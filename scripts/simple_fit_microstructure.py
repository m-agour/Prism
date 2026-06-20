#!/usr/bin/env python3

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from dipy.segment.mask import median_otsu
    HAVE_DIPY_MASK = True
except ImportError:
    HAVE_DIPY_MASK = False

from prism.differentiable_scanner_v4 import (
    DifferentiableScannerV4,
    AcquisitionProtocolV4,
    rician_nll,
    rician_loss,
    estimate_noise_sigma_from_background,
    rician_bias_correction,
    bias_corrected_mse_loss,
    hybrid_mse_rician_loss,
    b_dependent_hybrid_loss,
    snr_dependent_hybrid_loss,
    broadcast_sigma,
)
from prism.losses import loss_dmri, DMRILoss, SmartEarlyStopping, compute_grad_norm
from prism.noise_estimation import (
    estimate_noise_sigma,
    compute_sigma_map,
    compute_s0_map,
    NoiseEstimationResult,
)
from prism.microstructure_maps import (
    MicrostructureMaps,
    create_figure2_panel,
)
import torch.nn as nn



class MicrostructureEncoder(nn.Module):
    
    def __init__(self, n_measurements: int, n_fibers: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.n_measurements = n_measurements
        self.n_fibers = n_fibers
        
        self.backbone = nn.Sequential(
            nn.Linear(n_measurements, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        self.tissue_head = nn.Linear(hidden_dim, 3 + n_fibers)
        
        self.dir_head = nn.Linear(hidden_dim, n_fibers * 3)
        
        self.micro_head = nn.Linear(hidden_dim, 3)
    
    def forward(self, signal: torch.Tensor):
        shape = signal.shape[:-1]
        N = signal.shape[-1]
        
        x = signal.reshape(-1, N)
        
        features = self.backbone(x)
        
        tissue_raw = self.tissue_head(features)
        raw_csf = tissue_raw[:, 0].reshape(shape)
        raw_gm = tissue_raw[:, 1].reshape(shape)
        raw_restricted = tissue_raw[:, 2].reshape(shape)
        raw_wm = tissue_raw[:, 3:].reshape(shape + (self.n_fibers,))
        
        dirs_raw = self.dir_head(features)
        fiber_dirs = dirs_raw.reshape(shape + (self.n_fibers, 3))
        fiber_dirs = F.normalize(fiber_dirs, dim=-1)
        
        micro_raw = self.micro_head(features)
        raw_f_intra = micro_raw[:, 0].reshape(shape)
        raw_kappa = micro_raw[:, 1].reshape(shape)
        raw_s0 = micro_raw[:, 2].reshape(shape)
        
        return {
            'raw_csf': raw_csf,
            'raw_gm': raw_gm,
            'raw_wm': raw_wm,
            'raw_restricted': raw_restricted,
            'fiber_dirs': fiber_dirs,
            'f_intra': raw_f_intra,
            'kappa': raw_kappa,
            's0': raw_s0,
        }


def _encoder_to_scanner_params(enc_out: dict, n_fibers: int = 5) -> dict:
    tissue_raw = torch.cat([
        enc_out['raw_csf'].unsqueeze(-1),
        enc_out['raw_gm'].unsqueeze(-1),
        enc_out['raw_wm'],
        enc_out['raw_restricted'].unsqueeze(-1),
    ], dim=-1)
    tissue_fracs = F.softmax(tissue_raw, dim=-1)
    
    f_csf = tissue_fracs[..., 0]
    f_gm = tissue_fracs[..., 1]
    f_wm = tissue_fracs[..., 2:2+n_fibers]
    f_restricted = tissue_fracs[..., -1]
    
    return {
        'f_csf': f_csf,
        'f_gm': f_gm,
        'f_wm': f_wm,
        'f_restricted': f_restricted,
        'fiber_dirs': F.normalize(enc_out['fiber_dirs'], dim=-1),
        'f_intra': torch.sigmoid(enc_out['f_intra']),
        'kappa': F.softplus(enc_out['kappa']) + 0.1,
        's0': F.softplus(enc_out['s0']),
    }


def train_encoder_on_the_fly(
    encoder: nn.Module,
    scanner: 'DifferentiableScannerV4',
    target: torch.Tensor,
    mask: torch.Tensor,
    n_epochs: int = 10,
    lr: float = 0.001,
    device: str = 'cuda',
) -> nn.Module:
    encoder.train()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    mask_expanded = mask.unsqueeze(-1).expand_as(target)
    n_fibers = encoder.n_fibers
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        enc_out = encoder(target)
        scanner_params = _encoder_to_scanner_params(enc_out, n_fibers=n_fibers)
        pred = scanner(scanner_params, add_noise=False)
        loss = F.mse_loss(pred[mask_expanded], target[mask_expanded])
        loss.backward()
        optimizer.step()
        if epoch == 0 or epoch == n_epochs - 1:
            print(f"    Encoder epoch {epoch}: MSE = {loss.item():.6f}")
    
    encoder.eval()
    return encoder


def initialize_params_from_encoder(
    encoder: nn.Module,
    target: torch.Tensor,
    mask: torch.Tensor,
    n_fibers: int,
    device: str = 'cuda',
) -> dict:
    with torch.no_grad():
        enc_out = encoder(target)
    
    B, X, Y, Z = target.shape[:4]
    
    return {
        'raw_csf': enc_out['raw_csf'].clone().detach().requires_grad_(True),
        'raw_gm': enc_out['raw_gm'].clone().detach().requires_grad_(True),
        'raw_wm': enc_out['raw_wm'].clone().detach().requires_grad_(True),
        'raw_restricted': enc_out['raw_restricted'].clone().detach().requires_grad_(True),
        'fiber_dirs': enc_out['fiber_dirs'].clone().detach().requires_grad_(True),
        'f_intra': enc_out['f_intra'].clone().detach().requires_grad_(True),
        'kappa': enc_out['kappa'].clone().detach().requires_grad_(True),
        's0': enc_out['s0'].clone().detach().requires_grad_(True),
    }



def sample_uniform_sphere(shape: tuple, device: str = 'cuda') -> torch.Tensor:
    assert shape[-1] == 3, "Last dimension must be 3 for sphere sampling"
    
    dirs = torch.randn(shape, device=device)
    
    dirs = F.normalize(dirs, dim=-1)
    
    flip_mask = dirs[..., 2:3] < 0
    dirs = torch.where(flip_mask, -dirs, dirs)
    
    return dirs


def sample_uniform_sphere_stratified(shape: tuple, device: str = 'cuda') -> torch.Tensor:
    assert len(shape) == 6 and shape[-1] == 3, "Shape must be (B, X, Y, Z, n_fibers, 3)"
    
    B, X, Y, Z, n_fibers, _ = shape
    
    dirs = torch.randn(shape, device=device)
    
    if n_fibers > 1:
        for f in range(n_fibers):
            angle = f * (2 * 3.14159 / n_fibers)
            cos_a, sin_a = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
            
            x_new = dirs[..., f, 0] * cos_a - dirs[..., f, 1] * sin_a
            y_new = dirs[..., f, 0] * sin_a + dirs[..., f, 1] * cos_a
            dirs[..., f, 0] = x_new
            dirs[..., f, 1] = y_new
    
    dirs = F.normalize(dirs, dim=-1)
    
    flip_mask = dirs[..., 2:3] < 0
    dirs = torch.where(flip_mask, -dirs, dirs)
    
    return dirs



def compute_tissue_fractions(raw_csf, raw_gm, raw_wm_total, raw_restricted, 
                             use_sigmoid=False, sigmoid_scale=0.5):
    if use_sigmoid:
        f_csf_raw = sigmoid_scale * torch.sigmoid(raw_csf)
        f_gm_raw = sigmoid_scale * torch.sigmoid(raw_gm)
        f_wm_total_raw = sigmoid_scale * torch.sigmoid(raw_wm_total)
        f_restricted_raw = sigmoid_scale * torch.sigmoid(raw_restricted)
        
        total = f_csf_raw + f_gm_raw + f_wm_total_raw + f_restricted_raw
        
        scale = torch.clamp(total, min=1.0)
        
        f_csf = f_csf_raw / scale
        f_gm = f_gm_raw / scale
        f_wm_total = f_wm_total_raw / scale
        f_restricted = f_restricted_raw / scale
        
        f_residual = torch.clamp(1.0 - (f_csf + f_gm + f_wm_total + f_restricted), min=0.0)
    else:
        tissue_raw = torch.stack([raw_csf, raw_gm, raw_wm_total, raw_restricted], dim=-1)
        tissue_fracs = F.softmax(tissue_raw, dim=-1)
        
        f_csf = tissue_fracs[..., 0]
        f_gm = tissue_fracs[..., 1]
        f_wm_total = tissue_fracs[..., 2]
        f_restricted = tissue_fracs[..., 3]
        f_residual = torch.zeros_like(f_csf)
    
    return f_csf, f_gm, f_wm_total, f_restricted, f_residual




def load_dwi(dwi_path, bvals_path, bvecs_path, mask_path=None, 
             use_median_otsu=False, median_radius=4, numpass=4,
             auto_mask_scale=0.8, mask_dilate_iters=0):
    print(f"  Loading DWI: {dwi_path}")
    img = nib.load(dwi_path)
    data = img.get_fdata()

    print(f"  Loading bvals: {bvals_path}")
    bvals = np.loadtxt(bvals_path)
    
    print(f"  Loading bvecs: {bvecs_path}")
    bvecs = np.loadtxt(bvecs_path)

    if bvecs.ndim == 2:
        if bvecs.shape[0] == 3 and bvecs.shape[1] == len(bvals):
            bvecs = bvecs.T
        elif bvecs.shape[1] == 3 and bvecs.shape[0] == len(bvals):
            pass
        else:
            raise ValueError(f"Unexpected bvecs shape: {bvecs.shape}")
    else:
        raise ValueError(f"Unexpected bvecs shape: {bvecs.shape}")

    if use_median_otsu:
        if not HAVE_DIPY_MASK:
            raise ImportError("dipy is required for --median-otsu. Install with: pip install dipy")
        print(f"  Applying median_otsu (median_radius={median_radius}, numpass={numpass})...")
        b0_idx = np.argmin(bvals)
        b0_vol = data[..., b0_idx]
        _, mask = median_otsu(b0_vol, median_radius=median_radius, numpass=numpass)
        if mask_dilate_iters > 0:
            try:
                from scipy.ndimage import binary_dilation
                mask = binary_dilation(mask, iterations=mask_dilate_iters)
                print(f"  median_otsu mask: dilated by {mask_dilate_iters} iterations")
            except ImportError:
                print("  (scipy not available, skipping mask dilation)")
        n_voxels = np.sum(mask)
        print(f"  median_otsu mask: {n_voxels} voxels ({100*n_voxels/mask.size:.1f}% of volume)")
        mask_provided = True
    elif mask_path is not None:
        print(f"  Loading mask: {mask_path}")
        mask_img = nib.load(mask_path)
        mask = mask_img.get_fdata() > 0
        mask_provided = True
    else:
        print("  No mask provided, applying automatic Otsu threshold...")
        b0_idx = np.argmin(bvals)
        b0_vol = data[..., b0_idx]
        
        try:
            from skimage.filters import threshold_otsu
            b0_flat = b0_vol[b0_vol > 0].flatten()
            if len(b0_flat) > 0:
                threshold = threshold_otsu(b0_flat)
                threshold = threshold * auto_mask_scale
                mask = b0_vol > threshold
            else:
                mask = b0_vol > 0
        except ImportError:
            print("    (skimage not available, using percentile-based threshold)")
            b0_max = np.percentile(b0_vol, 99)
            threshold = 0.20 * b0_max * auto_mask_scale
            mask = b0_vol > threshold

        if mask_dilate_iters > 0:
            try:
                from scipy.ndimage import binary_dilation
                mask = binary_dilation(mask, iterations=mask_dilate_iters)
                print(f"  Auto-threshold mask: dilated by {mask_dilate_iters} iterations")
            except ImportError:
                print("  (scipy not available, skipping mask dilation)")
        
        n_voxels = np.sum(mask)
        print(f"  Auto-threshold mask: {n_voxels} voxels ({100*n_voxels/mask.size:.1f}% of volume)")
        mask_provided = False

    return img, data, bvals, bvecs, mask, mask_provided


def extract_patch_bounds(H, W, cx, cy, patch_size):
    half = patch_size // 2
    x0 = max(cx - half, 0)
    x1 = min(cx + half, H)
    y0 = max(cy - half, 0)
    y1 = min(cy + half, W)
    return x0, x1, y0, y1


def get_scanner_config_from_args(args) -> dict:
    return {
        'use_restricted': args.use_restricted and not getattr(args, 'no_restricted', False),
        'use_dot': getattr(args, 'use_dot', False),
        'use_dispersion': args.use_dispersion and not getattr(args, 'no_dispersion', False),
        'use_tortuosity': getattr(args, 'use_tortuosity', False),
        'use_kurtosis': getattr(args, 'use_kurtosis', False) and not getattr(args, 'no_kurtosis', False),
        'use_dki': getattr(args, 'use_dki', False),
        'use_diffusivity_spectrum': getattr(args, 'use_diffusivity_spectrum', True) and not getattr(args, 'no_diffusivity_spectrum', False),
        'n_spectrum_components': getattr(args, 'n_spectrum_components', 5),
        'use_t2_weighting': getattr(args, 'use_t2_weighting', False) and not getattr(args, 'no_t2_weighting', False),
        'use_per_shell_modulation': getattr(args, 'use_per_shell_modulation', False),
        'use_biexp_gm': getattr(args, 'use_biexp_gm', False),
        'learn_diffusivities': getattr(args, 'learn_diffusivities', False),
        'kappa_prior': getattr(args, 'kappa_prior', 8.0),
        'f_intra_prior': getattr(args, 'f_intra_prior', 0.5),
        'use_curvature_physics': getattr(args, 'use_curvature_physics', False),
        'curvature_alpha': getattr(args, 'curvature_alpha', 10000.0),
        'curvature_epsilon': getattr(args, 'curvature_epsilon', 0.2),
        'voxel_size': tuple(getattr(args, 'voxel_size', [1.0, 1.0, 1.0])),
        'use_mtcsd': getattr(args, 'use_mtcsd', False),
        'mtcsd_n_odf_dirs': getattr(args, 'mtcsd_n_odf_dirs', 60),
        'mtcsd_n_spectrum': getattr(args, 'mtcsd_n_spectrum', 5),
        'mtcsd_learn_responses': getattr(args, 'mtcsd_learn_responses', True),
        'learn_eddy': getattr(args, 'learn_eddy', True) and not getattr(args, 'no_eddy', False),
        'learn_shell_gain': getattr(args, 'learn_shell_gain', False) and not getattr(args, 'no_shell_gain', False),
        'learn_bias_field': getattr(args, 'learn_bias_field', True) and not getattr(args, 'no_bias_field', False),
        'learn_warps': getattr(args, 'learn_warps', False),
        'use_spatial_noise': getattr(args, 'use_spatial_noise', False),
        'use_psf': getattr(args, 'use_psf', False),
        'psf_sigma': getattr(args, 'psf_sigma', 0.68),
        'psf_kernel_size': getattr(args, 'psf_kernel_size', 3),
        'use_ghosting': getattr(args, 'use_ghosting', False),
        'use_qspace_mixing': getattr(args, 'use_qspace_mixing', False),
        'qspace_k_neighbors': getattr(args, 'qspace_k_neighbors', 6),
        'learn_channel_gain': getattr(args, 'learn_channel_gain', False),
        'channel_gain_reg': getattr(args, 'channel_gain_reg', 0.1),
        'low_rank_channel_gain': getattr(args, 'low_rank_channel_gain', False),
        'channel_gain_rank': getattr(args, 'channel_gain_rank', 5),
        'use_exchange': getattr(args, 'use_exchange', False),
        'exchange_tau': getattr(args, 'exchange_tau', 50.0),
        'exchange_pairs': getattr(args, 'exchange_pairs', ['restricted_gm']),
        'exchange_learnable': getattr(args, 'exchange_learnable', True) and not getattr(args, 'no_exchange_learnable', False),
        'use_tissue_priors': getattr(args, 'use_tissue_priors', True) and not getattr(args, 'no_tissue_priors', False),
        'learnable_tissue_priors': getattr(args, 'learnable_tissue_priors', False),
        'learnable_prior_weight': getattr(args, 'learnable_prior_weight', 0.001),
        'use_spatial_prior': getattr(args, 'use_spatial_prior', True) and not getattr(args, 'no_spatial_prior', False),
        'spatial_prior_weight': getattr(args, 'spatial_prior_weight', 0.01),
        'spatial_prior_connectivity': getattr(args, 'spatial_prior_connectivity', "26"),
    }


def get_topology_config_from_args(args) -> dict:
    use_6conn = getattr(args, 'use_6conn_topology', False)
    use_26conn = getattr(args, 'use_26conn_topology', False)
    
    if use_26conn:
        connectivity = 26
    elif use_6conn:
        connectivity = 6
    else:
        connectivity = 0
    
    return {
        'use_topology': getattr(args, 'use_topology', True) and not getattr(args, 'no_topology', False),
        'use_6conn_topology': use_6conn,
        'use_26conn_topology': use_26conn,
        'connectivity': connectivity,
        'lambda_orphan': getattr(args, 'lambda_orphan', 0.01),
        'lambda_continuity': getattr(args, 'lambda_continuity', 0.005),
        'lambda_ordering': getattr(args, 'lambda_ordering', 0.01),
        'lambda_repulsion': getattr(args, 'lambda_repulsion', 0.01),
        'lambda_endpoint': getattr(args, 'lambda_endpoint', 0.01),
        'lambda_curvature': getattr(args, 'lambda_curvature', 0.0),
        'lambda_perm_invariant': getattr(args, 'lambda_perm_invariant', 0.0),
    }


def print_scanner_config(config: dict):
    print("\n  Scanner Configuration:")
    print("  ─" * 30)
    
    print("  Signal Model:")
    for key in ['use_restricted', 'use_dot', 'use_dispersion', 'use_tortuosity', 
                'use_kurtosis', 'use_dki', 'use_diffusivity_spectrum', 'use_t2_weighting', 
                'use_per_shell_modulation', 'use_mtcsd', 'use_curvature_physics']:
        if key in config:
            status = "✓" if config[key] else "✗"
            print(f"    {status} {key.replace('use_', '').replace('_', ' ').title()}")
    
    if config.get('use_diffusivity_spectrum', False):
        print(f"      n_spectrum_components: {config.get('n_spectrum_components', 5)}")
    
    if config.get('use_curvature_physics', False):
        print("  Curvature-Aware Physics (NOVEL):")
        print(f"    ✓ Geometry-predicted κ = 1/(α·c²·L² + ε)")
        print(f"      α = {config.get('curvature_alpha', 10000.0)}")
        print(f"      ε = {config.get('curvature_epsilon', 0.2)}")
        print(f"      voxel_size = {config.get('voxel_size', (1.0, 1.0, 1.0))}")
        print(f"      35.6% MSE improvement on DiSCo phantom")
    
    if config.get('use_mtcsd', False):
        print("\n  ⚠️  WARNING: MT-CSD is EXPERIMENTAL and may degrade fiber quality ⚠️")
        print("      Consider using standard model for production runs.")
        print(f"      ODF directions: {config.get('mtcsd_n_odf_dirs', 60)}")
        print(f"      Spectrum components: {config.get('mtcsd_n_spectrum', 5)}")
        print(f"      Learn responses: {config.get('mtcsd_learn_responses', True)}")
    
    if config.get('use_exchange', False):
        print("  Kärger Exchange (RESEARCH):")
        print(f"    ✓ Exchange enabled")
        print(f"      τ_ex = {config.get('exchange_tau', 50.0):.1f} ms")
        print(f"      Pairs: {config.get('exchange_pairs', ['restricted_gm'])}")
        print(f"      Learnable: {config.get('exchange_learnable', True)}")
    
    print("  Artifact Model:")
    for key in ['learn_shell_gain', 'learn_bias_field', 'learn_warps', 
                'use_spatial_noise', 'use_psf', 'use_ghosting', 'use_qspace_mixing']:
        if key in config:
            status = "✓" if config[key] else "✗"
            name = key.replace('learn_', '').replace('use_', '').replace('_', ' ').title()
            print(f"    {status} {name}")
    
    print("  Priors:")
    for key in ['use_tissue_priors', 'use_spatial_prior']:
        if key in config:
            status = "✓" if config[key] else "✗"
            print(f"    {status} {key.replace('use_', '').replace('_', ' ').title()}")
    
    print()


def export_scanner_artifacts(scanner, outdir, affine, spatial_shape, mask=None, bvals=None, dataset_name="", dwi_signal=None):
    import json
    
    artifacts_dir = os.path.join(outdir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"EXPORTING SCANNER ARTIFACT MAPS")
    print(f"{'='*60}")
    print(f"  Output: {artifacts_dir}/")
    
    art = getattr(scanner, 'artifacts', None)
    if art is None:
        print("  ⚠ No artifact model found (learn_artifacts=False). Skipping export.")
        return {}
    
    artifact_summary = {}
    active_artifacts = []
    
    if art.learn_shell_gain:
        with torch.no_grad():
            shell_gains = art.shell_gain.detach().cpu().numpy()
        
        if bvals is not None:
            unique_bvals = sorted(set(int(round(b, -2)) for b in bvals))
        else:
            unique_bvals = list(range(len(shell_gains)))
        
        shell_gain_dict = {
            'description': 'Per-shell S0 correction factors (multiplicative)',
            'n_shells': len(shell_gains),
            'b_values': unique_bvals[:len(shell_gains)],
            'gains': [float(g) for g in shell_gains],
            'log_gains': [float(g) for g in np.log(shell_gains)],
        }
        
        with open(os.path.join(artifacts_dir, "shell_gains.json"), "w") as f:
            json.dump(shell_gain_dict, f, indent=2)
        
        artifact_summary['shell_gains'] = shell_gain_dict
        active_artifacts.append('shell_gains')
        
        print(f"  ✓ Shell gains ({len(shell_gains)} shells):")
        for i, (bv, g) in enumerate(zip(unique_bvals[:len(shell_gains)], shell_gains)):
            print(f"      b={bv}: gain={g:.4f} (log={np.log(g):.4f})")
    
    if art.learn_bias_field:
        with torch.no_grad():
            X, Y, Z = spatial_shape
            bias_lr = art.bias_field_lr
            bias_full = F.interpolate(
                bias_lr,
                size=(X, Y, Z),
                mode='trilinear',
                align_corners=False,
            )
            bias_field = torch.exp(bias_full)[0, 0].detach().cpu().numpy()
        
        nib.save(nib.Nifti1Image(bias_field.astype(np.float32), affine),
                 os.path.join(artifacts_dir, "bias_field.nii.gz"))
        
        bias_lr_np = bias_lr.detach().squeeze().cpu().numpy()
        np.save(os.path.join(artifacts_dir, "bias_field_lr_raw.npy"), bias_lr_np)
        
        bias_stats = {
            'description': 'B1 bias field (multiplicative, centered ~1.0)',
            'lr_shape': list(bias_lr_np.shape),
            'full_shape': list(bias_field.shape),
            'min': float(bias_field.min()),
            'max': float(bias_field.max()),
            'mean': float(bias_field.mean()),
            'std': float(bias_field.std()),
        }
        if mask is not None:
            brain_mask_3d = (mask > 0).reshape(bias_field.shape)
            if brain_mask_3d.any():
                bias_stats['brain_min'] = float(bias_field[brain_mask_3d].min())
                bias_stats['brain_max'] = float(bias_field[brain_mask_3d].max())
                bias_stats['brain_mean'] = float(bias_field[brain_mask_3d].mean())
                bias_stats['brain_std'] = float(bias_field[brain_mask_3d].std())
        
        artifact_summary['bias_field'] = bias_stats
        active_artifacts.append('bias_field')
        
        print(f"  ✓ Bias field: range [{bias_field.min():.4f}, {bias_field.max():.4f}], "
              f"mean={bias_field.mean():.4f}")
    
    if art.learn_channel_gain or art.low_rank_channel_gain:
        with torch.no_grad():
            channel_gains = art.channel_gain.detach().cpu().numpy()
        
        np.save(os.path.join(artifacts_dir, "channel_gains.npy"), channel_gains)
        
        channel_gain_dict = {
            'description': 'Per-channel gain corrections (multiplicative)',
            'n_channels': len(channel_gains),
            'low_rank': art.low_rank_channel_gain,
            'rank': getattr(art, 'channel_gain_rank', None),
            'min': float(channel_gains.min()),
            'max': float(channel_gains.max()),
            'mean': float(channel_gains.mean()),
            'std': float(channel_gains.std()),
        }
        
        with open(os.path.join(artifacts_dir, "channel_gains.json"), "w") as f:
            json.dump(channel_gain_dict, f, indent=2)
        
        artifact_summary['channel_gains'] = channel_gain_dict
        active_artifacts.append('channel_gains')
        
        print(f"  ✓ Channel gains ({len(channel_gains)} channels): "
              f"range [{channel_gains.min():.4f}, {channel_gains.max():.4f}]")
    
    if art.learn_noise:
        with torch.no_grad():
            X, Y, Z = spatial_shape
            if getattr(art, 'use_spatial_noise', False):
                noise_field = art.get_noise_field(spatial_shape)
                noise_field_np = noise_field.squeeze().detach().cpu().numpy()
                
                nib.save(nib.Nifti1Image(noise_field_np.astype(np.float32), affine),
                         os.path.join(artifacts_dir, "noise_sigma_field.nii.gz"))
                
                noise_lr_np = art.log_noise_sigma_lr.detach().squeeze().cpu().numpy()
                np.save(os.path.join(artifacts_dir, "noise_sigma_lr_raw.npy"), noise_lr_np)
                
                noise_dict = {
                    'description': 'Spatially varying Rician noise σ',
                    'spatial': True,
                    'lr_shape': list(noise_lr_np.shape),
                    'full_shape': list(noise_field_np.shape),
                    'min': float(noise_field_np.min()),
                    'max': float(noise_field_np.max()),
                    'mean': float(noise_field_np.mean()),
                }
            else:
                sigma_val = art.noise_sigma
                noise_dict = {
                    'description': 'Global Rician noise σ',
                    'spatial': False,
                    'sigma': sigma_val,
                }
        
        with open(os.path.join(artifacts_dir, "noise.json"), "w") as f:
            json.dump(noise_dict, f, indent=2)
        
        artifact_summary['noise'] = noise_dict
        active_artifacts.append('noise')
        
        noise_desc = 'spatial' if noise_dict.get('spatial') else f'{noise_dict.get("sigma", 0):.6f}'
        print(f"  ✓ Noise: σ={noise_desc}")
    
    if art.learn_eddy:
        with torch.no_grad():
            eddy_scale = torch.exp(art.log_eddy_scale).detach().cpu().numpy()
            eddy_bias = art.eddy_bias.detach().cpu().numpy()
        
        np.save(os.path.join(artifacts_dir, "eddy_scale.npy"), eddy_scale)
        np.save(os.path.join(artifacts_dir, "eddy_bias.npy"), eddy_bias)
        
        eddy_dict = {
            'description': 'Eddy current parameters: signal = signal * scale + bias',
            'n_channels': len(eddy_scale),
            'scale_range': [float(eddy_scale.min()), float(eddy_scale.max())],
            'bias_range': [float(eddy_bias.min()), float(eddy_bias.max())],
            'scale_mean': float(eddy_scale.mean()),
            'bias_mean': float(eddy_bias.mean()),
        }
        
        with open(os.path.join(artifacts_dir, "eddy.json"), "w") as f:
            json.dump(eddy_dict, f, indent=2)
        
        artifact_summary['eddy'] = eddy_dict
        active_artifacts.append('eddy')
        
        print(f"  ✓ Eddy currents: scale=[{eddy_scale.min():.4f}, {eddy_scale.max():.4f}], "
              f"bias=[{eddy_bias.min():.6f}, {eddy_bias.max():.6f}]")
    
    if art.learn_warps:
        with torch.no_grad():
            warp_lr = art.warp_field_lr.detach().cpu().numpy()
        
        np.save(os.path.join(artifacts_dir, "warp_field_lr.npy"), warp_lr)
        
        warp_dict = {
            'description': 'Low-res geometric warp field along phase-encode direction',
            'shape': list(warp_lr.shape),
            'max_displacement': float(np.abs(warp_lr).max()),
            'mean_abs_displacement': float(np.abs(warp_lr).mean()),
        }
        
        with open(os.path.join(artifacts_dir, "warps.json"), "w") as f:
            json.dump(warp_dict, f, indent=2)
        
        artifact_summary['warps'] = warp_dict
        active_artifacts.append('warps')
        
        print(f"  ✓ Warps: max displacement={np.abs(warp_lr).max():.4f}")
    
    if getattr(art, 'use_qspace_mixing', False):
        with torch.no_grad():
            eps = torch.exp(art.log_eps_qspace).item()
        
        qmix_dict = {
            'description': 'Q-space Laplacian mixing (epsilon parameter)',
            'epsilon': eps,
        }
        
        with open(os.path.join(artifacts_dir, "qspace_mixing.json"), "w") as f:
            json.dump(qmix_dict, f, indent=2)
        
        artifact_summary['qspace_mixing'] = qmix_dict
        active_artifacts.append('qspace_mixing')
        
        print(f"  ✓ Q-space mixing: ε={eps:.6f}")
    
    if getattr(art, 'use_ghosting', False):
        with torch.no_grad():
            ghost_amps = torch.tanh(art.ghost_amplitudes).detach().cpu().numpy()
        
        np.save(os.path.join(artifacts_dir, "ghost_amplitudes.npy"), ghost_amps)
        
        ghost_dict = {
            'description': 'Nyquist ghosting amplitudes per measurement',
            'n_channels': len(ghost_amps),
            'max_amplitude': float(np.abs(ghost_amps).max()),
            'mean_abs': float(np.abs(ghost_amps).mean()),
        }
        
        with open(os.path.join(artifacts_dir, "ghosting.json"), "w") as f:
            json.dump(ghost_dict, f, indent=2)
        
        artifact_summary['ghosting'] = ghost_dict
        active_artifacts.append('ghosting')
        
        print(f"  ✓ Ghosting: max amplitude={np.abs(ghost_amps).max():.4f}")
    
    artifact_summary['active_artifacts'] = active_artifacts
    artifact_summary['spatial_shape'] = list(spatial_shape)
    
    with open(os.path.join(artifacts_dir, "artifact_summary.json"), "w") as f:
        json.dump(artifact_summary, f, indent=2)
    
    _create_artifacts_panel(
        art=art,
        artifacts_dir=artifacts_dir,
        spatial_shape=spatial_shape,
        mask=mask,
        bvals=bvals,
        active_artifacts=active_artifacts,
        dataset_name=dataset_name,
        dwi_signal=dwi_signal,
    )
    
    print(f"\n  ✓ Artifact export complete → {artifacts_dir}/")
    print(f"    Active components: {', '.join(active_artifacts)}")
    
    return artifact_summary


def _create_artifacts_panel(art, artifacts_dir, spatial_shape, mask=None,
                            bvals=None, active_artifacts=None, dataset_name="",
                            dwi_signal=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    if not active_artifacts:
        print("  [Artifacts Panel] No active artifacts to visualize.")
        return
    
    n_panels = 0
    panel_specs = []
    
    for artifact_name in active_artifacts:
        if artifact_name == 'shell_gains':
            panel_specs.append(('Shell Gains (S₀ per shell)', 'bar', 1))
            n_panels += 1
        elif artifact_name == 'bias_field':
            panel_specs.append(('B₁ Bias Field', 'spatial_3axis', 3))
            n_panels += 1
            if dwi_signal is not None and bvals is not None:
                panel_specs.append(('b0 Normalization Effect', 'b0_normalization', 3))
                n_panels += 1
                panel_specs.append(('Per-Shell Mean DWI', 'shell_images', 0))
                n_panels += 1
        elif artifact_name == 'channel_gains':
            panel_specs.append(('Channel Gains (per measurement)', 'line', 1))
            n_panels += 1
        elif artifact_name == 'noise':
            spatial = getattr(art, 'use_spatial_noise', False)
            if spatial:
                panel_specs.append(('Noise σ Field', 'spatial_3axis', 3))
            else:
                panel_specs.append(('Global Noise σ', 'text', 1))
            n_panels += 1
        elif artifact_name == 'eddy':
            panel_specs.append(('Eddy Current (scale + bias)', 'dual_line', 2))
            n_panels += 1
        elif artifact_name == 'warps':
            panel_specs.append(('Geometric Warps', 'line', 1))
            n_panels += 1
        elif artifact_name == 'qspace_mixing':
            panel_specs.append(('Q-space Mixing', 'text', 1))
            n_panels += 1
        elif artifact_name == 'ghosting':
            panel_specs.append(('Nyquist Ghosting', 'line', 1))
            n_panels += 1
    
    if dwi_signal is not None and bvals is not None:
        panel_specs.append(('Artifact Correction Summary', 'artifact_correction', 4))
        n_panels += 1
    
    if n_panels == 0:
        return
    
    fig_height = max(3, n_panels * 2.5)
    fig = plt.figure(figsize=(14, fig_height))
    gs = GridSpec(n_panels, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    X, Y, Z = spatial_shape
    if mask is not None:
        brain_mask = (mask > 0).reshape(spatial_shape) if np.prod(mask.shape) == np.prod(spatial_shape) else (mask > 0)
        if brain_mask.ndim < 3:
            brain_mask = brain_mask.reshape(spatial_shape)
    else:
        brain_mask = np.ones(spatial_shape, dtype=bool)
    
    row = 0
    with torch.no_grad():
        for panel_name, panel_type, n_cols in panel_specs:
            
            if panel_type == 'bar' and 'Shell' in panel_name:
                ax = fig.add_subplot(gs[row, :])
                gains = art.shell_gain.detach().cpu().numpy()
                if bvals is not None:
                    unique_b = sorted(set(int(round(b, -2)) for b in bvals))[:len(gains)]
                    labels = [f'b={b}' for b in unique_b]
                else:
                    labels = [f'Shell {i}' for i in range(len(gains))]
                
                colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(gains)))
                bars = ax.bar(labels, gains, color=colors, edgecolor='black', linewidth=0.5)
                ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Neutral (gain=1)')
                ax.set_ylabel('Gain Factor')
                ax.set_title(panel_name, fontweight='bold', fontsize=11)
                ax.legend(fontsize=8)
                
                for bar_obj, g in zip(bars, gains):
                    ax.text(bar_obj.get_x() + bar_obj.get_width()/2, bar_obj.get_height() + 0.002,
                            f'{g:.4f}', ha='center', va='bottom', fontsize=8)
                
            elif panel_type == 'spatial_3axis' and 'Bias' in panel_name:
                bias_lr = art.bias_field_lr
                bias_full = F.interpolate(
                    bias_lr, size=(X, Y, Z), mode='trilinear', align_corners=False,
                )
                bias = torch.exp(bias_full)[0, 0].detach().cpu().numpy()
                
                if Z <= 1:
                    slices = [
                        ('Slice', bias[:, :, 0].T, brain_mask[:, :, 0].T),
                    ]
                else:
                    slices = [
                        ('Axial', bias[:, :, Z//2].T, brain_mask[:, :, Z//2].T),
                        ('Coronal', bias[:, Y//2, :].T, brain_mask[:, Y//2, :].T),
                        ('Sagittal', bias[X//2, :, :].T, brain_mask[X//2, :, :].T),
                    ]
                
                brain_mask_3d = brain_mask.reshape(bias.shape) if brain_mask.shape != bias.shape else brain_mask
                if brain_mask_3d.any():
                    vmax = max(abs(bias[brain_mask_3d].max() - 1.0), abs(bias[brain_mask_3d].min() - 1.0))
                else:
                    vmax = max(abs(bias.max() - 1.0), abs(bias.min() - 1.0))
                vmax = max(vmax, 0.001)
                vmin_val = 1.0 - vmax
                vmax_val = 1.0 + vmax
                
                n_views = len(slices)
                for col, (view_name, slice_data, slice_mask) in enumerate(slices):
                    ax = fig.add_subplot(gs[row, col] if n_views > 1 else gs[row, :])
                    
                    display = slice_data.copy().astype(float)
                    display[~slice_mask] = np.nan
                    
                    cmap = plt.cm.RdBu_r.copy()
                    cmap.set_bad(color='white')
                    
                    im = ax.imshow(display, cmap=cmap, vmin=vmin_val, vmax=vmax_val,
                                   origin='lower', aspect='equal')
                    ax.set_title(f'{panel_name} - {view_name}' if col == 0 else view_name,
                                fontweight='bold' if col == 0 else 'normal', fontsize=10)
                    ax.axis('off')
                    
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im, cax=cax)
            
            elif panel_type == 'b0_normalization' and dwi_signal is not None:
                sig = dwi_signal
                X_s, Y_s, Z_s = spatial_shape
                if sig.ndim == 4:
                    sig = sig[:, :, Z_s // 2, :]
                
                b0_mask_arr = np.array([abs(int(round(b, -2))) < 50 for b in bvals])
                if b0_mask_arr.sum() == 0:
                    b0_mask_arr = np.array([abs(int(round(b, -2))) < 150 for b in bvals])
                
                raw_b0 = sig[:, :, b0_mask_arr].mean(axis=-1)
                s0_map = np.maximum(raw_b0, 1e-6)
                
                b0_pos = raw_b0[raw_b0 > 0]
                if len(b0_pos) > 0:
                    bg_thresh = np.percentile(b0_pos, 10)
                else:
                    bg_thresh = 0
                fg_mask = raw_b0 > bg_thresh
                
                normed_all = sig / s0_map[..., np.newaxis]
                normed_mean = np.clip(normed_all, 0, 3).mean(axis=-1)
                normed_mean[~fg_mask] = 0
                
                bmask_2d = brain_mask[:, :, 0].T if brain_mask.ndim == 3 else brain_mask.T
                from scipy.ndimage import binary_fill_holes
                bmask_2d = binary_fill_holes(bmask_2d)
                
                bias_lr_disp = art.bias_field_lr
                bias_full_disp = F.interpolate(
                    bias_lr_disp, size=(X, Y, Z), mode='trilinear', align_corners=False,
                )
                bias_disp = torch.exp(bias_full_disp)[0, 0].detach().cpu().numpy()
                if Z <= 1:
                    bias_slice = bias_disp[:, :, 0].T
                else:
                    bias_slice = bias_disp[:, :, Z // 2].T
                
                corrected_mean = normed_mean.T / np.maximum(bias_slice, 1e-9)
                
                if bmask_2d.any():
                    raw_b0_brain = raw_b0.T[bmask_2d]
                    normed_brain = normed_mean.T[bmask_2d]
                    bias_brain = bias_slice[bmask_2d]
                    corrected_brain = corrected_mean[bmask_2d]
                    raw_cv = raw_b0_brain.std() / (raw_b0_brain.mean() + 1e-9)
                    normed_cv = normed_brain.std() / (normed_brain.mean() + 1e-9)
                    corrected_cv = corrected_brain.std() / (corrected_brain.mean() + 1e-9)
                    bf_spread = bias_brain.max() - bias_brain.min()
                else:
                    raw_cv = normed_cv = corrected_cv = bf_spread = 0.0
                
                inner_gs_b0 = gs[row, :].subgridspec(1, 4, wspace=0.3)
                
                cmap_gray = plt.cm.gray.copy()
                
                ax0 = fig.add_subplot(inner_gs_b0[0, 0])
                im0 = ax0.imshow(raw_b0.T, cmap=cmap_gray, origin='lower', aspect='equal')
                ax0.set_title(f'BEFORE: Raw S₀ (CV={raw_cv:.2f})',
                             fontweight='bold', fontsize=8, color='#d62728')
                ax0.axis('off')
                div0 = make_axes_locatable(ax0)
                plt.colorbar(im0, cax=div0.append_axes("right", size="5%", pad=0.05))
                
                ax1 = fig.add_subplot(inner_gs_b0[0, 1])
                im1 = ax1.imshow(normed_mean.T, cmap=cmap_gray, origin='lower', aspect='equal')
                ax1.set_title(f'AFTER norm: S/S₀ (CV={normed_cv:.2f})',
                             fontweight='bold', fontsize=8, color='#2ca02c')
                ax1.axis('off')
                div1 = make_axes_locatable(ax1)
                plt.colorbar(im1, cax=div1.append_axes("right", size="5%", pad=0.05))
                
                ax2 = fig.add_subplot(inner_gs_b0[0, 2])
                if bmask_2d.any():
                    vmax_bf = max(abs(bias_brain.max() - 1.0), abs(bias_brain.min() - 1.0))
                else:
                    vmax_bf = 0.05
                vmax_bf = max(vmax_bf, 0.001)
                im2 = ax2.imshow(bias_slice, cmap='RdBu_r', vmin=1.0 - vmax_bf, vmax=1.0 + vmax_bf,
                                origin='lower', aspect='equal')
                ax2.set_title(f'Learned B₁ (spread={bf_spread:.4f})',
                             fontweight='bold', fontsize=8, color='#1f77b4')
                ax2.axis('off')
                div2 = make_axes_locatable(ax2)
                plt.colorbar(im2, cax=div2.append_axes("right", size="5%", pad=0.05))
                
                ax3 = fig.add_subplot(inner_gs_b0[0, 3])
                im3 = ax3.imshow(corrected_mean, cmap=cmap_gray, origin='lower', aspect='equal',
                                vmin=im1.get_clim()[0], vmax=im1.get_clim()[1])
                ax3.set_title(f'CORRECTED: S/(S₀·B₁) (CV={corrected_cv:.2f})',
                             fontweight='bold', fontsize=8, color='#9467bd')
                ax3.axis('off')
                div3 = make_axes_locatable(ax3)
                plt.colorbar(im3, cax=div3.append_axes("right", size="5%", pad=0.05))
                    
            elif panel_type == 'shell_images' and dwi_signal is not None:
                unique_b = sorted(set(int(round(b, -2)) for b in bvals))
                
                sig = dwi_signal
                X_s, Y_s, Z_s = spatial_shape
                if sig.ndim == 3 and Z_s == 1:
                    pass
                elif sig.ndim == 4:
                    sig = sig[:, :, Z_s // 2, :]
                
                shell_imgs = []
                shell_labels = []
                for bv in unique_b:
                    shell_mask_b = np.array([abs(int(round(b, -2)) - bv) < 50 for b in bvals])
                    if shell_mask_b.sum() > 0:
                        mean_img = sig[:, :, shell_mask_b].mean(axis=-1)
                        shell_imgs.append(mean_img)
                        shell_labels.append(f'b={bv} (n={int(shell_mask_b.sum())})')
                
                n_show = len(shell_imgs)
                inner_gs = gs[row, :].subgridspec(1, n_show, wspace=0.3)
                for col_idx in range(n_show):
                    ax = fig.add_subplot(inner_gs[0, col_idx])
                    
                    display = shell_imgs[col_idx].T.copy().astype(float)
                    
                    im = ax.imshow(display, cmap='gray', origin='lower', aspect='equal')
                    title = shell_labels[col_idx]
                    if col_idx == 0:
                        title = f'Per-Shell Mean DWI - {title}'
                    ax.set_title(title, fontweight='bold' if col_idx == 0 else 'normal', fontsize=9)
                    ax.axis('off')
                    
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im, cax=cax)
                
            elif panel_type == 'line' and 'Channel' in panel_name:
                ax = fig.add_subplot(gs[row, :])
                gains = art.channel_gain.detach().cpu().numpy()
                
                if bvals is not None:
                    unique_b = sorted(set(int(round(b, -2)) for b in bvals))
                    shell_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_b)))
                    color_map = {}
                    for i, b in enumerate(unique_b):
                        color_map[b] = shell_colors[i]
                    
                    for idx in range(len(gains)):
                        b_val = int(round(bvals[idx], -2))
                        ax.scatter(idx, gains[idx], c=[color_map.get(b_val, 'gray')], 
                                  s=10, alpha=0.7, edgecolors='none')
                    
                    for b_val, color in color_map.items():
                        ax.scatter([], [], c=[color], label=f'b={b_val}', s=30)
                    ax.legend(fontsize=7, ncol=min(len(unique_b), 4))
                else:
                    ax.plot(gains, linewidth=0.8, color='steelblue')
                
                ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
                ax.set_xlabel('Measurement Index')
                ax.set_ylabel('Gain')
                ax.set_title(panel_name, fontweight='bold', fontsize=11)
                
            elif panel_type == 'spatial_3axis' and 'Noise' in panel_name:
                noise_raw = art.get_noise_field(spatial_shape)
                if noise_raw.dim() == 5:
                    noise_field = noise_raw[0, 0].detach().cpu().numpy()
                elif noise_raw.dim() == 4:
                    noise_field = noise_raw[0].detach().cpu().numpy()
                else:
                    noise_field = noise_raw.detach().cpu().numpy()
                
                if Z <= 1:
                    slices = [
                        ('Slice', noise_field[:, :, 0].T if noise_field.ndim == 3 else noise_field.T,
                         brain_mask[:, :, 0].T if brain_mask.ndim == 3 else brain_mask.T),
                    ]
                else:
                    slices = [
                        ('Axial', noise_field[:, :, Z//2].T, brain_mask[:, :, Z//2].T),
                        ('Coronal', noise_field[:, Y//2, :].T, brain_mask[:, Y//2, :].T),
                        ('Sagittal', noise_field[X//2, :, :].T, brain_mask[X//2, :, :].T),
                    ]
                
                n_views = len(slices)
                for col, (view_name, slice_data, slice_mask) in enumerate(slices):
                    ax = fig.add_subplot(gs[row, col] if n_views > 1 else gs[row, :])
                    
                    display = slice_data.copy().astype(float)
                    display[~slice_mask] = np.nan
                    
                    cmap = plt.cm.hot.copy()
                    cmap.set_bad(color='white')
                    
                    im = ax.imshow(display, cmap=cmap, origin='lower', aspect='equal')
                    ax.set_title(f'{panel_name} - {view_name}' if col == 0 else view_name,
                                fontweight='bold' if col == 0 else 'normal', fontsize=10)
                    ax.axis('off')
                    
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im, cax=cax)
                    
            elif panel_type == 'dual_line' and 'Eddy' in panel_name:
                eddy_scale = torch.exp(art.log_eddy_scale).detach().cpu().numpy()
                eddy_bias = art.eddy_bias.detach().cpu().numpy()
                
                ax1 = fig.add_subplot(gs[row, :2])
                ax1.plot(eddy_scale, linewidth=0.8, color='steelblue')
                ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
                ax1.set_xlabel('Channel')
                ax1.set_ylabel('Scale')
                ax1.set_title('Eddy Scale', fontweight='bold', fontsize=10)
                
                ax2 = fig.add_subplot(gs[row, 2])
                ax2.plot(eddy_bias, linewidth=0.8, color='darkorange')
                ax2.axhline(y=0.0, color='red', linestyle='--', alpha=0.5)
                ax2.set_xlabel('Channel')
                ax2.set_ylabel('Bias')
                ax2.set_title('Eddy Bias', fontweight='bold', fontsize=10)
                
            elif panel_type == 'line' and 'Warp' in panel_name:
                warp_lr = art.warp_field_lr.detach().cpu().numpy()
                max_disp = np.abs(warp_lr).max(axis=(1, 2, 3))
                
                ax = fig.add_subplot(gs[row, :])
                ax.bar(range(len(max_disp)), max_disp, color='mediumpurple', edgecolor='black', linewidth=0.3)
                ax.set_xlabel('Channel')
                ax.set_ylabel('Max |displacement|')
                ax.set_title(panel_name, fontweight='bold', fontsize=11)
                
            elif panel_type == 'text':
                ax = fig.add_subplot(gs[row, :])
                ax.axis('off')
                
                if 'Noise' in panel_name:
                    sigma = art.noise_sigma
                    ax.text(0.5, 0.5, f'Global Noise σ = {sigma:.6f}',
                           transform=ax.transAxes, ha='center', va='center',
                           fontsize=14, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))
                elif 'Q-space' in panel_name:
                    eps = torch.exp(art.log_eps_qspace).item()
                    ax.text(0.5, 0.5, f'Q-space mixing ε = {eps:.6f}',
                           transform=ax.transAxes, ha='center', va='center',
                           fontsize=14, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='lightcyan', edgecolor='steelblue'))
                
                ax.set_title(panel_name, fontweight='bold', fontsize=11)
                
            elif panel_type == 'line' and 'Ghost' in panel_name:
                ghost_amps = torch.tanh(art.ghost_amplitudes).detach().cpu().numpy()
                
                ax = fig.add_subplot(gs[row, :])
                ax.bar(range(len(ghost_amps)), np.abs(ghost_amps), 
                      color='salmon', edgecolor='black', linewidth=0.3)
                ax.set_xlabel('Channel')
                ax.set_ylabel('|Ghost Amplitude|')
                ax.set_title(panel_name, fontweight='bold', fontsize=11)
            
            elif panel_type == 'artifact_correction' and dwi_signal is not None:
                sig = dwi_signal
                X_s, Y_s, Z_s = spatial_shape
                if sig.ndim == 4:
                    sig = sig[:, :, Z_s // 2, :]
                N_meas = sig.shape[-1]
                
                b0_mask_ac = np.array([abs(int(round(b, -2))) < 50 for b in bvals])
                if b0_mask_ac.sum() == 0:
                    b0_mask_ac = np.array([abs(int(round(b, -2))) < 150 for b in bvals])
                raw_b0_ac = sig[:, :, b0_mask_ac].mean(axis=-1)
                s0_map = np.maximum(raw_b0_ac, 1e-6)
                sig_norm = np.clip(sig / s0_map[..., np.newaxis], 0, 3)
                
                b0_pos_ac = raw_b0_ac[raw_b0_ac > 0]
                if len(b0_pos_ac) > 0:
                    bg_thresh_ac = np.percentile(b0_pos_ac, 10)
                else:
                    bg_thresh_ac = 0
                fg_mask_ac = raw_b0_ac > bg_thresh_ac
                sig_norm[~fg_mask_ac] = 0
                
                compound = np.ones((X_s, Y_s, N_meas), dtype=np.float32)
                
                if getattr(art, 'learn_shell_gain', False):
                    sg = art.shell_gain.detach().cpu().numpy()
                    unique_b = sorted(set(int(round(b, -2)) for b in bvals))
                    for c_idx, bv in enumerate(bvals):
                        bv_round = int(round(bv, -2))
                        shell_idx = unique_b.index(min(unique_b, key=lambda x: abs(x - bv_round)))
                        compound[:, :, c_idx] *= sg[shell_idx]
                
                if getattr(art, 'learn_bias_field', False):
                    bias_full = F.interpolate(
                        art.bias_field_lr, size=(X, Y, Z),
                        mode='trilinear', align_corners=False,
                    )
                    bias_np = torch.exp(bias_full)[0, 0].detach().cpu().numpy()
                    if Z <= 1:
                        bias_2d = bias_np[:, :, 0]
                    else:
                        bias_2d = bias_np[:, :, Z // 2]
                    compound *= bias_2d[:, :, np.newaxis]
                
                if getattr(art, 'learn_eddy', False):
                    eddy_sc = torch.exp(art.log_eddy_scale).detach().cpu().numpy()
                    compound *= eddy_sc[np.newaxis, np.newaxis, :]
                
                corrected = sig_norm / np.maximum(compound, 1e-9)
                
                if getattr(art, 'learn_eddy', False):
                    eddy_bi = art.eddy_bias.detach().cpu().numpy()
                    corrected = corrected - eddy_bi[np.newaxis, np.newaxis, :] / np.maximum(compound, 1e-9)
                
                normed_mean_ac = sig_norm.mean(axis=-1)
                compound_mean = compound.mean(axis=-1)
                corrected_mean = corrected.mean(axis=-1)
                error_map = np.abs(sig_norm - corrected).mean(axis=-1)
                
                corrected_mean[~fg_mask_ac] = 0
                error_map[~fg_mask_ac] = 0
                
                bmask_2d = brain_mask[:, :, 0].T if brain_mask.ndim == 3 else brain_mask.T
                from scipy.ndimage import binary_fill_holes
                bmask_2d = binary_fill_holes(bmask_2d)
                
                inner_gs_ac = gs[row, :].subgridspec(1, 4, wspace=0.3)
                
                panels_data = [
                    (normed_mean_ac.T, 'Normalized S/S₀ (before)', 'gray', '#d62728'),
                    (compound_mean.T, 'Compound Artifacts', 'RdBu_r', '#1f77b4'),
                    (corrected_mean.T, 'Corrected S/S₀ (after)', 'gray', '#2ca02c'),
                    (error_map.T, '|Correction Effect|', 'gray', '#9467bd'),
                ]
                
                for col_idx, (data, title, cmap_name, color) in enumerate(panels_data):
                    ax_ac = fig.add_subplot(inner_gs_ac[0, col_idx])
                    
                    kwargs = {}
                    if 'Compound' in title and bmask_2d.any():
                        brain_vals = data[bmask_2d]
                        vmax_off = max(abs(brain_vals.max() - 1.0), abs(brain_vals.min() - 1.0))
                        vmax_off = max(vmax_off, 0.001)
                        kwargs['vmin'] = 1.0 - vmax_off
                        kwargs['vmax'] = 1.0 + vmax_off
                    
                    im_ac = ax_ac.imshow(data, cmap=cmap_name, origin='lower', aspect='equal', **kwargs)
                    
                    if bmask_2d.any():
                        brain_vals = data[bmask_2d]
                        stat = f'μ={brain_vals.mean():.3f} σ={brain_vals.std():.3f}'
                    else:
                        stat = ''
                    ax_ac.set_title(f'{title}\n{stat}', fontweight='bold', fontsize=7, color=color)
                    ax_ac.axis('off')
                    div_ac = make_axes_locatable(ax_ac)
                    plt.colorbar(im_ac, cax=div_ac.append_axes("right", size="5%", pad=0.05))
            
            row += 1
    
    title = f"Scanner Artifact Model - {dataset_name}" if dataset_name else "Scanner Artifact Model"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    panel_path = os.path.join(artifacts_dir, "artifacts_panel.png")
    plt.savefig(panel_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Visualization panel saved: {panel_path}")


def create_scanner(bvals, bvecs, n_fibers=3, device='cuda', 
                   use_restricted=True,
                   use_dot=False,
                   use_dispersion=True,
                   use_tortuosity=False,
                   use_kurtosis=False,
                   use_dki=False,
                   use_diffusivity_spectrum=True,
                   n_spectrum_components=5,
                   use_t2_weighting=False,
                   use_per_shell_modulation=False,
                   use_biexp_gm=False,
                   learn_diffusivities=False,
                   kappa_prior=8.0,
                   f_intra_prior=0.5,
                   use_curvature_physics=False,
                   curvature_alpha=10000.0,
                   curvature_epsilon=0.2,
                   voxel_size=(1.0, 1.0, 1.0),
                   use_mtcsd=False,
                   mtcsd_n_odf_dirs=60,
                   mtcsd_n_spectrum=5,
                   mtcsd_learn_responses=True,
                   learn_eddy=True,
                   learn_shell_gain=False,
                   learn_bias_field=True,
                   learn_warps=False,
                   use_spatial_noise=False,
                   use_psf=False,
                   psf_sigma=0.68,
                   psf_kernel_size=3,
                   use_ghosting=False,
                   use_qspace_mixing=False,
                   qspace_k_neighbors=6,
                   learn_channel_gain=False,
                   channel_gain_reg=0.1,
                   low_rank_channel_gain=False,
                   channel_gain_rank=5,
                   use_exchange=False,
                   exchange_tau=50.0,
                   exchange_pairs=None,
                   exchange_learnable=True,
                   use_tissue_priors=True,
                   learnable_tissue_priors=False,
                   learnable_prior_weight=0.001,
                   use_spatial_prior=False,
                   spatial_prior_weight=0.01,
                   spatial_prior_connectivity="6"):
    bvals_t = torch.tensor(bvals, dtype=torch.float32)
    bvecs_t = torch.tensor(bvecs, dtype=torch.float32)
    
    protocol = AcquisitionProtocolV4.from_tensors(
        bvals=bvals_t,
        bvecs=bvecs_t,
        te=89.0
    )
    
    scanner = DifferentiableScannerV4(
        protocol=protocol,
        n_fibers=n_fibers,
        use_restricted=use_restricted,
        use_dot=use_dot,
        use_dispersion=use_dispersion,
        use_tortuosity=use_tortuosity,
        use_kurtosis=use_kurtosis,
        use_dki=use_dki,
        use_diffusivity_spectrum=use_diffusivity_spectrum,
        n_spectrum_components=n_spectrum_components,
        use_t2_weighting=use_t2_weighting,
        use_per_shell_modulation=use_per_shell_modulation,
        use_biexp_gm=use_biexp_gm,
        learn_diffusivities=learn_diffusivities,
        use_curvature_physics=use_curvature_physics,
        curvature_alpha=curvature_alpha,
        curvature_epsilon=curvature_epsilon,
        voxel_size=voxel_size,
        use_mtcsd=use_mtcsd,
        n_odf_directions=mtcsd_n_odf_dirs,
        n_mtcsd_atoms=mtcsd_n_spectrum,
        learnable_responses=mtcsd_learn_responses,
        learn_eddy=learn_eddy,
        learn_shell_gain=learn_shell_gain,
        learn_bias_field=learn_bias_field,
        learn_warps=learn_warps,
        use_spatial_noise=use_spatial_noise,
        use_psf=use_psf,
        psf_sigma=psf_sigma,
        psf_kernel_size=psf_kernel_size,
        use_ghosting=use_ghosting,
        use_qspace_mixing=use_qspace_mixing,
        qspace_k_neighbors=qspace_k_neighbors,
        learn_channel_gain=learn_channel_gain,
        channel_gain_reg=channel_gain_reg,
        low_rank_channel_gain=low_rank_channel_gain,
        channel_gain_rank=channel_gain_rank,
    ).to(device)
    
    if use_curvature_physics:
        print(f"  [Curvature Physics] κ from geometry: α={curvature_alpha}, ε={curvature_epsilon}")
    
    scanner.use_tissue_priors = use_tissue_priors
    
    if kappa_prior != 8.0:
        scanner.signal_model.kappa_prior = torch.tensor(kappa_prior, device=device)
        print(f"  [Custom Prior] kappa = {kappa_prior:.1f} (default: 8.0)")
    if f_intra_prior != 0.5:
        scanner.signal_model.f_intra_prior = torch.tensor(f_intra_prior, device=device)
        print(f"  [Custom Prior] f_intra = {f_intra_prior:.2f} (default: 0.50)")
    
    if learnable_tissue_priors:
        scanner.set_learnable_tissue_priors(
            init_wm=0.50,
            init_gm=0.30,
            init_csf=0.10,
            prior_weight=learnable_prior_weight,
            spatial_mode='global',
        )
    
    if use_spatial_prior:
        scanner.set_spatial_prior(
            weight=spatial_prior_weight,
            connectivity=spatial_prior_connectivity,
            robust=True,
        )
    
    if use_exchange:
        if exchange_pairs is None:
            exchange_pairs = ['restricted_gm']
        print(f"\n  [RESEARCH] Enabling Kärger exchange:")
        print(f"    Exchange pairs: {exchange_pairs}")
        print(f"    Initial τ_ex: {exchange_tau:.1f} ms")
        print(f"    Learnable: {exchange_learnable}")
        scanner.set_exchange(
            tau_ex=exchange_tau,
            exchange_pairs=exchange_pairs,
            learnable=exchange_learnable
        )
    
    return scanner


def setup_signal_aware_priors(scanner: DifferentiableScannerV4, 
                               target_signal: torch.Tensor,
                               weight: float = 0.01,
                               wm_threshold: float = 0.2,
                               csf_threshold: float = 0.05):
    scanner.set_signal_aware_priors(
        signal=target_signal,
        wm_threshold=wm_threshold,
        csf_threshold=csf_threshold,
        weight=weight,
    )
    return scanner


def fit_patch(
    scanner: DifferentiableScannerV4,
    signal_patch: np.ndarray,
    mask_patch: np.ndarray,
    n_steps: int = 200,
    lr: float = 0.02,
    use_topology: bool = False,
    use_spatial_prior: bool = True,
    use_encoder: bool = False,
    encoder_epochs: int = 30,
    device: str = 'cuda',
    use_bochner: bool = False,
    bochner_weight: float = 0.01,
    bochner_m: int = 16,
    use_bernstein: bool = False,
    bernstein_weight: float = 0.01,
    bernstein_orders: int = 3,
    staged_training: bool = False,
    stage1_steps: int = 50,
    use_sigmoid_fracs: bool = False,
    sigmoid_scale: float = 0.5,
    use_rician: bool = False,
    rician_mode: str = 'hybrid',
    noise_sigma: float = None,
    hybrid_mse_weight: float = 0.7,
    huber_delta: float = 1.0,
    lambda_cosine: float = 0.05,
    mse_warmup_steps: int = 0,
    learn_noise_sigma: bool = False,
) -> dict:
    H, W, N = signal_patch.shape
    n_fibers = scanner.n_fibers
    
    if use_bochner:
        print(f"  Bochner constraint ENABLED (weight={bochner_weight}, m={bochner_m})")
    if use_bernstein:
        print(f"  Bernstein constraint ENABLED (weight={bernstein_weight}, orders={bernstein_orders})")
    if staged_training:
        print(f"  STAGED TRAINING ENABLED: Stage 1 = {stage1_steps} steps (global), Stage 2 = {n_steps - stage1_steps} steps (per-voxel)")
    
    b0_idx = scanner.protocol.bvals.cpu().numpy() < 100
    if b0_idx.sum() > 0:
        s0 = np.maximum(signal_patch[..., b0_idx].mean(axis=-1, keepdims=True), 1e-6)

        signal_norm = np.clip(signal_patch / s0, 0, 3)
    else:
        signal_norm = signal_patch / (signal_patch.max() + 1e-6)
    
    target = torch.tensor(signal_norm, dtype=torch.float32, device=device)
    target = target.unsqueeze(0).unsqueeze(-2)
    
    mask_t = torch.tensor(mask_patch, dtype=torch.bool, device=device)
    mask_t = mask_t.unsqueeze(0).unsqueeze(-1)
    
    B, X, Y, Z = 1, H, W, 1
    
    if use_encoder:
        print("  Training MLP encoder for warm-start initialization...")
        enc = MicrostructureEncoder(N, n_fibers, hidden_dim=128).to(device)
        enc = train_encoder_on_the_fly(enc, scanner, target, mask_t,
                                       n_epochs=encoder_epochs, lr=0.002, device=device)
        params = initialize_params_from_encoder(enc, target, mask_t, n_fibers, device)
    else:
        use_restricted = getattr(scanner, 'use_restricted', True)
        
        if use_restricted:
            raw_restricted_init = (torch.randn(B, X, Y, Z, device=device) * 0.5 - 0.5).requires_grad_(True)
        else:
            raw_restricted_init = torch.full((B, X, Y, Z), -10.0, device=device, requires_grad=False)
        
        params = {
            'raw_csf': (torch.randn(B, X, Y, Z, device=device) * 0.5 - 1.0).requires_grad_(True),
            'raw_gm': (torch.randn(B, X, Y, Z, device=device) * 0.5 - 0.5).requires_grad_(True),
            'raw_wm': (torch.randn(B, X, Y, Z, n_fibers, device=device) * 0.5 + 0.5).requires_grad_(True),
            'raw_restricted': raw_restricted_init,
            'fiber_dirs': sample_uniform_sphere_stratified(
                (B, X, Y, Z, n_fibers, 3), device=device
            ).requires_grad_(True),
            'f_intra': (torch.rand(B, X, Y, Z, device=device) * 0.3 + 0.5).requires_grad_(True),
            'kappa': (torch.ones(B, X, Y, Z, device=device) * 8.0).requires_grad_(True),
            's0': (torch.ones(B, X, Y, Z, device=device)).requires_grad_(True),
        }
    
    
    scanner.use_tissue_priors = use_topology
    
    
    voxel_param_keys = ['raw_csf', 'raw_gm', 'raw_wm', 'fiber_dirs', 'f_intra', 'kappa', 's0']
    if params['raw_restricted'].requires_grad:
        voxel_param_keys.append('raw_restricted')
    
    scanner_params_list = list(scanner.parameters())
    
    if hasattr(scanner, 'signal_model') and hasattr(scanner.signal_model, 'exchange_module'):
        if scanner.signal_model.exchange_module is not None:
            scanner_params_list.extend(scanner.signal_model.exchange_module.parameters())
            print(f"  [Exchange] Added τ_ex to optimizer (learnable)")
    
    if staged_training:
        print(f"  [Staged] Stage 1: Learning {len(scanner_params_list)} global response params")
        optimizer = torch.optim.Rprop(scanner_params_list, lr=lr)
        current_stage = 1
    else:
        optim_params = [p for p in params.values() if p.requires_grad] + scanner_params_list
        optimizer = torch.optim.Rprop(optim_params, lr=lr)
        current_stage = 0
    
    losses = []
    best_loss = float('inf')
    patience_counter = 0
    patience = 999999
    tol = 0.00000001
    
    pbar = tqdm(range(n_steps), desc="Fitting patch")
    
    sigmoid_scale_val = sigmoid_scale
    
    for step in pbar:
        if staged_training and current_stage == 1 and step == stage1_steps:
            print(f"\n  [Staged] Transitioning to Stage 2: Learning per-voxel params")
            for p in scanner_params_list:
                p.requires_grad_(False)
            optimizer = torch.optim.Rprop(list(params.values()), lr=lr)
            current_stage = 2
        
        optimizer.zero_grad()
        
        if use_sigmoid_fracs:
            f_csf_raw = sigmoid_scale_val * torch.sigmoid(params['raw_csf'])
            f_gm_raw = sigmoid_scale_val * torch.sigmoid(params['raw_gm'])
            f_wm_raw = sigmoid_scale_val * torch.sigmoid(params['raw_wm'])
            f_restricted_raw = sigmoid_scale_val * torch.sigmoid(params['raw_restricted'])
            
            total = f_csf_raw + f_gm_raw + f_wm_raw.sum(dim=-1) + f_restricted_raw
            scale = torch.clamp(total, min=1.0)
            
            f_csf = f_csf_raw / scale
            f_gm = f_gm_raw / scale
            f_wm = f_wm_raw / scale.unsqueeze(-1)
            f_restricted = f_restricted_raw / scale
        else:
            tissue_raw = torch.cat([
                params['raw_csf'].unsqueeze(-1),
                params['raw_gm'].unsqueeze(-1),
                params['raw_wm'],
                params['raw_restricted'].unsqueeze(-1),
            ], dim=-1)
            tissue_fracs = F.softmax(tissue_raw, dim=-1)
            
            f_csf = tissue_fracs[..., 0]
            f_gm = tissue_fracs[..., 1]
            f_wm = tissue_fracs[..., 2:2+n_fibers]
            f_restricted = tissue_fracs[..., -1]
        
        f_wm_total = f_wm.sum(dim=-1) if f_wm.dim() > len(f_csf.shape) else f_wm
        kappa_base = F.softplus(params['kappa'])
        kappa_wm_coupling = 5.0
        kappa = kappa_base + kappa_wm_coupling * f_wm_total + 0.1
        
        scanner_params = {
            'f_csf': f_csf,
            'f_gm': f_gm,
            'f_wm': f_wm,
            'f_restricted': f_restricted,
            'fiber_dirs': F.normalize(params['fiber_dirs'], dim=-1),
            'f_intra': torch.sigmoid(params['f_intra']),
            'kappa': kappa,
            's0': F.softplus(params['s0']),
        }
        
        
        pred = scanner(scanner_params, add_noise=False)
        
        mask_expanded = mask_t.unsqueeze(-1).expand_as(pred)
        
        use_rician_this_step = use_rician and (step >= mse_warmup_steps)
        if use_rician_this_step and noise_sigma is not None:
            sigma_t = torch.tensor(noise_sigma, dtype=torch.float32, device=device)
            if rician_mode == 'nll':
                nll = rician_nll(target, pred, sigma_t)
                loss = nll[mask_expanded].mean()
            elif rician_mode == 'hybrid':
                loss = hybrid_mse_rician_loss(
                    pred, target, sigma_t,
                    mse_weight=hybrid_mse_weight,
                    rician_weight=1.0 - hybrid_mse_weight,
                    mask=mask_t
                )
            else:
                loss = F.mse_loss(pred[mask_expanded], target[mask_expanded])
        else:
            loss = F.mse_loss(pred[mask_expanded], target[mask_expanded])
        
        reg_params = {'f_csf': f_csf, 'f_gm': f_gm, 'f_wm': f_wm, 'f_restricted': f_restricted}
        reg_loss = scanner.get_regularization_loss(reg_params, w_micro=0.1, mask=mask_t)
        
        with torch.no_grad():
            signal_mean = target.mean(dim=-1, keepdim=True)
            signal_std = target.std(dim=-1)
            cv = signal_std / (signal_mean.squeeze(-1) + 1e-6)
            cv_scaled = ((cv - 0.1) / 0.3).clamp(0, 1)
            aniso_weight = cv_scaled
        
        bvals_np = scanner.protocol.bvals.cpu().numpy()
        aniso_matching_loss = torch.tensor(0.0, device=pred.device)
        
        unique_bvals = np.unique(bvals_np[bvals_np > 100])
        for b in unique_bvals:
            shell_mask = np.abs(bvals_np - b) < 100
            if shell_mask.sum() < 6:
                continue
            
            pred_shell = pred[..., shell_mask]
            target_shell = target[..., shell_mask]
            
            pred_mean = pred_shell.mean(dim=-1) + 1e-6
            pred_std = pred_shell.std(dim=-1)
            pred_cv = pred_std / pred_mean
            
            target_mean = target_shell.mean(dim=-1) + 1e-6
            target_std = target_shell.std(dim=-1)
            target_cv = target_std / target_mean
            
            cv_diff = (pred_cv - target_cv) ** 2
            aniso_matching_loss = aniso_matching_loss + (cv_diff * mask_t.float()).sum() / (mask_t.float().sum() + 1e-6)
        
        if len(unique_bvals) > 0:
            aniso_matching_loss = aniso_matching_loss / len(unique_bvals)
        aniso_matching_weight = 0.1
            
        with torch.no_grad():
            bvals_np = scanner.protocol.bvals.cpu().numpy()
            high_b_mask = bvals_np >= 2000
            if high_b_mask.sum() > 0:
                high_b_signal = target[..., high_b_mask].mean(dim=-1)
                wm_from_highb = (high_b_signal > 0.15).float()
                gm_from_highb = (high_b_signal < 0.10).float()
            else:
                wm_from_highb = torch.zeros_like(aniso_weight)
                gm_from_highb = torch.zeros_like(aniso_weight)
        
        f_wm_total = f_wm.sum(dim=-1) if f_wm.dim() > 4 else f_wm
        iso_penalty = ((1 - aniso_weight) * f_wm_total * mask_t.float()).mean()
        aniso_prior_weight = 0.01
        
        highb_wm_consistency = ((1 - wm_from_highb) * (f_wm_total + f_restricted) * mask_t.float()).mean()
        highb_gm_consistency = ((1 - gm_from_highb) * f_gm * mask_t.float()).mean()
        highb_weight = 0.0
        
        if f_wm.dim() > 4 and f_wm.shape[-1] > 1:
            f_wm_sum = f_wm.sum(dim=-1)
            f_wm_sq_sum = (f_wm ** 2).sum(dim=-1)
            spread = f_wm_sum ** 2 - f_wm_sq_sum
            conc_penalty = ((1 - aniso_weight) * spread * mask_t.float()).mean()
            conc_weight = 0.02
        else:
            conc_penalty = 0.0
            conc_weight = 0.0
        
        if f_wm.dim() > 4 and f_wm.shape[-1] > 1:
            fiber_dirs_norm = F.normalize(params['fiber_dirs'], dim=-1)
            repel_loss = 0.0
            n_fib = f_wm.shape[-1]
            for i in range(n_fib):
                for j in range(i + 1, n_fib):
                    dot_ij = (fiber_dirs_norm[..., i, :] * fiber_dirs_norm[..., j, :]).sum(dim=-1)
                    f_prod = f_wm[..., i] * f_wm[..., j]
                    repel_loss = repel_loss + (f_prod * torch.abs(dot_ij) * mask_t.float()).mean()
            repel_weight = 0.01
        else:
            repel_loss = 0.0
            repel_weight = 0.0
        
        if use_bochner:
            bochner_loss = scanner.get_bochner_loss(
                pred, scanner_params, m=bochner_m, penalty_type='min_eigenvalue'
            )
        else:
            bochner_loss = 0.0
        
        if use_bernstein:
            bernstein_loss = scanner.get_bernstein_loss(
                pred, n_orders=bernstein_orders, mask=mask_t
            )
        else:
            bernstein_loss = 0.0
        
        gm_restricted_penalty = (f_gm * f_restricted * mask_t.float()).mean()
        gm_restricted_weight = 0.0
        
        f_wm_related = f_wm_total + f_restricted
        gm_wm_exclusion = (f_gm * f_wm_related * mask_t.float()).mean()
        gm_wm_exclusion_weight = 0.0
        
        total_loss = (loss + reg_loss + aniso_prior_weight * iso_penalty + 
                      conc_weight * conc_penalty + repel_weight * repel_loss + 
                      bochner_weight * bochner_loss + bernstein_weight * bernstein_loss +
                      gm_restricted_weight * gm_restricted_penalty +
                      gm_wm_exclusion_weight * gm_wm_exclusion +
                      highb_weight * (highb_wm_consistency + highb_gm_consistency) +
                      aniso_matching_weight * aniso_matching_loss)
        
        total_loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        postfix = {'MSE': f'{loss.item():.5f}'}
        if staged_training:
            postfix['Stage'] = current_stage
        if use_bochner and isinstance(bochner_loss, torch.Tensor):
            postfix['Boch'] = f'{bochner_loss.item():.4f}'
        if use_bernstein and isinstance(bernstein_loss, torch.Tensor):
            postfix['Bern'] = f'{bernstein_loss.item():.4f}'
        if isinstance(aniso_matching_loss, torch.Tensor) and step % 50 == 0:
            postfix['AnisoCV'] = f'{aniso_matching_loss.item():.4f}'
        if scanner.learn_diffusivities and step % 100 == 0:
            diff_vals = scanner.get_learned_diffusivities()
            postfix['d_gm'] = f'{diff_vals["d_gm"]:.2f}'
            postfix['d_perp'] = f'{diff_vals["d_perp"]:.2f}'
        pbar.set_postfix(postfix)
        
        current_loss = round(loss.item(), 3)
        if current_loss < round(best_loss, 3) - tol:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            pbar.set_description(f"Early stop @ epoch {step}")
            break
    
    with torch.no_grad():
        if use_sigmoid_fracs:
            f_csf_raw = sigmoid_scale_val * torch.sigmoid(params['raw_csf'])
            f_gm_raw = sigmoid_scale_val * torch.sigmoid(params['raw_gm'])
            f_wm_raw = sigmoid_scale_val * torch.sigmoid(params['raw_wm'])
            f_restricted_raw = sigmoid_scale_val * torch.sigmoid(params['raw_restricted'])
            
            total = f_csf_raw + f_gm_raw + f_wm_raw.sum(dim=-1) + f_restricted_raw
            scale = torch.clamp(total, min=1.0)
            
            f_csf_tmp = f_csf_raw / scale
            f_gm_tmp = f_gm_raw / scale
            f_wm_tmp = (f_wm_raw / scale.unsqueeze(-1)).sum(dim=-1)
            f_restricted_tmp = f_restricted_raw / scale
            f_residual_tmp = torch.clamp(1.0 - (f_csf_tmp + f_gm_tmp + f_wm_tmp + f_restricted_tmp), min=0.0)
        else:
            tissue_raw = torch.cat([
                params['raw_csf'].unsqueeze(-1),
                params['raw_gm'].unsqueeze(-1),
                params['raw_wm'],
                params['raw_restricted'].unsqueeze(-1),
            ], dim=-1)
            tissue_fracs = F.softmax(tissue_raw, dim=-1)
            
            f_csf_tmp = tissue_fracs[..., 0]
            f_gm_tmp = tissue_fracs[..., 1]
            f_wm_tmp = tissue_fracs[..., 2:2+n_fibers].sum(dim=-1)
            f_restricted_tmp = tissue_fracs[..., -1]
            f_residual_tmp = torch.zeros_like(f_csf_tmp)
        
        m = mask_t.bool()
        csf_vals = f_csf_tmp[m].cpu().numpy()
        gm_vals = f_gm_tmp[m].cpu().numpy()
        wm_vals = f_wm_tmp[m].cpu().numpy()
        res_vals = f_restricted_tmp[m].cpu().numpy()
        resid_vals = f_residual_tmp[m].cpu().numpy()
        
        print(f"\n  === TISSUE FRACTION STATS {'(SIGMOID mode)' if use_sigmoid_fracs else '(SOFTMAX mode)'} ===")
        print(f"  CSF:        {csf_vals.mean():.3f} ± {csf_vals.std():.3f}")
        print(f"  GM:         {gm_vals.mean():.3f} ± {gm_vals.std():.3f}")
        print(f"  WM (total): {wm_vals.mean():.3f} ± {wm_vals.std():.3f}")
        print(f"  Restricted: {res_vals.mean():.3f} ± {res_vals.std():.3f}")
        if use_sigmoid_fracs:
            print(f"  Residual:   {resid_vals.mean():.3f} ± {resid_vals.std():.3f}  <- unexplained signal")
        
        if scanner.learn_diffusivities:
            diff_vals = scanner.get_learned_diffusivities()
            print(f"\n  === LEARNED DIFFUSIVITY PRIORS ===")
            print(f"  d_gm (fast):       {diff_vals['d_gm']:.3f} × 10⁻³ mm²/s")
            print(f"  d_gm_slow (restr): {diff_vals['d_gm_slow']:.4f} × 10⁻³ mm²/s")
            print(f"  f_gm_restricted:   {diff_vals['f_gm_restricted']:.3f}")
            print(f"  d_parallel (WM):   {diff_vals['d_parallel']:.3f} × 10⁻³ mm²/s")
            print(f"  d_perp (WM):       {diff_vals['d_perp']:.3f} × 10⁻³ mm²/s")
        
        wm_related = wm_vals + res_vals
        
        tissues_3class = np.stack([csf_vals, gm_vals, wm_related], axis=-1)
        dominant_3class = np.argmax(tissues_3class, axis=-1)
        names_3class = ['CSF', 'GM', 'WM-related']
        print(f"\n  === DOMINANT TISSUE (3-class, WM+Restricted combined) ===")
        for i, name in enumerate(names_3class):
            pct = (dominant_3class == i).mean() * 100
            print(f"  {name}: {pct:.1f}%")
        
        tissues_4class = np.stack([csf_vals, gm_vals, wm_vals, res_vals], axis=-1)
        dominant_4class = np.argmax(tissues_4class, axis=-1)
        names_4class = ['CSF', 'GM', 'WM', 'Restricted']
        print(f"\n  === DOMINANT TISSUE (4-class, detailed) ===")
        for i, name in enumerate(names_4class):
            pct = (dominant_4class == i).mean() * 100
            print(f"  {name}: {pct:.1f}%")
        
        tissue_stats = {
            'csf_mean': float(csf_vals.mean()),
            'csf_std': float(csf_vals.std()),
            'gm_mean': float(gm_vals.mean()),
            'gm_std': float(gm_vals.std()),
            'wm_mean': float(wm_vals.mean()),
            'wm_std': float(wm_vals.std()),
            'restricted_mean': float(res_vals.mean()),
            'restricted_std': float(res_vals.std()),
            'dominant_csf_pct': float((dominant_3class == 0).mean() * 100),
            'dominant_gm_pct': float((dominant_3class == 1).mean() * 100),
            'dominant_wm_pct': float((dominant_3class == 2).mean() * 100),
            'dominant_wm_only_pct': float((dominant_4class == 2).mean() * 100),
            'dominant_restricted_pct': float((dominant_4class == 3).mean() * 100),
        }
    
    with torch.no_grad():
        if use_sigmoid_fracs:
            f_csf_raw = sigmoid_scale_val * torch.sigmoid(params['raw_csf'])
            f_gm_raw = sigmoid_scale_val * torch.sigmoid(params['raw_gm'])
            f_wm_raw = sigmoid_scale_val * torch.sigmoid(params['raw_wm'])
            f_restricted_raw = sigmoid_scale_val * torch.sigmoid(params['raw_restricted'])
            
            total = f_csf_raw + f_gm_raw + f_wm_raw.sum(dim=-1) + f_restricted_raw
            scale = torch.clamp(total, min=1.0)
            
            final_params = {
                'f_csf': f_csf_raw / scale,
                'f_gm': f_gm_raw / scale,
                'f_wm': f_wm_raw / scale.unsqueeze(-1),
                'f_restricted': f_restricted_raw / scale,
                'f_residual': torch.clamp(1.0 - (f_csf_raw/scale + f_gm_raw/scale + (f_wm_raw/scale.unsqueeze(-1)).sum(dim=-1) + f_restricted_raw/scale), min=0.0),
                'fiber_dirs': F.normalize(params['fiber_dirs'], dim=-1),
                'f_intra': torch.sigmoid(params['f_intra']),
                'kappa': F.softplus(params['kappa']) + 0.1,
                's0': F.softplus(params['s0']),
            }
        else:
            tissue_raw = torch.cat([
                params['raw_csf'].unsqueeze(-1),
                params['raw_gm'].unsqueeze(-1),
                params['raw_wm'],
                params['raw_restricted'].unsqueeze(-1),
            ], dim=-1)
            tissue_fracs = F.softmax(tissue_raw, dim=-1)
            
            final_params = {
                'f_csf': tissue_fracs[..., 0],
                'f_gm': tissue_fracs[..., 1],
                'f_wm': tissue_fracs[..., 2:2+n_fibers],
                'f_restricted': tissue_fracs[..., -1],
                'fiber_dirs': F.normalize(params['fiber_dirs'], dim=-1),
                'f_intra': torch.sigmoid(params['f_intra']),
                'kappa': F.softplus(params['kappa']) + 0.1,
                's0': F.softplus(params['s0']),
            }
        
        pred_signal = scanner(final_params, add_noise=False)
        final_params['pred_signal'] = pred_signal
        final_params['target_signal'] = target
        final_params['mask'] = mask_t
        final_params['mse'] = losses[-1]
        final_params['losses'] = losses
        final_params['tissue_stats'] = tissue_stats
        
        if getattr(scanner, 'learnable_tissue_priors', False):
            learned_priors = scanner.get_learned_tissue_priors()
            final_params['learned_tissue_priors'] = learned_priors
            print(f"\n  [Learned Priors] Final tissue priors:")
            print(f"    WM:  {learned_priors['wm']:.1%} (init: 50%)")
            print(f"    GM:  {learned_priors['gm']:.1%} (init: 30%)")
            print(f"    CSF: {learned_priors['csf']:.1%} (init: 10%)")
        
        final_params = sort_fibers_by_direction_coherence(final_params, mask_t)
    
    return final_params


def sort_fibers_by_direction_coherence(params, mask):
    f_wm = params['f_wm']
    fiber_dirs = params['fiber_dirs']
    
    orig_shape = f_wm.shape
    *batch_dims, n_fibers = f_wm.shape
    device = f_wm.device
    
    f_wm_2d = f_wm.squeeze().cpu().numpy()
    fiber_dirs_2d = fiber_dirs.squeeze().cpu().numpy()
    
    if mask.dim() == 2:
        mask_2d = mask.cpu().numpy().astype(bool)
    else:
        mask_2d = mask.squeeze().cpu().numpy().astype(bool)
    
    if mask_2d.ndim == 3:
        return params
    
    H, W = mask_2d.shape
    
    for i in range(H):
        for j in range(W):
            if mask_2d[i, j]:
                for f in range(n_fibers):
                    d = fiber_dirs_2d[i, j, f]
                    if d[2] < 0 or (d[2] == 0 and d[1] < 0) or (d[2] == 0 and d[1] == 0 and d[0] < 0):
                        fiber_dirs_2d[i, j, f] = -d
    
    for i in range(H):
        for j in range(W):
            if mask_2d[i, j]:
                order = np.argsort(f_wm_2d[i, j])[::-1]
                f_wm_2d[i, j] = f_wm_2d[i, j, order]
                fiber_dirs_2d[i, j] = fiber_dirs_2d[i, j, order]
    
    def get_neighbors(i, j):
        neighbors = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and mask_2d[ni, nj]:
                neighbors.append((ni, nj))
        return neighbors
    
    def compute_coherence(i, j):
        total = 0.0
        for ni, nj in get_neighbors(i, j):
            for f in range(n_fibers):
                if f_wm_2d[i, j, f] < 0.03 or f_wm_2d[ni, nj, f] < 0.03:
                    continue
                d1 = fiber_dirs_2d[i, j, f]
                d2 = fiber_dirs_2d[ni, nj, f]
                dot = np.abs(np.dot(d1, d2))
                total += dot * min(f_wm_2d[i, j, f], f_wm_2d[ni, nj, f])
        return total
    
    n_iterations = 3
    for iteration in range(n_iterations):
        n_swaps = 0
        for i in range(H):
            for j in range(W):
                if not mask_2d[i, j]:
                    continue
                
                best_swap = None
                best_improvement = 0
                
                current_coherence = compute_coherence(i, j)
                
                for f1 in range(n_fibers):
                    for f2 in range(f1 + 1, n_fibers):
                        w1, w2 = f_wm_2d[i, j, f1], f_wm_2d[i, j, f2]
                        if w1 < 0.02 and w2 < 0.02:
                            continue
                        if max(w1, w2) > 2 * min(w1, w2) + 0.01:
                            continue
                        
                        f_wm_2d[i, j, f1], f_wm_2d[i, j, f2] = f_wm_2d[i, j, f2], f_wm_2d[i, j, f1]
                        fiber_dirs_2d[i, j, f1], fiber_dirs_2d[i, j, f2] = fiber_dirs_2d[i, j, f2].copy(), fiber_dirs_2d[i, j, f1].copy()
                        
                        new_coherence = compute_coherence(i, j)
                        improvement = new_coherence - current_coherence
                        
                        f_wm_2d[i, j, f1], f_wm_2d[i, j, f2] = f_wm_2d[i, j, f2], f_wm_2d[i, j, f1]
                        fiber_dirs_2d[i, j, f1], fiber_dirs_2d[i, j, f2] = fiber_dirs_2d[i, j, f2].copy(), fiber_dirs_2d[i, j, f1].copy()
                        
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_swap = (f1, f2)
                
                if best_swap is not None:
                    f1, f2 = best_swap
                    f_wm_2d[i, j, f1], f_wm_2d[i, j, f2] = f_wm_2d[i, j, f2], f_wm_2d[i, j, f1]
                    fiber_dirs_2d[i, j, f1], fiber_dirs_2d[i, j, f2] = fiber_dirs_2d[i, j, f2].copy(), fiber_dirs_2d[i, j, f1].copy()
                    n_swaps += 1
        
        if n_swaps == 0:
            break
    
    reordered_f_wm_t = torch.tensor(f_wm_2d, dtype=f_wm.dtype, device=device)
    reordered_dirs_t = torch.tensor(fiber_dirs_2d, dtype=fiber_dirs.dtype, device=device)
    
    params['f_wm'] = reordered_f_wm_t.reshape(orig_shape)
    params['fiber_dirs'] = reordered_dirs_t.reshape(*orig_shape, 3)
    
    return params


def sort_fibers_by_weight(params):
    f_wm = params['f_wm']
    fiber_dirs = params['fiber_dirs']
    
    sort_idx = torch.argsort(f_wm, dim=-1, descending=True)
    
    params['f_wm'] = torch.gather(f_wm, -1, sort_idx)
    
    sort_idx_dirs = sort_idx.unsqueeze(-1).expand_as(fiber_dirs)
    params['fiber_dirs'] = torch.gather(fiber_dirs, -2, sort_idx_dirs)
    
    return params


def reorder_fibers_for_smoothness(fiber_dirs, f_wm, mask):
    H, W, n_fibers, _ = fiber_dirs.shape
    
    reordered_dirs = np.zeros_like(fiber_dirs)
    reordered_f_wm = np.zeros_like(f_wm)
    
    mask_ys, mask_xs = np.where(mask)
    n_voxels = len(mask_ys)
    
    if n_voxels == 0:
        return fiber_dirs, f_wm
    
    for idx in range(n_voxels):
        y, x = mask_ys[idx], mask_xs[idx]
        
        order = np.argsort(f_wm[y, x])[::-1]
        
        for out_slot, in_f in enumerate(order):
            d = fiber_dirs[y, x, in_f].copy()
            if d[2] < 0 or (d[2] == 0 and d[1] < 0) or (d[2] == 0 and d[1] == 0 and d[0] < 0):
                d = -d
            reordered_dirs[y, x, out_slot] = d
            reordered_f_wm[y, x, out_slot] = f_wm[y, x, in_f]
    
    return reordered_dirs, reordered_f_wm


def create_r2_mask_comparison(
    r2_map: np.ndarray,
    provided_mask: np.ndarray,
    output_path: str,
    thresholds: list = [0.5, 0.7, 0.9],
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
    n_thresh = len(thresholds)
    fig, axes = plt.subplots(2, n_thresh + 2, figsize=(4 * (n_thresh + 2), 8), facecolor='white')
    
    im = axes[0, 0].imshow(r2_map.T, cmap='RdYlGn', vmin=0, vmax=1, origin='lower')
    axes[0, 0].set_title('R² Fit Quality', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046)
    
    axes[0, 1].imshow(provided_mask.astype(float).T, cmap='gray', vmin=0, vmax=1, origin='lower')
    axes[0, 1].set_title('Provided Mask', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    n_provided = provided_mask.sum()
    axes[0, 1].text(0.5, -0.05, f'{n_provided:,} voxels', 
                    transform=axes[0, 1].transAxes, ha='center', fontsize=10)
    
    for i, thresh in enumerate(thresholds):
        r2_mask = ~np.isnan(r2_map) & (r2_map >= thresh)
        axes[0, i + 2].imshow(r2_mask.astype(float).T, cmap='gray', vmin=0, vmax=1, origin='lower')
        axes[0, i + 2].set_title(f'R² > {thresh}', fontsize=12, fontweight='bold')
        axes[0, i + 2].axis('off')
        n_r2 = r2_mask.sum()
        axes[0, i + 2].text(0.5, -0.05, f'{n_r2:,} voxels', 
                           transform=axes[0, i + 2].transAxes, ha='center', fontsize=10)
    
    r2_valid = r2_map[~np.isnan(r2_map)]
    axes[1, 0].hist(r2_valid, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    for thresh in thresholds:
        axes[1, 0].axvline(thresh, color='red', linestyle='--', linewidth=2, label=f'R²={thresh}')
    axes[1, 0].set_xlabel('R²', fontsize=11)
    axes[1, 0].set_ylabel('Count', fontsize=11)
    axes[1, 0].set_title('R² Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    
    colors = ['black', 'blue', 'red', 'green']
    cmap = ListedColormap(colors)
    
    for i, thresh in enumerate(thresholds):
        r2_mask = ~np.isnan(r2_map) & (r2_map >= thresh)
        
        overlap = np.zeros_like(r2_map, dtype=int)
        overlap[r2_mask & ~provided_mask] = 1
        overlap[~r2_mask & provided_mask] = 2
        overlap[r2_mask & provided_mask] = 3
        
        axes[1, i + 1].imshow(overlap.T, cmap=cmap, vmin=0, vmax=3, origin='lower')
        axes[1, i + 1].set_title(f'R²>{thresh} vs Provided', fontsize=12, fontweight='bold')
        axes[1, i + 1].axis('off')
        
        both = (r2_mask & provided_mask).sum()
        r2_only = (r2_mask & ~provided_mask).sum()
        prov_only = (~r2_mask & provided_mask).sum()
        dice = 2 * both / (r2_mask.sum() + provided_mask.sum() + 1e-8)
        iou = both / (r2_mask.sum() + provided_mask.sum() - both + 1e-8)
        
        stats_text = f'Dice={dice:.3f}\nIoU={iou:.3f}\nR²-only={r2_only}\nProv-only={prov_only}'
        axes[1, i + 1].text(0.02, 0.98, stats_text, transform=axes[1, i + 1].transAxes,
                           fontsize=9, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    legend_ax = axes[1, -1]
    legend_ax.axis('off')
    legend_items = [
        ('Black', 'Neither mask'),
        ('Blue', 'R² mask only'),
        ('Red', 'Provided mask only'),
        ('Green', 'Both agree'),
    ]
    y = 0.8
    for color, label in legend_items:
        legend_ax.text(0.1, y, '■', fontsize=20, color=color.lower(), 
                       transform=legend_ax.transAxes, verticalalignment='center')
        legend_ax.text(0.25, y, label, fontsize=11, transform=legend_ax.transAxes,
                       verticalalignment='center')
        y -= 0.15
    legend_ax.set_title('Legend', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_3axis_visualization(
    f_csf, f_gm, f_wm, f_restricted, fiber_dirs, mse, mask,
    output_path, dataset_name="", 
    axial_slice=None, coronal_slice=None, sagittal_slice=None
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
    X, Y, Z = f_csf.shape
    
    if axial_slice is None:
        axial_slice = Z // 2
    if coronal_slice is None:
        coronal_slice = Y // 2
    if sagittal_slice is None:
        sagittal_slice = X // 2
    
    fig, axes = plt.subplots(3, 6, figsize=(24, 12), facecolor='black')
    fig.suptitle(f'Whole Brain Microstructure - {dataset_name}' if dataset_name else 'Whole Brain Microstructure', 
                 fontsize=14, fontweight='bold', color='white')
    
    for ax in axes.flat:
        ax.set_facecolor('black')
        ax.axis('off')
    
    def get_cmap_black_bad(name):
        cmap = plt.cm.get_cmap(name).copy()
        cmap.set_bad(color='black')
        return cmap
    
    def masked_slice(vol, slc, axis, msk):
        if axis == 0:
            data = vol[slc, :, :].T
            m = msk[slc, :, :].T
        elif axis == 1:
            data = vol[:, slc, :].T
            m = msk[:, slc, :].T
        else:
            data = vol[:, :, slc].T
            m = msk[:, :, slc].T
        return np.ma.masked_where(~m.astype(bool), data)
    
    def fiber_rgb(dirs_vol, slc, axis, msk):
        if axis == 0:
            d = dirs_vol[slc, :, :, 0, :]
            m = msk[slc, :, :]
        elif axis == 1:
            d = dirs_vol[:, slc, :, 0, :]
            m = msk[:, slc, :]
        else:
            d = dirs_vol[:, :, slc, 0, :]
            m = msk[:, :, slc]
        
        rgb = np.abs(d)
        rgb = np.clip(rgb, 0, 1)
        if axis in [0, 1]:
            rgb = np.transpose(rgb, (1, 0, 2))
            m = m.T
        else:
            rgb = np.transpose(rgb, (1, 0, 2))
            m = m.T
        rgb[~m.astype(bool)] = 0
        return rgb
    
    row_labels = ['Axial', 'Coronal', 'Sagittal']
    col_labels = ['f_CSF', 'f_GM', 'f_WM', 'f_Restricted', 'Fiber RGB', 'MSE']
    slices = [axial_slice, coronal_slice, sagittal_slice]
    axis_map = [2, 1, 0]
    
    for row, (ax_type, slc) in enumerate(zip(axis_map, slices)):
        im = axes[row, 0].imshow(masked_slice(f_csf, slc, ax_type, mask), 
                                  cmap=get_cmap_black_bad('Blues'), vmin=0, vmax=0.8, origin='lower')
        if row == 0:
            axes[row, 0].set_title(col_labels[0], color='white', fontsize=12)
        axes[row, 0].set_ylabel(row_labels[row], color='white', fontsize=12)
        
        axes[row, 1].imshow(masked_slice(f_gm, slc, ax_type, mask), 
                            cmap=get_cmap_black_bad('Greens'), vmin=0, vmax=0.8, origin='lower')
        if row == 0:
            axes[row, 1].set_title(col_labels[1], color='white', fontsize=12)
        
        axes[row, 2].imshow(masked_slice(f_wm, slc, ax_type, mask), 
                            cmap=get_cmap_black_bad('plasma'), vmin=0, vmax=1.0, origin='lower')
        if row == 0:
            axes[row, 2].set_title(col_labels[2], color='white', fontsize=12)
        
        axes[row, 3].imshow(masked_slice(f_restricted, slc, ax_type, mask), 
                            cmap=get_cmap_black_bad('Purples'), vmin=0, vmax=0.5, origin='lower')
        if row == 0:
            axes[row, 3].set_title(col_labels[3], color='white', fontsize=12)
        
        axes[row, 4].imshow(fiber_rgb(fiber_dirs, slc, ax_type, mask), origin='lower')
        if row == 0:
            axes[row, 4].set_title(col_labels[4], color='white', fontsize=12)
        
        axes[row, 5].imshow(masked_slice(mse, slc, ax_type, mask), 
                            cmap=get_cmap_black_bad('hot'), vmin=0, vmax=0.05, origin='lower')
        if row == 0:
            axes[row, 5].set_title(col_labels[5], color='white', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Saved: {output_path}")


def create_3axis_panels_from_volumes(
    args, f_csf_vol, f_gm_vol, f_wm_vol, f_wm_fibers_vol, f_restricted_vol, 
    f_intra_vol, fiber_dirs_vol, mse_vol, mask, dwi, bvals, n_fibers, dataset_name=""
):
    X, Y, Z = f_csf_vol.shape
    
    slices = {
        'axial': (Z // 2, 2),
        'coronal': (Y // 2, 1),
        'sagittal': (X // 2, 0),
    }
    
    mask_provided = getattr(args, 'mask_provided', False)
    output_r2_thresh = getattr(args, 'output_r2_mask', None)
    output_rel_res_thresh = getattr(args, 'output_rel_res_mask', None)
    use_output_rel_res = output_rel_res_thresh is not None
    use_output_r2 = output_r2_thresh is not None
    
    for axis_name, (slice_idx, axis) in slices.items():
        print(f"  Creating {axis_name} panels (slice {slice_idx})...")
        
        if axis == 2:
            f_csf_2d = f_csf_vol[:, :, slice_idx]
            f_gm_2d = f_gm_vol[:, :, slice_idx]
            f_wm_2d = f_wm_vol[:, :, slice_idx]
            f_wm_fibers_2d = f_wm_fibers_vol[:, :, slice_idx, :]
            f_restricted_2d = f_restricted_vol[:, :, slice_idx]
            f_intra_2d = f_intra_vol[:, :, slice_idx]
            fiber_dirs_2d = fiber_dirs_vol[:, :, slice_idx, :, :]
            mask_2d = mask[:, :, slice_idx]
            dwi_2d = dwi[:, :, slice_idx, :]
        elif axis == 1:
            f_csf_2d = f_csf_vol[:, slice_idx, :].T
            f_gm_2d = f_gm_vol[:, slice_idx, :].T
            f_wm_2d = f_wm_vol[:, slice_idx, :].T
            f_wm_fibers_2d = np.transpose(f_wm_fibers_vol[:, slice_idx, :, :], (1, 0, 2))
            f_restricted_2d = f_restricted_vol[:, slice_idx, :].T
            f_intra_2d = f_intra_vol[:, slice_idx, :].T
            fiber_dirs_2d = np.transpose(fiber_dirs_vol[:, slice_idx, :, :, :], (1, 0, 2, 3))
            mask_2d = mask[:, slice_idx, :].T
            dwi_2d = np.transpose(dwi[:, slice_idx, :, :], (1, 0, 2))
        else:
            f_csf_2d = f_csf_vol[slice_idx, :, :].T
            f_gm_2d = f_gm_vol[slice_idx, :, :].T
            f_wm_2d = f_wm_vol[slice_idx, :, :].T
            f_wm_fibers_2d = np.transpose(f_wm_fibers_vol[slice_idx, :, :, :], (1, 0, 2))
            f_restricted_2d = f_restricted_vol[slice_idx, :, :].T
            f_intra_2d = f_intra_vol[slice_idx, :, :].T
            fiber_dirs_2d = np.transpose(fiber_dirs_vol[slice_idx, :, :, :, :], (1, 0, 2, 3))
            mask_2d = mask[slice_idx, :, :].T
            dwi_2d = np.transpose(dwi[slice_idx, :, :, :], (1, 0, 2))
        
        if axis == 2:
            mse_2d = mse_vol[:, :, slice_idx]
        elif axis == 1:
            mse_2d = mse_vol[:, slice_idx, :].T
        else:
            mse_2d = mse_vol[slice_idx, :, :].T
        
        H, W = f_csf_2d.shape
        
        kappa_2d = np.ones((H, W), dtype=np.float32) * 8.0
        
        params = {
            'f_csf': torch.tensor(f_csf_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(-1),
            'f_gm': torch.tensor(f_gm_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(-1),
            'f_wm': torch.tensor(f_wm_fibers_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(-2),
            'f_restricted': torch.tensor(f_restricted_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(-1),
            'f_intra': torch.tensor(f_intra_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(-1),
            'kappa': torch.tensor(kappa_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(-1),
            'fiber_dirs': torch.tensor(fiber_dirs_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(-3),
            'target_signal': torch.tensor(dwi_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(-2),
            'pred_signal': torch.tensor(dwi_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(-2),
        }
        
        mask_patch = mask_2d.astype(bool)
        
        viz_path = os.path.join(args.outdir, f"microstructure_panel_{axis_name}.png")
        create_simple_visualization(
            params, mask_patch, viz_path, f"{dataset_name} ({axis_name})", 
            data_mask=mask_patch, 
            use_relative_residual=use_output_rel_res,
            rel_res_threshold=output_rel_res_thresh,
            r2_threshold=output_r2_thresh,
            mask_provided=mask_provided,
            apply_viz_mask=mask_provided
        )
        
        try:
            maps = MicrostructureMaps.from_fitted_params(
                params,
                signal_pred=params['pred_signal'],
                signal_obs=params['target_signal'],
                wm_mask=mask_patch,
            )
            
            fig2_path = os.path.join(args.outdir, f"figure2_panel_{axis_name}.png")
            
            if use_output_rel_res:
                create_figure2_panel(maps, slice_idx=0, output_path=fig2_path, 
                                     use_r2_mask=False, 
                                     use_relative_residual_mask=True,
                                     rel_residual_threshold=output_rel_res_thresh,
                                     brain_mask=mask_patch if mask_provided else None)
            elif use_output_r2:
                create_figure2_panel(maps, slice_idx=0, output_path=fig2_path, 
                                     use_r2_mask=True, 
                                     r2_threshold=output_r2_thresh,
                                     use_relative_residual_mask=False,
                                     brain_mask=mask_patch if mask_provided else None)
            else:
                create_figure2_panel(maps, slice_idx=0, output_path=fig2_path, 
                                     use_r2_mask=False, 
                                     use_relative_residual_mask=False,
                                     brain_mask=mask_patch if mask_provided else None)
            print(f"    Saved: {fig2_path}")
        except Exception as e:
            import traceback
            print(f"    Note: Could not create figure2_panel_{axis_name}: {e}")
            traceback.print_exc()


def create_simple_visualization(params, mask, output_path, dataset_name="", data_mask=None, 
                                 r2_threshold=None, use_relative_residual=False, rel_res_threshold=None,
                                 mask_provided=True, apply_viz_mask=None):
    import matplotlib.pyplot as plt
    
    n_fibers = params['fiber_dirs'].shape[-2]
    
    n_cols = max(5, min(n_fibers + 1, 6))
    n_rows = 4
    
    if apply_viz_mask is not None:
        do_mask_viz = apply_viz_mask
    else:
        do_mask_viz = mask_provided
    
    H, W = mask.squeeze().shape
    scale = max(1.0, max(H, W) / 50)
    fig_width = max(20, 4 * scale * W / max(H, W) * n_cols)
    fig_height = max(20, 4 * scale * H / max(H, W) * n_rows)
    fig_width = min(max(fig_width, W * n_cols / 25), 60)
    fig_height = min(max(fig_height, H * n_rows / 25), 48)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), facecolor='black')
    fig.suptitle(f'Microstructure Fitting Results{" - " + dataset_name if dataset_name else ""}', 
                 fontsize=14, fontweight='bold', color='white')
    
    for ax in axes.flat:
        ax.set_facecolor('black')
    
    def masked_imshow(ax, data, mask, cmap='plasma', vmin=0, vmax=1, title=''):
        cmap_obj = plt.cm.get_cmap(cmap).copy()
        cmap_obj.set_bad(color='black')
        masked_data = np.ma.masked_where(~mask, data)
        im = ax.imshow(masked_data.T, cmap=cmap_obj, vmin=vmin, vmax=vmax, 
                       origin='lower', interpolation='nearest')
        if title:
            ax.set_title(title, color='white')
        ax.axis('off')
        return im
    
    def get_2d(t):
        arr = t.detach().cpu().numpy()
        if arr.ndim >= 4:
            return arr[0, :, :, 0]
        return arr[0] if arr.ndim == 3 else arr
    
    def get_2d_fiber(t, idx=0):
        arr = t.detach().cpu().numpy()
        return arr[0, :, :, 0, idx]
    
    mask_2d = mask.squeeze().astype(bool)
    
    if not do_mask_viz:
        viz_mask_2d = np.ones_like(mask_2d, dtype=bool)
    else:
        viz_mask_2d = mask_2d.astype(bool)
    
    if data_mask is not None and do_mask_viz:
        data_mask_2d = data_mask.squeeze() if hasattr(data_mask, 'squeeze') else np.asarray(data_mask).squeeze()
        data_mask_2d = data_mask_2d & viz_mask_2d
    else:
        data_mask_2d = viz_mask_2d
    
    pred = params['pred_signal'].detach().cpu().numpy()[0, :, :, 0]
    target = params['target_signal'].detach().cpu().numpy()[0, :, :, 0]
    
    target_mean = target.mean(axis=-1)
    pred_mean = pred.mean(axis=-1)
    residual_map = np.sqrt(((pred - target) ** 2).mean(axis=-1))
    
    ss_res = ((pred - target) ** 2).sum(axis=-1)
    ss_tot = ((target - target.mean(axis=-1, keepdims=True)) ** 2).sum(axis=-1)
    r2_map = np.where(ss_tot > 1e-6, 1 - ss_res / (ss_tot + 1e-8), np.nan)
    
    from scipy import ndimage
    
    signal_norm = np.linalg.norm(target, axis=-1)
    signal_norm = np.maximum(signal_norm, 1e-6)
    rel_residual = np.linalg.norm(pred - target, axis=-1) / signal_norm
    
    use_output_rel_res = rel_res_threshold is not None and use_relative_residual
    use_output_r2 = r2_threshold is not None and not use_output_rel_res
    
    if use_output_rel_res:
        raw_mask = (rel_residual < rel_res_threshold) & viz_mask_2d
        mask_type = f"rel_res<{rel_res_threshold}"
    elif use_output_r2:
        raw_mask = ~np.isnan(r2_map) & (r2_map >= r2_threshold) & viz_mask_2d
        mask_type = f"R²>{r2_threshold}"
    elif do_mask_viz:
        raw_mask = viz_mask_2d.astype(bool)
        mask_type = "input mask"
    else:
        raw_mask = np.ones_like(mask_2d, dtype=bool)
        mask_type = "all voxels"
    
    if use_output_rel_res or use_output_r2:
        labeled, n_components = ndimage.label(raw_mask)
        if n_components > 0:
            component_sizes = ndimage.sum(raw_mask, labeled, range(1, n_components + 1))
            largest_label = np.argmax(component_sizes) + 1
            mask_largest = (labeled == largest_label)
            
            r2_mask_2d = ndimage.binary_fill_holes(mask_largest)
            r2_mask_2d = r2_mask_2d & viz_mask_2d
        else:
            r2_mask_2d = raw_mask
    else:
        r2_mask_2d = raw_mask
    
    residuals = pred - target
    residuals_brain = residuals[mask_2d]
    sigma_global = 1.4826 * np.median(np.abs(residuals_brain - np.median(residuals_brain, axis=0, keepdims=True)), axis=0)
    sigma_global = np.maximum(sigma_global, 1e-6)
    
    z_scores = residuals / sigma_global[np.newaxis, np.newaxis, :]
    chi2_map = (z_scores ** 2).mean(axis=-1)
    
    brain_target = target_mean[data_mask_2d]
    vmin_sig = 0
    vmax_sig = np.percentile(brain_target, 99) * 0.9
    
    def get_cmap_black_bad(name):
        cmap = plt.cm.get_cmap(name).copy()
        cmap.set_bad(color='black')
        return cmap
    
    cmap_gray = get_cmap_black_bad('gray')
    im = axes[0, 0].imshow(np.ma.masked_where(~data_mask_2d, target_mean).T, cmap=cmap_gray, 
                           origin='lower', interpolation='nearest',
                           vmin=vmin_sig, vmax=vmax_sig)
    axes[0, 0].set_title('Observed (data mask)', color='white')
    axes[0, 0].axis('off')
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    im = axes[0, 1].imshow(np.ma.masked_where(~r2_mask_2d, pred_mean).T, cmap=cmap_gray, 
                           origin='lower', interpolation='nearest',
                           vmin=vmin_sig, vmax=vmax_sig)
    axes[0, 1].set_title(f'Predicted (R²>{r2_threshold})', color='white')
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    cmap_hot = get_cmap_black_bad('hot')
    im = axes[0, 2].imshow(np.ma.masked_where(~r2_mask_2d, residual_map).T, cmap=cmap_hot, 
                           origin='lower', interpolation='nearest',
                           vmin=0, vmax=0.12)
    axes[0, 2].set_title('RMSE (absolute)', color='white')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    cmap_coolwarm = get_cmap_black_bad('coolwarm')
    im = axes[0, 3].imshow(np.ma.masked_where(~r2_mask_2d, chi2_map).T, cmap=cmap_coolwarm, 
                           origin='lower', interpolation='nearest',
                           vmin=0.5, vmax=2.0)
    chi2_mean = np.nanmean(chi2_map[r2_mask_2d])
    axes[0, 3].set_title(f'χ² (mean={chi2_mean:.2f}, ideal=1)', color='white')
    axes[0, 3].axis('off')
    plt.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.04)
    
    cmap_rdylgn = get_cmap_black_bad('RdYlGn')
    im = axes[0, 4].imshow(np.ma.masked_where(~r2_mask_2d, r2_map).T, cmap=cmap_rdylgn, 
                           origin='lower', interpolation='nearest', vmin=0.5, vmax=1.0)
    axes[0, 4].set_title(f'R² (mean={np.nanmean(r2_map[r2_mask_2d]):.3f})', color='white')
    axes[0, 4].axis('off')
    plt.colorbar(im, ax=axes[0, 4], fraction=0.046, pad=0.04)
    
    for col in range(5, n_cols):
        axes[0, col].axis('off')
    
    f_csf = get_2d(params['f_csf'])
    f_gm = get_2d(params['f_gm'])
    f_wm = params['f_wm'].detach().cpu().numpy()[0, :, :, 0].sum(axis=-1)
    f_restricted = get_2d(params['f_restricted'])
    
    masked_imshow(axes[1, 0], f_csf, r2_mask_2d, cmap='Blues', vmin=0, vmax=0.6, title='f_CSF')
    masked_imshow(axes[1, 1], f_gm, r2_mask_2d, cmap='Greens', vmin=0, vmax=0.6, title='f_GM')
    masked_imshow(axes[1, 2], f_wm, r2_mask_2d, cmap='plasma', vmin=0, vmax=0.8, title='f_WM (total)')
    masked_imshow(axes[1, 3], f_restricted, r2_mask_2d, cmap='Purples', vmin=0, vmax=0.3, title='f_Restricted')
    
    if n_cols > 4:
        tissue_stack = np.stack([f_csf, f_gm, f_wm, f_restricted], axis=-1)
        dominant_tissue = np.argmax(tissue_stack, axis=-1)
        
        seg_rgb = np.zeros((*mask_2d.shape, 3))
        seg_rgb[dominant_tissue == 0] = [0.2, 0.4, 1.0]
        seg_rgb[dominant_tissue == 1] = [0.2, 0.8, 0.2]
        seg_rgb[dominant_tissue == 2] = [1.0, 0.3, 0.3]
        seg_rgb[dominant_tissue == 3] = [0.7, 0.3, 0.9]
        seg_rgb[~r2_mask_2d] = 0
        
        axes[1, 4].imshow(seg_rgb.transpose(1, 0, 2), origin='lower', interpolation='nearest')
        axes[1, 4].set_title('Tissue Segmentation')
        axes[1, 4].axis('off')
    
    for col in range(5, n_cols):
        axes[1, col].axis('off')
    
    all_fiber_dirs = params['fiber_dirs'].detach().cpu().numpy()[0, :, :, 0]
    all_f_wm = params['f_wm'].detach().cpu().numpy()[0, :, :, 0]
    
    smooth_dirs, smooth_f_wm = reorder_fibers_for_smoothness(all_fiber_dirs, all_f_wm, r2_mask_2d)
    
    def smooth_rgb_display(rgb, mask, sigma=1.5):
        from scipy.ndimage import gaussian_filter
        smoothed = np.zeros_like(rgb)
        for c in range(3):
            channel = rgb[:, :, c].copy()
            channel[~mask] = 0
            smoothed[:, :, c] = gaussian_filter(channel, sigma=sigma)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j]:
                    norm_orig = np.linalg.norm(rgb[i, j])
                    norm_smooth = np.linalg.norm(smoothed[i, j])
                    if norm_smooth > 1e-6 and norm_orig > 1e-6:
                        smoothed[i, j] = smoothed[i, j] * norm_orig / norm_smooth
        smoothed[~mask] = 0
        return smoothed
    
    fiber_names = ['Fiber 1 (Primary)', 'Fiber 2 (Secondary)', 'Fiber 3 (Tertiary)', 
                   'Fiber 4', 'Fiber 5']
    n_fibers_to_show = min(n_fibers, n_cols - 1)
    for f_idx in range(n_fibers_to_show):
        rgb = np.abs(smooth_dirs[:, :, f_idx, :])
        rgb = rgb / (np.linalg.norm(rgb, axis=-1, keepdims=True) + 1e-8)
        alpha = smooth_f_wm[:, :, f_idx:f_idx+1]
        rgb = rgb * alpha
        rgb = smooth_rgb_display(rgb, r2_mask_2d, sigma=1.5)
        rgb[~r2_mask_2d] = 0
        axes[2, f_idx].imshow(np.clip(rgb.transpose(1, 0, 2), 0, 1), origin='lower', interpolation='nearest')
        axes[2, f_idx].set_title(fiber_names[f_idx] if f_idx < len(fiber_names) else f'Fiber {f_idx+1}')
        axes[2, f_idx].axis('off')
        axes[2, f_idx].set_facecolor('black')
    
    f_intra = get_2d(params['f_intra'])
    ndi_col = n_cols - 1
    cmap_viridis = get_cmap_black_bad('viridis')
    axes[2, ndi_col].imshow(np.ma.masked_where(~r2_mask_2d, f_intra).T, cmap=cmap_viridis, vmin=0, vmax=1, origin='lower', interpolation='nearest')
    axes[2, ndi_col].set_title('NDI (f_intra)')
    axes[2, ndi_col].axis('off')
    
    for col in range(n_fibers_to_show, n_cols - 1):
        axes[2, col].axis('off')
    
    combined_rgb = np.zeros((mask_2d.shape[0], mask_2d.shape[1], 3))
    for f_idx in range(n_fibers):
        fiber_rgb = np.abs(smooth_dirs[:, :, f_idx, :])
        fiber_rgb = fiber_rgb / (np.linalg.norm(fiber_rgb, axis=-1, keepdims=True) + 1e-8)
        alpha = smooth_f_wm[:, :, f_idx:f_idx+1]
        combined_rgb = combined_rgb * (1 - alpha) + fiber_rgb * alpha
    combined_rgb[~r2_mask_2d] = 0
    axes[3, 0].imshow(np.clip(combined_rgb.transpose(1, 0, 2), 0, 1), origin='lower', interpolation='nearest')
    axes[3, 0].set_title(f'All {n_fibers} Fibers (opacity=fraction)')
    axes[3, 0].axis('off')
    axes[3, 0].set_facecolor('black')
    
    target_mean_safe = np.maximum(target_mean, 1e-6)
    rel_residual = residual_map / target_mean_safe
    cmap_hot2 = get_cmap_black_bad('hot')
    axes[3, 1].imshow(np.ma.masked_where(~r2_mask_2d, rel_residual).T, cmap=cmap_hot2, vmin=0, vmax=0.25, origin='lower', interpolation='nearest')
    axes[3, 1].set_title('Relative Error (RMSE/S)')
    axes[3, 1].axis('off')
    
    fiber_count = (all_f_wm > 0.1).sum(axis=-1).astype(float)
    cmap_plasma2 = get_cmap_black_bad('plasma')
    axes[3, 2].imshow(np.ma.masked_where(~r2_mask_2d, fiber_count).T, cmap=cmap_plasma2, vmin=0, vmax=n_fibers, origin='lower', interpolation='nearest')
    axes[3, 2].set_title(f'Fiber Count (f>0.1, max={n_fibers})', color='white')
    axes[3, 2].axis('off')
    
    if 'losses' in params and params['losses'] is not None:
        axes[3, 3].semilogy(params['losses'], color='cyan')
        axes[3, 3].set_xlabel('Step', color='white')
        axes[3, 3].set_ylabel('MSE', color='white')
        axes[3, 3].set_title(f'Final MSE: {params.get("mse", 0):.5f}', color='white')
        axes[3, 3].grid(True, alpha=0.3, color='gray')
        axes[3, 3].tick_params(colors='white')
        for spine in axes[3, 3].spines.values():
            spine.set_color('white')
    else:
        axes[3, 3].text(0.5, 0.5, 'No loss curve\n(3D slice view)', 
                        ha='center', va='center', color='white', fontsize=12,
                        transform=axes[3, 3].transAxes)
        axes[3, 3].set_title('Loss Curve', color='white')
        axes[3, 3].axis('off')
    
    for col in range(4, n_cols):
        axes[3, col].axis('off')
    
    for ax in axes.flat:
        if ax.get_title():
            ax.set_title(ax.get_title(), color='white')
    
    plt.tight_layout()
    save_dpi = max(150, min(300, max(H, W) * 2))
    plt.savefig(output_path, dpi=save_dpi, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Saved visualization: {output_path}")


def fit_whole_brain(args, img, dwi, bvals, bvecs, mask, dwi_for_sigma=None):
    if dwi_for_sigma is None:
        dwi_for_sigma = dwi
    import time
    
    X, Y, Z, N = dwi.shape
    n_fibers = args.n_fibers
    
    dev = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    use_topology = args.use_topology and not args.no_topology
    
    scanner_config = get_scanner_config_from_args(args)
    topology_config = get_topology_config_from_args(args)
    
    cube_size = getattr(args, 'cube_size', 20)
    overlap = getattr(args, 'overlap', 3)
    stride = cube_size - overlap
    
    print(f"\n{'='*60}")
    print(f"WHOLE BRAIN FITTING (3D CUBE-BASED)")
    print(f"{'='*60}")
    print(f"  Volume: {X} x {Y} x {Z}")
    print(f"  Device: {dev}")
    print(f"  Topology: {use_topology}")
    print(f"  6-conn Topology: {topology_config.get('use_6conn_topology', False)}")
    print(f"  26-conn Topology: {topology_config['use_26conn_topology']}")
    print(f"  Cube size: {cube_size}³ voxels")
    print(f"  Overlap: {overlap} voxels")
    print(f"  Stride: {stride} voxels")
    print(f"  N-steps: {args.n_steps}")
    print(f"  N-fibers: {n_fibers}")
    
    print_scanner_config(scanner_config)
    
    n_cubes_x = max(1, (X - overlap + stride - 1) // stride)
    n_cubes_y = max(1, (Y - overlap + stride - 1) // stride)
    n_cubes_z = max(1, (Z - overlap + stride - 1) // stride)
    total_cubes = n_cubes_x * n_cubes_y * n_cubes_z
    
    print(f"  Grid: {n_cubes_x} x {n_cubes_y} x {n_cubes_z} = {total_cubes} cubes")
    
    est_time = total_cubes * 15 / 60
    print(f"  Estimated time: ~{est_time:.1f} min")
    print()
    
    scanner = create_scanner(bvals, bvecs, n_fibers=n_fibers, device=dev, **scanner_config)
    
    f_csf_vol = np.zeros((X, Y, Z), dtype=np.float32)
    f_gm_vol = np.zeros((X, Y, Z), dtype=np.float32)
    f_wm_vol = np.zeros((X, Y, Z), dtype=np.float32)
    f_restricted_vol = np.zeros((X, Y, Z), dtype=np.float32)
    f_intra_vol = np.zeros((X, Y, Z), dtype=np.float32)
    mse_vol = np.zeros((X, Y, Z), dtype=np.float32)
    fiber_dirs_vol = np.zeros((X, Y, Z, n_fibers, 3), dtype=np.float32)
    f_wm_fibers_vol = np.zeros((X, Y, Z, n_fibers), dtype=np.float32)
    
    weight_vol = np.zeros((X, Y, Z), dtype=np.float32)
    
    start_time = time.time()
    cube_idx = 0
    
    cubes = []
    for ix in range(n_cubes_x):
        for iy in range(n_cubes_y):
            for iz in range(n_cubes_z):
                x0 = min(ix * stride, X - cube_size) if X > cube_size else 0
                y0 = min(iy * stride, Y - cube_size) if Y > cube_size else 0
                z0 = min(iz * stride, Z - cube_size) if Z > cube_size else 0
                x1 = min(x0 + cube_size, X)
                y1 = min(y0 + cube_size, Y)
                z1 = min(z0 + cube_size, Z)
                
                cube_mask = mask[x0:x1, y0:y1, z0:z1]
                if cube_mask.sum() < 50:
                    continue
                    
                cubes.append((x0, x1, y0, y1, z0, z1))
    
    print(f"  Non-empty cubes: {len(cubes)}")
    
    global_sigma_result = None
    global_sigma_map = None
    
    use_rician = getattr(args, 'use_rician', False)
    if use_rician and getattr(args, 'noise_sigma', None) is None:
        auto_method = getattr(args, 'auto_noise_sigma', 'background')
        if auto_method != 'none':
            print(f"\n[σ Estimation] Computing global σ from FULL volume before patch fitting...")
            
            signal_for_sigma = dwi_for_sigma
            print(f"[σ Estimation] Using {'original unmasked' if dwi_for_sigma is not dwi else 'masked'} DWI for σ estimation")
            
            signal_full_t = torch.tensor(signal_for_sigma, dtype=torch.float32, device=dev)
            mask_full_t = torch.tensor(mask > 0, dtype=torch.bool, device=dev)
            bvals_t = torch.tensor(bvals, dtype=torch.float32, device=dev)
            
            try:
                global_sigma_result = estimate_noise_sigma(
                    signal_full_t,
                    mask=mask_full_t,
                    bvals=bvals_t,
                    method=auto_method,
                    compute_map=getattr(args, 'use_sigma_map', False),
                    bg_margin=getattr(args, 'sigma_bg_margin', 3),
                    min_bg_voxels=2000,
                    print_diagnostics=True,
                )
                
                if getattr(args, 'use_sigma_map', False) and global_sigma_result.sigma_map is not None:
                    global_sigma_map = global_sigma_result.sigma_map
                    print(f"[σ Estimation] Global σ-map computed, shape {global_sigma_map.shape}")
                else:
                    print(f"[σ Estimation] Using scalar σ_norm = {global_sigma_result.sigma_normalized:.4f}")
                
                if getattr(args, 'save_sigma_map', False) and global_sigma_result.sigma_map is not None:
                    sigma_nii = nib.Nifti1Image(global_sigma_result.sigma_map.cpu().numpy(), img.affine)
                    sigma_path = os.path.join(args.outdir, 'sigma_map.nii.gz')
                    nib.save(sigma_nii, sigma_path)
                    print(f"[σ Estimation] Saved σ-map to {sigma_path}")
                    
            except Exception as e:
                print(f"[σ Estimation] Warning: Global estimation failed: {e}")
                print(f"[σ Estimation] Will use per-patch estimation (less reliable)")
    
    shared_encoder = None
    
    for cube_idx, (x0, x1, y0, y1, z0, z1) in enumerate(cubes):
        cube_start = time.time()
        
        signal_cube = dwi[x0:x1, y0:y1, z0:z1, :]
        mask_cube = mask[x0:x1, y0:y1, z0:z1] > 0
        
        cx, cy, cz = x1 - x0, y1 - y0, z1 - z0
        
        cube_noise_sigma = getattr(args, 'noise_sigma', None)
        cube_sigma_map = None
        cube_auto_noise_sigma = getattr(args, 'auto_noise_sigma', 'background')
        
        if global_sigma_result is not None:
            if global_sigma_map is not None:
                cube_sigma_map = global_sigma_map[x0:x1, y0:y1, z0:z1].clone()
            else:
                cube_noise_sigma = global_sigma_result.sigma_normalized
            cube_auto_noise_sigma = 'none'
        
        params = fit_patch_3d(
            scanner,
            signal_cube,
            mask_cube,
            n_steps=args.n_steps,
            use_topology=use_topology,
            use_spatial_prior=False,
            use_6conn_topology=topology_config.get('use_6conn_topology', False),
            use_26conn_topology=topology_config['use_26conn_topology'],
            lambda_orphan=topology_config['lambda_orphan'],
            lambda_continuity=topology_config['lambda_continuity'],
            lambda_ordering=topology_config['lambda_ordering'],
            lambda_repulsion=topology_config['lambda_repulsion'],
            lambda_curvature=topology_config.get('lambda_curvature', 0.0),
            lambda_perm_invariant=topology_config.get('lambda_perm_invariant', 0.0),
            device=dev,
            use_encoder=getattr(args, 'use_encoder', False),
            encoder_epochs=getattr(args, 'encoder_epochs', 30),
            encoder=shared_encoder,
            use_rician=use_rician,
            rician_mode=getattr(args, 'rician_mode', 'hybrid'),
            hybrid_mse_weight=getattr(args, 'hybrid_mse_weight', 0.7),
            noise_sigma=cube_noise_sigma,
            sigma_map_precomputed=cube_sigma_map,
            auto_noise_sigma=cube_auto_noise_sigma,
            use_sigma_map=getattr(args, 'use_sigma_map', False),
            sigma_bg_margin=getattr(args, 'sigma_bg_margin', 3),
            sigma_estimation_method=getattr(args, 'sigma_estimation_method', 'median'),
            learn_noise_sigma=getattr(args, 'learn_noise_sigma', False),
            staged_directions=getattr(args, 'staged_directions', False),
            stage1_freeze_steps=getattr(args, 'stage1_freeze_steps', 100),
            direction_lr_factor=getattr(args, 'direction_lr_factor', 0.1),
            signal_raw=signal_cube,
            bvals_np=bvals,
            b_switch=getattr(args, 'b_switch', 2000.0),
            b_slope=getattr(args, 'b_slope', 0.002),
            sparsity_weight=getattr(args, 'sparsity_weight', 0.02),
            entropy_weight=getattr(args, 'entropy_weight', 0.0),
            huber_delta=getattr(args, 'huber_delta', 1.0),
            lambda_cosine=getattr(args, 'lambda_cosine', 0.05),
            early_patience=getattr(args, 'early_patience', 30),
            early_warmup=getattr(args, 'early_warmup', 50),
            early_min_steps=getattr(args, 'early_min_steps', 100),
            early_rel_tol=getattr(args, 'early_rel_tol', 1e-4),
            no_early_stop=getattr(args, 'no_early_stop', False),
            mse_warmup_steps=getattr(args, 'mse_warmup_steps', 0),
        )
        
        if params.get('encoder') is not None:
            shared_encoder = params['encoder']
        
        at_start = (x0 == 0, y0 == 0, z0 == 0)
        at_end = (x1 >= X, y1 >= Y, z1 >= Z)
        
        use_blend = not getattr(args, 'no_blend', False)
        weight = _create_blend_weights(cx, cy, cz, overlap, at_start, at_end, use_blend=use_blend)
        
        f_csf_cube = params['f_csf'].detach().cpu().numpy()[0]
        f_gm_cube = params['f_gm'].detach().cpu().numpy()[0]
        f_wm_fibers_cube = params['f_wm'].detach().cpu().numpy()[0]
        f_wm_cube = f_wm_fibers_cube.sum(axis=-1)
        f_restricted_cube = params['f_restricted'].detach().cpu().numpy()[0]
        f_intra_cube = params['f_intra'].detach().cpu().numpy()[0]
        fiber_dirs_cube = params['fiber_dirs'].detach().cpu().numpy()[0]
        
        pred = params['pred_signal']
        target = params['target_signal']
        with torch.no_grad():
            mse_cube = ((pred - target) ** 2).mean(dim=-1)[0].cpu().numpy()
        
        f_csf_vol[x0:x1, y0:y1, z0:z1] += f_csf_cube * weight
        f_gm_vol[x0:x1, y0:y1, z0:z1] += f_gm_cube * weight
        f_wm_vol[x0:x1, y0:y1, z0:z1] += f_wm_cube * weight
        f_restricted_vol[x0:x1, y0:y1, z0:z1] += f_restricted_cube * weight
        f_intra_vol[x0:x1, y0:y1, z0:z1] += f_intra_cube * weight
        mse_vol[x0:x1, y0:y1, z0:z1] += mse_cube * weight
        f_wm_fibers_vol[x0:x1, y0:y1, z0:z1, :] += f_wm_fibers_cube * weight[..., np.newaxis]
        
        existing_dirs = fiber_dirs_vol[x0:x1, y0:y1, z0:z1, :, :]
        existing_weight = weight_vol[x0:x1, y0:y1, z0:z1]
        has_existing = existing_weight > 0
        if np.any(has_existing):
            dot_product = np.sum(existing_dirs * fiber_dirs_cube, axis=-1)
            sign_flip = np.where(dot_product < 0, -1.0, 1.0)
            fiber_dirs_cube_aligned = fiber_dirs_cube * sign_flip[..., np.newaxis]
            fiber_dirs_vol[x0:x1, y0:y1, z0:z1, :, :] += fiber_dirs_cube_aligned * weight[..., np.newaxis, np.newaxis]
        else:
            fiber_dirs_vol[x0:x1, y0:y1, z0:z1, :, :] += fiber_dirs_cube * weight[..., np.newaxis, np.newaxis]
        
        weight_vol[x0:x1, y0:y1, z0:z1] += weight
        
        cube_time = time.time() - cube_start
        elapsed = time.time() - start_time
        cubes_done = cube_idx + 1
        remaining = (len(cubes) - cubes_done) * (elapsed / cubes_done)
        
        print(f"  Cube {cube_idx+1:3d}/{len(cubes)} [{x0}:{x1},{y0}:{y1},{z0}:{z1}] | "
              f"MSE={params['mse']:.5f} | {cube_time:.1f}s | "
              f"ETA: {remaining/60:.1f} min")
    
    valid = weight_vol > 0
    for vol in [f_csf_vol, f_gm_vol, f_wm_vol, f_restricted_vol, f_intra_vol, mse_vol]:
        vol[valid] /= weight_vol[valid]
    f_wm_fibers_vol[valid] /= weight_vol[valid, np.newaxis]
    fiber_dirs_vol[valid] /= weight_vol[valid, np.newaxis, np.newaxis]
    
    fiber_norms = np.linalg.norm(fiber_dirs_vol, axis=-1, keepdims=True)
    fiber_norms[fiber_norms < 1e-6] = 1.0
    fiber_dirs_vol /= fiber_norms
    
    if hasattr(args, 'smooth_fibers') and args.smooth_fibers > 0:
        fiber_dirs_vol = smooth_fiber_directions(
            fiber_dirs_vol, 
            mask > 0, 
            f_wm=f_wm_vol,
            sigma=args.smooth_fibers, 
            n_fibers=n_fibers
        )
    
    if getattr(args, 'refine_boundaries', False) and len(cubes) > 1:
        fiber_dirs_vol, f_wm_fibers_vol = refine_boundary_regions(
            args=args,
            scanner=scanner,
            dwi=dwi,
            mask=mask,
            cubes=cubes,
            overlap=overlap,
            fiber_dirs_vol=fiber_dirs_vol,
            f_wm_fibers_vol=f_wm_fibers_vol,
            f_csf_vol=f_csf_vol,
            f_gm_vol=f_gm_vol,
            f_wm_vol=f_wm_vol,
            f_restricted_vol=f_restricted_vol,
            f_intra_vol=f_intra_vol,
            n_fibers=n_fibers,
            device=dev,
            bvals=bvals,
            topology_config=topology_config,
            use_topology=use_topology,
            use_rician=use_rician,
            global_sigma_result=global_sigma_result,
            global_sigma_map=global_sigma_map,
        )
    
    if getattr(args, 'refine_slicewise', False):
        fiber_dirs_vol, f_wm_fibers_vol = refine_slicewise(
            args=args,
            scanner=scanner,
            dwi=dwi,
            mask=mask,
            fiber_dirs_vol=fiber_dirs_vol,
            f_wm_fibers_vol=f_wm_fibers_vol,
            f_csf_vol=f_csf_vol,
            f_gm_vol=f_gm_vol,
            f_wm_vol=f_wm_vol,
            f_restricted_vol=f_restricted_vol,
            f_intra_vol=f_intra_vol,
            n_fibers=n_fibers,
            device=dev,
            n_sweeps=getattr(args, 'refine_sweeps', 50),
        )
    
    brain_mask = mask > 0
    for vol in [f_csf_vol, f_gm_vol, f_wm_vol, f_restricted_vol, f_intra_vol]:
        vol[~brain_mask] = 0
    f_wm_fibers_vol[~brain_mask] = 0
    fiber_dirs_vol[~brain_mask] = 0
    mse_vol[~brain_mask] = 0
    
    print("\n[Computing reliable microstructure mask]")
    
    fiber_threshold = 0.05
    fiber_count = np.zeros((X, Y, Z), dtype=np.uint8)
    if n_fibers > 1:
        for k in range(n_fibers):
            fiber_count += (f_wm_fibers_vol[..., k] > fiber_threshold).astype(np.uint8)
        
        f_wm_total = f_wm_fibers_vol.sum(axis=-1)
        f_wm_total[f_wm_total < 1e-6] = 1.0
        f_wm_normalized = f_wm_fibers_vol / f_wm_total[..., np.newaxis]
        max_fiber_frac = f_wm_normalized.max(axis=-1)
    else:
        fiber_count = np.ones((X, Y, Z), dtype=np.uint8)
        max_fiber_frac = np.ones((X, Y, Z))
    
    wm_dominant = f_wm_vol > 0.5
    single_fiber = fiber_count <= 1
    good_dominance = max_fiber_frac > 0.7
    good_fit = mse_vol < 0.01
    
    reliable_mask = brain_mask & wm_dominant & (single_fiber | good_dominance) & good_fit
    
    n_reliable = reliable_mask.sum()
    n_brain = brain_mask.sum()
    print(f"  Reliable voxels: {n_reliable:,} / {n_brain:,} ({100*n_reliable/n_brain:.1f}%)")
    print(f"    WM dominant: {wm_dominant[brain_mask].sum():,}")
    print(f"    Single fiber: {single_fiber[brain_mask].sum():,}")
    print(f"    Good fit (MSE<0.01): {good_fit[brain_mask].sum():,}")
    
    micro_fa = f_wm_vol * f_intra_vol
    ndi = f_wm_vol * f_intra_vol
    
    f_intra_reliable = np.where(reliable_mask, f_intra_vol, np.nan)
    micro_fa_reliable = np.where(reliable_mask, micro_fa, np.nan)
    
    print(f"\n  In reliable regions:")
    print(f"    f_intra: {np.nanmean(f_intra_reliable):.3f} ± {np.nanstd(f_intra_reliable):.3f}")
    print(f"    Micro FA: {np.nanmean(micro_fa_reliable):.3f} ± {np.nanstd(micro_fa_reliable):.3f}")
    
    total_time = time.time() - start_time
    print(f"\n  Total time: {total_time/60:.1f} min")
    
    print(f"\n[Saving NIfTI outputs to {args.outdir}/]")
    affine = img.affine
    
    nib.save(nib.Nifti1Image(f_csf_vol, affine), os.path.join(args.outdir, "f_csf.nii.gz"))
    nib.save(nib.Nifti1Image(f_gm_vol, affine), os.path.join(args.outdir, "f_gm.nii.gz"))
    nib.save(nib.Nifti1Image(f_wm_vol, affine), os.path.join(args.outdir, "f_wm.nii.gz"))
    nib.save(nib.Nifti1Image(f_restricted_vol, affine), os.path.join(args.outdir, "f_restricted.nii.gz"))
    nib.save(nib.Nifti1Image(f_intra_vol, affine), os.path.join(args.outdir, "f_intra.nii.gz"))
    nib.save(nib.Nifti1Image(mse_vol, affine), os.path.join(args.outdir, "mse.nii.gz"))
    nib.save(nib.Nifti1Image(f_wm_fibers_vol, affine), os.path.join(args.outdir, "f_wm_fibers.nii.gz"))
    fiber_dirs_flat = fiber_dirs_vol.reshape(X, Y, Z, n_fibers * 3)
    nib.save(nib.Nifti1Image(fiber_dirs_flat, affine), os.path.join(args.outdir, "fiber_dirs.nii.gz"))
    
    nib.save(nib.Nifti1Image(fiber_count.astype(np.float32), affine), os.path.join(args.outdir, "fiber_count.nii.gz"))
    nib.save(nib.Nifti1Image(reliable_mask.astype(np.float32), affine), os.path.join(args.outdir, "reliable_mask.nii.gz"))
    nib.save(nib.Nifti1Image(micro_fa, affine), os.path.join(args.outdir, "micro_fa.nii.gz"))
    
    print(f"  Saved: f_csf.nii.gz, f_gm.nii.gz, f_wm.nii.gz, f_restricted.nii.gz")
    print(f"  Saved: f_intra.nii.gz, mse.nii.gz, f_wm_fibers.nii.gz, fiber_dirs.nii.gz")
    print(f"  Saved: fiber_count.nii.gz, reliable_mask.nii.gz, micro_fa.nii.gz")
    
    summary = {
        'dataset': args.name or os.path.basename(args.dwi),
        'mode': 'whole_brain_cube',
        'cube_size': cube_size,
        'overlap': overlap,
        'volume_shape': [X, Y, Z],
        'n_fibers': n_fibers,
        'n_steps': args.n_steps,
        'use_topology': use_topology,
        'use_26conn_topology': getattr(args, 'use_26conn_topology', False),
        'cubes_fitted': len(cubes),
        'total_time_sec': total_time,
        'mean_mse': float(mse_vol[mask > 0].mean()),
        'reliable_voxels': int(n_reliable),
        'reliable_fraction': float(n_reliable / n_brain),
        'n_measurements': int(N),
        'b_values': [float(b) for b in np.unique(bvals.round(-2))],
    }
    
    if use_rician:
        summary['rician'] = {
            'enabled': True,
            'mode': getattr(args, 'rician_mode', 'hybrid'),
            'hybrid_mse_weight': getattr(args, 'hybrid_mse_weight', 0.7),
            'use_sigma_map': getattr(args, 'use_sigma_map', False),
        }
        if global_sigma_result is not None:
            summary['rician']['sigma_diagnostics'] = global_sigma_result.to_dict()
        elif getattr(args, 'noise_sigma', None) is not None:
            summary['rician']['sigma_diagnostics'] = {
                'sigma_method': 'manual',
                'sigma_normalized': getattr(args, 'noise_sigma'),
            }
        if getattr(args, 'rician_mode', 'hybrid') == 'b-dependent':
            summary['rician']['b_switch'] = getattr(args, 'b_switch', 2000.0)
            summary['rician']['b_slope'] = getattr(args, 'b_slope', 0.002)
    
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: summary.json")
    
    if getattr(args, 'export_artifacts', False):
        export_scanner_artifacts(
            scanner=scanner,
            outdir=args.outdir,
            affine=img.affine,
            spatial_shape=(X, Y, Z),
            mask=mask,
            bvals=bvals,
            dataset_name=args.name or os.path.basename(args.dwi),
            dwi_signal=dwi,
        )
    
    print(f"\n[Creating visualization panels for 3 axes...]")
    create_3axis_panels_from_volumes(
        args=args,
        f_csf_vol=f_csf_vol,
        f_gm_vol=f_gm_vol,
        f_wm_vol=f_wm_vol,
        f_wm_fibers_vol=f_wm_fibers_vol,
        f_restricted_vol=f_restricted_vol,
        f_intra_vol=f_intra_vol,
        fiber_dirs_vol=fiber_dirs_vol,
        mse_vol=mse_vol,
        mask=mask,
        dwi=dwi,
        bvals=bvals,
        n_fibers=n_fibers,
        dataset_name=args.name or os.path.basename(args.dwi),
    )
    
    print(f"\n✓ Whole brain fitting complete!")
    print(f"  Mean MSE: {summary['mean_mse']:.5f}")
    print(f"  Output: {args.outdir}/")


def fit_whole_brain_slabs(args, img, dwi, bvals, bvecs, mask, dwi_for_sigma=None):
    if dwi_for_sigma is None:
        dwi_for_sigma = dwi
    import time
    
    X, Y, Z, N = dwi.shape
    n_fibers = args.n_fibers
    
    dev = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    use_topology = args.use_topology and not args.no_topology
    
    scanner_config = get_scanner_config_from_args(args)
    topology_config = get_topology_config_from_args(args)
    
    slab_size = getattr(args, 'slab_size', 20)
    overlap = getattr(args, 'overlap', 2)
    stride = slab_size - overlap
    
    print(f"\n{'='*60}")
    print(f"WHOLE BRAIN FITTING (SLAB-BASED)")
    print(f"{'='*60}")
    print(f"  Volume: {X} x {Y} x {Z}")
    print(f"  Device: {dev}")
    print(f"  Topology: {use_topology}")
    print(f"  Slab size: {X} x {Y} x {slab_size} (full X×Y, {slab_size} slices)")
    print(f"  Overlap: {overlap} slices")
    print(f"  Stride: {stride} slices")
    print(f"  N-steps: {args.n_steps}")
    print(f"  N-fibers: {n_fibers}")
    
    print_scanner_config(scanner_config)
    
    n_slabs = max(1, (Z - overlap + stride - 1) // stride)
    
    print(f"  Number of slabs: {n_slabs}")
    
    voxels_per_slab = X * Y * slab_size
    voxels_ref = 50**3
    time_per_slab = 15 * (voxels_per_slab / voxels_ref) * (args.n_steps / 200)
    est_time = n_slabs * time_per_slab / 60
    print(f"  Estimated time: ~{est_time:.1f} min")
    print()
    
    scanner = create_scanner(bvals, bvecs, n_fibers=n_fibers, device=dev, **scanner_config)
    
    f_csf_vol = np.zeros((X, Y, Z), dtype=np.float32)
    f_gm_vol = np.zeros((X, Y, Z), dtype=np.float32)
    f_wm_vol = np.zeros((X, Y, Z), dtype=np.float32)
    f_restricted_vol = np.zeros((X, Y, Z), dtype=np.float32)
    f_intra_vol = np.zeros((X, Y, Z), dtype=np.float32)
    mse_vol = np.zeros((X, Y, Z), dtype=np.float32)
    fiber_dirs_vol = np.zeros((X, Y, Z, n_fibers, 3), dtype=np.float32)
    f_wm_fibers_vol = np.zeros((X, Y, Z, n_fibers), dtype=np.float32)
    weight_vol = np.zeros((X, Y, Z), dtype=np.float32)
    
    start_time = time.time()
    
    slabs = []
    for iz in range(n_slabs):
        z0 = min(iz * stride, Z - slab_size) if Z > slab_size else 0
        z1 = min(z0 + slab_size, Z)
        slab_mask = mask[:, :, z0:z1]
        if slab_mask.sum() < 100:
            continue
        slabs.append((z0, z1))
    
    print(f"  Non-empty slabs: {len(slabs)}")
    
    global_sigma_result = None
    global_sigma_map = None
    use_rician = getattr(args, 'use_rician', False)
    if use_rician and getattr(args, 'noise_sigma', None) is None:
        auto_method = getattr(args, 'auto_noise_sigma', 'background')
        if auto_method != 'none':
            print(f"\n[σ Estimation] Computing global σ from FULL volume...")
            signal_full_t = torch.tensor(dwi_for_sigma, dtype=torch.float32, device=dev)
            mask_full_t = torch.tensor(mask > 0, dtype=torch.bool, device=dev)
            bvals_t = torch.tensor(bvals, dtype=torch.float32, device=dev)
            try:
                global_sigma_result = estimate_noise_sigma(
                    signal_full_t, mask=mask_full_t, bvals=bvals_t,
                    method=auto_method,
                    compute_map=getattr(args, 'use_sigma_map', False),
                    bg_margin=getattr(args, 'sigma_bg_margin', 3),
                    min_bg_voxels=2000,
                    print_diagnostics=True,
                )
                if getattr(args, 'use_sigma_map', False) and global_sigma_result.sigma_map is not None:
                    global_sigma_map = global_sigma_result.sigma_map
            except Exception as e:
                print(f"[σ Estimation] Warning: Failed: {e}")
    
    for slab_idx, (z0, z1) in enumerate(slabs):
        slab_start = time.time()
        
        signal_slab = dwi[:, :, z0:z1, :]
        mask_slab = mask[:, :, z0:z1] > 0
        
        sz = z1 - z0
        
        slab_noise_sigma = getattr(args, 'noise_sigma', None)
        slab_sigma_map = None
        slab_auto_noise_sigma = getattr(args, 'auto_noise_sigma', 'background')
        
        if global_sigma_result is not None:
            if global_sigma_map is not None:
                slab_sigma_map = global_sigma_map[:, :, z0:z1].clone()
            else:
                slab_noise_sigma = global_sigma_result.sigma_normalized
            slab_auto_noise_sigma = 'none'
        
        params = fit_patch_3d(
            scanner, signal_slab, mask_slab,
            n_steps=args.n_steps,
            use_topology=use_topology,
            use_spatial_prior=False,
            use_6conn_topology=topology_config.get('use_6conn_topology', False),
            use_26conn_topology=topology_config['use_26conn_topology'],
            lambda_orphan=topology_config['lambda_orphan'],
            lambda_continuity=topology_config['lambda_continuity'],
            lambda_ordering=topology_config['lambda_ordering'],
            lambda_repulsion=topology_config['lambda_repulsion'],
            lambda_curvature=topology_config.get('lambda_curvature', 0.0),
            lambda_perm_invariant=topology_config.get('lambda_perm_invariant', 0.0),
            device=dev,
            use_encoder=getattr(args, 'use_encoder', False),
            encoder_epochs=getattr(args, 'encoder_epochs', 30),
            use_rician=use_rician,
            rician_mode=getattr(args, 'rician_mode', 'hybrid'),
            hybrid_mse_weight=getattr(args, 'hybrid_mse_weight', 0.7),
            noise_sigma=slab_noise_sigma,
            sigma_map_precomputed=slab_sigma_map,
            auto_noise_sigma=slab_auto_noise_sigma,
            use_sigma_map=getattr(args, 'use_sigma_map', False),
            sigma_bg_margin=getattr(args, 'sigma_bg_margin', 3),
            sigma_estimation_method=getattr(args, 'sigma_estimation_method', 'median'),
            learn_noise_sigma=getattr(args, 'learn_noise_sigma', False),
            staged_directions=getattr(args, 'staged_directions', False),
            stage1_freeze_steps=getattr(args, 'stage1_freeze_steps', 100),
            direction_lr_factor=getattr(args, 'direction_lr_factor', 0.1),
            signal_raw=signal_slab,
            bvals_np=bvals,
            b_switch=getattr(args, 'b_switch', 2000.0),
            b_slope=getattr(args, 'b_slope', 0.002),
            sparsity_weight=getattr(args, 'sparsity_weight', 0.02),
            entropy_weight=getattr(args, 'entropy_weight', 0.0),
            huber_delta=getattr(args, 'huber_delta', 1.0),
            lambda_cosine=getattr(args, 'lambda_cosine', 0.05),
            early_patience=getattr(args, 'early_patience', 30),
            early_warmup=getattr(args, 'early_warmup', 50),
            early_min_steps=getattr(args, 'early_min_steps', 100),
            early_rel_tol=getattr(args, 'early_rel_tol', 1e-4),
            no_early_stop=getattr(args, 'no_early_stop', False),
            mse_warmup_steps=getattr(args, 'mse_warmup_steps', 0),
        )
        
        at_start_z = (z0 == 0)
        at_end_z = (z1 >= Z)
        use_blend = not getattr(args, 'no_blend', False)
        if use_blend:
            weight = _create_blend_weights_1d(sz, overlap, at_start_z, at_end_z)
        else:
            weight = np.ones(sz, dtype=np.float32)
        weight = weight[np.newaxis, np.newaxis, :]
        
        f_csf_slab = params['f_csf'].detach().cpu().numpy()[0]
        f_gm_slab = params['f_gm'].detach().cpu().numpy()[0]
        f_wm_fibers_slab = params['f_wm'].detach().cpu().numpy()[0]
        f_wm_slab = f_wm_fibers_slab.sum(axis=-1)
        f_restricted_slab = params['f_restricted'].detach().cpu().numpy()[0]
        f_intra_slab = params['f_intra'].detach().cpu().numpy()[0]
        fiber_dirs_slab = params['fiber_dirs'].detach().cpu().numpy()[0]
        
        pred = params['pred_signal']
        target = params['target_signal']
        with torch.no_grad():
            mse_slab = ((pred - target) ** 2).mean(dim=-1)[0].cpu().numpy()
        
        f_csf_vol[:, :, z0:z1] += f_csf_slab * weight
        f_gm_vol[:, :, z0:z1] += f_gm_slab * weight
        f_wm_vol[:, :, z0:z1] += f_wm_slab * weight
        f_restricted_vol[:, :, z0:z1] += f_restricted_slab * weight
        f_intra_vol[:, :, z0:z1] += f_intra_slab * weight
        mse_vol[:, :, z0:z1] += mse_slab * weight
        f_wm_fibers_vol[:, :, z0:z1, :] += f_wm_fibers_slab * weight[..., np.newaxis]
        
        existing_dirs = fiber_dirs_vol[:, :, z0:z1, :, :]
        existing_weight = weight_vol[:, :, z0:z1]
        has_existing = existing_weight > 0
        if np.any(has_existing):
            dot_product = np.sum(existing_dirs * fiber_dirs_slab, axis=-1)
            sign_flip = np.where(dot_product < 0, -1.0, 1.0)
            fiber_dirs_slab_aligned = fiber_dirs_slab * sign_flip[..., np.newaxis]
            fiber_dirs_vol[:, :, z0:z1, :, :] += fiber_dirs_slab_aligned * weight[..., np.newaxis, np.newaxis]
        else:
            fiber_dirs_vol[:, :, z0:z1, :, :] += fiber_dirs_slab * weight[..., np.newaxis, np.newaxis]
        
        weight_vol[:, :, z0:z1] += weight[0, 0, :]
        
        slab_time = time.time() - slab_start
        elapsed = time.time() - start_time
        slabs_done = slab_idx + 1
        remaining = (len(slabs) - slabs_done) * (elapsed / slabs_done)
        
        print(f"  Slab {slab_idx+1:3d}/{len(slabs)} [z={z0}:{z1}] | "
              f"MSE={params['mse']:.5f} | {slab_time:.1f}s | "
              f"ETA: {remaining/60:.1f} min")
    
    valid = weight_vol > 0
    for vol in [f_csf_vol, f_gm_vol, f_wm_vol, f_restricted_vol, f_intra_vol, mse_vol]:
        vol[valid] /= weight_vol[valid]
    f_wm_fibers_vol[valid] /= weight_vol[valid, np.newaxis]
    fiber_dirs_vol[valid] /= weight_vol[valid, np.newaxis, np.newaxis]
    
    fiber_norms = np.linalg.norm(fiber_dirs_vol, axis=-1, keepdims=True)
    fiber_norms[fiber_norms < 1e-6] = 1.0
    fiber_dirs_vol /= fiber_norms
    
    brain_mask = mask > 0
    for vol in [f_csf_vol, f_gm_vol, f_wm_vol, f_restricted_vol, f_intra_vol]:
        vol[~brain_mask] = 0
    f_wm_fibers_vol[~brain_mask] = 0
    fiber_dirs_vol[~brain_mask] = 0
    mse_vol[~brain_mask] = 0
    
    total_time = time.time() - start_time
    print(f"\n  Total time: {total_time/60:.1f} min")
    
    print(f"\n[Saving NIfTI outputs to {args.outdir}/]")
    affine = img.affine
    
    nib.save(nib.Nifti1Image(f_csf_vol, affine), os.path.join(args.outdir, "f_csf.nii.gz"))
    nib.save(nib.Nifti1Image(f_gm_vol, affine), os.path.join(args.outdir, "f_gm.nii.gz"))
    nib.save(nib.Nifti1Image(f_wm_vol, affine), os.path.join(args.outdir, "f_wm.nii.gz"))
    nib.save(nib.Nifti1Image(f_restricted_vol, affine), os.path.join(args.outdir, "f_restricted.nii.gz"))
    nib.save(nib.Nifti1Image(f_intra_vol, affine), os.path.join(args.outdir, "f_intra.nii.gz"))
    nib.save(nib.Nifti1Image(mse_vol, affine), os.path.join(args.outdir, "mse.nii.gz"))
    nib.save(nib.Nifti1Image(f_wm_fibers_vol, affine), os.path.join(args.outdir, "f_wm_fibers.nii.gz"))
    fiber_dirs_flat = fiber_dirs_vol.reshape(X, Y, Z, n_fibers * 3)
    nib.save(nib.Nifti1Image(fiber_dirs_flat, affine), os.path.join(args.outdir, "fiber_dirs.nii.gz"))
    
    print(f"  Saved: f_csf.nii.gz, f_gm.nii.gz, f_wm.nii.gz, f_restricted.nii.gz")
    print(f"  Saved: f_intra.nii.gz, mse.nii.gz, f_wm_fibers.nii.gz, fiber_dirs.nii.gz")
    
    summary = {
        'dataset': args.name or os.path.basename(args.dwi),
        'mode': 'whole_brain_slab',
        'slab_size': slab_size,
        'overlap': overlap,
        'volume_shape': [X, Y, Z],
        'n_fibers': n_fibers,
        'n_steps': args.n_steps,
        'slabs_fitted': len(slabs),
        'total_time_sec': total_time,
        'mean_mse': float(mse_vol[mask > 0].mean()),
        'n_measurements': int(N),
        'b_values': [float(b) for b in np.unique(bvals.round(-2))],
    }
    
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: summary.json")
    
    if getattr(args, 'export_artifacts', False):
        export_scanner_artifacts(
            scanner=scanner,
            outdir=args.outdir,
            affine=affine,
            spatial_shape=(X, Y, Z),
            mask=mask,
            bvals=bvals,
            dataset_name=args.name or os.path.basename(args.dwi),
            dwi_signal=dwi,
        )
    
    print(f"\n[Creating visualization panels for 3 axes...]")
    create_3axis_panels_from_volumes(
        args=args,
        f_csf_vol=f_csf_vol,
        f_gm_vol=f_gm_vol,
        f_wm_vol=f_wm_vol,
        f_wm_fibers_vol=f_wm_fibers_vol,
        f_restricted_vol=f_restricted_vol,
        f_intra_vol=f_intra_vol,
        fiber_dirs_vol=fiber_dirs_vol,
        mse_vol=mse_vol,
        mask=mask,
        dwi=dwi,
        bvals=bvals,
        n_fibers=n_fibers,
        dataset_name=args.name or os.path.basename(args.dwi),
    )
    
    print(f"\n✓ Whole brain slab fitting complete!")
    print(f"  Mean MSE: {summary['mean_mse']:.5f}")
    print(f"  Output: {args.outdir}/")


def _create_blend_weights_1d(size, overlap, at_start, at_end):
    weight = np.ones(size, dtype=np.float32)
    
    if overlap > 0:
        if not at_start:
            taper = 0.5 * (1 - np.cos(np.pi * np.arange(overlap) / overlap))
            weight[:overlap] = taper
        
        if not at_end:
            taper = 0.5 * (1 + np.cos(np.pi * np.arange(overlap) / overlap))
            weight[-overlap:] = taper
    
    return weight


def smooth_fiber_directions(fiber_dirs, mask, f_wm=None, sigma=1.0, n_fibers=3):
    from scipy.ndimage import gaussian_filter
    from scipy.ndimage import label
    import numpy as np
    
    print(f"\n[Post-processing: Smoothing fiber directions (σ={sigma})]")
    
    X, Y, Z = fiber_dirs.shape[:3]
    smoothed = np.zeros_like(fiber_dirs)
    
    for fib in range(n_fibers):
        print(f"  Fiber {fib+1}/{n_fibers}...")
        dir_field = fiber_dirs[..., fib, :].copy()
        
        dir_norm = np.linalg.norm(dir_field, axis=-1)
        valid = mask & (dir_norm > 0.1)
        
        if valid.sum() < 10:
            smoothed[..., fib, :] = dir_field
            continue
        
        aligned = dir_field.copy()
        
        coords = np.array(np.where(valid)).T
        center = coords.mean(axis=0).astype(int)
        
        visited = np.zeros((X, Y, Z), dtype=bool)
        queue = [tuple(center)]
        visited[tuple(center)] = True
        
        neighbors = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
        
        while queue:
            cx, cy, cz = queue.pop(0)
            current_dir = aligned[cx, cy, cz]
            
            for dx, dy, dz in neighbors:
                nx, ny, nz = cx + dx, cy + dy, cz + dz
                
                if 0 <= nx < X and 0 <= ny < Y and 0 <= nz < Z:
                    if valid[nx, ny, nz] and not visited[nx, ny, nz]:
                        visited[nx, ny, nz] = True
                        queue.append((nx, ny, nz))
                        
                        neighbor_dir = aligned[nx, ny, nz]
                        dot = np.dot(current_dir, neighbor_dir)
                        if dot < 0:
                            aligned[nx, ny, nz] = -neighbor_dir
        
        if f_wm is not None:
            weight = f_wm * valid.astype(float)
        else:
            weight = valid.astype(float)
        
        smoothed_field = np.zeros_like(dir_field)
        weight_smooth = gaussian_filter(weight, sigma) + 1e-8
        
        for dim in range(3):
            weighted_dir = aligned[..., dim] * weight
            smoothed_field[..., dim] = gaussian_filter(weighted_dir, sigma) / weight_smooth
        
        smoothed_field[~valid] = 0
        
        norms = np.linalg.norm(smoothed_field, axis=-1, keepdims=True)
        norms[norms < 1e-6] = 1.0
        smoothed_field /= norms
        
        smoothed_field[~valid] = dir_field[~valid]
        
        smoothed[..., fib, :] = smoothed_field
    
    original_norms = np.linalg.norm(fiber_dirs, axis=-1)
    valid_voxels = mask & (original_norms[..., 0] > 0.1)
    if valid_voxels.sum() > 0:
        dot_products = np.sum(fiber_dirs * smoothed, axis=-1)
        dot_products = np.clip(dot_products, -1, 1)
        angles = np.degrees(np.arccos(np.abs(dot_products)))
        mean_change = np.nanmean(angles[valid_voxels, 0])
        print(f"  Mean angular change from smoothing: {mean_change:.2f}°")
    
    return smoothed


def _create_blend_weights(cx, cy, cz, overlap, at_start=(False, False, False), at_end=(False, False, False), use_blend=True):
    if not use_blend:
        weight = np.ones((cx, cy, cz), dtype=np.float32)
        return weight
    
    def taper_1d(size, overlap, taper_start=True, taper_end=True):
        w = np.ones(size, dtype=np.float32)
        if overlap > 0 and size > 2 * overlap:
            taper = 0.5 * (1 - np.cos(np.pi * np.arange(overlap) / overlap))
            if taper_start:
                w[:overlap] = taper
            if taper_end:
                w[-overlap:] = taper[::-1]
        return w
    
    wx = taper_1d(cx, overlap, taper_start=not at_start[0], taper_end=not at_end[0])
    wy = taper_1d(cy, overlap, taper_start=not at_start[1], taper_end=not at_end[1])
    wz = taper_1d(cz, overlap, taper_start=not at_start[2], taper_end=not at_end[2])
    
    weight = wx[:, np.newaxis, np.newaxis] * wy[np.newaxis, :, np.newaxis] * wz[np.newaxis, np.newaxis, :]
    return weight


def fit_patch_3d(
    scanner: DifferentiableScannerV4,
    signal_slab: np.ndarray,
    mask_slab: np.ndarray,
    n_steps: int = 200,
    lr: float = 0.02,
    use_topology: bool = False,
    use_spatial_prior: bool = False,
    use_6conn_topology: bool = False,
    use_26conn_topology: bool = False,
    lambda_orphan: float = 0.01,
    lambda_continuity: float = 0.005,
    lambda_ordering: float = 0.01,
    lambda_repulsion: float = 0.01,
    lambda_curvature: float = 0.0,
    lambda_perm_invariant: float = 0.0,
    device: str = 'cuda',
    use_encoder: bool = False,
    encoder_epochs: int = 50,
    encoder = None,
    use_rician: bool = False,
    rician_mode: str = 'hybrid',
    hybrid_mse_weight: float = 0.7,
    noise_sigma: float = None,
    sigma_map_precomputed: torch.Tensor = None,
    auto_noise_sigma: str = 'background',
    use_sigma_map: bool = False,
    sigma_bg_margin: int = 3,
    sigma_estimation_method: str = 'median',
    learn_noise_sigma: bool = False,
    staged_directions: bool = False,
    stage1_freeze_steps: int = 100,
    direction_lr_factor: float = 0.1,
    signal_raw: np.ndarray = None,
    bvals_np: np.ndarray = None,
    b_switch: float = 2000.0,
    b_slope: float = 0.002,
    sparsity_weight: float = 0.02,
    entropy_weight: float = 0.0,
    huber_delta: float = 1.0,
    lambda_cosine: float = 0.05,
    early_patience: int = 30,
    early_warmup: int = 50,
    early_min_steps: int = 100,
    early_rel_tol: float = 1e-4,
    no_early_stop: bool = False,
    mse_warmup_steps: int = 0,
) -> dict:
    H, W, D, N = signal_slab.shape
    n_fibers = scanner.n_fibers
    
    b0_idx = scanner.protocol.bvals.cpu().numpy() < 100
    if b0_idx.sum() > 0:
        s0 = np.maximum(signal_slab[..., b0_idx].mean(axis=-1, keepdims=True), 1e-6)
        signal_norm = np.clip(signal_slab / s0, 0, 3)
    else:
        signal_norm = signal_slab / (signal_slab.max() + 1e-6)
    
    target = torch.tensor(signal_norm, dtype=torch.float32, device=device)
    target = target.unsqueeze(0)
    
    mask_t = torch.tensor(mask_slab, dtype=torch.bool, device=device)
    mask_t = mask_t.unsqueeze(0)
    
    B, X, Y, Z = 1, H, W, D
    
    if use_encoder:
        if encoder is None:
            print("    Creating MLP encoder...")
            encoder = MicrostructureEncoder(N, n_fibers, hidden_dim=128).to(device)
            encoder = train_encoder_on_the_fly(
                encoder, scanner, target, mask_t,
                n_epochs=encoder_epochs, lr=0.001, device=device
            )
        else:
            encoder = train_encoder_on_the_fly(
                encoder, scanner, target, mask_t,
                n_epochs=min(10, encoder_epochs), lr=0.0003, device=device
            )
        params = initialize_params_from_encoder(encoder, target, mask_t, n_fibers, device)
    else:
        encoder = None
        params = {
            'raw_csf': (torch.randn(B, X, Y, Z, device=device) * 0.5 - 1.0).requires_grad_(True),
            'raw_gm': (torch.randn(B, X, Y, Z, device=device) * 0.5 - 0.5).requires_grad_(True),
            'raw_wm': (torch.randn(B, X, Y, Z, n_fibers, device=device) * 0.5 + 0.5).requires_grad_(True),
            'raw_restricted': (torch.randn(B, X, Y, Z, device=device) * 0.5 - 0.5).requires_grad_(True),
            'fiber_dirs': sample_uniform_sphere_stratified(
                (B, X, Y, Z, n_fibers, 3), device=device
            ).requires_grad_(True),
            'f_intra': (torch.rand(B, X, Y, Z, device=device) * 0.3 + 0.5).requires_grad_(True),
            'kappa': (torch.ones(B, X, Y, Z, device=device) * 8.0).requires_grad_(True),
            's0': (torch.ones(B, X, Y, Z, device=device)).requires_grad_(True),
        }
    
    scanner.use_tissue_priors = use_topology
    
    sigma_t = None
    sigma_map_t = None
    sigma_param = None
    sigma_estimation_result = None
    
    if use_rician:
        if sigma_map_precomputed is not None:
            sigma_map_t = sigma_map_precomputed.to(device=device, dtype=torch.float32)
            if sigma_map_t.dim() == 3:
                sigma_map_t = sigma_map_t.unsqueeze(0).unsqueeze(-1)
            elif sigma_map_t.dim() == 4:
                sigma_map_t = sigma_map_t.unsqueeze(-1)
            sigma_t = sigma_map_t
            
        elif noise_sigma is not None:
            sigma_t = torch.tensor(noise_sigma, dtype=torch.float32, device=device)
            print(f"[Rician] Using fixed noise sigma = {noise_sigma:.4f}")
            
        elif auto_noise_sigma == 'none':
            raise ValueError(
                "--use-rician requires noise sigma. Either provide --noise-sigma "
                "or use --auto-noise-sigma=background (recommended)."
            )
            
        elif auto_noise_sigma in ['background', 'b0']:
            print(f"[Rician] Estimating σ locally using method: {auto_noise_sigma}")
            
            if signal_raw is not None and bvals_np is not None:
                signal_raw_t = torch.tensor(signal_raw, dtype=torch.float32, device=device)
                bvals_t_est = torch.tensor(bvals_np, dtype=torch.float32, device=device)
                
                try:
                    sigma_estimation_result = estimate_noise_sigma(
                        signal_raw_t.unsqueeze(0),
                        mask=mask_t,
                        bvals=bvals_t_est,
                        method=auto_noise_sigma,
                        compute_map=use_sigma_map,
                        bg_margin=sigma_bg_margin,
                        min_bg_voxels=500,
                        print_diagnostics=True,
                    )
                    
                    if use_sigma_map and sigma_estimation_result.sigma_map is not None:
                        sigma_map_t = sigma_estimation_result.sigma_map.to(device)
                        sigma_t = sigma_map_t
                    else:
                        sigma_t = torch.tensor(
                            sigma_estimation_result.sigma_normalized,
                            dtype=torch.float32, device=device
                        )
                        
                except Exception as e:
                    print(f"[Rician] Warning: σ estimation failed ({e}), falling back to percentile method")
                    signal_flat = target[mask_t.unsqueeze(-1).expand_as(target)]
                    p5 = torch.quantile(signal_flat, 0.05)
                    sigma_est = (p5 / 0.320).clamp(min=0.01, max=0.2)
                    sigma_t = sigma_est
                    print(f"[Rician] Fallback σ = {sigma_t.item():.4f}")
            else:
                print(f"[Rician] Warning: Raw signal not available, using percentile fallback")
                signal_flat = target[mask_t.unsqueeze(-1).expand_as(target)]
                p5 = torch.quantile(signal_flat, 0.05)
                sigma_est = (p5 / 0.320).clamp(min=0.01, max=0.2)
                sigma_t = sigma_est
                print(f"[Rician] Estimated σ = {sigma_t.item():.4f} (percentile fallback)")
        
        if learn_noise_sigma:
            if sigma_map_t is not None:
                print("[Rician] Warning: learn_noise_sigma with σ-map not implemented, using scalar")
                sigma_scalar = sigma_estimation_result.sigma_normalized if sigma_estimation_result else 0.1
                sigma_t = torch.tensor(sigma_scalar, dtype=torch.float32, device=device)
            log_sigma = torch.log(sigma_t).clone().requires_grad_(True)
            sigma_param = log_sigma
    
    bvals_t = scanner.protocol.bvals
    
    unique_bvals = torch.unique(bvals_t.round(decimals=-2))
    shell_ids_t = torch.zeros(bvals_t.shape[0], dtype=torch.long, device=device)
    for i, bv in enumerate(unique_bvals):
        shell_ids_t[torch.abs(bvals_t - bv) < 100] = i
    
    scanner_params_list = list(scanner.parameters())
    
    if staged_directions:
        params['fiber_dirs'].requires_grad_(False)
        print(f"[Staged] Stage 1: Freezing fiber_dirs for {stage1_freeze_steps} steps")
        all_params_except_dirs = [p for k, p in params.items() if k != 'fiber_dirs']
        if sigma_param is not None:
            all_params_except_dirs.append(sigma_param)
        all_params_except_dirs.extend(scanner_params_list)
        optimizer = torch.optim.Rprop(all_params_except_dirs, lr=lr)
    else:
        all_params = list(params.values())
        if sigma_param is not None:
            all_params.append(sigma_param)
        all_params.extend(scanner_params_list)
        optimizer = torch.optim.Rprop(all_params, lr=lr)
    
    losses = []
    
    if no_early_stop:
        early_stopper = None
    else:
        early_stopper = SmartEarlyStopping(
            patience=early_patience,
            warmup=early_warmup,
            rel_tol=early_rel_tol,
            grad_tol=1e-6,
            ema_alpha=0.1,
            oscillation_window=20,
            min_steps=early_min_steps,
        )
    
    pbar = tqdm(range(n_steps), desc="Fitting slab")
    
    for step in pbar:
        optimizer.zero_grad()
        
        tissue_raw = torch.cat([
            params['raw_csf'].unsqueeze(-1),
            params['raw_gm'].unsqueeze(-1),
            params['raw_wm'],
            params['raw_restricted'].unsqueeze(-1),
        ], dim=-1)
        tissue_fracs = F.softmax(tissue_raw, dim=-1)
        
        f_csf = tissue_fracs[..., 0]
        f_gm = tissue_fracs[..., 1]
        f_wm = tissue_fracs[..., 2:2+n_fibers]
        f_restricted = tissue_fracs[..., -1]
        
        scanner_params = {
            'f_csf': f_csf,
            'f_gm': f_gm,
            'f_wm': f_wm,
            'f_restricted': f_restricted,
            'fiber_dirs': F.normalize(params['fiber_dirs'], dim=-1),
            'f_intra': torch.sigmoid(params['f_intra']),
            'kappa': F.softplus(params['kappa']) + 0.1,
            's0': F.softplus(params['s0']),
        }
        
        pred = scanner(scanner_params, add_noise=False)
        
        mask_expanded = mask_t.unsqueeze(-1).expand_as(pred)
        
        use_mse_phase = (mse_warmup_steps > 0 and step < mse_warmup_steps)
        
        if use_mse_phase:
            mse = F.mse_loss(pred * mask_expanded.float(), target * mask_expanded.float(), reduction='sum')
            loss = mse / mask_expanded.sum().clamp(min=1)
            if step == 0:
                tqdm.write(f"[Two-Phase] Using MSE for first {mse_warmup_steps} steps, then switching to {rician_mode}")
        elif use_rician:
            if step == mse_warmup_steps and mse_warmup_steps > 0:
                tqdm.write(f"[Two-Phase] Switching to {rician_mode} loss at step {step}")
            
            if sigma_param is not None:
                sigma_current = torch.exp(sigma_param)
            else:
                sigma_current = sigma_t
            
            if rician_mode == 'nll':
                nll = rician_nll(target, pred, sigma_current)
                loss = nll[mask_expanded].mean()
            elif rician_mode == 'bias-corrected':
                loss = bias_corrected_mse_loss(pred, target, sigma_current, mask=mask_t)
            elif rician_mode == 'hybrid':
                loss = hybrid_mse_rician_loss(
                    pred, target, sigma_current, 
                    mse_weight=hybrid_mse_weight, 
                    rician_weight=1.0 - hybrid_mse_weight,
                    mask=mask_t
                )
            elif rician_mode == 'b-dependent':
                loss = b_dependent_hybrid_loss(
                    pred, target, sigma_current, 
                    bvals=bvals_t,
                    b_switch=b_switch,
                    b_slope=b_slope,
                    mask=mask_t
                )
            elif rician_mode == 'snr-dependent':
                loss = snr_dependent_hybrid_loss(
                    pred, target, sigma_current,
                    snr_switch=3.0,
                    snr_slope=1.0,
                    mask=mask_t
                )
            elif rician_mode == 'huber':
                loss, _ = loss_dmri(
                    pred=pred, target=target,
                    shell_ids=shell_ids_t, bvals=bvals_t,
                    sigma=sigma_current,
                    mode='huber',
                    delta=huber_delta,
                    mask=mask_t
                )
            elif rician_mode == 'huber-rician':
                loss, _ = loss_dmri(
                    pred=pred, target=target,
                    shell_ids=shell_ids_t, bvals=bvals_t,
                    sigma=sigma_current,
                    mode='rician',
                    delta=huber_delta,
                    mask=mask_t
                )
            elif rician_mode == 'huber-debiased':
                loss, _ = loss_dmri(
                    pred=pred, target=target,
                    shell_ids=shell_ids_t, bvals=bvals_t,
                    sigma=sigma_current,
                    mode='huber_debiased',
                    delta=huber_delta,
                    mask=mask_t
                )
            elif rician_mode == 'huber-squared':
                loss, _ = loss_dmri(
                    pred=pred, target=target,
                    shell_ids=shell_ids_t, bvals=bvals_t,
                    sigma=sigma_current,
                    mode='huber_squared',
                    delta=huber_delta,
                    mask=mask_t
                )
            elif rician_mode == 'squared-magnitude':
                loss, _ = loss_dmri(
                    pred=pred, target=target,
                    shell_ids=shell_ids_t, bvals=bvals_t,
                    sigma=sigma_current,
                    mode='squared_magnitude',
                    delta=huber_delta,
                    mask=mask_t
                )
            elif rician_mode == 'sqmag-charb':
                loss, _ = loss_dmri(
                    pred=pred, target=target,
                    shell_ids=shell_ids_t, bvals=bvals_t,
                    sigma=sigma_current,
                    mode='sqmag_charb',
                    delta=huber_delta,
                    mask=mask_t
                )
            elif rician_mode == 'sqmag-charb-cosine':
                loss, info_loss = loss_dmri(
                    pred=pred, target=target,
                    shell_ids=shell_ids_t, bvals=bvals_t,
                    sigma=sigma_current,
                    mode='sqmag_charb_cosine',
                    delta=huber_delta,
                    lambda_cosine=lambda_cosine,
                    mask=mask_t
                )
                if step % 100 == 0 and step > 0:
                    print(f"    [SqMag+Cos] amp={info_loss.get('amp_loss', 0):.4f}, cos={info_loss.get('cos_loss', 0):.4f}")
            else:
                loss = bias_corrected_mse_loss(pred, target, sigma_current, mask=mask_t)
            
            if sigma_param is not None:
                sigma_reg = 0.1 * (sigma_param ** 2)
                loss = loss + sigma_reg
        else:
            loss = F.mse_loss(pred[mask_expanded], target[mask_expanded])
        
        if staged_directions and step == stage1_freeze_steps:
            print(f"[Staged] Step {step}: Unfreezing fiber_dirs with LR factor {direction_lr_factor}")
            params['fiber_dirs'].requires_grad_(True)
            other_params = [p for k, p in params.items() if k != 'fiber_dirs']
            other_params.extend(scanner_params_list)
            param_groups = [
                {'params': other_params, 'lr': lr},
                {'params': [params['fiber_dirs']], 'lr': lr * direction_lr_factor},
            ]
            if sigma_param is not None:
                param_groups[0]['params'].append(sigma_param)
            optimizer = torch.optim.Rprop(param_groups)
        
        reg_params = {'f_csf': f_csf, 'f_gm': f_gm, 'f_wm': f_wm, 'f_restricted': f_restricted}
        reg_loss = scanner.get_regularization_loss(reg_params, w_micro=0.1, mask=mask_t)
        
        with torch.no_grad():
            signal_mean = target.mean(dim=-1, keepdim=True)
            signal_std = target.std(dim=-1)
            cv = signal_std / (signal_mean.squeeze(-1) + 1e-6)
            cv_scaled = ((cv - 0.1) / 0.3).clamp(0, 1)
            aniso_weight = cv_scaled
        
        bvals_np = scanner.protocol.bvals.cpu().numpy()
        aniso_matching_loss = torch.tensor(0.0, device=pred.device)
        
        unique_bvals = np.unique(bvals_np[bvals_np > 100])
        for b in unique_bvals:
            shell_mask = np.abs(bvals_np - b) < 100
            if shell_mask.sum() < 6:
                continue
            
            pred_shell = pred[..., shell_mask]
            target_shell = target[..., shell_mask]
            
            pred_mean = pred_shell.mean(dim=-1) + 1e-6
            pred_std = pred_shell.std(dim=-1)
            pred_cv = pred_std / pred_mean
            
            target_mean = target_shell.mean(dim=-1) + 1e-6
            target_std = target_shell.std(dim=-1)
            target_cv = target_std / target_mean
            
            cv_diff = (pred_cv - target_cv) ** 2
            aniso_matching_loss = aniso_matching_loss + (cv_diff * mask_t.float()).sum() / (mask_t.float().sum() + 1e-6)
        
        if len(unique_bvals) > 0:
            aniso_matching_loss = aniso_matching_loss / len(unique_bvals)
        aniso_matching_weight = 0.1
        
        f_wm_total = f_wm.sum(dim=-1) if f_wm.dim() > 4 else f_wm
        iso_penalty = ((1 - aniso_weight) * f_wm_total * mask_t.float()).mean()
        aniso_prior_weight = 0.01
        
        if f_wm.dim() > 4 and f_wm.shape[-1] > 1:
            f_wm_sum = f_wm.sum(dim=-1)
            f_wm_sq_sum = (f_wm ** 2).sum(dim=-1)
            spread = f_wm_sum ** 2 - f_wm_sq_sum
            conc_penalty = ((1 - aniso_weight) * spread * mask_t.float()).mean()
            conc_weight = 0.05
            
            small_fractions = f_wm[f_wm < 0.15]
            sparsity_loss = small_fractions.abs().mean() if len(small_fractions) > 0 else 0.0
            
            f_wm_sum = f_wm.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            f_wm_prob = f_wm / f_wm_sum
            entropy_loss = -(f_wm_prob * (f_wm_prob + 1e-8).log()).sum(dim=-1)
            entropy_loss = (entropy_loss * mask_t.float()).mean()
        else:
            conc_penalty = 0.0
            conc_weight = 0.0
            sparsity_loss = 0.0
            entropy_loss = 0.0
        
        if f_wm.dim() > 4 and f_wm.shape[-1] > 1:
            fiber_dirs_norm = F.normalize(params['fiber_dirs'], dim=-1)
            repel_loss = 0.0
            n_fib = f_wm.shape[-1]
            for i in range(n_fib):
                for j in range(i + 1, n_fib):
                    dot_ij = (fiber_dirs_norm[..., i, :] * fiber_dirs_norm[..., j, :]).sum(dim=-1)
                    f_prod = f_wm[..., i] * f_wm[..., j]
                    repel_loss = repel_loss + (f_prod * torch.abs(dot_ij) * mask_t.float()).mean()
            repel_weight = 0.01
        else:
            repel_loss = 0.0
            repel_weight = 0.0
        
        topo_loss = 0.0
        use_topo_losses = use_6conn_topology or use_26conn_topology
        if use_topo_losses:
            connectivity = 26 if use_26conn_topology else 6
            topo_losses = scanner.get_topology_losses(
                scanner_params,
                lambda_orphan=lambda_orphan,
                lambda_continuity=lambda_continuity,
                lambda_endpoint=0.0,
                lambda_ordering=lambda_ordering,
                lambda_repulsion=lambda_repulsion,
                lambda_curvature=lambda_curvature,
                lambda_perm_invariant=lambda_perm_invariant,
                connectivity=connectivity,
            )
            topo_loss = topo_losses['total']
        
        total_loss = loss + reg_loss + aniso_prior_weight * iso_penalty + conc_weight * conc_penalty + repel_weight * repel_loss + topo_loss + aniso_matching_weight * aniso_matching_loss + sparsity_weight * sparsity_loss + entropy_weight * entropy_loss
        
        total_loss.backward()
        
        grad_norm = compute_grad_norm(params) if early_stopper is not None else None
        
        optimizer.step()
        
        losses.append(loss.item())
        if use_rician:
            postfix = {'NLL': f'{loss.item():.4f}'}
            if sigma_param is not None:
                postfix['σ'] = f'{torch.exp(sigma_param).item():.4f}'
        else:
            postfix = {'MSE': f'{loss.item():.5f}'}
        if use_topo_losses and step % 50 == 0:
            postfix[f'Topo{connectivity}'] = f'{topo_loss.item():.4f}'
        if isinstance(aniso_matching_loss, torch.Tensor) and step % 50 == 0:
            postfix['AnisoCV'] = f'{aniso_matching_loss.item():.4f}'
        pbar.set_postfix(postfix)
        
        if early_stopper is not None and early_stopper.step(loss.item(), step, grad_norm=grad_norm):
            pbar.set_description(f"Early stop @ {step}: {early_stopper.stop_reason}")
            pbar.close()
            break
    
    with torch.no_grad():
        tissue_raw = torch.cat([
            params['raw_csf'].unsqueeze(-1),
            params['raw_gm'].unsqueeze(-1),
            params['raw_wm'],
            params['raw_restricted'].unsqueeze(-1),
        ], dim=-1)
        tissue_fracs = F.softmax(tissue_raw, dim=-1)
        
        final_params = {
            'f_csf': tissue_fracs[..., 0],
            'f_gm': tissue_fracs[..., 1],
            'f_wm': tissue_fracs[..., 2:2+n_fibers],
            'f_restricted': tissue_fracs[..., -1],
            'fiber_dirs': F.normalize(params['fiber_dirs'], dim=-1),
            'f_intra': torch.sigmoid(params['f_intra']),
            'kappa': F.softplus(params['kappa']) + 0.1,
            's0': F.softplus(params['s0']),
            'mse': losses[-1] if losses else 0.0,
        }
        
        final_params = sort_fibers_by_direction_coherence(final_params, mask_t)
        
        pred_signal = scanner(final_params, add_noise=False)
        final_params['pred_signal'] = pred_signal
        final_params['target_signal'] = target
        final_params['mask'] = mask_t
        final_params['encoder'] = encoder
    
    return final_params


def main():
    parser = argparse.ArgumentParser(
        description="Simple microstructure fitting script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fit HCP data
  python simple_fit_microstructure.py \\
      --dwi /path/to/data.nii.gz \\
      --bvals /path/to/bvals \\
      --bvecs /path/to/bvecs \\
      --slice 72 --outdir outputs/fitting/hcp

  # Fit with topology prior
  python simple_fit_microstructure.py \\
      --dwi dwi.nii.gz --bvals bvals --bvecs bvecs \\
      --use-topology --n-steps 300 --outdir outputs/fitting/topo

  # Fit whole brain (all slices)
  python simple_fit_microstructure.py \\
      --dwi data.nii.gz --bvals bvals --bvecs bvecs --mask mask.nii.gz \\
      --all-slices --n-steps 500 --outdir outputs/fitting/wholebrain
"""
    )
    parser.add_argument("--dwi", required=True, help="Path to DWI NIfTI")
    parser.add_argument("--bvals", required=True, help="Path to bvals file")
    parser.add_argument("--bvecs", required=True, help="Path to bvecs file")
    
    input_mask_group = parser.add_argument_group('Input Masking (before fitting)')
    input_mask_group.add_argument("--mask", default=None, 
        help="Brain mask NIfTI - applied to DWI before fitting")
    input_mask_group.add_argument("--median-otsu", action="store_true",
        help="Compute brain mask using DIPY's median_otsu on b0 image (good for preclinical data)")
    input_mask_group.add_argument("--median-otsu-median-radius", type=int, default=2,
        help="Radius for median filter in median_otsu (default: 2, try 1-4)")
    input_mask_group.add_argument("--median-otsu-numpass", type=int, default=2,
        help="Number of passes for median_otsu (default: 2, try 1-4)")
    input_mask_group.add_argument("--auto-mask-scale", type=float, default=0.8,
        help="Scale factor applied to auto Otsu threshold (default: 0.8; lower = looser mask)")
    input_mask_group.add_argument("--mask-dilate-iters", type=int, default=0,
        help="Binary dilation iterations applied to auto/median_otsu masks (default: 0)")
    
    output_mask_group = parser.add_argument_group('Output Masking (after fitting, for visualization)')
    output_mask_group.add_argument("--output-r2-mask", type=float, default=None, metavar="THRESHOLD",
        help="Apply R²-based mask to output visualizations (e.g., 0.5). Only voxels with R²>threshold shown.")
    output_mask_group.add_argument("--output-rel-res-mask", type=float, default=None, metavar="THRESHOLD",
        help="Apply relative residual mask to output visualizations (e.g., 0.15). Only voxels with rel_res<threshold shown.")
    
    parser.add_argument("--apply-mask", action="store_true",
                        help="[DEPRECATED] Zero out DWI signal outside mask")
    parser.add_argument("--r2-threshold", type=float, default=None,
                        help="[DEPRECATED] Use --output-r2-mask instead")
    
    parser.add_argument("--slice", type=int, default=None, help="Axial slice index (default: middle)")
    parser.add_argument("--all-slices", action="store_true", help="Fit all slices (whole brain)")
    parser.add_argument("--slab-mode", action="store_true", help="Use slab mode: fit N slices at a time (full X,Y resolution)")
    parser.add_argument("--slab-size", type=int, default=5, help="Number of slices per slab in slab mode (default: 5)")
    parser.add_argument("--cube-size", type=int, default=50, help="Cube size for 3D fitting (default: 50)")
    parser.add_argument("--overlap", type=int, default=10, help="Overlap between cubes/slabs for blending (default: 10, increased from 3 for better direction continuity)")
    parser.add_argument("--smooth-fibers", type=float, default=0.0, 
                        help="Post-processing: Gaussian smoothing sigma for fiber directions (default: 0=off, try 0.5-1.0)")
    parser.add_argument("--no-blend", action="store_true",
                        help="Disable cosine blending in overlap regions (use uniform weights=1.0 instead of cosine taper)")
    parser.add_argument("--refine-boundaries", action="store_true",
                        help="Two-pass fitting: after first pass, re-fit boundary/overlap regions while keeping interior frozen")
    parser.add_argument("--refine-slicewise", action="store_true",
                        help="Two-pass fitting: Gauss-Seidel style - 1 step per slice, sweep all slices, repeat N times")
    parser.add_argument("--refine-sweeps", type=int, default=50,
                        help="Number of full sweeps through all slices for slicewise refinement (default: 50)")
    
    sweep_group = parser.add_argument_group('Sweep Mode (6 Orientations)')
    sweep_group.add_argument("--sweep-mode", action="store_true",
        help="Use sweep mode: process all slices with 6 different orientations (X+, X-, Y+, Y-, Z+, Z-). "
             "Each orientation sweeps through slices sequentially, doing N steps per slice. "
             "This avoids cube boundary artifacts and ensures global coherence.")
    sweep_group.add_argument("--sweeps-per-orientation", type=int, default=10,
        help="Number of sweeps per orientation (default: 10, total = 6 * N = 60)")
    sweep_group.add_argument("--steps-per-slice", type=int, default=5,
        help="Number of optimization steps per slice (default: 5)")
    
    parser.add_argument("--boundary-width", type=int, default=None,
                        help="Width of boundary region to refine (default: overlap/2)")
    parser.add_argument("--boundary-steps", type=int, default=200,
                        help="Number of optimization steps for boundary refinement (default: 200)")
    parser.add_argument("--patch-size", type=int, default=0, help="Patch size (0 = full slice)")
    parser.add_argument("--cx", type=int, default=None, help="Patch center x")
    parser.add_argument("--cy", type=int, default=None, help="Patch center y")
    parser.add_argument("--n-fibers", type=int, default=3, help="Number of fiber compartments")
    parser.add_argument("--n-steps", type=int, default=200, help="Optimization steps")
    
    staged_group = parser.add_argument_group('Staged Training [RESEARCH]')
    staged_group.add_argument("--staged-training", action="store_true",
        help="Enable 2-stage training: Stage 1 learns global responses, Stage 2 learns per-voxel fractions")
    staged_group.add_argument("--stage1-steps", type=int, default=50,
        help="Number of steps for Stage 1 (global response learning). Default: 50")
    
    signal_group = parser.add_argument_group('Signal Model Components')
    signal_group.add_argument("--use-restricted", action="store_true", default=True,
        help="Enable restricted/soma compartment D~0.2 (default: on, +15-22%% improvement)")
    signal_group.add_argument("--no-restricted", action="store_true",
        help="Disable restricted compartment")
    signal_group.add_argument("--use-dot", action="store_true",
        help="Enable DOT/sphere compartment for ultra-high b (>10000)")
    signal_group.add_argument("--use-dispersion", action="store_true", default=False,
        help="Enable Watson orientation dispersion (default: OFF - redundant with restricted)")
    signal_group.add_argument("--no-dispersion", action="store_true",
        help="Disable Watson dispersion")
    signal_group.add_argument("--use-tortuosity", action="store_true",
        help="Enable tortuosity constraint: d_perp = d_parallel * (1-f_intra)")
    signal_group.add_argument("--use-kurtosis", action="store_true", default=False,
        help="Enable scalar kurtosis correction for non-Gaussian diffusion (default: off, use spectrum instead)")
    signal_group.add_argument("--no-kurtosis", action="store_true",
        help="Disable kurtosis correction")
    signal_group.add_argument("--use-dki", action="store_true",
        help="Enable full DKI with K_parallel, K_perp (overrides --use-kurtosis)")
    signal_group.add_argument("--use-diffusivity-spectrum", action="store_true", default=True,
        help="Use diffusivity spectrum model instead of kurtosis (Bernstein-compliant, default: on)")
    signal_group.add_argument("--no-diffusivity-spectrum", action="store_true",
        help="Disable diffusivity spectrum model")
    signal_group.add_argument("--n-spectrum-components", type=int, default=5,
        help="Number of diffusivity spectrum components (default: 5)")
    signal_group.add_argument("--use-t2-weighting", action="store_true", default=False,
        help="Enable T2/TE weighting (default: off, causes GM/WM degeneracy)")
    signal_group.add_argument("--no-t2-weighting", action="store_true",
        help="Disable T2 weighting (default)")
    signal_group.add_argument("--use-per-shell-modulation", action="store_true",
        help="Enable per-shell diffusivity adjustments")
    signal_group.add_argument("--use-biexp-gm", action="store_true",
        help="Use bi-exponential GM model: fast extracellular (D=0.9) + slow restricted (D=0.15, 45%%). Default: single isotropic.")
    signal_group.add_argument("--learn-diffusivities", action="store_true",
        help="Make tissue diffusivity priors learnable (d_gm, d_gm_slow, f_gm_restricted, d_parallel, d_perp). Useful for adapting to different datasets.")
    signal_group.add_argument("--kappa-prior", type=float, default=8.0,
        help="Watson concentration prior for fiber dispersion (default: 8.0). Higher = sharper bundles (kappa=32 for corpus callosum)")
    signal_group.add_argument("--f-intra-prior", type=float, default=0.5,
        help="Intra-axonal fraction prior (default: 0.5). Higher = more restricted water (0.6-0.8 for dense WM)")
    signal_group.add_argument("--sigmoid-fractions", action="store_true",
        help="Use sigmoid instead of softmax for tissue fractions. Compartments don't need to sum to 1, allowing a residual component.")
    signal_group.add_argument("--sigmoid-scale", type=float, default=0.5,
        help="Max value per compartment with sigmoid (default: 0.5). Lower = more room for residual.")
    
    curv_group = parser.add_argument_group('Curvature-Aware Physics (Novel)')
    curv_group.add_argument("--use-curvature-physics", action="store_true",
        help="Compute κ from fiber geometry instead of fitting freely. Uses κ = 1/(α·c²·L² + ε).")
    curv_group.add_argument("--curvature-alpha", type=float, default=10000.0,
        help="Curvature-to-dispersion scaling (default: 10000, optimal from DiSCo)")
    curv_group.add_argument("--curvature-epsilon", type=float, default=0.2,
        help="Minimum variance regularizer (default: 0.2, optimal from DiSCo)")
    curv_group.add_argument("--voxel-size", type=float, nargs=3, default=[1.0, 1.0, 1.0],
        help="Voxel size in mm for curvature computation (default: 1.0 1.0 1.0)")
    
    exchange_group = parser.add_argument_group('Kärger Exchange Model (Research)')
    exchange_group.add_argument("--use-exchange", action="store_true",
        help="Enable Kärger exchange between compartments (research feature)")
    exchange_group.add_argument("--exchange-tau", type=float, default=50.0,
        help="Initial exchange time τ_ex in ms (default: 50.0, typical: 20-100)")
    exchange_group.add_argument("--exchange-pairs", nargs="+", default=["restricted_gm"],
        choices=["restricted_gm", "wm_gm"],
        help="Compartment pairs for exchange (default: restricted_gm)")
    exchange_group.add_argument("--exchange-learnable", action="store_true", default=True,
        help="Allow exchange time to be learned (default: on)")
    exchange_group.add_argument("--no-exchange-learnable", action="store_true",
        help="Fix exchange time during optimization")
    
    mtcsd_group = parser.add_argument_group('Multi-Tissue CSD Model [EXPERIMENTAL]')
    mtcsd_group.add_argument("--use-mtcsd", action="store_true",
        help="[EXPERIMENTAL] Use MT-CSD style model. WARNING: May degrade fiber quality. Not recommended.")
    
    artifact_group = parser.add_argument_group('Artifact Model Components')
    artifact_group.add_argument("--learn-shell-gain", action="store_true", default=True,
        help="Learn per-shell S0 correction (CRITICAL for HCP, default: on)")
    artifact_group.add_argument("--no-shell-gain", action="store_true",
        help="Disable per-shell S0 learning")
    artifact_group.add_argument("--no-eddy", action="store_true",
        help="Disable per-measurement affine intensity correction")
    artifact_group.add_argument("--learn-bias-field", action="store_true", default=True,
        help="Learn B1 bias field correction (default: on)")
    artifact_group.add_argument("--no-bias-field", action="store_true",
        help="Disable bias field learning")
    artifact_group.add_argument("--learn-warps", action="store_true",
        help="Learn geometric warp field for EPI distortions")
    artifact_group.add_argument("--use-spatial-noise", action="store_true",
        help="Enable spatially varying noise model")
    artifact_group.add_argument("--use-psf", action="store_true",
        help="Enable PSF blur for neighborhood coupling")
    artifact_group.add_argument("--psf-sigma", type=float, default=0.68,
        help="PSF Gaussian sigma in voxels (default: 0.68 = FWHM~1.6)")
    artifact_group.add_argument("--psf-kernel-size", type=int, default=3, choices=[3, 5],
        help="PSF kernel size (default: 3)")
    artifact_group.add_argument("--use-ghosting", action="store_true",
        help="Enable Nyquist ghosting artifacts")
    artifact_group.add_argument("--use-qspace-mixing", action="store_true",
        help="Enable Q-space graph Laplacian mixing (block-diagonal per shell)")
    artifact_group.add_argument("--qspace-k-neighbors", type=int, default=6,
        help="Number of k-NN for Q-space Laplacian (default: 6)")
    artifact_group.add_argument("--learn-channel-gain", action="store_true",
        help="Enable per-channel gain correction to absorb per-direction signal variations")
    artifact_group.add_argument("--channel-gain-reg", type=float, default=0.1,
        help="Regularization weight for channel gain (default: 0.1). Lower = more flexibility to absorb noise.")
    artifact_group.add_argument("--low-rank-channel-gain", action="store_true",
        help="Use low-rank channel gains (reduces DOF from N_meas to k)")
    artifact_group.add_argument("--channel-gain-rank", type=int, default=5,
        help="Rank of channel gain subspace (default: 5)")
    artifact_group.add_argument("--export-artifacts", action="store_true",
        help="Export all learned scanner artifact maps (shell gains, bias field, "
             "channel gains, noise field, eddy currents, warps) as NIfTI/JSON + visualization panel")
    
    reg_group = parser.add_argument_group('Regularization & Priors')
    reg_group.add_argument("--use-tissue-priors", action="store_true", default=True,
        help="Enable global tissue fraction priors (default: on)")
    reg_group.add_argument("--no-tissue-priors", action="store_true",
        help="Disable tissue priors (recommended for small ROI/patch fitting)")
    reg_group.add_argument("--learnable-tissue-priors", action="store_true",
        help="Learn optimal tissue prior targets (WM/GM/CSF fractions) during fitting")
    reg_group.add_argument("--learnable-prior-weight", type=float, default=0.001,
        help="Weight for learnable tissue priors (default: 0.001)")
    reg_group.add_argument("--signal-aware-priors", action="store_true",
        help="Use signal-based per-voxel tissue priors (high b3000 = WM)")
    reg_group.add_argument("--signal-aware-weight", type=float, default=0.25,
        help="Weight for signal-aware priors (default: 0.25)")
    reg_group.add_argument("--wm-threshold", type=float, default=0.18,
        help="b3000 signal threshold for WM classification (default: 0.18)")
    reg_group.add_argument("--tissue-prior-dir", type=str, default=None,
        help="Directory with external tissue priors (wm_mask.nii.gz, gm_mask.nii.gz, csf_mask.nii.gz)")
    reg_group.add_argument("--tissue-prior-weight", type=float, default=0.1,
        help="Weight for external tissue prior loss (default: 0.1)")
    reg_group.add_argument("--use-spatial-prior", action="store_true", default=True,
        help="Enable 3D Laplacian spatial prior (default: True)")
    reg_group.add_argument("--no-spatial-prior", action="store_true",
        help="Disable spatial prior")
    reg_group.add_argument("--spatial-prior-weight", type=float, default=0.01,
        help="Weight for spatial prior (default: 0.01)")
    reg_group.add_argument("--spatial-prior-connectivity", choices=["6", "26"], default="26",
        help="Spatial prior connectivity: 6 (face) or 26 (full, default)")
    
    topo_group = parser.add_argument_group('Topology Priors')
    topo_group.add_argument("--use-topology", action="store_true", default=True,
        help="Enable basic topology prior (default: True)")
    topo_group.add_argument("--no-topology", action="store_true",
        help="Disable topology prior")
    topo_group.add_argument("--use-6conn-topology", action="store_true",
        help="Enable 6-connectivity topology losses (face neighbors only, cheaper)")
    topo_group.add_argument("--use-26conn-topology", action="store_true",
        help="Enable 26-connectivity topology losses (all neighbors including diagonals)")
    topo_group.add_argument("--lambda-orphan", type=float, default=0.01,
        help="Weight for orphan WM loss (default: 0.01)")
    topo_group.add_argument("--lambda-continuity", type=float, default=0.005,
        help="Weight for directional continuity loss (default: 0.005)")
    topo_group.add_argument("--lambda-ordering", type=float, default=0.01,
        help="Weight for fiber ordering loss (default: 0.01)")
    topo_group.add_argument("--lambda-repulsion", type=float, default=0.01,
        help="Weight for fiber repulsion loss (default: 0.01)")
    topo_group.add_argument("--lambda-endpoint", type=float, default=0.01,
        help="Weight for endpoint-in-GM loss (default: 0.01)")
    topo_group.add_argument("--lambda-curvature", type=float, default=0.0,
        help="Weight for streamline curvature prior [NEW] (default: 0.0, recommended: 0.02)")
    topo_group.add_argument("--lambda-perm-invariant", type=float, default=0.0,
        help="Weight for permutation-invariant orientation [NEW] (default: 0.0, recommended: 0.02)")
    topo_group.add_argument("--sparsity-weight", type=float, default=0.02,
        help="Weight for L1 sparsity on small fiber fractions (default: 0.02)")
    topo_group.add_argument("--entropy-weight", type=float, default=0.0,
        help="Weight for entropy-based sparsity (default: 0.0)")
    
    parser.add_argument("--max-bval", type=float, default=None,
        help="Maximum b-value to include (filter out higher b-values)")
    parser.add_argument("--use-encoder", action="store_true",
        help="Use MLP encoder for warm-start initialization")
    parser.add_argument("--encoder-epochs", type=int, default=10,
        help="Encoder self-supervised training epochs (default: 10)")
    
    rician_group = parser.add_argument_group('Rician Likelihood Loss')
    rician_group.add_argument("--use-rician", action="store_true",
        help="Use Rician-aware loss. Mode is selected by --rician-mode.")
    rician_group.add_argument("--rician-mode", 
        choices=["nll", "bias-corrected", "hybrid", "b-dependent", "snr-dependent", 
                 "huber", "huber-rician", "huber-debiased", "huber-squared", 
                 "squared-magnitude", "sqmag-charb", "sqmag-charb-cosine"], 
        default="hybrid",
        help="Rician loss mode: 'nll' = full Rician NLL (unstable at low SNR), "
             "'bias-corrected' = MSE on bias-corrected target, "
             "'hybrid' = blend MSE + Rician NLL (RECOMMENDED, stable + correct), "
             "'b-dependent' = MSE at low b, Rician at high b, "
             "'snr-dependent' = MSE at high pred SNR, Rician at low, "
             "'huber' = heteroscedastic Huber loss (fast + robust), "
             "'huber-rician' = Huber loss with fast Rician fallback, "
             "'huber-debiased' = Huber with Rician bias correction, "
             "'huber-squared' = Huber in squared-magnitude space, "
             "'squared-magnitude' = variance-normalized squared-mag Charbonnier, "
             "'sqmag-charb' = alias for squared-magnitude, "
             "'sqmag-charb-cosine' = SqMag-Charbonnier + angular cosine (FAST + BIAS-AWARE). "
             "Default: hybrid")
    rician_group.add_argument("--huber-delta", type=float, default=1.0,
        help="Huber/Charbonnier delta parameter (default: 1.0). "
             "Lower = more robust to outliers, higher = more like MSE.")
    rician_group.add_argument("--lambda-cosine", type=float, default=0.05,
        help="Weight for per-shell cosine angular loss (default: 0.05). "
             "Used with sqmag-charb-cosine mode.")
    rician_group.add_argument("--hybrid-mse-weight", type=float, default=0.7,
        help="MSE weight for hybrid mode (default: 0.7, recommended 0.7-0.9)")
    rician_group.add_argument("--b-switch", type=float, default=2000.0,
        help="B-value transition point for b-dependent mode (default: 2000). "
             "Below this: mostly MSE. Above this: mostly Rician NLL.")
    rician_group.add_argument("--b-slope", type=float, default=0.002,
        help="Slope of sigmoid transition for b-dependent mode (default: 0.002). "
             "Higher = sharper transition. 0.002 gives smooth blend over ~500 s/mm².")
    rician_group.add_argument("--noise-sigma", type=float, default=None,
        help="Fixed noise sigma for Rician loss. If not provided, will be estimated "
             "from background (air) voxels. IMPORTANT: Specify in S0-normalized units "
             "(typically 0.05-0.15 for SNR 10-50).")
    rician_group.add_argument("--auto-noise-sigma", choices=["none", "background", "b0"], default="background",
        help="Automatic σ estimation method: "
             "'none' = require --noise-sigma, "
             "'background' = estimate from air voxels using Rayleigh statistics (RECOMMENDED), "
             "'b0' = estimate from variance of repeated b0 acquisitions. "
             "Default: background")
    rician_group.add_argument("--use-sigma-map", action="store_true",
        help="Use spatially-varying σ-map instead of scalar σ. "
             "This is CRITICAL when S0 varies across the brain (always the case). "
             "σ-map: σ(x) = σ_raw / S0(x). Highly recommended for accurate Rician correction.")
    rician_group.add_argument("--sigma-bg-margin", type=int, default=3,
        help="Number of voxels to erode from background mask for safe σ estimation (default: 3). "
             "Avoids edge artifacts and partial volume.")
    rician_group.add_argument("--sigma-estimation-method", choices=["median", "mean"], default="median",
        help="Rayleigh statistic for background σ estimation: "
             "'median' = σ = median / 1.177 (robust to outliers, recommended), "
             "'mean' = σ = mean / 1.253. Default: median")
    rician_group.add_argument("--save-sigma-map", action="store_true",
        help="Save estimated σ-map as NIfTI for visualization/debugging.")
    rician_group.add_argument("--learn-noise-sigma", action="store_true",
        help="Learn noise sigma during fitting (MAP with regularization). "
             "Useful when background is not available for estimation.")
    rician_group.add_argument("--staged-directions", action="store_true",
        help="Use staged optimization: freeze fiber directions initially, then "
             "unfreeze with small LR. Helps at low SNR where direction estimates "
             "are ill-conditioned.")
    rician_group.add_argument("--stage1-freeze-steps", type=int, default=100,
        help="Number of steps to freeze fiber directions in Stage 1 (default: 100)")
    rician_group.add_argument("--direction-lr-factor", type=float, default=0.1,
        help="LR multiplier for fiber directions when unfrozen (default: 0.1)")
    
    early_group = parser.add_argument_group('Smart Early Stopping')
    early_group.add_argument("--early-patience", type=int, default=30,
        help="Base patience for early stopping (default: 30). Adapts based on step number.")
    early_group.add_argument("--early-warmup", type=int, default=50,
        help="Warmup steps before early stopping can trigger (default: 50)")
    early_group.add_argument("--early-min-steps", type=int, default=100,
        help="Minimum steps before any early stopping (default: 100)")
    early_group.add_argument("--early-rel-tol", type=float, default=1e-4,
        help="Relative improvement threshold for early stopping (default: 1e-4 = 0.01%%)")
    early_group.add_argument("--no-early-stop", action="store_true",
        help="Disable early stopping completely (run all --n-steps)")
    early_group.add_argument("--mse-warmup-steps", type=int, default=0,
        help="Two-phase optimization: use MSE for first N steps, then switch to Rician. "
             "Recommended: 100-150 for faster convergence with similar accuracy. (default: 0 = disabled)")
    
    bochner_group = parser.add_argument_group('Bochner Physics Constraint')
    bochner_group.add_argument("--use-bochner", action="store_true",
        help="Enable Bochner PSD constraint (ensures physically valid signals)")
    bochner_group.add_argument("--bochner-weight", type=float, default=0.01,
        help="Weight for Bochner loss term (default: 0.01)")
    bochner_group.add_argument("--bochner-m", type=int, default=16,
        help="Gram matrix size for Bochner constraint (default: 16, range: 8-32)")
    
    bernstein_group = parser.add_argument_group('Bernstein Complete Monotonicity')
    bernstein_group.add_argument("--use-bernstein", action="store_true",
        help="Enable Bernstein complete monotonicity constraint (valid b-curve shape)")
    bernstein_group.add_argument("--bernstein-weight", type=float, default=0.01,
        help="Weight for Bernstein loss term (default: 0.01)")
    bernstein_group.add_argument("--bernstein-orders", type=int, default=3,
        help="Number of derivative orders to check (default: 3, range: 1-4)")
    
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda",
        help="Device")
    parser.add_argument("--outdir", required=True,
        help="Output directory")
    parser.add_argument("--name", default="",
        help="Dataset name for titles")
    parser.add_argument("--save-raw", action="store_true",
        help="Save raw fitted parameters (mu, fractions, kappa, etc.) as .npy files")
    parser.add_argument("--config", type=str, default=None,
        help="Path to YAML config file (overrides command line args)")
    parser.add_argument("--seed", type=int, default=0,
        help="Random seed for reproducibility (default: 0)")

    args = parser.parse_args()
    
    args.cube_size = getattr(args, 'cube_size', 50)
    args.overlap = getattr(args, 'overlap', 3)

    os.makedirs(args.outdir, exist_ok=True)

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"  Seed: {args.seed}")

    print("\n[1] Loading data...")
    img, dwi, bvals, bvecs, mask, mask_provided = load_dwi(
        args.dwi, args.bvals, args.bvecs, args.mask,
        use_median_otsu=getattr(args, 'median_otsu', False),
        median_radius=getattr(args, 'median_otsu_median_radius', 4),
        numpass=getattr(args, 'median_otsu_numpass', 4),
        auto_mask_scale=getattr(args, 'auto_mask_scale', 0.8),
        mask_dilate_iters=getattr(args, 'mask_dilate_iters', 0)
    )
    
    args.mask_provided = mask_provided

    X, Y, Z, N = dwi.shape
    print(f"  DWI shape: {dwi.shape} (X, Y, Z, N_meas)")
    print(f"  B-values: {np.unique(bvals.round(-2))}")
    
    dwi_for_sigma = dwi
    
    if getattr(args, 'apply_mask', False) and mask is not None:
        n_brain = mask.sum()
        n_total = mask.size
        print(f"  Applying mask: {n_brain:.0f}/{n_total} voxels ({100*n_brain/n_total:.1f}% brain)")
        mask_3d = ~(mask > 0)
        dwi[mask_3d] = 0
        print(f"  DWI masked in-place: non-brain voxels set to 0")

    if args.max_bval is not None:
        keep_idx = bvals <= args.max_bval
        n_orig = len(bvals)
        dwi = dwi[:, :, :, keep_idx]
        bvals = bvals[keep_idx]
        bvecs = bvecs[keep_idx]
        print(f"  Filtered to b ≤ {args.max_bval}: {keep_idx.sum()}/{n_orig} volumes kept")
        print(f"  New B-values: {np.unique(bvals.round(-2))}")

    if getattr(args, 'sweep_mode', False):
        fit_sweep_mode(args, img, dwi, bvals, bvecs, mask, dwi_for_sigma=dwi_for_sigma)
        return

    if args.all_slices:
        if getattr(args, 'slab_mode', False):
            fit_whole_brain_slabs(args, img, dwi, bvals, bvecs, mask, dwi_for_sigma=dwi_for_sigma)
        else:
            fit_whole_brain(args, img, dwi, bvals, bvecs, mask, dwi_for_sigma=dwi_for_sigma)
        return

    if args.slice is None:
        sl = Z // 2
    else:
        sl = int(args.slice)
    sl = max(0, min(sl, Z - 1))

    slab_mode = getattr(args, 'slab_mode', False)
    slab_size = getattr(args, 'slab_size', 5)
    
    if slab_mode:
        half = slab_size // 2
        z0 = max(0, sl - half)
        z1 = min(Z, z0 + slab_size)
        z0 = max(0, z1 - slab_size)
        
        print(f"  SLAB MODE: Fitting slices {z0}-{z1} (centered on {sl})")
        signal_slab = dwi[:, :, z0:z1, :]
        mask_slab = mask[:, :, z0:z1] > 0
        
        dev = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
        scanner_config = get_scanner_config_from_args(args)
        topology_config = get_topology_config_from_args(args)
        
        print(f"\n[2] Creating scanner (n_fibers={args.n_fibers}, device={dev})...")
        print_scanner_config(scanner_config)
        scanner = create_scanner(bvals, bvecs, n_fibers=args.n_fibers, device=dev, **scanner_config)
        
        use_topology = topology_config['use_topology']
        use_rician = getattr(args, 'use_rician', False)
        
        noise_sigma = None
        if getattr(args, 'noise_sigma', None):
            noise_sigma = args.noise_sigma
        else:
            b0_idx = bvals < 100
            if b0_idx.sum() > 0 and mask is not None:
                bg = ~(mask > 0)
                bg_median = 0.0
                if bg.sum() > 100:
                    bg_signal = dwi_for_sigma[bg][:, b0_idx]
                    bg_median = np.median(bg_signal)
                
                s0_median = np.median(dwi_for_sigma[mask > 0][:, b0_idx])
                
                if bg_median > 1e-6:
                    sigma_raw = bg_median / np.sqrt(np.pi / 2)
                    noise_sigma = sigma_raw / s0_median
                    print(f"[Rician] Estimated σ from background = {noise_sigma:.4f}")
                elif b0_idx.sum() > 1:
                    b0_vols = dwi_for_sigma[mask > 0][:, b0_idx]
                    noise_per_voxel = np.std(b0_vols, axis=1)
                    sigma_raw = np.median(noise_per_voxel)
                    noise_sigma = sigma_raw / s0_median
                    print(f"[Rician] Estimated σ from b0 temporal std = {noise_sigma:.4f}")
                
                if noise_sigma is None or noise_sigma < 1e-6:
                    noise_sigma = 1.0 / 30.0
                    print(f"[Rician] Using fallback σ = {noise_sigma:.4f} (SNR ≈ 30)")
        
        print(f"\n[3] Fitting slab (n_steps={args.n_steps}, topology={use_topology})...")
        
        params = fit_patch_3d(
            scanner,
            signal_slab,
            mask_slab,
            n_steps=args.n_steps,
            use_topology=use_topology,
            use_spatial_prior=scanner_config.get('use_spatial_prior', True),
            use_6conn_topology=topology_config.get('use_6conn_topology', False),
            use_26conn_topology=topology_config['use_26conn_topology'],
            lambda_orphan=topology_config['lambda_orphan'],
            lambda_continuity=topology_config['lambda_continuity'],
            lambda_ordering=topology_config['lambda_ordering'],
            lambda_repulsion=topology_config['lambda_repulsion'],
            lambda_curvature=topology_config.get('lambda_curvature', 0.0),
            lambda_perm_invariant=topology_config.get('lambda_perm_invariant', 0.0),
            device=dev,
            use_rician=use_rician,
            rician_mode=getattr(args, 'rician_mode', 'hybrid'),
            hybrid_mse_weight=getattr(args, 'hybrid_mse_weight', 0.7),
            noise_sigma=noise_sigma,
            auto_noise_sigma='none',
            sparsity_weight=getattr(args, 'sparsity_weight', 0.02),
            entropy_weight=getattr(args, 'entropy_weight', 0.0),
            huber_delta=getattr(args, 'huber_delta', 1.0),
            lambda_cosine=getattr(args, 'lambda_cosine', 0.05),
            early_patience=getattr(args, 'early_patience', 30),
            early_warmup=getattr(args, 'early_warmup', 50),
            early_min_steps=getattr(args, 'early_min_steps', 100),
            early_rel_tol=getattr(args, 'early_rel_tol', 1e-4),
            no_early_stop=getattr(args, 'no_early_stop', False),
            mse_warmup_steps=getattr(args, 'mse_warmup_steps', 0),
        )
        
        print(f"\n[4] Saving results to {args.outdir}...")
        
        f_csf = params['f_csf'].detach().cpu().numpy()[0]
        f_gm = params['f_gm'].detach().cpu().numpy()[0]
        f_wm_fibers = params['f_wm'].detach().cpu().numpy()[0]
        fiber_dirs = params['fiber_dirs'].detach().cpu().numpy()[0]
        f_restricted = params['f_restricted'].detach().cpu().numpy()[0] if 'f_restricted' in params else np.zeros_like(f_csf)
        
        f_wm = f_wm_fibers.sum(axis=-1)
        
        slab_affine = img.affine.copy()
        slab_affine[:3, 3] += slab_affine[:3, 2] * z0
        
        nib.save(nib.Nifti1Image(f_csf, slab_affine), f"{args.outdir}/f_csf.nii.gz")
        nib.save(nib.Nifti1Image(f_gm, slab_affine), f"{args.outdir}/f_gm.nii.gz")
        nib.save(nib.Nifti1Image(f_wm, slab_affine), f"{args.outdir}/f_wm.nii.gz")
        nib.save(nib.Nifti1Image(f_restricted, slab_affine), f"{args.outdir}/f_restricted.nii.gz")
        nib.save(nib.Nifti1Image(f_wm_fibers, slab_affine), f"{args.outdir}/f_wm_fibers.nii.gz")
        nib.save(nib.Nifti1Image(fiber_dirs.reshape(*fiber_dirs.shape[:3], -1), slab_affine), 
                 f"{args.outdir}/fiber_dirs.nii.gz")
        
        print(f"  Saved slab outputs: z={z0}-{z1}, shape={f_csf.shape}")
        
        if 'pred_signal' in params and 'target_signal' in params:
            pred_sig = params['pred_signal'].detach().cpu().numpy()[0]
            targ_sig = params['target_signal'].detach().cpu().numpy()[0]
            if pred_sig.ndim == 4 and pred_sig.shape[2] == 1:
                pred_sig = pred_sig[:, :, 0, :]
                targ_sig = targ_sig[:, :, 0, :]
            elif pred_sig.ndim == 4:
                pass
            
            ss_res = np.sum((targ_sig - pred_sig)**2, axis=-1)
            ss_tot = np.sum((targ_sig - targ_sig.mean(axis=-1, keepdims=True))**2, axis=-1)
            r2_map = np.where(ss_tot > 1e-6, 1.0 - ss_res / (ss_tot + 1e-8), np.nan)
            
            sig_norm = np.linalg.norm(targ_sig, axis=-1)
            rel_res = np.linalg.norm(targ_sig - pred_sig, axis=-1) / (sig_norm + 1e-8)
            
            nib.save(nib.Nifti1Image(r2_map.astype(np.float32), slab_affine), 
                     f"{args.outdir}/r2_map.nii.gz")
            nib.save(nib.Nifti1Image(rel_res.astype(np.float32), slab_affine), 
                     f"{args.outdir}/rel_residual.nii.gz")
            
            mask_np = mask_slab if mask_slab is not None else np.ones(r2_map.shape, dtype=bool)
            for rr_thr in [0.10, 0.15, 0.20, 0.25]:
                err_mask = (rel_res < rr_thr) & mask_np
                nib.save(nib.Nifti1Image(err_mask.astype(np.uint8), slab_affine),
                         f"{args.outdir}/error_mask_rr{int(rr_thr*100):02d}.nii.gz")
            for r2_thr in [0.5, 0.7, 0.9]:
                r2_mask = (~np.isnan(r2_map)) & (r2_map >= r2_thr) & mask_np
                nib.save(nib.Nifti1Image(r2_mask.astype(np.uint8), slab_affine),
                         f"{args.outdir}/r2_mask_{int(r2_thr*100)}.nii.gz")
            
            r2_valid = r2_map[mask_np & ~np.isnan(r2_map)]
            rr_valid = rel_res[mask_np]
            print(f"  R² map: mean={np.mean(r2_valid):.4f}, median={np.median(r2_valid):.4f}")
            print(f"  Rel residual: mean={np.mean(rr_valid):.4f}, median={np.median(rr_valid):.4f}")
            print(f"  Saved: r2_map.nii.gz, rel_residual.nii.gz, error_mask_rr*.nii.gz, r2_mask_*.nii.gz")
        
        if getattr(args, 'export_artifacts', False):
            export_scanner_artifacts(
                scanner=scanner,
                outdir=args.outdir,
                affine=slab_affine,
                spatial_shape=f_csf.shape,
                mask=mask_slab,
                bvals=bvals,
                dataset_name=args.name or os.path.basename(args.dwi),
                dwi_signal=dwi[:, :, z0:z1, :] if dwi is not None else None,
            )
        
        print(f"\n✓ Slab fitting complete!")
        return

    print(f"  Using slice {sl}")
    signal_slice = dwi[:, :, sl, :]
    mask_slice = mask[:, :, sl] > 0

    H, W, _ = signal_slice.shape

    if args.patch_size > 0:
        cx = args.cx if args.cx is not None else H // 2
        cy = args.cy if args.cy is not None else W // 2
        x0, x1, y0, y1 = extract_patch_bounds(H, W, cx, cy, args.patch_size)
        signal_patch = signal_slice[x0:x1, y0:y1, :]
        mask_patch = mask_slice[x0:x1, y0:y1]
        print(f"  Patch: center=({cx},{cy}), bounds=({x0}:{x1}, {y0}:{y1}), shape={signal_patch.shape[:2]}")
    else:
        signal_patch = signal_slice
        mask_patch = mask_slice
        print(f"  Using full slice: shape={signal_patch.shape[:2]}")

    dev = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    if args.device == "cuda" and dev == "cpu":
        print("  Warning: CUDA not available, using CPU")

    scanner_config = get_scanner_config_from_args(args)
    topology_config = get_topology_config_from_args(args)
    
    print(f"\n[2] Creating scanner (n_fibers={args.n_fibers}, device={dev})...")
    print_scanner_config(scanner_config)
    scanner = create_scanner(bvals, bvecs, n_fibers=args.n_fibers, device=dev, **scanner_config)
    
    if getattr(args, 'signal_aware_priors', False):
        signal_t = torch.tensor(signal_patch, dtype=torch.float32).unsqueeze(0).unsqueeze(-2)
        setup_signal_aware_priors(
            scanner, 
            signal_t,
            weight=getattr(args, 'signal_aware_weight', 0.01),
            wm_threshold=getattr(args, 'wm_threshold', 0.2),
        )

    use_topology = topology_config['use_topology']
    
    use_bochner = getattr(args, 'use_bochner', False)
    bochner_weight = getattr(args, 'bochner_weight', 0.01)
    bochner_m = getattr(args, 'bochner_m', 16)
    
    use_bernstein = getattr(args, 'use_bernstein', False)
    bernstein_weight = getattr(args, 'bernstein_weight', 0.01)
    bernstein_orders = getattr(args, 'bernstein_orders', 3)
    
    staged_training = getattr(args, 'staged_training', False)
    stage1_steps = getattr(args, 'stage1_steps', 50)
    
    use_sigmoid_fracs = getattr(args, 'sigmoid_fractions', False)
    sigmoid_scale = getattr(args, 'sigmoid_scale', 0.5)
    
    use_rician = getattr(args, 'use_rician', False)
    noise_sigma = None
    if use_rician:
        if getattr(args, 'noise_sigma', None):
            noise_sigma = args.noise_sigma
        else:
            b0_idx = bvals < 100
            if b0_idx.sum() > 0 and mask is not None:
                bg = ~(mask > 0)
                bg_median = 0.0
                if bg.sum() > 100:
                    bg_signal = dwi_for_sigma[bg][:, b0_idx]
                    bg_median = np.median(bg_signal)
                
                s0_median = np.median(dwi_for_sigma[mask > 0][:, b0_idx])
                
                if bg_median > 1e-6:
                    sigma_raw = bg_median / np.sqrt(np.pi / 2)
                    noise_sigma = sigma_raw / s0_median
                    print(f"  [Rician] Estimated σ from background = {noise_sigma:.4f} (SNR ≈ {1/noise_sigma:.0f})")
                elif b0_idx.sum() > 1:
                    b0_vols = dwi_for_sigma[mask > 0][:, b0_idx]
                    noise_per_voxel = np.std(b0_vols, axis=1)
                    sigma_raw = np.median(noise_per_voxel)
                    noise_sigma = sigma_raw / s0_median
                    print(f"  [Rician] Estimated σ from b0 temporal std = {noise_sigma:.4f} (SNR ≈ {1/noise_sigma:.0f})")
                else:
                    high_b = bvals > 2500
                    if high_b.sum() > 5:
                        hb_signal = dwi_for_sigma[mask > 0][:, high_b]
                        mad = np.median(np.abs(hb_signal - np.median(hb_signal, axis=1, keepdims=True)))
                        sigma_raw = mad * 1.4826
                        noise_sigma = sigma_raw / s0_median
                        print(f"  [Rician] Estimated σ from high-b MAD = {noise_sigma:.4f} (SNR ≈ {1/noise_sigma:.0f})")
                
                if noise_sigma is None or noise_sigma < 1e-6:
                    noise_sigma = 1.0 / 30.0
                    print(f"  [Rician] Using fallback σ = {noise_sigma:.4f} (SNR ≈ 30)")
    
    print(f"\n[3] Fitting (n_steps={args.n_steps}, topology={use_topology}, rician={use_rician}, bochner={use_bochner}, bernstein={use_bernstein}, staged={staged_training}, sigmoid_fracs={use_sigmoid_fracs})...")
    params = fit_patch(
        scanner,
        signal_patch,
        mask_patch,
        n_steps=args.n_steps,
        use_topology=use_topology,
        device=dev,
        use_encoder=args.use_encoder,
        encoder_epochs=args.encoder_epochs,
        use_bochner=use_bochner,
        bochner_weight=bochner_weight,
        bochner_m=bochner_m,
        use_bernstein=use_bernstein,
        bernstein_weight=bernstein_weight,
        bernstein_orders=bernstein_orders,
        staged_training=staged_training,
        stage1_steps=stage1_steps,
        use_sigmoid_fracs=use_sigmoid_fracs,
        sigmoid_scale=sigmoid_scale,
        use_rician=use_rician,
        rician_mode=getattr(args, 'rician_mode', 'hybrid'),
        noise_sigma=noise_sigma,
        hybrid_mse_weight=getattr(args, 'hybrid_mse_weight', 0.7),
        huber_delta=getattr(args, 'huber_delta', 1.0),
        lambda_cosine=getattr(args, 'lambda_cosine', 0.05),
        mse_warmup_steps=getattr(args, 'mse_warmup_steps', 0),
        learn_noise_sigma=getattr(args, 'learn_noise_sigma', False),
    )

    print(f"\n[4] Saving results to {args.outdir}/")
    
    summary = {
        'dataset': args.name or os.path.basename(args.dwi),
        'slice': sl,
        'patch_size': args.patch_size if args.patch_size > 0 else 'full',
        'n_fibers': args.n_fibers,
        'n_steps': args.n_steps,
        'use_topology': use_topology,
        'use_bochner': use_bochner,
        'bochner_weight': bochner_weight if use_bochner else None,
        'bochner_m': bochner_m if use_bochner else None,
        'use_bernstein': use_bernstein,
        'bernstein_weight': bernstein_weight if use_bernstein else None,
        'bernstein_orders': bernstein_orders if use_bernstein else None,
        'scanner_config': scanner_config,
        'topology_config': topology_config,
        'final_mse': float(params['mse']),
        'n_measurements': int(N),
        'b_values': [float(b) for b in np.unique(bvals.round(-2))],
        'mask_voxels': int(mask_patch.sum()),
        'total_voxels': int(np.prod(mask_patch.shape)),
    }
    
    if 'tissue_stats' in params:
        summary.update(params['tissue_stats'])
    
    if scanner_config.get('use_exchange', False):
        exchange_times = scanner.get_exchange_times()
        summary['exchange'] = {
            'enabled': True,
            'tau_ex_ms': exchange_times.get('tau_ex', None),
            'initial_tau_ex_ms': scanner_config['exchange_tau'],
            'pairs': scanner_config['exchange_pairs'],
            'learnable': scanner_config['exchange_learnable'],
        }
        if exchange_times.get('tau_ex') is not None:
            print(f"  Exchange τ_ex: {exchange_times['tau_ex']:.2f} ms")
    
    summary_path = os.path.join(args.outdir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")
    
    if getattr(args, 'export_artifacts', False):
        H_patch, W_patch = signal_patch.shape[:2]
        export_scanner_artifacts(
            scanner=scanner,
            outdir=args.outdir,
            affine=img.affine,
            spatial_shape=(H_patch, W_patch, 1),
            mask=mask_patch[:, :, np.newaxis] if mask_patch.ndim == 2 else mask_patch,
            bvals=bvals,
            dataset_name=args.name or os.path.basename(args.dwi),
            dwi_signal=signal_patch,
        )
    
    if args.save_raw:
        print("  Saving raw parameters...")
        
        def to_numpy(x):
            if hasattr(x, 'cpu'):
                return x.cpu().detach().numpy()
            return np.asarray(x)
        
        if 'fiber_dirs' in params:
            arr = to_numpy(params['fiber_dirs'])
            np.save(os.path.join(args.outdir, "fiber_dirs.npy"), arr)
            print(f"    fiber_dirs.npy: {arr.shape}")
        if 'f_intra' in params:
            arr = to_numpy(params['f_intra'])
            np.save(os.path.join(args.outdir, "f_intra.npy"), arr)
            print(f"    f_intra.npy: {arr.shape}")
        if 'kappa' in params:
            arr = to_numpy(params['kappa'])
            np.save(os.path.join(args.outdir, "kappa.npy"), arr)
            print(f"    kappa.npy: {arr.shape}")
        if 'f_csf' in params:
            arr = to_numpy(params['f_csf'])
            np.save(os.path.join(args.outdir, "f_csf.npy"), arr)
            print(f"    f_csf.npy: {arr.shape}")
        if 'f_gm' in params:
            arr = to_numpy(params['f_gm'])
            np.save(os.path.join(args.outdir, "f_gm.npy"), arr)
            print(f"    f_gm.npy: {arr.shape}")
        if 'f_wm' in params:
            arr = to_numpy(params['f_wm'])
            np.save(os.path.join(args.outdir, "f_wm.npy"), arr)
            print(f"    f_wm.npy: {arr.shape}")
        if 'f_restricted' in params:
            arr = to_numpy(params['f_restricted'])
            np.save(os.path.join(args.outdir, "f_restricted.npy"), arr)
            print(f"    f_restricted.npy: {arr.shape}")
        if 's0' in params:
            arr = to_numpy(params['s0'])
            np.save(os.path.join(args.outdir, "s0.npy"), arr)
            print(f"    s0.npy: {arr.shape}")
        if 'pred_signal' in params:
            arr = to_numpy(params['pred_signal'])
            np.save(os.path.join(args.outdir, "pred_signal.npy"), arr)
            print(f"    pred_signal.npy: {arr.shape}")
        if 'target_signal' in params:
            arr = to_numpy(params['target_signal'])
            np.save(os.path.join(args.outdir, "target_signal.npy"), arr)
            print(f"    target_signal.npy: {arr.shape}")
        np.save(os.path.join(args.outdir, "mask.npy"), mask_patch)
        print(f"    mask.npy: {mask_patch.shape}")
    
    
    mask_provided = getattr(args, 'mask_provided', False)
    output_r2_thresh = getattr(args, 'output_r2_mask', None)
    output_rel_res_thresh = getattr(args, 'output_rel_res_mask', None)
    
    use_output_r2 = output_r2_thresh is not None
    use_output_rel_res = output_rel_res_thresh is not None
    
    apply_viz_mask = mask_provided
    
    final_viz_mask = mask_patch.copy()
    
    print(f"\n  Visualization Masking:")
    print(f"    Input mask provided: {mask_provided}")
    print(f"    Output R² mask: {output_r2_thresh}")
    print(f"    Output rel_res mask: {output_rel_res_thresh}")
    
    viz_path = os.path.join(args.outdir, "microstructure_panel.png")
    
    create_simple_visualization(params, mask_patch, viz_path, args.name, 
                                data_mask=mask_patch, 
                                use_relative_residual=use_output_rel_res,
                                rel_res_threshold=output_rel_res_thresh,
                                r2_threshold=output_r2_thresh,
                                mask_provided=mask_provided,
                                apply_viz_mask=mask_provided)
    
    try:
        print("\n[5] Creating detailed microstructure maps...")
        maps = MicrostructureMaps.from_fitted_params(
            params,
            signal_pred=params['pred_signal'],
            signal_obs=params['target_signal'],
            wm_mask=mask_patch,
        )
        
        fig2_path = os.path.join(args.outdir, "figure2_panel.png")
        
        if use_output_rel_res:
            create_figure2_panel(maps, slice_idx=0, output_path=fig2_path, 
                                 use_r2_mask=False, 
                                 use_relative_residual_mask=True,
                                 rel_residual_threshold=output_rel_res_thresh,
                                 brain_mask=mask_patch.squeeze() if mask_provided else None)
        elif use_output_r2:
            create_figure2_panel(maps, slice_idx=0, output_path=fig2_path, 
                                 use_r2_mask=True, 
                                 r2_threshold=output_r2_thresh,
                                 use_relative_residual_mask=False,
                                 brain_mask=mask_patch.squeeze() if mask_provided else None)
        else:
            create_figure2_panel(maps, slice_idx=0, output_path=fig2_path, 
                                 use_r2_mask=False, 
                                 use_relative_residual_mask=False,
                                 brain_mask=mask_patch.squeeze() if mask_provided else None)
        print(f"  Saved: {fig2_path}")
        
        r2 = maps.get_r_squared()
        if r2 is not None:
            np.save(os.path.join(args.outdir, "r2_map.npy"), r2)
            print(f"  Saved: r2_map.npy")
            
            for thresh in [0.5, 0.7, 0.9]:
                r2_mask = maps.get_r2_brain_mask(threshold=thresh)
                fname = f"r2_brain_mask_{int(thresh*100)}.npy"
                np.save(os.path.join(args.outdir, fname), r2_mask)
                
                r2_mask_clean = maps.get_r2_brain_mask(
                    threshold=thresh, 
                    largest_component=True, 
                    fill_holes=True
                )
                fname_clean = f"r2_brain_mask_{int(thresh*100)}_clean.npy"
                np.save(os.path.join(args.outdir, fname_clean), r2_mask_clean)
                
            print(f"  Saved: r2_brain_mask_50/70/90.npy (R²-based brain masks)")
            print(f"  Saved: r2_brain_mask_50/70/90_clean.npy (largest component + filled)")
            
            create_r2_mask_comparison(
                r2_map=r2.squeeze(),
                provided_mask=mask_patch.squeeze(),
                output_path=os.path.join(args.outdir, "r2_mask_comparison.png"),
            )
            print(f"  Saved: r2_mask_comparison.png")
        
        maps.print_summary(f"Slice {sl}")
        
    except Exception as e:
        print(f"  Note: Could not create detailed maps: {e}")
        print("  (This is OK - basic visualization was saved)")

    print("\n✓ Done!")
    print(f"  Output directory: {args.outdir}")


if __name__ == "__main__":
    main()
