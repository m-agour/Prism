#!/usr/bin/env python3

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DWI_BVALS = [0, 1000, 2000, 3000]


def get_environment_info() -> dict:
    import sys as _sys

    info = {
        "python": _sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }

    try:
        import dipy

        info["dipy"] = dipy.__version__
    except Exception:
        info["dipy"] = None

    try:
        import scipy

        info["scipy"] = scipy.__version__
    except Exception:
        info["scipy"] = None

    if torch.cuda.is_available():
        try:
            info["cuda_available"] = True
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = torch.backends.cudnn.version()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_capability"] = list(torch.cuda.get_device_capability(0))
        except Exception:
            pass
    else:
        info["cuda_available"] = False

    return info


def _torch_synchronize_if_needed(device: str | torch.device) -> None:
    if isinstance(device, torch.device):
        device = device.type
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed_call(fn, *args, sync_device: str = "cpu", warmup: int = 0, repeats: int = 1, **kwargs):
    repeats = int(max(1, repeats))
    warmup = int(max(0, warmup))

    for _ in range(warmup):
        _ = fn(*args, **kwargs)

    times = []
    last = None
    for _ in range(repeats):
        _torch_synchronize_if_needed(sync_device)
        t0 = time.time()
        last = fn(*args, **kwargs)
        _torch_synchronize_if_needed(sync_device)
        times.append(time.time() - t0)

    times_arr = np.asarray(times, dtype=np.float64)
    timing = {
        "repeats": repeats,
        "warmup": warmup,
        "mean_s": float(times_arr.mean()),
        "median_s": float(np.median(times_arr)),
        "std_s": float(times_arr.std(ddof=0)),
        "all_s": [float(x) for x in times_arr],
    }
    return last, timing


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in fieldnames})


def _format_advantage(metric_name: str, prism_value, competitor_value, higher_is_better: bool) -> str:
    if prism_value is None or competitor_value is None:
        return "N/A"
    if higher_is_better:
        return f"{(prism_value - competitor_value):.1f} points"
    prism_value = float(prism_value)
    competitor_value = float(competitor_value)
    if prism_value <= 0 or competitor_value <= 0:
        return "N/A"
    if prism_value < competitor_value:
        ratio = competitor_value / prism_value
        pct = (1.0 - prism_value / competitor_value) * 100.0
        return f"{ratio:.2f}× better ({pct:.0f}%)"
    if prism_value > competitor_value:
        ratio = prism_value / competitor_value
        pct = (1.0 - competitor_value / prism_value) * 100.0
        return f"{ratio:.2f}× worse ({pct:.0f}%)"
    return "tie"


def _pick_best_competitor(
    values_by_method: dict[str, float | None],
    exclude: str,
    higher_is_better: bool,
) -> tuple[str | None, float | None]:
    items = [(m, v) for m, v in values_by_method.items() if m != exclude and v is not None]
    if not items:
        return None, None
    key = (lambda x: x[1]) if higher_is_better else (lambda x: -x[1])
    best = max(items, key=key)
    return best[0], float(best[1])


def set_global_seed(seed: int) -> None:
    import os
    import random as _random
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    z = rng.uniform(-1.0, 1.0)
    phi = rng.uniform(0.0, 2 * np.pi)
    r_xy = np.sqrt(max(0.0, 1.0 - z * z))
    return np.array([r_xy * np.cos(phi), r_xy * np.sin(phi), z], dtype=np.float64)


def unit_vector_to_spherical_angles(v: np.ndarray) -> tuple[float, float]:
    v = np.asarray(v, dtype=np.float64)
    v = v / (np.linalg.norm(v) + 1e-12)
    theta = np.arccos(np.clip(v[2], -1.0, 1.0))
    phi = np.arctan2(v[1], v[0])
    if phi < 0:
        phi += 2 * np.pi
    return float(np.degrees(theta)), float(np.degrees(phi))


def make_crossing_pair(
    rng: np.random.Generator, angle_deg: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    v1 = random_unit_vector(rng)

    for _ in range(100):
        w = random_unit_vector(rng)
        u = w - np.dot(w, v1) * v1
        u_norm = np.linalg.norm(u)
        if u_norm > 1e-8:
            u = u / u_norm
            break
    else:
        u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        u = u - np.dot(u, v1) * v1
        u = u / (np.linalg.norm(u) + 1e-12)

    a = np.deg2rad(angle_deg)
    v2 = np.cos(a) * v1 + np.sin(a) * u
    return unit_vector_to_spherical_angles(v1), unit_vector_to_spherical_angles(v2)


def subset_measurements_by_bvals(
    signals: np.ndarray,
    gtab,
    bvals_keep: list[float],
    atol: float = 50.0,
):
    from dipy.core.gradients import gradient_table

    bvals = np.asarray(gtab.bvals)
    sel = np.zeros_like(bvals, dtype=bool)
    for b in bvals_keep:
        sel |= np.isclose(bvals, b, atol=atol)
    if not np.any(sel):
        raise ValueError(f"No measurements selected for bvals_keep={bvals_keep}")

    sub_signals = signals[:, sel]
    sub_gtab = gradient_table(bvals[sel], bvecs=gtab.bvecs[sel])
    return sub_signals, sub_gtab, sel


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def get_dipy_sphere(name: str):
    from dipy.data import default_sphere, get_sphere

    if name in (None, "", "default"):
        return default_sphere
    try:
        return get_sphere(name)
    except Exception as e:
        raise ValueError(f"Unknown DIPY sphere '{name}'. Try 'default' or e.g. 'repulsion724'.") from e


def angular_error(dir1, dir2):
    dir1 = np.asarray(dir1)
    dir2 = np.asarray(dir2)
    dir1 = dir1 / (np.linalg.norm(dir1) + 1e-8)
    dir2 = dir2 / (np.linalg.norm(dir2) + 1e-8)
    dot = np.abs(np.dot(dir1, dir2))
    dot = np.clip(dot, 0, 1)
    return np.arccos(dot) * 180 / np.pi


def best_match_angular_error(pred_dirs, gt_dirs, weight_threshold=0.01):
    errors = []
    for gt_dir in gt_dirs:
        if np.linalg.norm(gt_dir) < 0.1:
            continue
        best_err = 90.0
        for pred_dir in pred_dirs:
            if np.linalg.norm(pred_dir) < 0.1:
                continue
            err = angular_error(pred_dir, gt_dir)
            if err < best_err:
                best_err = err
        errors.append(best_err)
    return np.mean(errors) if errors else 90.0


def generate_multitensor_phantom(
    n_voxels_per_config: int = 100,
    crossing_angles: list = [30, 45, 60, 75, 90],
    snr: float = 30.0,
    b_values: list = [0, 1000, 2000, 3000],
    n_directions: int = 64,
    include_single_fiber: bool = True,
    fraction_mode: str = "both",
    seed: int = 42,
):
    from dipy.core.gradients import gradient_table
    from dipy.core.sphere import HemiSphere, disperse_charges
    from dipy.sims.voxel import multi_tensor
    
    rng = np.random.default_rng(seed)
    
    print(f"Creating gradient table: b-values={b_values}, n_dirs={n_directions}")
    
    theta = np.pi * rng.random(size=n_directions)
    phi = 2 * np.pi * rng.random(size=n_directions)
    hsph_initial = HemiSphere(theta=theta, phi=phi)
    hsph_updated, _ = disperse_charges(hsph_initial, 5000)
    vertices = hsph_updated.vertices
    
    bvecs_list = []
    bvals_list = []
    
    for b in b_values:
        if b == 0:
            bvecs_list.append(np.array([[0, 0, 0]]))
            bvals_list.append(np.array([0]))
        else:
            bvecs_list.append(vertices)
            bvals_list.append(np.ones(len(vertices)) * b)
    
    bvecs = np.vstack(bvecs_list)
    bvals = np.hstack(bvals_list)
    gtab = gradient_table(bvals, bvecs=bvecs)
    
    print(f"Total measurements: {len(bvals)}")
    
    mevals = np.array([
        [0.0017, 0.0003, 0.0003],
        [0.0017, 0.0003, 0.0003],
    ])
    
    signals = []
    gt_dirs = []
    gt_fractions = []
    config_labels = []
    
    if include_single_fiber:
        print(f"Generating {n_voxels_per_config} single-fiber voxels...")
        for i in range(n_voxels_per_config):
            v = random_unit_vector(rng)
            angles = [unit_vector_to_spherical_angles(v)]
            fractions = [100]
            
            signal, sticks = multi_tensor(
                gtab, mevals[:1], S0=1.0, angles=angles, 
                fractions=fractions, snr=snr, rng=rng
            )
            
            signals.append(signal)
            gt_dirs.append([sticks[0], np.zeros(3)])
            gt_fractions.append([1.0, 0.0])
            config_labels.append('single')
    
    for cross_angle in crossing_angles:
        print(f"Generating {n_voxels_per_config} voxels with {cross_angle}° crossing...")
        
        for i in range(n_voxels_per_config):
            angles = list(make_crossing_pair(rng, cross_angle))
            
            if fraction_mode == "equal":
                f1, f2 = 0.5, 0.5
            elif fraction_mode == "random":
                f1 = rng.uniform(0.3, 0.7)
                f2 = 1.0 - f1
            else:
                if i % 2 == 0:
                    f1, f2 = 0.5, 0.5
                else:
                    f1 = rng.uniform(0.3, 0.7)
                    f2 = 1.0 - f1
            fractions = [f1 * 100, f2 * 100]
            
            signal, sticks = multi_tensor(
                gtab, mevals, S0=1.0, angles=angles,
                fractions=fractions, snr=snr, rng=rng
            )
            
            signals.append(signal)
            gt_dirs.append([sticks[0], sticks[1]])
            gt_fractions.append([f1, f2])
            config_labels.append(f'cross_{cross_angle}')
    
    signals = np.array(signals)
    gt_dirs = np.array(gt_dirs)
    gt_fractions = np.array(gt_fractions)
    config_labels = np.array(config_labels)
    
    print(f"Generated {len(signals)} voxels total")
    
    return signals, gt_dirs, gt_fractions, gtab, config_labels


def fit_prism(
    signals,
    gtab,
    n_fibers=2,
    n_iters=200,
    device='cuda',
    seed: int = 42,
    repulsion_weight: float = 0.005,
    sparsity_weight: float = 0.0,
    merge_angle_deg: float = 0.0,
    loss_type: str = 'mse',
    snr: float = None,
    mse_warmup_frac: float = 0.3,
):
    from prism.differentiable_scanner_v4 import DifferentiableScannerV4, AcquisitionProtocolV4
    if loss_type in ('nll', 'nll_auto'):
        from prism.differentiable_scanner_v4 import rician_nll
    if loss_type == 'nll':
        if snr is None:
            raise ValueError("snr must be provided when loss_type='nll'")
        sigma_val = torch.tensor(1.0 / snr, dtype=torch.float32)
        print(f"  [NLL mode] sigma = 1/SNR = {1.0/snr:.4f}")
    elif loss_type == 'nll_auto':
        mse_warmup_steps = int(n_iters * mse_warmup_frac)
        print(f"  [NLL-auto mode] MSE warmup for {mse_warmup_steps}/{n_iters} iters, then NLL with learned σ")
    
    N_voxels, N_meas = signals.shape
    
    batch_size = min(1000, N_voxels)
    
    bvals_t = torch.from_numpy(gtab.bvals.astype(np.float32)).to(device)
    bvecs_t = torch.from_numpy(gtab.bvecs.astype(np.float32)).to(device)
    protocol = AcquisitionProtocolV4.from_tensors(bvals=bvals_t, bvecs=bvecs_t, te=89.0)
    
    all_dirs = []
    all_fracs = []
    all_mse = []
    
    def compute_repulsion_loss(dirs, weights):
        n_fib = dirs.shape[-2]
        if n_fib < 2:
            return torch.tensor(0.0, device=dirs.device)
        
        loss = torch.tensor(0.0, device=dirs.device)
        for i in range(n_fib):
            for j in range(i + 1, n_fib):
                cos_sim = torch.abs((dirs[..., i, :] * dirs[..., j, :]).sum(dim=-1))
                pair_weight = weights[..., i] * weights[..., j]
                penalty = F.relu(cos_sim - 0.8) * pair_weight
                loss = loss + penalty.mean()
        return loss
    
    def compute_sparsity_loss(wm_fracs):
        total = wm_fracs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        probs = wm_fracs / total
        
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)
        return entropy.mean()
    
    for start in range(0, N_voxels, batch_size):
        end = min(start + batch_size, N_voxels)
        batch_signals = signals[start:end]
        B = end - start
        
        signal_t = torch.from_numpy(batch_signals).float().to(device)
        signal_t = signal_t.view(B, 1, 1, 1, N_meas)
        
        scanner = DifferentiableScannerV4(protocol=protocol, n_fibers=n_fibers).to(device)

        torch.manual_seed(seed + start)
        if device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + start)
        
        raw_params = {
            'raw_csf': (torch.randn(B, 1, 1, 1, device=device) * 0.5 - 1.0).requires_grad_(True),
            'raw_gm': (torch.randn(B, 1, 1, 1, device=device) * 0.5 - 1.0).requires_grad_(True),
            'raw_wm': (torch.randn(B, 1, 1, 1, n_fibers, device=device) * 0.5 + 0.5).requires_grad_(True),
            'raw_restricted': (torch.randn(B, 1, 1, 1, device=device) * 0.5 - 1.0).requires_grad_(True),
            'fiber_dirs': F.normalize(torch.randn(B, 1, 1, 1, n_fibers, 3, device=device), dim=-1).requires_grad_(True),
            'f_intra': (torch.rand(B, 1, 1, 1, device=device) * 0.3 + 0.5).requires_grad_(True),
            'kappa': (torch.ones(B, 1, 1, 1, device=device) * 8.0).requires_grad_(True),
            's0': (torch.ones(B, 1, 1, 1, device=device)).requires_grad_(True),
        }
        
        if loss_type == 'nll_auto':
            log_sigma = torch.tensor(-3.0, device=device, requires_grad=True)
        
        def make_params(raw):
            raw_stack = torch.stack([
                raw['raw_csf'], raw['raw_gm'], 
                raw['raw_wm'].sum(dim=-1), raw['raw_restricted']
            ], dim=-1)
            fractions = F.softmax(raw_stack, dim=-1)
            wm_per_fiber = F.softmax(raw['raw_wm'], dim=-1)
            return {
                'f_csf': fractions[..., 0],
                'f_gm': fractions[..., 1],
                'f_wm': wm_per_fiber * fractions[..., 2:3],
                'f_restricted': fractions[..., 3],
                'fiber_dirs': F.normalize(raw['fiber_dirs'], dim=-1),
                'f_intra': torch.sigmoid(raw['f_intra']),
                'kappa': F.softplus(raw['kappa']),
                's0': F.softplus(raw['s0']),
            }
        
        optim_params = list(raw_params.values())
        if loss_type == 'nll_auto':
            optim_params.append(log_sigma)
        optimizer = torch.optim.Rprop(optim_params, lr=0.01)
        
        for it in range(n_iters):
            optimizer.zero_grad()
            params = make_params(raw_params)
            pred = scanner(params)
            
            if loss_type == 'nll':
                nll = rician_nll(signal_t, pred, sigma_val.to(pred.device))
                data_loss = nll.mean()
            elif loss_type == 'nll_auto':
                if it < mse_warmup_steps:
                    data_loss = F.mse_loss(pred, signal_t)
                else:
                    sigma_learned = torch.exp(log_sigma).clamp(min=1e-4, max=1.0)
                    nll = rician_nll(signal_t, pred, sigma_learned)
                    data_loss = nll.mean()
            else:
                data_loss = F.mse_loss(pred, signal_t)
            
            total_loss = data_loss
            if repulsion_weight > 0:
                rep_loss = compute_repulsion_loss(params['fiber_dirs'], params['f_wm'])
                total_loss = total_loss + repulsion_weight * rep_loss
            
            if sparsity_weight > 0:
                sparse_loss = compute_sparsity_loss(params['f_wm'])
                total_loss = total_loss + sparsity_weight * sparse_loss
            
            total_loss.backward()
            optimizer.step()
        
        if loss_type == 'nll_auto' and start == 0:
            learned_sigma = torch.exp(log_sigma).item()
            print(f"  [NLL-auto] Learned σ = {learned_sigma:.4f} (≈ 1/SNR={1.0/learned_sigma:.0f})")
        
        with torch.no_grad():
            final_params = make_params(raw_params)
            dirs = final_params['fiber_dirs'].cpu().numpy()[:, 0, 0, 0, :, :]
            fracs = final_params['f_wm'].cpu().numpy()[:, 0, 0, 0, :]
            pred = scanner(final_params)
            mse = F.mse_loss(pred, signal_t, reduction='none').mean(dim=-1).cpu().numpy()[:, 0, 0, 0]
        
        all_dirs.append(dirs)
        all_fracs.append(fracs)
        all_mse.append(mse)
    
    dirs_out = np.vstack(all_dirs)
    fracs_out = np.vstack(all_fracs)
    mse_out = np.hstack(all_mse)
    
    if merge_angle_deg > 0 and n_fibers >= 2:
        dirs_out, fracs_out = _merge_close_peaks(dirs_out, fracs_out, merge_angle_deg)
    
    return dirs_out, fracs_out, mse_out


def _merge_close_peaks(dirs, fracs, merge_angle_deg):
    N, n_fibers, _ = dirs.shape
    fracs_out = fracs.copy()
    
    merge_cos = np.cos(np.radians(merge_angle_deg))
    
    for i in range(N):
        order = np.argsort(-fracs[i])
        
        for j in range(1, n_fibers):
            fib_j = order[j]
            fib_0 = order[0]
            
            d1 = dirs[i, fib_0]
            d2 = dirs[i, fib_j]
            n1 = np.linalg.norm(d1)
            n2 = np.linalg.norm(d2)
            if n1 < 1e-8 or n2 < 1e-8:
                continue
            
            cos_sep = abs(np.dot(d1 / n1, d2 / n2))
            
            if cos_sep >= merge_cos:
                fracs_out[i, fib_j] = 0.0
    
    return dirs, fracs_out


def fit_dipy_dti(signals, gtab, gtab_full=None, signals_full=None):
    from dipy.reconst.dti import TensorModel
    
    data = signals[:, np.newaxis, np.newaxis, :]
    
    model = TensorModel(gtab)
    fit = model.fit(data)
    
    dirs = fit.evecs[..., 0]
    dirs = dirs[:, 0, 0, :]
    
    pred_fit = fit.predict(gtab)[:, 0, 0, :]
    mse_fit = np.mean((signals - pred_fit) ** 2, axis=1)

    mse_full = None
    if gtab_full is not None and signals_full is not None:
        try:
            pred_full = fit.predict(gtab_full)[:, 0, 0, :]
            mse_full = np.mean((signals_full - pred_full) ** 2, axis=1)
        except Exception:
            mse_full = None
    
    return dirs[:, np.newaxis, :], np.ones((len(signals), 1)), mse_fit, mse_full


def fit_dipy_msmtcsd(
    signals,
    gtab,
    npeaks: int = 2,
    response_mode: str = "oracle",
    sh_order_max: int = 8,
    relative_peak_threshold: float = 0.15,
    min_separation_angle: float = 10,
    sphere=None,
    parallel: bool = True,
    num_processes: int | None = None,
    response_mask=None,
    gtab_full=None,
    signals_full=None,
):
    from dipy.direction import peaks_from_model
    from dipy.reconst.mcsd import (
        MultiShellDeconvModel,
        multi_shell_fiber_response,
        auto_response_msmt,
        response_from_mask_msmt,
    )
    from dipy.core.gradients import unique_bvals_tolerance
    from dipy.reconst.csdeconv import auto_response_ssst
    
    data = signals[:, np.newaxis, np.newaxis, :]
    mask = np.ones((len(signals), 1, 1), dtype=bool)
    if sphere is None:
        sphere = get_dipy_sphere("default")

    bvals_unique = unique_bvals_tolerance(gtab.bvals, tol=20)
    n_shells = int(np.sum(bvals_unique > 0))

    response = None
    response_error = None

    response_mode = (response_mode or "oracle").lower().strip()

    if response_mode == "oracle":
        ubvals = unique_bvals_tolerance(gtab.bvals, tol=20)
        ubvals = np.asarray([b for b in ubvals if b > 50], dtype=np.float64)
        if ubvals.size == 0:
            raise ValueError("MSMT-CSD oracle response requires at least one non-zero shell.")

        wm_evals = (1.7e-3, 0.3e-3, 0.3e-3)
        gm_iso = 0.9e-3
        csf_iso = 3.0e-3
        s0 = 1.0

        wm_rf = np.zeros((len(ubvals), 4), dtype=np.float64)
        wm_rf[:, 0] = wm_evals[0]
        wm_rf[:, 1] = wm_evals[1]
        wm_rf[:, 2] = wm_evals[2]
        wm_rf[:, 3] = s0

        gm_rf = np.zeros((len(ubvals), 4), dtype=np.float64)
        gm_rf[:, 0] = gm_iso
        gm_rf[:, 1] = gm_iso
        gm_rf[:, 2] = gm_iso
        gm_rf[:, 3] = s0

        csf_rf = np.zeros((len(ubvals), 4), dtype=np.float64)
        csf_rf[:, 0] = csf_iso
        csf_rf[:, 1] = csf_iso
        csf_rf[:, 2] = csf_iso
        csf_rf[:, 3] = s0

        response = multi_shell_fiber_response(
            sh_order_max=sh_order_max,
            bvals=ubvals,
            wm_rf=wm_rf,
            gm_rf=gm_rf,
            csf_rf=csf_rf,
        )

    elif response_mode == "mask":
        if response_mask is None:
            raise ValueError("response_mode='mask' requires response_mask.")
        try:
            if isinstance(response_mask, dict):
                mask_wm = np.asarray(response_mask["wm"], dtype=bool)
                mask_gm = np.asarray(response_mask["gm"], dtype=bool)
                mask_csf = np.asarray(response_mask["csf"], dtype=bool)
            elif isinstance(response_mask, (tuple, list)) and len(response_mask) == 3:
                mask_wm = np.asarray(response_mask[0], dtype=bool)
                mask_gm = np.asarray(response_mask[1], dtype=bool)
                mask_csf = np.asarray(response_mask[2], dtype=bool)
            else:
                raise TypeError("response_mask must be dict or (wm, gm, csf) tuple/list")

            mask_wm = mask_wm.reshape(len(signals), 1, 1)
            mask_gm = mask_gm.reshape(len(signals), 1, 1)
            mask_csf = mask_csf.reshape(len(signals), 1, 1)

            resp_wm, resp_gm, resp_csf = response_from_mask_msmt(
                gtab, data, mask_wm, mask_gm, mask_csf, tol=20
            )
            response = np.stack([resp_wm, resp_gm, resp_csf], axis=0)
        except Exception as e:
            response_error = e
            response = None

    elif response_mode == "auto":
        try:
            resp_wm, resp_gm, resp_csf = auto_response_msmt(
                gtab,
                data,
                tol=20,
                roi_radii=10,
                wm_fa_thr=0.7,
                gm_fa_thr=0.3,
                csf_fa_thr=0.15,
                gm_md_thr=0.001,
                csf_md_thr=0.0032,
            )
            response = np.stack([resp_wm, resp_gm, resp_csf], axis=0)
        except Exception as e:
            response_error = e
            response = None

    elif response_mode == "ssst_fallback":
        response = None
    else:
        raise ValueError(f"Unknown MSMT-CSD response_mode='{response_mode}' (use oracle/auto/mask/ssst_fallback).")

    if response is None:
        try:
            b_nonzero = float(np.max(bvals_unique)) if len(bvals_unique) else float(np.max(gtab.bvals))
            sub_signals, sub_gtab, _ = subset_measurements_by_bvals(signals, gtab, [0, b_nonzero])
            sub_data = sub_signals[:, np.newaxis, np.newaxis, :]
            (wm_evals, wm_s0), _ = auto_response_ssst(sub_gtab, sub_data, roi_radii=10, fa_thr=0.5)
        except Exception:
            wm_evals = np.array([1.5e-3, 0.5e-3, 0.5e-3], dtype=np.float64)
            wm_s0 = 1.0

        wm_row = np.array([wm_evals[0], wm_evals[1], wm_evals[2], float(wm_s0)], dtype=np.float64)
        gm_iso = 0.9e-3
        csf_iso = 3.0e-3
        gm_row = np.array([gm_iso, gm_iso, gm_iso, float(wm_s0)], dtype=np.float64)
        csf_row = np.array([csf_iso, csf_iso, csf_iso, float(wm_s0)], dtype=np.float64)

        wm_rf = np.tile(wm_row, (max(1, n_shells), 1))
        gm_rf = np.tile(gm_row, (max(1, n_shells), 1))
        csf_rf = np.tile(csf_row, (max(1, n_shells), 1))
        response = np.stack([wm_rf, gm_rf, csf_rf], axis=0)

    try:
        model = MultiShellDeconvModel(gtab, response, sh_order_max=sh_order_max)
    except Exception as e:
        msg = f"  MSMT-CSD model construction failed: {e}"
        if response_error is not None:
            msg += f" (response estimation error: {response_error})"
        print(msg)
        raise
    
    peaks = peaks_from_model(
        model, data, sphere,
        relative_peak_threshold=relative_peak_threshold,
        min_separation_angle=min_separation_angle,
        mask=mask,
        npeaks=npeaks,
        parallel=parallel,
        num_processes=num_processes,
    )
    
    dirs = peaks.peak_dirs[:, 0, 0, :, :]
    values = peaks.peak_values[:, 0, 0, :]
    
    mse_fit = None
    mse_full = None
    
    return dirs, values, mse_fit, mse_full


def fit_dipy_forecast(
    signals,
    gtab,
    npeaks: int = 2,
    sh_order_max: int = 8,
    relative_peak_threshold: float = 0.15,
    min_separation_angle: float = 10,
    sphere=None,
    parallel: bool = True,
    num_processes: int | None = None,
    gtab_full=None,
    signals_full=None,
):
    from dipy.direction import peaks_from_model
    from dipy.reconst.forecast import ForecastModel
    
    data = signals[:, np.newaxis, np.newaxis, :]
    mask = np.ones((len(signals), 1, 1), dtype=bool)
    if sphere is None:
        sphere = get_dipy_sphere("default")
    
    model = ForecastModel(gtab, sh_order_max=sh_order_max, dec_alg='CSD')
    
    peaks = peaks_from_model(
        model, data, sphere,
        relative_peak_threshold=relative_peak_threshold,
        min_separation_angle=min_separation_angle,
        mask=mask,
        npeaks=npeaks,
        parallel=parallel,
        num_processes=num_processes,
    )
    
    dirs = peaks.peak_dirs[:, 0, 0, :, :]
    values = peaks.peak_values[:, 0, 0, :]
    
    mse_fit = None
    mse_full = None
    try:
        fit = model.fit(data, mask=mask)
        pred_fit = fit.predict(gtab)[:, 0, 0, :]
        mse_fit = np.mean((signals - pred_fit) ** 2, axis=1)
        if gtab_full is not None and signals_full is not None:
            try:
                pred_full = fit.predict(gtab_full)[:, 0, 0, :]
                mse_full = np.mean((signals_full - pred_full) ** 2, axis=1)
            except Exception:
                mse_full = None
    except Exception:
        mse_fit = None
        mse_full = None
    
    return dirs, values, mse_fit, mse_full


def fit_dipy_odffp(
    signals,
    gtab,
    npeaks: int = 2,
    dict_size: int = 1000000,
    gtab_full=None,
    signals_full=None,
):
    import sys
    import os
    _local_dir = os.path.dirname(__file__)
    if _local_dir not in sys.path:
        sys.path.insert(0, _local_dir)
    from odffp_local import OdffpModel, OdffpDictionary, dsiSphere8Fold

    N = signals.shape[0]

    tess = dsiSphere8Fold()
    odf_dict = OdffpDictionary(gtab, tessellation=tess)
    odf_dict.generate(
        dict_size=dict_size, max_peaks_num=3,
        D_a=[1.2, 2.2], D_e=[0.1, 0.6], D_r=[0.1, 0.6],
    )

    model = OdffpModel(gtab, odf_dict)

    data = signals.reshape(N, 1, 1, -1)
    odffp_fit = model.fit(data)

    raw_dirs = odffp_fit.peak_dirs()
    raw_odf = odffp_fit.odf()

    if raw_dirs.ndim == 5:
        dirs = raw_dirs[:, 0, 0, :npeaks, :]
    else:
        dirs = raw_dirs[:, :npeaks, :]

    values = np.zeros((N, npeaks))
    for i in range(N):
        for p in range(npeaks):
            d = dirs[i, p]
            if np.linalg.norm(d) < 0.1:
                continue
            d_unit = d / (np.linalg.norm(d) + 1e-12)
            cos_sim = np.abs(tess.vertices @ d_unit)
            best_idx = np.argmax(cos_sim)
            if raw_odf.ndim >= 4:
                odf_val = raw_odf[i, 0, 0, best_idx % raw_odf.shape[-1]]
            else:
                odf_val = raw_odf[i, 0, 0]
            values[i, p] = max(0, odf_val)

    max_val = values.max()
    if max_val > 0:
        values /= max_val

    mse_fit = None
    mse_full = None
    try:
        pred_fit = odffp_fit.predict()
        if pred_fit.ndim == 4:
            pred_fit = pred_fit[:, 0, 0, :]
        mse_fit = np.mean((signals - pred_fit) ** 2, axis=1)
    except Exception:
        mse_fit = None

    return dirs, values, mse_fit, mse_full


def fit_dipy_force(
    signals,
    gtab,
    npeaks: int = 2,
    n_simulations: int = 100000,
    relative_peak_threshold: float = 0.15,
    min_separation_angle: float = 10,
    sphere=None,
    gtab_full=None,
    signals_full=None,
):
    from dipy.reconst.force import FORCEModel
    from dipy.direction.peaks import peak_directions

    N = signals.shape[0]
    if sphere is None:
        sphere = get_dipy_sphere("default")

    cache_key = (n_simulations,
                 tuple(np.round(gtab.bvals, 1).tolist()),
                 tuple(gtab.bvecs.round(4).ravel().tolist()))
    if not hasattr(fit_dipy_force, '_model_cache'):
        fit_dipy_force._model_cache = {}
    model = fit_dipy_force._model_cache.get(cache_key)
    if model is None:
        model = FORCEModel(
            gtab,
            penalty=1e-5,
            n_neighbors=50,
            use_posterior=True,
            posterior_beta=500.0,
            compute_odf=True,
        )
        model.generate(
            num_simulations=n_simulations,
            num_cpus=4,
            verbose=True,
            use_cache=True,
        )
        fit_dipy_force._model_cache[cache_key] = model
    else:
        print("  [FORCE] Reusing in-process cached model")

    data_4d = signals[:, np.newaxis, np.newaxis, :]
    all_fits = model.fit(data_4d)

    all_dirs = np.zeros((N, npeaks, 3))
    all_vals = np.zeros((N, npeaks))
    mse_fit = np.zeros(N)

    for i in range(N):
        fit = all_fits[i, 0, 0]
        odf = fit.odf
        pred = fit.predicted_signal

        if odf is not None and np.max(odf) > 0:
            dirs, vals, _ = peak_directions(
                odf.astype(np.float64), sphere,
                relative_peak_threshold=relative_peak_threshold,
                min_separation_angle=min_separation_angle,
            )
            n = min(len(dirs), npeaks)
            all_dirs[i, :n] = dirs[:n]
            all_vals[i, :n] = vals[:n]

        if pred is not None:
            mse_fit[i] = np.mean((signals[i] - pred) ** 2)

    mx = all_vals.max()
    if mx > 0:
        all_vals /= mx

    mse_full = None
    if gtab_full is not None and signals_full is not None:
        try:
            full_key = (n_simulations,
                        tuple(np.round(gtab_full.bvals, 1).tolist()),
                        tuple(gtab_full.bvecs.round(4).ravel().tolist()))
            model_full = fit_dipy_force._model_cache.get(full_key)
            if model_full is None:
                model_full = FORCEModel(
                    gtab_full, penalty=1e-5, n_neighbors=50,
                    use_posterior=True, posterior_beta=500.0,
                    compute_odf=False,
                )
                model_full.generate(num_simulations=n_simulations, num_cpus=4,
                                    verbose=False, use_cache=True)
                fit_dipy_force._model_cache[full_key] = model_full
            data_full_4d = signals_full[:, np.newaxis, np.newaxis, :]
            fits_full = model_full.fit(data_full_4d)
            mse_full_arr = np.zeros(N)
            for i in range(N):
                pred_f = fits_full[i, 0, 0].predicted_signal
                if pred_f is not None:
                    mse_full_arr[i] = np.mean((signals_full[i] - pred_f) ** 2)
            mse_full = mse_full_arr
        except Exception:
            mse_full = None

    return all_dirs, all_vals, mse_fit, mse_full


def fit_dipy_peaks(
    signals,
    gtab,
    model_type='csd',
    npeaks: int = 2,
    response_mask=None,
    sh_order_max: int = 8,
    relative_peak_threshold: float = 0.15,
    min_separation_angle: float = 10,
    sphere=None,
    parallel: bool = True,
    num_processes: int | None = None,
    gtab_full=None,
    signals_full=None,
):
    from dipy.direction import peaks_from_model
    from dipy.reconst.csdeconv import (
        auto_response_ssst,
        ConstrainedSphericalDeconvModel,
        response_from_mask_ssst,
    )
    from dipy.reconst.shm import CsaOdfModel
    from dipy.reconst.dki import DiffusionKurtosisModel
    from dipy.reconst.sfm import SparseFascicleModel
    
    data = signals[:, np.newaxis, np.newaxis, :]
    mask = np.ones((len(signals), 1, 1), dtype=bool)
    if sphere is None:
        sphere = get_dipy_sphere("default")
    
    if model_type == 'csd':
        if response_mask is not None:
            resp_mask_3d = np.asarray(response_mask, dtype=bool).reshape(len(signals), 1, 1)
            response, _ = response_from_mask_ssst(gtab, data, resp_mask_3d)
        else:
            response, _ = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.5)
        model = ConstrainedSphericalDeconvModel(gtab, response, sh_order_max=sh_order_max)
    elif model_type == 'qball':
        model = CsaOdfModel(gtab, sh_order_max=sh_order_max)
    elif model_type == 'dki':
        model = DiffusionKurtosisModel(gtab)
    elif model_type == 'sfm':
        model = SparseFascicleModel(gtab, sphere=sphere, l1_ratio=0.5, alpha=0.001)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    peaks = peaks_from_model(
        model, data, sphere,
        relative_peak_threshold=relative_peak_threshold,
        min_separation_angle=min_separation_angle,
        mask=mask,
        npeaks=npeaks,
        parallel=parallel,
        num_processes=num_processes,
    )
    
    dirs = peaks.peak_dirs[:, 0, 0, :, :]
    values = peaks.peak_values[:, 0, 0, :]
    
    mse_fit = None
    mse_full = None
    try:
        fit = model.fit(data, mask=mask)
        pred_fit = fit.predict(gtab)[:, 0, 0, :]
        mse_fit = np.mean((signals - pred_fit) ** 2, axis=1)
        if gtab_full is not None and signals_full is not None:
            try:
                pred_full = fit.predict(gtab_full)[:, 0, 0, :]
                mse_full = np.mean((signals_full - pred_full) ** 2, axis=1)
            except Exception:
                mse_full = None
    except Exception:
        mse_fit = None
        mse_full = None
    
    return dirs, values, mse_fit, mse_full


def evaluate_method(
    pred_dirs,
    pred_weights,
    gt_dirs,
    gt_fractions,
    config_labels,
    method_name,
    weight_threshold: float = 0.05,
):
    from scipy.optimize import linear_sum_assignment

    def _safe_unit(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        return v / (np.linalg.norm(v) + 1e-12)

    def _normalize_weights(w):
        if w is None:
            return None
        w = np.asarray(w, dtype=np.float64)
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        w = np.clip(w, 0.0, None)
        s = float(w.sum())
        if s > 1e-12:
            w = w / s
        return w

    def _merge_antipodal_and_nearby_axes(
        dirs: np.ndarray,
        weights: np.ndarray,
        merge_angle_deg: float = 10.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        dirs = np.asarray(dirs, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        if dirs.shape[0] <= 1:
            return dirs, weights

        norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
        dirs_u = dirs / (norms + 1e-12)

        order = np.argsort(weights)[::-1]
        cluster_dirs: list[np.ndarray] = []
        cluster_w: list[float] = []
        cluster_vec: list[np.ndarray] = []

        for idx in order:
            d = dirs_u[idx]
            w = float(weights[idx])
            if not cluster_dirs:
                cluster_dirs.append(d)
                cluster_w.append(w)
                cluster_vec.append(d * w)
                continue

            assigned = False
            for ci, cdir in enumerate(cluster_dirs):
                if angular_error(d, cdir) < merge_angle_deg:
                    if float(np.dot(d, cdir)) < 0.0:
                        d = -d
                    cluster_vec[ci] = cluster_vec[ci] + d * w
                    cluster_w[ci] = float(cluster_w[ci] + w)
                    v = cluster_vec[ci]
                    cluster_dirs[ci] = v / (np.linalg.norm(v) + 1e-12)
                    assigned = True
                    break

            if not assigned:
                cluster_dirs.append(d)
                cluster_w.append(w)
                cluster_vec.append(d * w)

        merged_dirs = np.stack(cluster_dirs, axis=0)
        merged_w = _normalize_weights(np.asarray(cluster_w, dtype=np.float64))
        return merged_dirs, merged_w

    def _valid_peaks(dirs: np.ndarray, weights):
        dirs = np.asarray(dirs, dtype=np.float64)
        norms = np.linalg.norm(dirs, axis=-1)
        w = _normalize_weights(weights)
        if w is None:
            w = np.ones(dirs.shape[0], dtype=np.float64) / max(1, dirs.shape[0])
        keep = (norms > 0.1) & (w > weight_threshold)
        dirs_k = dirs[keep]
        w_k = w[keep]
        if dirs_k.shape[0] <= 1:
            return dirs_k, w_k
        return _merge_antipodal_and_nearby_axes(dirs_k, w_k, merge_angle_deg=10.0)

    results = {
        'method': method_name,
        'overall': {},
        'by_config': {},
        'by_n_fibers': {},
        'peak_detection': {},
        'peak_count': {},
    }
    
    all_errors_primary = []
    all_errors_all = []
    all_errors_all_weighted = []
    all_errors_bestmatch = []
    success_5 = 0
    success_10 = 0
    success_15 = 0
    success_20 = 0
    success_25 = 0
    
    spurious_peaks = []
    n_pred_peaks_list = []
    n_gt_peaks_list = []
    secondary_detected = []
    secondary_angular_errors = []

    peak_count_confusion: dict[int, dict[int, int]] = {}
    peak_count_total = 0
    peak_count_correct = 0
    
    single_fiber_errors = []
    two_fiber_errors_primary = []
    two_fiber_errors_all = []
    three_fiber_errors_primary = []
    three_fiber_errors_all = []
    
    single_success = {5: 0, 10: 0, 15: 0, 20: 0, 25: 0}
    single_total = 0
    two_success = {5: 0, 10: 0, 15: 0, 20: 0, 25: 0}
    two_total = 0
    three_success = {5: 0, 10: 0, 15: 0, 20: 0, 25: 0}
    three_total = 0
    
    single_fraction_errors = []
    two_fraction_errors = []
    three_fraction_errors = []
    
    fraction_errors = []
    
    for i in range(len(gt_dirs)):
        gt_w = _normalize_weights(gt_fractions[i])
        gt_d, gt_w = _valid_peaks(gt_dirs[i], gt_w)
        n_gt = gt_d.shape[0]
        if n_gt == 0:
            continue
        n_gt_peaks_list.append(n_gt)

        pw = pred_weights[i] if pred_weights is not None else None
        pred_d, pred_w = _valid_peaks(pred_dirs[i], pw)
        n_pred = pred_d.shape[0]
        n_pred_peaks_list.append(n_pred)

        peak_count_total += 1
        if n_pred == n_gt:
            peak_count_correct += 1
        peak_count_confusion.setdefault(n_gt, {})
        peak_count_confusion[n_gt][n_pred] = peak_count_confusion[n_gt].get(n_pred, 0) + 1
        
        spurious = max(0, n_pred - n_gt)
        spurious_peaks.append(spurious)

        if pred_d.shape[0] == 0:
            all_errors_primary.append(90.0)
            all_errors_all.append(90.0)
            all_errors_all_weighted.append(90.0)
            all_errors_bestmatch.append(90.0)
            if n_gt == 1:
                single_fiber_errors.append(90.0)
                single_total += 1
            elif n_gt == 2:
                two_fiber_errors_primary.append(90.0)
                two_fiber_errors_all.append(90.0)
                two_total += 1
                secondary_detected.append(False)
            elif n_gt >= 3:
                three_fiber_errors_primary.append(90.0)
                three_fiber_errors_all.append(90.0)
                three_total += 1
                secondary_detected.append(False)
            continue
        
        gt_primary = gt_d[int(np.argmax(gt_w))]
        pred_primary = pred_d[int(np.argmax(pred_w))]
        err_primary = angular_error(pred_primary, gt_primary)
        all_errors_primary.append(err_primary)
        
        cost = np.full((n_gt, max(n_gt, n_pred)), 90.0, dtype=np.float64)
        for gi in range(n_gt):
            for pj in range(n_pred):
                cost[gi, pj] = angular_error(_safe_unit(pred_d[pj]), _safe_unit(gt_d[gi]))

        row_ind, col_ind = linear_sum_assignment(cost)
        per_gt_errors = cost[row_ind, col_ind]
        err_all = float(per_gt_errors.mean())
        err_all_w = float((gt_w[row_ind] * per_gt_errors).sum() / (gt_w.sum() + 1e-12))
        all_errors_all.append(err_all)
        all_errors_all_weighted.append(err_all_w)
        
        bm_errors = []
        for gi in range(n_gt):
            best = min(angular_error(_safe_unit(pred_d[pj]), _safe_unit(gt_d[gi]))
                       for pj in range(n_pred))
            bm_errors.append(best)
        all_errors_bestmatch.append(float(np.mean(bm_errors)))
        
        if n_gt == 1:
            single_fiber_errors.append(err_all)
            single_total += 1
            for thr in [5, 10, 15, 20, 25]:
                if err_primary < thr:
                    single_success[thr] += 1
        elif n_gt == 2:
            two_fiber_errors_primary.append(err_primary)
            two_fiber_errors_all.append(err_all)
            two_total += 1
            for thr in [5, 10, 15, 20, 25]:
                if err_primary < thr:
                    two_success[thr] += 1
            if len(per_gt_errors) >= 2:
                secondary_err = per_gt_errors[1] if gt_w[0] > gt_w[1] else per_gt_errors[0]
                secondary_angular_errors.append(secondary_err)
                secondary_detected.append(secondary_err < 25.0)
            elif n_pred >= 2:
                gt_secondary_idx = 1 if gt_w[0] > gt_w[1] else 0
                best_match = min(angular_error(pred_d[j], gt_d[gt_secondary_idx]) for j in range(n_pred))
                secondary_angular_errors.append(best_match)
                secondary_detected.append(best_match < 25.0)
            else:
                secondary_detected.append(False)
        elif n_gt >= 3:
            three_fiber_errors_primary.append(err_primary)
            three_fiber_errors_all.append(err_all)
            three_total += 1
            for thr in [5, 10, 15, 20, 25]:
                if err_primary < thr:
                    three_success[thr] += 1
            if len(per_gt_errors) >= 2:
                secondary_err = per_gt_errors[1] if gt_w[0] > gt_w[1] else per_gt_errors[0]
                secondary_angular_errors.append(secondary_err)
                secondary_detected.append(secondary_err < 25.0)
            elif n_pred >= 2:
                gt_secondary_idx = 1 if gt_w[0] > gt_w[1] else 0
                best_match = min(angular_error(pred_d[j], gt_d[gt_secondary_idx]) for j in range(n_pred))
                secondary_angular_errors.append(best_match)
                secondary_detected.append(best_match < 25.0)
            else:
                secondary_detected.append(False)
        
        if err_primary < 5:
            success_5 += 1
        if err_primary < 10:
            success_10 += 1
        if err_primary < 15:
            success_15 += 1
        if err_primary < 20:
            success_20 += 1
        if err_primary < 25:
            success_25 += 1
        
        if n_gt >= 2 and pred_d.shape[0] >= 2:
            for gi, pi in zip(row_ind, col_ind):
                if pi < n_pred:
                    ferr = abs(float(gt_w[gi]) - float(pred_w[pi]))
                    fraction_errors.append(ferr)
                    if n_gt == 2:
                        two_fraction_errors.append(ferr)
                    elif n_gt >= 3:
                        three_fraction_errors.append(ferr)
        elif n_gt == 1 and pred_d.shape[0] >= 1:
            ferr = abs(float(gt_w[0]) - float(pred_w[int(np.argmax(pred_w))]))
            fraction_errors.append(ferr)
            single_fraction_errors.append(ferr)
    
    all_errors_primary = np.array(all_errors_primary)
    all_errors_all = np.array(all_errors_all)
    all_errors_all_weighted = np.array(all_errors_all_weighted)
    all_errors_bestmatch = np.array(all_errors_bestmatch)
    
    results['overall'] = {
        'primary_peak': {
            'mean': float(np.mean(all_errors_primary)),
            'median': float(np.median(all_errors_primary)),
            'std': float(np.std(all_errors_primary)),
            'p90': float(np.percentile(all_errors_primary, 90)),
            'p95': float(np.percentile(all_errors_primary, 95)),
        },
        'all_peaks': {
            'mean': float(np.mean(all_errors_all)),
            'median': float(np.median(all_errors_all)),
            'std': float(np.std(all_errors_all)),
        },
        'all_peaks_weighted_by_gt_fractions': {
            'mean': float(np.mean(all_errors_all_weighted)),
            'median': float(np.median(all_errors_all_weighted)),
            'std': float(np.std(all_errors_all_weighted)),
        },
        'best_match': {
            'mean': float(np.mean(all_errors_bestmatch)),
            'median': float(np.median(all_errors_bestmatch)),
            'std': float(np.std(all_errors_bestmatch)),
        },
        'success_rate_5deg': float(success_5 / len(all_errors_primary) * 100),
        'success_rate_10deg': float(success_10 / len(all_errors_primary) * 100),
        'success_rate_15deg': float(success_15 / len(all_errors_primary) * 100),
        'success_rate_20deg': float(success_20 / len(all_errors_primary) * 100),
        'success_rate_25deg': float(success_25 / len(all_errors_primary) * 100),
        'n_voxels': len(all_errors_primary),
    }
    
    results['fraction_estimation'] = {
        'mean_abs_error': float(np.mean(fraction_errors)) if fraction_errors else None,
        'median_abs_error': float(np.median(fraction_errors)) if fraction_errors else None,
        'rmse': float(np.sqrt(np.mean(np.array(fraction_errors)**2))) if fraction_errors else None,
        'n_pairs': len(fraction_errors),
    }
    
    def _fiber_count_stats(errors, errors_primary, n_total, success_dict, frac_errors):
        if not errors:
            return {'n_voxels': 0}
        arr = np.array(errors)
        arr_p = np.array(errors_primary) if errors_primary else arr
        stats = {
            'n_voxels': len(errors),
            'all_peaks_mean': float(np.mean(arr)),
            'all_peaks_median': float(np.median(arr)),
            'all_peaks_std': float(np.std(arr)),
            'all_peaks_p90': float(np.percentile(arr, 90)),
            'all_peaks_p95': float(np.percentile(arr, 95)),
            'primary_mean': float(np.mean(arr_p)),
            'primary_median': float(np.median(arr_p)),
            'primary_std': float(np.std(arr_p)),
        }
        if n_total > 0:
            for thr in [5, 10, 15, 20, 25]:
                stats[f'success_rate_{thr}deg'] = float(success_dict[thr] / n_total * 100)
        if frac_errors:
            stats['fraction_rmse'] = float(np.sqrt(np.mean(np.array(frac_errors)**2)))
            stats['fraction_mae'] = float(np.mean(frac_errors))
        return stats

    results['by_n_fibers'] = {
        '1_fiber': _fiber_count_stats(
            single_fiber_errors, single_fiber_errors, single_total, single_success, single_fraction_errors),
        '2_fiber': _fiber_count_stats(
            two_fiber_errors_all, two_fiber_errors_primary, two_total, two_success, two_fraction_errors),
        '3_fiber': _fiber_count_stats(
            three_fiber_errors_all, three_fiber_errors_primary, three_total, three_success, three_fraction_errors),
    }
    
    avg_pred_peaks = float(np.mean(n_pred_peaks_list)) if n_pred_peaks_list else 0
    avg_gt_peaks = float(np.mean(n_gt_peaks_list)) if n_gt_peaks_list else 0
    spurious_rate = float(np.mean(spurious_peaks)) if spurious_peaks else 0
    secondary_recall = float(np.mean(secondary_detected)) * 100 if secondary_detected else 0
    
    hardi_pd_values = []
    n_under_list = []
    n_over_list = []
    for gt_n, pred_n in zip(n_gt_peaks_list, n_pred_peaks_list):
        hardi_pd_values.append(abs(gt_n - pred_n) / gt_n)
        n_under_list.append(max(gt_n - pred_n, 0))
        n_over_list.append(max(pred_n - gt_n, 0))
    hardi_P_d = float(np.mean(hardi_pd_values)) if hardi_pd_values else None
    hardi_eps_theta = float(np.mean(all_errors_bestmatch)) if len(all_errors_bestmatch) > 0 else None
    hardi_n_minus = float(np.mean(n_under_list)) if n_under_list else None
    hardi_n_plus = float(np.mean(n_over_list)) if n_over_list else None
    
    tp_total = 0
    fp_total = 0
    fn_total = 0
    for i in range(len(gt_dirs)):
        gt_w_i = _normalize_weights(gt_fractions[i])
        gt_d_i, gt_w_i = _valid_peaks(gt_dirs[i], gt_w_i)
        n_gt_i = gt_d_i.shape[0]
        pw_i = pred_weights[i] if pred_weights is not None else None
        pred_d_i, pred_w_i = _valid_peaks(pred_dirs[i], pw_i)
        n_pred_i = pred_d_i.shape[0]
        
        if n_gt_i == 0 and n_pred_i == 0:
            continue
        if n_gt_i == 0:
            fp_total += n_pred_i
            continue
        if n_pred_i == 0:
            fn_total += n_gt_i
            continue
        
        cost_i = np.full((n_gt_i, n_pred_i), 90.0, dtype=np.float64)
        for gi in range(n_gt_i):
            for pj in range(n_pred_i):
                cost_i[gi, pj] = angular_error(_safe_unit(pred_d_i[pj]), _safe_unit(gt_d_i[gi]))
        row_idx_i, col_idx_i = linear_sum_assignment(cost_i)
        
        matched_gt = set()
        matched_pred = set()
        for gi, pi in zip(row_idx_i, col_idx_i):
            if cost_i[gi, pi] < 25.0:
                tp_total += 1
                matched_gt.add(gi)
                matched_pred.add(pi)
        fn_total += (n_gt_i - len(matched_gt))
        fp_total += (n_pred_i - len(matched_pred))
    
    precision = tp_total / max(1, tp_total + fp_total) * 100
    recall_pct = tp_total / max(1, tp_total + fn_total) * 100
    f1 = 2 * precision * recall_pct / max(1e-8, precision + recall_pct)
    
    results['peak_detection'] = {
        'avg_predicted_peaks': avg_pred_peaks,
        'avg_gt_peaks': avg_gt_peaks,
        'spurious_peak_rate': spurious_rate,
        'secondary_fiber_recall': secondary_recall,
        'secondary_mean_error': float(np.mean(secondary_angular_errors)) if secondary_angular_errors else None,
        'precision_pct': float(precision),
        'recall_pct': float(recall_pct),
        'f1_pct': float(f1),
        'tp': int(tp_total),
        'fp': int(fp_total),
        'fn': int(fn_total),
        'hardi_P_d': hardi_P_d,
        'hardi_eps_theta': hardi_eps_theta,
        'hardi_n_minus': hardi_n_minus,
        'hardi_n_plus': hardi_n_plus,
    }
    
    for config in np.unique(config_labels):
        mask = config_labels == config
        errors_config = all_errors_all[mask]
        errors_bm_config = all_errors_bestmatch[mask]
        
        results['by_config'][config] = {
            'mean': float(np.mean(errors_config)),
            'median': float(np.median(errors_config)),
            'std': float(np.std(errors_config)),
            'best_match_mean': float(np.mean(errors_bm_config)),
            'n_voxels': int(np.sum(mask)),
        }

    gt_labels = sorted(peak_count_confusion.keys())
    pred_labels = sorted({p for row in peak_count_confusion.values() for p in row.keys()})
    confusion_counts: dict[str, dict[str, int]] = {}
    accuracy_by_gt_n: dict[str, dict] = {}
    for gt_n in gt_labels:
        row = peak_count_confusion.get(gt_n, {})
        n_vox = int(sum(row.values()))
        correct = int(row.get(gt_n, 0))
        confusion_counts[str(gt_n)] = {str(p): int(row.get(p, 0)) for p in pred_labels}
        accuracy_by_gt_n[str(gt_n)] = {
            "n_voxels": n_vox,
            "accuracy_pct": float(correct / n_vox * 100.0) if n_vox > 0 else None,
        }

    results['peak_count'] = {
        "n_voxels": int(peak_count_total),
        "accuracy_overall_pct": float(peak_count_correct / peak_count_total * 100.0) if peak_count_total > 0 else None,
        "accuracy_by_gt_n": accuracy_by_gt_n,
        "confusion_counts": confusion_counts,
        "labels_pred": [int(p) for p in pred_labels],
        "labels_gt": [int(g) for g in gt_labels],
        "weight_threshold": float(weight_threshold),
    }
    
    return results


def run_benchmark(
    quick: bool = False,
    crossing_angles: list = None,
    snr: float = 30.0,
    seed: int = 42,
    device: str = "auto",
    n_voxels_per_config: int | None = None,
    prism_iters: int | None = None,
    weight_threshold: float = 0.05,
    dipy_sphere: str = "default",
    dipy_sh_order_max: int = 8,
    dipy_relative_peak_threshold: float = 0.15,
    dipy_min_separation_angle: float = 10,
    dipy_parallel: bool = True,
    dipy_num_processes: int | None = None,
    timing_warmup: int = 0,
    timing_repeats: int = 1,
    msmtcsd_response_mode: str = "oracle",
    csd_response_mode: str = "oracle",
    shell_mode: str = "standard",
    methods: list[str] | None = None,
    fraction_mode: str = "both",
    prism_loss_type: str = "mse",
):
    print_header("Multi-Tensor Simulation Benchmark")
    
    if device in (None, "", "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available; falling back to CPU.")
        device = "cpu"
    print(f"Device: {device}")
    set_global_seed(seed)
    
    if crossing_angles is None:
        crossing_angles = [30, 45, 60, 75, 90]
    
    n_voxels = int(n_voxels_per_config) if n_voxels_per_config is not None else (50 if quick else 200)
    n_iters = int(prism_iters) if prism_iters is not None else (100 if quick else 200)
    sphere = get_dipy_sphere(dipy_sphere)
    
    print(f"Configuration:")
    print(f"  Crossing angles: {crossing_angles}")
    print(f"  Voxels per config: {n_voxels}")
    print(f"  SNR: {snr}")
    print(f"  Fraction mode: {fraction_mode}")
    print(f"  Shell mode: {shell_mode}")
    print(f"  CSD response mode: {csd_response_mode}")
    print(f"  MSMT-CSD response mode: {msmtcsd_response_mode}")
    print(f"  PRISM iterations: {n_iters}")
    print(f"  PRISM loss: {prism_loss_type}")
    print(f"  DIPY sphere: {dipy_sphere} ({sphere.vertices.shape[0]} vertices)")
    print(f"  Peak params: sh_order_max={dipy_sh_order_max}, rel_thr={dipy_relative_peak_threshold}, min_sep={dipy_min_separation_angle}°")
    print(f"  Eval weight_threshold: {weight_threshold}")
    print(f"  Timing: warmup={timing_warmup}, repeats={timing_repeats} (reported=median)")
    
    print_header("1. Generating Multi-Tensor Phantom")
    signals, gt_dirs, gt_fractions, gtab, config_labels = generate_multitensor_phantom(
        n_voxels_per_config=n_voxels,
        crossing_angles=crossing_angles,
        snr=snr,
        b_values=DEFAULT_DWI_BVALS,
        n_directions=64,
        include_single_fiber=True,
        fraction_mode=fraction_mode,
        seed=seed,
    )

    if shell_mode == "matched_single":
        matched_bvals = [0, 3000]
        _ms, _mg, _ = subset_measurements_by_bvals(signals, gtab, matched_bvals)
        dti_signals, dti_gtab = _ms, _mg
        csd_signals, csd_gtab = _ms, _mg
        qball_signals, qball_gtab = _ms, _mg
        dki_signals, dki_gtab = _ms, _mg
        sfm_signals, sfm_gtab = _ms, _mg
        prism_signals, prism_gtab = _ms, _mg
        msmtcsd_signals, msmtcsd_gtab = _ms, _mg
        forecast_signals, forecast_gtab = _ms, _mg
        odffp_signals, odffp_gtab = _ms, _mg
        force_signals, force_gtab = _ms, _mg
        print(f"  [matched_single] All methods use b={matched_bvals}")
    elif shell_mode == "matched_two":
        matched_bvals = [0, 1000, 3000]
        _ms, _mg, _ = subset_measurements_by_bvals(signals, gtab, matched_bvals)
        dti_signals, dti_gtab = _ms, _mg
        csd_signals, csd_gtab = _ms, _mg
        qball_signals, qball_gtab = _ms, _mg
        dki_signals, dki_gtab = _ms, _mg
        sfm_signals, sfm_gtab = _ms, _mg
        prism_signals, prism_gtab = _ms, _mg
        msmtcsd_signals, msmtcsd_gtab = _ms, _mg
        forecast_signals, forecast_gtab = _ms, _mg
        odffp_signals, odffp_gtab = _ms, _mg
        force_signals, force_gtab = _ms, _mg
        print(f"  [matched_two] All methods use b={matched_bvals}")
    else:
        dti_signals, dti_gtab, _ = subset_measurements_by_bvals(signals, gtab, [0, 1000])
        csd_signals, csd_gtab, _ = subset_measurements_by_bvals(signals, gtab, [0, 3000])
        qball_signals, qball_gtab, _ = subset_measurements_by_bvals(signals, gtab, [0, 3000])
        dki_signals, dki_gtab, _ = subset_measurements_by_bvals(signals, gtab, [0, 1000, 2000])
        sfm_signals, sfm_gtab, _ = subset_measurements_by_bvals(signals, gtab, [0, 3000])
        prism_signals, prism_gtab = signals, gtab
        msmtcsd_signals, msmtcsd_gtab = signals, gtab
        forecast_signals, forecast_gtab = signals, gtab
        odffp_signals, odffp_gtab = signals, gtab
        force_signals, force_gtab = signals, gtab
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'environment': get_environment_info(),
        'config': {
            'crossing_angles': crossing_angles,
            'n_voxels_per_config': n_voxels,
            'snr': snr,
            'fraction_mode': fraction_mode,
            'n_measurements': len(gtab.bvals),
            'seed': seed,
            'device': device,
            'dipy_sphere': dipy_sphere,
            'dipy_sh_order_max': dipy_sh_order_max,
            'dipy_relative_peak_threshold': dipy_relative_peak_threshold,
            'dipy_min_separation_angle': dipy_min_separation_angle,
            'eval_weight_threshold': weight_threshold,
            'timing_warmup': timing_warmup,
            'timing_repeats': timing_repeats,
            'msmtcsd_response_mode': msmtcsd_response_mode,
            'csd_response_mode': csd_response_mode,
            'shell_mode': shell_mode,
            'simulation_mevals': [[0.0017, 0.0003, 0.0003], [0.0017, 0.0003, 0.0003]],
            'prism_model_note': 'stick-and-zeppelin with d_par=1.7e-3, d_perp=0.4e-3 (model mismatch by design)',
            'msmt_response': f'{msmtcsd_response_mode} response',
            'csd_response': f'{csd_response_mode} response' + (' (single-fiber GT mask)' if csd_response_mode == 'oracle' else ' (auto_response_ssst)'),
            'bval_usage_note': (f'Shell mode: {shell_mode}. '
                + ('Standard per method: DTI=[0,1000], CSD/Q-ball/SFM=[0,3000], DKI=[0,1000,2000], PRISM/MSMT-CSD/FORECAST/ODF-FP/FORCE=all shells'
                   if shell_mode == 'standard' else
                   f'All methods use matched shells')),
            'fraction_note': 'PRISM fraction = biophysical WM sub-fractions; '
                'DIPY fraction = normalized ODF peak amplitudes (proxy, not true volume fractions)',
            'reproducibility': 'Seeded RNG passed to DIPY multi_tensor noise; '
                'torch deterministic mode; bit-exact across runs',
        },
        'methods': {},
    }

    method_order = methods or ['prism', 'dti', 'csd', 'qball', 'dki', 'sfm', 'msmtcsd', 'forecast', 'odffp', 'force']
    method_order = [m.lower().strip() for m in method_order]
    
    prism_variants = []
    if 'prism' in method_order:
        prism_variants.append(('prism', prism_loss_type))
    if 'prism_mse' in method_order:
        prism_variants.append(('prism_mse', 'mse'))
    if 'prism_nll' in method_order:
        prism_variants.append(('prism_nll', 'nll_auto'))
    if 'prism_nll_oracle' in method_order:
        prism_variants.append(('prism_nll_oracle', 'nll'))
    if 'prism_nll_auto' in method_order:
        prism_variants.append(('prism_nll_auto', 'nll_auto'))
    
    for prism_name, prism_lt in prism_variants:
        print_header(f"2. Fitting {prism_name.upper()} (loss={prism_lt})")
        try:
            (prism_dirs, prism_fracs, prism_mse), prism_timing = _timed_call(
                fit_prism,
                prism_signals,
                prism_gtab,
                n_fibers=2,
                n_iters=n_iters,
                device=device,
                seed=seed,
                loss_type=prism_lt,
                snr=snr,
                sync_device=device,
                warmup=timing_warmup,
                repeats=timing_repeats,
            )
            prism_time = prism_timing["median_s"]
            print(f"  Time: {prism_time:.2f}s, MSE: {np.mean(prism_mse):.6f}")

            prism_results = evaluate_method(
                prism_dirs,
                prism_fracs,
                gt_dirs,
                gt_fractions,
                config_labels,
                prism_name.upper(),
                weight_threshold=weight_threshold,
            )
            prism_results['timing'] = prism_timing
            prism_results['time'] = prism_time
            prism_results['mse_fit'] = float(np.mean(prism_mse))
            prism_results['mse_full'] = float(np.mean(prism_mse))
            prism_results['fit_bvals'] = sorted(set(int(round(b)) for b in prism_gtab.bvals if b > 50)) or DEFAULT_DWI_BVALS
            results['methods'][prism_name] = prism_results
        except Exception as e:
            print(f"  {prism_name.upper()} failed: {e}")
    
    if 'dti' in method_order:
        print_header("3. Fitting DTI")
        try:
            (dti_dirs, dti_fracs, dti_mse_fit, dti_mse_full), dti_timing = _timed_call(
                fit_dipy_dti,
                dti_signals,
                dti_gtab,
                gtab_full=gtab,
                signals_full=signals,
                sync_device="cpu",
                warmup=timing_warmup,
                repeats=timing_repeats,
            )
            dti_time = dti_timing["median_s"]
            print(f"  Time: {dti_time:.2f}s, MSE_fit: {np.mean(dti_mse_fit):.6f}")

            dti_results = evaluate_method(
                dti_dirs,
                dti_fracs,
                gt_dirs,
                gt_fractions,
                config_labels,
                'DTI',
                weight_threshold=weight_threshold,
            )
            dti_results['timing'] = dti_timing
            dti_results['time'] = dti_time
            dti_results['mse_fit'] = float(np.mean(dti_mse_fit)) if dti_mse_fit is not None else None
            dti_results['mse_full'] = float(np.mean(dti_mse_full)) if dti_mse_full is not None else None
            dti_results['fit_bvals'] = [0, 1000]
            results['methods']['dti'] = dti_results
        except Exception as e:
            print(f"  DTI failed: {e}")
    
    if 'csd' in method_order:
        print_header("4. Fitting CSD")
        try:
            _csd_response_mask = (config_labels == 'single') if csd_response_mode == 'oracle' else None
            (csd_dirs, csd_vals, csd_mse_fit, csd_mse_full), csd_timing = _timed_call(
                fit_dipy_peaks,
                csd_signals,
                csd_gtab,
                'csd',
                npeaks=2,
                response_mask=_csd_response_mask,
                sh_order_max=dipy_sh_order_max,
                relative_peak_threshold=dipy_relative_peak_threshold,
                min_separation_angle=dipy_min_separation_angle,
                sphere=sphere,
                parallel=dipy_parallel,
                num_processes=dipy_num_processes,
                gtab_full=gtab,
                signals_full=signals,
                sync_device="cpu",
                warmup=timing_warmup,
                repeats=timing_repeats,
            )
            csd_time = csd_timing["median_s"]
            print(f"  Time: {csd_time:.2f}s")

            csd_results = evaluate_method(
                csd_dirs,
                csd_vals,
                gt_dirs,
                gt_fractions,
                config_labels,
                'CSD',
                weight_threshold=weight_threshold,
            )
            csd_results['timing'] = csd_timing
            csd_results['time'] = csd_time
            csd_results['mse_fit'] = float(np.mean(csd_mse_fit)) if csd_mse_fit is not None else None
            csd_results['mse_full'] = float(np.mean(csd_mse_full)) if csd_mse_full is not None else None
            csd_results['fit_bvals'] = [0, 3000]
            results['methods']['csd'] = csd_results
        except Exception as e:
            print(f"  CSD failed: {e}")
    
    if 'qball' in method_order:
        print_header("5. Fitting Q-Ball")
        try:
            (qball_dirs, qball_vals, qball_mse_fit, qball_mse_full), qball_timing = _timed_call(
                fit_dipy_peaks,
                qball_signals,
                qball_gtab,
                'qball',
                npeaks=2,
                sh_order_max=dipy_sh_order_max,
                relative_peak_threshold=dipy_relative_peak_threshold,
                min_separation_angle=dipy_min_separation_angle,
                sphere=sphere,
                parallel=dipy_parallel,
                num_processes=dipy_num_processes,
                gtab_full=gtab,
                signals_full=signals,
                sync_device="cpu",
                warmup=timing_warmup,
                repeats=timing_repeats,
            )
            qball_time = qball_timing["median_s"]
            print(f"  Time: {qball_time:.2f}s")

            qball_results = evaluate_method(
                qball_dirs,
                qball_vals,
                gt_dirs,
                gt_fractions,
                config_labels,
                'Q-Ball',
                weight_threshold=weight_threshold,
            )
            qball_results['timing'] = qball_timing
            qball_results['time'] = qball_time
            qball_results['mse_fit'] = float(np.mean(qball_mse_fit)) if qball_mse_fit is not None else None
            qball_results['mse_full'] = float(np.mean(qball_mse_full)) if qball_mse_full is not None else None
            qball_results['fit_bvals'] = [0, 3000]
            results['methods']['qball'] = qball_results
        except Exception as e:
            print(f"  Q-Ball failed: {e}")
    
    if 'dki' in method_order:
        print_header("6. Fitting DKI")
        try:
            (dki_dirs, dki_vals, dki_mse_fit, dki_mse_full), dki_timing = _timed_call(
                fit_dipy_peaks,
                dki_signals,
                dki_gtab,
                'dki',
                npeaks=2,
                sh_order_max=dipy_sh_order_max,
                relative_peak_threshold=dipy_relative_peak_threshold,
                min_separation_angle=dipy_min_separation_angle,
                sphere=sphere,
                parallel=dipy_parallel,
                num_processes=dipy_num_processes,
                gtab_full=gtab,
                signals_full=signals,
                sync_device="cpu",
                warmup=timing_warmup,
                repeats=timing_repeats,
            )
            dki_time = dki_timing["median_s"]
            print(f"  Time: {dki_time:.2f}s")

            dki_results = evaluate_method(
                dki_dirs,
                dki_vals,
                gt_dirs,
                gt_fractions,
                config_labels,
                'DKI',
                weight_threshold=weight_threshold,
            )
            dki_results['timing'] = dki_timing
            dki_results['time'] = dki_time
            dki_results['mse_fit'] = float(np.mean(dki_mse_fit)) if dki_mse_fit is not None else None
            dki_results['mse_full'] = float(np.mean(dki_mse_full)) if dki_mse_full is not None else None
            dki_results['fit_bvals'] = [0, 1000, 2000]
            results['methods']['dki'] = dki_results
        except Exception as e:
            print(f"  DKI failed: {e}")
    
    if 'sfm' in method_order:
        print_header("7. Fitting SFM (Sparse Fascicle Model)")
        try:
            (sfm_dirs, sfm_vals, sfm_mse_fit, sfm_mse_full), sfm_timing = _timed_call(
                fit_dipy_peaks,
                sfm_signals,
                sfm_gtab,
                'sfm',
                npeaks=2,
                sh_order_max=dipy_sh_order_max,
                relative_peak_threshold=dipy_relative_peak_threshold,
                min_separation_angle=dipy_min_separation_angle,
                sphere=sphere,
                parallel=dipy_parallel,
                num_processes=dipy_num_processes,
                gtab_full=gtab,
                signals_full=signals,
                sync_device="cpu",
                warmup=timing_warmup,
                repeats=timing_repeats,
            )
            sfm_time = sfm_timing["median_s"]
            print(f"  Time: {sfm_time:.2f}s")

            sfm_results = evaluate_method(
                sfm_dirs,
                sfm_vals,
                gt_dirs,
                gt_fractions,
                config_labels,
                'SFM',
                weight_threshold=weight_threshold,
            )
            sfm_results['timing'] = sfm_timing
            sfm_results['time'] = sfm_time
            sfm_results['mse_fit'] = float(np.mean(sfm_mse_fit)) if sfm_mse_fit is not None else None
            sfm_results['mse_full'] = float(np.mean(sfm_mse_full)) if sfm_mse_full is not None else None
            sfm_results['fit_bvals'] = [0, 3000]
            results['methods']['sfm'] = sfm_results
        except Exception as e:
            print(f"  SFM failed: {e}")
    
    if 'msmtcsd' in method_order:
        print_header("8. Fitting MSMT-CSD (Multi-Shell Multi-Tissue)")
        msmt_sh_order = max(dipy_sh_order_max, 12)
        try:
            (msmtcsd_dirs, msmtcsd_vals, msmtcsd_mse_fit, msmtcsd_mse_full), msmtcsd_timing = _timed_call(
                fit_dipy_msmtcsd,
                msmtcsd_signals,
                msmtcsd_gtab,
                npeaks=2,
                response_mode=msmtcsd_response_mode,
                sh_order_max=msmt_sh_order,
                relative_peak_threshold=dipy_relative_peak_threshold,
                min_separation_angle=dipy_min_separation_angle,
                sphere=sphere,
                parallel=dipy_parallel,
                num_processes=dipy_num_processes,
                gtab_full=gtab,
                signals_full=signals,
                sync_device="cpu",
                warmup=timing_warmup,
                repeats=timing_repeats,
            )
            msmtcsd_time = msmtcsd_timing["median_s"]
            print(f"  Time: {msmtcsd_time:.2f}s")

            msmtcsd_results = evaluate_method(
                msmtcsd_dirs,
                msmtcsd_vals,
                gt_dirs,
                gt_fractions,
                config_labels,
                'MSMT-CSD',
                weight_threshold=weight_threshold,
            )
            msmtcsd_results['timing'] = msmtcsd_timing
            msmtcsd_results['time'] = msmtcsd_time
            msmtcsd_results['mse_fit'] = float(np.mean(msmtcsd_mse_fit)) if msmtcsd_mse_fit is not None else None
            msmtcsd_results['mse_full'] = float(np.mean(msmtcsd_mse_full)) if msmtcsd_mse_full is not None else None
            msmtcsd_results['fit_bvals'] = DEFAULT_DWI_BVALS
            results['methods']['msmtcsd'] = msmtcsd_results
        except Exception as e:
            print(f"  MSMT-CSD failed: {e}")
    
    if 'forecast' in method_order:
        print_header("9. Fitting FORECAST")
        try:
            (forecast_dirs, forecast_vals, forecast_mse_fit, forecast_mse_full), forecast_timing = _timed_call(
                fit_dipy_forecast,
                forecast_signals,
                forecast_gtab,
                npeaks=2,
                sh_order_max=dipy_sh_order_max,
                relative_peak_threshold=dipy_relative_peak_threshold,
                min_separation_angle=dipy_min_separation_angle,
                sphere=sphere,
                parallel=dipy_parallel,
                num_processes=dipy_num_processes,
                gtab_full=gtab,
                signals_full=signals,
                sync_device="cpu",
                warmup=timing_warmup,
                repeats=timing_repeats,
            )
            forecast_time = forecast_timing["median_s"]
            print(f"  Time: {forecast_time:.2f}s")

            forecast_results = evaluate_method(
                forecast_dirs,
                forecast_vals,
                gt_dirs,
                gt_fractions,
                config_labels,
                'FORECAST',
                weight_threshold=weight_threshold,
            )
            forecast_results['timing'] = forecast_timing
            forecast_results['time'] = forecast_time
            forecast_results['mse_fit'] = float(np.mean(forecast_mse_fit)) if forecast_mse_fit is not None else None
            forecast_results['mse_full'] = float(np.mean(forecast_mse_full)) if forecast_mse_full is not None else None
            forecast_results['fit_bvals'] = DEFAULT_DWI_BVALS
            results['methods']['forecast'] = forecast_results
        except Exception as e:
            print(f"  FORECAST failed: {e}")
    
    if 'odffp' in method_order:
        print_header("10. Fitting ODF-FP")
        try:
            (odffp_dirs, odffp_vals, odffp_mse_fit, odffp_mse_full), odffp_timing = _timed_call(
                fit_dipy_odffp,
                odffp_signals,
                odffp_gtab,
                npeaks=2,
                dict_size=1000000,
                gtab_full=gtab,
                signals_full=signals,
                sync_device="cpu",
                warmup=timing_warmup,
                repeats=timing_repeats,
            )
            odffp_time = odffp_timing["median_s"]
            print(f"  Time: {odffp_time:.2f}s")

            odffp_results = evaluate_method(
                odffp_dirs,
                odffp_vals,
                gt_dirs,
                gt_fractions,
                config_labels,
                'ODF-FP',
                weight_threshold=weight_threshold,
            )
            odffp_results['timing'] = odffp_timing
            odffp_results['time'] = odffp_time
            odffp_results['mse_fit'] = float(np.mean(odffp_mse_fit)) if odffp_mse_fit is not None else None
            odffp_results['mse_full'] = float(np.mean(odffp_mse_full)) if odffp_mse_full is not None else None
            odffp_results['fit_bvals'] = DEFAULT_DWI_BVALS
            results['methods']['odffp'] = odffp_results
        except Exception as e:
            print(f"  ODF-FP failed: {e}")
    
    if 'force' in method_order:
        print_header("11. Fitting FORCE")
        try:
            (force_dirs, force_vals, force_mse_fit, force_mse_full), force_timing = _timed_call(
                fit_dipy_force,
                force_signals,
                force_gtab,
                npeaks=2,
                n_simulations=1000000,
                relative_peak_threshold=dipy_relative_peak_threshold,
                min_separation_angle=dipy_min_separation_angle,
                sphere=sphere,
                gtab_full=gtab,
                signals_full=signals,
                sync_device="cpu",
                warmup=timing_warmup,
                repeats=timing_repeats,
            )
            force_time = force_timing["median_s"]
            print(f"  Time: {force_time:.2f}s")

            force_results = evaluate_method(
                force_dirs,
                force_vals,
                gt_dirs,
                gt_fractions,
                config_labels,
                'FORCE',
                weight_threshold=weight_threshold,
            )
            force_results['timing'] = force_timing
            force_results['time'] = force_time
            force_results['mse_fit'] = float(np.mean(force_mse_fit)) if force_mse_fit is not None else None
            force_results['mse_full'] = float(np.mean(force_mse_full)) if force_mse_full is not None else None
            force_results['fit_bvals'] = DEFAULT_DWI_BVALS
            results['methods']['force'] = force_results
        except Exception as e:
            print(f"  FORCE failed: {e}")
            import traceback; traceback.print_exc()

    print_header("RESULTS SUMMARY")
    
    print("\n=== Angular Error vs Ground Truth ===")
    print(f"{'Method':<12} {'PrimMean':>9} {'AllMean':>9} {'BestMatch':>10} {'AllMed':>9} {'<5°':>7} {'<10°':>7} {'<15°':>7} {'<20°':>7} {'<25°':>7}")
    print("-" * 105)
    
    for method_name, method_results in results['methods'].items():
        overall = method_results['overall']
        bm = overall.get('best_match', {}).get('mean', float('nan'))
        print(f"{method_name.upper():<12} "
              f"{overall['primary_peak']['mean']:>8.2f}° "
              f"{overall['all_peaks']['mean']:>8.2f}° "
              f"{bm:>9.2f}° "
              f"{overall['all_peaks']['median']:>8.2f}° "
              f"{overall['success_rate_5deg']:>6.1f}% "
              f"{overall['success_rate_10deg']:>6.1f}% "
              f"{overall['success_rate_15deg']:>6.1f}% "
              f"{overall['success_rate_20deg']:>6.1f}% "
              f"{overall['success_rate_25deg']:>6.1f}%")
    
    print("\n=== Angular Error by Crossing Angle (Hungarian w/ 90° penalty) ===")
    print(f"{'Method':<12}", end="")
    configs = ['single'] + [f'cross_{a}' for a in crossing_angles]
    for config in configs:
        label = config.replace('cross_', '').replace('single', '1-fib')
        print(f"{label:>10}", end="")
    print()
    print("-" * (12 + 10 * len(configs)))
    
    for method_name, method_results in results['methods'].items():
        print(f"{method_name.upper():<12}", end="")
        for config in configs:
            if config in method_results['by_config']:
                mean = method_results['by_config'][config]['mean']
                print(f"{mean:>9.1f}°", end="")
            else:
                print(f"{'N/A':>10}", end="")
        print()

    print("\n=== Angular Error by Crossing Angle (Best-Match, no penalty) ===")
    print(f"{'Method':<12}", end="")
    for config in configs:
        label = config.replace('cross_', '').replace('single', '1-fib')
        print(f"{label:>10}", end="")
    print()
    print("-" * (12 + 10 * len(configs)))
    
    for method_name, method_results in results['methods'].items():
        print(f"{method_name.upper():<12}", end="")
        for config in configs:
            if config in method_results['by_config']:
                bm = method_results['by_config'][config].get('best_match_mean')
                if bm is not None:
                    print(f"{bm:>9.1f}°", end="")
                else:
                    print(f"{'N/A':>10}", end="")
            else:
                print(f"{'N/A':>10}", end="")
        print()
    
    print("\n=== Peak Detection (Precision/Recall/F1 @ 25° match threshold) ===")
    print(f"{'Method':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'2nd Recall':>11} {'Spurious':>10} {'FracRMSE':>10}")
    print("-" * 85)
    
    for method_name, method_results in results['methods'].items():
        peak = method_results.get('peak_detection', {})
        frac = method_results.get('fraction_estimation', {})
        
        prec = peak.get('precision_pct', None)
        rec = peak.get('recall_pct', None)
        f1_val = peak.get('f1_pct', None)
        secondary_rec = peak.get('secondary_fiber_recall', 0)
        spurious = peak.get('spurious_peak_rate', 0)
        frac_rmse = frac.get('rmse', None)
        
        prec_str = f"{prec:>9.1f}%" if prec is not None else f"{'N/A':>10}"
        rec_str = f"{rec:>9.1f}%" if rec is not None else f"{'N/A':>10}"
        f1_str = f"{f1_val:>9.1f}%" if f1_val is not None else f"{'N/A':>10}"
        frac_str = f"{frac_rmse:>9.4f}" if frac_rmse is not None else f"{'N/A':>10}"
        
        print(f"{method_name.upper():<12} {prec_str} {rec_str} {f1_str} {secondary_rec:>10.1f}% {spurious:>9.2f} {frac_str}")
    
    fiber_groups = [
        ('1_fiber', '1-FIBER'),
        ('2_fiber', '2-FIBER'),
        ('3_fiber', '3-FIBER'),
    ]
    
    for fkey, flabel in fiber_groups:
        any_data = any(
            method_results.get('by_n_fibers', {}).get(fkey, {}).get('n_voxels', 0) > 0
            for method_results in results['methods'].values()
        )
        if not any_data:
            continue
        
        print(f"\n=== {flabel} Voxels (Separate Report) ===")
        print(f"{'Method':<12} {'PrimMean':>9} {'AllMean':>9} {'AllMed':>9} {'Std':>7} {'P90':>7} {'<5°':>7} {'<10°':>7} {'<15°':>7} {'<20°':>7} {'<25°':>7} {'FracRMSE':>10} {'N':>6}")
        print("-" * 120)
        
        for method_name, method_results in results['methods'].items():
            by_n = method_results.get('by_n_fibers', {})
            stats = by_n.get(fkey, {})
            n_vox = stats.get('n_voxels', 0)
            
            if n_vox == 0:
                print(f"{method_name.upper():<12} {'(no data)':>9}")
                continue
            
            pm = stats.get('primary_mean')
            am = stats.get('all_peaks_mean')
            amd = stats.get('all_peaks_median')
            astd = stats.get('all_peaks_std')
            ap90 = stats.get('all_peaks_p90')
            frmse = stats.get('fraction_rmse')
            
            pm_s = f"{pm:>8.2f}°" if pm is not None else f"{'N/A':>9}"
            am_s = f"{am:>8.2f}°" if am is not None else f"{'N/A':>9}"
            amd_s = f"{amd:>8.2f}°" if amd is not None else f"{'N/A':>9}"
            astd_s = f"{astd:>6.2f}°" if astd is not None else f"{'N/A':>7}"
            ap90_s = f"{ap90:>6.2f}°" if ap90 is not None else f"{'N/A':>7}"
            fr_s = f"{frmse:>9.4f}" if frmse is not None else f"{'N/A':>10}"
            
            sr = {t: stats.get(f'success_rate_{t}deg', None) for t in [5, 10, 15, 20, 25]}
            sr_s = {t: f"{v:>6.1f}%" if v is not None else f"{'N/A':>7}" for t, v in sr.items()}
            
            print(f"{method_name.upper():<12} {pm_s} {am_s} {amd_s} {astd_s} {ap90_s} "
                  f"{sr_s[5]} {sr_s[10]} {sr_s[15]} {sr_s[20]} {sr_s[25]} {fr_s} {n_vox:>6}")
    
    print("\n=== Timing ===")
    for method_name, method_results in results['methods'].items():
        print(f"  {method_name.upper()}: {method_results['time']:.2f}s")
    
    return results


def write_benchmark_tables(results: dict, output_stem: Path) -> dict[str, str]:

    def _get(d: dict, *keys, default=None):
        cur = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    from collections.abc import Callable

    output_stem = Path(output_stem)
    crossing_angles = list(_get(results, "config", "crossing_angles", default=[]))
    methods = results.get("methods", {})

    summary_rows: list[dict] = []
    for method_key, method_results in methods.items():
        overall = method_results.get("overall", {})
        by_n = method_results.get("by_n_fibers", {})
        row = {
            "method_key": method_key,
            "method": method_results.get("method", method_key.upper()),
            "time_s": method_results.get("time", None),
            "primary_mean_deg": _get(overall, "primary_peak", "mean"),
            "all_mean_deg": _get(overall, "all_peaks", "mean"),
            "all_median_deg": _get(overall, "all_peaks", "median"),
            "best_match_mean_deg": _get(overall, "best_match", "mean"),
            "1fib_n_voxels": _get(by_n, "1_fiber", "n_voxels"),
            "1fib_primary_mean_deg": _get(by_n, "1_fiber", "primary_mean"),
            "1fib_all_mean_deg": _get(by_n, "1_fiber", "all_peaks_mean"),
            "1fib_all_median_deg": _get(by_n, "1_fiber", "all_peaks_median"),
            "1fib_all_std_deg": _get(by_n, "1_fiber", "all_peaks_std"),
            "1fib_all_p90_deg": _get(by_n, "1_fiber", "all_peaks_p90"),
            "1fib_success_5deg_pct": _get(by_n, "1_fiber", "success_rate_5deg"),
            "1fib_success_10deg_pct": _get(by_n, "1_fiber", "success_rate_10deg"),
            "1fib_success_15deg_pct": _get(by_n, "1_fiber", "success_rate_15deg"),
            "1fib_success_20deg_pct": _get(by_n, "1_fiber", "success_rate_20deg"),
            "1fib_success_25deg_pct": _get(by_n, "1_fiber", "success_rate_25deg"),
            "1fib_fraction_rmse": _get(by_n, "1_fiber", "fraction_rmse"),
            "2fib_n_voxels": _get(by_n, "2_fiber", "n_voxels"),
            "2fib_primary_mean_deg": _get(by_n, "2_fiber", "primary_mean"),
            "2fib_all_mean_deg": _get(by_n, "2_fiber", "all_peaks_mean"),
            "2fib_all_median_deg": _get(by_n, "2_fiber", "all_peaks_median"),
            "2fib_all_std_deg": _get(by_n, "2_fiber", "all_peaks_std"),
            "2fib_all_p90_deg": _get(by_n, "2_fiber", "all_peaks_p90"),
            "2fib_success_5deg_pct": _get(by_n, "2_fiber", "success_rate_5deg"),
            "2fib_success_10deg_pct": _get(by_n, "2_fiber", "success_rate_10deg"),
            "2fib_success_15deg_pct": _get(by_n, "2_fiber", "success_rate_15deg"),
            "2fib_success_20deg_pct": _get(by_n, "2_fiber", "success_rate_20deg"),
            "2fib_success_25deg_pct": _get(by_n, "2_fiber", "success_rate_25deg"),
            "2fib_fraction_rmse": _get(by_n, "2_fiber", "fraction_rmse"),
            "3fib_n_voxels": _get(by_n, "3_fiber", "n_voxels"),
            "3fib_primary_mean_deg": _get(by_n, "3_fiber", "primary_mean"),
            "3fib_all_mean_deg": _get(by_n, "3_fiber", "all_peaks_mean"),
            "3fib_all_median_deg": _get(by_n, "3_fiber", "all_peaks_median"),
            "3fib_all_std_deg": _get(by_n, "3_fiber", "all_peaks_std"),
            "3fib_all_p90_deg": _get(by_n, "3_fiber", "all_peaks_p90"),
            "3fib_success_5deg_pct": _get(by_n, "3_fiber", "success_rate_5deg"),
            "3fib_success_10deg_pct": _get(by_n, "3_fiber", "success_rate_10deg"),
            "3fib_success_15deg_pct": _get(by_n, "3_fiber", "success_rate_15deg"),
            "3fib_success_20deg_pct": _get(by_n, "3_fiber", "success_rate_20deg"),
            "3fib_success_25deg_pct": _get(by_n, "3_fiber", "success_rate_25deg"),
            "3fib_fraction_rmse": _get(by_n, "3_fiber", "fraction_rmse"),
            "success_5deg_pct": _get(overall, "success_rate_5deg"),
            "success_10deg_pct": _get(overall, "success_rate_10deg"),
            "success_15deg_pct": _get(overall, "success_rate_15deg"),
            "success_20deg_pct": _get(overall, "success_rate_20deg"),
            "success_25deg_pct": _get(overall, "success_rate_25deg"),
            "precision_pct": _get(method_results, "peak_detection", "precision_pct"),
            "recall_pct": _get(method_results, "peak_detection", "recall_pct"),
            "f1_pct": _get(method_results, "peak_detection", "f1_pct"),
            "secondary_recall_pct": _get(method_results, "peak_detection", "secondary_fiber_recall"),
            "spurious_peak_rate": _get(method_results, "peak_detection", "spurious_peak_rate"),
            "fraction_rmse": _get(method_results, "fraction_estimation", "rmse"),
            "fraction_mean_abs_error": _get(method_results, "fraction_estimation", "mean_abs_error"),
            "peak_count_acc_overall_pct": _get(method_results, "peak_count", "accuracy_overall_pct"),
            "peak_count_acc_gt1_pct": _get(method_results, "peak_count", "accuracy_by_gt_n", "1", "accuracy_pct"),
            "peak_count_acc_gt2_pct": _get(method_results, "peak_count", "accuracy_by_gt_n", "2", "accuracy_pct"),
            "peak_count_acc_gt3_pct": _get(method_results, "peak_count", "accuracy_by_gt_n", "3", "accuracy_pct"),
            "mse_fit": method_results.get("mse_fit", None),
            "mse_full": method_results.get("mse_full", None),
            "fit_bvals": ",".join(str(b) for b in method_results.get("fit_bvals", []) if b is not None),
        }
        for a in crossing_angles:
            row[f"cross_{a}_deg"] = _get(method_results, "by_config", f"cross_{a}", "mean")
            row[f"cross_{a}_bestmatch_deg"] = _get(method_results, "by_config", f"cross_{a}", "best_match_mean")
        summary_rows.append(row)

    summary_path = output_stem.parent / f"{output_stem.name}_summary.csv"
    summary_fields = list(summary_rows[0].keys()) if summary_rows else []
    if summary_rows:
        _write_csv(summary_path, summary_rows, summary_fields)

    by_config_rows: list[dict] = []
    for method_key, method_results in methods.items():
        for cfg, stats in (method_results.get("by_config", {}) or {}).items():
            by_config_rows.append(
                {
                    "method_key": method_key,
                    "method": method_results.get("method", method_key.upper()),
                    "config": cfg,
                    "mean_deg": stats.get("mean"),
                    "best_match_mean_deg": stats.get("best_match_mean"),
                    "median_deg": stats.get("median"),
                    "std_deg": stats.get("std"),
                    "n_voxels": stats.get("n_voxels"),
                }
            )

    by_config_path = output_stem.parent / f"{output_stem.name}_by_config.csv"
    if by_config_rows:
        _write_csv(
            by_config_path,
            by_config_rows,
            ["method_key", "method", "config", "mean_deg", "best_match_mean_deg", "median_deg", "std_deg", "n_voxels"],
        )

    paper_path = output_stem.parent / f"{output_stem.name}_paper.md"

    prism_key = "prism"
    competitor_pool = {k: v for k, v in methods.items() if k not in (prism_key, "dti")}

    def _val_deg(method_key: str, cfg: str) -> float | None:
        v = _get(methods.get(method_key, {}), "by_config", cfg, "mean")
        return float(v) if v is not None else None

    def _val_recall(method_key: str) -> float | None:
        v = _get(methods.get(method_key, {}), "peak_detection", "secondary_fiber_recall")
        return float(v) if v is not None else None

    def _val_time(method_key: str) -> float | None:
        v = methods.get(method_key, {}).get("time", None)
        return float(v) if v is not None else None

    def _val_count_acc(method_key: str, gt_n: int | None = None) -> float | None:
        if gt_n is None:
            v = _get(methods.get(method_key, {}), "peak_count", "accuracy_overall_pct")
        else:
            v = _get(methods.get(method_key, {}), "peak_count", "accuracy_by_gt_n", str(gt_n), "accuracy_pct")
        return float(v) if v is not None else None

    def _val_precision(method_key: str) -> float | None:
        v = _get(methods.get(method_key, {}), "peak_detection", "precision_pct")
        return float(v) if v is not None else None

    def _val_f1(method_key: str) -> float | None:
        v = _get(methods.get(method_key, {}), "peak_detection", "f1_pct")
        return float(v) if v is not None else None

    def _val_frac_rmse(method_key: str) -> float | None:
        v = _get(methods.get(method_key, {}), "fraction_estimation", "rmse")
        return float(v) if v is not None else None

    def _val_by_nfib(method_key: str, fib_key: str, stat: str) -> float | None:
        v = _get(methods.get(method_key, {}), "by_n_fibers", fib_key, stat)
        return float(v) if v is not None else None

    def _fmt_deg(x):
        return "N/A" if x is None else f"{x:.1f}°"

    def _fmt_pct(x):
        return "N/A" if x is None else f"{x:.1f}%"

    def _fmt_s(x):
        return "N/A" if x is None else f"{x:.2f}s"

    def _fmt_frac(x):
        return "N/A" if x is None else f"{x:.4f}"

    key_metrics: list[tuple[str, Callable[[str], float | None], bool]] = [
        ("1-Fiber: mean error", lambda k: _val_by_nfib(k, "1_fiber", "all_peaks_mean"), False),
        ("1-Fiber: <10° success", lambda k: _val_by_nfib(k, "1_fiber", "success_rate_10deg"), True),
        ("1-Fiber: <25° success", lambda k: _val_by_nfib(k, "1_fiber", "success_rate_25deg"), True),
        ("2-Fiber: primary mean", lambda k: _val_by_nfib(k, "2_fiber", "primary_mean"), False),
        ("2-Fiber: all-peaks mean", lambda k: _val_by_nfib(k, "2_fiber", "all_peaks_mean"), False),
        ("2-Fiber: <10° success", lambda k: _val_by_nfib(k, "2_fiber", "success_rate_10deg"), True),
        ("2-Fiber: <25° success", lambda k: _val_by_nfib(k, "2_fiber", "success_rate_25deg"), True),
        ("2-Fiber: fraction RMSE", lambda k: _val_by_nfib(k, "2_fiber", "fraction_rmse"), False),
        ("Secondary fiber recall", lambda k: _val_recall(k), True),
        ("Precision (25° match)", lambda k: _val_precision(k), True),
        ("F1 (25° match)", lambda k: _val_f1(k), True),
        ("Peak-count accuracy (1-fiber)", lambda k: _val_count_acc(k, 1), True),
        ("Peak-count accuracy (2-fiber)", lambda k: _val_count_acc(k, 2), True),
    ]
    for a in crossing_angles:
        key_metrics.append((f"{a}° crossing", lambda k, _cfg=f"cross_{a}": _val_deg(k, _cfg), False))
    key_metrics.append(("Speed (wall)", lambda k: _val_time(k), False))

    configs = ["single"] + [f"cross_{a}" for a in crossing_angles]
    win_counts: dict[str, int] = {k: 0 for k in methods.keys()}
    for cfg in configs:
        best_k = None
        best_v = None
        for mk in methods.keys():
            v = _val_deg(mk, cfg)
            if v is None:
                continue
            if best_v is None or v < best_v:
                best_v = v
                best_k = mk
        if best_k is not None:
            win_counts[best_k] = win_counts.get(best_k, 0) + 1

    prism_wins = win_counts.get(prism_key, 0)
    total_configs = len(configs)

    with open(paper_path, "w") as f:
        f.write("# Synthetic Multi-Tensor Benchmark (DIPY multi_tensor)\n\n")
        f.write(f"- Timestamp: {results.get('timestamp')}\n")
        f.write(f"- Seed: {_get(results, 'config', 'seed')}\n")
        f.write(f"- SNR: {_get(results, 'config', 'snr')}\n")
        f.write(f"- Voxels/config: {_get(results, 'config', 'n_voxels_per_config')}\n")
        f.write(f"- Device: {_get(results, 'config', 'device')}\n")
        f.write(f"- DIPY sphere: {_get(results, 'config', 'dipy_sphere')}\n")
        f.write(f"- PRISM wins: {prism_wins}/{total_configs} configs by mean angular error\n\n")
        f.write("## Key Metrics (PRISM vs best non-PRISM competitor)\n\n")
        f.write("| Metric | PRISM | Best Competitor | Advantage |\n")
        f.write("|---|---:|---:|---:|\n")

        for metric_name, get_value, higher_is_better in key_metrics:
            values_all = {k: get_value(k) for k in methods.keys()}
            prism_val = values_all.get(prism_key, None)
            comp_values = {k: values_all.get(k) for k in competitor_pool.keys()}
            best_key, best_val = _pick_best_competitor(comp_values, exclude=prism_key, higher_is_better=higher_is_better)

            if higher_is_better:
                prism_str = _fmt_pct(prism_val)
                best_str = f"{_fmt_pct(best_val)} ({methods[best_key].get('method', best_key.upper())})" if best_key else "N/A"
            elif metric_name.startswith("Speed"):
                prism_str = _fmt_s(prism_val)
                best_str = f"{_fmt_s(best_val)} ({methods[best_key].get('method', best_key.upper())})" if best_key else "N/A"
            elif "fraction rmse" in metric_name.lower() or metric_name.startswith("Fraction"):
                prism_str = _fmt_frac(prism_val)
                best_str = f"{_fmt_frac(best_val)} ({methods[best_key].get('method', best_key.upper())})" if best_key else "N/A"
            else:
                prism_str = _fmt_deg(prism_val)
                best_str = f"{_fmt_deg(best_val)} ({methods[best_key].get('method', best_key.upper())})" if best_key else "N/A"

            adv = _format_advantage(metric_name, prism_val, best_val, higher_is_better=higher_is_better)
            f.write(f"| {metric_name} | {prism_str} | {best_str} | {adv} |\n")

        for fib_key, fib_label in [("1_fiber", "1-Fiber"), ("2_fiber", "2-Fiber"), ("3_fiber", "3-Fiber")]:
            any_data = any(
                _get(methods.get(mk, {}), "by_n_fibers", fib_key, "n_voxels", default=0) > 0
                for mk in methods.keys()
            )
            if not any_data:
                continue

            f.write(f"\n## {fib_label} Voxels — Per-Method Breakdown\n\n")
            f.write(f"| Method | Prim Mean | All Mean | All Med | Std | P90 | <5° | <10° | <15° | <20° | <25° | Frac RMSE | N |\n")
            f.write(f"|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for mk, mr in methods.items():
                stats = _get(mr, "by_n_fibers", fib_key, default={})
                n_v = stats.get("n_voxels", 0)
                if n_v == 0:
                    f.write(f"| {mr.get('method', mk.upper())} | — | — | — | — | — | — | — | — | — | — | — | 0 |\n")
                    continue
                f.write(f"| {mr.get('method', mk.upper())}")
                f.write(f" | {_fmt_deg(stats.get('primary_mean'))}")
                f.write(f" | {_fmt_deg(stats.get('all_peaks_mean'))}")
                f.write(f" | {_fmt_deg(stats.get('all_peaks_median'))}")
                f.write(f" | {_fmt_deg(stats.get('all_peaks_std'))}")
                f.write(f" | {_fmt_deg(stats.get('all_peaks_p90'))}")
                for thr in [5, 10, 15, 20, 25]:
                    v = stats.get(f"success_rate_{thr}deg")
                    f.write(f" | {_fmt_pct(v)}")
                f.write(f" | {_fmt_frac(stats.get('fraction_rmse'))}")
                f.write(f" | {n_v} |\n")

        f.write("\n")
        f.write("### Fairness Notes\n\n")
        f.write("- **Simulation model**: DIPY `multi_tensor` with λ=[1.7, 0.3, 0.3]×10⁻³ mm²/s\n")
        f.write("- **PRISM model**: stick-and-zeppelin with d_par=1.7, d_perp=0.4 (deliberate model mismatch)\n")
        f.write("- **MSMT-CSD response**: oracle (eigenvalues match simulation exactly)\n")
        f.write("- **CSD response**: estimated from single-fiber voxels in the simulation\n")
        f.write("- **Peak detection**: all ODF methods use identical sphere, SH order, thresholds\n")
        f.write("- **Evaluation**: Hungarian matching with 90° penalty for unmatched peaks\n")
        f.write("- **Fraction RMSE**: PRISM reports biophysical WM sub-fractions; DIPY methods use normalized ODF peak amplitudes as fraction proxy\n")
        f.write("- **Fraction mode**: " + str(_get(results, 'config', 'fraction_mode', default='both')) + "\n")
        f.write("\n*Best competitor selection excludes DTI (single-fiber-only baseline).*\n")

    return {
        "summary_csv": str(summary_path),
        "by_config_csv": str(by_config_path),
        "paper_md": str(paper_path),
    }


def _aggregate_multi_seed(all_seed_results: list[dict]) -> dict:
    if len(all_seed_results) == 1:
        return all_seed_results[0]
    
    template = all_seed_results[0]
    combined = json.loads(json.dumps(template))
    combined['config']['n_seeds'] = len(all_seed_results)
    combined['config']['seeds_used'] = [r['config']['seed'] for r in all_seed_results]
    
    for method_key in combined.get('methods', {}):
        seed_method_data = [r['methods'][method_key] for r in all_seed_results if method_key in r.get('methods', {})]
        if not seed_method_data:
            continue
        
        for metric_group in ['primary_peak', 'all_peaks', 'all_peaks_weighted_by_gt_fractions']:
            for stat in ['mean', 'median', 'std']:
                vals = [d.get('overall', {}).get(metric_group, {}).get(stat) for d in seed_method_data]
                vals = [v for v in vals if v is not None]
                if vals:
                    combined['methods'][method_key]['overall'][metric_group][stat] = float(np.mean(vals))
                    combined['methods'][method_key]['overall'][metric_group][f'{stat}_seed_std'] = float(np.std(vals))
        
        for thr in [5, 10, 15, 20, 25]:
            key = f'success_rate_{thr}deg'
            vals = [d.get('overall', {}).get(key) for d in seed_method_data]
            vals = [v for v in vals if v is not None]
            if vals:
                combined['methods'][method_key]['overall'][key] = float(np.mean(vals))
                combined['methods'][method_key]['overall'][f'{key}_seed_std'] = float(np.std(vals))
        
        for det_key in ['precision_pct', 'recall_pct', 'f1_pct', 'secondary_fiber_recall']:
            vals = [d.get('peak_detection', {}).get(det_key) for d in seed_method_data]
            vals = [v for v in vals if v is not None]
            if vals:
                combined['methods'][method_key]['peak_detection'][det_key] = float(np.mean(vals))
                combined['methods'][method_key]['peak_detection'][f'{det_key}_seed_std'] = float(np.std(vals))
        
        vals = [d.get('fraction_estimation', {}).get('rmse') for d in seed_method_data]
        vals = [v for v in vals if v is not None]
        if vals:
            combined['methods'][method_key]['fraction_estimation']['rmse'] = float(np.mean(vals))
            combined['methods'][method_key]['fraction_estimation']['rmse_seed_std'] = float(np.std(vals))
        
        for cfg in combined['methods'][method_key].get('by_config', {}):
            vals = [d.get('by_config', {}).get(cfg, {}).get('mean') for d in seed_method_data]
            vals = [v for v in vals if v is not None]
            if vals:
                combined['methods'][method_key]['by_config'][cfg]['mean'] = float(np.mean(vals))
                combined['methods'][method_key]['by_config'][cfg]['mean_seed_std'] = float(np.std(vals))
        
        for fkey in ['1_fiber', '2_fiber', '3_fiber']:
            fdata = [d.get('by_n_fibers', {}).get(fkey, {}) for d in seed_method_data]
            for stat_key in ['all_peaks_mean', 'primary_mean', 'fraction_rmse']:
                vals = [fd.get(stat_key) for fd in fdata]
                vals = [v for v in vals if v is not None]
                if vals and fkey in combined['methods'][method_key].get('by_n_fibers', {}):
                    combined['methods'][method_key]['by_n_fibers'][fkey][stat_key] = float(np.mean(vals))
                    combined['methods'][method_key]['by_n_fibers'][fkey][f'{stat_key}_seed_std'] = float(np.std(vals))
    
    return combined


def run_sensitivity_analysis(
    base_args: dict,
    output_dir: Path,
    snr: float = 30.0,
) -> dict:
    lmax_vals = [6, 8, 10]
    peak_thr_vals = [0.1, 0.25, 0.4]
    min_sep_vals = [20, 25, 30]
    
    sensitivity_results = {
        'snr': snr,
        'sweeps': {
            'lmax': {'values': lmax_vals, 'results': {}},
            'peak_threshold': {'values': peak_thr_vals, 'results': {}},
            'min_separation': {'values': min_sep_vals, 'results': {}},
        }
    }
    
    print_header("SENSITIVITY: Sweeping SH order (lmax)")
    for lmax in lmax_vals:
        print(f"\n--- lmax = {lmax} ---")
        kwargs = dict(base_args)
        kwargs['dipy_sh_order_max'] = lmax
        kwargs['snr'] = snr
        try:
            res = run_benchmark(**kwargs)
            sensitivity_results['sweeps']['lmax']['results'][str(lmax)] = {
                mk: {
                    'all_mean': r.get('overall', {}).get('all_peaks', {}).get('mean'),
                    'f1': r.get('peak_detection', {}).get('f1_pct'),
                }
                for mk, r in res.get('methods', {}).items()
            }
        except Exception as e:
            print(f"  lmax={lmax} failed: {e}")
    
    print_header("SENSITIVITY: Sweeping peak threshold")
    for thr in peak_thr_vals:
        print(f"\n--- peak_threshold = {thr} ---")
        kwargs = dict(base_args)
        kwargs['dipy_relative_peak_threshold'] = thr
        kwargs['snr'] = snr
        try:
            res = run_benchmark(**kwargs)
            sensitivity_results['sweeps']['peak_threshold']['results'][str(thr)] = {
                mk: {
                    'all_mean': r.get('overall', {}).get('all_peaks', {}).get('mean'),
                    'f1': r.get('peak_detection', {}).get('f1_pct'),
                }
                for mk, r in res.get('methods', {}).items()
            }
        except Exception as e:
            print(f"  peak_threshold={thr} failed: {e}")
    
    print_header("SENSITIVITY: Sweeping min separation angle")
    for sep in min_sep_vals:
        print(f"\n--- min_sep = {sep}° ---")
        kwargs = dict(base_args)
        kwargs['dipy_min_separation_angle'] = sep
        kwargs['snr'] = snr
        try:
            res = run_benchmark(**kwargs)
            sensitivity_results['sweeps']['min_separation']['results'][str(sep)] = {
                mk: {
                    'all_mean': r.get('overall', {}).get('all_peaks', {}).get('mean'),
                    'f1': r.get('peak_detection', {}).get('f1_pct'),
                }
                for mk, r in res.get('methods', {}).items()
            }
        except Exception as e:
            print(f"  min_sep={sep} failed: {e}")
    
    print_header("SENSITIVITY SUMMARY")
    for sweep_name, sweep_data in sensitivity_results['sweeps'].items():
        print(f"\n--- {sweep_name} ---")
        vals = sweep_data['values']
        results_map = sweep_data['results']
        all_methods = set()
        for v_res in results_map.values():
            all_methods.update(v_res.keys())
        
        print(f"{'Method':<12}", end="")
        for v in vals:
            print(f"{'=' + str(v):>12}", end="")
        print()
        
        for mk in sorted(all_methods):
            print(f"{mk.upper():<12}", end="")
            for v in vals:
                r = results_map.get(str(v), {}).get(mk, {})
                mean_err = r.get('all_mean')
                if mean_err is not None:
                    print(f"{mean_err:>11.2f}°", end="")
                else:
                    print(f"{'N/A':>12}", end="")
            print()
    
    sens_path = output_dir / 'sensitivity_analysis.json'
    with open(sens_path, 'w') as f:
        json.dump(sensitivity_results, f, indent=2)
    print(f"\nSensitivity results saved to: {sens_path}")
    
    return sensitivity_results


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Tensor Simulation Benchmark (Literature-Standard)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard single-SNR run (paper results)
  python benchmark_multitensor_sim.py --snr 30 --seed 203

  # Multi-SNR sweep (literature standard)
  python benchmark_multitensor_sim.py --snr-levels 10 20 30 50

  # Quick test
  python benchmark_multitensor_sim.py --quick --methods prism msmtcsd

  # Equal fractions only (like Tournier 2007)
  python benchmark_multitensor_sim.py --fraction-mode equal

  # Reviewer-proof: matched shells + auto response + 3 seeds
  python benchmark_multitensor_sim.py --snr-levels 10 20 30 50 \\
      --shell-mode matched_two --csd-response-mode auto \\
      --msmtcsd-response-mode auto --seeds 3

  # Sensitivity analysis (peak-finding hyperparameters)
  python benchmark_multitensor_sim.py --sensitivity --methods prism csd msmtcsd qball

  # Full reviewer-proof experiment matrix
  python benchmark_multitensor_sim.py --reviewer-proof
""",
    )
    parser.add_argument('--quick', action='store_true', help='Quick mode with fewer voxels')
    parser.add_argument('--crossing-angles', type=int, nargs='+', default=[30, 45, 60, 75, 90],
                       help='Crossing angles to simulate')
    parser.add_argument('--snr', type=float, default=30.0, help='Signal-to-noise ratio (single-SNR mode)')
    parser.add_argument('--snr-levels', type=float, nargs='+', default=None,
                       help='Multiple SNR levels to sweep (e.g. 10 20 30 50). Overrides --snr.')
    parser.add_argument('--fraction-mode', type=str, default='both',
                       choices=['equal', 'random', 'both'],
                       help='Volume fraction regime: equal (50/50), random (30-70%%), or both')
    parser.add_argument('--seed', type=int, default=203, help='Random seed (203 for paper results)')
    parser.add_argument('--seeds', type=int, default=None,
                       help='Run N seeds (e.g. 3 → seeds 203,204,205). Reports mean±std.')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], help='Compute device for PRISM')
    parser.add_argument('--n-voxels-per-config', type=int, default=None, help='Override voxels per configuration')
    parser.add_argument('--prism-iters', type=int, default=None, help='Override PRISM optimization iterations')
    parser.add_argument('--weight-threshold', type=float, default=0.05, help='Min normalized peak weight for evaluation')
    parser.add_argument('--dipy-sphere', type=str, default='default', help="Sphere for DIPY peak finding (e.g. 'default', 'repulsion724')")
    parser.add_argument('--dipy-sh-order-max', type=int, default=8, help='Spherical harmonic order (lmax) for DIPY models')
    parser.add_argument('--dipy-relative-peak-threshold', type=float, default=0.25, help='DIPY peaks_from_model relative_peak_threshold')
    parser.add_argument('--dipy-min-separation-angle', type=float, default=10, help='DIPY peaks_from_model min_separation_angle (deg)')
    parser.add_argument('--prism-loss', type=str, default='mse', choices=['mse', 'nll'],
                        help='PRISM data fidelity loss: "mse" (default) or "nll" (Rician NLL)')
    parser.add_argument('--dipy-parallel', action=argparse.BooleanOptionalAction, default=True, help='Enable DIPY parallel peak extraction')
    parser.add_argument('--dipy-num-processes', type=int, default=None, help='DIPY peaks_from_model num_processes (optional)')
    parser.add_argument('--timing-warmup', type=int, default=0, help='Warmup runs per method (excluded from timing)')
    parser.add_argument('--timing-repeats', type=int, default=1, help='Timing repeats per method (reported=median)')
    parser.add_argument('--msmtcsd-response-mode', type=str, default='oracle',
                        choices=['oracle', 'auto', 'mask', 'ssst_fallback'],
                        help='Response mode for MSMT-CSD (synthetic default: oracle)')
    parser.add_argument('--csd-response-mode', type=str, default='oracle',
                        choices=['oracle', 'auto'],
                        help='CSD response: "oracle" (GT single-fiber mask) or "auto" (DIPY auto_response_ssst)')
    parser.add_argument('--shell-mode', type=str, default='standard',
                        choices=['standard', 'matched_single', 'matched_two'],
                        help='Shell assignment: "standard" (per-method conventional), '
                             '"matched_single" ([0,3000] for all), '
                             '"matched_two" ([0,1000,3000] for all)')
    parser.add_argument('--sensitivity', action='store_true',
                        help='Run peak-parameter sensitivity analysis (lmax, threshold, separation)')
    parser.add_argument('--reviewer-proof', action='store_true',
                        help='Run full reviewer-proof experiment matrix: '
                             'standard+matched shells, oracle+auto response, 3 seeds, sensitivity')
    parser.add_argument('--methods', type=str, nargs='+', default=None,
                        help='Subset of methods to run (e.g. prism csd qball msmtcsd)')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file')
    parser.add_argument('--no-tables', action='store_true', help='Do not write CSV/Markdown tables')
    args = parser.parse_args()
    
    print("=" * 70)
    print("  MULTI-TENSOR SIMULATION BENCHMARK (Literature-Standard)")
    print("  Ground Truth Comparison: PRISM vs DTI vs CSD vs DKI vs Q-Ball vs SFM")
    print("  vs MSMT-CSD vs FORECAST vs FORCE")
    print("=" * 70)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print(f"Fraction mode: {args.fraction_mode}")
    print(f"Shell mode: {args.shell_mode}")
    print(f"CSD response: {args.csd_response_mode}")
    print(f"MSMT-CSD response: {args.msmtcsd_response_mode}")
    if args.seeds:
        print(f"Multi-seed: {args.seeds} seeds starting at {args.seed}")
    
    output_dir = Path(__file__).parent / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.reviewer_proof:
        print_header("REVIEWER-PROOF EXPERIMENT MATRIX")
        snr_levels = args.snr_levels or [10, 20, 30, 50]
        n_seeds = args.seeds or 3
        methods = args.methods or ['prism', 'msmtcsd', 'csd', 'qball', 'forecast', 'odffp', 'force']
        
        base_kwargs = dict(
            quick=args.quick,
            crossing_angles=args.crossing_angles,
            device=args.device,
            n_voxels_per_config=args.n_voxels_per_config,
            prism_iters=args.prism_iters,
            weight_threshold=args.weight_threshold,
            dipy_sphere=args.dipy_sphere,
            dipy_sh_order_max=args.dipy_sh_order_max,
            dipy_relative_peak_threshold=args.dipy_relative_peak_threshold,
            dipy_min_separation_angle=args.dipy_min_separation_angle,
            dipy_parallel=args.dipy_parallel,
            dipy_num_processes=args.dipy_num_processes,
            timing_warmup=args.timing_warmup,
            timing_repeats=args.timing_repeats,
            methods=methods,
            fraction_mode=args.fraction_mode,
        )
        
        conditions = [
            ('A_standard_oracle', 'standard', 'oracle', 'oracle'),
            ('B_standard_auto',   'standard', 'auto',   'auto'),
            ('C_matched2_oracle', 'matched_two', 'oracle', 'oracle'),
            ('D_matched2_auto',   'matched_two', 'auto',   'auto'),
        ]
        
        master_results = {
            'experiment': 'reviewer_proof',
            'snr_levels': snr_levels,
            'n_seeds': n_seeds,
            'conditions': {},
        }
        
        for cond_label, shell_mode, csd_resp, msmtcsd_resp in conditions:
            print(f"\n{'='*70}")
            print(f"  CONDITION: {cond_label}")
            print(f"  shell_mode={shell_mode}, csd={csd_resp}, msmtcsd={msmtcsd_resp}")
            print(f"{'='*70}")
            
            cond_all_snr = {}
            for snr in snr_levels:
                seed_results = []
                for si in range(n_seeds):
                    seed = args.seed + si
                    print(f"\n  --- {cond_label} | SNR={int(snr)} | seed={seed} ---")
                    res = run_benchmark(
                        **base_kwargs,
                        snr=snr,
                        seed=seed,
                        shell_mode=shell_mode,
                        csd_response_mode=csd_resp,
                        msmtcsd_response_mode=msmtcsd_resp,
                    )
                    seed_results.append(res)
                
                cond_all_snr[f"snr_{int(snr)}"] = _aggregate_multi_seed(seed_results)
            
            master_results['conditions'][cond_label] = {
                'shell_mode': shell_mode,
                'csd_response_mode': csd_resp,
                'msmtcsd_response_mode': msmtcsd_resp,
                'results_by_snr': cond_all_snr,
            }
        
        print_header("SENSITIVITY ANALYSIS")
        sens_kwargs = dict(base_kwargs)
        sens_kwargs['shell_mode'] = 'standard'
        sens_kwargs['csd_response_mode'] = 'oracle'
        sens_kwargs['msmtcsd_response_mode'] = 'oracle'
        sens_kwargs['seed'] = args.seed
        sens_results = run_sensitivity_analysis(sens_kwargs, output_dir, snr=30.0)
        master_results['sensitivity'] = sens_results
        
        rp_path = output_dir / 'reviewer_proof.json'
        with open(rp_path, 'w') as f:
            json.dump(master_results, f, indent=2)
        print(f"\nReviewer-proof results saved to: {rp_path}")
        
        if not args.no_tables:
            cond_a = master_results['conditions'].get('A_standard_oracle', {})
            cond_a_snr30 = cond_a.get('results_by_snr', {}).get('snr_30')
            if cond_a_snr30:
                stem = output_dir / 'reviewer_proof_condA'
                table_paths = write_benchmark_tables(cond_a_snr30, stem)
                print("Tables (Condition A, SNR=30):")
                for k, v in table_paths.items():
                    print(f"  {k}: {v}")
        
        print_header("CROSS-CONDITION COMPARISON (SNR=30, All-Peaks Mean)")
        print(f"{'Method':<12}", end="")
        for cond_label, _, _, _ in conditions:
            short = cond_label.split('_', 1)[0]
            print(f"{short:>14}", end="")
        print()
        print("-" * (12 + 14 * len(conditions)))
        
        all_mk = set()
        for cond_label, _, _, _ in conditions:
            cond = master_results['conditions'].get(cond_label, {})
            snr30 = cond.get('results_by_snr', {}).get('snr_30', {})
            all_mk.update(snr30.get('methods', {}).keys())
        
        for mk in sorted(all_mk):
            print(f"{mk.upper():<12}", end="")
            for cond_label, _, _, _ in conditions:
                cond = master_results['conditions'].get(cond_label, {})
                snr30 = cond.get('results_by_snr', {}).get('snr_30', {})
                m = snr30.get('methods', {}).get(mk, {}).get('overall', {}).get('all_peaks', {})
                val = m.get('mean')
                std = m.get('mean_seed_std')
                if val is not None and std is not None:
                    print(f"{val:>8.2f}±{std:>4.2f}", end="")
                elif val is not None:
                    print(f"{val:>12.2f}°", end="")
                else:
                    print(f"{'N/A':>14}", end="")
            print()
        
        return
    
    if args.sensitivity:
        base_kwargs = dict(
            quick=args.quick,
            crossing_angles=args.crossing_angles,
            seed=args.seed,
            device=args.device,
            n_voxels_per_config=args.n_voxels_per_config,
            prism_iters=args.prism_iters,
            weight_threshold=args.weight_threshold,
            dipy_sphere=args.dipy_sphere,
            dipy_sh_order_max=args.dipy_sh_order_max,
            dipy_relative_peak_threshold=args.dipy_relative_peak_threshold,
            dipy_min_separation_angle=args.dipy_min_separation_angle,
            dipy_parallel=args.dipy_parallel,
            dipy_num_processes=args.dipy_num_processes,
            timing_warmup=args.timing_warmup,
            timing_repeats=args.timing_repeats,
            msmtcsd_response_mode=args.msmtcsd_response_mode,
            csd_response_mode=args.csd_response_mode,
            shell_mode=args.shell_mode,
            methods=args.methods,
            fraction_mode=args.fraction_mode,
        )
        run_sensitivity_analysis(base_kwargs, output_dir, snr=args.snr)
        return
    
    snr_levels = args.snr_levels if args.snr_levels else [args.snr]
    n_seeds = args.seeds or 1
    seed_list = [args.seed + i for i in range(n_seeds)]
    
    all_results = {}
    for snr in snr_levels:
        seed_results_for_snr = []
        for seed in seed_list:
            print(f"\n{'='*70}")
            print(f"  Running SNR = {snr}" + (f", seed = {seed}" if n_seeds > 1 else ""))
            print(f"{'='*70}")
            
            results = run_benchmark(
                quick=args.quick,
                crossing_angles=args.crossing_angles,
                snr=snr,
                seed=seed,
                device=args.device,
                n_voxels_per_config=args.n_voxels_per_config,
                prism_iters=args.prism_iters,
                weight_threshold=args.weight_threshold,
                dipy_sphere=args.dipy_sphere,
                dipy_sh_order_max=args.dipy_sh_order_max,
                dipy_relative_peak_threshold=args.dipy_relative_peak_threshold,
                dipy_min_separation_angle=args.dipy_min_separation_angle,
                dipy_parallel=args.dipy_parallel,
                dipy_num_processes=args.dipy_num_processes,
                timing_warmup=args.timing_warmup,
                timing_repeats=args.timing_repeats,
                msmtcsd_response_mode=args.msmtcsd_response_mode,
                csd_response_mode=args.csd_response_mode,
                shell_mode=args.shell_mode,
                methods=args.methods,
                fraction_mode=args.fraction_mode,
                prism_loss_type=args.prism_loss,
            )
            seed_results_for_snr.append(results)
        
        aggregated = _aggregate_multi_seed(seed_results_for_snr)
        all_results[f"snr_{int(snr)}"] = aggregated
    
    if len(snr_levels) == 1:
        output_file = args.output or (output_dir / 'multitensor_benchmark.json')
        save_results = aggregated
    else:
        output_file = args.output or (output_dir / 'multitensor_benchmark_multisnr.json')
        save_results = {
            'snr_levels': snr_levels,
            'fraction_mode': args.fraction_mode,
            'shell_mode': args.shell_mode,
            'csd_response_mode': args.csd_response_mode,
            'msmtcsd_response_mode': args.msmtcsd_response_mode,
            'n_seeds': n_seeds,
            'results_by_snr': all_results,
        }
    
    output_file = Path(output_file)
    with open(output_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    if not args.no_tables:
        primary_results = all_results[f"snr_{int(snr_levels[-1])}"] if len(snr_levels) > 1 else aggregated
        stem = output_file.with_suffix("")
        table_paths = write_benchmark_tables(primary_results, stem)
        print("Tables written:")
        for k, v in table_paths.items():
            print(f"  {k}: {v}")
    
    if len(snr_levels) > 1:
        print_header("CROSS-SNR SUMMARY (All-Peaks Mean Angular Error)")
        has_seed_std = n_seeds > 1
        print(f"{'Method':<12}", end="")
        for snr in snr_levels:
            print(f"{'SNR='+str(int(snr)):>14}", end="")
        print()
        print("-" * (12 + 14 * len(snr_levels)))
        
        all_method_names = set()
        for snr_key, res in all_results.items():
            all_method_names.update(res.get('methods', {}).keys())
        
        for method_name in sorted(all_method_names):
            print(f"{method_name.upper():<12}", end="")
            for snr in snr_levels:
                snr_key = f"snr_{int(snr)}"
                method_res = all_results.get(snr_key, {}).get('methods', {}).get(method_name, {})
                m = method_res.get('overall', {}).get('all_peaks', {})
                val = m.get('mean')
                std = m.get('mean_seed_std')
                if val is not None and has_seed_std and std is not None:
                    print(f"{val:>8.2f}±{std:>4.2f}", end="")
                elif val is not None:
                    print(f"{val:>13.2f}°", end="")
                else:
                    print(f"{'N/A':>14}", end="")
            print()


if __name__ == '__main__':
    main()
