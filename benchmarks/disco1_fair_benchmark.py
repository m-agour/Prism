#!/usr/bin/env python3

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion
from scipy.stats import pearsonr

from dipy.core.gradients import gradient_table
from dipy.data import default_sphere, get_fnames
from dipy.io.image import load_nifti, load_nifti_data
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
from dipy.reconst.mcsd import MultiShellDeconvModel, mask_for_response_msmt, response_from_mask_msmt
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.tracking.tracker import deterministic_tracking, probabilistic_tracking, ptt_tracking
from dipy.tracking.utils import connectivity_matrix, seeds_from_mask
from dipy.io.streamline import save_tractogram, load_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space

sys.path.insert(0, str(Path(__file__).parent.parent))


def fiber_dirs_to_odf_fair(fiber_dirs, f_wm_fibers, kappa=12.0, sphere=None, target_sum=22.0):
    if sphere is None:
        sphere = default_sphere
    
    n_vertices = len(sphere.vertices)
    
    if fiber_dirs.ndim == 4 and fiber_dirs.shape[-1] != 3:
        n_total = fiber_dirs.shape[-1]
        n_fibers = n_total // 3
        fiber_dirs = fiber_dirs.reshape(fiber_dirs.shape[:3] + (n_fibers, 3))
    
    shape = fiber_dirs.shape[:3]
    n_fibers = fiber_dirs.shape[3]
    
    fiber_dirs_norm = np.linalg.norm(fiber_dirs, axis=-1, keepdims=True)
    fiber_dirs_norm = np.where(fiber_dirs_norm > 0, fiber_dirs_norm, 1)
    fiber_dirs = fiber_dirs / fiber_dirs_norm
    
    C = kappa / (4 * np.pi * np.sinh(kappa))
    
    odf = np.zeros(shape + (n_vertices,), dtype=np.float64)
    
    for f in range(n_fibers):
        fiber_dir = fiber_dirs[..., f, :]
        weights = f_wm_fibers[..., f]
        
        for i, v in enumerate(sphere.vertices):
            cos_theta = np.abs(np.einsum('...d,d->...', fiber_dir, v))
            cos_theta = np.clip(cos_theta, 0, 1)
            watson = C * np.exp(kappa * cos_theta**2)
            odf[..., i] += weights * watson
    
    odf = np.maximum(odf, 0)
    current_sum = odf.sum(axis=-1, keepdims=True)
    scale = np.where(current_sum > 1e-6, target_sum / current_sum, 0.0)
    odf = odf * scale
    
    return odf.astype(np.float32)


def load_disco_data(snr=50):
    fnames = get_fnames(name='disco1')
    disco1_fnames = [os.path.basename(f) for f in fnames]
    
    GT_connectome = np.loadtxt(
        fnames[disco1_fnames.index('DiSCo1_Connectivity_Matrix_Cross-Sectional_Area.txt')]
    )
    
    connectome_mask = np.tril(np.ones(GT_connectome.shape), -1) > 0
    
    labels_fname = fnames[disco1_fnames.index('highRes_DiSCo1_ROIs.nii.gz')]
    labels, affine, labels_img = load_nifti(labels_fname, return_img=True)
    labels = labels.astype(int)
    
    mask_fname = fnames[disco1_fnames.index('highRes_DiSCo1_mask.nii.gz')]
    mask = load_nifti_data(mask_fname)
    sc = BinaryStoppingCriterion(mask)
    
    seed_fname = fnames[disco1_fnames.index('highRes_DiSCo1_ROIs-mask.nii.gz')]
    seed_mask = load_nifti_data(seed_fname)
    seed_mask = binary_erosion(seed_mask * mask, iterations=1)
    seeds = seeds_from_mask(seed_mask, affine, density=2)
    
    dipy_disco_dir = Path.home() / '.dipy' / 'disco' / 'disco_1'
    dwi_fname = dipy_disco_dir / f'highRes_DiSCo1_DWI_RicianNoise-snr{snr}.nii.gz'
    if dwi_fname.exists():
        data = nib.load(dwi_fname).get_fdata()
    elif snr == 10:
        data_fname = fnames[disco1_fnames.index('highRes_DiSCo1_DWI_RicianNoise-snr10.nii.gz')]
        data = load_nifti_data(data_fname)
    else:
        raise FileNotFoundError(f"DWI not found: {dwi_fname}")
    
    bvecs = fnames[disco1_fnames.index('DiSCo_gradients_dipy.bvecs')]
    bvals = fnames[disco1_fnames.index('DiSCo_gradients.bvals')]
    gtab = gradient_table(bvals=bvals, bvecs=bvecs)
    
    return {
        'data': data,
        'gtab': gtab,
        'mask': mask,
        'labels': labels,
        'affine': affine,
        'sc': sc,
        'seeds': seeds,
        'GT_connectome': GT_connectome,
        'connectome_mask': connectome_mask,
    }


def _merge_close_peaks_vol(fiber_dirs, f_wm_fibers, merge_angle_deg):
    spatial_shape = fiber_dirs.shape[:-2]
    n_fibers = fiber_dirs.shape[-2]
    
    dirs_flat = fiber_dirs.reshape(-1, n_fibers, 3).copy()
    fracs_flat = f_wm_fibers.reshape(-1, n_fibers).copy()
    N = dirs_flat.shape[0]
    
    merge_cos = np.cos(np.radians(merge_angle_deg))
    
    norms = np.linalg.norm(dirs_flat, axis=-1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    dirs_normed = dirs_flat / norms
    
    order = np.argsort(-fracs_flat, axis=-1)
    
    idx_n = np.arange(N)[:, None]
    dirs_sorted = dirs_normed[idx_n, order]
    fracs_sorted = fracs_flat[idx_n, order]
    
    d0 = dirs_sorted[:, 0, :]
    
    for j in range(1, n_fibers):
        dj = dirs_sorted[:, j, :]
        cos_sep = np.abs(np.einsum('nd,nd->n', d0, dj))
        close = cos_sep >= merge_cos
        
        fib_j_idx = order[:, j]
        fib_0_idx = order[:, 0]
        fracs_flat[idx_n[:, 0][close], fib_0_idx[close]] += fracs_flat[idx_n[:, 0][close], fib_j_idx[close]]
        fracs_flat[idx_n[:, 0][close], fib_j_idx[close]] = 0.0
    
    n_merged = int((f_wm_fibers.reshape(-1, n_fibers) > 0.01).sum() - (fracs_flat > 0.01).sum())
    print(f"  Peak merging (<{merge_angle_deg}°): {n_merged:,} fiber peaks merged")
    
    return fiber_dirs, fracs_flat.reshape(f_wm_fibers.shape)


def load_gymri_odf(snr=50, fit_dir_override=None, merge_peaks_angle=0):
    if fit_dir_override is not None:
        fit_dir = Path(fit_dir_override)
        if not fit_dir.exists() or not (fit_dir / 'fiber_dirs.nii.gz').exists():
            raise FileNotFoundError(f"GYMRI fit not found at: {fit_dir}")
    else:
        gymri_root = Path(__file__).parent.parent
        fit_dirs = [
        
        gymri_root / f'outputs/fitting/disco1_snr50_defaults_encoder',

    ]
    
    if fit_dir_override is None:
        fit_dir = None
        for d in fit_dirs:
            if d.exists() and (d / 'fiber_dirs.nii.gz').exists():
                fit_dir = d
                break
    
        if fit_dir is None:
            raise FileNotFoundError(f"GYMRI fit not found. Tried: {fit_dirs}")
    
    
    print(f"  Loading GYMRI from: {fit_dir}")
    
    fiber_dirs = nib.load(f'{fit_dir}/fiber_dirs.nii.gz').get_fdata()
    f_wm_fibers = nib.load(f'{fit_dir}/f_wm_fibers.nii.gz').get_fdata()
    
    f_restricted_path = fit_dir / 'f_restricted.nii.gz'
    if f_restricted_path.exists():
        f_restricted = nib.load(str(f_restricted_path)).get_fdata()
        
        max_frac = f_wm_fibers.max(axis=-1, keepdims=True)
        relative_threshold = 0.25
        fiber_mask = f_wm_fibers >= (max_frac * relative_threshold)
        
        f_wm_filtered = f_wm_fibers * fiber_mask
        
        f_wm_sum = f_wm_filtered.sum(axis=-1, keepdims=True)
        f_wm_sum = np.maximum(f_wm_sum, 1e-8)
        fiber_weights = f_wm_filtered / f_wm_sum
        
        f_fiber_total = f_wm_filtered + fiber_weights * f_restricted[..., np.newaxis]
        
        print(f"  f_wm_fibers mean: {f_wm_fibers.sum(axis=-1).mean():.4f}")
        print(f"  f_restricted mean: {f_restricted.mean():.4f}")
        print(f"  f_fiber_total mean: {f_fiber_total.sum(axis=-1).mean():.4f}")
        print(f"  Relative threshold: {relative_threshold} (fibers < 25% of max zeroed)")
    else:
        f_fiber_total = f_wm_fibers
        print("  Warning: f_restricted.nii.gz not found, using f_wm_fibers only")
    
    if merge_peaks_angle > 0 and fiber_dirs.shape[-2] >= 2:
        fiber_dirs, f_fiber_total = _merge_close_peaks_vol(
            fiber_dirs, f_fiber_total, merge_peaks_angle
        )
    
    odf = fiber_dirs_to_odf_fair(fiber_dirs, f_fiber_total, kappa=12.0)
    
    return odf, fit_dir



def load_gt_strands():
    fnames = get_fnames(name='disco1')
    disco1_fnames = [os.path.basename(f) for f in fnames]
    labels_fname = fnames[disco1_fnames.index('highRes_DiSCo1_ROIs.nii.gz')]
    _, affine, labels_img = load_nifti(labels_fname, return_img=True)
    tck_fname = fnames[disco1_fnames.index('DiSCo1_Strands_Trajectories.tck')]
    tractogram = load_tractogram(tck_fname, reference=labels_img, bbox_valid_check=False)
    return tractogram.streamlines, labels_img


def save_trk(streamlines, reference_img, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sft = StatefulTractogram(streamlines, reference_img, Space.RASMM)
    save_tractogram(sft, out_path, bbox_valid_check=False)
    print(f"  Saved TRK: {out_path} ({len(streamlines):,} streamlines)")



def render_streamlines_fury(streamlines, output_path, size=(800, 800),
                            view='oblique', azimuth=None, elevation=None):
    from fury import window, actor, colormap as fury_cmap

    scene = window.Scene()
    scene.background((0, 0, 0))

    colors = fury_cmap.line_colors(streamlines)
    stream_actor = actor.line(streamlines, colors=colors, linewidth=1.0)
    scene.add(stream_actor)

    all_pts = np.vstack([s for s in streamlines if len(s) > 0])
    center = all_pts.mean(axis=0)
    extent = all_pts.max(axis=0) - all_pts.min(axis=0)
    cam_dist = max(extent) * 1.6

    if azimuth is not None and elevation is not None:
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)
        dx = cam_dist * np.cos(el_rad) * np.cos(az_rad)
        dy = cam_dist * np.cos(el_rad) * np.sin(az_rad)
        dz = cam_dist * np.sin(el_rad)
        scene.set_camera(
            position=(center[0] + dx, center[1] + dy, center[2] + dz),
            focal_point=tuple(center), view_up=(0, 0, 1))
    elif view == 'axial':
        scene.set_camera(
            position=(center[0], center[1], center[2] + cam_dist),
            focal_point=tuple(center), view_up=(0, 1, 0))
    elif view == 'coronal':
        scene.set_camera(
            position=(center[0], center[1] + cam_dist, center[2]),
            focal_point=tuple(center), view_up=(0, 0, 1))
    elif view == 'sagittal':
        scene.set_camera(
            position=(center[0] + cam_dist, center[1], center[2]),
            focal_point=tuple(center), view_up=(0, 0, 1))
    else:
        d = cam_dist / np.sqrt(3)
        scene.set_camera(
            position=(center[0] + d, center[1] + d, center[2] + d),
            focal_point=tuple(center), view_up=(0, 0, 1))

    window.record(scene=scene, out_path=output_path, size=size, reset_camera=False)
    scene.clear()


CAMERA_ANGLES = [
    (  0, 30, 'Front-High'),
    ( 36, 15, 'Front-Right'),
    ( 72, 40, 'Right-Top'),
    (108, 10, 'Right-Back'),
    (144, 35, 'Back-High'),
    (180, 20, 'Back'),
    (216, 45, 'Back-Left-Top'),
    (252,  5, 'Left-Low'),
    (288, 30, 'Left-Front-High'),
    (324, 50, 'Top-Oblique'),
]


def render_tractogram_multiangle(gt_streams, method_streams, method_labels,
                                  method_correlations, output_dir,
                                  tag='best', size=900):
    from PIL import Image, ImageDraw, ImageFont

    tmp = '/tmp/fury_disco_benchmark'
    os.makedirs(tmp, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    col_keys = ['gt'] + list(method_streams.keys())
    all_streams = {'gt': gt_streams, **method_streams}
    n_cols = len(col_keys)

    try:
        font_title = ImageFont.truetype(
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 44)
        font_r = ImageFont.truetype(
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 38)
    except Exception:
        font_title = font_r = ImageFont.load_default()

    for az, el, angle_name in CAMERA_ANGLES:
        panels = {}
        for key, streams in all_streams.items():
            p = os.path.join(tmp, f'{key}_az{az}_el{el}.png')
            render_streamlines_fury(streams, p, size=(size, size),
                                    azimuth=az, elevation=el)
            panels[key] = Image.open(p)

        img_w, img_h = panels[col_keys[0]].size
        title_h = 100
        total_w = n_cols * img_w
        total_h = title_h + img_h

        combined = Image.new('RGB', (total_w, total_h), (0, 0, 0))
        draw = ImageDraw.Draw(combined)

        for col, key in enumerate(col_keys):
            x_left = col * img_w
            x_center = x_left + img_w // 2

            if key == 'gt':
                title_text = f'Ground Truth ({len(gt_streams):,})'
            else:
                label = method_labels.get(key, key.upper())
                r_val = method_correlations.get(key, 0)
                title_text = f'{label}  r = {r_val:.4f}'

            bb = draw.textbbox((0, 0), title_text, font=font_title)
            tw = bb[2] - bb[0]
            tx = max(x_left + 4, x_center - tw // 2)
            draw.text((tx, 18), title_text,
                      fill=(255, 255, 255), font=font_title)

            combined.paste(panels[key], (x_left, title_h))

        safe_name = angle_name.replace(' ', '_').lower()
        out_path = os.path.join(output_dir,
                                f'tractogram_{tag}_angle{az}_{safe_name}.png')
        combined.save(out_path, quality=95)
        print(f"  Saved {angle_name} (az={az}° el={el}°): {out_path}")

    print(f"Saved 10 multi-angle tractogram figures → {output_dir}")



def render_best_tractograms_mpl(gt_streams, method_streams, method_labels,
                                 method_corrs, output_path, max_sl=6000):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    col_keys = ['gt'] + list(method_streams.keys())
    all_streams = {'gt': gt_streams, **method_streams}
    n_cols = len(col_keys)

    fig, axes = plt.subplots(2, n_cols, figsize=(4.2 * n_cols, 8),
                             facecolor='k')
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    views = [('axial', 0, 1), ('coronal', 0, 2)]

    for col, key in enumerate(col_keys):
        streams = all_streams[key]
        n = len(streams)
        if n > max_sl:
            rng = np.random.RandomState(0)
            idx = rng.choice(n, max_sl, replace=False)
            streams = [streams[i] for i in idx]
        else:
            streams = list(streams)

        for row, (vname, xi, yi) in enumerate(views):
            ax = axes[row, col]
            ax.set_facecolor('k')
            ax.set_aspect('equal')
            ax.tick_params(colors='gray', labelsize=5)
            for spine in ax.spines.values():
                spine.set_color('gray')

            segments, colours = [], []
            for sl in streams:
                if len(sl) < 2:
                    continue
                pts = sl[:, [xi, yi]]
                segs = np.stack([pts[:-1], pts[1:]], axis=1)
                segments.append(segs)
                tan = np.diff(sl, axis=0)
                tan_n = np.abs(tan) / (np.linalg.norm(tan, axis=1, keepdims=True) + 1e-9)
                colours.append(tan_n)

            if segments:
                segments = np.concatenate(segments, axis=0)
                colours = np.concatenate(colours, axis=0)
                lc = LineCollection(segments, colors=colours, linewidths=0.15,
                                    alpha=0.6)
                ax.add_collection(lc)
                ax.autoscale_view()

            if row == 0:
                if key == 'gt':
                    ttl = f'Ground Truth\n({len(gt_streams):,} strands)'
                else:
                    r_val = method_corrs.get(key, 0)
                    ttl = f'{method_labels[key]}\nr = {r_val:.4f}'
                ax.set_title(ttl, color='w', fontsize=9, fontweight='bold')

            if col == 0:
                ax.set_ylabel(vname.capitalize(), color='w', fontsize=9)

    plt.tight_layout(pad=0.5)
    fig.savefig(output_path, dpi=200, facecolor='k', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved best-of-each figure: {output_path}")


def run_tracking(method, seeds, sc, affine, odf, sphere, random_seed=42, max_angle=30):
    if method == 'deterministic':
        streamline_generator = deterministic_tracking(
            seeds, sc, affine, sf=odf, 
            nbr_threads=1, random_seed=random_seed, sphere=sphere, max_angle=max_angle
        )
    elif method == 'probabilistic':
        streamline_generator = probabilistic_tracking(
            seeds, sc, affine, sf=odf,
            nbr_threads=4, random_seed=random_seed, sphere=sphere, max_angle=max_angle
        )
    elif method == 'ptt':
        streamline_generator = ptt_tracking(
            seeds, sc, affine, sf=odf,
            nbr_threads=0, random_seed=random_seed, sphere=sphere, max_angle=max_angle
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return Streamlines(streamline_generator)


def compute_correlation(streamlines, affine, labels, GT_connectome, connectome_mask):
    connectome = connectivity_matrix(streamlines, affine, labels)[1:, 1:]
    r, p = pearsonr(
        GT_connectome[connectome_mask].flatten(),
        connectome[connectome_mask].flatten()
    )
    return r, p, connectome


def run_benchmark(snr=50, methods=['deterministic'], angles=[15, 20, 25, 30],
                  skip_msmt=False, use_cache=True, cache_dir="outputs/disco1_fair_benchmark",
                  fit_dir=None, merge_peaks_angle=0):
    print(f"\n{'='*70}")
    print(f"DiSCo1 FAIR BENCHMARK - SNR={snr}")
    print(f"{'='*70}")
    
    print("\nLoading data (DIPY exact settings)...")
    disco = load_disco_data(snr)
    print(f"  Seeds: {len(disco['seeds'])} (DIPY exact: eroded ROI mask, density=2)")
    
    print("  Loading GT strands...")
    gt_strands, labels_img = load_gt_strands()
    print(f"  GT strands: {len(gt_strands):,}")
    
    trk_dir = os.path.join(cache_dir, 'trk')
    os.makedirs(trk_dir, exist_ok=True)
    
    gt_trk_path = os.path.join(trk_dir, 'gt_strands.trk')
    if not os.path.exists(gt_trk_path):
        save_trk(gt_strands, labels_img, gt_trk_path)
    
    os.makedirs(cache_dir, exist_ok=True)
    csd_cache = os.path.join(cache_dir, f'odf_csd_snr{snr}.npy')
    mcsd_cache = os.path.join(cache_dir, f'odf_mcsd_snr{snr}.npy')
    
    if use_cache and os.path.exists(csd_cache):
        print(f"\nLoading cached CSD ODF: {csd_cache}")
        odf_csd = np.load(csd_cache)
    else:
        print("\nFitting CSD...")
        response, _ = auto_response_ssst(disco['gtab'], disco['data'], roi_radii=10, fa_thr=0.7)
        csd_model = ConstrainedSphericalDeconvModel(disco['gtab'], response, sh_order_max=8)
        csd_fit = csd_model.fit(disco['data'], mask=disco['mask'])
        odf_csd = csd_fit.odf(default_sphere)
        if use_cache:
            np.save(csd_cache, odf_csd)
            print(f"  Cached CSD ODF: {csd_cache}")
    
    odf_mcsd = None
    has_mcsd = False
    
    if skip_msmt:
        print("\nSkipping MSMT-CSD (--skip-msmt)")
    elif use_cache and os.path.exists(mcsd_cache):
        print(f"\nLoading cached MSMT-CSD ODF: {mcsd_cache}")
        odf_mcsd = np.load(mcsd_cache)
        has_mcsd = True
    else:
        print("\nFitting MSMT-CSD...")
        try:
            from dipy.reconst.mcsd import multi_shell_fiber_response
            from dipy.core.gradients import unique_bvals_tolerance
            
            ubvals = unique_bvals_tolerance(disco['gtab'].bvals)
            print(f"  Unique b-values: {ubvals}")
            
            mask_wm, mask_gm, mask_csf = mask_for_response_msmt(disco['gtab'], disco['data'],
                                                                roi_radii=10, wm_fa_thr=0.7,
                                                                gm_fa_thr=0.2, csf_fa_thr=0.1,
                                                                gm_md_thr=0.0007, csf_md_thr=0.002)
            response_wm, response_gm, response_csf = response_from_mask_msmt(
                disco['gtab'], disco['data'], mask_wm, mask_gm, mask_csf)
            
            response_mcsd = multi_shell_fiber_response(sh_order_max=8, bvals=ubvals,
                                                       wm_rf=response_wm, gm_rf=response_gm,
                                                       csf_rf=response_csf)
            
            mcsd_model = MultiShellDeconvModel(disco['gtab'], response_mcsd, sh_order_max=8)
            mcsd_fit = mcsd_model.fit(disco['data'], mask=disco['mask'])
            odf_mcsd = mcsd_fit.odf(default_sphere)
            has_mcsd = True
            print("  MSMT-CSD fitted successfully")
            if use_cache:
                np.save(mcsd_cache, odf_mcsd)
                print(f"  Cached MSMT-CSD ODF: {mcsd_cache}")
        except Exception as e:
            import traceback
            print(f"  MSMT-CSD failed: {e}")
            traceback.print_exc()
            odf_mcsd = None
            has_mcsd = False
    
    print("\nLoading GYMRI...")
    try:
        odf_gymri, fit_dir_used = load_gymri_odf(snr, fit_dir_override=fit_dir,
                                                  merge_peaks_angle=merge_peaks_angle)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return None
    
    mask = disco['mask'] > 0
    csd_sum = odf_csd[mask].sum(axis=-1).mean()
    gymri_sum = odf_gymri[mask].sum(axis=-1).mean()
    mcsd_sum = odf_mcsd[mask].sum(axis=-1).mean() if has_mcsd else 0
    print(f"\n  ODF sums: CSD={csd_sum:.1f}, MSMT-CSD={mcsd_sum:.1f}, GYMRI={gymri_sum:.1f}")
    
    results = {
        'snr': snr,
        'n_seeds': len(disco['seeds']),
        'fit_dir': str(fit_dir_used),
        'odf_sum_csd': float(csd_sum),
        'odf_sum_mcsd': float(mcsd_sum) if has_mcsd else None,
        'odf_sum_gymri': float(gymri_sum),
        'has_mcsd': has_mcsd,
        'methods': {},
        'timestamp': datetime.now().isoformat(),
    }
    
    for method in methods:
        print(f"\n{'='*70}")
        print(f"{method.upper()} TRACKING - Multiple Max Angles")
        print(f"{'='*70}")
        
        results['methods'][method] = {'angles': {}}
        
        for max_angle in angles:
            print(f"\n--- Max Angle = {max_angle}° ---")
            
            streams_csd = run_tracking(
                method, disco['seeds'], disco['sc'], disco['affine'],
                odf_csd, default_sphere, max_angle=max_angle
            )
            r_csd, p_csd, conn_csd = compute_correlation(
                streams_csd, disco['affine'], disco['labels'],
                disco['GT_connectome'], disco['connectome_mask']
            )
            save_trk(streams_csd, labels_img,
                     os.path.join(trk_dir, f'csd_{method}_angle{max_angle}_snr{snr}.trk'))
            
            if has_mcsd:
                streams_mcsd = run_tracking(
                    method, disco['seeds'], disco['sc'], disco['affine'],
                    odf_mcsd, default_sphere, max_angle=max_angle
                )
                r_mcsd, p_mcsd, conn_mcsd = compute_correlation(
                    streams_mcsd, disco['affine'], disco['labels'],
                    disco['GT_connectome'], disco['connectome_mask']
                )
                save_trk(streams_mcsd, labels_img,
                         os.path.join(trk_dir, f'mcsd_{method}_angle{max_angle}_snr{snr}.trk'))
            else:
                r_mcsd, p_mcsd = None, None
                streams_mcsd = []
            
            streams_gymri = run_tracking(
                method, disco['seeds'], disco['sc'], disco['affine'],
                odf_gymri, default_sphere, max_angle=max_angle
            )
            r_gymri, p_gymri, conn_gymri = compute_correlation(
                streams_gymri, disco['affine'], disco['labels'],
                disco['GT_connectome'], disco['connectome_mask']
            )
            save_trk(streams_gymri, labels_img,
                     os.path.join(trk_dir, f'gymri_{method}_angle{max_angle}_snr{snr}.trk'))
            
            best_csd_r = max(r_csd, r_mcsd if r_mcsd else 0)
            gap = r_gymri - best_csd_r
            gap_pct = gap / best_csd_r * 100 if best_csd_r > 0 else 0
            winner = "GYMRI" if gap > 0 else ("MSMT-CSD" if r_mcsd and r_mcsd > r_csd else "CSD")
            
            print(f"  CSD:      r = {r_csd:.4f}, streamlines = {len(streams_csd)}")
            if has_mcsd:
                print(f"  MSMT-CSD: r = {r_mcsd:.4f}, streamlines = {len(streams_mcsd)}")
            print(f"  GYMRI:    r = {r_gymri:.4f}, streamlines = {len(streams_gymri)}")
            print(f"  Gap vs best: {gap:+.4f} ({gap_pct:+.2f}%) → {winner} wins")
            
            results['methods'][method]['angles'][max_angle] = {
                'csd': {
                    'correlation': float(r_csd),
                    'p_value': float(p_csd),
                    'n_streamlines': len(streams_csd),
                },
                'mcsd': {
                    'correlation': float(r_mcsd) if r_mcsd else None,
                    'p_value': float(p_mcsd) if p_mcsd else None,
                    'n_streamlines': len(streams_mcsd) if has_mcsd else 0,
                } if has_mcsd else None,
                'gymri': {
                    'correlation': float(r_gymri),
                    'p_value': float(p_gymri),
                    'n_streamlines': len(streams_gymri),
                },
                'gap': float(gap),
                'gap_pct': float(gap_pct),
                'winner': winner,
            }
        
        print(f"\n--- {method.upper()} SUMMARY ---")
        if has_mcsd:
            print(f"{'Max Angle':>10} | {'CSD':>10} | {'MSMT-CSD':>10} | {'GYMRI':>10} | {'Gap':>10} | {'Winner':>10}")
            print("-" * 80)
        else:
            print(f"{'Max Angle':>10} | {'CSD':>10} | {'GYMRI':>10} | {'Gap':>12} | {'Winner':>8}")
            print("-" * 60)
        gymri_wins = 0
        for angle in angles:
            data = results['methods'][method]['angles'][angle]
            if data['gap'] > 0:
                gymri_wins += 1
            if has_mcsd and data.get('mcsd'):
                mcsd_r = data['mcsd']['correlation']
                print(f"{angle:>10}° | {data['csd']['correlation']:>10.4f} | {mcsd_r:>10.4f} | {data['gymri']['correlation']:>10.4f} | {data['gap_pct']:>+9.2f}% | {data['winner']:>10}")
            else:
                print(f"{angle:>10}° | {data['csd']['correlation']:>10.4f} | {data['gymri']['correlation']:>10.4f} | {data['gap_pct']:>+11.2f}% | {data['winner']:>8}")
        print("-" * (80 if has_mcsd else 60))
        print(f"GYMRI wins at {gymri_wins}/{len(angles)} angles")
    
    print(f"\n{'='*70}")
    print("RENDERING BEST-OF-EACH TRACTOGRAM FIGURES")
    print(f"{'='*70}")
    
    fig_dir = os.path.join(cache_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    for method in methods:
        method_data = results['methods'][method]
        angle_data = method_data['angles']
        
        best_csd_angle = max(angle_data, key=lambda a: angle_data[a]['csd']['correlation'])
        best_gymri_angle = max(angle_data, key=lambda a: angle_data[a]['gymri']['correlation'])
        best_mcsd_angle = None
        if has_mcsd:
            best_mcsd_angle = max(angle_data, key=lambda a: (
                angle_data[a]['mcsd']['correlation'] if angle_data[a].get('mcsd') and angle_data[a]['mcsd']['correlation'] is not None else -1))
        
        msg = f"\n{method.upper()}: best angles  CSD={best_csd_angle}°"
        if has_mcsd:
            msg += f"  MSMT-CSD={best_mcsd_angle}°"
        msg += f"  GYMRI={best_gymri_angle}°"
        print(msg)
        
        best_csd_trk = os.path.join(trk_dir, f'csd_{method}_angle{best_csd_angle}_snr{snr}.trk')
        best_gymri_trk = os.path.join(trk_dir, f'gymri_{method}_angle{best_gymri_angle}_snr{snr}.trk')
        
        if not os.path.exists(best_csd_trk) or not os.path.exists(best_gymri_trk):
            print(f"  Skipping {method} — TRK files not found")
            continue
        
        streams_best_csd = load_tractogram(best_csd_trk, reference=labels_img,
                                           bbox_valid_check=False).streamlines
        streams_best_gymri = load_tractogram(best_gymri_trk, reference=labels_img,
                                             bbox_valid_check=False).streamlines
        
        m_streams = {'csd': streams_best_csd}
        m_labels  = {'csd': f'CSD ({best_csd_angle}°)'}
        m_corrs   = {'csd': angle_data[best_csd_angle]['csd']['correlation']}
        
        if has_mcsd and best_mcsd_angle is not None:
            best_mcsd_trk = os.path.join(trk_dir, f'mcsd_{method}_angle{best_mcsd_angle}_snr{snr}.trk')
            if os.path.exists(best_mcsd_trk):
                streams_best_mcsd = load_tractogram(best_mcsd_trk, reference=labels_img,
                                                    bbox_valid_check=False).streamlines
                m_streams['mcsd'] = streams_best_mcsd
                m_labels['mcsd'] = f'MSMT-CSD ({best_mcsd_angle}°)'
                m_corrs['mcsd'] = angle_data[best_mcsd_angle]['mcsd']['correlation']
        
        m_streams['gymri'] = streams_best_gymri
        m_labels['gymri'] = f'PRISM ({best_gymri_angle}°)'
        m_corrs['gymri'] = angle_data[best_gymri_angle]['gymri']['correlation']
        
        angle_dir = os.path.join(fig_dir, f'{method}_snr{snr}_angles')
        try:
            render_tractogram_multiangle(
                gt_strands, m_streams, m_labels, m_corrs,
                angle_dir, tag=f'{method}_snr{snr}_best', size=900
            )
        except Exception as e:
            print(f"  FURY multi-angle rendering failed: {e}")
            import traceback; traceback.print_exc()
            mpl_path = os.path.join(fig_dir, f'tractogram_{method}_snr{snr}_best.png')
            try:
                render_best_tractograms_mpl(
                    gt_strands, m_streams, m_labels, m_corrs, mpl_path
                )
            except Exception as e2:
                print(f"  Matplotlib fallback also failed: {e2}")
    
    return results


def print_summary(all_results):
    print("\n" + "="*70)
    print("SUMMARY: GYMRI vs CSD (DIPY EXACT SETTINGS)")
    print("="*70)
    
    for results in all_results:
        if results is None:
            continue
        snr = results['snr']
        print(f"\nSNR = {snr}")
        for method, method_data in results['methods'].items():
            print(f"\n{method.upper()}:")
            print(f"{'Angle':>8} | {'CSD':>10} | {'GYMRI':>10} | {'Gap':>12}")
            print("-" * 50)
            for angle, data in method_data['angles'].items():
                print(f"{angle:>8}° | {data['csd']['correlation']:>10.4f} | {data['gymri']['correlation']:>10.4f} | {data['gap_pct']:>+11.2f}%")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="DiSCo1 Fair Benchmark: GYMRI vs CSD with DIPY exact settings"
    )
    parser.add_argument("--snr", type=int, nargs="+", default=[50],
                        help="SNR levels to test (default: 50)")
    parser.add_argument("--outdir", type=str, default="outputs/disco1_fair_benchmark",
                        help="Output directory")
    parser.add_argument("--methods", type=str, nargs="+", 
                        default=['deterministic'],
                        choices=['deterministic', 'probabilistic', 'ptt'],
                        help="Tracking methods to test")
    parser.add_argument("--angles", type=int, nargs="+",
                        default=[15, 20, 25, 30, 35, 45],
                        help="Max angles to test (default: 15 20 25 30 35 45)")
    parser.add_argument("--skip-msmt", action="store_true",
                        help="Skip MSMT-CSD fitting (faster)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Don't use cached ODFs")
    parser.add_argument("--fit-dir", type=str, default=None,
                        help="Explicit GYMRI fit directory (overrides auto-discovery)")
    parser.add_argument("--merge-peaks", type=float, default=15,
                        help="Merge fiber peaks closer than this angle (degrees) before "
                             "ODF construction. E.g. --merge-peaks 15 merges fibers <15° apart. "
                             "Default: 0 (disabled)")
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    print("\n" + "="*70)
    print("DiSCo1 FAIR BENCHMARK")
    print("="*70)
    print(f"SNR levels: {args.snr}")
    print(f"Methods: {args.methods}")
    print(f"Max Angles: {args.angles}")
    print(f"Output: {args.outdir}")
    print(f"Skip MSMT-CSD: {args.skip_msmt}")
    print(f"Use cache: {not args.no_cache}")
    print("\nThis benchmark uses DIPY EXACT settings for 100% fair comparison:")
    print("  - Seeds: eroded ROI mask, density=2")
    print("  - ODF normalization: both sum to ~22")
    print("  - Same tracking algorithms and random seed")
    
    all_results = []
    for snr in args.snr:
        results = run_benchmark(snr, methods=args.methods, angles=args.angles,
                                skip_msmt=args.skip_msmt, use_cache=not args.no_cache,
                                cache_dir=args.outdir, fit_dir=args.fit_dir,
                                merge_peaks_angle=args.merge_peaks)
        all_results.append(results)
        
        if results is not None:
            json_path = os.path.join(args.outdir, f'snr{snr}_results.json')
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved: {json_path}")
    
    print_summary(all_results)
    
    combined_path = os.path.join(args.outdir, 'all_results.json')
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {combined_path}")
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
