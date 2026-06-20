#!/usr/bin/env python3

import os
import sys
import json
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.stats import pearsonr

PROJECT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT)



def load_freesurfer_seg(hcp_dir):
    seg_path = os.path.join(hcp_dir, 't1_segmentation', 't1_tissue_gt.nii.gz')
    if not os.path.exists(seg_path):
        raise FileNotFoundError(
            f"FreeSurfer segmentation not found: {seg_path}\n"
            f"Expected t1_tissue_gt.nii.gz with labels 0=WM, 1=GM, 2=CSF"
        )
    
    seg = nib.load(seg_path).get_fdata()
    return {
        'wm': (seg == 0).astype(np.float32),
        'gm': (seg == 1).astype(np.float32),
        'csf': (seg == 2).astype(np.float32),
        'any': (seg >= 0).astype(bool),
        'raw': seg,
    }


def load_prism_fractions(prism_dir, slab_z=None):
    prism_dir = Path(prism_dir)
    
    def _load(name):
        nii = prism_dir / f'{name}.nii.gz'
        npy = prism_dir / f'{name}.npy'
        if nii.exists():
            return nib.load(str(nii)).get_fdata().astype(np.float32)
        elif npy.exists():
            arr = np.load(str(npy)).astype(np.float32)
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]
            return arr
        else:
            raise FileNotFoundError(f"Cannot find {name}.nii.gz or {name}.npy in {prism_dir}")
    
    f_wm = _load('f_wm')
    f_gm = _load('f_gm')
    f_csf = _load('f_csf')
    
    if slab_z is not None:
        z0, z1 = slab_z
        f_wm = f_wm[:, :, z0:z1]
        f_gm = f_gm[:, :, z0:z1]
        f_csf = f_csf[:, :, z0:z1]
    
    return {'wm': f_wm, 'gm': f_gm, 'csf': f_csf}



def dice_score(pred_binary, gt_binary, mask):
    p = pred_binary[mask].astype(bool)
    g = gt_binary[mask].astype(bool)
    intersection = (p & g).sum()
    return 2.0 * intersection / (p.sum() + g.sum() + 1e-8)


def compute_dice_argmax(prism, fs_seg, mask):
    fracs = np.stack([prism['csf'], prism['gm'], prism['wm']], axis=-1)
    pred_label = np.argmax(fracs, axis=-1)
    
    pred_csf = (pred_label == 0).astype(np.float32)
    pred_gm = (pred_label == 1).astype(np.float32)
    pred_wm = (pred_label == 2).astype(np.float32)
    
    d_csf = dice_score(pred_csf, fs_seg['csf'], mask)
    d_gm = dice_score(pred_gm, fs_seg['gm'], mask)
    d_wm = dice_score(pred_wm, fs_seg['wm'], mask)
    
    return {
        'dice_csf': round(float(d_csf), 4),
        'dice_gm': round(float(d_gm), 4),
        'dice_wm': round(float(d_wm), 4),
        'dice_mean_wmgm': round(float(0.5 * (d_wm + d_gm)), 4),
        'dice_mean_all': round(float((d_csf + d_gm + d_wm) / 3), 4),
    }


def compute_dice_threshold_sweep(prism, fs_seg, mask,
                                  thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.20, 0.55, 0.05)
    
    results = {}
    for t in thresholds:
        t_key = f'{t:.2f}'
        results[t_key] = {}
        for tissue in ['wm', 'gm', 'csf']:
            pred = (prism[tissue] >= t).astype(np.float32)
            d = dice_score(pred, fs_seg[tissue], mask)
            results[t_key][tissue] = round(float(d), 4)
        results[t_key]['mean_wmgm'] = round(
            0.5 * (results[t_key]['wm'] + results[t_key]['gm']), 4)
    
    return results


def compute_pearson(prism, fs_seg, mask):
    results = {}
    for tissue in ['wm', 'gm', 'csf']:
        r, p = pearsonr(prism[tissue][mask], fs_seg[tissue][mask])
        results[f'r_{tissue}'] = round(float(r), 4)
        results[f'p_{tissue}'] = float(p)
    results['r_mean_wmgm'] = round(
        0.5 * (results['r_wm'] + results['r_gm']), 4)
    return results



def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PRISM tissue fractions against FreeSurfer T1 segmentations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--prism-dir', type=str, required=True,
                        help='Directory containing PRISM fit results (f_wm.nii.gz, etc.)')
    parser.add_argument('--hcp-dir', type=str,
                        default='/home/moabouag/studies/.assignmentss/dmrl/data/HCP/100307',
                        help='HCP subject directory with t1_segmentation/t1_tissue_gt.nii.gz')
    parser.add_argument('--slab-z', type=int, nargs=2, default=None,
                        help='Z-range of the PRISM slab in full-brain coordinates. '
                             'e.g. --slab-z 62 82 means PRISM covers z=62:82 of the full brain. '
                             'FreeSurfer will be cropped to match.')
    parser.add_argument('--outdir', type=str, default=None,
                        help='Output dir for results JSON (default: same as --prism-dir)')
    
    args = parser.parse_args()
    outdir = args.outdir or args.prism_dir
    os.makedirs(outdir, exist_ok=True)
    
    print("=" * 60)
    print("HCP DICE vs FREESURFER EVALUATION")
    print("=" * 60)
    
    print(f"\n1. Loading FreeSurfer segmentation from {args.hcp_dir}")
    fs = load_freesurfer_seg(args.hcp_dir)
    
    print(f"2. Loading PRISM fractions from {args.prism_dir}")
    prism = load_prism_fractions(args.prism_dir)
    
    mask_path = os.path.join(args.prism_dir, 'brain_mask.nii.gz')
    if os.path.exists(mask_path):
        dwi_mask = nib.load(mask_path).get_fdata().astype(bool)
    else:
        total = prism['wm'] + prism['gm'] + prism['csf']
        dwi_mask = total > 0.1
    
    if args.slab_z:
        z0, z1 = args.slab_z
        fs_seg_slab = {
            'wm': fs['wm'][:, :, z0:z1],
            'gm': fs['gm'][:, :, z0:z1],
            'csf': fs['csf'][:, :, z0:z1],
        }
        print(f"  FreeSurfer cropped to z=[{z0},{z1})")
    else:
        fs_seg_slab = {'wm': fs['wm'], 'gm': fs['gm'], 'csf': fs['csf']}
    
    prism_shape = prism['wm'].shape
    fs_shape = fs_seg_slab['wm'].shape
    if prism_shape != fs_shape:
        print(f"  ⚠ Shape mismatch: PRISM={prism_shape}, FreeSurfer={fs_shape}")
        min_shape = tuple(min(p, f) for p, f in zip(prism_shape, fs_shape))
        for tissue in ['wm', 'gm', 'csf']:
            prism[tissue] = prism[tissue][:min_shape[0], :min_shape[1], :min_shape[2]]
            fs_seg_slab[tissue] = fs_seg_slab[tissue][:min_shape[0], :min_shape[1], :min_shape[2]]
        dwi_mask = dwi_mask[:min_shape[0], :min_shape[1], :min_shape[2]]
    
    fs_labeled = (fs_seg_slab['wm'] + fs_seg_slab['gm'] + fs_seg_slab['csf']) > 0
    joint_mask = dwi_mask & fs_labeled
    
    n_joint = int(joint_mask.sum())
    print(f"  PRISM shape: {prism['wm'].shape}")
    print(f"  FreeSurfer shape: {fs_seg_slab['wm'].shape}")
    print(f"  DWI mask voxels: {dwi_mask.sum():,}")
    print(f"  Joint mask voxels: {n_joint:,}")
    
    print(f"\n3. Computing Dice (argmax)...")
    dice_argmax = compute_dice_argmax(prism, fs_seg_slab, joint_mask)
    print(f"  WM Dice:  {dice_argmax['dice_wm']:.4f}")
    print(f"  GM Dice:  {dice_argmax['dice_gm']:.4f}")
    print(f"  CSF Dice: {dice_argmax['dice_csf']:.4f}")
    print(f"  Mean WM+GM: {dice_argmax['dice_mean_wmgm']:.4f}")
    
    print(f"\n4. Threshold sweep...")
    sweep = compute_dice_threshold_sweep(prism, fs_seg_slab, joint_mask)
    
    best_wm_t = max(sweep.keys(), key=lambda t: sweep[t]['wm'])
    best_gm_t = max(sweep.keys(), key=lambda t: sweep[t]['gm'])
    print(f"  Best WM Dice:  {sweep[best_wm_t]['wm']:.4f} at t={best_wm_t}")
    print(f"  Best GM Dice:  {sweep[best_gm_t]['gm']:.4f} at t={best_gm_t}")
    
    for t_key in sorted(sweep.keys()):
        s = sweep[t_key]
        print(f"    t={t_key}: WM={s['wm']:.4f}  GM={s['gm']:.4f}  CSF={s.get('csf', 'N/A')}")
    
    print(f"\n5. Pearson correlations...")
    pearson = compute_pearson(prism, fs_seg_slab, joint_mask)
    print(f"  r_WM:  {pearson['r_wm']:.4f}")
    print(f"  r_GM:  {pearson['r_gm']:.4f}")
    print(f"  r_CSF: {pearson['r_csf']:.4f}")
    print(f"  r_mean(WM+GM): {pearson['r_mean_wmgm']:.4f}")
    
    results = {
        'prism_dir': str(args.prism_dir),
        'hcp_dir': str(args.hcp_dir),
        'slab_z': args.slab_z,
        'joint_voxels': n_joint,
        'argmax_dice': dice_argmax,
        'threshold_sweep': sweep,
        'best_wm_threshold': best_wm_t,
        'best_wm_dice': sweep[best_wm_t]['wm'],
        'best_gm_threshold': best_gm_t,
        'best_gm_dice': sweep[best_gm_t]['gm'],
        'pearson': pearson,
    }
    
    out_path = os.path.join(outdir, 'dice_vs_freesurfer.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Argmax Dice:  WM={dice_argmax['dice_wm']:.3f}  GM={dice_argmax['dice_gm']:.3f}")
    print(f"  Best Dice:    WM={sweep[best_wm_t]['wm']:.3f} (t={best_wm_t})  "
          f"GM={sweep[best_gm_t]['gm']:.3f} (t={best_gm_t})")
    print(f"  Pearson:      r_WM={pearson['r_wm']:.3f}  r_GM={pearson['r_gm']:.3f}")
    print(f"  Saved: {out_path}")
    print(f"  Done! ✓")


if __name__ == '__main__':
    main()
