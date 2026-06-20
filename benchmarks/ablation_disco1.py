#!/usr/bin/env python3

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

import numpy as np

GYMRI_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = GYMRI_ROOT / "scripts" / "simple_fit_microstructure.py"
OUTDIR_BASE = GYMRI_ROOT / "outputs" / "ablation_disco1"

DIPY_DISCO = Path.home() / ".dipy" / "disco" / "disco_1"
DWI  = DIPY_DISCO / "highRes_DiSCo1_DWI_RicianNoise-snr50.nii.gz"
BVALS = DIPY_DISCO / "DiSCo_gradients.bvals"
BVECS = DIPY_DISCO / "DiSCo_gradients_dipy.bvecs"
MASK  = DIPY_DISCO / "highRes_DiSCo1_mask.nii.gz"

COMMON_FLAGS = [
    "--dwi", str(DWI),
    "--bvals", str(BVALS),
    "--bvecs", str(BVECS),
    "--mask", str(MASK),
    "--all-slices", "--slab-mode", "--slab-size", "40",
    "--n-fibers", "5",
    "--n-steps", "300",
    "--no-early-stop",
    "--seed", "2000",
    "--device", "cuda",
]


FULL_PRISM_OUTDIR = GYMRI_ROOT / "outputs" / "fitting" / "disco1_snr50_defaults_encoder"

CONFIGS = OrderedDict([
    ("vanilla", {
        "desc": "Vanilla (5-fiber, no extras)",
        "flags": [
            "--no-restricted",
            "--no-dispersion",
            "--no-shell-gain", "--no-bias-field", "--no-eddy",
            "--no-topology",
            "--no-spatial-prior",
            "--no-tissue-priors",
            "--sparsity-weight", "0",
        ],
    }),
    ("+restricted", {
        "desc": "+ Restricted compartment",
        "flags": [
            "--use-restricted",
            "--no-dispersion",
            "--no-shell-gain", "--no-bias-field", "--no-eddy",
            "--no-topology",
            "--no-spatial-prior",
            "--no-tissue-priors",
            "--sparsity-weight", "0",
        ],
    }),
    ("+nuisance", {
        "desc": "+ Nuisance calibration (bias + gains)",
        "flags": [
            "--use-restricted",
            "--no-dispersion",
            "--learn-shell-gain", "--learn-bias-field", "--no-eddy",
            "--no-topology",
            "--no-spatial-prior",
            "--no-tissue-priors",
            "--sparsity-weight", "0",
        ],
    }),
    ("full_nll", {
        "desc": "Full PRISM + learned σ (NLL)",
        "flags": [
            "--use-rician",
        ],
        "outdir": str(FULL_PRISM_OUTDIR),
    }),
])


def fit_config(name, cfg, dry_run=False):
    outdir = Path(cfg.get("outdir", str(OUTDIR_BASE / name)))
    
    if (outdir / "fiber_dirs.nii.gz").exists():
        print(f"  ✓ {name:20s} - already fitted at {outdir}")
        return True
    
    cmd = [
        sys.executable, str(SCRIPT),
        *COMMON_FLAGS,
        "--outdir", str(outdir),
        "--name", f"ablation_{name}",
        *cfg["flags"],
    ]
    
    print(f"\n{'='*70}")
    print(f"FITTING: {name} - {cfg['desc']}")
    print(f"{'='*70}")
    print(f"  Output: {outdir}")
    print(f"  Command: {' '.join(cmd)}")
    
    if dry_run:
        print("  [DRY RUN - skipping]")
        return True
    
    os.makedirs(outdir, exist_ok=True)
    
    with open(outdir / "ablation_command.txt", "w") as f:
        f.write(" ".join(cmd) + "\n")
    
    result = subprocess.run(cmd, cwd=str(GYMRI_ROOT))
    
    if result.returncode != 0:
        print(f"  ✗ FAILED (exit code {result.returncode})")
        return False
    
    print(f"  ✓ Done: {outdir}")
    return True


def eval_config(name, cfg, disco_data, angles, merge_peaks_angle=15):
    fit_dir = Path(cfg.get("outdir", str(OUTDIR_BASE / name)))
    
    if not (fit_dir / "fiber_dirs.nii.gz").exists():
        print(f"  ✗ {name:20s} - not fitted yet (run with --fit first)")
        return None
    
    sys.path.insert(0, str(Path(__file__).parent))
    from disco1_fair_benchmark import (
        load_gymri_odf, run_tracking, compute_correlation
    )
    from dipy.data import default_sphere
    
    print(f"\n--- Evaluating: {name} ({cfg['desc']}) ---")
    
    try:
        odf, _ = load_gymri_odf(snr=50, fit_dir_override=str(fit_dir),
                                 merge_peaks_angle=merge_peaks_angle)
    except Exception as e:
        print(f"  ✗ Failed to load ODF: {e}")
        return None
    
    mse_val = None
    summary_path = fit_dir / "summary.json"
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            mse_val = summary.get("final_mse") or summary.get("mse")
        except:
            pass
    
    if mse_val is None:
        loss_path = fit_dir / "loss_curve.npy"
        if loss_path.exists():
            try:
                losses = np.load(str(loss_path))
                mse_val = float(losses[-1])
            except:
                pass
    
    results = {"name": name, "desc": cfg["desc"], "mse": mse_val, "angles": {}}
    
    for angle in angles:
        streamlines = run_tracking(
            "deterministic",
            disco_data["seeds"], disco_data["sc"], disco_data["affine"],
            odf, default_sphere, max_angle=angle
        )
        r, p, conn = compute_correlation(
            streamlines, disco_data["affine"], disco_data["labels"],
            disco_data["GT_connectome"], disco_data["connectome_mask"]
        )
        results["angles"][angle] = {
            "correlation": float(r),
            "p_value": float(p),
            "n_streamlines": len(streamlines),
        }
        print(f"    {angle:3d}° → r = {r:.4f}  ({len(streamlines):,} streamlines)")
    
    best_angle = max(results["angles"], key=lambda a: results["angles"][a]["correlation"])
    best_r = results["angles"][best_angle]["correlation"]
    results["best_angle"] = best_angle
    results["best_r"] = best_r
    
    return results


def print_ablation_table(all_results, angles):
    print(f"\n{'='*90}")
    print("ABLATION RESULTS - DiSCo1 (SNR=50, deterministic tracking)")
    print(f"{'='*90}")
    
    angle_cols = "".join(f" {a:>6d}°" for a in angles)
    header = f"{'Configuration':<35s} {'MSE':>10s}{angle_cols}  {'Best':>8s}"
    print(header)
    print("-" * len(header))
    
    for res in all_results:
        if res is None:
            continue
        
        name = res["name"]
        desc = res["desc"]
        mse_str = f"{res['mse']:.2e}" if res["mse"] is not None else "---"
        
        angle_strs = []
        for a in angles:
            if a in res["angles"]:
                r = res["angles"][a]["correlation"]
                angle_strs.append(f"  {r:.4f}")
            else:
                angle_strs.append(f"     ---")
        
        best_str = f"{res['best_r']:.4f}" if "best_r" in res else "---"
        label = f"{name} ({desc})" if len(name) + len(desc) < 32 else name
        
        row = f"{label:<35s} {mse_str:>10s}{''.join(angle_strs)}  {best_str:>8s}"
        print(row)
    
    print("-" * len(header))
    
    vanilla_res = next((r for r in all_results if r and r["name"] == "vanilla"), None)
    if vanilla_res:
        print(f"\n{'Δ vs vanilla':<35s} {'':>10s}", end="")
        ref_angle = angles[-1] if angles else 25
        for res in all_results:
            if res is None or res["name"] == "vanilla":
                continue
            if ref_angle in res["angles"] and ref_angle in vanilla_res["angles"]:
                delta = res["angles"][ref_angle]["correlation"] - vanilla_res["angles"][ref_angle]["correlation"]
                print(f"  {res['name']}: Δr={delta:+.4f}", end="")
        print()


def main():
    parser = argparse.ArgumentParser(description="PRISM Ablation Study on DiSCo1")
    parser.add_argument("--fit", action="store_true", help="Run fitting for each config")
    parser.add_argument("--eval", action="store_true", help="Evaluate fitted results")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--configs", nargs="*", default=None,
                        help=f"Specific configs to run (default: all). Choices: {list(CONFIGS.keys())}")
    parser.add_argument("--angles", type=int, nargs="+", default=[15, 20, 25, 30],
                        help="Tracking angles to evaluate (default: 15 20 25 30)")
    parser.add_argument("--merge-peaks", type=float, default=15,
                        help="Merge fiber peaks closer than this angle (default: 15)")
    args = parser.parse_args()
    
    if not args.fit and not args.eval:
        parser.print_help()
        print("\nError: specify --fit, --eval, or both.")
        sys.exit(1)
    
    if args.configs:
        selected = OrderedDict()
        for name in args.configs:
            if name not in CONFIGS:
                print(f"Unknown config: {name}. Choose from: {list(CONFIGS.keys())}")
                sys.exit(1)
            selected[name] = CONFIGS[name]
    else:
        selected = CONFIGS
    
    print(f"\n{'='*70}")
    print(f"PRISM ABLATION STUDY - DiSCo1 (SNR=50)")
    print(f"{'='*70}")
    print(f"Configs:  {list(selected.keys())}")
    print(f"Output:   {OUTDIR_BASE}")
    print(f"Fit:      {args.fit}")
    print(f"Eval:     {args.eval}")
    print(f"Angles:   {args.angles}")
    
    if args.fit:
        print(f"\n{'='*70}")
        print("PHASE 1: FITTING")
        print(f"{'='*70}")
        
        for name, cfg in selected.items():
            success = fit_config(name, cfg, dry_run=args.dry_run)
            if not success:
                print(f"\n⚠ Config '{name}' failed. Continuing with remaining configs...")
    
    if args.eval:
        print(f"\n{'='*70}")
        print("PHASE 2: EVALUATION")
        print(f"{'='*70}")
        
        sys.path.insert(0, str(Path(__file__).parent))
        from disco1_fair_benchmark import load_disco_data
        
        print("\nLoading DiSCo1 data...")
        disco = load_disco_data(snr=50)
        print(f"  Seeds: {len(disco['seeds']):,}")
        
        all_results = []
        for name, cfg in selected.items():
            res = eval_config(name, cfg, disco, args.angles,
                             merge_peaks_angle=args.merge_peaks)
            all_results.append(res)
        
        print_ablation_table(all_results, args.angles)
        
        os.makedirs(OUTDIR_BASE, exist_ok=True)
        json_path = OUTDIR_BASE / "ablation_results.json"
        save_results = [r for r in all_results if r is not None]
        with open(json_path, "w") as f:
            json.dump(save_results, f, indent=2)
        print(f"\nSaved: {json_path}")
    
    print(f"\n{'='*70}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
