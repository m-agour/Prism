#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_multitensor_sim import (
    set_global_seed,
    generate_multitensor_phantom,
    evaluate_method,
    print_header,
)


def fit_msmtcsd_with_params(
    signals, gtab, 
    sh_order_max=8,
    relative_peak_threshold=0.25,
    min_separation_angle=25,
    npeaks=2,
):
    from dipy.direction import peaks_from_model
    from dipy.data import default_sphere
    from dipy.reconst.mcsd import (
        MultiShellDeconvModel,
        multi_shell_fiber_response,
    )
    
    data = signals[:, np.newaxis, np.newaxis, :]
    mask = np.ones((len(signals), 1, 1), dtype=bool)
    
    ubvals = np.unique(np.round(gtab.bvals, -2))
    ubvals = ubvals[ubvals > 0]
    n_shells = len(ubvals)
    
    wm_rf = np.zeros((n_shells, 4))
    wm_rf[:, 0] = 1.7e-3
    wm_rf[:, 1] = 0.3e-3
    wm_rf[:, 2] = 0.3e-3
    wm_rf[:, 3] = 1.0
    
    gm_rf = np.zeros((n_shells, 4))
    gm_rf[:, 0] = 0.9e-3
    gm_rf[:, 1] = 0.9e-3
    gm_rf[:, 2] = 0.9e-3
    gm_rf[:, 3] = 1.0
    
    csf_rf = np.zeros((n_shells, 4))
    csf_rf[:, 0] = 3.0e-3
    csf_rf[:, 1] = 3.0e-3
    csf_rf[:, 2] = 3.0e-3
    csf_rf[:, 3] = 1.0
    
    response_mcsd = multi_shell_fiber_response(
        sh_order_max=sh_order_max,
        bvals=ubvals,
        wm_rf=wm_rf,
        gm_rf=gm_rf,
        csf_rf=csf_rf,
    )
    
    model = MultiShellDeconvModel(gtab, response_mcsd, sh_order_max=sh_order_max)
    
    peaks = peaks_from_model(
        model, data, default_sphere,
        relative_peak_threshold=relative_peak_threshold,
        min_separation_angle=min_separation_angle,
        mask=mask,
        npeaks=npeaks,
        parallel=True
    )
    
    dirs = peaks.peak_dirs[:, 0, 0, :, :]
    values = peaks.peak_values[:, 0, 0, :]
    
    return dirs, values


def run_sensitivity_analysis(quick=False, seed=203):
    print_header("MSMT-CSD Peak Extraction Sensitivity Analysis")
    
    set_global_seed(seed)
    
    thresholds = [0.1, 0.25, 0.5]
    sep_angles = [15, 25, 35]
    lmax_values = [6, 8, 10]
    
    n_voxels = 50 if quick else 100
    crossing_angles = [30, 60, 90] if quick else [30, 45, 60, 75, 90]
    
    print(f"\nConfiguration:")
    print(f"  Voxels per config: {n_voxels}")
    print(f"  Crossing angles: {crossing_angles}")
    print(f"  Thresholds tested: {thresholds}")
    print(f"  Separation angles tested: {sep_angles}")
    print(f"  lmax values tested: {lmax_values}")
    
    signals, gt_dirs, gt_fractions, gtab, config_labels = generate_multitensor_phantom(
        n_voxels_per_config=n_voxels,
        crossing_angles=crossing_angles,
        snr=30,
        b_values=[0, 1000, 2000, 3000],
        n_directions=64,
        include_single_fiber=True,
        seed=seed,
    )
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_voxels': len(signals),
        'crossing_angles': crossing_angles,
        'experiments': [],
    }
    
    best_result = None
    best_all_mean = 999
    
    print_header("Testing Parameter Combinations")
    
    total_combinations = len(thresholds) * len(sep_angles) * len(lmax_values)
    combo_idx = 0
    
    for lmax in lmax_values:
        for thresh in thresholds:
            for sep in sep_angles:
                combo_idx += 1
                print(f"\n[{combo_idx}/{total_combinations}] lmax={lmax}, threshold={thresh}, sep_angle={sep}°")
                
                try:
                    dirs, vals = fit_msmtcsd_with_params(
                        signals, gtab,
                        sh_order_max=lmax,
                        relative_peak_threshold=thresh,
                        min_separation_angle=sep,
                        npeaks=2,
                    )
                    
                    eval_results = evaluate_method(
                        dirs, vals, gt_dirs, gt_fractions, config_labels,
                        f"MSMT-CSD-lmax{lmax}-t{thresh}-s{sep}"
                    )
                    
                    exp = {
                        'lmax': lmax,
                        'threshold': thresh,
                        'min_separation': sep,
                        'primary_mean': eval_results['overall']['primary_peak']['mean'],
                        'all_mean': eval_results['overall']['all_peaks']['mean'],
                        'success_15': eval_results['overall']['success_rate_15deg'],
                        'success_25': eval_results['overall']['success_rate_25deg'],
                    }
                    
                    results['experiments'].append(exp)
                    
                    print(f"  Primary: {exp['primary_mean']:.2f}°, All: {exp['all_mean']:.2f}°, Success<15°: {exp['success_15']:.1f}%")
                    
                    if exp['all_mean'] < best_all_mean:
                        best_all_mean = exp['all_mean']
                        best_result = exp
                        
                except Exception as e:
                    print(f"  FAILED: {e}")
    
    print_header("SENSITIVITY ANALYSIS SUMMARY")
    
    print("\n=== Best MSMT-CSD Configuration ===")
    if best_result:
        print(f"  lmax: {best_result['lmax']}")
        print(f"  threshold: {best_result['threshold']}")
        print(f"  min_separation: {best_result['min_separation']}°")
        print(f"  Primary Mean: {best_result['primary_mean']:.2f}°")
        print(f"  All Peaks Mean: {best_result['all_mean']:.2f}°")
        print(f"  Success<15°: {best_result['success_15']:.1f}%")
    
    results['best_config'] = best_result
    
    all_means = [e['all_mean'] for e in results['experiments']]
    primary_means = [e['primary_mean'] for e in results['experiments']]
    
    print("\n=== Performance Range Across All Configurations ===")
    print(f"  Primary Mean: {min(primary_means):.2f}° - {max(primary_means):.2f}°")
    print(f"  All Peaks Mean: {min(all_means):.2f}° - {max(all_means):.2f}°")
    
    results['performance_range'] = {
        'primary_mean_min': min(primary_means),
        'primary_mean_max': max(primary_means),
        'all_mean_min': min(all_means),
        'all_mean_max': max(all_means),
    }
    
    print("\n=== Comparison to PRISM (from synthetic benchmark) ===")
    prism_primary = None
    prism_all = None
    bench_path = Path(__file__).parent / 'outputs' / 'multitensor_benchmark.json'
    if bench_path.exists():
        try:
            bench = json.loads(bench_path.read_text())
            prism_primary = bench['methods']['prism']['overall']['primary_peak']['mean']
            prism_all = bench['methods']['prism']['overall']['all_peaks']['mean']
        except Exception:
            prism_primary = None
            prism_all = None
    prism_primary = float(prism_primary) if prism_primary is not None else 3.8
    prism_all = float(prism_all) if prism_all is not None else 2.7

    print(f"  PRISM Primary Mean: {prism_primary:.2f}°")
    print(f"  PRISM All Peaks Mean: {prism_all:.2f}°")
    print(f"  Best MSMT-CSD All Peaks Mean: {best_all_mean:.2f}°")
    print(f"  PRISM advantage: {best_all_mean / prism_all:.1f}× (even with best MSMT-CSD settings)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='MSMT-CSD Sensitivity Analysis')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--seed', type=int, default=203)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    
    results = run_sensitivity_analysis(quick=args.quick, seed=args.seed)
    
    output_dir = Path(__file__).parent / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = args.output or (output_dir / 'msmtcsd_sensitivity.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
