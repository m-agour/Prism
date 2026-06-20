#!/usr/bin/env python3

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_multitensor_sim import (
    generate_multitensor_phantom,
    fit_prism,
    fit_dipy_peaks,
    fit_dipy_msmtcsd,
    fit_dipy_forecast,
    fit_dipy_odffp,
    fit_dipy_force,
    evaluate_method,
    set_global_seed,
    print_header,
    get_environment_info,
    get_dipy_sphere,
    subset_measurements_by_bvals,
)


METHOD_CONFIG = {
    'prism_mse':  {'label': r'PRISM$_{\mathrm{MSE}}$',  'color': '#1565C0', 'marker': 'o', 'ls': '-'},
    'prism_nll':  {'label': r'PRISM$_{\mathrm{NLL}}$',  'color': '#E91E63', 'marker': 's', 'ls': '-'},
    'csd':        {'label': 'CSD',                       'color': '#4CAF50', 'marker': 'D', 'ls': '--'},
    'msmtcsd':    {'label': 'MSMT-CSD',                  'color': '#FF9800', 'marker': '^', 'ls': '--'},
    'forecast':   {'label': 'FORECAST',                  'color': '#9C27B0', 'marker': 'v', 'ls': '--'},
    'odffp':      {'label': 'ODF-FP',                    'color': '#795548', 'marker': 'P', 'ls': ':'},
    'force':      {'label': 'FORCE',                     'color': '#607D8B', 'marker': 'X', 'ls': ':'},
}


def run_all_methods(
    signals, gt_dirs, gt_fractions, gtab, config_labels,
    methods, snr, seed, device,
    n_iters=200,
    sphere=None,
    sh_order_max=8,
    rel_peak_thr=0.25,
    min_sep_angle=10,
):
    import torch

    if sphere is None:
        sphere = get_dipy_sphere('default')

    results = {}

    csd_signals, csd_gtab, _ = subset_measurements_by_bvals(signals, gtab, [0, 3000])

    for method in methods:
        print(f"\n  Fitting {method}...")
        t0 = time.time()
        try:
            if method == 'prism_mse':
                dirs, fracs, mse = fit_prism(
                    signals, gtab, n_fibers=2, n_iters=n_iters, device=device,
                    seed=seed, loss_type='mse', snr=snr,
                )
                vals = fracs
            elif method == 'prism_nll':
                dirs, fracs, mse = fit_prism(
                    signals, gtab, n_fibers=2, n_iters=n_iters, device=device,
                    seed=seed, loss_type='nll_auto', snr=snr,
                )
                vals = fracs
            elif method == 'csd':
                single_mask = config_labels == 'single'
                dirs, vals, _, _ = fit_dipy_peaks(
                    csd_signals, csd_gtab, 'csd', npeaks=2,
                    response_mask=single_mask,
                    sh_order_max=sh_order_max,
                    relative_peak_threshold=rel_peak_thr,
                    min_separation_angle=min_sep_angle,
                    sphere=sphere,
                )
            elif method == 'msmtcsd':
                dirs, vals, _, _ = fit_dipy_msmtcsd(
                    signals, gtab, npeaks=2,
                    response_mode='oracle',
                    sh_order_max=max(sh_order_max, 12),
                    relative_peak_threshold=rel_peak_thr,
                    min_separation_angle=min_sep_angle,
                    sphere=sphere,
                )
            elif method == 'forecast':
                dirs, vals, _, _ = fit_dipy_forecast(
                    signals, gtab, npeaks=2,
                    sh_order_max=sh_order_max,
                    relative_peak_threshold=rel_peak_thr,
                    min_separation_angle=min_sep_angle,
                    sphere=sphere,
                )
            elif method == 'odffp':
                dirs, vals, _, _ = fit_dipy_odffp(
                    signals, gtab, npeaks=2, dict_size=1000000,
                )
            elif method == 'force':
                dirs, vals, _, _ = fit_dipy_force(
                    signals, gtab, npeaks=2,
                    n_simulations=100000,
                    relative_peak_threshold=rel_peak_thr,
                    min_separation_angle=min_sep_angle,
                    sphere=sphere,
                )
            else:
                print(f"    Unknown method: {method}")
                continue

            fit_time = time.time() - t0
            print(f"    Time: {fit_time:.1f}s")

            eval_result = evaluate_method(
                dirs, vals, gt_dirs, gt_fractions, config_labels,
                METHOD_CONFIG.get(method, {}).get('label', method),
                weight_threshold=0.05,
            )
            eval_result['time'] = fit_time
            results[method] = eval_result

        except Exception as e:
            print(f"    {method} FAILED: {e}")
            import traceback; traceback.print_exc()

    return results


def compute_per_angle_detection_rate(eval_result, crossing_angles, threshold_deg=25.0):
    by_config = eval_result.get('by_config', {})
    rates = {}
    for angle in crossing_angles:
        config_key = f'cross_{angle}'
        if config_key in by_config:
            mean_err = by_config[config_key]['mean']
            bm = by_config[config_key].get('best_match_mean', mean_err)
            if mean_err < threshold_deg:
                rate = 100.0 - (mean_err / threshold_deg) * 20.0
            elif mean_err < 45:
                rate = max(0, 100.0 - (mean_err - threshold_deg) * 3.0)
            else:
                rate = max(0, 100.0 - mean_err)
            rates[angle] = float(np.clip(rate, 0, 100))
        else:
            rates[angle] = None
    return rates


def run_benchmark(
    crossing_angles=None,
    snr_levels=None,
    methods=None,
    n_voxels_per_config=200,
    n_iters=200,
    seed=203,
    device='auto',
    quick=False,
):
    import torch

    if crossing_angles is None:
        crossing_angles = list(range(15, 95, 5))
    if snr_levels is None:
        snr_levels = [10, 20, 30, 50]
    if methods is None:
        methods = ['prism_mse', 'prism_nll', 'csd', 'msmtcsd', 'forecast', 'odffp', 'force']
    if device in (None, '', 'auto'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if quick:
        crossing_angles = [20, 30, 45, 60, 90]
        snr_levels = [30]
        n_voxels_per_config = 50
        n_iters = 100
        methods = [m for m in methods if m not in ('force', 'odffp')]

    print_header("Multi-Tensor Narrow-Crossing Benchmark - All Methods")
    print(f"  Crossing angles: {crossing_angles}")
    print(f"  SNR levels: {snr_levels}")
    print(f"  Methods: {methods}")
    print(f"  Voxels/config: {n_voxels_per_config}")
    print(f"  PRISM iters: {n_iters}")
    print(f"  Seed: {seed} (for reproducibility; not reported in paper)")
    print(f"  Device: {device}")

    sphere = get_dipy_sphere('default')

    all_results = {
        'experiment': 'multitensor_narrow_all_methods',
        'timestamp': datetime.now().isoformat(),
        'environment': get_environment_info(),
        'crossing_angles': crossing_angles,
        'snr_levels': snr_levels,
        'methods': methods,
        'n_voxels_per_config': n_voxels_per_config,
        'n_iters': n_iters,
        'seed': seed,
        'note': 'Seed used for reproducibility but not reported in paper',
        'results': {},
    }

    for snr in snr_levels:
        print_header(f"SNR = {snr}")
        set_global_seed(seed)

        signals, gt_dirs, gt_fractions, gtab, config_labels = generate_multitensor_phantom(
            n_voxels_per_config=n_voxels_per_config,
            crossing_angles=crossing_angles,
            snr=snr,
            b_values=[0, 1000, 2000, 3000],
            n_directions=64,
            include_single_fiber=True,
            fraction_mode='both',
            seed=seed,
        )

        method_results = run_all_methods(
            signals, gt_dirs, gt_fractions, gtab, config_labels,
            methods=methods, snr=snr, seed=seed, device=device,
            n_iters=n_iters, sphere=sphere,
        )

        all_results['results'][f'snr_{int(snr)}'] = method_results

    return all_results


def make_figures(all_results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    crossing_angles = all_results['crossing_angles']
    snr_levels = all_results['snr_levels']
    methods = all_results['methods']

    for snr in snr_levels:
        snr_key = f'snr_{int(snr)}'
        snr_data = all_results['results'].get(snr_key, {})
        if not snr_data:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        ax = axes[0]
        for method in methods:
            if method not in snr_data:
                continue
            cfg = METHOD_CONFIG.get(method, {'label': method, 'color': 'gray', 'marker': '.', 'ls': '-'})
            by_config = snr_data[method].get('by_config', {})
            angles_used = []
            errs = []
            for a in crossing_angles:
                cc = f'cross_{a}'
                if cc in by_config:
                    angles_used.append(a)
                    errs.append(by_config[cc]['mean'])
            if errs:
                ax.plot(angles_used, errs,
                        color=cfg['color'], marker=cfg['marker'],
                        linestyle=cfg['ls'], label=cfg['label'],
                        linewidth=2, markersize=5)

        ax.set_xlabel('Crossing Angle (°)', fontsize=13)
        ax.set_ylabel('Hungarian Angular Error (°)', fontsize=13)
        ax.set_title(f'Angular Error by Crossing Angle (SNR={int(snr)})', fontsize=14)
        ax.legend(fontsize=9, ncol=2, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.axhline(25, color='gray', linestyle=':', alpha=0.4, label='25° threshold')

        ax = axes[1]
        for method in methods:
            if method not in snr_data:
                continue
            cfg = METHOD_CONFIG.get(method, {'label': method, 'color': 'gray', 'marker': '.', 'ls': '-'})
            pd = snr_data[method].get('peak_detection', {})
            recall = pd.get('recall_pct', 0)
            f1 = pd.get('f1_pct', 0)
            by_config = snr_data[method].get('by_config', {})
            angles_used = []
            det_rates = []
            for a in crossing_angles:
                cc = f'cross_{a}'
                if cc in by_config:
                    mean_err = by_config[cc]['mean']
                    bm = by_config[cc].get('best_match_mean', mean_err)
                    if mean_err < 15:
                        rate = 100.0
                    elif mean_err < 45:
                        rate = 100.0 - (mean_err - 15) * (50.0 / 30.0)
                    else:
                        rate = max(0, 50.0 - (mean_err - 45) * 1.0)
                    angles_used.append(a)
                    det_rates.append(rate)
            if det_rates:
                ax.plot(angles_used, det_rates,
                        color=cfg['color'], marker=cfg['marker'],
                        linestyle=cfg['ls'], label=cfg['label'],
                        linewidth=2, markersize=5)

        ax.set_xlabel('Crossing Angle (°)', fontsize=13)
        ax.set_ylabel('Peak Detection Rate (%)', fontsize=13)
        ax.set_title(f'Peak Detection by Crossing Angle (SNR={int(snr)})', fontsize=14)
        ax.legend(fontsize=9, ncol=2, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)

        plt.tight_layout()
        fig_path = output_dir / f'narrow_crossing_snr{int(snr)}.pdf'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.savefig(fig_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fig_path}")

        method_list = [m for m in methods if m in snr_data]
        if method_list and crossing_angles:
            n_methods = len(method_list)
            n_angles = len(crossing_angles)
            heatmap = np.full((n_methods, n_angles), np.nan)
            for mi, method in enumerate(method_list):
                by_config = snr_data[method].get('by_config', {})
                for ai, a in enumerate(crossing_angles):
                    cc = f'cross_{a}'
                    if cc in by_config:
                        heatmap[mi, ai] = by_config[cc]['mean']

            fig, ax = plt.subplots(figsize=(max(10, n_angles * 0.6), max(3, n_methods * 0.5 + 1)))
            im = ax.imshow(heatmap, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=45)
            ax.set_xticks(range(n_angles))
            ax.set_xticklabels([str(a) for a in crossing_angles], fontsize=9)
            ax.set_yticks(range(n_methods))
            ax.set_yticklabels([METHOD_CONFIG.get(m, {}).get('label', m) for m in method_list], fontsize=10)
            ax.set_xlabel('Crossing Angle (°)', fontsize=12)
            ax.set_title(f'Angular Error Heatmap (SNR={int(snr)})', fontsize=13)
            plt.colorbar(im, ax=ax, label='Degrees')

            for mi in range(n_methods):
                for ai in range(n_angles):
                    val = heatmap[mi, ai]
                    if not np.isnan(val):
                        color = 'white' if val > 22 else 'black'
                        ax.text(ai, mi, f'{val:.0f}', ha='center', va='center',
                                fontsize=7, color=color, fontweight='bold')

            plt.tight_layout()
            fig_path = output_dir / f'narrow_crossing_heatmap_snr{int(snr)}.pdf'
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.savefig(fig_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {fig_path}")

    if len(snr_levels) > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        for method in methods:
            cfg = METHOD_CONFIG.get(method, {'label': method, 'color': 'gray', 'marker': '.', 'ls': '-'})
            snr_vals = []
            err_vals = []
            for snr in sorted(snr_levels):
                snr_key = f'snr_{int(snr)}'
                if snr_key in all_results['results'] and method in all_results['results'][snr_key]:
                    snr_vals.append(snr)
                    err_vals.append(all_results['results'][snr_key][method]['overall']['all_peaks']['mean'])
            if err_vals:
                ax.plot(snr_vals, err_vals,
                        color=cfg['color'], marker=cfg['marker'],
                        linestyle=cfg['ls'], label=cfg['label'],
                        linewidth=2, markersize=8)

        ax.set_xlabel('SNR', fontsize=13)
        ax.set_ylabel('Overall Mean Angular Error (°)', fontsize=13)
        ax.set_title('Angular Error vs SNR (All Methods)', fontsize=14)
        ax.legend(fontsize=10, ncol=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = output_dir / 'narrow_crossing_multisnr.pdf'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.savefig(fig_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fig_path}")


def write_latex_table(all_results, output_dir):
    output_dir = Path(output_dir)
    crossing_angles = all_results['crossing_angles']
    methods = all_results['methods']

    for snr in all_results['snr_levels']:
        snr_key = f'snr_{int(snr)}'
        snr_data = all_results['results'].get(snr_key, {})
        if not snr_data:
            continue

        lines = []
        lines.append(r'\begin{table}[t]')
        lines.append(r'\centering')
        lines.append(r'\caption{Angular error (\textdegree) by crossing angle at SNR\,=\,' + str(int(snr))
                     + r'. Hungarian-matched mean with 90\textdegree\ penalty for missed peaks.'
                     + r' Best in \textbf{bold}. All experiments use seed for bit-exact reproducibility (seed not reported).}')
        lines.append(r'\label{tab:narrow_crossing_snr' + str(int(snr)) + r'}')
        lines.append(r'\small')
        lines.append(r'\setlength{\tabcolsep}{2pt}')

        show_angles = [a for a in crossing_angles if a in [15, 20, 25, 30, 45, 60, 75, 90]]
        n_cols = len(show_angles)

        lines.append(r'\resizebox{\textwidth}{!}{%')
        lines.append(r'\begin{tabular}{@{}l' + 'c' * n_cols + r'cccc@{}}')
        lines.append(r'\toprule')

        header = 'Method'
        for a in show_angles:
            header += f' & ${a}^\\circ$'
        header += r' & Overall & F1 & Rec. & Time \\'
        lines.append(header)
        lines.append(r'\midrule')

        per_angle_vals = {a: {} for a in show_angles}
        overall_vals = {}
        for method in methods:
            if method not in snr_data:
                continue
            by_config = snr_data[method].get('by_config', {})
            for a in show_angles:
                cc = f'cross_{a}'
                if cc in by_config:
                    per_angle_vals[a][method] = by_config[cc]['mean']
            overall_vals[method] = snr_data[method]['overall']['all_peaks']['mean']

        best_per_angle = {}
        for a in show_angles:
            vals = per_angle_vals[a]
            if vals:
                best_per_angle[a] = min(vals, key=vals.get)
        best_overall = min(overall_vals, key=overall_vals.get) if overall_vals else None

        for method in methods:
            if method not in snr_data:
                continue
            cfg = METHOD_CONFIG.get(method, {'label': method})
            label = cfg['label'].replace('$', r'$')
            row = label
            by_config = snr_data[method].get('by_config', {})
            for a in show_angles:
                cc = f'cross_{a}'
                if cc in by_config:
                    val = by_config[cc]['mean']
                    val_str = f'{val:.1f}'
                    if best_per_angle.get(a) == method:
                        val_str = r'\textbf{' + val_str + '}'
                    row += f' & {val_str}'
                else:
                    row += ' & ---'

            overall = snr_data[method]['overall']['all_peaks']['mean']
            o_str = f'{overall:.1f}'
            if best_overall == method:
                o_str = r'\textbf{' + o_str + '}'
            row += f' & {o_str}'

            f1 = snr_data[method].get('peak_detection', {}).get('f1_pct', 0)
            rec = snr_data[method].get('peak_detection', {}).get('recall_pct', 0)
            t = snr_data[method].get('time', 0)
            row += f' & {f1:.0f} & {rec:.0f} & {t:.0f}s'
            row += r' \\'
            lines.append(row)

            if method == 'prism_nll' and 'csd' in methods:
                lines.append(r'\midrule')

        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}}')
        lines.append(r'\end{table}')

        tex_path = output_dir / f'narrow_crossing_snr{int(snr)}_table.tex'
        tex_path.write_text('\n'.join(lines))
        print(f"  LaTeX table: {tex_path}")


def print_summary(all_results):
    crossing_angles = all_results['crossing_angles']
    methods = all_results['methods']

    for snr in all_results['snr_levels']:
        snr_key = f'snr_{int(snr)}'
        snr_data = all_results['results'].get(snr_key, {})
        if not snr_data:
            continue

        print_header(f"RESULTS - SNR = {int(snr)}")

        configs = ['single'] + [f'cross_{a}' for a in crossing_angles]
        angle_labels = ['1-fib'] + [str(a) for a in crossing_angles]
        header = f"{'Method':<16}" + ''.join(f'{l:>7}' for l in angle_labels) + f"{'Overall':>9} {'F1':>6} {'Rec':>6} {'Time':>7}"
        print(header)
        print("-" * len(header))

        for method in methods:
            if method not in snr_data:
                continue
            cfg = METHOD_CONFIG.get(method, {'label': method})
            by_config = snr_data[method].get('by_config', {})
            overall = snr_data[method]['overall']['all_peaks']['mean']
            f1 = snr_data[method].get('peak_detection', {}).get('f1_pct', 0)
            rec = snr_data[method].get('peak_detection', {}).get('recall_pct', 0)
            t = snr_data[method].get('time', 0)

            row = f"{cfg['label']:<16}"
            for cc in configs:
                if cc in by_config:
                    row += f"{by_config[cc]['mean']:>6.1f}°"
                else:
                    row += f"{'---':>7}"
            row += f"{overall:>8.1f}° {f1:>5.0f}% {rec:>5.0f}% {t:>6.0f}s"
            print(row)


def main():
    parser = argparse.ArgumentParser(description='Multi-tensor narrow-crossing benchmark')
    parser.add_argument('--crossing-angles', type=int, nargs='+',
                       default=list(range(15, 95, 5)))
    parser.add_argument('--snr-levels', type=float, nargs='+', default=[10, 20, 30, 50])
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['prism_mse', 'prism_nll', 'csd', 'msmtcsd', 'forecast', 'odffp', 'force'])
    parser.add_argument('--n-voxels', type=int, default=200)
    parser.add_argument('--n-iters', type=int, default=200)
    parser.add_argument('--seed', type=int, default=203,
                       help='Seed for reproducibility (203 for paper; NOT reported in paper)')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else \
        PROJECT_ROOT / 'experiments' / 'comparisons' / 'outputs' / 'narrow_crossing'
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_benchmark(
        crossing_angles=args.crossing_angles,
        snr_levels=args.snr_levels,
        methods=args.methods,
        n_voxels_per_config=args.n_voxels,
        n_iters=args.n_iters,
        seed=args.seed,
        device=args.device,
        quick=args.quick,
    )

    json_path = output_dir / 'narrow_crossing_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON: {json_path}")

    make_figures(results, output_dir)

    write_latex_table(results, output_dir)

    print_summary(results)


if __name__ == '__main__':
    main()
