#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'higgs'


def boot_mean_ci(x, n_boot=5000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    boots = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [alpha/2, 1-alpha/2])
    return float(lo), float(hi)


def main():
    src = DATA / 'higgs_seed_results_v5.csv'
    if not src.exists():
        src = DATA / 'higgs_seed_results.csv'
    df = pd.read_csv(src)

    rows = []
    for protocol, g in df.groupby('protocol'):
        lo_gap, hi_gap = boot_mean_ci(g['optimism_gap'].values, seed=123)
        lo_pr, hi_pr = boot_mean_ci(g['optimism_gap_pr'].values, seed=456)
        rows.append({
            'dataset': 'higgs',
            'protocol': protocol,
            'n_seeds': int(g['seed'].nunique()),
            'mean_gap': float(g['optimism_gap'].mean()),
            'sd_gap': float(g['optimism_gap'].std(ddof=1)),
            'ci95_gap_lo': lo_gap,
            'ci95_gap_hi': hi_gap,
            'mean_gap_pr': float(g['optimism_gap_pr'].mean()),
            'sd_gap_pr': float(g['optimism_gap_pr'].std(ddof=1)),
            'ci95_gap_pr_lo': lo_pr,
            'ci95_gap_pr_hi': hi_pr,
            'bootstrap_method': 'percentile, seed=123/456, n_boot=5000',
        })

    out_summary = pd.DataFrame(rows)
    out_summary.to_csv(DATA / 'higgs_summary_v5.csv', index=False)

    pvt_gap = df.pivot(index='seed', columns='protocol', values='optimism_gap').dropna()
    diffs_gap = (pvt_gap['leaky'] - pvt_gap['anti_leakage']).values
    pvt_pr = df.pivot(index='seed', columns='protocol', values='optimism_gap_pr').dropna()
    diffs_pr = (pvt_pr['leaky'] - pvt_pr['anti_leakage']).values

    def paired_stats(diffs, alpha=0.05, power=0.8):
        n = len(diffs)
        sd = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
        mean = float(np.mean(diffs))
        dz = float(mean / sd) if sd > 0 else 0.0
        if np.allclose(diffs, 0.0):
            stat, p = 0.0, 1.0
        else:
            stat, p = wilcoxon(diffs, alternative='greater', zero_method='wilcox')
        # approximate paired-t MDE for two-sided alpha at target power
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power)
        mde = float((z_alpha + z_beta) * (sd / np.sqrt(max(n, 1))))
        lo, hi = boot_mean_ci(diffs, seed=789)
        return {
            'n_seeds': int(n),
            'mean_diff': mean,
            'sd_diff': sd,
            'ci95_diff_lo': lo,
            'ci95_diff_hi': hi,
            'cohen_dz': dz,
            'wilcoxon_stat': float(stat),
            'p_value_one_sided': float(p),
            'alpha': alpha,
            'power_target': power,
            'minimum_detectable_effect': mde,
            'mde_method': 'paired-normal approximation',
        }

    sig_gap = paired_stats(diffs_gap)
    sig_pr = paired_stats(diffs_pr)

    sig = pd.DataFrame([
        {'metric': 'roc_auc_gap', **sig_gap},
        {'metric': 'pr_auc_gap', **sig_pr},
    ])
    sig.to_csv(DATA / 'higgs_significance_v5.csv', index=False)

    manifest = {
        'dataset': 'higgs',
        'n_seeds': int(df['seed'].nunique()),
        'metrics': ['roc_auc_gap', 'pr_auc_gap'],
        'outputs': [
            'data/higgs/higgs_summary_v5.csv',
            'data/higgs/higgs_significance_v5.csv'
        ]
    }
    (DATA / 'manifest_v5_stats.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
