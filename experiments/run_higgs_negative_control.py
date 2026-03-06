#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split

from run_higgs_leakage import load_higgs, run_protocol

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'higgs_negative_control'
OUT.mkdir(parents=True, exist_ok=True)


def main():
    csv_path = ROOT / 'data' / 'raw' / 'HIGGS.csv'
    X, y = load_higgs(csv_path, max_rows=1000000)

    seeds = 10
    trials = 20
    rows = []

    for seed in range(1, seeds + 1):
        rng = np.random.default_rng(70000 + seed)
        y_shuf = y.copy()
        rng.shuffle(y_shuf)
        for protocol in ['leaky', 'anti_leakage']:
            out = run_protocol(X, y_shuf, 91000 + seed, trials, protocol)
            rows.append({
                'dataset': 'higgs_label_shuffle',
                'seed': seed,
                'protocol': protocol,
                'trials': trials,
                **out,
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'seed_results.csv', index=False)

    pvt = df.pivot(index='seed', columns='protocol', values='optimism_gap').dropna()
    diffs = (pvt['leaky'] - pvt['anti_leakage']).values
    if np.allclose(diffs, 0.0):
        stat, p = 0.0, 1.0
    else:
        stat, p = wilcoxon(diffs, alternative='greater', zero_method='wilcox')

    summary = {
        'n_seeds': int(len(diffs)),
        'mean_diff_roc_gap': float(np.mean(diffs)),
        'sd_diff_roc_gap': float(np.std(diffs, ddof=1)),
        'wilcoxon_stat': float(stat),
        'p_value_one_sided': float(p),
        'interpretation': 'No systematic optimism inflation expected under label shuffle negative control.'
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
