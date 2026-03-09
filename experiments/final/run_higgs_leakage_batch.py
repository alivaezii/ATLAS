#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import time

from run_higgs_leakage import load_higgs, run_protocol

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'higgs'
OUT.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--max-rows', type=int, default=1000000)
    ap.add_argument('--seed-start', type=int, required=True)
    ap.add_argument('--seed-count', type=int, required=True)
    ap.add_argument('--trials', type=int, default=20)
    ap.add_argument('--out', default='higgs_seed_results_v5.csv')
    args = ap.parse_args()

    X, y = load_higgs(Path(args.csv), max_rows=args.max_rows)
    rows = []
    for seed in range(args.seed_start, args.seed_start + args.seed_count):
        for protocol in ['leaky', 'anti_leakage']:
            s0 = time.perf_counter()
            out = run_protocol(X, y, 90_000 + seed, args.trials, protocol)
            rows.append({
                'dataset': 'higgs',
                'seed': seed,
                'protocol': protocol,
                'trials': args.trials,
                'runtime_sec': float(time.perf_counter() - s0),
                **out,
            })

    out_path = OUT / args.out
    df = pd.DataFrame(rows)
    if out_path.exists():
        prev = pd.read_csv(out_path)
        df = pd.concat([prev, df], ignore_index=True)
        df = df.drop_duplicates(subset=['seed','protocol','trials'], keep='last').sort_values(['seed','protocol'])
    df.to_csv(out_path, index=False)
    print(out_path)


if __name__ == '__main__':
    main()
