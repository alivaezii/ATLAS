#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "higgs"
OUT.mkdir(parents=True, exist_ok=True)


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def safe_pr_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float(np.mean(y_true))
    return float(average_precision_score(y_true, y_score))


def load_higgs(csv_path: Path, max_rows: int | None = None):
    # Expected Kaggle-style HIGGS CSV: first column label, remaining 28 features.
    df = pd.read_csv(csv_path, header=None, nrows=max_rows)
    y = df.iloc[:, 0].astype(int).values
    X = df.iloc[:, 1:].astype(float).values
    return X, y


def build_model(seed: int, alpha: float):
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=alpha,
                max_iter=1000,
                tol=1e-3,
                random_state=seed,
            ),
        ),
    ])


def run_protocol(X, y, seed: int, trials: int, protocol: str):
    X_dev, X_ext, y_dev, y_ext = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    X_train, X_tmp, y_train, y_tmp = train_test_split(X_dev, y_dev, test_size=0.4, stratify=y_dev, random_state=seed + 1)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed + 2)

    X_fit = X_train if protocol == "anti_leakage" else np.vstack([X_train, X_val])
    y_fit = y_train if protocol == "anti_leakage" else np.hstack([y_train, y_val])

    rng = np.random.default_rng(seed + 10)
    best = None
    for _ in range(trials):
        alpha = float(np.exp(rng.uniform(np.log(1e-6), np.log(1e-3))))
        m = build_model(seed, alpha)
        m.fit(X_fit, y_fit)
        p_sel = m.predict_proba(X_test if protocol == "leaky" else X_val)[:, 1]
        y_sel = y_test if protocol == "leaky" else y_val
        score = safe_auc(y_sel, p_sel)
        if best is None or score > best[0]:
            best = (score, m, alpha)

    model = best[1]
    p_test = model.predict_proba(X_test)[:, 1]
    p_ext = model.predict_proba(X_ext)[:, 1]
    return {
        "selected_alpha": float(best[2]),
        "internal_auc": safe_auc(y_test, p_test),
        "external_auc": safe_auc(y_ext, p_ext),
        "optimism_gap": safe_auc(y_test, p_test) - safe_auc(y_ext, p_ext),
        "internal_pr_auc": safe_pr_auc(y_test, p_test),
        "external_pr_auc": safe_pr_auc(y_ext, p_ext),
        "optimism_gap_pr": safe_pr_auc(y_test, p_test) - safe_pr_auc(y_ext, p_ext),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "n_ext": int(len(y_ext)),
    }


def main():
    ap = argparse.ArgumentParser(description="HIGGS large-scale leakage benchmark")
    ap.add_argument("--csv", required=True, help="Path to HIGGS CSV")
    ap.add_argument("--max-rows", type=int, default=1000000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--trials", type=int, default=20)
    args = ap.parse_args()

    X, y = load_higgs(Path(args.csv), max_rows=args.max_rows)

    rows = []
    t0 = time.perf_counter()
    for seed in range(1, args.seeds + 1):
        for protocol in ["leaky", "anti_leakage"]:
            s0 = time.perf_counter()
            out = run_protocol(X, y, 90_000 + seed, args.trials, protocol)
            rows.append(
                {
                    "dataset": "higgs",
                    "seed": seed,
                    "protocol": protocol,
                    "trials": args.trials,
                    "runtime_sec": float(time.perf_counter() - s0),
                    **out,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "higgs_seed_results.csv", index=False)

    summary = (
        df.groupby(["dataset", "protocol", "trials"])
        .agg(
            mean_gap=("optimism_gap", "mean"),
            sd_gap=("optimism_gap", "std"),
            mean_gap_pr=("optimism_gap_pr", "mean"),
            sd_gap_pr=("optimism_gap_pr", "std"),
            mean_internal_auc=("internal_auc", "mean"),
            mean_external_auc=("external_auc", "mean"),
            mean_internal_pr_auc=("internal_pr_auc", "mean"),
            mean_external_pr_auc=("external_pr_auc", "mean"),
            mean_runtime_sec=("runtime_sec", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(OUT / "higgs_summary.csv", index=False)

    pvt = df.pivot(index="seed", columns="protocol", values="optimism_gap")
    diffs = (pvt["leaky"] - pvt["anti_leakage"]).dropna().values
    if np.allclose(diffs, 0.0):
        stat, p_val = 0.0, 1.0
    else:
        stat, p_val = wilcoxon(diffs, alternative="greater", zero_method="wilcox")

    sig = pd.DataFrame(
        [
            {
                "dataset": "higgs",
                "n_seeds": int(len(diffs)),
                "mean_diff": float(np.mean(diffs)),
                "cohen_dz": float(np.mean(diffs) / max(np.std(diffs, ddof=1), 1e-12)),
                "wilcoxon_stat": float(stat),
                "p_value": float(p_val),
                "primary_absdiff_metric": "paired_absdiff_roc_auc_gap",
            }
        ]
    )
    sig.to_csv(OUT / "higgs_significance.csv", index=False)

    manifest = {
        "dataset": "higgs",
        "input_csv": str(Path(args.csv)),
        "max_rows": int(args.max_rows),
        "seeds": int(args.seeds),
        "trials": int(args.trials),
        "metrics": ["roc_auc", "pr_auc"],
        "primary_comparison_metric": "paired_absdiff_roc_auc_gap",
        "compute": {
            "total_runtime_sec": float(time.perf_counter() - t0),
            "n_samples": int(len(y)),
            "n_features": int(X.shape[1]),
        },
        "outputs": [
            "data/higgs/higgs_seed_results.csv",
            "data/higgs/higgs_summary.csv",
            "data/higgs/higgs_significance.csv",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
