#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, load_iris, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from figure_style import apply_journal_style, export_figure

ROOT = Path(__file__).resolve().parents[1]
OUT_DATA = ROOT / "data" / "realworld"
OUT_DATA.mkdir(parents=True, exist_ok=True)

N_SEEDS = 50
TRIALS = [1, 5, 10, 20, 40]
DATASETS = ["breast_cancer", "wine", "digits", "iris", "diabetes_binary"]


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def safe_pr_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float(np.mean(y_true))
    return float(average_precision_score(y_true, y_score))


def _dataset_binary(name: str):
    if name == "breast_cancer":
        d = load_breast_cancer()
        X, y = d.data, d.target
    elif name == "wine":
        d = load_wine()
        X, y = d.data, (d.target == 0).astype(int)
    elif name == "digits":
        d = load_digits()
        X, y = d.data, (d.target >= 5).astype(int)
    elif name == "iris":
        d = load_iris()
        X, y = d.data, (d.target == 0).astype(int)
    elif name == "diabetes_binary":
        d = load_diabetes()
        X = d.data
        y = (d.target >= np.median(d.target)).astype(int)
    else:
        raise ValueError(name)

    return X, y


def make_dataset(seed, dataset_name):
    X, y = _dataset_binary(dataset_name)

    X_dev, X_ext, y_dev, y_ext = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X_dev, y_dev, test_size=0.4, stratify=y_dev, random_state=seed + 1
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed + 2
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, X_ext, y_ext


def sample_model(seed, trial_id):
    rng = np.random.default_rng(seed + trial_id)
    c = float(np.exp(rng.uniform(np.log(0.01), np.log(30.0))))
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                solver="liblinear",
                C=c,
                max_iter=200,
                random_state=seed + trial_id,
            ),
        ),
    ])


def run_protocol(seed, trials, protocol, dataset_name):
    X_train, y_train, X_val, y_val, X_test, y_test, X_ext, y_ext = make_dataset(seed, dataset_name)
    X_fit = X_train if protocol == "anti_leakage" else np.vstack([X_train, X_val])
    y_fit = y_train if protocol == "anti_leakage" else np.hstack([y_train, y_val])

    best = None
    for t in range(trials):
        model = sample_model(seed, 1000 + t)
        model.fit(X_fit, y_fit)
        score = safe_auc(y_test, model.predict_proba(X_test)[:, 1]) if protocol == "leaky" else safe_auc(y_val, model.predict_proba(X_val)[:, 1])
        if best is None or score > best[0]:
            best = (score, model)

    selected = best[1]
    p_test = selected.predict_proba(X_test)[:, 1]
    p_ext = selected.predict_proba(X_ext)[:, 1]
    internal_auc = safe_auc(y_test, p_test)
    external_auc = safe_auc(y_ext, p_ext)
    internal_pr_auc = safe_pr_auc(y_test, p_test)
    external_pr_auc = safe_pr_auc(y_ext, p_ext)
    return internal_auc, external_auc, internal_pr_auc, external_pr_auc


def bootstrap_ci(values, n_boot=4000, alpha=0.05, seed=7):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    boots = []
    for _ in range(n_boot):
        boots.append(np.mean(rng.choice(arr, size=len(arr), replace=True)))
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = (m / rank) * ranked[i]
        prev = min(prev, val)
        adj[i] = prev
    out = np.empty(m, dtype=float)
    out[order] = np.minimum(adj, 1.0)
    return out


def main():
    apply_journal_style()
    rows = []
    for dataset_name in DATASETS:
        for seed in range(1, N_SEEDS + 1):
            for protocol in ["leaky", "anti_leakage"]:
                for trials in TRIALS:
                    internal_auc, external_auc, internal_pr_auc, external_pr_auc = run_protocol(50_000 + seed, trials, protocol, dataset_name)
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "seed": seed,
                            "protocol": protocol,
                            "trials": trials,
                            "internal_auc": internal_auc,
                            "external_auc": external_auc,
                            "optimism_gap": internal_auc - external_auc,
                            "internal_pr_auc": internal_pr_auc,
                            "external_pr_auc": external_pr_auc,
                            "optimism_gap_pr": internal_pr_auc - external_pr_auc,
                        }
                    )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DATA / "pressure_seed_results.csv", index=False)

    summary = (
        df.groupby(["dataset", "protocol", "trials"])
        .agg(
            mean_gap=("optimism_gap", "mean"),
            sd_gap=("optimism_gap", "std"),
            mean_internal_auc=("internal_auc", "mean"),
            mean_external_auc=("external_auc", "mean"),
            mean_gap_pr=("optimism_gap_pr", "mean"),
            sd_gap_pr=("optimism_gap_pr", "std"),
            mean_internal_pr_auc=("internal_pr_auc", "mean"),
            mean_external_pr_auc=("external_pr_auc", "mean"),
        )
        .reset_index()
    )

    ci_rows = []
    for (dataset_name, protocol, trials), g in df.groupby(["dataset", "protocol", "trials"]):
        lo, hi = bootstrap_ci(g["optimism_gap"].values, seed=71 + trials)
        ci_rows.append({"dataset": dataset_name, "protocol": protocol, "trials": trials, "ci95_lo": lo, "ci95_hi": hi})
    summary = summary.merge(pd.DataFrame(ci_rows), on=["dataset", "protocol", "trials"], how="left")
    summary.to_csv(OUT_DATA / "pressure_summary.csv", index=False)

    # Paired significance and effect sizes (leaky - anti_leakage) across matched seeds.
    paired_rows = []
    for (dataset_name, trials), g in df.groupby(["dataset", "trials"]):
        pvt = g.pivot(index="seed", columns="protocol", values="optimism_gap")
        pvt_pr = g.pivot(index="seed", columns="protocol", values="optimism_gap_pr")
        diffs = (pvt["leaky"] - pvt["anti_leakage"]).dropna().values
        diffs_pr = (pvt_pr["leaky"] - pvt_pr["anti_leakage"]).dropna().values
        if np.allclose(diffs, 0.0):
            # Degenerate paired case (e.g., perfectly matched protocols):
            # avoid scipy RuntimeWarning in Wilcoxon z-normalization.
            stat, p_value = 0.0, 1.0
        else:
            stat, p_value = wilcoxon(diffs, alternative="greater", zero_method="wilcox")
        ci_lo, ci_hi = bootstrap_ci(diffs, seed=113 + int(trials))
        ci_lo_pr, ci_hi_pr = bootstrap_ci(diffs_pr, seed=313 + int(trials))
        dz = float(np.mean(diffs) / max(np.std(diffs, ddof=1), 1e-12))
        n_pos = int(np.sum(diffs > 0))
        n_neg = int(np.sum(diffs < 0))
        rank_biserial = float((n_pos - n_neg) / max(len(diffs), 1))
        paired_rows.append(
            {
                "dataset": dataset_name,
                "trials": trials,
                "n_seeds": len(diffs),
                "mean_diff": float(np.mean(diffs)),
                "median_diff": float(np.median(diffs)),
                "ci95_lo_mean_diff": ci_lo,
                "ci95_hi_mean_diff": ci_hi,
                "mean_diff_pr": float(np.mean(diffs_pr)),
                "ci95_lo_mean_diff_pr": ci_lo_pr,
                "ci95_hi_mean_diff_pr": ci_hi_pr,
                "primary_absdiff_metric": "paired_absdiff_roc_auc_gap",
                "cohen_dz": dz,
                "rank_biserial_sign": rank_biserial,
                "wilcoxon_stat": float(stat),
                "p_value": float(p_value),
                "alpha": 0.05,
            }
        )

    paired = pd.DataFrame(paired_rows)
    paired["p_bh_fdr"] = np.nan
    for trials in TRIALS:
        mask = paired["trials"] == trials
        paired.loc[mask, "p_bh_fdr"] = bh_adjust(paired.loc[mask, "p_value"].values)
    paired.to_csv(OUT_DATA / "paired_significance.csv", index=False)

    # Real-world explicit gap plot for Breast Cancer: Δopt_real = leaky - anti_leakage.
    breast = summary[summary["dataset"] == "breast_cancer"].copy()
    b_leaky = breast[breast["protocol"] == "leaky"].sort_values("trials")
    b_anti = breast[breast["protocol"] == "anti_leakage"].sort_values("trials")
    gap_real = b_leaky["mean_gap"].values - b_anti["mean_gap"].values

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(b_leaky["trials"].values, gap_real, marker="o", linewidth=2.6, color="#0072B2")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Hyperparameter trials (log scale)")
    ax.set_ylabel("Δopt_real = leaky − anti (AUC units)")
    ax.set_title("Real-world pressure gap (Breast Cancer)")
    export_figure(fig, ROOT / "figures" / "realworld_pressure_breast_cancer")
    plt.close(fig)

    manifest = {
        "datasets": DATASETS,
        "seeds": N_SEEDS,
        "trials": TRIALS,
        "metrics": ["roc_auc", "pr_auc"],
        "primary_comparison_metric": "paired_absdiff_roc_auc_gap",
        "test": "One-sided Wilcoxon signed-rank on paired optimism-gap difference (leaky - anti_leakage), alpha=0.05, BH-FDR within each trial across datasets",
        "effect_sizes": ["cohen_dz", "rank_biserial_sign"],
        "outputs": [
            "data/realworld/pressure_seed_results.csv",
            "data/realworld/pressure_summary.csv",
            "data/realworld/paired_significance.csv",
        ],
    }
    (OUT_DATA / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
