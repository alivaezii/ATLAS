#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from figure_style import apply_journal_style

N_SEEDS = 50
TRIALS = [1, 5, 10, 20, 40]

ROOT = Path(__file__).resolve().parents[1]
OUT_DATA = ROOT / "data" / "synthetic"
OUT_DATA.mkdir(parents=True, exist_ok=True)


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_score)


def evaluate_model(model_name, seed, Xtr, ytr, Xte, yte, Xex, yex):
    if model_name == "logreg":
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=220, random_state=seed)),
        ])
    elif model_name == "rf":
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=60, max_depth=6, random_state=seed, n_jobs=1)),
        ])
    elif model_name == "hgb":
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.1, random_state=seed)),
        ])
    else:
        raise ValueError(model_name)

    model.fit(Xtr, ytr)
    p_te = model.predict_proba(Xte)[:, 1]
    p_ex = model.predict_proba(Xex)[:, 1]
    return safe_auc(yte, p_te), safe_auc(yex, p_ex)


def simulate_s1(seed, protocol):
    rng = np.random.default_rng(seed)
    n, p = 2600, 25
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[:6] = np.array([1.0, -1.1, 0.9, -0.8, 0.7, -0.6])
    logits = X @ beta + rng.normal(0, 0.8, size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(int)

    idx = rng.permutation(n)
    tr, va, te, ex = idx[:1200], idx[1200:1600], idx[1600:2000], idx[2000:]

    if protocol == "leaky":
        scaler = StandardScaler().fit(X[np.r_[tr, va, te]])
    else:
        scaler = StandardScaler().fit(X[tr])

    Xs = scaler.transform(X)
    if protocol == "partial":
        X_train = np.vstack([Xs[tr], Xs[va]])
        y_train = np.hstack([y[tr], y[va]])
    else:
        X_train, y_train = Xs[tr], y[tr]

    best = None
    for c in [0.05, 0.2, 1.0, 3.0]:
        m = LogisticRegression(C=c, max_iter=300, random_state=seed)
        m.fit(X_train, y_train)
        score = safe_auc(y[va], m.predict_proba(Xs[va])[:, 1])
        if best is None or score > best[0]:
            best = (score, m)
    model = best[1]
    return safe_auc(y[te], model.predict_proba(Xs[te])[:, 1]), safe_auc(y[ex], model.predict_proba(Xs[ex])[:, 1])


def simulate_s2(seed, protocol):
    rng = np.random.default_rng(seed)
    n, p = 700, 1600
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[:15] = rng.normal(0, 0.9, size=15)
    logits = X @ beta / np.sqrt(15) + rng.normal(0, 0.8, size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(int)
    idx = rng.permutation(n)
    tr, va, te, ex = idx[:320], idx[320:440], idx[440:560], idx[560:]

    k = 60
    if protocol == "leaky":
        sel = SelectKBest(f_classif, k=k).fit(X[np.r_[tr, va, te]], y[np.r_[tr, va, te]])
    else:
        sel = SelectKBest(f_classif, k=k).fit(X[tr], y[tr])
    Xk = sel.transform(X)

    if protocol == "partial":
        X_train, y_train = np.vstack([Xk[tr], Xk[va]]), np.hstack([y[tr], y[va]])
    else:
        X_train, y_train = Xk[tr], y[tr]

    m = LogisticRegression(max_iter=300, C=0.8, random_state=seed)
    m.fit(X_train, y_train)
    return safe_auc(y[te], m.predict_proba(Xk[te])[:, 1]), safe_auc(y[ex], m.predict_proba(Xk[ex])[:, 1])


def simulate_s3(seed, protocol):
    rng = np.random.default_rng(seed)
    n, p = 2200, 28
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[:8] = rng.normal(0, 0.75, size=8)
    logits = X @ beta + rng.normal(0, 1.0, size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(int)

    idx = rng.permutation(n)
    tr, va, te, ex = idx[:1000], idx[1000:1400], idx[1400:1800], idx[1800:]

    if protocol == "leaky":
        select_split = te
    else:
        select_split = va

    best = None
    for t in range(25):
        C = float(np.exp(rng.uniform(np.log(0.03), np.log(8.0))))
        m = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=300, random_state=seed + t)),
        ])
        m.fit(X[tr], y[tr])
        score = safe_auc(y[select_split], m.predict_proba(X[select_split])[:, 1])
        if best is None or score > best[0]:
            best = (score, m)
    model = best[1]
    if protocol == "partial":
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=300, random_state=seed)),
        ]).fit(X[tr], y[tr])

    return safe_auc(y[te], model.predict_proba(X[te])[:, 1]), safe_auc(y[ex], model.predict_proba(X[ex])[:, 1])


def simulate_s4(seed, protocol):
    rng = np.random.default_rng(seed)
    n_groups = 220
    group_size = 12
    n = n_groups * group_size
    groups = np.repeat(np.arange(n_groups), group_size)
    group_latent = rng.normal(0, 1.0, size=n_groups)
    X = rng.normal(size=(n, 15))
    X[:, 0] += group_latent[groups]
    logits = 0.9 * X[:, 0] - 0.7 * X[:, 1] + 0.8 * group_latent[groups] + rng.normal(0, 0.9, size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(int)

    if protocol == "leaky":
        idx = rng.permutation(n)
        tr, te, ex = idx[:1500], idx[1500:2000], idx[2000:]
    else:
        gss = GroupShuffleSplit(n_splits=1, train_size=0.65, random_state=seed)
        tr, rest = next(gss.split(X, y, groups))
        gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=seed + 1)
        te_rel, ex_rel = next(gss2.split(X[rest], y[rest], groups[rest]))
        te, ex = rest[te_rel], rest[ex_rel]

    Xtr, ytr = X[tr], y[tr]
    if protocol == "partial":
        Xtr, ytr = np.vstack([X[tr], X[te][:120]]), np.hstack([y[tr], y[te][:120]])

    aucs = []
    for model_name in ["logreg", "rf", "hgb"]:
        it, exm = evaluate_model(model_name, seed, Xtr, ytr, X[te], y[te], X[ex], y[ex])
        aucs.append((it, exm))
    return float(np.mean([a for a, _ in aucs])), float(np.mean([b for _, b in aucs]))


def simulate_s5(seed, protocol):
    rng = np.random.default_rng(seed)
    n = 2600
    p = 10
    X = np.zeros((n, p))
    eps = rng.normal(size=(n, p))
    for t in range(1, n):
        X[t] = 0.75 * X[t - 1] + 0.45 * eps[t]
    drift = np.linspace(-0.6, 0.8, n)
    logits = 1.1 * X[:, 0] - 0.9 * X[:, 1] + 0.7 * drift + rng.normal(0, 0.8, size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(int)

    if protocol == "leaky":
        idx = rng.permutation(n)
        tr, te, ex = idx[:1300], idx[1300:1900], idx[1900:]
    else:
        tr, te, ex = np.arange(0, 1300), np.arange(1300, 1900), np.arange(1900, n)

    Xtr, ytr = X[tr], y[tr]
    if protocol == "partial":
        mix = np.r_[tr[:1100], te[:200]]
        Xtr, ytr = X[mix], y[mix]

    aucs = []
    for model_name in ["logreg", "rf", "hgb"]:
        it, exm = evaluate_model(model_name, seed, Xtr, ytr, X[te], y[te], X[ex], y[ex])
        aucs.append((it, exm))
    return float(np.mean([a for a, _ in aucs])), float(np.mean([b for _, b in aucs]))


def simulate_s6(seed, protocol):
    rng = np.random.default_rng(seed)
    base_n, p = 1800, 18
    Xb = rng.normal(size=(base_n, p))
    logits = 0.8 * Xb[:, 0] - 0.75 * Xb[:, 2] + 0.6 * Xb[:, 4] + rng.normal(0, 1.0, size=base_n)
    yb = (rng.random(base_n) < 1 / (1 + np.exp(-logits))).astype(int)

    n_dup = 250
    dup_idx = rng.choice(base_n, size=n_dup, replace=False)
    Xdup = Xb[dup_idx] + rng.normal(0, 0.015, size=(n_dup, p))
    ydup = yb[dup_idx]

    X = np.vstack([Xb, Xdup])
    y = np.hstack([yb, ydup])

    idx = rng.permutation(len(y))
    tr, te, ex = idx[:1100], idx[1100:1500], idx[1500:]

    if protocol == "leaky":
        Xtr, ytr = X[tr], y[tr]
    else:
        nn = NearestNeighbors(n_neighbors=1).fit(X[np.r_[te, ex]])
        d, _ = nn.kneighbors(X[tr])
        keep = d[:, 0] > (0.03 if protocol == "anti_leakage" else 0.015)
        Xtr, ytr = X[tr][keep], y[tr][keep]

    if protocol == "partial" and len(Xtr) < 100:
        Xtr, ytr = X[tr], y[tr]

    aucs = []
    for model_name in ["logreg", "rf", "hgb"]:
        it, exm = evaluate_model(model_name, seed, Xtr, ytr, X[te], y[te], X[ex], y[ex])
        aucs.append((it, exm))
    return float(np.mean([a for a, _ in aucs])), float(np.mean([b for _, b in aucs]))


def run_s3_trials(seed, protocol):
    rng = np.random.default_rng(seed)
    n, p = 2200, 24
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[:6] = rng.normal(0, 0.8, size=6)
    y = (rng.random(n) < 1 / (1 + np.exp(-(X @ beta + rng.normal(0, 1.0, size=n))))).astype(int)
    idx = rng.permutation(n)
    tr, va, te, ex = idx[:900], idx[900:1200], idx[1200:1700], idx[1700:]

    out = []
    for trials in TRIALS:
        best = None
        for t in range(trials):
            c = float(np.exp(rng.uniform(np.log(0.03), np.log(8.0))))
            m = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=c, max_iter=300, random_state=seed + t)),
            ])
            m.fit(X[tr], y[tr])
            score = safe_auc(y[te], m.predict_proba(X[te])[:, 1]) if protocol == "leaky" else safe_auc(y[va], m.predict_proba(X[va])[:, 1])
            if best is None or score > best[0]:
                best = (score, m)
        model = best[1]
        int_auc = safe_auc(y[te], model.predict_proba(X[te])[:, 1])
        ext_auc = safe_auc(y[ex], model.predict_proba(X[ex])[:, 1])
        out.append((trials, int_auc - ext_auc))
    return out


def bootstrap_ci(values, n_boot=4000, alpha=0.05, seed=123):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boots.append(np.mean(sample))
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def main():
    apply_journal_style()
    scenarios = {
        "S1_preprocess": simulate_s1,
        "S2_feature_select": simulate_s2,
        "S3_test_peeking": simulate_s3,
        "S4_group_leakage": simulate_s4,
        "S5_temporal_leakage": simulate_s5,
        "S6_duplicate_leakage": simulate_s6,
    }
    protocols = ["leaky", "partial", "anti_leakage"]

    rows = []
    for sname, fn in scenarios.items():
        for seed in range(1, N_SEEDS + 1):
            for protocol in protocols:
                int_auc, ext_auc = fn(10_000 * seed + hash((sname, protocol)) % 9999, protocol)
                rows.append({
                    "scenario": sname,
                    "seed": seed,
                    "protocol": protocol,
                    "internal_auc": int_auc,
                    "external_auc": ext_auc,
                    "optimism_gap": int_auc - ext_auc,
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DATA / "s1_s6_seed_results.csv", index=False)

    summary_rows = []
    for (scenario, protocol), g in df.groupby(["scenario", "protocol"]):
        ci_lo, ci_hi = bootstrap_ci(g["optimism_gap"].values)
        summary_rows.append({
            "scenario": scenario,
            "protocol": protocol,
            "n": len(g),
            "mean_gap": g["optimism_gap"].mean(),
            "sd_gap": g["optimism_gap"].std(ddof=1),
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
            "mean_internal_auc": g["internal_auc"].mean(),
            "mean_external_auc": g["external_auc"].mean(),
        })
    summary = pd.DataFrame(summary_rows).sort_values(["scenario", "protocol"])
    summary.to_csv(OUT_DATA / "s1_s6_summary.csv", index=False)

    piv = summary.pivot(index="scenario", columns="protocol", values="mean_gap")
    reduction = pd.DataFrame({
        "scenario": piv.index,
        "gap_leaky": piv["leaky"].values,
        "gap_partial": piv["partial"].values,
        "gap_anti_leakage": piv["anti_leakage"].values,
    })
    reduction["anti_vs_leaky_reduction_pct"] = 100 * (reduction["gap_leaky"] - reduction["gap_anti_leakage"]) / np.maximum(reduction["gap_leaky"], 1e-6)
    reduction.to_csv(OUT_DATA / "s1_s6_reduction_table.csv", index=False)

    sig_rows = []
    for scenario, g in df.groupby("scenario"):
        pvt = g.pivot(index="seed", columns="protocol", values="optimism_gap")
        diffs = (pvt["leaky"] - pvt["anti_leakage"]).dropna()
        stat, p_value = wilcoxon(diffs.values, alternative="greater", zero_method="wilcox")
        ci_lo, ci_hi = bootstrap_ci(diffs.values, seed=881)
        dz = float(np.mean(diffs.values) / max(np.std(diffs.values, ddof=1), 1e-12))
        sig_rows.append(
            {
                "scenario": scenario,
                "n_seeds": len(diffs),
                "mean_diff": float(np.mean(diffs.values)),
                "ci95_lo_mean_diff": ci_lo,
                "ci95_hi_mean_diff": ci_hi,
                "cohen_dz": dz,
                "wilcoxon_stat": float(stat),
                "p_value": float(p_value),
                "alpha": 0.05,
            }
        )
    pd.DataFrame(sig_rows).to_csv(OUT_DATA / "s1_s6_significance.csv", index=False)

    s3_rows = []
    for seed in range(1, N_SEEDS + 1):
        for protocol in ["leaky", "anti_leakage"]:
            for t, gap in run_s3_trials(77_000 + seed, protocol):
                s3_rows.append({"seed": seed, "protocol": protocol, "trials": t, "optimism_gap": gap})
    s3df = pd.DataFrame(s3_rows)
    s3df.to_csv(OUT_DATA / "s3_selection_intensity.csv", index=False)

    grp = s3df.groupby(["protocol", "trials"])["optimism_gap"].mean().reset_index()
    leaky = grp[grp.protocol == "leaky"].sort_values("trials")
    smooth = gaussian_filter1d(leaky["optimism_gap"].values, sigma=1)
    log_x = np.log(leaky["trials"].values)
    slope, intercept = np.polyfit(log_x, leaky["optimism_gap"].values, deg=1)

    intensity_summary = grp.copy()
    intensity_summary["log_trials"] = np.log(intensity_summary["trials"])
    intensity_summary.to_csv(OUT_DATA / "s3_selection_intensity_summary.csv", index=False)

    manifest = {
        "seeds": N_SEEDS,
        "scenarios": list(scenarios.keys()),
        "protocols": protocols,
        "trials_for_s3_curve": TRIALS,
        "s3_leaky_log_slope": float(slope),
        "s3_leaky_log_intercept": float(intercept),
        "s3_leaky_smoothed": [float(x) for x in smooth],
        "test": "One-sided Wilcoxon signed-rank on paired optimism-gap difference (leaky - anti_leakage), alpha=0.05",
        "effect_sizes": ["cohen_dz"],
        "outputs": [
            "data/synthetic/s1_s6_seed_results.csv",
            "data/synthetic/s1_s6_summary.csv",
            "data/synthetic/s1_s6_reduction_table.csv",
            "data/synthetic/s1_s6_significance.csv",
            "data/synthetic/s3_selection_intensity.csv",
            "data/synthetic/s3_selection_intensity_summary.csv",
        ],
    }
    (OUT_DATA / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
