#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from figure_style import MARKERS, PALETTE, apply_journal_style, cycle_colors, export_figure

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
REAL = ROOT / "data" / "realworld"
SYN = ROOT / "data" / "synthetic"

DATASET_ORDER = ["breast_cancer", "diabetes_binary", "digits", "iris", "wine"]


def _ordered_datasets(existing):
    keep = [d for d in DATASET_ORDER if d in existing]
    rem = [d for d in existing if d not in keep]
    return keep + sorted(rem)


def _add_ci(ax, x, m, lo, hi, color, label, marker):
    ax.fill_between(x, lo, hi, color=color, alpha=0.14, linewidth=0, zorder=1)
    ax.plot(x, m, color=color, marker=marker, label=label, linewidth=2.6, zorder=2)


def _bootstrap_ci_mean(values, n_boot=3000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    boots = []
    for _ in range(n_boot):
        boots.append(np.mean(rng.choice(arr, size=len(arr), replace=True)))
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def fig1_multidataset_pressure():
    d = pd.read_csv(REAL / "pressure_summary.csv")
    datasets = _ordered_datasets(sorted(d.dataset.unique().tolist()))
    colors = cycle_colors(datasets)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharex=True)
    ax = axes[0]
    leaky = d[d.protocol == "leaky"].sort_values(["dataset", "trials"])

    for i, ds in enumerate(datasets):
        g = leaky[leaky.dataset == ds].sort_values("trials")
        _add_ci(
            ax,
            g.trials.values,
            g.mean_gap.values,
            g.ci95_lo.values,
            g.ci95_hi.values,
            colors[i],
            ds.replace("_", " "),
            MARKERS[i % len(MARKERS)],
        )

    ax.set_xscale("log")
    ax.set_xlabel("Hyperparameter trials (log scale)")
    ax.set_ylabel("Optimism gap Δopt (AUC units)")
    ax.set_title("Leaky pressure across datasets")

    ax2 = axes[1]
    for i, ds in enumerate(datasets):
        g_leaky = d[(d.dataset == ds) & (d.protocol == "leaky")].sort_values("trials")
        g_anti = d[(d.dataset == ds) & (d.protocol == "anti_leakage")].sort_values("trials")
        diff = g_leaky.mean_gap.values - g_anti.mean_gap.values
        ax2.plot(g_leaky.trials.values, diff, color=colors[i], marker=MARKERS[i % len(MARKERS)], label=ds.replace("_", " "))

    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xscale("log")
    ax2.set_xlabel("Hyperparameter trials (log scale)")
    ax2.set_ylabel("Gap difference (leaky − anti) (AUC units)")
    ax2.set_title("Pressure difference view")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.subplots_adjust(top=0.84, wspace=0.30)

    export_figure(fig, FIG_DIR / "fig1_multi_dataset_pressure")
    plt.close(fig)


def fig2_paired_difference():
    seed = pd.read_csv(REAL / "pressure_seed_results.csv")
    trials = int(seed.trials.max())
    s = seed[seed.trials == trials]

    rows = []
    for ds in _ordered_datasets(sorted(s.dataset.unique().tolist())):
        g = s[s.dataset == ds]
        p = g.pivot(index="seed", columns="protocol", values="optimism_gap")
        diff = (p["leaky"] - p["anti_leakage"]).dropna().values
        rows.append(
            {
                "dataset": ds,
                "mean": float(np.mean(diff)),
                "lo": float(np.quantile(diff, 0.025)),
                "hi": float(np.quantile(diff, 0.975)),
                "vals": diff,
            }
        )

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(rows))
    means = [r["mean"] for r in rows]
    yerr = np.vstack([[r["mean"] - r["lo"] for r in rows], [r["hi"] - r["mean"] for r in rows]])
    ax.errorbar(x, means, yerr=yerr, fmt="o", color=PALETTE["blue"], ecolor=PALETTE["gray"], capsize=3, label="mean ± 95% interval")

    for i, r in enumerate(rows):
        jitter = np.linspace(-0.09, 0.09, len(r["vals"]))
        ax.scatter(np.full_like(r["vals"], i) + jitter, r["vals"], color=PALETTE["orange"], s=12, alpha=0.35)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r["dataset"].replace("_", " ") for r in rows], rotation=18, ha="right")
    ax.set_ylabel("Paired Δopt difference: leaky − anti (AUC units)")
    ax.set_xlabel(f"Dataset (trials = {trials})")
    ax.set_title("Paired protocol differences")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    export_figure(fig, FIG_DIR / "fig2_paired_difference")
    plt.close(fig)


def fig3_selection_intensity_loglog():
    # Per-dataset slope audit in log-log space from real-world seed-level outputs.
    d = pd.read_csv(REAL / "pressure_seed_results.csv")
    d = d[d.protocol == "leaky"].copy()
    d["abs_optimism_gap"] = d["optimism_gap"].abs()

    datasets = _ordered_datasets(sorted(d.dataset.unique().tolist()))
    colors = cycle_colors(datasets)

    eps = 1e-6
    slope_rows = []
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for i, ds in enumerate(datasets):
        g = d[d.dataset == ds].copy()
        # Mean curve + bootstrap CI at each trials level.
        summary_rows = []
        for t, gt in g.groupby("trials"):
            vals = np.maximum(gt["abs_optimism_gap"].values, eps)
            m = float(np.mean(vals))
            lo, hi = _bootstrap_ci_mean(vals, seed=1234 + int(t) + i)
            summary_rows.append({"trials": t, "mean": m, "lo": max(lo, eps), "hi": max(hi, eps)})
        s = pd.DataFrame(summary_rows).sort_values("trials")

        # Per-seed slope distribution for CI.
        seed_slopes = []
        for seed, gs in g.groupby("seed"):
            gs = gs.sort_values("trials")
            x = np.log(gs["trials"].values.astype(float))
            y = np.log(np.maximum(gs["abs_optimism_gap"].values.astype(float), eps))
            if np.isfinite(y).all() and len(np.unique(x)) >= 2:
                b1, _ = np.polyfit(x, y, deg=1)
                seed_slopes.append(float(b1))

        slope = float(np.mean(seed_slopes)) if seed_slopes else float("nan")
        slo, shi = _bootstrap_ci_mean(np.array(seed_slopes), seed=900 + i) if seed_slopes else (float("nan"), float("nan"))
        slope_rows.append(
            {
                "dataset": ds,
                "n_seed_slopes": len(seed_slopes),
                "slope_loglog": slope,
                "ci95_lo": slo,
                "ci95_hi": shi,
                "ci_includes_zero": bool(slo <= 0 <= shi) if np.isfinite(slo) and np.isfinite(shi) else True,
            }
        )

        label = f"{ds.replace('_',' ')} (slope={slope:.2f}, 95% CI [{slo:.2f},{shi:.2f}])"
        _add_ci(ax, s["trials"].values, s["mean"].values, s["lo"].values, s["hi"].values, colors[i], label, MARKERS[i % len(MARKERS)])

    pd.DataFrame(slope_rows).to_csv(REAL / "fig3_dataset_slopes.csv", index=False)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Selection intensity: number of trials (log)")
    ax.set_ylabel("|Δopt| optimism magnitude (AUC units, log)")
    ax.set_title("Per-dataset selection intensity scaling")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)

    export_figure(fig, FIG_DIR / "fig3_selection_intensity_loglog")
    plt.close(fig)


def fig4_synthetic_vs_real():
    syn = pd.read_csv(SYN / "s1_s6_seed_results.csv")
    real = pd.read_csv(REAL / "pressure_seed_results.csv")

    syn2 = syn[syn.protocol.isin(["leaky", "anti_leakage"])]
    syn2 = syn2.assign(domain="synthetic")
    real2 = real.assign(domain="real")

    syn_ag = syn2.groupby(["domain", "protocol", "seed"], as_index=False)["optimism_gap"].mean()
    real_ag = real2.groupby(["domain", "protocol", "seed"], as_index=False)["optimism_gap"].mean()
    both = pd.concat([syn_ag, real_ag], ignore_index=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    positions = {("synthetic", "leaky"): 0, ("synthetic", "anti_leakage"): 1, ("real", "leaky"): 3, ("real", "anti_leakage"): 4}
    colors = {"leaky": PALETTE["red"], "anti_leakage": PALETTE["blue"]}

    for key, pos in positions.items():
        g = both[(both.domain == key[0]) & (both.protocol == key[1])]["optimism_gap"].values
        vp = ax.violinplot(g, positions=[pos], widths=0.8, showmeans=False, showmedians=True, bw_method=0.3)
        for b in vp["bodies"]:
            b.set_facecolor(colors[key[1]])
            b.set_alpha(0.18)
        vp["cmedians"].set_color(colors[key[1]])
        m, lo, hi = np.mean(g), np.quantile(g, 0.025), np.quantile(g, 0.975)
        ax.errorbar([pos], [m], yerr=[[m - lo], [hi - m]], fmt="o", color=colors[key[1]], capsize=3)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks([0, 1, 3, 4])
    ax.set_xticklabels(["Syn leaky", "Syn anti", "Real leaky", "Real anti"])
    ax.set_ylabel("Optimism gap Δopt (AUC units)")
    ax.set_title("Synthetic vs real distributions")

    export_figure(fig, FIG_DIR / "fig4_synthetic_vs_real")
    plt.close(fig)


def fig5_pipeline_schematic():
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    ax.axis("off")

    boxes = [
        (0.04, 0.55, "Raw dataset"),
        (0.27, 0.55, "Split before fit\n(train/val/test/external)"),
        (0.54, 0.55, "Fit transforms/model\non train only"),
        (0.79, 0.55, "Single-use test +\nexternal evaluation"),
        (0.54, 0.16, "Leakage audit:\noverlap/group/time/duplicate checks"),
    ]

    for x, y, txt in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.17, 0.26, fill=False, linewidth=1.1, edgecolor=PALETTE["black"]))
        ax.text(x + 0.085, y + 0.13, txt, ha="center", va="center", fontsize=10)

    arrows = [
        ((0.21, 0.68), (0.27, 0.68)),
        ((0.44, 0.68), (0.54, 0.68)),
        ((0.71, 0.68), (0.79, 0.68)),
        ((0.625, 0.55), (0.625, 0.44)),
    ]
    for s, e in arrows:
        ax.annotate("", xy=e, xytext=s, arrowprops=dict(arrowstyle="->", linewidth=1.4, color=PALETTE["black"]))

    ax.text(0.275, 0.48, "No information from test/external flows backward", fontsize=10, color=PALETTE["red"])
    ax.set_title("Split-before-fit anti-leakage pipeline", fontsize=11)

    export_figure(fig, FIG_DIR / "fig5_pipeline_schematic")
    plt.close(fig)


def main():
    apply_journal_style()
    fig1_multidataset_pressure()
    fig2_paired_difference()
    fig3_selection_intensity_loglog()
    fig4_synthetic_vs_real()
    fig5_pipeline_schematic()


if __name__ == "__main__":
    main()
