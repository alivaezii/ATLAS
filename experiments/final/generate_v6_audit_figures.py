#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

from figure_style import apply_journal_style, export_figure

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data" / "audit"
FIG = ROOT / "figures"

CHECK_ORDER = ["ALAV-01", "ALAV-02", "ALAV-03", "ALAV-04", "ALAV-05", "ALAV-06"]
CHECK_LABELS = ["Overlap", "Fit scope", "Augment", "Test reuse", "Time/Group", "Cache"]
PROFILE_FILES = {
    "Anti-leakage": AUDIT / "alav_report_anti_leakage_v6.json",
    "Leaky": AUDIT / "alav_report_leaky_v6.json",
}

STATUS_TO_RISK = {"PASS": 0.0, "WARN": 0.5, "FAIL": 1.0}
STATUS_TO_COLOR = {"PASS": "#1b9e77", "WARN": "#d95f02", "FAIL": "#7570b3"}


def load_statuses(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    by_id = {c["check_id"]: c["status"] for c in data["checks"]}
    return [by_id[k] for k in CHECK_ORDER]


def main():
    apply_journal_style()

    profiles = list(PROFILE_FILES.keys())
    statuses = [load_statuses(PROFILE_FILES[p]) for p in profiles]
    risks = np.array([[STATUS_TO_RISK[s] for s in row] for row in statuses])

    fig, ax = plt.subplots(figsize=(8.6, 3.6))

    # draw clean matrix as colored cells
    for i in range(risks.shape[0]):
        for j in range(risks.shape[1]):
            s = statuses[i][j]
            rect = plt.Rectangle((j, i), 1, 1, color=STATUS_TO_COLOR[s], alpha=0.22, ec="#444444", lw=0.6)
            ax.add_patch(rect)
            ax.text(j + 0.5, i + 0.5, s, ha="center", va="center", fontsize=9)

    ax.set_xlim(0, len(CHECK_ORDER))
    ax.set_ylim(0, len(profiles))
    ax.set_xticks(np.arange(len(CHECK_ORDER)) + 0.5)
    ax.set_xticklabels(CHECK_LABELS)
    ax.set_yticks(np.arange(len(profiles)) + 0.5)
    ax.set_yticklabels(profiles)
    ax.invert_yaxis()
    ax.set_title("ALAV check outcomes by protocol profile")
    ax.set_xlabel("Audit checks")
    ax.set_ylabel("Profile")
    ax.grid(False)

    # Clean legend
    handles = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=STATUS_TO_COLOR[k], markeredgecolor="#444444", markersize=10, label=k)
        for k in ["PASS", "WARN", "FAIL"]
    ]
    ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=3)

    fig.subplots_adjust(top=0.72, bottom=0.18, left=0.10, right=0.98)
    export_figure(fig, FIG / "fig10_alav_audit_matrix_v6")
    plt.close(fig)
    print("Generated:", FIG / "fig10_alav_audit_matrix_v6.pdf")


if __name__ == "__main__":
    main()
