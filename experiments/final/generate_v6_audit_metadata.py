#!/usr/bin/env python3
"""Generate explicit augmentation/cache audit metadata from experiment code.

Purpose: close audit-completeness gaps with reproducible, code-backed metadata.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
AUDIT = ROOT / "data" / "audit"
AUDIT.mkdir(parents=True, exist_ok=True)

TARGET_SCRIPTS = [
    EXP / "run_higgs_leakage.py",
    EXP / "run_realworld_leakage.py",
    EXP / "run_synthetic_leakage.py",
]

AUG_KEYWORDS = ["augment", "augmentation", "albumentations", "imgaug", "torchvision.transforms"]
CACHE_KEYWORDS = ["joblib", "memory=", "cache", "cached", "diskcache", "redis"]


def keyword_hits(text: str, keywords: list[str]) -> list[str]:
    low = text.lower()
    return [k for k in keywords if k.lower() in low]


def main():
    inspected = []
    aug_hits = {}
    cache_hits = {}

    for p in TARGET_SCRIPTS:
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8")
        inspected.append(str(p.relative_to(ROOT)))
        ah = keyword_hits(txt, AUG_KEYWORDS)
        ch = keyword_hits(txt, CACHE_KEYWORDS)
        if ah:
            aug_hits[str(p.relative_to(ROOT))] = ah
        if ch:
            cache_hits[str(p.relative_to(ROOT))] = ch

    aug_meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inspected_scripts": inspected,
        "policy": "No data augmentation stage in current benchmark pipelines.",
        "augmentation_used": bool(aug_hits),
        "keyword_hits": aug_hits,
        "status": "pass" if not aug_hits else "warn",
        "evidence_rule": "Static scan over benchmark runner scripts.",
    }

    cache_meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inspected_scripts": inspected,
        "policy": "Cross-split cache reuse is disabled in current benchmark pipelines.",
        "cache_used": bool(cache_hits),
        "cache_namespace_mode": "disabled" if not cache_hits else "detected",
        "keyword_hits": cache_hits,
        "status": "pass" if not cache_hits else "warn",
        "evidence_rule": "Static scan over benchmark runner scripts.",
    }

    (AUDIT / "augmentation_metadata_v6.json").write_text(json.dumps(aug_meta, indent=2), encoding="utf-8")
    (AUDIT / "cache_lineage_v6.json").write_text(json.dumps(cache_meta, indent=2), encoding="utf-8")

    print("Generated:")
    print(" - data/audit/augmentation_metadata_v6.json")
    print(" - data/audit/cache_lineage_v6.json")


if __name__ == "__main__":
    main()
