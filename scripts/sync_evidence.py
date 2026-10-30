#!/usr/bin/env python3
"""Sync evaluation evidence and map points to web/public.

Copies outputs/evidence/*.json and outputs/map_points.json into web/public/
and web/public/evidence/ so the frontend can serve them statically or display them.
"""

import glob
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs"
EVIDENCE_SRC = OUTPUTS_DIR / "evidence"
WEB_PUBLIC = ROOT / "web" / "public"
WEB_EVIDENCE = WEB_PUBLIC / "evidence"


def sync():
    WEB_PUBLIC.mkdir(parents=True, exist_ok=True)
    WEB_EVIDENCE.mkdir(parents=True, exist_ok=True)

    # Copy map_points.json
    map_src = OUTPUTS_DIR / "map_points.json"
    if map_src.exists():
        shutil.copy2(map_src, WEB_PUBLIC / "map_points.json")
        print(f"✓ Copied {map_src.name} -> {WEB_PUBLIC / 'map_points.json'}")
    else:
        print(f"⚠ Warning: {map_src} not found")

    # Copy evidence files
    json_files = list(EVIDENCE_SRC.glob("*.json"))
    for f in json_files:
        shutil.copy2(f, WEB_EVIDENCE / f.name)
        print(f"✓ Copied {f.name} -> {WEB_EVIDENCE / f.name}")

    csv_files = list(EVIDENCE_SRC.glob("*.csv"))
    for f in csv_files:
        shutil.copy2(f, WEB_EVIDENCE / f.name)
        print(f"✓ Copied {f.name} -> {WEB_EVIDENCE / f.name}")

    print("Evidence sync complete!")


if __name__ == "__main__":
    sync()
