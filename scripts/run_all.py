#!/usr/bin/env python3
"""Master pipeline runner — train, evaluate, map, and sync evidence.

Regenerates every artifact in outputs/ from data/ and syncs to web/public:
1. scripts/train_a1.py    -> outputs/a1_model.joblib
2. scripts/eval_a1.py     -> outputs/evidence/a1.json, a1_errors.csv
3. scripts/eval_a2.py     -> outputs/evidence/a2.json, a2_results.md, a2_samples.csv
4. scripts/build_map.py   -> outputs/map_points.json
5. scripts/sync_evidence.py -> web/public/evidence/* & web/public/map_points.json

Usage:
    python3 scripts/run_all.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_step(step_num: int, total_steps: int, title: str, script_name: str, args: list[str] = None):
    print("\n" + "=" * 70)
    print(f"STEP {step_num}/{total_steps}: {title} ({script_name})")
    print("=" * 70)
    t0 = time.time()
    cmd = [sys.executable, str(ROOT / "scripts" / script_name)] + (args or [])
    res = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"❌ Error in {script_name} (exit code {res.returncode})")
        sys.exit(res.returncode)
    print(f"✓ Step {step_num} completed in {elapsed:.1f}s")


def main():
    total_start = time.time()
    print("=" * 70)
    print("🚀 NLP SUPPORT SYSTEM — COMPLETE PIPELINE REGENERATION")
    print("=" * 70)

    steps = [
        ("Train Agent A1 (IntentAnalyst Classifier)", "train_a1.py", []),
        ("Evaluate Agent A1 on TEST Split", "eval_a1.py", []),
        ("Evaluate Agent A2 (ResponseSynthesizer) on TEST Split", "eval_a2.py", []),
        ("Generate 2D Projection Map (UMAP/t-SNE)", "build_map.py", []),
        ("Sync Evidence to Next.js Frontend", "sync_evidence.py", []),
    ]

    for idx, (title, script, args) in enumerate(steps, 1):
        run_step(idx, len(steps), title, script, args)

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"🎉 All pipeline stages successfully completed in {total_elapsed:.1f}s!")
    print("Artifacts generated in outputs/ and synced to web/public/")
    print("=" * 70)


if __name__ == "__main__":
    main()
