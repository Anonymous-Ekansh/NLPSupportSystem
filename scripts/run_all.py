"""Master runner – split, train, evaluate, generate examples.

Usage:  python scripts/run_all.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.split_data import run_split
from nlp.agent_a1 import train_a1
from nlp.agent_a2 import build_a2_index
from nlp.evaluate import evaluate
from nlp.examples import generate_examples


def main() -> None:
    print("=" * 60)
    print("STEP 1 / 5 – Data split")
    print("=" * 60)
    run_split()

    print("\n" + "=" * 60)
    print("STEP 2 / 5 – Train A1 (Intent Analyst)")
    print("=" * 60)
    train_a1()

    print("\n" + "=" * 60)
    print("STEP 3 / 5 – Build A2 index (Response Specialist)")
    print("=" * 60)
    build_a2_index()

    print("\n" + "=" * 60)
    print("STEP 4 / 5 – Evaluate on TEST")
    print("=" * 60)
    evaluate("test")

    print("\n" + "=" * 60)
    print("STEP 5 / 5 – Generate example traces")
    print("=" * 60)
    generate_examples()

    print("\n✅  All done.  Outputs in outputs/")


if __name__ == "__main__":
    main()
