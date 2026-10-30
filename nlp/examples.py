"""Generate 5 end-to-end example traces and save to outputs/examples/.

Picks one query per category to show diversity.
"""

import json
import os

import pandas as pd

from nlp.agent_a1 import IntentAnalyst
from nlp.agent_a2 import ResponseSpecialist
from nlp.pipeline import run_pipeline

EXAMPLES_DIR = "outputs/examples"


def generate_examples(n: int = 5) -> list[dict]:
    """Pick n diverse test queries, run the pipeline, save traces."""
    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    df = pd.read_csv("outputs/splits/test.csv")

    # One per category for diversity, then truncate to n
    sampled = df.groupby("category").first().reset_index()
    sampled = sampled.head(n)

    a1 = IntentAnalyst()
    a2 = ResponseSpecialist()

    traces = []
    for _, row in sampled.iterrows():
        q = str(row["instruction"])
        result = run_pipeline(q, a1, a2)
        result["true_intent"] = row["intent"]
        result["true_category"] = row["category"]
        traces.append(result)

        print(f"\n--- Example: {row['category']} / {row['intent']} ---")
        print(f"  Query     : {q}")
        print(f"  A1 intent : {result['a1_output']['intent']} "
              f"(conf={result['a1_output']['confidence']:.2%})")
        print(f"  A2 resp   : {result['a2_output']['response'][:100]}…")
        print(f"  Self-check: {result['a2_output']['self_check']['check_passed']}")

    with open(os.path.join(EXAMPLES_DIR, "traces.json"), "w") as f:
        json.dump(traces, f, indent=2)

    # Human-readable markdown
    md_lines = ["# End-to-End Example Traces\n"]
    for i, t in enumerate(traces, 1):
        a1o = t["a1_output"]
        a2o = t["a2_output"]
        md_lines.append(f"## Example {i}: {t['true_category']} / {t['true_intent']}\n")
        md_lines.append(f"**Query:** {t['query']}\n")
        md_lines.append(f"**A1 → Intent:** `{a1o['intent']}` | "
                        f"**Category:** `{a1o['category']}` | "
                        f"**Confidence:** {a1o['confidence']:.2%} | "
                        f"**Margin:** {a1o['margin']:.4f}\n")
        md_lines.append(f"**Needs clarification:** {a1o['needs_clarification']}\n")
        md_lines.append(f"**A2 → Response:**\n> {a2o['response'][:300]}{'…' if len(a2o['response'])>300 else ''}\n")
        md_lines.append(f"**Retrieval score:** {a2o['retrieval_score']:.4f} | "
                        f"**Self-check passed:** {a2o['self_check']['check_passed']}\n")
        md_lines.append("---\n")

    with open(os.path.join(EXAMPLES_DIR, "traces.md"), "w") as f:
        f.write("\n".join(md_lines))

    print(f"\n[examples] Saved {len(traces)} traces to {EXAMPLES_DIR}/")
    return traces


if __name__ == "__main__":
    generate_examples()
