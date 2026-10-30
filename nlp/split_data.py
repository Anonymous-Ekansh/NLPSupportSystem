"""Split raw data into TRAIN / DEV / TEST with deduplication and leak removal.

Procedure:
1. Deduplicate provided train by normalized instruction.
2. Remove from validation any row whose norm instruction is in train (leak).
3. Deduplicate validation.
4. Stratified 90/10 split of deduplicated train → TRAIN / DEV.
5. Cleaned validation → TEST.
6. Save outputs/splits/{train,dev,test}.csv and split_report.json.
"""

import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from nlp.normalize import normalize

SEED = 42


def run_split() -> dict:
    """Execute the full split pipeline.  Returns report dict."""
    os.makedirs("outputs/splits", exist_ok=True)

    raw_train = pd.read_csv("data/train.csv")
    raw_val = pd.read_csv("data/validation.csv")

    raw_train["_norm"] = raw_train["instruction"].apply(normalize)
    raw_val["_norm"] = raw_val["instruction"].apply(normalize)

    # --- deduplicate train ---
    train_deduped = raw_train.drop_duplicates(subset="_norm", keep="first").copy()

    # --- leak removal from validation ---
    train_norms = set(train_deduped["_norm"])
    val_clean = raw_val[~raw_val["_norm"].isin(train_norms)].copy()
    leaks_removed = len(raw_val) - len(val_clean)

    # --- deduplicate validation ---
    val_before = len(val_clean)
    val_clean = val_clean.drop_duplicates(subset="_norm", keep="first")
    val_dups_removed = val_before - len(val_clean)

    # --- stratified split of train ---
    tr, dv = train_test_split(
        train_deduped, test_size=0.10, random_state=SEED,
        stratify=train_deduped["intent"],
    )

    # --- drop helper col, save ---
    for df in (tr, dv, val_clean):
        df.drop(columns=["_norm"], inplace=True)

    tr.to_csv("outputs/splits/train.csv", index=False)
    dv.to_csv("outputs/splits/dev.csv", index=False)
    val_clean.to_csv("outputs/splits/test.csv", index=False)

    report = {
        "raw_train_rows": len(raw_train),
        "raw_validation_rows": len(raw_val),
        "train_duplicates_removed": len(raw_train) - len(train_deduped),
        "validation_leaks_removed": leaks_removed,
        "validation_duplicates_removed": val_dups_removed,
        "final_train": len(tr),
        "final_dev": len(dv),
        "final_test": len(val_clean),
        "train_intents": int(tr["intent"].nunique()),
        "train_categories": int(tr["category"].nunique()),
        "dev_intents": int(dv["intent"].nunique()),
        "dev_categories": int(dv["category"].nunique()),
        "test_intents": int(val_clean["intent"].nunique()),
        "test_categories": int(val_clean["category"].nunique()),
    }
    with open("outputs/splits/split_report.json", "w") as f:
        json.dump(report, f, indent=2)

    for k, v in report.items():
        print(f"  {k:35s}: {v}")
    return report


if __name__ == "__main__":
    run_split()
