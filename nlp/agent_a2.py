"""Agent A2 – Response Specialist.

Retrieval via fastembed cosine similarity over training examples filtered
by A1's predicted intent.  Includes self-check.

Contract:
    A2.respond(query, a1_output) -> {response, supporting_examples,
      used_intents, retrieval_score, low_confidence,
      self_check: {placeholder_safe, source_similarity_ok,
                   response_intent_ok, check_passed, candidates_rejected}}
"""

import json
import os

import joblib
import numpy as np
import pandas as pd

from nlp.normalize import normalize
from nlp.embeddings import embed_texts

SEED = 42
MODEL_DIR = "outputs/models"
TOP_K = 5
LOW_CONF = 0.40
MIN_SIMILARITY = 0.55  # below this a retrieved candidate is suspicious


# ── Index building ───────────────────────────────────────────────────

def build_a2_index(train_csv: str = "outputs/splits/train.csv") -> None:
    """Embed all training instructions and save the index."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(train_csv)
    normed = df["instruction"].apply(normalize).tolist()

    print(f"[A2] Embedding {len(df)} training instructions …")
    matrix = embed_texts(normed)  # (N, 384)

    joblib.dump(matrix, os.path.join(MODEL_DIR, "a2_emb_matrix.joblib"))
    df.to_pickle(os.path.join(MODEL_DIR, "a2_train_df.pkl"))
    print("[A2] Index saved.")


# ── Inference ────────────────────────────────────────────────────────

class ResponseSpecialist:
    """Agent A2 – retrieval-based response with self-check."""

    def __init__(self):
        self.emb_matrix = joblib.load(os.path.join(MODEL_DIR, "a2_emb_matrix.joblib"))
        self.train_df = pd.read_pickle(os.path.join(MODEL_DIR, "a2_train_df.pkl"))
        # Precompute norms for cosine
        self._norms = np.linalg.norm(self.emb_matrix, axis=1, keepdims=True) + 1e-9
        self._normed_matrix = self.emb_matrix / self._norms

    def respond(self, query: str, a1_output: dict) -> dict:
        """Retrieve response and supporting examples.  Returns A2 contract."""
        predicted_intent = a1_output["intent"]
        confidence = a1_output["confidence"]
        low_confidence = confidence < LOW_CONF

        # Filter to predicted intent
        intent_mask = (self.train_df["intent"] == predicted_intent).values
        intent_idx = np.where(intent_mask)[0]
        if len(intent_idx) < 3:          # fallback to full corpus
            intent_idx = np.arange(len(self.train_df))

        # Cosine similarity
        q_emb = embed_texts([normalize(query)])[0]
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        sims = self._normed_matrix[intent_idx] @ q_norm  # (M,)

        top_local = np.argsort(sims)[::-1][:TOP_K]
        top_global = intent_idx[top_local]
        top_sims = sims[top_local].tolist()

        best_idx = top_global[0]
        best_sim = top_sims[0]
        response_text = str(self.train_df.iloc[best_idx]["response"])

        supporting = [
            {
                "instruction": str(self.train_df.iloc[int(top_global[i])]["instruction"]),
                "response": str(self.train_df.iloc[int(top_global[i])]["response"]),
                "similarity": round(top_sims[i], 4),
            }
            for i in range(min(TOP_K, len(top_global)))
        ]

        used_intents = list(
            self.train_df.iloc[top_global.tolist()]["intent"].unique()
        )

        # Self-check
        placeholder_safe = "{{" not in response_text
        source_similarity_ok = best_sim >= MIN_SIMILARITY
        response_intent_ok = (
            str(self.train_df.iloc[best_idx]["intent"]) == predicted_intent
        )
        candidates_rejected = sum(1 for s in top_sims if s < MIN_SIMILARITY)
        check_passed = placeholder_safe and source_similarity_ok and response_intent_ok

        return {
            "response": response_text,
            "supporting_examples": supporting,
            "used_intents": used_intents,
            "retrieval_score": round(best_sim, 4),
            "low_confidence": low_confidence,
            "self_check": {
                "placeholder_safe": placeholder_safe,
                "source_similarity_ok": source_similarity_ok,
                "response_intent_ok": response_intent_ok,
                "check_passed": check_passed,
                "candidates_rejected": candidates_rejected,
            },
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_a2_index()
    else:
        from nlp.agent_a1 import IntentAnalyst
        a1 = IntentAnalyst()
        a2 = ResponseSpecialist()
        q = "I want to cancel my subscription"
        out = a2.respond(q, a1.analyze(q))
        print(json.dumps(out, indent=2))
