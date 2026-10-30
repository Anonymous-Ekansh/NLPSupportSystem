"""Agent A1 – Intent Analyst with model ladder.

Ladder levels (all fit on TRAIN):
    L0  majority-class baseline
    L1  TF-IDF word + char_wb → LogisticRegression
    L2  embedding → LogisticRegression  (C tuned on DEV)
    L3  similarity-weighted kNN vote    (k, temperature tuned on DEV)

The SERVED model is chosen on DEV log-loss among {L1, L2, L3} (must need
only numpy + embedder at inference).  Its weights are exported to
outputs/models/a1.npz.  Temperature scaling is applied on DEV.

Contract:
    A1.analyze(query) → {
        query, intent, category, confidence, margin,
        top3: [{intent, prob}], top1_similarity,
        needs_clarification
    }
"""

import json
import os

import numpy as np

from nlp.normalize import normalize
from nlp.embed import embed, EMBED_MODEL

MODEL_DIR = "outputs/models"
CONFIG_PATH = os.path.join(MODEL_DIR, "a1_config.json")
NPZ_PATH = os.path.join(MODEL_DIR, "a1.npz")


def _load_config() -> dict:
    """Load thresholds and model metadata."""
    with open(CONFIG_PATH) as f:
        return json.load(f)


class IntentAnalyst:
    """Agent A1 – loaded from a1.npz (no scikit-learn at inference)."""

    def __init__(self):
        cfg = _load_config()
        blob = np.load(NPZ_PATH, allow_pickle=True)

        self.classes: list[str] = blob["classes"].tolist()
        self.intent_to_cat: dict = json.loads(str(blob["intent_to_cat"]))
        self.temperature: float = float(cfg["temperature"])
        self.tau: float = float(cfg["tau"])
        self.s_min: float = float(cfg["s_min"])
        self.served_level: str = cfg["served_level"]

        # Inference path depends on served level
        if self.served_level in ("L2", "L3"):
            self._embed_model = cfg.get("embed_model", EMBED_MODEL)

        if self.served_level == "L1":
            # TF-IDF + LR: we'd need the vectoriser — not pure numpy.
            # L1 is kept for comparison only; L2/L3 are the served candidates.
            raise RuntimeError("L1 cannot be served without scikit-learn.")

        elif self.served_level == "L2":
            self.W = blob["W"]      # (n_features, n_classes)
            self.b = blob["b"]      # (n_classes,)

        elif self.served_level == "L3":
            self.train_embs = blob["train_embs"]    # (N_train, 384)
            self.train_labels = blob["train_labels"] # (N_train,) int indices
            self.knn_k = int(cfg["knn_k"])
            self.knn_temp = float(cfg["knn_temp"])

    # ── softmax with temperature ──────────────────────────────────────

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        e = np.exp(logits - logits.max())
        return e / e.sum()

    # ── predict_proba ─────────────────────────────────────────────────

    def _predict_proba(self, q_emb: np.ndarray) -> np.ndarray:
        """Return calibrated class probabilities for one query embedding."""
        if self.served_level == "L2":
            logits = q_emb @ self.W + self.b          # (n_classes,)
            logits = logits / self.temperature
            return self._softmax(logits)

        elif self.served_level == "L3":
            # cosine similarities (inputs are L2-normed)
            sims = self.train_embs @ q_emb             # (N_train,)
            top_idx = np.argpartition(sims, -self.knn_k)[-self.knn_k:]
            top_sims = sims[top_idx]
            top_labels = self.train_labels[top_idx]

            # similarity-weighted soft vote with temperature
            weights = np.exp(top_sims / self.knn_temp)
            votes = np.zeros(len(self.classes), dtype=np.float64)
            for s, w, lbl in zip(top_sims, weights, top_labels):
                votes[lbl] += w

            # temperature-scale the log-votes
            log_votes = np.log(votes + 1e-30)
            log_votes = log_votes / self.temperature
            return self._softmax(log_votes)

        else:
            raise RuntimeError(f"Unknown served level: {self.served_level}")

    # ── analyze ───────────────────────────────────────────────────────

    def analyze(self, query: str) -> dict:
        """Classify a single query.  Returns the A1 contract dict."""
        normed = normalize(query)
        q_emb = embed([normed], model=self._embed_model)[0]  # (384,)

        probs = self._predict_proba(q_emb)
        ranked = np.argsort(probs)[::-1]

        intent = self.classes[ranked[0]]
        confidence = float(probs[ranked[0]])
        margin = float(probs[ranked[0]] - probs[ranked[1]])

        top3 = [
            {"intent": self.classes[ranked[i]],
             "prob": round(float(probs[ranked[i]]), 4)}
            for i in range(min(3, len(self.classes)))
        ]

        category = self.intent_to_cat.get(intent, "UNKNOWN")

        # top1_similarity: cosine to nearest TRAIN neighbour
        if hasattr(self, "train_embs"):
            all_sims = self.train_embs @ q_emb
        else:
            # For L2: load train embeddings for similarity
            blob = np.load(NPZ_PATH, allow_pickle=True)
            all_sims = blob["train_embs"] @ q_emb
        top1_similarity = float(np.max(all_sims))

        needs_clarification = (
            confidence < self.tau or top1_similarity < self.s_min
        )

        return {
            "query": query,
            "intent": intent,
            "category": category,
            "confidence": round(confidence, 4),
            "margin": round(margin, 4),
            "top3": top3,
            "top1_similarity": round(top1_similarity, 4),
            "needs_clarification": needs_clarification,
        }


if __name__ == "__main__":
    a1 = IntentAnalyst()
    result = a1.analyze("I want to cancel my subscription")
    print(json.dumps(result, indent=2))
