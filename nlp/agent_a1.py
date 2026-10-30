"""Agent A1 – Intent Analyst.

Model: TF-IDF (1,2)-grams + multinomial Logistic Regression.
Also stores per-intent mean embeddings to compute top1_similarity.

Contract:
    A1.analyze(query) -> {query, intent, category, confidence, margin,
                          top3, top1_similarity, needs_clarification}
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from nlp.normalize import normalize
from nlp.embeddings import embed_texts

SEED = 42
MODEL_DIR = "outputs/models"
CONFIDENCE_THRESHOLD = 0.40


# ── Training ─────────────────────────────────────────────────────────

def train_a1(train_csv: str = "outputs/splits/train.csv") -> None:
    """Train TF-IDF + LogReg + per-intent centroids.  Saves to MODEL_DIR."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(train_csv)

    X_text = df["instruction"].apply(normalize).tolist()
    y = df["intent"]

    # Intent -> category mapping (majority vote)
    intent_to_cat = (
        df.groupby("intent")["category"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )

    # TF-IDF + LogReg pipeline
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), max_features=30_000,
            sublinear_tf=True, min_df=2,
        )),
        ("clf", LogisticRegression(
            max_iter=1000, C=5.0, solver="lbfgs",
            random_state=SEED, n_jobs=-1,
        )),
    ])
    pipe.fit(X_text, y)
    print(f"[A1] Classifier trained on {len(df)} rows, {y.nunique()} intents.")

    # Per-intent mean embeddings (for top1_similarity)
    print("[A1] Computing per-intent centroids …")
    all_embs = embed_texts(X_text)
    intent_labels = y.values
    unique_intents = sorted(y.unique())
    centroids = {}
    for intent in unique_intents:
        mask = intent_labels == intent
        centroids[intent] = all_embs[mask].mean(axis=0)
    centroid_matrix = np.stack([centroids[i] for i in unique_intents])  # (27, 384)

    joblib.dump(pipe, os.path.join(MODEL_DIR, "a1_pipe.joblib"))
    joblib.dump(intent_to_cat, os.path.join(MODEL_DIR, "a1_intent_to_cat.joblib"))
    joblib.dump({"intents": unique_intents, "centroids": centroid_matrix},
                os.path.join(MODEL_DIR, "a1_centroids.joblib"))
    print("[A1] Saved.")


# ── Inference ────────────────────────────────────────────────────────

class IntentAnalyst:
    """Agent A1 – loaded and ready for .analyze(query)."""

    def __init__(self):
        self.pipe = joblib.load(os.path.join(MODEL_DIR, "a1_pipe.joblib"))
        self.intent_to_cat = joblib.load(os.path.join(MODEL_DIR, "a1_intent_to_cat.joblib"))
        blob = joblib.load(os.path.join(MODEL_DIR, "a1_centroids.joblib"))
        self.centroid_intents = blob["intents"]
        self.centroids = blob["centroids"]          # (27, 384)
        self.classes = list(self.pipe.classes_)

    def analyze(self, query: str) -> dict:
        """Classify a single query.  Returns the A1 contract dict."""
        normed = normalize(query)

        # Classification
        probs = self.pipe.predict_proba([normed])[0]
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

        # Similarity to predicted-intent centroid
        q_emb = embed_texts([normed])[0]
        idx = self.centroid_intents.index(intent)
        top1_similarity = float(
            np.dot(q_emb, self.centroids[idx])
            / (np.linalg.norm(q_emb) * np.linalg.norm(self.centroids[idx]) + 1e-9)
        )

        needs_clarification = confidence < CONFIDENCE_THRESHOLD

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
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_a1()
    else:
        a1 = IntentAnalyst()
        print(json.dumps(a1.analyze("I want to cancel my subscription"), indent=2))
