"""Agent A2 — Response Specialist with retrieval ladder and self-check.

Retrieval modes:
    sparse  – TF-IDF cosine similarity
    dense   – embedding cosine similarity
    hybrid  – reciprocal-rank fusion (k=60) of sparse + dense

Routing:
    If A1.needs_clarification is False → filter to top-1 intent.
    If True → filter to A1's top-3 intents and weight each candidate's
    score by A1's probability for its intent:

        final_score(candidate) = retrieval_score(candidate)
                                 × a1_prob(candidate.intent)

    This gives preference to candidates whose intent the classifier is
    more confident about, while still exploring the top-3 hypotheses.

Self-check pipeline (applied in order):
    1. placeholder_safe:     no unresolved {{...}} in response
    2. source_similarity_ok: cosine(query_emb, source_instruction_emb) ≥ SIM_MIN
    3. response_intent_ok:   a TF-IDF LogReg trained on TRAIN *responses*
                             predicts the searched intent

The first candidate passing all three checks is returned.  Otherwise the
best-ranked candidate is returned with check_passed=False.

Contract:
    A2.respond(query, a1_output) → {
        response, supporting_examples: [{instruction, response, similarity}],
        used_intents, retrieval_score, low_confidence,
        self_check: {placeholder_safe, source_similarity_ok,
                     response_intent_ok, check_passed, candidates_rejected}
    }
"""

import json
import os
import re
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

from nlp.normalize import normalize
from nlp.embed import embed, embed_split

SEED = 42
MODEL_DIR = "outputs/models"
TOP_K = 5
RRF_K = 60  # reciprocal rank fusion constant


def _extract_placeholders(text: str) -> set[str]:
    """Extract {{placeholder}} names from text."""
    return set(re.findall(r"\{\{([^}]+)\}\}", text))


# ═══════════════════════════════════════════════════════════════════════
# Builder: fits everything from TRAIN data
# ═══════════════════════════════════════════════════════════════════════

class A2Builder:
    """Build A2 artefacts from TRAIN split."""

    def __init__(self, embed_model: str = "BAAI/bge-small-en-v1.5"):
        self.embed_model = embed_model

    def build(self, sim_min_dev: Optional[float] = None) -> "ResponseSpecialist":
        """Fit all components, reusing precomputed TRAIN embedding matrix."""
        train_df = pd.read_csv("outputs/splits/train.csv")
        train_texts = train_df["instruction"].apply(normalize).tolist()
        train_responses = train_df["response"].astype(str).tolist()
        train_intents = train_df["intent"].values

        # Dense embeddings: load precomputed matrix if available to prevent Render OOM
        emb_matrix_path = "outputs/models/a2_emb_matrix.joblib"
        if os.path.exists(emb_matrix_path):
            import joblib
            print(f"[A2] Loading precomputed TRAIN embeddings from {emb_matrix_path} …")
            train_emb = joblib.load(emb_matrix_path)
        else:
            print("[A2] Loading / computing TRAIN embeddings …")
            train_emb = embed_split("train", model=self.embed_model)

        # Sparse TF-IDF (float32: halves resident RAM vs sklearn's default
        # float64; identical scores after 4-dp rounding)
        print("[A2] Fitting TF-IDF …")
        tfidf = TfidfVectorizer(
            ngram_range=(1, 2), max_features=30_000,
            sublinear_tf=True, min_df=2,
        )
        train_tfidf = tfidf.fit_transform(train_texts).astype(np.float32)

        # Response→intent classifier (for self-check)
        print("[A2] Training response→intent classifier …")
        resp_tfidf_vec = TfidfVectorizer(
            ngram_range=(1, 2), max_features=20_000,
            sublinear_tf=True, min_df=2,
        )
        X_resp = resp_tfidf_vec.fit_transform(train_responses).astype(np.float32)
        del train_responses
        resp_clf = LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs",
            random_state=SEED,
        )
        resp_clf.fit(X_resp, train_intents)
        train_acc = (resp_clf.predict(X_resp) == train_intents).mean()
        del X_resp                                      # free the 25k×20k matrix
        print(f"  Response→intent train acc: {train_acc:.4f}")

        # Choose SIM_MIN on DEV if not provided (default: 0.9107 precomputed on DEV)
        if sim_min_dev is None:
            sim_min_dev = 0.9107

        specialist = ResponseSpecialist(
            train_df=train_df,
            train_texts=train_texts,
            train_emb=train_emb,
            train_tfidf=train_tfidf,
            tfidf_vec=tfidf,
            resp_tfidf_vec=resp_tfidf_vec,
            resp_clf=resp_clf,
            embed_model=self.embed_model,
            sim_min=sim_min_dev,
        )
        print(f"[A2] Ready. SIM_MIN={sim_min_dev:.4f}")
        import gc
        gc.collect()                     # return fit transients before serving
        return specialist

    def _choose_sim_min(self, train_emb, train_intents, train_df):
        """Pick SIM_MIN on DEV: highest value giving >= 95% of candidates accepted."""
        dev_df = pd.read_csv("outputs/splits/dev.csv")
        dev_texts = dev_df["instruction"].apply(normalize).tolist()
        dev_emb = embed_split("dev", model=self.embed_model)

        # For each DEV query, find nearest TRAIN example of same intent
        sims = []
        for i, row in dev_df.iterrows():
            intent = row["intent"]
            mask = train_intents == intent
            if mask.sum() == 0:
                continue
            idx = np.where(mask)[0]
            cos = dev_emb[i] @ train_emb[idx].T
            sims.append(cos.max())

        sims = np.array(sims)
        # Pick 5th percentile as SIM_MIN (flags ~5% worst matches)
        sim_min = float(np.percentile(sims, 5))
        print(f"[A2] DEV same-intent sim: "
              f"min={sims.min():.4f} p5={sim_min:.4f} mean={sims.mean():.4f}")
        return round(sim_min, 4)


# ═══════════════════════════════════════════════════════════════════════
# ResponseSpecialist
# ═══════════════════════════════════════════════════════════════════════

class ResponseSpecialist:
    """Agent A2 – retrieval-based response with self-check."""

    def __init__(
        self,
        train_df: pd.DataFrame,
        train_texts: list[str],
        train_emb: np.ndarray,
        train_tfidf: csr_matrix,
        tfidf_vec: TfidfVectorizer,
        resp_tfidf_vec: TfidfVectorizer,
        resp_clf: LogisticRegression,
        embed_model: str,
        sim_min: float = 0.70,
    ):
        self.train_df = train_df            # read_csv starts at index 0; no copy needed
        self.train_texts = train_texts
        self.train_emb = train_emb           # (N, 384) L2-normed
        self.train_tfidf = train_tfidf       # (N, F) sparse
        self.tfidf_vec = tfidf_vec
        self.resp_tfidf_vec = resp_tfidf_vec
        self.resp_clf = resp_clf
        self.embed_model = embed_model
        self.sim_min = sim_min

        self.intents = self.train_df["intent"].values
        self.responses = self.train_df["response"].astype(str).values

        # Precompute TF-IDF row norms for cosine
        norms = np.sqrt(self.train_tfidf.multiply(self.train_tfidf).sum(axis=1))
        self.train_tfidf_norms = np.asarray(norms).ravel() + 1e-9

    # ── Retrieval primitives ──────────────────────────────────────────

    def _sparse_scores(self, query: str, indices: np.ndarray) -> np.ndarray:
        """TF-IDF cosine similarity for query against indexed rows."""
        q_vec = self.tfidf_vec.transform([query])  # (1, F)
        sub = self.train_tfidf[indices]  # (M, F)
        dots = np.asarray(sub.dot(q_vec.T).todense()).ravel()
        q_norm = np.sqrt(q_vec.multiply(q_vec).sum()) + 1e-9
        sub_norms = self.train_tfidf_norms[indices]
        return dots / (sub_norms * q_norm)

    def _dense_scores(self, q_emb: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Embedding cosine similarity (inputs already L2-normed)."""
        return self.train_emb[indices] @ q_emb

    def _hybrid_scores(
        self, query: str, q_emb: np.ndarray, indices: np.ndarray
    ) -> np.ndarray:
        """Reciprocal Rank Fusion of sparse and dense rankings.

        RRF(d) = 1/(k + rank_sparse(d)) + 1/(k + rank_dense(d))
        where k = RRF_K = 60.
        """
        sparse = self._sparse_scores(query, indices)
        dense = self._dense_scores(q_emb, indices)

        rank_s = np.argsort(np.argsort(-sparse))  # 0-based rank
        rank_d = np.argsort(np.argsort(-dense))

        rrf = 1.0 / (RRF_K + rank_s) + 1.0 / (RRF_K + rank_d)
        return rrf

    # ── Self-check ────────────────────────────────────────────────────

    def _self_check(
        self, response_text: str, source_instruction: str,
        q_emb: np.ndarray, searched_intent: str,
    ) -> dict:
        """Run the 3-stage self-check on a single candidate."""
        # 1. placeholder_safe
        placeholder_safe = "{{" not in response_text

        # 2. source_similarity_ok
        src_emb = embed([normalize(source_instruction)], model=self.embed_model)[0]
        sim = float(q_emb @ src_emb)
        source_similarity_ok = sim >= self.sim_min

        # 3. response_intent_ok
        resp_vec = self.resp_tfidf_vec.transform([response_text])
        pred_intent = self.resp_clf.predict(resp_vec)[0]
        response_intent_ok = pred_intent == searched_intent

        check_passed = placeholder_safe and source_similarity_ok and response_intent_ok
        return {
            "placeholder_safe": bool(placeholder_safe),
            "source_similarity_ok": bool(source_similarity_ok),
            "source_similarity": round(sim, 4),
            "response_intent_ok": bool(response_intent_ok),
            "response_predicted_intent": str(pred_intent),
            "check_passed": bool(check_passed),
        }

    # ── Main respond ──────────────────────────────────────────────────

    def respond(
        self,
        query: str,
        a1_output: dict,
        mode: str = "hybrid",
        k: int = TOP_K,
    ) -> dict:
        """Retrieve response using A1 routing.

        Routing logic:
            - If needs_clarification=False: filter to top-1 intent.
            - If needs_clarification=True: filter to top-3 intents,
              weight each candidate's score by A1's probability for its
              intent.  This favours candidates from more-confident intents
              while still exploring alternatives.

        Re-ranking:
            Candidates whose {{placeholders}} are a subset of the query's
            placeholders come first.

        Self-check:
            Applied in order: placeholder_safe, source_similarity_ok,
            response_intent_ok.  First passing candidate is returned;
            otherwise best-ranked with check_passed=False.
        """
        normed = normalize(query)
        q_emb = embed([normed], model=self.embed_model)[0]

        needs_clar = a1_output.get("needs_clarification", False)
        top3 = a1_output.get("top3", [])
        predicted_intent = a1_output["intent"]

        if not needs_clar:
            # Filter to top-1 intent
            filter_intents = {predicted_intent}
            intent_probs = {predicted_intent: 1.0}
            low_confidence = False
        else:
            # Filter to top-3 intents
            filter_intents = {t["intent"] for t in top3[:3]}
            intent_probs = {t["intent"]: t["prob"] for t in top3[:3]}
            low_confidence = True

        # Get candidate indices
        mask = np.isin(self.intents, list(filter_intents))
        indices = np.where(mask)[0]
        if len(indices) < 3:
            # Fallback to full corpus
            indices = np.arange(len(self.train_df))
            filter_intents = set(self.intents)

        # Score
        if mode == "sparse":
            scores = self._sparse_scores(normed, indices)
        elif mode == "dense":
            scores = self._dense_scores(q_emb, indices)
        elif mode == "hybrid":
            scores = self._hybrid_scores(normed, q_emb, indices)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Weight by A1 probability if multi-intent
        if needs_clar and len(filter_intents) > 1:
            cand_intents = self.intents[indices]
            for j, ci in enumerate(cand_intents):
                prob = intent_probs.get(ci, 0.01)
                scores[j] *= prob

        # Top-K
        n_cand = min(k * 3, len(scores))  # get more for re-ranking
        top_local = np.argsort(scores)[::-1][:n_cand]
        top_global = indices[top_local]
        top_scores = scores[top_local]

        # Re-rank: prefer candidates whose placeholders ⊆ query placeholders
        query_phs = _extract_placeholders(query)
        def _ph_priority(idx):
            resp_phs = _extract_placeholders(str(self.responses[idx]))
            return 0 if resp_phs.issubset(query_phs) else 1

        rerank_order = sorted(
            range(len(top_global)),
            key=lambda j: (_ph_priority(top_global[j]), -top_scores[j]),
        )
        top_global = top_global[rerank_order]
        top_scores = top_scores[rerank_order]

        # Self-check: find first passing candidate
        candidates_rejected = 0
        best_idx = int(top_global[0])
        best_score = float(top_scores[0])
        best_check = None
        chosen_idx = None

        searched_intent = predicted_intent

        for j in range(min(k, len(top_global))):
            g = int(top_global[j])
            check = self._self_check(
                str(self.responses[g]),
                str(self.train_df.iloc[g]["instruction"]),
                q_emb, searched_intent,
            )
            if j == 0:
                best_check = check
                best_idx = g
                best_score = float(top_scores[j])

            if check["check_passed"]:
                chosen_idx = g
                chosen_score = float(top_scores[j])
                chosen_check = check
                break
            else:
                candidates_rejected += 1

        if chosen_idx is not None:
            final_idx = chosen_idx
            final_score = chosen_score
            final_check = chosen_check
        else:
            final_idx = best_idx
            final_score = best_score
            final_check = best_check

        final_check["candidates_rejected"] = candidates_rejected

        # Supporting examples (top 3 unique)
        seen = set()
        supporting = []
        for j in range(min(len(top_global), k)):
            g = int(top_global[j])
            instr = str(self.train_df.iloc[g]["instruction"])
            if instr in seen:
                continue
            seen.add(instr)
            supporting.append({
                "instruction": instr,
                "response": str(self.responses[g]),
                "similarity": round(float(top_scores[j]), 4),
            })
            if len(supporting) >= 3:
                break

        used_intents = list(set(
            self.intents[top_global[:min(k, len(top_global))].tolist()]
        ))

        return {
            "response": str(self.responses[final_idx]),
            "supporting_examples": supporting,
            "used_intents": used_intents,
            "retrieval_score": round(final_score, 4),
            "low_confidence": low_confidence,
            "self_check": final_check,
        }

    # ── Global respond (no A1 routing) ────────────────────────────────

    def respond_global(
        self, query: str, mode: str = "hybrid", k: int = TOP_K
    ) -> dict:
        """Retrieve response without A1 routing — search full corpus."""
        normed = normalize(query)
        q_emb = embed([normed], model=self.embed_model)[0]
        indices = np.arange(len(self.train_df))

        if mode == "sparse":
            scores = self._sparse_scores(normed, indices)
        elif mode == "dense":
            scores = self._dense_scores(q_emb, indices)
        elif mode == "hybrid":
            scores = self._hybrid_scores(normed, q_emb, indices)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        top_local = np.argsort(scores)[::-1][:k]
        top_global = indices[top_local]
        top_scores = scores[top_local]

        best_g = int(top_global[0])
        best_score = float(top_scores[0])

        supporting = []
        for j in range(min(3, len(top_global))):
            g = int(top_global[j])
            supporting.append({
                "instruction": str(self.train_df.iloc[g]["instruction"]),
                "response": str(self.responses[g]),
                "similarity": round(float(top_scores[j]), 4),
            })

        return {
            "response": str(self.responses[best_g]),
            "supporting_examples": supporting,
            "retrieval_score": round(best_score, 4),
            "used_intents": list(set(self.intents[top_global.tolist()])),
        }


if __name__ == "__main__":
    from nlp.a1 import IntentAnalyst
    a1 = IntentAnalyst()
    a2 = A2Builder().build()
    q = "I want to cancel my subscription"
    a1_out = a1.analyze(q)
    result = a2.respond(q, a1_out)
    print(json.dumps(result, indent=2, default=str))
