"""SVM + Platt Scaling Domain Router (Section 4.1).

Routes incoming legal queries to the highest-probability domain partition
(labor, insurance, financial) to narrow search space and improve precision.
Uses scikit-learn SVC(probability=True) (SMO + Platt scaling) when installed,
with a deterministic TF-IDF + calibrated linear-margin NumPy fallback otherwise.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import SVC

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False


def _tokenize_mixed(text: str) -> List[str]:
    """Tokenize mixed English + CJK strings cleanly."""
    tokens = re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", text.lower())
    return tokens


class SVMDomainRouter:
    """Platt-calibrated SVM classifier for legal domain routing."""

    def __init__(self, svm_c: float = 1.0, min_confidence: float = 0.35) -> None:
        self.svm_c = svm_c
        self.min_confidence = min_confidence
        self.classes_: List[str] = []
        self._fitted = False

        # sklearn backend
        self._vectorizer = None
        self._svc = None

        # lightweight fallback backend (TF-IDF centroids + Platt sigmoid)
        self._vocab: Dict[str, int] = {}
        self._idf: np.ndarray | None = None
        self._centroids: np.ndarray | None = None

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "SVMDomainRouter":
        if not texts or len(texts) != len(labels):
            raise ValueError("texts and labels must be non-empty and equal length.")
        self.classes_ = sorted(set(labels))
        if len(self.classes_) < 2:
            self._fitted = True
            return self

        if _HAS_SKLEARN:
            self._vectorizer = TfidfVectorizer(
                tokenizer=_tokenize_mixed, token_pattern=None, ngram_range=(1, 2)
            )
            X = self._vectorizer.fit_transform(texts)
            self._svc = SVC(kernel="linear", C=self.svm_c, probability=True, random_state=42)
            self._svc.fit(X, labels)
            self.classes_ = list(self._svc.classes_)
            self._fitted = True
            return self

        # Pure-NumPy TF-IDF + sigmoid-calibrated margin fallback
        docs = [_tokenize_mixed(t) for t in texts]
        df: Counter[str] = Counter()
        for doc in docs:
            df.update(set(doc))
        self._vocab = {tok: idx for idx, tok in enumerate(sorted(df.keys()))}
        n_docs = len(docs)
        self._idf = np.array(
            [math.log((1.0 + n_docs) / (1.0 + df[tok])) + 1.0 for tok in sorted(df.keys())],
            dtype=np.float32,
        )
        mat = self._transform_numpy(docs)
        centroids = []
        for cls in self.classes_:
            mask = [i for i, y in enumerate(labels) if y == cls]
            c = mat[mask].mean(axis=0)
            c /= max(float(np.linalg.norm(c)), 1e-12)
            centroids.append(c)
        self._centroids = np.stack(centroids, axis=0)
        self._fitted = True
        return self

    def _transform_numpy(self, docs: Sequence[Sequence[str]]) -> np.ndarray:
        mat = np.zeros((len(docs), len(self._vocab)), dtype=np.float32)
        for i, doc in enumerate(docs):
            counts = Counter(doc)
            for tok, cnt in counts.items():
                if tok in self._vocab:
                    j = self._vocab[tok]
                    mat[i, j] = (1.0 + math.log(cnt)) * self._idf[j]
            norm = float(np.linalg.norm(mat[i]))
            if norm > 1e-12:
                mat[i] /= norm
        return mat

    def predict_proba(self, query: str) -> Dict[str, float]:
        if not self._fitted:
            raise RuntimeError("SVMDomainRouter must be fitted before calling predict_proba.")
        if len(self.classes_) == 1:
            return {self.classes_[0]: 1.0}

        if _HAS_SKLEARN and self._svc is not None and self._vectorizer is not None:
            X = self._vectorizer.transform([query])
            probs = self._svc.predict_proba(X)[0]
            return {cls: float(p) for cls, p in zip(self.classes_, probs)}

        q_vec = self._transform_numpy([_tokenize_mixed(query)])[0]
        margins = self._centroids @ q_vec  # cosine margins
        # Platt-style logistic calibration (temperature = 0.25)
        scaled = (margins - np.max(margins)) / 0.25
        exp_v = np.exp(scaled)
        probs = exp_v / np.sum(exp_v)
        return {cls: float(p) for cls, p in zip(self.classes_, probs)}

    def route(self, query: str) -> Tuple[str | None, Dict[str, float]]:
        """Return the top legal domain if confidence >= min_confidence, else None (search all)."""
        probs = self.predict_proba(query)
        best_domain, best_prob = max(probs.items(), key=lambda kv: kv[1])
        if best_prob < self.min_confidence:
            return None, probs
        return best_domain, probs
