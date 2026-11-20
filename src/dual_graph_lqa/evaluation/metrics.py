"""Exact Section 3 Evaluation Metrics (Eqs. 1-29).

Implements:
  - Answerability TP/FP/FN/TN, Precision, Recall, F1 (Eqs. 1-13)
  - Article-level TP/FP/FN/TN, Precision, Recall, F1 (Eqs. 14-26)
  - Semantic similarity rho_i and overall rho (Eqs. 27-28)
  - Joint performance objective P_M = alpha*F1_ans + beta*F1_art + gamma*rho (Eq. 29)
"""

from __future__ import annotations

from typing import Dict, Sequence, Set

import numpy as np


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def compute_answerability_metrics(
    y_true: Sequence[bool | int], y_pred: Sequence[bool | int]
) -> Dict[str, float]:
    """Eqs. (1)-(13): Answerability classification precision, recall, and F1."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length.")
    tp = fp = fn = tn = 0
    for yt, yp in zip(y_true, y_pred):
        gt, pr = int(bool(yt)), int(bool(yp))
        tp += gt * pr
        fp += (1 - gt) * pr
        fn += gt * (1 - pr)
        tn += (1 - gt) * (1 - pr)

    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * prec * rec, prec + rec)
    return {
        "TP": float(tp),
        "FP": float(fp),
        "FN": float(fn),
        "TN": float(tn),
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def compute_article_level_metrics(
    gold_article_sets: Sequence[Set[str]],
    pred_article_sets: Sequence[Set[str]],
    universe_articles: Set[str] | None = None,
) -> Dict[str, float]:
    """Eqs. (14)-(26): Article-level citation TP/FP/FN/TN, Precision, Recall, F1."""
    if len(gold_article_sets) != len(pred_article_sets):
        raise ValueError("gold_article_sets and pred_article_sets must have equal length.")
    if universe_articles is None:
        universe_articles = set().union(*gold_article_sets, *pred_article_sets)

    tp = fp = fn = tn = 0
    for gold, pred in zip(gold_article_sets, pred_article_sets):
        for art in universe_articles:
            c_gt = 1 if art in gold else 0
            c_pr = 1 if art in pred else 0
            tp += c_gt * c_pr
            fp += (1 - c_gt) * c_pr
            fn += c_gt * (1 - c_pr)
            tn += (1 - c_gt) * (1 - c_pr)

    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * prec * rec, prec + rec)
    return {
        "TP": float(tp),
        "FP": float(fp),
        "FN": float(fn),
        "TN": float(tn),
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def compute_semantic_similarity(
    gold_embeddings: np.ndarray, pred_embeddings: np.ndarray
) -> Dict[str, float | list]:
    """Eqs. (27)-(28): Cosine semantic similarity rho_i and mean rho."""
    if gold_embeddings.shape != pred_embeddings.shape:
        raise ValueError("gold_embeddings and pred_embeddings must have matching shape.")
    if len(gold_embeddings) == 0:
        return {"rho": 0.0, "per_query": []}
    g_norms = np.maximum(np.linalg.norm(gold_embeddings, axis=1), 1e-12)
    p_norms = np.maximum(np.linalg.norm(pred_embeddings, axis=1), 1e-12)
    per_q = np.sum(gold_embeddings * pred_embeddings, axis=1) / (g_norms * p_norms)
    return {"rho": float(np.mean(per_q)), "per_query": [float(x) for x in per_q]}


def compute_joint_objective(
    f1_answerability: float,
    f1_article: float,
    semantic_rho: float,
    alpha: float = 0.30,
    beta: float = 0.40,
    gamma: float = 0.30,
) -> float:
    """Eq. (29): P_M = alpha * F1_answerability + beta * F1_article + gamma * rho."""
    if abs((alpha + beta + gamma) - 1.0) > 1e-6:
        raise ValueError("Weights alpha, beta, gamma must satisfy alpha + beta + gamma = 1.")
    return float(alpha * f1_answerability + beta * f1_article + gamma * semantic_rho)
