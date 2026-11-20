"""Evidence Constraint Filters (Section 3.4 & Section 4.4, Eqs. 31-33).

Implements the three formal problem constraints validated in Exp 2-4 and Exp 2-5:
  - Eq. (31) Coverage Constraint:       cosine(V_C, V_q) >= delta_cov
  - Eq. (32) Cross-Source Consistency:  ||(V_QA + V_law + V_case)/3 - V_cand|| <= epsilon
  - Eq. (33) Redundancy Penalty:        H(A_ref) - H(A_pred) <= delta_ent
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..config import ConstraintsConfig
from ..data.schemas import EvidenceSegment


def shannon_entropy(text: str) -> float:
    """Compute token-level Shannon entropy H(A) in bits (Eq. 33)."""
    tokens = re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", text.lower())
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


class EvidenceConstraintFilter:
    """Applies coverage (Eq. 31), cross-source consistency (Eq. 32), and redundancy (Eq. 33)."""

    def __init__(self, cfg: ConstraintsConfig) -> None:
        self.cfg = cfg

    def check_answer_constraints(
        self,
        predicted_answer: str,
        predicted_vec: np.ndarray,
        supporting_vecs: Sequence[np.ndarray],
        reference_answer: str | None = None,
        source_triplet_vecs: Tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> Dict[str, bool | float]:
        """Verify Eqs. (31)-(33) directly on a candidate or generated answer."""
        max_cov = max((cosine_sim(sv, predicted_vec) for sv in supporting_vecs), default=0.0)
        cov_ok = (not self.cfg.enable_coverage) or (max_cov >= self.cfg.delta_cov)

        cons_dev = 0.0
        cons_ok = True
        if self.cfg.enable_consistency and source_triplet_vecs is not None:
            centroid = (source_triplet_vecs[0] + source_triplet_vecs[1] + source_triplet_vecs[2]) / 3.0
            cons_dev = float(np.linalg.norm(centroid - predicted_vec))
            cons_ok = cons_dev <= self.cfg.epsilon

        ent_diff = 0.0
        red_ok = True
        if self.cfg.enable_redundancy and reference_answer is not None:
            ent_diff = shannon_entropy(reference_answer) - shannon_entropy(predicted_answer)
            red_ok = ent_diff <= self.cfg.delta_ent

        return {
            "coverage_score": max_cov,
            "coverage_ok": cov_ok,
            "consistency_deviation": cons_dev,
            "consistency_ok": cons_ok,
            "entropy_diff": ent_diff,
            "redundancy_ok": red_ok,
            "all_satisfied": bool(cov_ok and cons_ok and red_ok),
        }

    def filter_evidence_pool(
        self,
        query_vec: np.ndarray,
        candidates: Sequence[EvidenceSegment],
        node_vectors: Dict[str, np.ndarray],
    ) -> List[EvidenceSegment]:
        """Filter and de-duplicate retrieved evidence pool prior to Top-K budget capping."""
        if not candidates:
            return []

        ranked = sorted(candidates, key=lambda s: s.score, reverse=True)

        # Build multi-source centroid from top basic, applied, and contrastive hits (Eq. 32)
        source_reps: List[np.ndarray] = []
        for src_prefix in ("basic", "applied", "contrastive"):
            for seg in ranked:
                if seg.source_type.startswith(src_prefix) and seg.node_id in node_vectors:
                    source_reps.append(node_vectors[seg.node_id])
                    break
        centroid = np.mean(source_reps, axis=0) if len(source_reps) >= 2 else None

        kept: List[EvidenceSegment] = []
        kept_vecs: List[np.ndarray] = []
        best_entropy = max((shannon_entropy(s.text) for s in ranked[:3]), default=0.0)

        for seg in ranked:
            vec = node_vectors.get(seg.node_id)

            # Eq. (31): Coverage floor (always keep at least top-1 anchor)
            if self.cfg.enable_coverage and vec is not None and kept:
                sim_q = cosine_sim(query_vec, vec)
                # Relax slightly for graph-linked multi-hop & contrastive hits
                floor = self.cfg.delta_cov * 0.35
                if sim_q < floor and seg.score < floor:
                    continue

            # Eq. (32): Cross-source consistency check
            if self.cfg.enable_consistency and centroid is not None and vec is not None and len(kept) >= 2:
                dev = float(np.linalg.norm(centroid - vec))
                # Soft consistency bound scaled to unit-sphere distance (max L2 = 2.0)
                if dev > (self.cfg.epsilon + 1.0):
                    continue

            # Eq. (33): Redundancy / near-duplicate pruning
            if self.cfg.enable_redundancy and vec is not None:
                is_redundant = any(cosine_sim(vec, kv) >= 0.96 for kv in kept_vecs)
                if is_redundant:
                    continue
                if len(kept) >= 3 and (best_entropy - shannon_entropy(seg.text)) > (self.cfg.delta_ent * 6.0):
                    continue

            kept.append(seg)
            if vec is not None:
                kept_vecs.append(vec)

        return kept if kept else ranked[:1]
