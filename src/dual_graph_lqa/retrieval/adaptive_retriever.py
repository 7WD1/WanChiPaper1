"""Section 4.5: Uncertainty-Gated Adaptive Dual-KG Retriever (Eqs. 51-77, Algorithm 5 / Table 6).

Faithfully implements:
  - Eqs. (51)-(60): One-hop dynamic thresholding theta_basic(Q), theta_applied(Q) = mu + lambda*sigma
  - Eq.  (75):      Normalized similarity-distribution entropy uncertainty U(Q) in [0, 1]
  - Eq.  (76):      Bounded hop-depth stopping rule h*(Q) in {1, 2} (H_max = 2)
  - Eqs. (61)-(67): Hop-decayed multi-hop expansion s_u^mh = max[eta^hop(u,v) * Sim(q, u)]
  - Eqs. (68)-(74): Contrastively aligned retrieval R_ctr(Q) with adaptive threshold theta_ctr(Q)
  - Eq.  (77):      Decoupling of dynamic thresholding from Top-K upper context-budget cap
                    |R_final(Q)| = min(|R_pool(Q)|, K)

Also exposes explicit ablation switches matching Modules 1, 2, and 4:
  - use_basic_kg, use_applied_kg, use_contrastive (Exp 1-4, 1-5, 1-6)
  - use_multihop (Exp 2-2)
  - threshold_mode: "adaptive" | "fixed_threshold" | "fixed_topk" (Exp 2-3 & Exp 4-1 vs Exp 4-2)
  - constraints config flags (Exp 2-4, Exp 2-5)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import RetrievalConfig
from ..contrastive.trainer import ContrastiveTrainer, UnifiedGraphIndex
from ..data.schemas import EvidenceSegment
from ..kg.applied_kg import AppliedKnowledgeGraph
from ..kg.basic_kg import BasicKnowledgeGraph
from ..kg.domain_router import SVMDomainRouter
from .constraints import EvidenceConstraintFilter


@dataclass
class RetrievalTrace:
    """Diagnostic trace recording uncertainty U(Q), chosen hop depth h*(Q), and retained count."""

    query: str
    routed_domain: Optional[str]
    domain_probs: Dict[str, float]
    uncertainty_u: float
    selected_hop_depth: int
    theta_basic: float
    theta_applied: float
    theta_mh: float
    theta_ctr: float
    num_basic_hits: int
    num_applied_1hop_hits: int
    num_applied_mh_hits: int
    num_contrastive_hits: int
    num_final_retained: int
    top_k_cap: int
    evidence: List[EvidenceSegment] = field(default_factory=list)
    threshold_mode: str = "adaptive"


def compute_query_uncertainty(scores: np.ndarray, top_m: int = 20, temp: float = 0.1) -> float:
    """Eq. (75): Normalized Shannon entropy U(Q) in [0, 1] over top-M similarity scores."""
    if scores.size <= 1:
        return 0.0
    m = min(top_m, scores.size)
    top_scores = np.sort(scores)[::-1][:m]
    scaled = (top_scores - np.max(top_scores)) / max(temp, 1e-6)
    exp_s = np.exp(scaled)
    probs = exp_s / np.sum(exp_s)
    entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12))))
    return float(entropy / max(math.log(m), 1e-12))


class AdaptiveDualKGRetriever:
    """Complete inference-stage Multi-KG Retriever (Algorithm 5 / Table 6)."""

    def __init__(
        self,
        basic_kg: BasicKnowledgeGraph,
        applied_kg: AppliedKnowledgeGraph,
        graph_index: UnifiedGraphIndex,
        contrastive_trainer: ContrastiveTrainer,
        cfg: RetrievalConfig,
        domain_router: Optional[SVMDomainRouter] = None,
    ) -> None:
        self.basic_kg = basic_kg
        self.applied_kg = applied_kg
        self.index = graph_index
        self.trainer = contrastive_trainer
        self.cfg = cfg
        self.domain_router = domain_router
        self.constraint_filter = EvidenceConstraintFilter(cfg.constraints)

        # Precompute contrastive and raw node vector lookups for constraint verification
        self._ctr_node_vecs: Dict[str, np.ndarray] = {
            nid: self.index.contrastive_embeddings[idx]
            for idx, nid in enumerate(self.index.node_ids)
        }
        self._raw_node_vecs: Dict[str, np.ndarray] = {
            nid: self.index.raw_node_embeddings[idx]
            for idx, nid in enumerate(self.index.node_ids)
        }
        self._node_vecs: Dict[str, np.ndarray] = self._ctr_node_vecs

    def _dynamic_threshold(self, scores: np.ndarray, lam: float, threshold_mode: str) -> float:
        if scores.size == 0:
            return 1.0
        if threshold_mode == "fixed_topk":
            return 0.0
        if threshold_mode == "fixed_threshold":
            return float(self.cfg.fixed_threshold_fallback)
        mu = float(np.mean(scores))
        sigma = float(np.std(scores))
        return mu + lam * sigma

    def retrieve(
        self,
        query: str,
        *,
        use_basic_kg: bool = True,
        use_applied_kg: bool = True,
        use_contrastive: bool = True,
        use_multihop: bool = True,
        use_adaptive_threshold: bool = True,
        threshold_mode: Optional[str] = None,
        use_constraints: bool = True,
        top_k_cap: Optional[int] = None,
    ) -> RetrievalTrace:
        """Execute Steps 1-5 of Algorithm 5 with full ablation configurability.

        Args:
            threshold_mode: One of ``"adaptive"`` (mu + lambda*sigma + entropy gate + Top-K cap,
                Full / Exp 4-2), ``"fixed_threshold"`` (fixed cosine cutoff + Top-K cap, Exp 2-3),
                or ``"fixed_topk"`` (direct Top-K without thresholding, Exp 4-1). When ``None``,
                falls back to ``"adaptive"`` if ``use_adaptive_threshold`` is True else ``"fixed_topk"``.
        """
        if threshold_mode is None:
            mode = "adaptive" if use_adaptive_threshold else "fixed_topk"
        else:
            mode = threshold_mode
        if mode not in ("adaptive", "fixed_threshold", "fixed_topk"):
            raise ValueError(
                f"Invalid threshold_mode={mode!r}; expected 'adaptive', 'fixed_threshold', or 'fixed_topk'."
            )

        k_cap = self.cfg.top_k_cap if top_k_cap is None else top_k_cap

        # Step 1: Query encoding & optional SVM + Platt domain routing
        raw_q, ctr_q = self.trainer.encode_query(query)
        routed_domain: Optional[str] = None
        domain_probs: Dict[str, float] = {}
        if self.domain_router is not None:
            routed_domain, domain_probs = self.domain_router.route(query)

        # Step 2: One-hop retrieval from Basic KG (Eqs. 51, 53-54, 57, 59)
        basic_segments: List[EvidenceSegment] = []
        theta_basic = 0.0
        if use_basic_kg and self.index.basic_indices:
            b_idxs = self.index.basic_indices
            if routed_domain is not None:
                routed_b_idxs = [
                    g_idx
                    for g_idx in b_idxs
                    if self.basic_kg.articles[self.index.node_ids[g_idx]].domain == routed_domain
                ]
                if routed_b_idxs:
                    b_idxs = routed_b_idxs

            b_vecs = self.index.raw_node_embeddings[b_idxs]
            b_scores = b_vecs @ raw_q
            theta_basic = self._dynamic_threshold(b_scores, self.cfg.lambda_one_hop, mode)

            if mode == "fixed_topk":
                # Exp 4-1: Fixed Top-K mode bypasses thresholding and selects top-ranked nodes directly
                selected = list(np.argsort(b_scores)[::-1][:k_cap])
            else:
                # Both "adaptive" (mu + lambda*sigma) and "fixed_threshold" (Exp 2-3 fixed cutoff)
                selected = [i for i, sc in enumerate(b_scores) if float(sc) >= theta_basic]

            for local_i in selected:
                g_idx = b_idxs[local_i]
                aid = self.index.node_ids[g_idx]
                art = self.basic_kg.articles[aid]
                basic_segments.append(
                    EvidenceSegment(
                        node_id=aid,
                        source_type="basic",
                        score=float(b_scores[local_i]),
                        text=art.to_segment_text(),
                        article_id=aid,
                        hop=1,
                        metadata={"domain": art.domain, "law_name": art.law_name},
                    )
                )

        # Step 2b: One-hop retrieval from Applied KG (Eqs. 52, 55-56, 58, 60)
        applied_1hop_segments: List[EvidenceSegment] = []
        seed_triple_ids: List[str] = []
        theta_applied = 0.0
        uncertainty_u = 0.0
        a_scores_all = np.zeros(0, dtype=np.float32)

        if use_applied_kg and self.index.applied_indices:
            a_idxs = self.index.applied_indices
            a_vecs = self.index.raw_node_embeddings[a_idxs]
            a_scores_all = a_vecs @ raw_q
            theta_applied = self._dynamic_threshold(a_scores_all, self.cfg.lambda_one_hop, mode)
            # Eq. (75): Query uncertainty from normalized similarity entropy
            uncertainty_u = compute_query_uncertainty(
                a_scores_all,
                top_m=self.cfg.uncertainty_top_m,
                temp=self.cfg.uncertainty_temp,
            )

            if mode == "fixed_topk":
                selected = list(np.argsort(a_scores_all)[::-1][:k_cap])
            else:
                selected = [i for i, sc in enumerate(a_scores_all) if float(sc) >= theta_applied]

            for local_i in selected:
                g_idx = a_idxs[local_i]
                tid = self.index.node_ids[g_idx]
                triple = self.applied_kg.triple_by_id[tid]
                seed_triple_ids.append(tid)
                applied_1hop_segments.append(
                    EvidenceSegment(
                        node_id=tid,
                        source_type="applied_1hop",
                        score=float(a_scores_all[local_i]),
                        text=triple.to_segment_text(),
                        article_id=triple.tail if triple.tail in self.basic_kg.articles else None,
                        hop=1,
                        metadata={"domain": triple.domain},
                    )
                )

        # Step 3: Uncertainty-gated bounded multi-hop expansion (Eqs. 61-67, 75-76)
        applied_mh_segments: List[EvidenceSegment] = []
        theta_mh = 0.0
        selected_hop_depth = 1

        if use_applied_kg and use_multihop and seed_triple_ids and self.cfg.max_hops >= 2:
            # Eq. (76): Trigger 2-hop expansion only when U(Q) > tau_unc (or fixed_topk bypass)
            trigger_mh = (uncertainty_u > self.cfg.uncertainty_threshold) or (mode == "fixed_topk")
            if trigger_mh:
                mh_scores_by_tid: Dict[str, Tuple[float, int]] = {}
                seed_set = set(seed_triple_ids)
                for seed_tid in seed_triple_ids:
                    neighbors = self.applied_kg.k_hop_neighbors(seed_tid, max_hops=self.cfg.max_hops)
                    for nb_tid, dist in neighbors.items():
                        if nb_tid in seed_set or nb_tid not in self.index.node_id_to_idx:
                            continue
                        nb_idx = self.index.node_id_to_idx[nb_tid]
                        sim_val = float(np.dot(self.index.raw_node_embeddings[nb_idx], raw_q))
                        # Eq. (62): Hop-decayed relevance eta^{hop(u,v)} * Sim(q, u)
                        decayed = (self.cfg.hop_decay ** dist) * sim_val
                        prev = mh_scores_by_tid.get(nb_tid)
                        if prev is None or decayed > prev[0]:
                            mh_scores_by_tid[nb_tid] = (decayed, dist)

                if mh_scores_by_tid:
                    # Eqs. (63)-(66): Distribution statistics over V^applied
                    mh_vector = np.zeros(len(self.index.applied_indices), dtype=np.float32)
                    for local_i, g_idx in enumerate(self.index.applied_indices):
                        tid = self.index.node_ids[g_idx]
                        if tid in mh_scores_by_tid:
                            mh_vector[local_i] = mh_scores_by_tid[tid][0]
                    theta_mh = self._dynamic_threshold(mh_vector, self.cfg.lambda_multi_hop, mode)
                    if mode == "fixed_topk":
                        ranked_mh = sorted(
                            mh_scores_by_tid.items(), key=lambda kv: kv[1][0], reverse=True
                        )
                        mh_selected = ranked_mh[:k_cap]
                    else:
                        mh_selected = [
                            (nb_tid, (sc, dist))
                            for nb_tid, (sc, dist) in mh_scores_by_tid.items()
                            if sc >= theta_mh
                        ]
                    for nb_tid, (sc, dist) in mh_selected:
                        triple = self.applied_kg.triple_by_id[nb_tid]
                        applied_mh_segments.append(
                            EvidenceSegment(
                                node_id=nb_tid,
                                source_type="applied_mh",
                                score=float(sc),
                                text=triple.to_segment_text(),
                                article_id=triple.tail if triple.tail in self.basic_kg.articles else None,
                                hop=dist,
                                metadata={"domain": triple.domain},
                            )
                        )
                    if applied_mh_segments:
                        selected_hop_depth = 2

        # Step 4: Contrastive learning-based retrieval over unified V (Eqs. 68-74)
        contrastive_segments: List[EvidenceSegment] = []
        theta_ctr = 0.0
        if use_contrastive and len(self.index.node_ids) > 0:
            ctr_scores = self.index.contrastive_embeddings @ ctr_q
            theta_ctr = self._dynamic_threshold(ctr_scores, self.cfg.lambda_contrastive, mode)
            if mode == "fixed_topk":
                selected = list(np.argsort(ctr_scores)[::-1][:k_cap])
            else:
                selected = [i for i, sc in enumerate(ctr_scores) if float(sc) >= theta_ctr]

            for g_idx in selected:
                nid = self.index.node_ids[g_idx]
                ntype = self.index.node_types[g_idx]
                art_id = nid if ntype == "basic" else None
                contrastive_segments.append(
                    EvidenceSegment(
                        node_id=nid,
                        source_type="contrastive",
                        score=float(ctr_scores[g_idx]),
                        text=f"[Contrastive-Aligned] {self.index.node_texts[g_idx]}",
                        article_id=art_id,
                        hop=1,
                        metadata={"node_type": ntype},
                    )
                )

        # Step 5: Merge, enforce constraints (Eqs. 31-33), and apply Top-K budget cap (Eq. 77)
        merged_by_id: Dict[str, EvidenceSegment] = {}
        for seg in basic_segments + applied_1hop_segments + applied_mh_segments + contrastive_segments:
            prev = merged_by_id.get(seg.node_id)
            if prev is None or seg.score > prev.score:
                merged_by_id[seg.node_id] = seg

        pool = list(merged_by_id.values())
        if use_constraints and mode != "fixed_topk":
            filter_q = ctr_q if use_contrastive else raw_q
            filter_vecs = self._ctr_node_vecs if use_contrastive else self._raw_node_vecs
            pool = self.constraint_filter.filter_evidence_pool(filter_q, pool, filter_vecs)

        # Eq. (77): Top-K operates strictly as an upper context-window ceiling
        pool_sorted = sorted(pool, key=lambda s: s.score, reverse=True)
        final_evidence = pool_sorted[:k_cap]

        return RetrievalTrace(
            query=query,
            routed_domain=routed_domain,
            domain_probs=domain_probs,
            uncertainty_u=uncertainty_u,
            selected_hop_depth=selected_hop_depth,
            theta_basic=theta_basic,
            theta_applied=theta_applied,
            theta_mh=theta_mh,
            theta_ctr=theta_ctr,
            num_basic_hits=len(basic_segments),
            num_applied_1hop_hits=len(applied_1hop_segments),
            num_applied_mh_hits=len(applied_mh_segments),
            num_contrastive_hits=len(contrastive_segments),
            num_final_retained=len(final_evidence),
            top_k_cap=k_cap,
            evidence=final_evidence,
            threshold_mode=mode,
        )
