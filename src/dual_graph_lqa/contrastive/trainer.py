"""Unified trainer for Module 3: Dual-KG GCN + Residual Contrastive Alignment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..config import ContrastiveConfig
from ..data.schemas import QAPair
from ..kg.applied_kg import AppliedKnowledgeGraph
from ..kg.basic_kg import BasicKnowledgeGraph
from .gcn_encoder import GCNEncoder, TextEncoder, build_normalized_adjacency
from .residual_alignment import NumpyResidualContrastiveAligner, ResidualContrastiveOutput


@dataclass
class UnifiedGraphIndex:
    """Stores unified node table V = V^basic U V^applied and learned contrastive vectors."""

    node_ids: List[str]
    node_id_to_idx: Dict[str, int]
    node_types: List[str]  # "basic" | "applied"
    node_texts: List[str]
    raw_node_embeddings: np.ndarray
    frozen_gcn_embeddings: np.ndarray
    contrastive_embeddings: np.ndarray
    basic_indices: List[int]
    applied_indices: List[int]


class ContrastiveTrainer:
    """Builds unified dual-KG index and trains residual contrastive representations."""

    def __init__(
        self,
        cfg: ContrastiveConfig,
        use_pretrained_roberta: bool = False,
        use_residual: bool = True,
        seed: int = 42,
    ) -> None:
        self.cfg = cfg
        self.use_residual = use_residual
        self.seed = seed
        self.text_encoder = TextEncoder(
            model_name=cfg.encoder_model,
            embedding_dim=cfg.embedding_dim,
            use_pretrained_roberta=use_pretrained_roberta,
        )
        self.aligner: NumpyResidualContrastiveAligner | None = None

    def build_unified_nodes_and_edges(
        self,
        basic_kg: BasicKnowledgeGraph,
        applied_kg: AppliedKnowledgeGraph,
    ) -> Tuple[List[str], List[str], List[str], List[Tuple[int, int, float]], List[Tuple[int, int]]]:
        """Collect V = V^basic U V^applied, topology edges E, and cross-graph R_align."""
        node_ids: List[str] = []
        node_types: List[str] = []
        node_texts: List[str] = []
        id_to_idx: Dict[str, int] = {}

        # Basic KG article nodes
        for aid, art in basic_kg.articles.items():
            id_to_idx[aid] = len(node_ids)
            node_ids.append(aid)
            node_types.append("basic")
            node_texts.append(art.to_segment_text())

        # Applied KG triple nodes
        for triple in applied_kg.triples:
            tid = triple.node_id
            id_to_idx[tid] = len(node_ids)
            node_ids.append(tid)
            node_types.append("applied")
            node_texts.append(triple.to_segment_text())

        edges: List[Tuple[int, int, float]] = []
        # Co-law weak smoothness edges inside Basic KG (preserve law-article grouping)
        law_to_arts: Dict[str, List[int]] = {}
        for aid, art in basic_kg.articles.items():
            law_to_arts.setdefault(art.law_id, []).append(id_to_idx[aid])
        for arts in law_to_arts.values():
            for i in range(len(arts)):
                for j in range(i + 1, len(arts)):
                    edges.append((arts[i], arts[j], 0.5))

        # Applied KG entity-sharing edges
        for u_tid, v_tid in applied_kg.triple_graph.edges():
            if u_tid in id_to_idx and v_tid in id_to_idx:
                edges.append((id_to_idx[u_tid], id_to_idx[v_tid], 1.0))

        # Cross-graph alignment pairs R_align (Eq. 47): Applied triples referencing Basic articles
        align_pairs: List[Tuple[int, int]] = []
        for triple in applied_kg.triples:
            t_idx = id_to_idx[triple.node_id]
            for cand in (triple.head, triple.tail):
                if cand in id_to_idx and node_types[id_to_idx[cand]] == "basic":
                    b_idx = id_to_idx[cand]
                    align_pairs.append((b_idx, t_idx))
                    edges.append((b_idx, t_idx, 1.0))

        return node_ids, node_types, node_texts, edges, align_pairs

    def fit(
        self,
        basic_kg: BasicKnowledgeGraph,
        applied_kg: AppliedKnowledgeGraph,
        train_qa_pairs: Sequence[QAPair],
    ) -> Tuple[UnifiedGraphIndex, List[ResidualContrastiveOutput]]:
        """Run GCN pre-encoding + residual multi-positive InfoNCE training (Algorithm 3)."""
        node_ids, node_types, node_texts, edges, align_pairs = self.build_unified_nodes_and_edges(
            basic_kg, applied_kg
        )
        id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

        # Step 1: RoBERTa node encoding + frozen GCN base embeddings \tilde{z}_v
        raw_node_embs = self.text_encoder.encode(node_texts, normalize=True)
        norm_adj = build_normalized_adjacency(len(node_ids), edges)
        gcn = GCNEncoder(
            in_dim=raw_node_embs.shape[1],
            hidden_dim=raw_node_embs.shape[1],
            num_layers=self.cfg.gcn_num_layers,
            seed=self.seed,
        )
        frozen_gcn = gcn.encode_graph(raw_node_embs, norm_adj)

        # Step 2: Build positive node sets V_i^+ = V_{a_i}^{basic} U V_{a_i}^{applied} (Eq. 37)
        qa_questions = [qa.question for qa in train_qa_pairs]
        raw_q_vecs = self.text_encoder.encode(qa_questions, normalize=True)
        positive_sets: List[List[int]] = []
        for qa in train_qa_pairs:
            pos: List[int] = []
            for aid in qa.positive_articles:
                if aid in id_to_idx:
                    pos.append(id_to_idx[aid])
            for triple in applied_kg.triples:
                if triple.source_qa_id == qa.qa_id and triple.node_id in id_to_idx:
                    pos.append(id_to_idx[triple.node_id])
            positive_sets.append(sorted(set(pos)))

        # Step 3: Residual contrastive optimization (Eqs. 34-49)
        self.aligner = NumpyResidualContrastiveAligner(
            frozen_base_embeddings=frozen_gcn,
            temperature=self.cfg.temperature,
            lambda_res=self.cfg.lambda_res,
            lambda_smooth=self.cfg.lambda_smooth,
            lambda_align=self.cfg.lambda_align,
            lambda_head=self.cfg.lambda_head,
            use_residual=self.use_residual,
        )
        history: List[ResidualContrastiveOutput] = []
        for _ in range(self.cfg.epochs):
            out = self.aligner.step(
                raw_query_vecs=raw_q_vecs,
                positive_sets=positive_sets,
                graph_edges=edges,
                align_pairs=align_pairs,
                lr=self.cfg.learning_rate,
            )
            history.append(out)

        contrastive_embs = self.aligner.get_normalized_node_embeddings()
        index = UnifiedGraphIndex(
            node_ids=node_ids,
            node_id_to_idx=id_to_idx,
            node_types=node_types,
            node_texts=node_texts,
            raw_node_embeddings=raw_node_embs,
            frozen_gcn_embeddings=frozen_gcn,
            contrastive_embeddings=contrastive_embs,
            basic_indices=[i for i, t in enumerate(node_types) if t == "basic"],
            applied_indices=[i for i, t in enumerate(node_types) if t == "applied"],
        )
        return index, history

    def encode_query(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (raw_query_vec, contrastively_aligned_query_vec)."""
        raw = self.text_encoder.encode([query], normalize=True)
        if self.aligner is None:
            return raw[0], raw[0]
        _, _, h_hat = self.aligner.encode_queries(raw)
        return raw[0], h_hat[0]

    def save_checkpoint(self, path: str | Path) -> None:
        if self.aligner is None:
            raise RuntimeError("Call fit() before saving a contrastive checkpoint.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            base_z=self.aligner.base_z,
            r=self.aligner.r,
            W=self.aligner.W,
            Q_proj=self.aligner.Q_proj,
        )
