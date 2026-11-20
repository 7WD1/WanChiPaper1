"""Strong Flat-Text Hybrid RAG Baseline (Dense + BM25 + Rerank).

Implements the matched-corpus flat-chunk baseline used in Exp 1-3 (LLAMA3-TAIDE-7B)
and Exp 3-2 (GPT-4 + Flat-Text RAG with matched access) in Tables 11 and 15.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from ..contrastive.gcn_encoder import TextEncoder
from ..data.schemas import ArticleNode, EvidenceSegment, JudicialDecision
from ..kg.applied_kg import segment_and_clean


@dataclass
class FlatChunk:
    chunk_id: str
    text: str
    article_id: str | None = None
    tokens: List[str] | None = None


class FlatTextHybridRAG:
    """Flat-chunk hybrid retriever combining BM25, dense cosine similarity, and RRF reranking."""

    def __init__(
        self,
        text_encoder: TextEncoder,
        chunk_size_chars: int = 320,
        chunk_overlap: int = 64,
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ) -> None:
        self.text_encoder = text_encoder
        self.chunk_size_chars = chunk_size_chars
        self.chunk_overlap = chunk_overlap
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

        self.chunks: List[FlatChunk] = []
        self.chunk_vecs: np.ndarray | None = None
        self._doc_freqs: Counter[str] = Counter()
        self._avg_dl: float = 1.0

    def index_corpus(
        self,
        articles: Sequence[ArticleNode],
        cases: Sequence[JudicialDecision] = (),
    ) -> None:
        """Split statutes and cases into flat overlapping text chunks (conventional RAG)."""
        self.chunks.clear()
        for art in articles:
            full_text = f"{art.law_name} {art.article_number}: {art.text}"
            step = max(self.chunk_size_chars - self.chunk_overlap, 1)
            for idx, start in enumerate(range(0, max(len(full_text), 1), step)):
                piece = full_text[start : start + self.chunk_size_chars].strip()
                if piece:
                    self.chunks.append(
                        FlatChunk(
                            chunk_id=f"flat_{art.article_id}_c{idx}",
                            text=piece,
                            article_id=art.article_id,
                            tokens=segment_and_clean(piece, use_jieba=False),
                        )
                    )
        for case in cases:
            piece = f"{case.title}: {case.summary}"
            self.chunks.append(
                FlatChunk(
                    chunk_id=f"flat_{case.case_id}",
                    text=piece,
                    article_id=case.cited_articles[0] if case.cited_articles else None,
                    tokens=segment_and_clean(piece, use_jieba=False),
                )
            )

        self._doc_freqs.clear()
        for ch in self.chunks:
            self._doc_freqs.update(set(ch.tokens or []))
        self._avg_dl = sum(len(ch.tokens or []) for ch in self.chunks) / max(len(self.chunks), 1)
        self.chunk_vecs = self.text_encoder.encode([ch.text for ch in self.chunks], normalize=True)

    def _bm25_scores(self, query_tokens: Sequence[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        n_docs = max(len(self.chunks), 1)
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        for i, ch in enumerate(self.chunks):
            toks = ch.tokens or []
            tf = Counter(toks)
            dl = len(toks)
            s = 0.0
            for qt in query_tokens:
                if qt not in tf:
                    continue
                df = self._doc_freqs.get(qt, 1)
                idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
                f = tf[qt]
                s += idf * (f * (k1 + 1.0)) / (f + k1 * (1.0 - b + b * dl / max(self._avg_dl, 1.0)))
            scores[i] = s
        max_s = float(np.max(scores)) if scores.size > 0 else 0.0
        return scores / max(max_s, 1e-12)

    def retrieve(self, query: str, top_k: int = 20) -> List[EvidenceSegment]:
        if not self.chunks or self.chunk_vecs is None:
            return []
        q_vec = self.text_encoder.encode([query], normalize=True)[0]
        dense = self.chunk_vecs @ q_vec
        bm25 = self._bm25_scores(segment_and_clean(query, use_jieba=False))

        # Hybrid score + reciprocal-rank fusion reranking
        dense_rank = np.argsort(np.argsort(-dense)) + 1
        bm25_rank = np.argsort(np.argsort(-bm25)) + 1
        rrf = (1.0 / (60.0 + dense_rank)) + (1.0 / (60.0 + bm25_rank))
        hybrid = self.dense_weight * dense + self.bm25_weight * bm25 + 10.0 * rrf

        top_indices = np.argsort(hybrid)[::-1][:top_k]
        return [
            EvidenceSegment(
                node_id=self.chunks[idx].chunk_id,
                source_type="flat_rag",
                score=float(hybrid[idx]),
                text=f"[Flat-RAG Chunk] {self.chunks[idx].text}",
                article_id=self.chunks[idx].article_id,
                hop=1,
            )
            for idx in top_indices
        ]
