"""Four-stage anti-leakage and deduplication protocol (Section 5.1, S1-S4)."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, List, Sequence, Set, Tuple

import numpy as np

from .schemas import QAPair


def normalize_legal_text(text: str) -> str:
    """Unicode NFKC normalization, lowercasing, and punctuation stripping."""
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
    return text


def char_ngrams(text: str, n: int = 3) -> Set[str]:
    norm = normalize_legal_text(text)
    if len(norm) < n:
        return {norm} if norm else set()
    return {norm[i : i + n] for i in range(len(norm) - n + 1)}


def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class AntiLeakageReport:
    total_input: int
    retained: int
    removed_s1_source: int = 0
    removed_s2_lexical: int = 0
    removed_s3_paraphrase: int = 0
    removed_qa_ids: List[str] = field(default_factory=list)


class AntiLeakageFilter:
    """Enforces Safeguards S1-S4 from Section 5.1 between D^QA and held-out test sets.

    S1: Physical source isolation (construction source must differ from eval source).
    S2: Exact normalized match + character 3-gram Jaccard >= 0.80 removal.
    S3: Dense embedding cosine similarity >= 0.85 semantic paraphrase removal.
    S4: Strict answer-derived triple quarantine (triples built solely from retained train pairs).
    """

    def __init__(
        self,
        ngram_size: int = 3,
        jaccard_threshold: float = 0.80,
        cosine_threshold: float = 0.85,
        forbidden_sources: Sequence[str] = ("lawbench", "moj_faq_eval"),
    ) -> None:
        self.ngram_size = ngram_size
        self.jaccard_threshold = jaccard_threshold
        self.cosine_threshold = cosine_threshold
        self.forbidden_sources = set(forbidden_sources)

    def filter_training_pairs(
        self,
        train_pairs: Sequence[QAPair],
        eval_questions: Sequence[str],
        embed_fn: Callable[[Sequence[str]], np.ndarray] | None = None,
    ) -> Tuple[List[QAPair], AntiLeakageReport]:
        """Remove any overlapping or near-paraphrase construction QA pairs."""
        eval_norms = {normalize_legal_text(q) for q in eval_questions}
        eval_grams = [char_ngrams(q, self.ngram_size) for q in eval_questions]

        stage2_kept: List[QAPair] = []
        report = AntiLeakageReport(total_input=len(train_pairs), retained=0)

        for qa in train_pairs:
            # S1: Physical source isolation
            if qa.source in self.forbidden_sources:
                report.removed_s1_source += 1
                report.removed_qa_ids.append(qa.qa_id)
                continue

            # S2: Exact & character n-gram Jaccard filtering
            q_norm = normalize_legal_text(qa.question)
            q_grams = char_ngrams(qa.question, self.ngram_size)
            is_lex_dup = q_norm in eval_norms or any(
                jaccard_similarity(q_grams, eg) >= self.jaccard_threshold for eg in eval_grams
            )
            if is_lex_dup:
                report.removed_s2_lexical += 1
                report.removed_qa_ids.append(qa.qa_id)
                continue

            stage2_kept.append(qa)

        # S3: Dense semantic paraphrase filtering
        if embed_fn is None or not stage2_kept or not eval_questions:
            report.retained = len(stage2_kept)
            return stage2_kept, report

        train_vecs = embed_fn([qa.question for qa in stage2_kept])
        eval_vecs = embed_fn(list(eval_questions))

        train_vecs = train_vecs / np.maximum(np.linalg.norm(train_vecs, axis=1, keepdims=True), 1e-12)
        eval_vecs = eval_vecs / np.maximum(np.linalg.norm(eval_vecs, axis=1, keepdims=True), 1e-12)
        sim_matrix = train_vecs @ eval_vecs.T

        final_kept: List[QAPair] = []
        for idx, qa in enumerate(stage2_kept):
            if float(np.max(sim_matrix[idx])) >= self.cosine_threshold:
                report.removed_s3_paraphrase += 1
                report.removed_qa_ids.append(qa.qa_id)
            else:
                final_kept.append(qa)

        report.retained = len(final_kept)
        return final_kept, report
