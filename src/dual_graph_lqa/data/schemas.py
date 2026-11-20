"""Core data structures matching Section 3 notations (D^law, D^case, D^QA)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ArticleNode:
    """Statutory article node v_{i,j} = (phi_{i,j}, Case_{i,j}) in Basic KG."""

    article_id: str
    law_id: str
    law_name: str
    article_number: str
    domain: str
    text: str
    jurisdiction: str = "Taiwan (R.O.C.)"
    effective_date: str = "2024-06-01"
    source: str = ""
    cited_case_ids: List[str] = field(default_factory=list)
    cited_case_summaries: List[str] = field(default_factory=list)

    def to_segment_text(self) -> str:
        """Format Seg^basic(v) = [Article Citation, phi_{x,y}, Case_{x,y}] (Sec. 4.4)."""
        citation = f"{self.law_name} {self.article_number} ({self.article_id})"
        case_str = (
            "; ".join(self.cited_case_summaries)
            if self.cited_case_summaries
            else "No linked judicial decisions."
        )
        return f"[Basic-KG | {citation}] {self.text} | Related Cases: {case_str}"


@dataclass
class LawRegulation:
    """Regulation L_i = {phi_{i,1}, ..., phi_{i,|L_i|}} in D^law."""

    law_id: str
    law_name: str
    domain: str
    jurisdiction: str
    effective_date: str
    source: str
    articles: List[ArticleNode] = field(default_factory=list)


@dataclass
class JudicialDecision:
    """Judicial decision j_i^case in D^case."""

    case_id: str
    title: str
    domain: str
    date: str
    cited_articles: List[str]
    summary: str


@dataclass
class SemanticTriple:
    """Semantic triple (e_x, r_{x,y}, e_y) in Applied KG (Sec. 4.2)."""

    head: str
    relation: str
    tail: str
    source_qa_id: str = ""
    domain: str = ""

    @property
    def node_id(self) -> str:
        return f"AKG::{self.head}::{self.relation}::{self.tail}"

    def to_segment_text(self) -> str:
        """Format Seg^applied(h, r, t) (Sec. 4.4)."""
        return f"[Applied-KG Triple] ({self.head}) --[{self.relation}]--> ({self.tail})"


@dataclass
class QAPair:
    """Historical corporate QA pair (q_i, a_i) in D^QA."""

    qa_id: str
    domain: str
    question: str
    answer: str
    positive_articles: List[str] = field(default_factory=list)
    triples: List[SemanticTriple] = field(default_factory=list)
    source: str = "corporate_qa"


@dataclass
class EvidenceSegment:
    """Unified retrieved evidence node passed to constraint filtering and LLM prompting."""

    node_id: str
    source_type: str  # "basic" | "applied_1hop" | "applied_mh" | "contrastive"
    score: float
    text: str
    article_id: Optional[str] = None
    hop: int = 1
    metadata: Dict[str, str] = field(default_factory=dict)
