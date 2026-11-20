"""Module 1: Basic Knowledge Graph Construction (Section 4.1, Algorithm 1 / Table 2).

Organizes statutory regulations D^law = {L_1, ..., L_q} and judicial decisions D^case
into a unified star-structured graph G^basic = (V^basic, E^basic) where:
  - Each law L_i is a central hub node.
  - Each article node v_{i,j} = (phi_{i,j}, Case_{i,j}) stores full article text
    (preserving internal paragraphs/subparagraphs) together with citing decisions Case_{i,j}.
  - Law-article membership edges r_{i,j} = (L_i, phi_{i,j}) connect laws to articles
    without direct cross-article edges.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import networkx as nx

from ..data.schemas import ArticleNode, JudicialDecision, LawRegulation


def clean_statutory_text(text: str) -> str:
    """Remove redundant symbols, erratic punctuation spacing, and duplicate whitespace."""
    text = re.sub(r"[\r\t]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def parse_raw_law_text(
    law_id: str,
    law_name: str,
    raw_text: str,
    domain: str,
    jurisdiction: str = "Taiwan (R.O.C.)",
    effective_date: str = "2024-06-01",
    source: str = "",
) -> LawRegulation:
    """Segment plain-text statutory body into articles using regex (Section 4.1)."""
    pattern = re.compile(r"(第\s*\d+(?:-\d+)?\s*條|Article\s+\d+(?:-\d+)?)", re.IGNORECASE)
    splits = pattern.split(clean_statutory_text(raw_text))
    articles: List[ArticleNode] = []

    if len(splits) <= 1:
        articles.append(
            ArticleNode(
                article_id=f"{law_id}_Art1",
                law_id=law_id,
                law_name=law_name,
                article_number="Article 1",
                domain=domain,
                text=clean_statutory_text(raw_text),
                jurisdiction=jurisdiction,
                effective_date=effective_date,
                source=source,
            )
        )
    else:
        for idx in range(1, len(splits), 2):
            art_header = splits[idx].strip()
            art_body = splits[idx + 1].strip() if idx + 1 < len(splits) else ""
            art_num_digits = re.sub(r"[^\d-]", "", art_header) or str(len(articles) + 1)
            articles.append(
                ArticleNode(
                    article_id=f"{law_id}_Art{art_num_digits}",
                    law_id=law_id,
                    law_name=law_name,
                    article_number=art_header,
                    domain=domain,
                    text=clean_statutory_text(art_body),
                    jurisdiction=jurisdiction,
                    effective_date=effective_date,
                    source=source,
                )
            )

    return LawRegulation(
        law_id=law_id,
        law_name=law_name,
        domain=domain,
        jurisdiction=jurisdiction,
        effective_date=effective_date,
        source=source,
        articles=articles,
    )


class BasicKnowledgeGraph:
    """Unified Basic KG G^basic = (V^basic, E^basic)."""

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self.laws: Dict[str, LawRegulation] = {}
        self.articles: Dict[str, ArticleNode] = {}
        self.cases: Dict[str, JudicialDecision] = {}

    @property
    def num_laws(self) -> int:
        return len(self.laws)

    @property
    def num_articles(self) -> int:
        return len(self.articles)

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()

    def get_articles(self, domain: Optional[str] = None) -> List[ArticleNode]:
        if domain is None:
            return list(self.articles.values())
        filtered = [a for a in self.articles.values() if a.domain == domain]
        return filtered if filtered else list(self.articles.values())

    def to_dict(self) -> Dict[str, object]:
        return {
            "num_laws": self.num_laws,
            "num_articles": self.num_articles,
            "num_cases": len(self.cases),
            "num_edges": self.num_edges,
            "articles": {
                aid: {
                    "law_id": a.law_id,
                    "law_name": a.law_name,
                    "article_number": a.article_number,
                    "domain": a.domain,
                    "text": a.text,
                    "cited_case_ids": a.cited_case_ids,
                }
                for aid, a in self.articles.items()
            },
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


class BasicKGBuilder:
    """Implements Algorithm 1 (Table 2): Construction of the Basic KG."""

    @staticmethod
    def load_regulations(statutes_path: str | Path) -> List[LawRegulation]:
        raw = json.loads(Path(statutes_path).read_text(encoding="utf-8"))
        laws: List[LawRegulation] = []
        for item in raw:
            law = LawRegulation(
                law_id=item["law_id"],
                law_name=item["law_name"],
                domain=item["domain"],
                jurisdiction=item.get("jurisdiction", "Taiwan (R.O.C.)"),
                effective_date=item.get("effective_date", "2024-06-01"),
                source=item.get("source", ""),
                articles=[],
            )
            for art in item.get("articles", []):
                law.articles.append(
                    ArticleNode(
                        article_id=art["article_id"],
                        law_id=law.law_id,
                        law_name=law.law_name,
                        article_number=art["article_number"],
                        domain=law.domain,
                        text=clean_statutory_text(art["text"]),
                        jurisdiction=law.jurisdiction,
                        effective_date=law.effective_date,
                        source=law.source,
                    )
                )
            laws.append(law)
        return laws

    @staticmethod
    def load_cases(cases_path: str | Path) -> List[JudicialDecision]:
        raw = json.loads(Path(cases_path).read_text(encoding="utf-8"))
        return [
            JudicialDecision(
                case_id=c["case_id"],
                title=c["title"],
                domain=c["domain"],
                date=c.get("date", ""),
                cited_articles=list(c.get("cited_articles", [])),
                summary=clean_statutory_text(c["summary"]),
            )
            for c in raw
        ]

    def build(
        self,
        regulations: Sequence[LawRegulation],
        cases: Sequence[JudicialDecision],
    ) -> BasicKnowledgeGraph:
        """Construct G^basic = union_i G_i^basic with Case_{i,j} attached to each v_{i,j}."""
        kg = BasicKnowledgeGraph()

        # Index citing judicial decisions by article_id -> Case_{i,j}
        citations_by_article: Dict[str, List[JudicialDecision]] = {}
        for case in cases:
            kg.cases[case.case_id] = case
            for aid in case.cited_articles:
                citations_by_article.setdefault(aid, []).append(case)

        for law in regulations:
            kg.laws[law.law_id] = law
            kg.graph.add_node(
                law.law_id,
                node_type="law",
                law_name=law.law_name,
                domain=law.domain,
            )
            for art in law.articles:
                linked_cases = citations_by_article.get(art.article_id, [])
                art.cited_case_ids = [c.case_id for c in linked_cases]
                art.cited_case_summaries = [f"{c.title}: {c.summary}" for c in linked_cases]

                kg.articles[art.article_id] = art
                kg.graph.add_node(
                    art.article_id,
                    node_type="article",
                    law_id=law.law_id,
                    domain=art.domain,
                    text=art.text,
                    case_ids=art.cited_case_ids,
                )
                # Law-article membership edge r_{i,j} = (L_i, phi_{i,j})
                kg.graph.add_edge(law.law_id, art.article_id, relation="contains_article")

        return kg
