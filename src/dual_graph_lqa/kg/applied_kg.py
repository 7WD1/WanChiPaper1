r"""Module 2: Applied Knowledge Graph Construction (Section 4.2, Algorithm 2 / Table 3).

Constructs G^applied = (V^applied, E^applied) from historical training QA pairs D^QA:
  Step 1: Jieba tokenization + stopword removal -> T_i.
  Step 2: BM25 keyword extraction -> \hat{W}_i; synonym expansion S(\cdot) -> W_i.
  Step 3: Answer Revision enforcing explicit statutory-basis prefix ("According to Article...").
  Step 4: LLM-assisted (or pre-extracted/heuristic) semantic triple generation
          Phi_i^text = {(e_{i,x}, r_{i,x,y}, e_{i,y})} aggregated into G^applied.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx

from ..data.schemas import QAPair, SemanticTriple

try:
    import jieba  # type: ignore

    _HAS_JIEBA = True
except ImportError:  # pragma: no cover
    _HAS_JIEBA = False

DEFAULT_STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "in", "on", "for",
    "and", "or", "with", "that", "this", "it", "if", "can", "may", "must", "shall",
    "what", "when", "how", "does", "do", "by", "as", "at", "from", "our", "my", "i",
    "的", "了", "是", "在", "和", "與", "或", "是否", "如何", "請問", "如果", "可以",
}


def segment_and_clean(text: str, use_jieba: bool = True, stopwords: Set[str] = DEFAULT_STOPWORDS) -> List[str]:
    """Step 1: Tokenize text with Jieba (if available) and strip stopwords."""
    if use_jieba and _HAS_JIEBA:
        raw_tokens = list(jieba.cut(text))
    else:
        raw_tokens = re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]{1,2}", text)
    cleaned = [tok.strip().lower() for tok in raw_tokens if tok.strip()]
    return [tok for tok in cleaned if tok not in stopwords and len(tok) > 1 or re.match(r"[\u4e00-\u9fff]", tok)]


class BM25KeywordExtractor:
    """Pure-Python BM25 keyword scorer matching Robertson & Zaragoza [3] (Section 4.2)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_freqs: Counter[str] = Counter()
        self.num_docs: int = 0
        self.avg_dl: float = 1.0

    def fit(self, tokenized_docs: Sequence[Sequence[str]]) -> "BM25KeywordExtractor":
        self.num_docs = max(len(tokenized_docs), 1)
        self.avg_dl = sum(len(d) for d in tokenized_docs) / self.num_docs
        self.doc_freqs.clear()
        for doc in tokenized_docs:
            self.doc_freqs.update(set(doc))
        return self

    def extract_top_k(self, tokens: Sequence[str], top_k: int = 8) -> List[str]:
        if not tokens:
            return []
        tf = Counter(tokens)
        dl = len(tokens)
        scores: Dict[str, float] = {}
        for tok, freq in tf.items():
            df = self.doc_freqs.get(tok, 1)
            idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)
            denom = freq + self.k1 * (1.0 - self.b + self.b * dl / max(self.avg_dl, 1.0))
            scores[tok] = idf * (freq * (self.k1 + 1.0)) / max(denom, 1e-12)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [tok for tok, _ in ranked[:top_k]]


class AppliedKnowledgeGraph:
    """Unified Applied KG G^applied = (V^applied, E^applied) supporting 1- and 2-hop traversal."""

    def __init__(self) -> None:
        self.graph: nx.Graph = nx.Graph()
        self.triples: List[SemanticTriple] = []
        self.triple_by_id: Dict[str, SemanticTriple] = {}
        self.entity_to_triples: Dict[str, List[str]] = {}
        # Triple adjacency graph for multi-hop expansion N_k(v) in Eqs. (61)-(67)
        self.triple_graph: nx.Graph = nx.Graph()

    @property
    def num_entities(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_triples(self) -> int:
        return len(self.triples)

    def add_triple(self, triple: SemanticTriple) -> None:
        tid = triple.node_id
        if tid in self.triple_by_id:
            return
        self.triples.append(triple)
        self.triple_by_id[tid] = triple

        self.graph.add_node(triple.head, domain=triple.domain)
        self.graph.add_node(triple.tail, domain=triple.domain)
        self.graph.add_edge(triple.head, triple.tail, relation=triple.relation, triple_id=tid)

        # Connect triples sharing an entity in triple_graph for k-hop neighbor lookup N_k(v)
        self.triple_graph.add_node(tid, text=triple.to_segment_text(), domain=triple.domain)
        for ent in (triple.head, triple.tail):
            existing = self.entity_to_triples.setdefault(ent, [])
            for other_tid in existing:
                if other_tid != tid:
                    self.triple_graph.add_edge(tid, other_tid, shared_entity=ent)
            existing.append(tid)

    def k_hop_neighbors(self, seed_triple_id: str, max_hops: int = 2) -> Dict[str, int]:
        """Return mapping {neighbor_triple_id: hop_distance} for 1 <= hop <= max_hops."""
        if seed_triple_id not in self.triple_graph:
            return {}
        lengths = nx.single_source_shortest_path_length(self.triple_graph, seed_triple_id, cutoff=max_hops)
        return {nid: dist for nid, dist in lengths.items() if 1 <= dist <= max_hops}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "num_entities": self.num_entities,
            "num_triples": self.num_triples,
            "triples": [
                {
                    "node_id": t.node_id,
                    "head": t.head,
                    "relation": t.relation,
                    "tail": t.tail,
                    "source_qa_id": t.source_qa_id,
                    "domain": t.domain,
                }
                for t in self.triples
            ],
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class AppliedKGBuilder:
    """Implements Algorithm 2 (Table 3): Construction of the Applied KG."""

    def __init__(
        self,
        synonyms: Optional[Dict[str, List[str]]] = None,
        bm25_top_keywords: int = 8,
        use_jieba: bool = True,
        enforce_statutory_prefix: bool = True,
        llm_triple_extractor: Optional[Callable[[str, str, List[str]], List[Tuple[str, str, str]]]] = None,
    ) -> None:
        self.synonyms = {k.lower(): v for k, v in (synonyms or {}).items()}
        self.bm25_top_keywords = bm25_top_keywords
        self.use_jieba = use_jieba
        self.enforce_statutory_prefix = enforce_statutory_prefix
        self.llm_triple_extractor = llm_triple_extractor

    @staticmethod
    def load_qa_pairs(qa_path: str | Path) -> List[QAPair]:
        raw = json.loads(Path(qa_path).read_text(encoding="utf-8"))
        pairs: List[QAPair] = []
        for item in raw:
            triples = [
                SemanticTriple(
                    head=t[0], relation=t[1], tail=t[2], source_qa_id=item["qa_id"], domain=item["domain"]
                )
                for t in item.get("triples", [])
            ]
            pairs.append(
                QAPair(
                    qa_id=item["qa_id"],
                    domain=item["domain"],
                    question=item["question"],
                    answer=item["answer"],
                    positive_articles=list(item.get("positive_articles", [])),
                    triples=triples,
                    source=item.get("source", "corporate_qa"),
                )
            )
        return pairs

    @staticmethod
    def load_synonyms(synonyms_path: str | Path) -> Dict[str, List[str]]:
        return json.loads(Path(synonyms_path).read_text(encoding="utf-8"))

    def expand_synonyms(self, keywords: Sequence[str], raw_text: str) -> List[str]:
        r"""Compute W_i = union_j S(\hat{w}_{i,j}) (Section 4.2)."""
        expanded: List[str] = list(keywords)
        lower_text = raw_text.lower()
        for kw in keywords:
            for syn in self.synonyms.get(kw.lower(), []):
                if syn not in expanded:
                    expanded.append(syn)
        # Also catch multi-word colloquial phrases present in raw text
        for phrase, syns in self.synonyms.items():
            if phrase in lower_text:
                for syn in [phrase, *syns]:
                    if syn not in expanded:
                        expanded.append(syn)
        return expanded

    def revise_answer_with_statutory_prefix(self, answer: str, positive_articles: Sequence[str]) -> str:
        """Step 3: Revise A_i to start explicitly with a statutory citation if absent."""
        if not self.enforce_statutory_prefix:
            return answer
        if re.match(r"^(according to|pursuant to|under|依|根據)", answer.strip(), re.IGNORECASE):
            return answer.strip()
        if positive_articles:
            cite_str = ", ".join(positive_articles)
            return f"According to {cite_str}, {answer.strip()}"
        return answer.strip()

    def build(self, qa_pairs: Sequence[QAPair]) -> AppliedKnowledgeGraph:
        """Run Steps 1-4 of Algorithm 2 over strictly training-partition QA pairs."""
        tokenized_docs = [
            segment_and_clean(f"{qa.question} {qa.answer}", use_jieba=self.use_jieba)
            for qa in qa_pairs
        ]
        bm25 = BM25KeywordExtractor().fit(tokenized_docs)
        akg = AppliedKnowledgeGraph()

        for qa, tokens in zip(qa_pairs, tokenized_docs):
            # Step 2: BM25 keywords + synonym expansion
            keywords = bm25.extract_top_k(tokens, top_k=self.bm25_top_keywords)
            expanded_vocab = self.expand_synonyms(keywords, f"{qa.question} {qa.answer}")

            # Step 3: Answer revision
            revised_answer = self.revise_answer_with_statutory_prefix(qa.answer, qa.positive_articles)

            # Step 4: Triple extraction (LLM callback -> pre-extracted -> keyword/synonym fallback)
            if self.llm_triple_extractor is not None:
                raw_triples = self.llm_triple_extractor(qa.question, revised_answer, expanded_vocab)
                extracted = [
                    SemanticTriple(h, r, t, source_qa_id=qa.qa_id, domain=qa.domain)
                    for h, r, t in raw_triples
                ]
            elif qa.triples:
                extracted = list(qa.triples)
            else:
                extracted = []

            # Enrich with synonym-bridge triples linking colloquial expressions to legal terms
            lower_q = qa.question.lower()
            for phrase, syns in self.synonyms.items():
                if phrase in lower_q and syns:
                    extracted.append(
                        SemanticTriple(
                            head=phrase,
                            relation="synonym_of",
                            tail=syns[0],
                            source_qa_id=qa.qa_id,
                            domain=qa.domain,
                        )
                    )
                    for art_id in qa.positive_articles:
                        extracted.append(
                            SemanticTriple(
                                head=syns[0],
                                relation="governed_by",
                                tail=art_id,
                                source_qa_id=qa.qa_id,
                                domain=qa.domain,
                            )
                        )

            for triple in extracted:
                akg.add_triple(triple)

        return akg
