"""Retrieval modules: uncertainty-gated adaptive dual-KG retriever, constraints, and flat RAG."""

from .adaptive_retriever import AdaptiveDualKGRetriever, RetrievalTrace
from .constraints import EvidenceConstraintFilter, shannon_entropy
from .flat_rag_baseline import FlatTextHybridRAG

__all__ = [
    "EvidenceConstraintFilter",
    "shannon_entropy",
    "AdaptiveDualKGRetriever",
    "RetrievalTrace",
    "FlatTextHybridRAG",
]
