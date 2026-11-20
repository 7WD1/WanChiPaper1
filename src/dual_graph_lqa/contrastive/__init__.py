"""Module 3: GCN + RoBERTa Residual Contrastive Alignment (Section 4.3, Eqs. 34-49)."""

from .gcn_encoder import GCNEncoder, TextEncoder, build_normalized_adjacency
from .residual_alignment import (
    NumpyResidualContrastiveAligner,
    ResidualContrastiveOutput,
)
from .trainer import ContrastiveTrainer

try:
    from .residual_alignment import TorchResidualContrastiveAligner
except ImportError:  # pragma: no cover
    TorchResidualContrastiveAligner = None  # type: ignore

__all__ = [
    "TextEncoder",
    "GCNEncoder",
    "build_normalized_adjacency",
    "ResidualContrastiveOutput",
    "NumpyResidualContrastiveAligner",
    "TorchResidualContrastiveAligner",
    "ContrastiveTrainer",
]
