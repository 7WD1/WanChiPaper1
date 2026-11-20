r"""RoBERTa text encoder and multi-layer GCN producing frozen base vectors \tilde{z}_v (Section 4.3).

When PyTorch and HuggingFace Transformers are installed and `use_pretrained_roberta=True`,
this module encodes texts with RoBERTa (`hfl/chinese-roberta-wwm-ext`) and propagates
features through a 2-layer Kipf & Welling [17] GCN.
In lightweight CPU/NumPy environments (e.g., CI smoke tests), it uses a deterministic
character/word n-gram random-projection encoder paired with exact normalized-Laplacian
GCN propagation \hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}.
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False

try:
    from transformers import AutoModel, AutoTokenizer

    _HAS_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    _HAS_TRANSFORMERS = False


def build_normalized_adjacency(
    num_nodes: int, edges: Sequence[Tuple[int, int, float]]
) -> np.ndarray:
    r"""Compute symmetric normalized adjacency \hat{A} = D^{-1/2} (A + I) D^{-1/2} [17]."""
    adj = np.eye(num_nodes, dtype=np.float32)
    for u, v, w in edges:
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            adj[u, v] = max(adj[u, v], float(w))
            adj[v, u] = max(adj[v, u], float(w))
    deg = np.sum(adj, axis=1)
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    return (deg_inv_sqrt[:, None] * adj) * deg_inv_sqrt[None, :]


class TextEncoder:
    """Dual-mode RoBERTa query/node encoder (Eq. 36) with portable NumPy fallback."""

    def __init__(
        self,
        model_name: str = "hfl/chinese-roberta-wwm-ext",
        embedding_dim: int = 256,
        use_pretrained_roberta: bool = False,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.device = device
        self.use_pretrained_roberta = bool(use_pretrained_roberta and _HAS_TORCH and _HAS_TRANSFORMERS)

        self.tokenizer = None
        self.model = None
        if self.use_pretrained_roberta:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device)
            self.embedding_dim = int(self.model.config.hidden_size)

    def _hash_encode_one(self, text: str) -> np.ndarray:
        """Deterministic bag-of-tokens + char-3gram signed random projection."""
        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        lower = text.lower()
        words = re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", lower)
        features = list(words)
        # Add character 3-grams for morphological/paraphrase overlap
        compact = "".join(words)
        if len(compact) >= 3:
            features.extend(compact[i : i + 3] for i in range(len(compact) - 2))
        if not features:
            vec[0] = 1.0
            return vec
        for feat in features:
            digest = hashlib.md5(feat.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.embedding_dim
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        return vec / max(norm, 1e-12)

    def encode(self, texts: Sequence[str], normalize: bool = True) -> np.ndarray:
        """Encode texts into shape [N, d] embeddings."""
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        if self.use_pretrained_roberta and self.model is not None and self.tokenizer is not None:
            self.model.eval()
            with torch.no_grad():
                enc = self.tokenizer(
                    list(texts), padding=True, truncation=True, max_length=512, return_tensors="pt"
                ).to(self.device)
                out = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (out * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
                arr = pooled.cpu().numpy().astype(np.float32)
        else:
            arr = np.stack([self._hash_encode_one(t) for t in texts], axis=0)

        if normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / np.maximum(norms, 1e-12)
        return arr


class GCNEncoder:
    r"""Multi-layer Graph Convolutional Encoder producing frozen base vectors \tilde{z}_v."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        seed: int = 42,
    ) -> None:
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        rng = np.random.default_rng(seed)

        # Residual-initialized orthogonal/near-identity weights preserve initial RoBERTa semantics
        self.weights: List[np.ndarray] = []
        dims = [in_dim] + [hidden_dim] * num_layers
        for l_idx in range(num_layers):
            d_in, d_out = dims[l_idx], dims[l_idx + 1]
            if d_in == d_out:
                w = 0.85 * np.eye(d_in, dtype=np.float32) + rng.normal(
                    0.0, 0.05 / np.sqrt(d_in), size=(d_in, d_out)
                ).astype(np.float32)
            else:
                w = rng.normal(0.0, np.sqrt(2.0 / d_in), size=(d_in, d_out)).astype(np.float32)
            self.weights.append(w)

    def encode_graph(self, node_features: np.ndarray, norm_adj: np.ndarray) -> np.ndarray:
        r"""Propagate RoBERTa node features over dual-KG topology -> frozen \tilde{z}_v."""
        h = node_features.astype(np.float32)
        for l_idx, w in enumerate(self.weights):
            h_prop = norm_adj @ h @ w
            if l_idx < self.num_layers - 1:
                # Residual skip + LeakyReLU preserves distinctness on sparse legal graphs
                h = 0.5 * h + 0.5 * np.where(h_prop > 0, h_prop, 0.1 * h_prop)
            else:
                h = 0.5 * h + 0.5 * h_prop
        norms = np.linalg.norm(h, axis=1, keepdims=True)
        return (h / np.maximum(norms, 1e-12)).astype(np.float32)


if _HAS_TORCH:

    class TorchGCNEncoder(nn.Module):
        """PyTorch multi-layer GCN encoder matching Kipf & Welling [17] (Section 4.3)."""

        def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1) -> None:
            super().__init__()
            self.layers = nn.ModuleList()
            dims = [in_dim] + [hidden_dim] * num_layers
            for i in range(num_layers):
                self.layers.append(nn.Linear(dims[i], dims[i + 1], bias=False))
            self.dropout = dropout

        def forward(self, x: "torch.Tensor", norm_adj: "torch.Tensor") -> "torch.Tensor":
            h = x
            for idx, layer in enumerate(self.layers):
                h_next = torch.matmul(norm_adj, layer(h))
                if idx < len(self.layers) - 1:
                    h = F.relu(h_next)
                    h = F.dropout(h, p=self.dropout, training=self.training)
                else:
                    h = h_next
            return F.normalize(h, p=2, dim=-1)
