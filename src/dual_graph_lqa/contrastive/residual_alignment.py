r"""Module 3: Residual Contrastive Learning over KG Nodes (Section 4.3, Eqs. 34-49).

Implements exact equations from Section 4.3:
  Eq. (34): \tilde{y}_v = \tilde{z}_v + r_v + W \tilde{z}_v   (with \tilde{z}_v FROZEN)
  Eq. (35): \hat{z}_v = \tilde{y}_v / ||\tilde{y}_v||
  Eq. (36): \hat{h}_i = h_{q_i} / ||h_{q_i}||
  Eq. (37): V_i^+ = V_{a_i}^{basic} U V_{a_i}^{applied},  V_i^- = V \ V_i^+
  Eq. (38): S_{i,v} = (\hat{h}_i^T \hat{z}_v) / \tau
  Eq. (39): p_{i,v} = exp(S_{i,v}) / sum_{u in C_i} exp(S_{i,u})
  Eq. (40): t_{i,v} = 1/|V_i^+| if v in V_i^+ else 0
  Eq. (41): L_i = -sum_{v in C_i} t_{i,v} log p_{i,v},  L = (1/n) sum_i L_i
  Eq. (42)-(44): Exact analytical gradients w.r.t. \hat{z}_v, r_v, W, and h_{q_i}
  Eq. (45): L_res    = \lambda_r sum_v ||r_v||^2
  Eq. (46): L_smooth = \lambda_s sum_{(u,v) in E} w_{uv} ||r_u - r_v||^2
  Eq. (47): L_align  = \lambda_a sum_{(u,v) in R_align} ||\hat{z}_u - \hat{z}_v||^2
  Eq. (48): L_head   = \lambda_w ||W||_F^2
  Eq. (49): J        = L + L_res + L_smooth + L_align + L_head

Provides both:
  1. `NumpyResidualContrastiveAligner` (uses exact closed-form gradients Eqs. 42-44,
     zero external DL dependencies, runnable anywhere in milliseconds).
  2. `TorchResidualContrastiveAligner` (full PyTorch nn.Module for end-to-end GPU
     fine-tuning jointly with a HuggingFace RoBERTa encoder).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


@dataclass
class ResidualContrastiveOutput:
    total_loss: float
    infonce_loss: float
    l_res: float
    l_smooth: float
    l_align: float
    l_head: float


class NumpyResidualContrastiveAligner:
    """Exact NumPy implementation of Algorithm 3 (Eqs. 34-49) with analytical gradients."""

    def __init__(
        self,
        frozen_base_embeddings: np.ndarray,
        temperature: float = 0.07,
        lambda_res: float = 0.01,
        lambda_smooth: float = 0.01,
        lambda_align: float = 0.05,
        lambda_head: float = 0.001,
        use_residual: bool = True,
    ) -> None:
        self.base_z = frozen_base_embeddings.astype(np.float32).copy()
        self.base_z.setflags(write=False)  # Guarantee \tilde{z}_v stays strictly frozen!
        self.num_nodes, self.dim = self.base_z.shape

        self.temperature = temperature
        self.lambda_res = lambda_res
        self.lambda_smooth = lambda_smooth
        self.lambda_align = lambda_align
        self.lambda_head = lambda_head
        self.use_residual = use_residual

        # Trainable residual parameters r_v in R^d and shared linear head W in R^{d x d}
        self.r = np.zeros((self.num_nodes, self.dim), dtype=np.float32)
        self.W = np.zeros((self.dim, self.dim), dtype=np.float32)
        # Trainable lightweight query linear adapter simulating RoBERTa projection head
        self.Q_proj = np.eye(self.dim, dtype=np.float32)

    def compute_node_representations(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Eqs. (34)-(35): \tilde{y}_v = \tilde{z}_v + r_v + W \tilde{z}_v; \hat{z}_v = \tilde{y}_v / ||\tilde{y}_v||."""
        if self.use_residual:
            y_tilde = self.base_z + self.r + (self.base_z @ self.W.T)
        else:
            # Exp 2-1 ablation: w/o residual fine-tuning (r_v, W)
            y_tilde = self.base_z.copy()
        y_norms = np.maximum(np.linalg.norm(y_tilde, axis=1, keepdims=True), 1e-12)
        z_hat = y_tilde / y_norms
        return y_tilde, y_norms, z_hat

    def encode_queries(self, raw_query_vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Eq. (36): h_{q_i} = Enc(q_i), \hat{h}_i = h_{q_i} / ||h_{q_i}||."""
        h_q = raw_query_vecs.astype(np.float32) @ self.Q_proj.T
        h_norms = np.maximum(np.linalg.norm(h_q, axis=1, keepdims=True), 1e-12)
        h_hat = h_q / h_norms
        return h_q, h_norms, h_hat

    def step(
        self,
        raw_query_vecs: np.ndarray,
        positive_sets: Sequence[Sequence[int]],
        graph_edges: Sequence[Tuple[int, int, float]],
        align_pairs: Sequence[Tuple[int, int]],
        lr: float = 1.0e-2,
    ) -> ResidualContrastiveOutput:
        """Compute exact objective J (Eq. 49) and apply analytical gradient update (Eqs. 42-44)."""
        n_queries = len(positive_sets)
        if n_queries == 0:
            return ResidualContrastiveOutput(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        y_tilde, y_norms, z_hat = self.compute_node_representations()
        h_q, h_norms, h_hat = self.encode_queries(raw_query_vecs)

        # Eq. (38)-(39): Temperature-scaled cosine similarity & softmax
        logits = (h_hat @ z_hat.T) / self.temperature
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Eq. (40)-(41): Uniform target distribution t_{i,v} & Multi-positive InfoNCE
        targets = np.zeros_like(probs)
        loss_list: List[float] = []
        for i, pos_ids in enumerate(positive_sets):
            valid_pos = [pid for pid in pos_ids if 0 <= pid < self.num_nodes]
            if not valid_pos:
                continue
            weight = 1.0 / len(valid_pos)
            for pid in valid_pos:
                targets[i, pid] = weight
            loss_i = -float(np.sum(targets[i, valid_pos] * np.log(np.maximum(probs[i, valid_pos], 1e-12))))
            loss_list.append(loss_i)

        infonce_loss = float(np.mean(loss_list)) if loss_list else 0.0

        # Eq. (45)-(48): Regularization terms
        l_res = float(self.lambda_res * np.sum(self.r ** 2)) if self.use_residual else 0.0
        l_smooth = 0.0
        grad_r_smooth = np.zeros_like(self.r)
        if self.use_residual and graph_edges:
            for u, v, w_uv in graph_edges:
                if 0 <= u < self.num_nodes and 0 <= v < self.num_nodes:
                    diff = self.r[u] - self.r[v]
                    l_smooth += float(self.lambda_smooth * w_uv * np.sum(diff ** 2))
                    g = 2.0 * self.lambda_smooth * w_uv * diff
                    grad_r_smooth[u] += g
                    grad_r_smooth[v] -= g

        l_align = 0.0
        grad_zhat_align = np.zeros_like(z_hat)
        for u, v in align_pairs:
            if 0 <= u < self.num_nodes and 0 <= v < self.num_nodes:
                diff = z_hat[u] - z_hat[v]
                l_align += float(self.lambda_align * np.sum(diff ** 2))
                g = 2.0 * self.lambda_align * diff
                grad_zhat_align[u] += g
                grad_zhat_align[v] -= g

        l_head = float(self.lambda_head * np.sum(self.W ** 2)) if self.use_residual else 0.0
        total_loss = infonce_loss + l_res + l_smooth + l_align + l_head

        # Analytical gradients (Eqs. 42-44)
        coeff = (probs - targets) / (self.temperature * max(len(loss_list), 1))  # [n, |V|]

        # Eq. (44): Query gradient dL/dh_{q_i} and update Q_proj
        grad_h_hat = coeff @ z_hat  # [n, d]
        dot_h = np.sum(grad_h_hat * h_hat, axis=1, keepdims=True)
        grad_h_q = (grad_h_hat - dot_h * h_hat) / h_norms
        grad_Q_proj = grad_h_q.T @ raw_query_vecs.astype(np.float32)
        self.Q_proj -= lr * np.clip(grad_Q_proj, -1.0, 1.0)

        # Eq. (42)-(43): Node residual & shared head updates (only if use_residual=True)
        if self.use_residual:
            # dL/dz_hat (Eq. 42) + cross-graph alignment gradient
            grad_z_hat = (coeff.T @ h_hat) + grad_zhat_align  # [|V|, d]
            # Chain rule through L2 normalization (Eq. 43)
            dot_z = np.sum(grad_z_hat * z_hat, axis=1, keepdims=True)
            grad_y_tilde = (grad_z_hat - dot_z * z_hat) / y_norms

            grad_r = grad_y_tilde + (2.0 * self.lambda_res * self.r) + grad_r_smooth
            grad_W = (grad_y_tilde.T @ self.base_z) + (2.0 * self.lambda_head * self.W)

            self.r -= lr * np.clip(grad_r, -1.0, 1.0)
            self.W -= lr * np.clip(grad_W, -1.0, 1.0)

        return ResidualContrastiveOutput(
            total_loss=total_loss,
            infonce_loss=infonce_loss,
            l_res=l_res,
            l_smooth=l_smooth,
            l_align=l_align,
            l_head=l_head,
        )

    def get_normalized_node_embeddings(self) -> np.ndarray:
        _, _, z_hat = self.compute_node_representations()
        return z_hat


if _HAS_TORCH:

    class TorchResidualContrastiveAligner(nn.Module):
        """Full PyTorch nn.Module implementing Section 4.3 (Eqs. 34-49) for GPU training."""

        def __init__(
            self,
            frozen_base_embeddings: "torch.Tensor",
            temperature: float = 0.07,
            lambda_res: float = 0.01,
            lambda_smooth: float = 0.01,
            lambda_align: float = 0.05,
            lambda_head: float = 0.001,
            use_residual: bool = True,
        ) -> None:
            super().__init__()
            # Register \tilde{z}_v as a non-trainable buffer so it stays strictly frozen
            self.register_buffer("base_z", frozen_base_embeddings.detach().clone().float())
            num_nodes, dim = self.base_z.shape
            self.temperature = temperature
            self.lambda_res = lambda_res
            self.lambda_smooth = lambda_smooth
            self.lambda_align = lambda_align
            self.lambda_head = lambda_head
            self.use_residual = use_residual

            self.r = nn.Parameter(torch.zeros(num_nodes, dim))
            self.W = nn.Parameter(torch.zeros(dim, dim))

        def get_normalized_node_embeddings(self) -> "torch.Tensor":
            """Eqs. (34)-(35)."""
            if self.use_residual:
                y_tilde = self.base_z + self.r + F.linear(self.base_z, self.W)
            else:
                y_tilde = self.base_z
            return F.normalize(y_tilde, p=2, dim=-1)

        def forward(
            self,
            query_embeddings: "torch.Tensor",
            positive_mask: "torch.Tensor",
            edge_index: "torch.Tensor | None" = None,
            edge_weight: "torch.Tensor | None" = None,
            align_pairs: "torch.Tensor | None" = None,
        ) -> Dict[str, "torch.Tensor"]:
            """Compute total objective J = L + L_res + L_smooth + L_align + L_head (Eq. 49)."""
            z_hat = self.get_normalized_node_embeddings()
            h_hat = F.normalize(query_embeddings, p=2, dim=-1)

            # Eqs. (38)-(41): Multi-positive InfoNCE
            logits = torch.matmul(h_hat, z_hat.T) / self.temperature
            log_probs = F.log_softmax(logits, dim=-1)
            pos_counts = torch.clamp(positive_mask.sum(dim=-1, keepdim=True), min=1.0)
            targets = positive_mask / pos_counts
            infonce = -(targets * log_probs).sum(dim=-1).mean()

            # Eq. (45): Residual penalty
            l_res = self.lambda_res * (self.r.pow(2).sum()) if self.use_residual else logits.new_tensor(0.0)

            # Eq. (46): Graph smoothness
            if self.use_residual and edge_index is not None and edge_index.numel() > 0:
                u, v = edge_index[0], edge_index[1]
                w = edge_weight if edge_weight is not None else torch.ones_like(u, dtype=z_hat.dtype)
                l_smooth = self.lambda_smooth * (w * (self.r[u] - self.r[v]).pow(2).sum(dim=-1)).sum()
            else:
                l_smooth = logits.new_tensor(0.0)

            # Eq. (47): Cross-graph alignment
            if align_pairs is not None and align_pairs.numel() > 0:
                u_a, v_a = align_pairs[:, 0], align_pairs[:, 1]
                l_align = self.lambda_align * (z_hat[u_a] - z_hat[v_a]).pow(2).sum()
            else:
                l_align = logits.new_tensor(0.0)

            # Eq. (48): Shared linear head Frobenius decay
            l_head = self.lambda_head * (self.W.pow(2).sum()) if self.use_residual else logits.new_tensor(0.0)

            total = infonce + l_res + l_smooth + l_align + l_head
            return {
                "total_loss": total,
                "infonce_loss": infonce,
                "l_res": l_res,
                "l_smooth": l_smooth,
                "l_align": l_align,
                "l_head": l_head,
            }
