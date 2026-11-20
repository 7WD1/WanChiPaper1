#!/usr/bin/env python3
"""CLI Step 2: Train Module 3 (GCN + RoBERTa Residual Contrastive Alignment, Eqs. 34-49)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dual_graph_lqa.config import load_config
from dual_graph_lqa.contrastive import ContrastiveTrainer
from dual_graph_lqa.kg import AppliedKGBuilder, BasicKGBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Train residual contrastive alignment (Module 3).")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--epochs", type=int, default=None, help="Override contrastive epochs.")
    parser.add_argument("--no-residual", action="store_true", help="Exp 2-1: disable (r_v, W).")
    parser.add_argument(
        "--use-pretrained-roberta",
        action="store_true",
        help="Use HuggingFace RoBERTa encoder (requires torch & transformers).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.contrastive.epochs = args.epochs
    out_dir = ROOT / cfg.data.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    laws = BasicKGBuilder.load_regulations(ROOT / cfg.data.statutes_path)
    cases = BasicKGBuilder.load_cases(ROOT / cfg.data.cases_path)
    qa_pairs = AppliedKGBuilder.load_qa_pairs(ROOT / cfg.data.qa_train_path)
    synonyms = AppliedKGBuilder.load_synonyms(ROOT / cfg.data.synonyms_path)

    basic_kg = BasicKGBuilder().build(laws, cases)
    applied_kg = AppliedKGBuilder(synonyms=synonyms).build(qa_pairs)

    trainer = ContrastiveTrainer(
        cfg=cfg.contrastive,
        use_pretrained_roberta=args.use_pretrained_roberta,
        use_residual=not args.no_residual,
        seed=cfg.seed,
    )
    index, history = trainer.fit(basic_kg, applied_kg, qa_pairs)

    first, last = history[0], history[-1]
    print(
        f"[Contrastive Training] Nodes={len(index.node_ids)} "
        f"(Basic={len(index.basic_indices)}, Applied={len(index.applied_indices)}) | "
        f"Residual={'ON' if not args.no_residual else 'OFF (Exp 2-1)'}"
    )
    print(
        f"  Epoch 1   : J={first.total_loss:.4f} (InfoNCE={first.infonce_loss:.4f}, "
        f"L_res={first.l_res:.4f}, L_smooth={first.l_smooth:.4f}, L_align={first.l_align:.4f})"
    )
    print(
        f"  Epoch {len(history):<3d} : J={last.total_loss:.4f} (InfoNCE={last.infonce_loss:.4f}, "
        f"L_res={last.l_res:.4f}, L_smooth={last.l_smooth:.4f}, L_align={last.l_align:.4f})"
    )

    ckpt_path = out_dir / "contrastive_alignment.npz"
    trainer.save_checkpoint(ckpt_path)
    print(f"[Done] Saved residual contrastive checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
