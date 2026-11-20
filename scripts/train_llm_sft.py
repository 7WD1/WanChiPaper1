#!/usr/bin/env python3
"""CLI Step 3: Build Evidence-Augmented SFT Dataset and (optionally) Fine-Tune LLAMA3-TAIDE-7B."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dual_graph_lqa.config import load_config
from dual_graph_lqa.contrastive import ContrastiveTrainer
from dual_graph_lqa.kg import AppliedKGBuilder, BasicKGBuilder, SVMDomainRouter
from dual_graph_lqa.llm import EvidencePromptBuilder, LegalLoRASFTTrainer
from dual_graph_lqa.retrieval import AdaptiveDualKGRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare evidence-augmented SFT data & run LoRA SFT.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument(
        "--run-training",
        action="store_true",
        help="Execute HuggingFace PEFT LoRA fine-tuning on LLAMA3-TAIDE-7B (requires GPU + torch/peft).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = ROOT / cfg.data.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    laws = BasicKGBuilder.load_regulations(ROOT / cfg.data.statutes_path)
    cases = BasicKGBuilder.load_cases(ROOT / cfg.data.cases_path)
    qa_pairs = AppliedKGBuilder.load_qa_pairs(ROOT / cfg.data.qa_train_path)
    synonyms = AppliedKGBuilder.load_synonyms(ROOT / cfg.data.synonyms_path)

    basic_kg = BasicKGBuilder().build(laws, cases)
    applied_kg = AppliedKGBuilder(synonyms=synonyms).build(qa_pairs)

    router = SVMDomainRouter(svm_c=cfg.basic_kg.svm_c, min_confidence=cfg.basic_kg.min_confidence)
    router.fit(
        [a.text for a in basic_kg.articles.values()] + [qa.question for qa in qa_pairs],
        [a.domain for a in basic_kg.articles.values()] + [qa.domain for qa in qa_pairs],
    )

    c_trainer = ContrastiveTrainer(cfg=cfg.contrastive, seed=cfg.seed)
    graph_index, _ = c_trainer.fit(basic_kg, applied_kg, qa_pairs)

    retriever = AdaptiveDualKGRetriever(
        basic_kg=basic_kg,
        applied_kg=applied_kg,
        graph_index=graph_index,
        contrastive_trainer=c_trainer,
        cfg=cfg.retrieval,
        domain_router=router,
    )

    prompt_builder = EvidencePromptBuilder()
    sft_records = []
    for qa in qa_pairs:
        trace = retriever.retrieve(qa.question, top_k_cap=cfg.retrieval.top_k_cap)
        sft_records.append(prompt_builder.build_sft_example(qa.question, trace.evidence, qa.answer))

    sft_trainer = LegalLoRASFTTrainer(cfg.llm_sft)
    jsonl_path = sft_trainer.export_sft_dataset(sft_records, out_dir / "sft_train.jsonl")
    print(f"[SFT Data] Exported {len(sft_records)} constraint-verified evidence prompts to {jsonl_path}")

    if args.run_training:
        adapter_dir = sft_trainer.train(jsonl_path, out_dir / "llama3_taide_lora")
        print(f"[Done] Saved LoRA adapter checkpoint to {adapter_dir}")
    else:
        print(
            "[Dry-Run] SFT dataset prepared successfully. Pass --run-training on a CUDA GPU "
            "machine with torch/transformers/peft installed to launch LLAMA3-TAIDE-7B LoRA SFT."
        )


if __name__ == "__main__":
    main()
