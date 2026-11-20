#!/usr/bin/env python3
"""CLI Step 4: End-to-End Dual-Graph Adaptive Retrieval & Citation-Grounded Generation.

Supports ALL experimental configurations and ablation variants from Tables 11, 12, and 14:
  full | flat_rag | bkg_only | akg_only | dual_kg_no_cle |
  no_residual | no_multihop | fixed_threshold | no_constraints | no_consistency | fixed_topk
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dual_graph_lqa.config import load_config
from dual_graph_lqa.contrastive import ContrastiveTrainer
from dual_graph_lqa.kg import AppliedKGBuilder, BasicKGBuilder, SVMDomainRouter
from dual_graph_lqa.llm import CitationGroundedGenerator
from dual_graph_lqa.retrieval import AdaptiveDualKGRetriever, FlatTextHybridRAG

PRESETS = {
    "full": dict(bkg=True, akg=True, cle=True, res=True, mh=True, mode="adaptive", cons=True, cross=True),
    "flat_rag": dict(flat=True),
    "bkg_only": dict(bkg=True, akg=False, cle=False, res=True, mh=False, mode="adaptive", cons=True, cross=True),
    "akg_only": dict(bkg=False, akg=True, cle=False, res=True, mh=True, mode="adaptive", cons=True, cross=True),
    "dual_kg_no_cle": dict(bkg=True, akg=True, cle=False, res=True, mh=True, mode="adaptive", cons=True, cross=True),
    "no_residual": dict(bkg=True, akg=True, cle=True, res=False, mh=True, mode="adaptive", cons=True, cross=True),
    "no_multihop": dict(bkg=True, akg=True, cle=True, res=True, mh=False, mode="adaptive", cons=True, cross=True),
    "fixed_threshold": dict(bkg=True, akg=True, cle=True, res=True, mh=True, mode="fixed_threshold", cons=True, cross=True),
    "no_constraints": dict(bkg=True, akg=True, cle=True, res=True, mh=True, mode="adaptive", cons=False, cross=False),
    "no_consistency": dict(bkg=True, akg=True, cle=True, res=True, mh=True, mode="adaptive", cons=True, cross=False),
    "fixed_topk": dict(bkg=True, akg=True, cle=True, res=True, mh=True, mode="fixed_topk", cons=True, cross=True),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with Contrastive Dual-Graph Legal RAG.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument(
        "--query",
        type=str,
        default="Can my employer fire me during probation without advance notice?",
        help="Legal question to answer.",
    )
    parser.add_argument("--preset", type=str, default="full", choices=sorted(PRESETS.keys()))
    parser.add_argument("--top-k", type=int, default=None, help="Override Top-K context budget cap.")
    parser.add_argument("--use-hf-llm", action="store_true", help="Load LLAMA3-TAIDE-7B via HuggingFace.")
    parser.add_argument("--adapter-path", type=str, default=None, help="Optional LoRA adapter directory.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    preset = PRESETS[args.preset]
    k_cap = args.top_k if args.top_k is not None else cfg.retrieval.top_k_cap

    laws = BasicKGBuilder.load_regulations(ROOT / cfg.data.statutes_path)
    cases = BasicKGBuilder.load_cases(ROOT / cfg.data.cases_path)
    qa_pairs = AppliedKGBuilder.load_qa_pairs(ROOT / cfg.data.qa_train_path)
    synonyms = AppliedKGBuilder.load_synonyms(ROOT / cfg.data.synonyms_path)

    basic_kg = BasicKGBuilder().build(laws, cases)
    applied_kg = AppliedKGBuilder(synonyms=synonyms).build(qa_pairs)
    generator = CitationGroundedGenerator(
        cfg=cfg.llm_sft, adapter_path=args.adapter_path, use_hf_model=args.use_hf_llm
    )

    if preset.get("flat"):
        c_trainer = ContrastiveTrainer(cfg=cfg.contrastive, seed=cfg.seed)
        flat_rag = FlatTextHybridRAG(c_trainer.text_encoder)
        flat_rag.index_corpus(list(basic_kg.articles.values()), list(basic_kg.cases.values()))
        ev = flat_rag.retrieve(args.query, top_k=k_cap)
        result = generator.generate(args.query, ev)
        print(f"=== Preset: {args.preset} (Exp 1-3 Flat-Text Hybrid RAG) | Retained={len(ev)}/{k_cap} ===")
        print("Cited Articles:", result["cited_articles"])
        print("Answer:", result["answer"])
        return

    cfg.retrieval.constraints.enable_consistency = bool(preset["cross"])
    router = SVMDomainRouter(svm_c=cfg.basic_kg.svm_c, min_confidence=cfg.basic_kg.min_confidence)
    router.fit(
        [a.text for a in basic_kg.articles.values()] + [qa.question for qa in qa_pairs],
        [a.domain for a in basic_kg.articles.values()] + [qa.domain for qa in qa_pairs],
    )
    c_trainer = ContrastiveTrainer(cfg=cfg.contrastive, use_residual=bool(preset["res"]), seed=cfg.seed)
    index, _ = c_trainer.fit(basic_kg, applied_kg, qa_pairs)

    retriever = AdaptiveDualKGRetriever(
        basic_kg=basic_kg,
        applied_kg=applied_kg,
        graph_index=index,
        contrastive_trainer=c_trainer,
        cfg=cfg.retrieval,
        domain_router=router,
    )
    trace = retriever.retrieve(
        args.query,
        use_basic_kg=bool(preset["bkg"]),
        use_applied_kg=bool(preset["akg"]),
        use_contrastive=bool(preset["cle"]),
        use_multihop=bool(preset["mh"]),
        threshold_mode=str(preset["mode"]),
        use_constraints=bool(preset["cons"]),
        top_k_cap=k_cap,
    )
    result = generator.generate(args.query, trace.evidence)

    print(f"=== Preset: {args.preset} (mode={trace.threshold_mode}) | Top-K Cap: {k_cap} ===")
    print(f"Routed Domain : {trace.routed_domain} ({trace.domain_probs})")
    print(
        f"Uncertainty U : {trace.uncertainty_u:.4f} (gate={cfg.retrieval.uncertainty_threshold}) -> "
        f"Selected Hop Depth h*(Q) = {trace.selected_hop_depth}"
    )
    print(
        f"Retained Nodes: {trace.num_final_retained} / {trace.top_k_cap} "
        f"(Basic={trace.num_basic_hits}, Applied-1hop={trace.num_applied_1hop_hits}, "
        f"Applied-MH={trace.num_applied_mh_hits}, Contrastive={trace.num_contrastive_hits})"
    )
    print(f"Cited Articles: {result['cited_articles']}")
    print(f"Answer        : {result['answer']}")


if __name__ == "__main__":
    main()
