#!/usr/bin/env python3
"""CLI Step 5: Evaluate Section 3 Metrics, FAQ Rubric Scores, and Adaptive Budget Decoupling."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dual_graph_lqa.config import load_config
from dual_graph_lqa.contrastive import ContrastiveTrainer
from dual_graph_lqa.evaluation import (
    FAQRubricJudge,
    compute_answerability_metrics,
    compute_article_level_metrics,
    compute_joint_objective,
    compute_semantic_similarity,
)
from dual_graph_lqa.kg import AppliedKGBuilder, BasicKGBuilder, SVMDomainRouter
from dual_graph_lqa.llm import CitationGroundedGenerator
from dual_graph_lqa.retrieval import AdaptiveDualKGRetriever


def evaluate_strategy(
    retriever: AdaptiveDualKGRetriever,
    generator: CitationGroundedGenerator,
    eval_items: list,
    universe_articles: set,
    use_adaptive: bool,
    top_k: int,
    alpha: float,
    beta: float,
    gamma: float,
) -> dict:
    y_true, y_pred = [], []
    gold_sets, pred_sets = [], []
    ref_texts, pred_texts = [], []
    retained_counts = []
    rubric_records = []

    for item in eval_items:
        trace = retriever.retrieve(
            item["question"],
            use_adaptive_threshold=use_adaptive,
            top_k_cap=top_k,
        )
        out = generator.generate(item["question"], trace.evidence)

        y_true.append(bool(item["is_answerable"]))
        y_pred.append(bool(out["predicted_answerable"]))
        gold_sets.append(set(item.get("gold_articles", [])))
        pred_sets.append(set(out["cited_articles"]))
        ref_texts.append(item["reference_answer"])
        pred_texts.append(out["answer"])
        retained_counts.append(trace.num_final_retained)

        rubric_records.append(
            {
                "question": item["question"],
                "reference_answer": item["reference_answer"],
                "model_response": out["answer"],
                "gold_articles": item.get("gold_articles", []),
                "cited_articles": out["cited_articles"],
            }
        )

    ans_m = compute_answerability_metrics(y_true, y_pred)
    art_m = compute_article_level_metrics(gold_sets, pred_sets, universe_articles)

    enc = retriever.trainer.text_encoder
    gold_vecs = enc.encode(ref_texts, normalize=True)
    pred_vecs = enc.encode(pred_texts, normalize=True)
    sem_m = compute_semantic_similarity(gold_vecs, pred_vecs)

    for rec, sim_val in zip(rubric_records, sem_m["per_query"]):
        rec["semantic_sim"] = sim_val
    rubric = FAQRubricJudge().evaluate_dataset(rubric_records)

    joint = compute_joint_objective(
        ans_m["f1"], art_m["f1"], float(sem_m["rho"]), alpha=alpha, beta=beta, gamma=gamma
    )
    return {
        "avg_retained_nodes": float(np.mean(retained_counts)),
        "answerability_f1": ans_m["f1"],
        "article_f1": art_m["f1"],
        "semantic_rho": float(sem_m["rho"]),
        "joint_P_M": joint,
        "faq_accuracy_pct": rubric["accuracy_pct"],
        "faq_explainability_pct": rubric["explainability_pct"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end evaluation on sample FAQ benchmark.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    laws = BasicKGBuilder.load_regulations(ROOT / cfg.data.statutes_path)
    cases = BasicKGBuilder.load_cases(ROOT / cfg.data.cases_path)
    qa_pairs = AppliedKGBuilder.load_qa_pairs(ROOT / cfg.data.qa_train_path)
    synonyms = AppliedKGBuilder.load_synonyms(ROOT / cfg.data.synonyms_path)
    eval_items = json.loads((ROOT / cfg.data.faq_eval_path).read_text(encoding="utf-8"))

    basic_kg = BasicKGBuilder().build(laws, cases)
    applied_kg = AppliedKGBuilder(synonyms=synonyms).build(qa_pairs)
    router = SVMDomainRouter(svm_c=cfg.basic_kg.svm_c, min_confidence=cfg.basic_kg.min_confidence)
    router.fit(
        [a.text for a in basic_kg.articles.values()] + [qa.question for qa in qa_pairs],
        [a.domain for a in basic_kg.articles.values()] + [qa.domain for qa in qa_pairs],
    )
    c_trainer = ContrastiveTrainer(cfg=cfg.contrastive, seed=cfg.seed)
    index, _ = c_trainer.fit(basic_kg, applied_kg, qa_pairs)

    retriever = AdaptiveDualKGRetriever(
        basic_kg=basic_kg,
        applied_kg=applied_kg,
        graph_index=index,
        contrastive_trainer=c_trainer,
        cfg=cfg.retrieval,
        domain_router=router,
    )
    generator = CitationGroundedGenerator(cfg=cfg.llm_sft, use_hf_model=False)
    universe = set(basic_kg.articles.keys())

    for label, adapt in [("Exp 4-1 Fixed Top-K=20", False), ("Exp 4-2 Adaptive theta(Q) + Top-K=20 Cap", True)]:
        res = evaluate_strategy(
            retriever,
            generator,
            eval_items,
            universe,
            use_adaptive=adapt,
            top_k=cfg.retrieval.top_k_cap,
            alpha=cfg.evaluation.alpha_answerability,
            beta=cfg.evaluation.beta_article,
            gamma=cfg.evaluation.gamma_semantic,
        )
        print(f"=== {label} ===")
        for k, v in res.items():
            print(f"  {k:24s}: {v:.4f}")


if __name__ == "__main__":
    main()
