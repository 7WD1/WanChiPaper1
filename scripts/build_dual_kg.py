#!/usr/bin/env python3
"""CLI Step 1: Build Basic KG, Applied KG, SVM+Platt Domain Router, and Anti-Leakage Audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dual_graph_lqa.config import load_config
from dual_graph_lqa.contrastive.gcn_encoder import TextEncoder
from dual_graph_lqa.data.anti_leakage import AntiLeakageFilter
from dual_graph_lqa.kg import AppliedKGBuilder, BasicKGBuilder, SVMDomainRouter


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Basic KG and Applied KG (Modules 1 & 2).")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = ROOT / cfg.data.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load corpora
    laws = BasicKGBuilder.load_regulations(ROOT / cfg.data.statutes_path)
    cases = BasicKGBuilder.load_cases(ROOT / cfg.data.cases_path)
    qa_pairs = AppliedKGBuilder.load_qa_pairs(ROOT / cfg.data.qa_train_path)
    synonyms = AppliedKGBuilder.load_synonyms(ROOT / cfg.data.synonyms_path)
    eval_faqs = json.loads((ROOT / cfg.data.faq_eval_path).read_text(encoding="utf-8"))
    eval_questions = [item["question"] for item in eval_faqs]

    # 2. Enforce 4-stage Anti-Leakage protocol (Section 5.1, S1-S4)
    encoder = TextEncoder(embedding_dim=cfg.contrastive.embedding_dim, use_pretrained_roberta=False)
    leak_filter = AntiLeakageFilter(
        ngram_size=cfg.data.anti_leakage.ngram_size,
        jaccard_threshold=cfg.data.anti_leakage.jaccard_threshold,
        cosine_threshold=cfg.data.anti_leakage.cosine_threshold,
    )
    clean_qa, leak_report = leak_filter.filter_training_pairs(
        qa_pairs, eval_questions, embed_fn=lambda xs: encoder.encode(xs, normalize=True)
    )
    print(
        f"[Anti-Leakage S1-S4] Input={leak_report.total_input}, Retained={leak_report.retained}, "
        f"Removed(S1={leak_report.removed_s1_source}, S2={leak_report.removed_s2_lexical}, "
        f"S3={leak_report.removed_s3_paraphrase})"
    )

    # 3. Module 1: Build Basic KG + SVM Platt Domain Router
    basic_kg = BasicKGBuilder().build(laws, cases)
    basic_kg.save(out_dir / "basic_kg.json")
    print(
        f"[Basic KG] Built {basic_kg.num_laws} laws, {basic_kg.num_articles} articles, "
        f"{len(basic_kg.cases)} cases, {basic_kg.num_edges} law-article edges."
    )

    router = SVMDomainRouter(
        svm_c=cfg.basic_kg.svm_c, min_confidence=cfg.basic_kg.min_confidence
    )
    router_texts = [a.text for a in basic_kg.articles.values()] + [qa.question for qa in clean_qa]
    router_labels = [a.domain for a in basic_kg.articles.values()] + [qa.domain for qa in clean_qa]
    router.fit(router_texts, router_labels)
    print(f"[Domain Router] Fitted SVM+Platt router over domains: {router.classes_}")

    # 4. Module 2: Build Applied KG strictly from deduplicated training QA pairs
    akg_builder = AppliedKGBuilder(
        synonyms=synonyms if cfg.applied_kg.use_synonym_expansion else {},
        bm25_top_keywords=cfg.applied_kg.bm25_top_keywords,
        use_jieba=cfg.applied_kg.use_jieba,
        enforce_statutory_prefix=cfg.applied_kg.enforce_statutory_prefix,
    )
    applied_kg = akg_builder.build(clean_qa)
    applied_kg.save(out_dir / "applied_kg.json")
    print(
        f"[Applied KG] Built {applied_kg.num_entities} entities and {applied_kg.num_triples} semantic triples."
    )
    print(f"[Done] Saved dual-KG artifacts to {out_dir}")


if __name__ == "__main__":
    main()
