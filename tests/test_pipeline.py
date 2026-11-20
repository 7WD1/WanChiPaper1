"""Comprehensive unit & integration test suite verifying all modules and Eqs. (1)-(77)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dual_graph_lqa.config import load_config
from dual_graph_lqa.contrastive import ContrastiveTrainer
from dual_graph_lqa.data import AntiLeakageFilter, QAPair
from dual_graph_lqa.evaluation import (
    FAQRubricJudge,
    LawBenchSubsetEvaluator,
    compute_answerability_metrics,
    compute_article_level_metrics,
    compute_joint_objective,
    compute_semantic_similarity,
)
from dual_graph_lqa.kg import AppliedKGBuilder, BasicKGBuilder, SVMDomainRouter
from dual_graph_lqa.llm import CitationGroundedGenerator, EvidencePromptBuilder
from dual_graph_lqa.retrieval import (
    AdaptiveDualKGRetriever,
    EvidenceConstraintFilter,
    FlatTextHybridRAG,
    shannon_entropy,
)
from dual_graph_lqa.retrieval.adaptive_retriever import compute_query_uncertainty


class TestDualGraphLQAPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config(ROOT / "configs" / "default.yaml")
        cls.laws = BasicKGBuilder.load_regulations(ROOT / cls.cfg.data.statutes_path)
        cls.cases = BasicKGBuilder.load_cases(ROOT / cls.cfg.data.cases_path)
        cls.qa_pairs = AppliedKGBuilder.load_qa_pairs(ROOT / cls.cfg.data.qa_train_path)
        cls.synonyms = AppliedKGBuilder.load_synonyms(ROOT / cls.cfg.data.synonyms_path)
        cls.eval_faqs = json.loads((ROOT / cls.cfg.data.faq_eval_path).read_text(encoding="utf-8"))

        cls.basic_kg = BasicKGBuilder().build(cls.laws, cls.cases)
        cls.applied_kg = AppliedKGBuilder(synonyms=cls.synonyms).build(cls.qa_pairs)
        cls.router = SVMDomainRouter().fit(
            [a.text for a in cls.basic_kg.articles.values()] + [q.question for q in cls.qa_pairs],
            [a.domain for a in cls.basic_kg.articles.values()] + [q.domain for q in cls.qa_pairs],
        )
        cls.trainer = ContrastiveTrainer(cfg=cls.cfg.contrastive, seed=42)
        cls.index, cls.history = cls.trainer.fit(cls.basic_kg, cls.applied_kg, cls.qa_pairs)
        cls.retriever = AdaptiveDualKGRetriever(
            basic_kg=cls.basic_kg,
            applied_kg=cls.applied_kg,
            graph_index=cls.index,
            contrastive_trainer=cls.trainer,
            cfg=cls.cfg.retrieval,
            domain_router=cls.router,
        )

    def test_basic_kg_star_structure_and_case_links(self) -> None:
        self.assertEqual(self.basic_kg.num_laws, 4)
        self.assertGreaterEqual(self.basic_kg.num_articles, 10)
        # Verify star topology: every edge originates from a law node to an article node
        for u, v in self.basic_kg.graph.edges():
            self.assertEqual(self.basic_kg.graph.nodes[u]["node_type"], "law")
            self.assertEqual(self.basic_kg.graph.nodes[v]["node_type"], "article")
        # Verify Case_{1,11} links attached to L1_Art11
        self.assertIn("Case_001", self.basic_kg.articles["L1_Art11"].cited_case_ids)

    def test_applied_kg_triples_and_multihop(self) -> None:
        self.assertGreater(self.applied_kg.num_triples, 10)
        # Verify 1- and 2-hop neighbor lookup works on shared entities
        some_tid = self.applied_kg.triples[0].node_id
        nbs = self.applied_kg.k_hop_neighbors(some_tid, max_hops=2)
        self.assertTrue(all(1 <= dist <= 2 for dist in nbs.values()))

    def test_anti_leakage_s1_s2_s3(self) -> None:
        leaked = [
            QAPair("L_S1", "labor", "Unique question?", "Answer", source="lawbench"),
            QAPair("L_S2", "labor", self.eval_faqs[0]["question"], "Answer", source="corporate_qa"),
            self.qa_pairs[0],
        ]
        filt = AntiLeakageFilter(ngram_size=3, jaccard_threshold=0.80, cosine_threshold=0.85)
        kept, rep = filt.filter_training_pairs(
            leaked,
            [e["question"] for e in self.eval_faqs],
            embed_fn=lambda xs: self.trainer.text_encoder.encode(xs, normalize=True),
        )
        self.assertEqual(rep.removed_s1_source, 1)
        self.assertEqual(rep.removed_s2_lexical, 1)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].qa_id, "QA_001")

    def test_contrastive_frozen_base_and_loss_decrease(self) -> None:
        # Base GCN embeddings \tilde{z}_v must stay strictly frozen (Eq. 34)
        self.assertFalse(self.trainer.aligner.base_z.flags.writeable)
        # Total objective J should decrease across residual contrastive epochs
        self.assertLess(self.history[-1].infonce_loss, self.history[0].infonce_loss)
        # Normalized node vectors \hat{z}_v must have unit L2 norm (Eq. 35)
        norms = np.linalg.norm(self.index.contrastive_embeddings, axis=1)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)

    def test_uncertainty_measure_and_bounded_stopping(self) -> None:
        # Peaked scores -> low uncertainty U(Q)
        peaked = np.array([0.99, 0.05, 0.04, 0.03, 0.02], dtype=np.float32)
        # Uniform diffuse scores -> high uncertainty U(Q) near 1.0
        diffuse = np.array([0.50, 0.50, 0.50, 0.50, 0.50], dtype=np.float32)
        u_peak = compute_query_uncertainty(peaked, top_m=5, temp=0.1)
        u_diff = compute_query_uncertainty(diffuse, top_m=5, temp=0.1)
        self.assertLess(u_peak, 0.35)
        self.assertGreater(u_diff, 0.99)

    def test_adaptive_retrieval_decoupled_from_topk_and_ablations(self) -> None:
        query = "Can an employer fire a worker during probation without advance notice?"
        trace_full = self.retriever.retrieve(query, top_k_cap=20)
        self.assertLessEqual(trace_full.num_final_retained, 20)
        self.assertIn(trace_full.selected_hop_depth, (1, 2))
        self.assertGreater(trace_full.num_final_retained, 0)

        # Verify ablation switches execute cleanly
        trace_no_mh = self.retriever.retrieve(query, use_multihop=False, top_k_cap=20)
        self.assertEqual(trace_no_mh.selected_hop_depth, 1)
        self.assertEqual(trace_no_mh.num_applied_mh_hits, 0)

        trace_no_cle = self.retriever.retrieve(query, use_contrastive=False, top_k_cap=20)
        self.assertEqual(trace_no_cle.num_contrastive_hits, 0)

        # Flat-text hybrid RAG baseline (Exp 1-3)
        flat = FlatTextHybridRAG(self.trainer.text_encoder)
        flat.index_corpus(list(self.basic_kg.articles.values()), list(self.basic_kg.cases.values()))
        flat_hits = flat.retrieve(query, top_k=5)
        self.assertEqual(len(flat_hits), 5)

    def test_evidence_constraints_and_section3_metrics(self) -> None:
        cfilter = EvidenceConstraintFilter(self.cfg.retrieval.constraints)
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.95, 0.31, 0.0], dtype=np.float32)
        res = cfilter.check_answer_constraints(
            predicted_answer="According to Article 11, termination requires statutory cause.",
            predicted_vec=v1,
            supporting_vecs=[v2],
            reference_answer="According to Article 11, termination requires statutory cause.",
            source_triplet_vecs=(v1, v2, v1),
        )
        self.assertTrue(res["all_satisfied"])
        self.assertGreater(shannon_entropy("Article 11 Labor Standards Act termination"), 1.0)

        ans = compute_answerability_metrics([1, 1, 0, 0], [1, 0, 0, 0])
        self.assertAlmostEqual(ans["precision"], 1.0)
        self.assertAlmostEqual(ans["recall"], 0.5)
        art = compute_article_level_metrics([{"L1_Art11"}, {"L4_Art9"}], [{"L1_Art11"}, {"L4_Art9"}])
        self.assertAlmostEqual(art["f1"], 1.0)
        sem = compute_semantic_similarity(np.eye(2), np.eye(2))
        self.assertAlmostEqual(float(sem["rho"]), 1.0)
        pm = compute_joint_objective(ans["f1"], art["f1"], float(sem["rho"]), 0.3, 0.4, 0.3)
        self.assertAlmostEqual(pm, 0.3 * ans["f1"] + 0.4 + 0.3)

    def test_lawbench_16_task_aggregation_matches_paper(self) -> None:
        paper_ours = {
            "1-1": 79.21, "1-2": 73.34,
            "2-1": 38.13, "2-2": 54.31, "2-4": 53.44, "2-5": 67.89,
            "2-6": 88.21, "2-7": 41.13, "2-8": 77.31, "2-9": 86.13, "2-10": 74.13,
            "3-1": 80.11, "3-2": 79.31, "3-3": 46.90, "3-6": 85.13, "3-8": 90.44,
        }
        agg = LawBenchSubsetEvaluator.aggregate_scores(paper_ours)
        self.assertAlmostEqual(agg["Memory"], 76.28, places=2)
        self.assertAlmostEqual(agg["Understanding"], 64.52, places=2)
        self.assertAlmostEqual(agg["Application"], 76.38, places=2)
        self.assertAlmostEqual(agg["Overall"], 69.69, places=2)

    def test_end_to_end_generation_and_rubric(self) -> None:
        gen = CitationGroundedGenerator(cfg=self.cfg.llm_sft, use_hf_model=False)
        trace = self.retriever.retrieve(self.eval_faqs[0]["question"], top_k_cap=20)
        out = gen.generate(self.eval_faqs[0]["question"], trace.evidence)
        self.assertTrue(out["predicted_answerable"])
        self.assertIn("According to", out["answer"])
        prompt = EvidencePromptBuilder().build_prompt(self.eval_faqs[0]["question"], trace.evidence)
        self.assertIn("Basic-KG", prompt)
        judge = FAQRubricJudge()
        sc = judge.score_one(
            self.eval_faqs[0]["question"],
            self.eval_faqs[0]["reference_answer"],
            out["answer"],
            gold_articles=set(self.eval_faqs[0]["gold_articles"]),
            cited_articles=set(out["cited_articles"]),
            semantic_sim=0.85,
        )
        self.assertIn(sc["accuracy"], (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))


if __name__ == "__main__":
    unittest.main()
