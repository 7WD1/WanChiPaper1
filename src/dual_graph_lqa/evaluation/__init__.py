"""Evaluation suite: Section 3 metrics, LawBench 16-task adapter, and FAQ LLM judge."""

from .faq_rubric_judge import RUBRIC_LEVELS, FAQRubricJudge
from .lawbench_adapter import EXCLUDED_TASKS, RETAINED_LAWBENCH_TASKS, LawBenchSubsetEvaluator
from .metrics import (
    compute_answerability_metrics,
    compute_article_level_metrics,
    compute_joint_objective,
    compute_semantic_similarity,
)

__all__ = [
    "compute_answerability_metrics",
    "compute_article_level_metrics",
    "compute_semantic_similarity",
    "compute_joint_objective",
    "RUBRIC_LEVELS",
    "FAQRubricJudge",
    "EXCLUDED_TASKS",
    "RETAINED_LAWBENCH_TASKS",
    "LawBenchSubsetEvaluator",
]
