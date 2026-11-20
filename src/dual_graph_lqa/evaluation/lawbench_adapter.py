"""16-Task Domain-Filtered LawBench Adapter (Section 5.1, Table 7).

Excludes the 4 non-corporate tasks (2-3 Marital Disputes, 3-4/3-5 Prison Term
Prediction, 3-7 Criminal Damages Calculation) and applies unified output parsing
across all evaluated models (Exp 1-1..1-6, Exp 2-1..2-5, Exp 3-1..3-3).
"""

from __future__ import annotations

import re
from typing import Dict, List

EXCLUDED_TASKS = ("2-3", "3-4", "3-5", "3-7")

RETAINED_LAWBENCH_TASKS: Dict[str, Dict[str, object]] = {
    "1-1": {"level": "Memory", "name": "Statute Paraphrasing", "metric": "ROUGE-L", "retained_n": 500},
    "1-2": {"level": "Memory", "name": "Knowledge Question Answering", "metric": "Accuracy", "retained_n": 500},
    "2-1": {"level": "Understanding", "name": "Document Alignment", "metric": "F0.5", "retained_n": 500},
    "2-2": {"level": "Understanding", "name": "Dispute Focus Identification", "metric": "F1", "retained_n": 500},
    "2-4": {"level": "Understanding", "name": "Issue Topic Identification", "metric": "Accuracy", "retained_n": 500},
    "2-5": {"level": "Understanding", "name": "Reading Comprehension", "metric": "rc-F1", "retained_n": 500},
    "2-6": {"level": "Understanding", "name": "Named Entity Recognition", "metric": "soft-F1", "retained_n": 500},
    "2-7": {"level": "Understanding", "name": "Case Summary", "metric": "ROUGE-L", "retained_n": 500},
    "2-8": {"level": "Understanding", "name": "Argument Mining", "metric": "Accuracy", "retained_n": 500},
    "2-9": {"level": "Understanding", "name": "Event Detection", "metric": "F1", "retained_n": 500},
    "2-10": {"level": "Understanding", "name": "Trigger Word Extraction", "metric": "soft-F1", "retained_n": 500},
    "3-1": {"level": "Application", "name": "Statute Prediction (Fact-based)", "metric": "F1", "retained_n": 500},
    "3-2": {"level": "Application", "name": "Statute Prediction (Scenario-based)", "metric": "ROUGE-L", "retained_n": 500},
    "3-3": {"level": "Application", "name": "Charge Prediction", "metric": "F1", "retained_n": 500},
    "3-6": {"level": "Application", "name": "Case Analysis", "metric": "Accuracy", "retained_n": 500},
    "3-8": {"level": "Application", "name": "Legal Consultation", "metric": "ROUGE-L", "retained_n": 500},
}


class LawBenchSubsetEvaluator:
    """Unified task registry, choice parser, and category-mean aggregator for the 16 tasks."""

    @staticmethod
    def filter_tasks(task_scores: Dict[str, float]) -> Dict[str, float]:
        return {tid: sc for tid, sc in task_scores.items() if tid in RETAINED_LAWBENCH_TASKS}

    @staticmethod
    def parse_single_choice(raw_output: str) -> str:
        """Unified single-choice option parser applied identically across all compared systems."""
        m = re.search(r"\b([A-D])\b", raw_output.upper())
        return m.group(1) if m else ""

    @staticmethod
    def aggregate_scores(task_scores: Dict[str, float]) -> Dict[str, float]:
        """Compute category means (Memory, Understanding, Application) and overall 16-task mean."""
        by_cat: Dict[str, List[float]] = {"Memory": [], "Understanding": [], "Application": []}
        all_scores: List[float] = []
        for tid, meta in RETAINED_LAWBENCH_TASKS.items():
            if tid in task_scores:
                val = float(task_scores[tid])
                by_cat[str(meta["level"])].append(val)
                all_scores.append(val)
        return {
            "Memory": sum(by_cat["Memory"]) / max(len(by_cat["Memory"]), 1),
            "Understanding": sum(by_cat["Understanding"]) / max(len(by_cat["Understanding"]), 1),
            "Application": sum(by_cat["Application"]) / max(len(by_cat["Application"]), 1),
            "Overall": sum(all_scores) / max(len(all_scores), 1),
        }
