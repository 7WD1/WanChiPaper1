"""Six-level Rubric-Based FAQ Evaluation Harness (Section 5.3.2, Table 13)."""

from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional, Sequence, Set

RUBRIC_LEVELS: Dict[float, str] = {
    0.0: "Does not answer the question and cites no relevant statutes.",
    0.2: "Slightly related but fails to solve the problem; cited statutes are unrelated.",
    0.4: "Captures part of key points with obvious errors/omissions; citations misapplied.",
    0.6: "Addresses main part with minor errors/omissions; citations mostly correct.",
    0.8: "Clear and accurate with minor flaws; cited statutes are correct and aligned.",
    1.0: "Fully correct, detailed, error-free; cited statutes fully relevant and clear.",
}

JUDGE_PROMPT_TEMPLATE = """You are an impartial legal evaluator grading a Legal QA response against a reference answer.
Focus strictly on substantive legal correctness and statutory citation grounding (ignore length and stylistic fluency).

Rubric Scale (choose ONLY from 0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
- 0.0: Does not answer and cites no relevant statutes.
- 0.2: Slightly related; fails to solve; unrelated statutes.
- 0.4: Partial key points with major errors/omissions; misapplied statutes.
- 0.6: Main point addressed with minor errors; mostly correct statutes.
- 0.8: Clear, accurate, minor flaws; correct statutes aligned with explanation.
- 1.0: Fully correct and error-free; completely relevant statutes and clear legal basis.

Question: {question}
Reference Answer: {reference_answer}
Model Response: {model_response}

Return valid JSON only: {{"accuracy": <score>, "explainability": <score>}}
"""


def _snap_to_rubric(val: float) -> float:
    return min(RUBRIC_LEVELS.keys(), key=lambda lvl: abs(lvl - val))


class FAQRubricJudge:
    """Computes six-level Accuracy (%) and Explainability (%) using LLM judge or heuristic fallback."""

    def __init__(self, llm_judge_fn: Optional[Callable[[str], str]] = None) -> None:
        self.llm_judge_fn = llm_judge_fn

    def score_one(
        self,
        question: str,
        reference_answer: str,
        model_response: str,
        gold_articles: Set[str] | None = None,
        cited_articles: Set[str] | None = None,
        semantic_sim: float | None = None,
    ) -> Dict[str, float]:
        if self.llm_judge_fn is not None:
            prompt = JUDGE_PROMPT_TEMPLATE.format(
                question=question,
                reference_answer=reference_answer,
                model_response=model_response,
            )
            raw = self.llm_judge_fn(prompt)
            parsed = json.loads(raw)
            return {
                "accuracy": _snap_to_rubric(float(parsed["accuracy"])),
                "explainability": _snap_to_rubric(float(parsed["explainability"])),
            }

        # Deterministic proxy scorer for offline/unit-test verification
        sim = 0.0 if semantic_sim is None else max(0.0, min(1.0, semantic_sim))
        gold = gold_articles or set()
        cited = cited_articles or set()
        if not gold:
            cite_f1 = 1.0 if not cited else 0.2
        else:
            inter = len(gold & cited)
            cite_f1 = (2.0 * inter / (len(gold) + len(cited))) if (gold or cited) else 0.0
        acc_raw = 0.55 * sim + 0.45 * cite_f1
        exp_raw = 0.70 * cite_f1 + 0.30 * sim
        return {
            "accuracy": _snap_to_rubric(acc_raw),
            "explainability": _snap_to_rubric(exp_raw),
        }

    def evaluate_dataset(self, records: Sequence[Dict[str, object]]) -> Dict[str, float]:
        if not records:
            return {"accuracy_pct": 0.0, "explainability_pct": 0.0}
        accs: List[float] = []
        exps: List[float] = []
        for rec in records:
            sc = self.score_one(
                question=str(rec["question"]),
                reference_answer=str(rec["reference_answer"]),
                model_response=str(rec["model_response"]),
                gold_articles=set(rec.get("gold_articles", [])),  # type: ignore[arg-type]
                cited_articles=set(rec.get("cited_articles", [])),  # type: ignore[arg-type]
                semantic_sim=float(rec.get("semantic_sim", 0.0)),  # type: ignore[arg-type]
            )
            accs.append(sc["accuracy"])
            exps.append(sc["explainability"])
        return {
            "accuracy_pct": float(100.0 * sum(accs) / len(accs)),
            "explainability_pct": float(100.0 * sum(exps) / len(exps)),
        }
