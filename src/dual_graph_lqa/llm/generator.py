"""Inference-stage citation-grounded answer generator A(Q) = LLM(X(Q)) (Section 4.5)."""

from __future__ import annotations

from typing import List, Optional, Sequence

from ..config import LLMSFTConfig
from ..data.schemas import EvidenceSegment
from .prompt_builder import EvidencePromptBuilder


class CitationGroundedGenerator:
    """Wraps fine-tuned LLAMA3-TAIDE-7B + LoRA adapter with a dry-run CPU synthesis mode."""

    def __init__(
        self,
        cfg: LLMSFTConfig,
        adapter_path: Optional[str] = None,
        use_hf_model: bool = False,
        answerability_min_score: float = 0.12,
    ) -> None:
        self.cfg = cfg
        self.adapter_path = adapter_path
        self.use_hf_model = use_hf_model
        self.answerability_min_score = answerability_min_score
        self.prompt_builder = EvidencePromptBuilder()

        self._tokenizer = None
        self._model = None
        if use_hf_model:
            self._load_hf_model()

    def _load_hf_model(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.base_model, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.base_model,
            torch_dtype=torch.float16 if self.cfg.fp16 else torch.float32,
            device_map="auto",
        )
        if self.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_path)
        model.eval()
        self._model = model

    def generate(self, query: str, evidence: Sequence[EvidenceSegment]) -> dict:
        """Generate structured, citation-grounded answer A(Q) = LLM(X(Q))."""
        prompt = self.prompt_builder.build_prompt(query, evidence)
        top_score = max((seg.score for seg in evidence), default=0.0)
        is_answerable = bool(evidence) and top_score >= self.answerability_min_score

        cited_articles: List[str] = []
        for seg in evidence:
            if seg.article_id and seg.article_id not in cited_articles:
                cited_articles.append(seg.article_id)

        if self.use_hf_model and self._model is not None and self._tokenizer is not None:
            import torch

            inputs = self._tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=self.cfg.cutoff_length
            ).to(self._model.device)
            with torch.no_grad():
                out_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.cfg.max_new_tokens,
                    temperature=self.cfg.temperature,
                    do_sample=self.cfg.temperature > 0,
                )
            gen_ids = out_ids[0][inputs["input_ids"].shape[1] :]
            answer_text = self._tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        else:
            # Deterministic citation-grounded synthesis for CPU verification / smoke testing
            if not is_answerable:
                answer_text = (
                    "This query falls outside the scope of the indexed labor, insurance, and "
                    "financial statutory corpus; no controlling article applies."
                )
                cited_articles = []
            else:
                cite_header = ", ".join(cited_articles[:3]) if cited_articles else "retrieved legal provisions"
                snippets = [seg.text for seg in evidence[:3]]
                answer_text = (
                    f"According to {cite_header}, the controlling legal basis is grounded in the "
                    f"following retrieved evidence: " + " | ".join(snippets)
                )

        return {
            "query": query,
            "answer": answer_text,
            "predicted_answerable": is_answerable,
            "cited_articles": cited_articles,
            "prompt": prompt,
        }
