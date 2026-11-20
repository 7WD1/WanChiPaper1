"""Evidence-augmented prompt builder (Section 4.4 & Section 4.5, Algorithm 4 & 5).

Constructs:
  x_i = [Q_k, Instr, Input_k^basic, Input_k^applied, Input_k^contrast]
  X(Q) = (Instr, Q, E_basic(Q), E_applied(Q), E_ctr(Q))
"""

from __future__ import annotations

from typing import List, Sequence

from ..data.schemas import EvidenceSegment

DEFAULT_LEGAL_INSTRUCTION = (
    "You are an authoritative corporate legal assistant specializing in labor, insurance, "
    "and financial law. Answer the user question strictly grounded in the retrieved statutory "
    "provisions, judicial decisions, and semantic relations below:\n"
    "1. Cite the controlling statute and article number first.\n"
    "2. Provide concise clause-level legal justification tied directly to the cited provisions.\n"
    "3. Explain any multi-step statutory conditions clearly without hallucinating unsupported articles.\n"
    "4. If the query lies outside the retrieved legal corpus, state explicitly that it is out of scope."
)


class EvidencePromptBuilder:
    """Formats retrieved dual-KG + contrastive evidence segments into structured LLM prompts."""

    def __init__(self, instruction: str = DEFAULT_LEGAL_INSTRUCTION) -> None:
        self.instruction = instruction

    def build_prompt(self, query: str, evidence: Sequence[EvidenceSegment]) -> str:
        """Assemble X(Q) = (Instr, Q, E_basic(Q), E_applied(Q), E_ctr(Q))."""
        basic_lines: List[str] = []
        applied_lines: List[str] = []
        contrast_lines: List[str] = []
        other_lines: List[str] = []

        for seg in evidence:
            if seg.source_type == "basic":
                basic_lines.append(f"- (score={seg.score:.3f}) {seg.text}")
            elif seg.source_type in ("applied_1hop", "applied_mh"):
                hop_tag = f"hop={seg.hop}"
                applied_lines.append(f"- ({hop_tag}, score={seg.score:.3f}) {seg.text}")
            elif seg.source_type == "contrastive":
                contrast_lines.append(f"- (score={seg.score:.3f}) {seg.text}")
            else:
                other_lines.append(f"- (score={seg.score:.3f}) {seg.text}")

        sections = [
            f"### Instruction\n{self.instruction}",
            f"### User Query\n{query}",
            "### Basic-KG Statutory & Case Evidence (E_basic)\n"
            + ("\n".join(basic_lines) if basic_lines else "None"),
            "### Applied-KG Semantic Relations (E_applied)\n"
            + ("\n".join(applied_lines) if applied_lines else "None"),
            "### Contrastively Aligned Evidence (E_ctr)\n"
            + ("\n".join(contrast_lines) if contrast_lines else "None"),
        ]
        if other_lines:
            sections.append("### Retrieved Flat-Text Evidence\n" + "\n".join(other_lines))
        sections.append("### Citation-Grounded Legal Answer\n")
        return "\n\n".join(sections)

    def build_sft_example(self, query: str, evidence: Sequence[EvidenceSegment], gold_answer: str) -> dict:
        """Construct labeled SFT pair (x_i, y_i) for Eq. (50) next-token prediction."""
        return {
            "prompt": self.build_prompt(query, evidence),
            "completion": gold_answer,
        }
