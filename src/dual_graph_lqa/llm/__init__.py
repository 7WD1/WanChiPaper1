"""Module 4: Evidence-Augmented Prompt Construction, LoRA SFT, and Answer Generation."""

from .generator import CitationGroundedGenerator
from .prompt_builder import DEFAULT_LEGAL_INSTRUCTION, EvidencePromptBuilder
from .sft_trainer import LegalLoRASFTTrainer

__all__ = [
    "DEFAULT_LEGAL_INSTRUCTION",
    "EvidencePromptBuilder",
    "LegalLoRASFTTrainer",
    "CitationGroundedGenerator",
]
