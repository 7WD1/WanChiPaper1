"""Module 4: Evidence-Augmented LoRA Supervised Fine-Tuning (Section 4.4, Eq. 50, Table 10).

Fine-tunes LLAMA3-TAIDE-7B using HuggingFace Transformers + PEFT LoRA with exact
Table 10 hyperparameters:
  - Cutoff length: 4096
  - Learning rate: 5e-5 (cosine schedule)
  - Epochs: 40
  - Batch size: 4, Gradient accumulation steps: 64
  - Max grad norm: 1.0
  - LoRA rank: 16, dropout: 0.1
  - Precision: FP16
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

from ..config import LLMSFTConfig


class LegalLoRASFTTrainer:
    """Supervised fine-tuning pipeline for LLAMA3-TAIDE-7B (Algorithm 4 / Table 5)."""

    def __init__(self, cfg: LLMSFTConfig) -> None:
        self.cfg = cfg

    @staticmethod
    def export_sft_dataset(examples: Sequence[Dict[str, str]], output_jsonl: str | Path) -> Path:
        """Persist evidence-augmented (x_i, y_i) training pairs to JSONL."""
        out_path = Path(output_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for ex in examples:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
        return out_path

    def train(self, dataset_jsonl: str | Path, output_dir: str | Path) -> Path:
        """Launch PEFT LoRA supervised fine-tuning optimizing L_SFT (Eq. 50)."""
        try:
            import torch
            from datasets import load_dataset
            from peft import LoraConfig, TaskType, get_peft_model
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                DataCollatorForSeq2Seq,
                Trainer,
                TrainingArguments,
            )
        except ImportError as exc:
            raise ImportError(
                "Full LLM LoRA fine-tuning requires torch, transformers, peft, and datasets. "
                "Install them via `pip install -r requirements.txt`."
            ) from exc

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(self.cfg.base_model, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.base_model,
            torch_dtype=torch.float16 if self.cfg.fp16 else torch.float32,
            device_map="auto",
        )

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.cfg.lora_rank,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=list(self.cfg.lora_target_modules),
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)

        raw_ds = load_dataset("json", data_files=str(dataset_jsonl), split="train")

        def tokenize_pair(sample: Dict[str, str]) -> Dict[str, List[int]]:
            prompt_ids = tokenizer(sample["prompt"], add_special_tokens=True)["input_ids"]
            comp_ids = tokenizer(sample["completion"], add_special_tokens=False)["input_ids"] + [
                tokenizer.eos_token_id
            ]
            input_ids = (prompt_ids + comp_ids)[: self.cfg.cutoff_length]
            # Mask prompt tokens with -100 so Eq. (50) L_SFT computes loss strictly over y_t
            labels = ([-100] * len(prompt_ids) + comp_ids)[: self.cfg.cutoff_length]
            return {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            }

        tokenized = raw_ds.map(tokenize_pair, remove_columns=raw_ds.column_names)

        training_args = TrainingArguments(
            output_dir=str(out_dir),
            num_train_epochs=self.cfg.epochs,
            per_device_train_batch_size=self.cfg.batch_size,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            learning_rate=self.cfg.learning_rate,
            lr_scheduler_type=self.cfg.lr_scheduler_type,
            max_grad_norm=self.cfg.max_grad_norm,
            fp16=self.cfg.fp16 and torch.cuda.is_available(),
            logging_steps=10,
            save_strategy="epoch",
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        )
        trainer.train()
        model.save_pretrained(str(out_dir / "lora_adapter"))
        tokenizer.save_pretrained(str(out_dir / "lora_adapter"))
        return out_dir / "lora_adapter"
