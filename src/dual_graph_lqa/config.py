"""Configuration dataclasses aligned with Tables 9 and 10 of the paper."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class AntiLeakageConfig:
    ngram_size: int = 3
    jaccard_threshold: float = 0.80
    cosine_threshold: float = 0.85


@dataclass
class DataConfig:
    statutes_path: str = "data/sample/statutes.json"
    cases_path: str = "data/sample/cases.json"
    qa_train_path: str = "data/sample/qa_train.json"
    synonyms_path: str = "data/sample/synonyms.json"
    faq_eval_path: str = "data/sample/faq_eval_sample.json"
    output_dir: str = "outputs"
    train_val_ratio: float = 0.8
    anti_leakage: AntiLeakageConfig = field(default_factory=AntiLeakageConfig)


@dataclass
class BasicKGConfig:
    domains: List[str] = field(default_factory=lambda: ["labor", "insurance", "financial"])
    use_platt_scaling: bool = True
    svm_c: float = 1.0
    min_confidence: float = 0.35


@dataclass
class AppliedKGConfig:
    bm25_top_keywords: int = 8
    use_jieba: bool = True
    use_synonym_expansion: bool = True
    enforce_statutory_prefix: bool = True


@dataclass
class ContrastiveConfig:
    encoder_model: str = "hfl/chinese-roberta-wwm-ext"
    embedding_dim: int = 256
    gcn_hidden_dim: int = 256
    gcn_num_layers: int = 2
    gcn_dropout: float = 0.1
    temperature: float = 0.07
    lambda_res: float = 0.01
    lambda_smooth: float = 0.01
    lambda_align: float = 0.05
    lambda_head: float = 0.001
    learning_rate: float = 1.0e-3
    epochs: int = 25
    batch_size: int = 16
    freeze_base_gcn: bool = True


@dataclass
class ConstraintsConfig:
    enable_coverage: bool = True
    delta_cov: float = 0.55
    enable_consistency: bool = True
    epsilon: float = 0.45
    enable_redundancy: bool = True
    delta_ent: float = 0.35


@dataclass
class RetrievalConfig:
    lambda_one_hop: float = 1.0
    lambda_multi_hop: float = 1.0
    lambda_contrastive: float = 1.0
    hop_decay: float = 0.8
    max_hops: int = 2
    uncertainty_top_m: int = 20
    uncertainty_temp: float = 0.1
    uncertainty_threshold: float = 0.65
    top_k_cap: int = 20
    fixed_threshold_fallback: float = 0.45
    constraints: ConstraintsConfig = field(default_factory=ConstraintsConfig)


@dataclass
class LLMSFTConfig:
    base_model: str = "taide/Llama3-TAIDE-LX-8B-Chat-Alpha1"
    cutoff_length: int = 4096
    learning_rate: float = 5.0e-5
    epochs: int = 40
    batch_size: int = 4
    gradient_accumulation_steps: int = 64
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    fp16: bool = True
    max_new_tokens: int = 512
    temperature: float = 0.2


@dataclass
class EvaluationConfig:
    alpha_answerability: float = 0.30
    beta_article: float = 0.40
    gamma_semantic: float = 0.30
    excluded_lawbench_tasks: List[str] = field(
        default_factory=lambda: ["2-3", "3-4", "3-5", "3-7"]
    )


@dataclass
class PipelineConfig:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    basic_kg: BasicKGConfig = field(default_factory=BasicKGConfig)
    applied_kg: AppliedKGConfig = field(default_factory=AppliedKGConfig)
    contrastive: ContrastiveConfig = field(default_factory=ContrastiveConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm_sft: LLMSFTConfig = field(default_factory=LLMSFTConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def _merge_dict(dc_cls: type, raw: Dict[str, Any] | None) -> Any:
    if not raw:
        return dc_cls()
    valid = {k: v for k, v in raw.items() if k in dc_cls.__dataclass_fields__}
    return dc_cls(**valid)


def load_config(path: str | Path) -> PipelineConfig:
    """Load YAML configuration into typed PipelineConfig."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    data_raw = dict(raw.get("data", {}))
    al_raw = data_raw.pop("anti_leakage", None)
    data_cfg = _merge_dict(DataConfig, data_raw)
    data_cfg.anti_leakage = _merge_dict(AntiLeakageConfig, al_raw)

    bkg_raw = dict(raw.get("basic_kg", {}))
    router_raw = bkg_raw.pop("domain_router", {}) or {}
    bkg_raw.update(router_raw)
    bkg_cfg = _merge_dict(BasicKGConfig, bkg_raw)

    ret_raw = dict(raw.get("retrieval", {}))
    cons_raw = ret_raw.pop("constraints", None)
    ret_cfg = _merge_dict(RetrievalConfig, ret_raw)
    ret_cfg.constraints = _merge_dict(ConstraintsConfig, cons_raw)

    return PipelineConfig(
        seed=int(raw.get("seed", 42)),
        data=data_cfg,
        basic_kg=bkg_cfg,
        applied_kg=_merge_dict(AppliedKGConfig, raw.get("applied_kg")),
        contrastive=_merge_dict(ContrastiveConfig, raw.get("contrastive")),
        retrieval=ret_cfg,
        llm_sft=_merge_dict(LLMSFTConfig, raw.get("llm_sft")),
        evaluation=_merge_dict(EvaluationConfig, raw.get("evaluation")),
    )
