# Data Preparation & Anti-Leakage Guide

## 1. Included Runnable Sample Corpus (`data/sample/`)

To support direct reproducibility without access to confidential corporate records, `data/sample/` provides a self-contained multi-domain legal sample corpus covering labor, insurance, and financial law:

- `statutes.json`: Structured statutory provisions ($D^{\text{law}}$) for Basic KG construction.
- `cases.json`: Judicial decisions ($D^{\text{case}}$) citing articles in `statutes.json`.
- `qa_train.json`: Historical QA pairs ($D^{\text{QA}}$) with positive statutory citations for Applied KG construction and contrastive alignment.
- `synonyms.json`: Legal synonym expansion lexicon $S(\cdot)$ mapping everyday phrasing to statutory concepts.
- `faq_eval_sample.json`: Held-out evaluation questions with gold article citations and answerability labels.

## 2. Full Paper Datasets

| Dataset | Jurisdiction | Version / Collection | Access |
| :--- | :--- | :--- | :--- |
| **Statutory Regulations ($D^{\text{law}}$)** | Taiwan (R.O.C.) & PRC ref. | Effective June 2024 | Public: [MOJ Laws & Regulations Database](https://law.moj.gov.tw) & [NPC FLK](https://flk.npc.gov.cn) |
| **Judicial Decisions ($D^{\text{case}}$)** | Taiwan (R.O.C.) & PRC ref. | Through June 2024 | Public: Judicial Yuan Retrieving System |
| **Corporate QA Pairs ($D^{\text{QA}}$)** | Taiwan (R.O.C.) | 2022–June 2024 | Confidential under NDA (80% train / 20% val; 0% test) |
| **LawBench 16-Task Subset** | Mainland China (PRC) | LawBench v1.0 (2024) | Public: [open-compass/LawBench](https://github.com/open-compass/LawBench/) (excludes Tasks 2-3, 3-4, 3-5, 3-7) |
| **Real-World MOJ FAQ Set ($N=1{,}000$)** | Taiwan (R.O.C.) | Through June 2024 | Public FAQ portal of [law.moj.gov.tw](https://law.moj.gov.tw) (100% held-out test) |

## 3. Four-Stage Anti-Leakage Verification (`src/dual_graph_lqa/data/anti_leakage.py`)

Before building the Applied KG or fine-tuning, run `AntiLeakageFilter` to enforce:
1. **S1**: Physical source separation (`train_source != eval_source`).
2. **S2**: Unicode normalization with exact string and character 3-gram Jaccard filtering ($\ge 0.80$).
3. **S3**: Dense embedding cosine-similarity paraphrase filtering ($\ge 0.85$).
4. **S4**: Answer-triple isolation ensuring that no evaluation reference answers or answer-derived triples enter $G^{\text{applied}}$, $V_i^+$, or $L_{\text{SFT}}$.
