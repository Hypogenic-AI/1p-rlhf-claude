# Resources Catalog

## Summary

This document catalogs all resources gathered for investigating whether 1-person RLHF still produces verbose, bland, or homogeneous outputs, and how much this is attributable to pretraining, SFT, or RLHF.

## Papers

Total papers downloaded: 24 unique papers (29 files including duplicates/versions)

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| InstructGPT | Ouyang et al. | 2022 | `2203.02155_instructgpt_...` | Foundational RLHF pipeline |
| Training Helpful & Harmless | Bai et al. | 2022 | `2204.05862_training_...` | HH-RLHF dataset, helpfulness-harmlessness tension |
| Learning to Summarize | Stiennon et al. | 2020 | `2009.01325_learning_...` | Early RLHF, 1/3 quality gap from length |
| Scaling Laws for RM Overoptimization | Gao et al. | 2023 | `2210.10760_scaling_laws_...` | RM degradation under optimization |
| Constitutional AI | Bai et al. | 2022 | `2212.08073_constitutional_ai...` | AI-feedback alternative |
| DPO | Rafailov et al. | 2023 | `2305.18290_dpo_...` | Direct preference optimization |
| RLAIF | Lee et al. | 2023 | `2309.00267_rlaif_...` | AI feedback scaling |
| RM Ensembles | Coste et al. | 2023 | `2310.02743_reward_model_ensembles...` | Ensembles reduce overoptimization |
| Constrained RLHF | Moskovitz et al. | 2023 | `2310.04373_confronting_...` | Multi-faceted RM constraints |
| Loose Lips Sink Ships | Shen et al. | 2023 | `2310.05199_loose_lips_...` | Length bias in RM, PoE mitigation |
| Verbosity Bias in Preference Labeling | Saito et al. | 2023 | `2310.10076v1_verbosity_bias_...` | Quantified verbosity bias in LLM+human judges |
| Understanding Sycophancy | Sharma et al. | 2024 | `2310.13548v4_towards_understanding_...` | Multi-stage sycophancy causes |
| MaxMin-RLHF | Chakraborty et al. | 2024 | `2402.08925_maxmin_rlhf_...` | Impossibility of single RM for diverse preferences |
| Directional Preference Alignment | Li et al. | 2024 | `2402.18571_directional_...` | Multi-objective user preferences |
| Heterogeneous RLHF | Park et al. | 2024 | `2405.00254_rlhf_heterogeneous_...` | Personalization vs. aggregation framework |
| Preference Collapse | Li et al. | 2024 | `2405.16455_algorithmic_bias_...` | KL regularization causes mode collapse |
| Interpretable Preferences | Wang et al. | 2024 | `2406.12845_interpretable_...` | Multi-objective reward modeling |
| Reward Shaping | 2025 | 2025 | `2502.18770_reward_shaping_...` | Mitigating reward hacking |
| Shared LoRA Personalized RLHF | 2025 | 2025 | `2503.19201_shared_lora_...` | Per-user LoRA reward models |
| Survey: Personalized Alignment | Xie et al. | 2025 | `2504.07070_survey_...` | Comprehensive taxonomy |
| Bias Fitting for Length Bias | 2025 | 2025 | `2505.12843_bias_fitting_...` | Alternative length bias mitigation |
| RM Evaluation & Overoptimization | 2025 | 2025 | `2505.12763_rethinking_...` | RM evaluation methodology |
| Iterated RLHF Overoptimization | 2025 | 2025 | `2505.18126_reward_model_...` | Overoptimization in iterated RLHF |
| Causal Length Bias | 2025 | 2025 | `2511.12573_mitigating_...` | Causal framework for length bias |
| Swap-Guided Personalized RLHF | 2026 | 2026 | `2603.12595_swap_guided_...` | Posterior collapse in personalized RLHF |

See `papers/` directory for all PDFs.

## Datasets

Total datasets downloaded: 5 (7 subdirectories, ~424MB on disk)

| Name | Source | Size | Task | Location | Per-Annotator? |
|------|--------|------|------|----------|----------------|
| Anthropic HH-RLHF | HuggingFace | 169K pairs | Helpfulness/Harmlessness | `datasets/anthropic-hh-rlhf/` | No |
| PRISM | HuggingFace | 68K utterances, 1,396 users | Multi-model rating | `datasets/prism-*/` | **Yes** |
| PersonalLLM | HuggingFace | 10K prompts, 10 simulated annotators | Personalized alignment | `datasets/personal-llm/` | Yes (simulated) |
| OpenAI TL;DR | HuggingFace | 92K comparisons | Summarization | `datasets/openai-summarize-comparisons/` | No (in this version) |
| SHP (50K sample) | HuggingFace | 50K preferences | Reddit Q&A | `datasets/shp-sample-50k/` | No |

See `datasets/README.md` for detailed schemas and download instructions.

## Code Repositories

Total repositories cloned: 2

| Name | URL | Purpose | Location |
|------|-----|---------|----------|
| TRL | github.com/huggingface/trl | RLHF/DPO training framework | `code/trl/` |
| PAL | github.com/RamyaLab/pluralistic-alignment | Per-user personalized reward models | `code/pluralistic-alignment/` |

See `code/README.md` for detailed descriptions and experiment plan.

## Resource Gathering Notes

### Search Strategy
1. **Paper search:** Used arxiv API with 7 targeted queries covering RLHF length bias, personalized RLHF, reward hacking, sycophancy, DPO, and reward overoptimization. 82 unique papers found, 24 most relevant downloaded.
2. **Dataset search:** Identified datasets from literature review (especially the personalized alignment survey). Focused on datasets with per-annotator metadata for 1-person experiments.
3. **Code search:** Searched GitHub for RLHF training frameworks and personalized alignment implementations.

### Selection Criteria
- Papers: Prioritized (a) direct relevance to verbosity/blandness/sycophancy in RLHF, (b) personalized/individual preference modeling, (c) foundational RLHF methodology, (d) recency (2023-2026)
- Datasets: Prioritized per-annotator metadata availability (PRISM is the only real-human dataset with user IDs)
- Code: Selected mature, well-documented frameworks that support our specific experiment needs

### Challenges Encountered
- Two arxiv PDFs (2307.11760, 2310.12036) served wrong papers for their IDs; re-downloaded with correct versions
- Paper-finder service was unavailable; used direct arxiv API search as fallback
- Most preference datasets lack per-annotator metadata, limiting 1-person experiments to PRISM and PersonalLLM

### Gaps and Workarounds
- **No existing 1-person RLHF experiments:** This is a genuine research gap. Our experiments will be novel.
- **Limited per-annotator datasets:** PRISM is the only large real-human dataset with user IDs. PersonalLLM provides controlled simulated annotators as supplement.
- **OpenAI TL;DR worker IDs:** The HuggingFace version strips worker IDs. Raw data from OpenAI's blob storage preserves them if needed.

## Recommendations for Experiment Design

### Primary Dataset
**PRISM** -- 1,396 real users with individual scores and preferences. Train DPO/reward models on individual users and compare to aggregate.

### Baseline Methods
1. **SFT-only:** Isolates pretraining+SFT contribution to verbosity/blandness
2. **DPO on all annotators:** Standard multi-annotator RLHF baseline
3. **DPO on single annotator:** The 1-person RLHF condition
4. **DPO on size-matched random sample:** Controls for data quantity vs. annotator identity

### Evaluation Metrics
1. **Output length** (tokens): Primary verbosity measure
2. **Lexical diversity** (distinct-n, self-BLEU): Measures output homogeneity
3. **Sycophancy rate**: How often model agrees with factually incorrect user statements
4. **Win rate vs. SFT**: Quality check via LLM judge
5. **Inter-model similarity**: Compare outputs across different 1-person models to test "all AI sounds the same"

### Code to Adapt/Reuse
- **TRL's DPOTrainer:** Primary training framework. Filter PRISM data to individual users and train per-user models.
- **PAL:** Reference for per-user reward model training. Compare PAL's multi-user approach vs. our single-user DPO.

### Suggested Experiment Pipeline
1. Select 10-20 PRISM users with sufficient preference data (50+ comparisons each)
2. For each user: train DPO model, train aggregate DPO model, train SFT-only baseline
3. Generate outputs on held-out prompts from all models
4. Measure verbosity, diversity, sycophancy, quality across all conditions
5. Ablate: base model → SFT → DPO to isolate each stage's contribution
