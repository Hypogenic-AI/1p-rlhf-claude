# Datasets for 1-Person RLHF Research

Downloaded and saved in HuggingFace `datasets` Arrow format. Load with:
```python
from datasets import load_from_disk
ds = load_from_disk("datasets/<name>")
```

## Dataset Inventory

### 1. Anthropic HH-RLHF (`anthropic-hh-rlhf/`)
- **Source:** `Anthropic/hh-rlhf`
- **Size:** 160,800 train / 8,552 test (~185MB on disk)
- **Schema:** `chosen`, `rejected` (full conversation strings with `\n\nHuman:` / `\n\nAssistant:` turns)
- **Per-annotator info:** No. Aggregated preferences only.
- **Use case:** Standard RLHF baseline. Helpful for training reward models and DPO, but cannot isolate individual annotator effects.

### 2. PRISM (`prism-conversations/`, `prism-utterances/`, `prism-survey/`)
- **Source:** `HannahRoseKirk/prism-alignment`
- **Size:**
  - Conversations: 8,011 rows (conversation-level, ~26MB)
  - Utterances: 68,371 rows (turn-level with scores, ~25MB)
  - Survey: 1,500 rows (user profiles, ~1MB)
- **Schema (utterances):** `utterance_id`, `user_id`, `conversation_id`, `user_prompt`, `model_response`, `model_name`, `score`, `if_chosen`, etc.
- **Schema (survey):** `user_id`, `self_description`, `stated_prefs` (values, creativity, fluency, factuality, diversity, safety, helpfulness scores), demographics
- **Per-annotator info:** YES -- 1,396 unique users with `user_id` on every utterance. Each user rated multiple model responses with numeric scores and chose preferred responses.
- **Use case:** CRITICAL for 1-person experiments. Can isolate individual user preferences, train per-user reward models, and compare single-user vs. aggregated alignment. User profiles in survey data enable analysis of preference diversity.

### 3. PersonalLLM (`personal-llm/`)
- **Source:** `namkoong-lab/PersonalLLM`
- **Size:** 9,402 train / 1,000 test (~92MB on disk)
- **Schema:** `prompt`, `response_1` through `response_8` (from different models), plus scores from 10 different simulated reward models (e.g., `response_1_gemma_2b`, `response_1_mistral_raft`, etc.)
- **Per-annotator info:** YES (simulated). Each "annotator" is a different reward model with distinct preferences. Scores from 10 reward models per response.
- **Use case:** Benchmark for personalized alignment. Simulated users allow controlled experiments on how single-user training differs from population-level training.

### 4. OpenAI TL;DR Summarization (`openai-summarize-comparisons/`)
- **Source:** `CarperAI/openai_summarize_comparisons` (reformatted from OpenAI's "Learning to Summarize from Human Feedback")
- **Size:** 92,534 train / 83,629 test / 33,082 valid1 / 50,715 valid2 (~74MB on disk)
- **Schema:** `prompt`, `chosen`, `rejected`
- **Per-annotator info:** No. The original OpenAI dataset has `worker` IDs in the raw JSON, but this HF version strips them. The raw data at `openaipublic.blob.core.windows.net/summarize-from-feedback/` preserves worker IDs if needed.
- **Use case:** Summarization-specific preference data. Good for domain-specific reward modeling.

### 5. Stanford Human Preferences - SHP (`shp-sample-50k/`)
- **Source:** `stanfordnlp/SHP` (50K sample from 348K train)
- **Size:** 50,000 rows (~22MB on disk)
- **Schema:** `history` (prompt), `human_ref_A`, `human_ref_B`, `score_A`, `score_B`, `labels` (which is preferred), `domain` (subreddit), `score_ratio`, `seconds_difference`
- **Per-annotator info:** No. Preferences derived from Reddit upvote counts (crowd signal, not individual annotators).
- **Use case:** Large-scale naturally occurring preferences. Domain labels (subreddits) can serve as a proxy for user-type segmentation.

## Per-Annotator Summary

| Dataset | Per-Annotator IDs? | Annotator Count | Best for 1-Person RLHF? |
|---------|-------------------|-----------------|--------------------------|
| PRISM | YES (real users) | 1,396 | **PRIMARY** -- real humans with profiles |
| PersonalLLM | YES (simulated) | 10 reward models | **SECONDARY** -- controlled experiments |
| Anthropic HH-RLHF | No | Unknown | Baseline comparison only |
| OpenAI TL;DR | No (in this version) | ~80 workers in raw data | Need raw data for per-worker |
| SHP | No | Crowd (Reddit) | Domain segmentation only |

## Regenerating Datasets

All datasets can be re-downloaded from HuggingFace Hub:
```bash
source .venv/bin/activate
python3 -c "
from datasets import load_dataset
# Example: re-download PRISM
ds = load_dataset('HannahRoseKirk/prism-alignment', 'utterances')
ds.save_to_disk('datasets/prism-utterances')
"
```
