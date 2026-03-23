# Code Repositories for 1-Person RLHF Research

This directory contains external code repositories relevant to our investigation
of whether training on a single person's preferences (1-person RLHF) produces
verbose/bland outputs compared to standard multi-annotator RLHF.

## Cloned Repositories

### 1. TRL (Transformers Reinforcement Learning) by HuggingFace

- **Path:** `./trl/`
- **Source:** https://github.com/huggingface/trl
- **License:** Apache-2.0
- **Purpose:** Primary framework for running RLHF/DPO experiments

TRL is the standard library for post-training foundation models. It provides
trainers for the full RLHF pipeline:

| Trainer | Purpose |
|---------|---------|
| `SFTTrainer` | Supervised fine-tuning |
| `RewardTrainer` | Train reward models from preference data |
| `DPOTrainer` | Direct Preference Optimization (no separate reward model needed) |
| `PPOTrainer` | Proximal Policy Optimization (classic RLHF) |
| `GRPOTrainer` | Group Relative Policy Optimization |
| `KTOTrainer` | Kahneman-Tversky Optimization |
| `RLOOTrainer` | REINFORCE Leave-One-Out |

**Why it matters for 1-person RLHF:** TRL's `DPOTrainer` and `RewardTrainer`
can be used directly with a single annotator's preference dataset. DPO is
especially convenient since it skips the reward model step entirely -- we can
train directly on one person's pairwise preferences.

Key paths:
- Trainers: `trl/trl/trainer/`
- Examples: `trl/examples/` (if available)

### 2. PAL (Pluralistic Alignment Framework)

- **Path:** `./pluralistic-alignment/`
- **Source:** https://github.com/RamyaLab/pluralistic-alignment
- **License:** Apache-2.0
- **Paper:** "PAL: Sample-Efficient Personalized Reward Modeling for Pluralistic Alignment" (ICLR 2025)
- **Purpose:** Reference implementation for per-user personalized reward models

PAL is directly relevant to our research question. It trains reward models that
maintain per-user preference embeddings, allowing the model to learn both shared
preferences across users AND individual-specific preferences. Key concepts:

- **User IDs in training data:** Each preference pair is tagged with a user_id
- **Preference groups (k):** The model learns k latent preference clusters
- **Few-shot new user adaptation:** Can adapt to a new user with as few as 2-5
  preference pairs
- **Converts to standard reward model:** Trained PAL models can be exported as
  standard scalar reward models for use in RLHF pipelines

**Why it matters for 1-person RLHF:** PAL provides a principled baseline for
understanding how single-user preferences relate to the population. With k=1
(single preference group), it approximates standard RLHF. With k>1, it can
capture preference diversity. This lets us compare 1-person training against
PAL's multi-user approach.

Key paths:
- Main training scripts: `pluralistic-alignment/main_pal_b.py`, `main_pal_b_fix_llm.py`
- Config files: `pluralistic-alignment/config/`
- Dataset handling: `pluralistic-alignment/dataset_factory.py`
- RM conversion: `pluralistic-alignment/integration/`

## Repositories Considered But Not Cloned

### OpenRLHF
- **Source:** https://github.com/OpenRLHF/OpenRLHF
- **Reason skipped:** Focused on large-scale distributed RLHF training (Ray + vLLM,
  70B+ models). Overkill for our single-annotator experiments where TRL suffices.
  Could revisit if we need to scale up.

### Federated-RLHF
- **Source:** https://github.com/flint-xf-fan/Federated-RLHF
- **Reason skipped:** Addresses privacy-preserving personalized RLHF via federated
  learning (AAMAS 2025). Interesting conceptually but the federated setting adds
  unnecessary complexity for our 1-person case.

### RLHFlow/RLHF-Reward-Modeling
- **Source:** https://github.com/RLHFlow/RLHF-Reward-Modeling
- **Reason skipped:** General reward model training recipes. TRL already covers
  this use case for us.

## Suggested Experiment Plan

1. **Baseline (TRL DPO):** Train with DPO on a single annotator's preferences
   from an existing dataset (e.g., filter Anthropic HH or OpenAssistant by
   annotator ID). Evaluate output verbosity, diversity, and quality.

2. **Multi-annotator baseline (TRL DPO):** Train on the same dataset but with
   all annotators mixed together. Compare outputs against the 1-person model.

3. **PAL comparison:** Use PAL to train a per-user reward model, then compare
   the reward landscape of a single user vs. the population average.
