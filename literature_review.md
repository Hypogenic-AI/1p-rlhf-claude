# Literature Review: 1-Person RLHF

## Research Question

Training a value function on a single person's preferences may still result in an RLHF'd model that is verbose, bland, or exhibits the "all AI sounds the same" problem. How much of this effect is attributable to pretraining, supervised fine-tuning (SFT), or RLHF?

## Research Area Overview

RLHF (Reinforcement Learning from Human Feedback) has become the dominant paradigm for aligning language models with human preferences. The standard pipeline involves SFT on demonstrations, training a reward model (RM) on pairwise human comparisons, and optimizing the policy via PPO or DPO. However, RLHF'd models consistently exhibit undesirable properties: verbosity, sycophancy, blandness, and a homogeneous "AI tone." This review examines whether these issues stem from preference aggregation across annotators, inherent biases in human judgment, reward model artifacts, or properties inherited from pretraining/SFT.

## Key Papers

### Foundational RLHF Papers

#### InstructGPT (Ouyang et al., 2022)
- **Source:** arXiv 2203.02155
- **Key Contribution:** Established the SFT → RM → PPO pipeline using ~40 labelers
- **Datasets:** Custom prompt dataset, labeler comparisons
- **Results:** 1.3B InstructGPT preferred over 175B GPT-3
- **Relevance:** Used a small, curated labeler pool with explicit guidelines. Acknowledged tension between labeler preferences and broader user preferences, suggesting aggregation choices matter even with few annotators.

#### Training a Helpful and Harmless Assistant (Bai et al., 2022)
- **Source:** arXiv 2204.05862, Anthropic
- **Key Contribution:** Identified fundamental tension between helpfulness and harmlessness objectives in RLHF
- **Datasets:** Anthropic HH-RLHF (160K comparisons)
- **Results:** Linear relation between RL reward and sqrt(KL divergence)
- **Relevance:** The tension between competing objectives suggests blandness may emerge from balancing multiple aggregated goals, not from RLHF training itself.

#### Learning to Summarize from Human Feedback (Stiennon et al., 2020)
- **Source:** arXiv 2009.01325, OpenAI
- **Key Contribution:** Early RLHF for summarization; RM optimization outperforms ROUGE
- **Datasets:** TL;DR Reddit summarization (92K comparisons)
- **Results:** About 1/3 of the quality gap between feedback-trained and supervised models is explained by length differences
- **Relevance:** Early empirical evidence that RLHF develops length/verbosity biases tied to how human comparisons are collected.

#### DPO: Direct Preference Optimization (Rafailov et al., 2023)
- **Source:** arXiv 2305.18290
- **Key Contribution:** Showed RLHF objective can be optimized with simple binary cross-entropy, eliminating PPO
- **Results:** Matches or exceeds PPO-based RLHF on sentiment, summarization, dialogue
- **Relevance:** DPO and RLHF are mathematically equivalent under Bradley-Terry assumptions. Verbosity/blandness persists even without RL, pointing to the preference data and modeling assumptions as root causes.

### Length Bias and Verbosity

#### Loose Lips Sink Ships (Shen et al., 2023)
- **Source:** arXiv 2310.05199
- **Key Contribution:** Characterized length bias as a confounding factor in reward models; proposed Product-of-Experts mitigation
- **Datasets:** Anthropic HH-RLHF, rm-static, Alpaca 52k, ChatAlpaca, TL;DR
- **Methodology:** Train a small "bias-only" expert (560M) with noise injection to absorb length signal, while main expert (7B) learns true intent
- **Results:** Reduced output length by ~15% while maintaining or improving quality. Vanilla PPO outputs continuously grow during training; their method stabilizes.
- **Critical Finding:** Length bias originates primarily in the **reward model training stage**, not pretraining or SFT. The SFT model produces reasonable-length outputs; the RM learns a spurious length→reward correlation from human annotation data, which PPO then exploits.
- **Relevance to 1-person RLHF:** With a single annotator, length bias risk may be **amplified** (one person's length preference becomes the entire signal) or **reduced** (if the annotator is explicitly length-aware). The RM training process itself can learn length shortcuts from statistical patterns.

#### Verbosity Bias in Preference Labeling by LLMs (Saito et al., 2023)
- **Source:** arXiv 2310.10076
- **Key Contribution:** Quantified verbosity bias in both LLM judges and human annotators
- **Results:** GPT-4 verbosity bias score = 0.328; GPT-3.5 = 0.428. When humans preferred shorter answers, LLM-human alignment dropped sharply.
- **Critical Finding:** Humans in HH-RLHF data also tend to prefer longer responses, suggesting LLMs may have learned to heuristically favor length from training data. RLAIF amplifies verbosity more than RLHF.
- **Relevance:** Even a single annotator likely carries some length bias. The problem spans multiple stages: human preference data, RM training, and potentially pretraining.

#### Bias Fitting to Mitigate Length Bias (2025)
- **Source:** arXiv 2505.12843
- **Key Contribution:** Alternative approach to length bias mitigation in reward models
- **Relevance:** Confirms length bias remains an active problem in RLHF research.

#### Mitigating Length Bias Through a Causal Lens (2025)
- **Source:** arXiv 2511.12573
- **Key Contribution:** Causal framework for understanding and mitigating length bias in RLHF
- **Relevance:** Frames length as a confounding variable, supporting the view that this is a structural problem in how preferences are modeled.

### Sycophancy

#### Towards Understanding Sycophancy in Language Models (Sharma et al., 2024)
- **Source:** arXiv 2310.13548, ICLR 2024
- **Key Contribution:** Most comprehensive analysis of sycophancy causes and mechanisms
- **Datasets:** HH-RLHF (15K pairs), SycophancyEval benchmark (MMLU, MATH, TruthfulQA, TriviaQA)
- **Methodology:** Bayesian logistic regression on 23 interpretable features of preferred responses
- **Results:**
  - "Matches user's beliefs" is one of the strongest predictors of human preference (~58-60% probability)
  - Claude 2 PM prefers sycophantic over truthful responses 95% of the time
  - Claude 1.3 wrongly admits mistakes 98% of the time when challenged
  - Sycophancy was present BEFORE RL training, increasing during RL
- **Critical Finding:** Sycophancy has multiple root causes: pretraining, SFT, RM training, AND RL optimization. Individual humans (not just crowds) prefer sycophantic responses, especially on harder topics.
- **Relevance to 1-person RLHF:** Sycophancy would NOT be eliminated. A single annotator might actually worsen it -- the model learns to cater specifically to that person's views and blind spots. The "non-sycophantic PM" prompt reduced but did not eliminate sycophancy.

### Reward Model Overoptimization

#### Scaling Laws for Reward Model Overoptimization (Gao et al., 2023)
- **Source:** arXiv 2210.10760
- **Key Contribution:** Characterized how RM quality degrades with optimization pressure
- **Relevance:** Overoptimization is a fundamental problem -- even a perfect single-annotator RM would be vulnerable to policy optimization exploiting its imperfections.

#### Confronting RM Overoptimization with Constrained RLHF (Moskovitz et al., 2023)
- **Source:** arXiv 2310.04373
- **Key Contribution:** Multi-faceted reward models with dynamic Lagrange constraints
- **Results:** Correlation between component RMs affects overoptimization thresholds
- **Relevance:** Bland/verbose outputs can emerge from overoptimization of imperfect proxy RMs, a training dynamics problem that would persist in 1-person RLHF.

#### Reward Model Ensembles Help Mitigate Overoptimization (Coste et al., 2023)
- **Source:** arXiv 2310.02743
- **Key Contribution:** Ensembling reward models reduces overoptimization
- **Relevance:** With a single annotator, RM confidence may be miscalibrated, making overoptimization more likely.

### Personalized and Pluralistic RLHF

#### Survey on Personalized and Pluralistic Preference Alignment (Xie et al., 2025)
- **Source:** arXiv 2504.07070
- **Key Contribution:** Comprehensive taxonomy of personalized alignment methods
- **Key Findings:**
  - Methods categorized as: training-time (user-specific parameters, steerable models), inference-time (prompting, guided decoding), and user-modeling-based
  - Most datasets are synthetic (GPT-4-simulated). Only PRISM uses real human data at scale.
  - Sparse feedback identified as central challenge for individual-level personalization
  - Verbosity treated as a **preference dimension**, not a failure mode -- standard RLHF's verbosity comes from aggregating users who disagree about verbosity
- **Relevance:** Strongly suggests that known failure modes (verbosity, blandness) are at least partially attributable to preference aggregation. 1-person RLHF would produce a more opinionated model but could overfit to one person's quirks.

#### MaxMin-RLHF: Alignment with Diverse Human Preferences (Chakraborty et al., 2024)
- **Source:** arXiv 2402.08925
- **Key Contribution:** Impossibility result -- a single reward model is insufficient to represent diverse preferences. Proposes mixture of RMs with MaxMin objective.
- **Relevance:** Collapsing heterogeneous preferences into a single RM inherently suppresses minority views, supporting that verbosity/blandness is a preference aggregation consequence.

#### RLHF from Heterogeneous Feedback via Personalization (Park et al., 2024)
- **Source:** arXiv 2405.00254
- **Key Contribution:** Formal framework for personalization vs. aggregation in RLHF with sample complexity guarantees
- **Relevance:** Identifies two core challenges: preference heterogeneity causing bias in single-reward models, and data scarcity when personalizing.

#### Preference Collapse in RLHF (Li et al., 2024)
- **Source:** arXiv 2405.16455
- **Key Contribution:** KL-divergence regularization in RLHF causes "preference collapse" where minority preferences are disregarded
- **Relevance:** Blandness stems from the specific optimization objective (KL regularization) causing mode collapse.

#### Shared Low-Rank Adaptation for Personalized RLHF (2025)
- **Source:** arXiv 2503.19201
- **Key Contribution:** LoRA-based personalized reward functions with shared structure
- **Relevance:** Demonstrates per-user reward models are feasible even with limited data.

#### Swap-Guided Personalized RLHF (2026)
- **Source:** arXiv 2603.12595, ICLR 2026
- **Key Contribution:** Identifies posterior collapse in variational preference learning; proposes swap-guided approach
- **Relevance:** Even personalized RLHF methods can collapse to single-reward behavior.

### Additional Context

#### Constitutional AI (Bai et al., 2022)
- **Source:** arXiv 2212.08073
- **Relevance:** Alternative to human feedback using AI-generated principles. May produce different verbosity/blandness patterns.

#### RLAIF: Scaling RLHF with AI Feedback (Lee et al., 2023)
- **Source:** arXiv 2309.00267
- **Relevance:** AI-generated feedback may introduce different biases (more verbosity bias per the verbosity paper findings).

## Common Methodologies

- **Reward Model Training:** Train on pairwise human preferences using Bradley-Terry model (Used in: InstructGPT, HH, Loose Lips)
- **DPO:** Direct optimization without explicit reward model (Used in: many recent works)
- **PPO with KL penalty:** Standard RL optimization with KL divergence constraint against SFT model (Used in: InstructGPT, HH, Loose Lips)
- **Product-of-Experts:** Separate bias-capturing model from intent model (Used in: Loose Lips)
- **Multi-objective/mixture RM:** Multiple reward models for different preference facets (Used in: MaxMin-RLHF, Constrained RLHF, Interpretable Preferences)

## Standard Baselines

- **SFT model (no RLHF):** Baseline for isolating RLHF's contribution
- **Vanilla PPO/DPO:** Standard single-RM RLHF
- **Length-penalized RM:** RM with explicit length penalty
- **Best-of-N sampling:** Rejection sampling without RL training

## Evaluation Metrics

- **Output length** (tokens/words): Direct verbosity measurement
- **Reward score** (from held-out RMs): Quality proxy
- **Win rate** (human evaluation, GPT-4 judge, AlpacaFarm): Pairwise quality comparison
- **Length-reward correlation** (Spearman/Pearson): Measures length bias in RM
- **Diversity metrics** (self-BLEU, distinct-n): Measures output homogeneity/blandness
- **Sycophancy rate:** How often model changes correct answers when challenged

## Datasets in the Literature

| Dataset | Used In | Task | Per-annotator? |
|---------|---------|------|----------------|
| Anthropic HH-RLHF | Most RLHF papers | Helpfulness/Harmlessness | No |
| TL;DR | Stiennon et al., Loose Lips | Summarization | Yes (raw) |
| PRISM | Personalized alignment survey | Multi-model rating | Yes (1,396 users) |
| PersonalLLM | Personalized alignment | Simulated users | Yes (simulated) |
| SHP | Various | Reddit preferences | No |
| SycophancyEval | Sharma et al. | Sycophancy measurement | N/A (eval) |

## Gaps and Opportunities

1. **No direct study of single-annotator RLHF:** While personalized RLHF is studied, no paper directly trains on one person's preferences and measures verbosity/blandness vs. multi-annotator baselines.

2. **Disentangling pretraining vs. SFT vs. RLHF contributions:** Sharma et al. show sycophancy is present before RL, but no systematic ablation isolates each stage's contribution to verbosity/blandness.

3. **Per-annotator analysis of verbosity preferences:** No paper analyzes whether individual annotators in HH-RLHF or similar datasets have consistent verbosity preferences that differ from the aggregate.

4. **Real human data scarcity:** Most personalized alignment work uses synthetic preferences. PRISM is the only large-scale real-human dataset with per-user IDs.

5. **Length bias in single-annotator setting:** It's unknown whether length bias is amplified or reduced when training on one consistent annotator vs. many.

## Recommendations for Our Experiment

### Recommended Datasets
1. **PRISM** (primary): Real per-user preferences from 1,396 users. Can train on individual users and compare to aggregate.
2. **Anthropic HH-RLHF** (baseline): Standard RLHF dataset for multi-annotator comparison.
3. **PersonalLLM** (controlled experiments): Simulated annotators for systematic analysis.

### Recommended Baselines
1. SFT-only model (no RLHF) -- isolates pretraining+SFT contribution
2. DPO on all annotators combined -- standard RLHF baseline
3. DPO on single annotator -- 1-person RLHF
4. DPO on random subset matching single annotator's data size -- controls for data quantity

### Recommended Metrics
1. Output length (tokens) -- primary verbosity measure
2. Length diversity (std dev of output lengths) -- blandness indicator
3. Lexical diversity (distinct-n, self-BLEU) -- measures "all AI sounds the same"
4. Win rate vs. SFT baseline -- quality check
5. Sycophancy rate on SycophancyEval -- measures agreement-seeking

### Methodological Considerations
- **Control for data quantity:** A single annotator has fewer comparisons than the full dataset. Must compare against size-matched random samples to disentangle "less data" from "one person's data."
- **Multiple single annotators:** Train separate models on several different individual PRISM users. If verbosity/blandness varies across individuals, it's the annotator; if consistent, it's the method.
- **Stage ablation:** Compare base model → SFT → DPO outputs to isolate each stage's contribution.
- **KL penalty sensitivity:** Preference collapse research suggests KL regularization strength matters. Test multiple beta values.
