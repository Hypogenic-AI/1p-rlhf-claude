# Research Plan: 1-Person RLHF

## Motivation & Novelty Assessment

### Why This Research Matters
RLHF-trained models consistently exhibit verbosity, blandness, and a homogeneous "AI tone" — the "all AI sounds the same" problem. This is widely attributed to RLHF, but the pipeline has three stages (pretraining → SFT → RLHF), and recent work (Sharma et al. 2024, Shen et al. 2023) shows some of these behaviors exist *before* RLHF. Meanwhile, standard RLHF aggregates preferences from many annotators, which provably loses minority preferences (Chakraborty et al. 2024). The question of whether training on a *single person's* preferences avoids these failure modes — or reveals that they're inherent to the method itself — has direct implications for personalized AI alignment.

### Gap in Existing Work
Based on the literature review:
1. **No direct study of single-annotator RLHF** exists — personalized RLHF papers focus on methods, not on diagnosing verbosity/blandness
2. **No systematic ablation** isolates pretraining vs SFT vs RLHF contributions to verbosity across individual annotators
3. **No per-annotator analysis** of whether individual humans in RLHF datasets have consistent (or divergent) verbosity preferences

### Our Novel Contribution
We conduct the first empirical study that:
1. Analyzes whether individual annotators systematically differ in verbosity preferences (using PRISM dataset)
2. Trains DPO models on individual simulated annotators vs. aggregated preferences (using PersonalLLM)
3. Measures how verbosity, lexical diversity, and output homogeneity change at each pipeline stage
4. Disentangles "preference aggregation" effects from "RLHF method" effects

### Experiment Justification
- **Experiment 1 (PRISM Preference Analysis):** Directly tests whether individual users prefer different verbosity levels or whether length bias is universal across humans. If individual users have diverse length preferences but aggregation converges to "prefer longer," this implicates aggregation.
- **Experiment 2 (DPO Training):** Tests whether 1-person RLHF produces less verbose/bland outputs than aggregate RLHF. Uses PersonalLLM (10 simulated annotators, 9400 prompts) for sufficient per-annotator data.
- **Experiment 3 (Stage Ablation):** Compares base model → DPO outputs to isolate RLHF's incremental contribution to verbosity/blandness.
- **Experiment 4 (LLM Judge):** Uses GPT-4.1 to evaluate quality, detecting whether 1-person models sacrifice quality for personality.

## Research Question
Does training a language model on a single person's preferences (1-person RLHF) still produce verbose, bland, or homogeneous outputs? How much of the "all AI sounds the same" problem is attributable to pretraining, SFT, or RLHF?

## Hypothesis Decomposition
- **H1:** Individual annotators in PRISM have significantly different verbosity preferences (measured by chosen-vs-rejected length ratios). Aggregation compresses this variation.
- **H2:** DPO trained on a single annotator produces outputs with different verbosity levels than DPO trained on aggregated annotators.
- **H3:** Models trained on different individual annotators produce *distinguishable* outputs (lower inter-model similarity), while aggregate-trained models converge to a single style.
- **H4:** Some verbosity is already present in the base/SFT model before any preference training, suggesting pretraining/SFT contribution.

## Proposed Methodology

### Approach
We use two complementary datasets:
- **PRISM** (1,396 real users, ~19 pairs/user): For analyzing real human preference patterns
- **PersonalLLM** (10 simulated annotators, 9,400 prompts × 8 responses): For DPO training with sufficient per-annotator data

We train a small model (Qwen2.5-1.5B-Instruct) with DPO under different conditions and measure outputs.

### Experimental Steps

#### Experiment 1: PRISM Per-User Preference Analysis
1. For each PRISM user, compute: mean chosen length, mean rejected length, length preference ratio
2. Compute length-score correlation per user
3. Statistical test: Do individual users differ significantly in length preference? (Kruskal-Wallis)
4. Visualize distribution of per-user length preferences

#### Experiment 2: DPO Training (PersonalLLM)
1. Convert PersonalLLM reward scores to DPO pairs (for each prompt, use highest-scoring and lowest-scoring responses per annotator)
2. Train DPO on:
   a. Individual annotators (10 models)
   b. All annotators combined (1 aggregate model)
   c. Size-matched random subsets (10 control models)
3. Use Qwen2.5-1.5B-Instruct as base model with LoRA
4. Generate responses on held-out test prompts

#### Experiment 3: Stage Ablation
1. Generate responses from base model (no DPO) on same test prompts
2. Compare base → individual DPO → aggregate DPO
3. Measure verbosity and diversity at each stage

#### Experiment 4: LLM Judge Evaluation
1. Use GPT-4.1 to evaluate generated responses on: helpfulness, verbosity, personality/distinctiveness
2. Pairwise comparisons between 1-person and aggregate models

### Baselines
1. **Base model (Qwen2.5-1.5B-Instruct):** No DPO — isolates pretraining+SFT contribution
2. **Aggregate DPO:** Standard multi-annotator RLHF
3. **Size-matched random DPO:** Controls for data quantity vs. annotator identity

### Evaluation Metrics
1. **Output length** (tokens): Primary verbosity measure
2. **Length standard deviation**: Measures response length variety
3. **Distinct-n** (n=1,2,3): Lexical diversity (higher = less bland)
4. **Self-BLEU**: Inter-response similarity (lower = more diverse)
5. **Inter-model cosine similarity**: Whether different 1-person models produce similar outputs (lower = more personalized)
6. **GPT-4.1 judge scores**: Quality and distinctiveness ratings

### Statistical Analysis Plan
- **H1:** Kruskal-Wallis test across user length preferences, with Dunn's post-hoc
- **H2:** Mann-Whitney U comparing individual vs aggregate output lengths
- **H3:** Permutation test on inter-model similarity scores
- **H4:** Paired t-test comparing base model vs DPO model output lengths
- Significance level: α = 0.05 with Bonferroni correction for multiple comparisons
- Report effect sizes (Cohen's d) and 95% confidence intervals

## Expected Outcomes
- **H1 supported:** Individual users show diverse length preferences; aggregate preference is biased toward longer responses
- **H2 supported:** 1-person DPO produces outputs matching the individual's length preference (some shorter, some longer than aggregate)
- **H3 supported:** Different 1-person models produce distinguishable outputs; aggregate model produces homogeneous outputs
- **H4 partially supported:** Base model already shows some verbosity, but DPO amplifies it (especially aggregate DPO)

## Timeline and Milestones
1. Environment setup + data prep: 15 min
2. Experiment 1 (PRISM analysis): 20 min
3. Experiment 2 (DPO training): 60 min (parallelized across GPUs)
4. Experiment 3 (generation + measurement): 20 min
5. Experiment 4 (GPT-4.1 evaluation): 20 min
6. Analysis + visualization: 20 min
7. Documentation: 25 min

## Potential Challenges
1. **Limited per-user data in PRISM:** Mitigated by using PersonalLLM for training experiments
2. **PersonalLLM uses simulated annotators:** Less ecologically valid than real humans, but enables controlled experiments
3. **Small model (1.5B):** May not fully exhibit verbosity patterns of larger models; acknowledged as limitation
4. **LoRA DPO convergence with small data:** May need careful hyperparameter tuning

## Success Criteria
1. Clear statistical evidence of whether individual annotators differ in verbosity preferences
2. Measurable differences in output characteristics between 1-person and aggregate DPO
3. Quantified contribution of each pipeline stage to verbosity/blandness
4. At least 3 of 4 hypotheses tested with statistical rigor
