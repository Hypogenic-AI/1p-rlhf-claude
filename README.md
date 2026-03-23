# 1-Person RLHF: Disentangling Verbosity and Blandness

Does training on a single person's preferences avoid the "all AI sounds the same" problem? This research disentangles how much the verbosity and blandness of RLHF'd models is due to pretraining, SFT, or RLHF.

## Key Findings

- **84.1% of individual users prefer longer responses**, but preferences vary significantly (Kruskal-Wallis H=920.81, p<10^-93)
- **Aggregate RLHF amplifies verbosity by 21%** over the base model (143.6 vs 118.8 words)
- **A brevity-preferring annotator produces 30% shorter outputs** than aggregate (100.8 vs 143.6, p=0.042)
- **Blandness/low distinctiveness persists across ALL conditions** (GPT-4.1 rates 1.3-1.6/5) — this is a pretraining/SFT artifact, not RLHF
- **Decomposition: ~80% of verbosity from pretraining/SFT, ~20% added by RLHF aggregation**

## Project Structure

```
├── REPORT.md                  # Full research report with results
├── planning.md                # Experimental design and motivation
├── src/
│   ├── experiment1_prism_analysis.py    # PRISM per-user preference analysis
│   ├── experiment2_v3.py               # DPO training pipeline
│   ├── experiment3_generate_evaluate.py # Generation and metric computation
│   ├── experiment4_llm_judge.py        # GPT-4.1 evaluation
│   └── analysis_and_plots.py           # Statistical tests and visualization
├── results/
│   ├── exp1_results.json       # PRISM preference statistics
│   ├── exp2_pair_stats.json    # DPO training data properties
│   ├── exp3_metrics.json       # Generation metrics
│   ├── exp3_all_responses.json # All generated responses
│   ├── exp4_summary.json       # GPT-4.1 judge scores
│   ├── statistical_tests.json  # Statistical test results
│   └── models/                 # Trained LoRA adapters
├── figures/                    # Plots and visualizations
├── datasets/                   # Pre-downloaded datasets
├── papers/                     # Reference papers
└── code/                       # Reference code repositories
```

## Reproducing

```bash
# Setup
uv venv && source .venv/bin/activate
uv add torch transformers datasets accelerate peft trl scipy scikit-learn matplotlib seaborn pandas numpy openai

# Run experiments (requires GPU and OPENAI_API_KEY)
CUDA_VISIBLE_DEVICES=0 python src/experiment1_prism_analysis.py
CUDA_VISIBLE_DEVICES=0 python src/experiment2_v3.py
CUDA_VISIBLE_DEVICES=0 python src/experiment3_generate_evaluate.py
python src/experiment4_llm_judge.py
python src/analysis_and_plots.py
```

## Hardware
- GPU: NVIDIA RTX A6000 (49GB)
- Training time: ~5 min per DPO model (4 models total)
- Generation: ~5 min per model (5 models x 100 prompts)

See [REPORT.md](REPORT.md) for full details.
