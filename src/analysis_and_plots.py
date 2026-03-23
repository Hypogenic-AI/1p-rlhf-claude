"""
Statistical analysis and visualization for all experiments.

Creates publication-quality figures and runs statistical tests.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

sns.set_theme(style="whitegrid", font_scale=1.1)


def analyze_experiment2():
    """Analyze DPO pair statistics - what do different annotators prefer?"""
    pair_stats = json.load(open(RESULTS_DIR / "exp2_pair_stats.json"))

    print("\n=== Experiment 2: DPO Training Data Analysis ===")
    print(f"{'Condition':<30s} {'Pairs':>6s} {'Chosen Len':>11s} {'Rejected Len':>13s} {'Ratio':>7s}")
    print("-" * 70)
    for name, s in pair_stats.items():
        print(f"{name:<30s} {s['n_pairs']:>6d} {s['mean_chosen_length']:>11.0f} {s['mean_rejected_length']:>13.0f} {s['length_ratio']:>7.3f}")

    return pair_stats


def analyze_experiment3():
    """Analyze generation metrics across models."""
    metrics = json.load(open(RESULTS_DIR / "exp3_metrics.json"))
    similarities = json.load(open(RESULTS_DIR / "exp3_similarities.json"))

    print("\n=== Experiment 3: Generation Metrics ===")
    print(f"{'Model':<30s} {'Len(w)':>8s} {'Len SD':>8s} {'D-1':>8s} {'D-2':>8s} {'D-3':>8s} {'Self-BLEU':>10s} {'TTR':>8s}")
    print("-" * 95)
    for m in metrics:
        print(f"{m['model']:<30s} {m['mean_length_words']:>8.1f} {m['std_length_words']:>8.1f} "
              f"{m['distinct_1']:>8.4f} {m['distinct_2']:>8.4f} {m['distinct_3']:>8.4f} "
              f"{m['self_bleu']:>10.4f} {m['type_token_ratio']:>8.4f}")

    print("\n=== Inter-Model Similarity (TF-IDF cosine) ===")
    for pair, sim in sorted(similarities.items()):
        print(f"  {pair}: {sim:.4f}")

    return metrics, similarities


def analyze_experiment4():
    """Analyze GPT-4.1 judge scores."""
    if not (RESULTS_DIR / "exp4_summary.json").exists():
        print("\nExperiment 4 results not available.")
        return None

    summary = json.load(open(RESULTS_DIR / "exp4_summary.json"))

    print("\n=== Experiment 4: GPT-4.1 Judge Scores ===")
    print(f"{'Model':<30s} {'Helpful':>10s} {'Verbose':>10s} {'Distinct':>10s} {'Quality':>10s}")
    print("-" * 75)
    for model, scores in summary.items():
        h = scores.get("helpfulness", {}).get("mean", 0)
        v = scores.get("verbosity", {}).get("mean", 0)
        d = scores.get("distinctiveness", {}).get("mean", 0)
        q = scores.get("quality", {}).get("mean", 0)
        print(f"{model:<30s} {h:>10.2f} {v:>10.2f} {d:>10.2f} {q:>10.2f}")

    return summary


def statistical_tests(metrics):
    """Run statistical tests comparing conditions."""
    print("\n=== Statistical Tests ===")

    # Load all responses for detailed testing
    all_responses = json.load(open(RESULTS_DIR / "exp3_all_responses.json"))

    # H2: Individual vs aggregate output lengths
    results = {}

    # Get per-response lengths
    model_lengths = {}
    for model_name, pairs in all_responses.items():
        lengths = [len(resp.split()) for _, resp in pairs]
        model_lengths[model_name] = lengths

    # Compare each individual model to aggregate
    if "aggregate" in model_lengths:
        agg_lens = model_lengths["aggregate"]
        for name, lens in model_lengths.items():
            if name == "aggregate" or name == "base":
                continue
            u_stat, p_val = stats.mannwhitneyu(lens, agg_lens, alternative="two-sided")
            n1, n2 = len(lens), len(agg_lens)
            effect_r = 1 - 2 * u_stat / (n1 * n2)
            d = (np.mean(lens) - np.mean(agg_lens)) / np.sqrt((np.std(lens)**2 + np.std(agg_lens)**2) / 2)
            results[f"{name}_vs_aggregate"] = {
                "U": float(u_stat), "p": float(p_val), "effect_r": float(effect_r), "cohens_d": float(d),
                "mean_individual": float(np.mean(lens)), "mean_aggregate": float(np.mean(agg_lens)),
            }
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"  {name} vs aggregate: U={u_stat:.0f}, p={p_val:.4f} {sig}, d={d:.3f}")

    # H4: Base vs all DPO models
    if "base" in model_lengths:
        base_lens = model_lengths["base"]
        for name, lens in model_lengths.items():
            if name == "base":
                continue
            u_stat, p_val = stats.mannwhitneyu(lens, base_lens, alternative="two-sided")
            d = (np.mean(lens) - np.mean(base_lens)) / np.sqrt((np.std(lens)**2 + np.std(base_lens)**2) / 2)
            results[f"{name}_vs_base"] = {
                "U": float(u_stat), "p": float(p_val), "cohens_d": float(d),
                "mean_model": float(np.mean(lens)), "mean_base": float(np.mean(base_lens)),
            }
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"  {name} vs base: U={u_stat:.0f}, p={p_val:.4f} {sig}, d={d:.3f}")

    with open(RESULTS_DIR / "statistical_tests.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def create_summary_plots(metrics, similarities, exp4_summary=None):
    """Create publication-quality summary plots."""

    # Figure 1: Verbosity comparison across models
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    models = [m["model"] for m in metrics]
    lengths = [m["mean_length_words"] for m in metrics]
    length_stds = [m["std_length_words"] for m in metrics]
    distinct2 = [m["distinct_2"] for m in metrics]
    self_bleu = [m["self_bleu"] for m in metrics]

    # Color coding
    colors = []
    for m in models:
        if m == "base":
            colors.append("#2196F3")  # blue
        elif m == "aggregate":
            colors.append("#F44336")  # red
        elif "individual" in m:
            colors.append("#4CAF50")  # green
        else:
            colors.append("#FF9800")  # orange

    # 1a: Mean response length
    ax = axes[0]
    bars = ax.bar(range(len(models)), lengths, yerr=length_stds, color=colors, alpha=0.8,
                  capsize=3, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace("individual_", "").replace("_", "\n") for m in models],
                       rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean Response Length (words)")
    ax.set_title("(a) Verbosity: Response Length")

    # 1b: Distinct-2 (lexical diversity)
    ax = axes[1]
    ax.bar(range(len(models)), distinct2, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace("individual_", "").replace("_", "\n") for m in models],
                       rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Distinct-2 (higher = more diverse)")
    ax.set_title("(b) Lexical Diversity")

    # 1c: Self-BLEU (output homogeneity)
    ax = axes[2]
    ax.bar(range(len(models)), self_bleu, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace("individual_", "").replace("_", "\n") for m in models],
                       rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Self-BLEU (lower = more diverse)")
    ax.set_title("(c) Output Homogeneity")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "summary_verbosity_diversity.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 2: Inter-model similarity heatmap
    if similarities:
        model_names = list(set(
            [k.split("_vs_")[0] for k in similarities.keys()] +
            [k.split("_vs_")[1] for k in similarities.keys()]
        ))
        model_names.sort()
        n = len(model_names)
        sim_matrix = np.eye(n)
        for i, m1 in enumerate(model_names):
            for j, m2 in enumerate(model_names):
                key1 = f"{m1}_vs_{m2}"
                key2 = f"{m2}_vs_{m1}"
                if key1 in similarities:
                    sim_matrix[i, j] = similarities[key1]
                elif key2 in similarities:
                    sim_matrix[i, j] = similarities[key2]

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(sim_matrix, annot=True, fmt=".3f", cmap="YlOrRd",
                    xticklabels=[m.replace("individual_", "").replace("_", "\n") for m in model_names],
                    yticklabels=[m.replace("individual_", "").replace("_", "\n") for m in model_names],
                    ax=ax, vmin=0, vmax=1)
        ax.set_title("Inter-Model Output Similarity (TF-IDF Cosine)")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "inter_model_similarity.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Figure 3: GPT-4.1 judge scores (if available)
    if exp4_summary:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        dims = ["helpfulness", "verbosity", "distinctiveness", "quality"]
        dim_titles = ["Helpfulness", "Verbosity", "Distinctiveness", "Quality"]

        judge_models = list(exp4_summary.keys())
        judge_colors = []
        for m in judge_models:
            if m == "base":
                judge_colors.append("#2196F3")
            elif m == "aggregate":
                judge_colors.append("#F44336")
            elif "individual" in m:
                judge_colors.append("#4CAF50")
            else:
                judge_colors.append("#FF9800")

        for idx, (dim, title) in enumerate(zip(dims, dim_titles)):
            ax = axes[idx]
            means = [exp4_summary[m].get(dim, {}).get("mean", 0) for m in judge_models]
            stds = [exp4_summary[m].get(dim, {}).get("std", 0) for m in judge_models]
            ax.bar(range(len(judge_models)), means, yerr=stds, color=judge_colors,
                   alpha=0.8, capsize=3, edgecolor="black", linewidth=0.5)
            ax.set_xticks(range(len(judge_models)))
            ax.set_xticklabels([m.replace("individual_", "").replace("_", "\n") for m in judge_models],
                              rotation=45, ha="right", fontsize=8)
            ax.set_ylabel(f"{title} Score (1-5)")
            ax.set_title(f"{title}")
            ax.set_ylim(0, 5.5)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "judge_scores.png", dpi=150, bbox_inches="tight")
        plt.close()

    print("\nAll plots saved to figures/")


def main():
    print("=" * 60)
    print("Statistical Analysis and Visualization")
    print("=" * 60)

    pair_stats = analyze_experiment2()
    metrics, similarities = analyze_experiment3()
    exp4_summary = analyze_experiment4()
    stat_results = statistical_tests(metrics)
    create_summary_plots(metrics, similarities, exp4_summary)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
