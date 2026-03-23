"""
Experiment 1: PRISM Per-User Preference Analysis

Analyzes whether individual PRISM users have different verbosity preferences
compared to the aggregate. Tests H1: individual annotators have significantly
different length preferences.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datasets import load_from_disk
from pathlib import Path

SEED = 42
np.random.seed(SEED)

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

def load_prism_utterances():
    """Load PRISM utterances and convert to DataFrame."""
    ds = load_from_disk("datasets/prism-utterances")["train"]
    df = pd.DataFrame(ds)
    df["response_length"] = df["model_response"].str.split().str.len()
    df["response_char_length"] = df["model_response"].str.len()
    return df

def compute_per_user_preferences(df):
    """Compute per-user length preference statistics."""
    # For each user, compute mean length of chosen vs rejected responses
    user_stats = []

    for user_id, user_df in df.groupby("user_id"):
        chosen = user_df[user_df["if_chosen"] == True]
        rejected = user_df[user_df["if_chosen"] == False]

        if len(chosen) < 3 or len(rejected) < 3:
            continue

        chosen_len = chosen["response_length"].values
        rejected_len = rejected["response_length"].values

        # Length preference ratio: >1 means prefers longer, <1 means prefers shorter
        mean_chosen_len = np.mean(chosen_len)
        mean_rejected_len = np.mean(rejected_len)
        length_pref_ratio = mean_chosen_len / max(mean_rejected_len, 1)

        # Correlation between score and length for this user
        if len(user_df) >= 5:
            corr, p_corr = stats.spearmanr(user_df["score"], user_df["response_length"])
        else:
            corr, p_corr = np.nan, np.nan

        user_stats.append({
            "user_id": user_id,
            "n_chosen": len(chosen),
            "n_rejected": len(rejected),
            "mean_chosen_length": mean_chosen_len,
            "mean_rejected_length": mean_rejected_len,
            "length_pref_ratio": length_pref_ratio,
            "chosen_length_std": np.std(chosen_len),
            "length_score_corr": corr,
            "length_score_p": p_corr,
            "prefers_longer": mean_chosen_len > mean_rejected_len,
        })

    return pd.DataFrame(user_stats)

def test_h1_user_differences(df, user_stats):
    """Test H1: Do individual users differ in length preferences?"""
    results = {}

    # Kruskal-Wallis test across users' length preference ratios
    # Group responses by user and test if length distributions differ
    groups = []
    user_ids = []
    for user_id, user_df in df.groupby("user_id"):
        chosen = user_df[user_df["if_chosen"] == True]["response_length"].values
        if len(chosen) >= 3:
            groups.append(chosen)
            user_ids.append(user_id)

    if len(groups) >= 3:
        stat, p_value = stats.kruskal(*groups[:200])  # Limit to avoid memory issues
        results["kruskal_wallis_stat"] = float(stat)
        results["kruskal_wallis_p"] = float(p_value)
        results["kruskal_wallis_n_groups"] = min(len(groups), 200)

    # Summary statistics
    results["n_users_analyzed"] = len(user_stats)
    results["pct_prefer_longer"] = float(user_stats["prefers_longer"].mean() * 100)
    results["pct_prefer_shorter"] = float((~user_stats["prefers_longer"]).mean() * 100)
    results["mean_length_pref_ratio"] = float(user_stats["length_pref_ratio"].mean())
    results["std_length_pref_ratio"] = float(user_stats["length_pref_ratio"].std())
    results["median_length_pref_ratio"] = float(user_stats["length_pref_ratio"].median())

    # Correlation between score and length (across all users)
    valid_corrs = user_stats.dropna(subset=["length_score_corr"])
    results["mean_length_score_corr"] = float(valid_corrs["length_score_corr"].mean())
    results["std_length_score_corr"] = float(valid_corrs["length_score_corr"].std())
    results["pct_positive_corr"] = float((valid_corrs["length_score_corr"] > 0).mean() * 100)
    results["pct_sig_positive_corr"] = float(
        ((valid_corrs["length_score_corr"] > 0) & (valid_corrs["length_score_p"] < 0.05)).mean() * 100
    )

    # Aggregate length preference
    all_chosen = df[df["if_chosen"] == True]["response_length"]
    all_rejected = df[df["if_chosen"] == False]["response_length"]
    results["aggregate_chosen_mean_length"] = float(all_chosen.mean())
    results["aggregate_rejected_mean_length"] = float(all_rejected.mean())
    results["aggregate_length_pref_ratio"] = float(all_chosen.mean() / max(all_rejected.mean(), 1))

    # Mann-Whitney U: aggregate chosen vs rejected lengths
    u_stat, u_p = stats.mannwhitneyu(all_chosen, all_rejected, alternative="two-sided")
    results["aggregate_mannwhitney_u"] = float(u_stat)
    results["aggregate_mannwhitney_p"] = float(u_p)

    # Effect size (rank-biserial correlation)
    n1, n2 = len(all_chosen), len(all_rejected)
    results["aggregate_effect_size_r"] = float(1 - 2 * u_stat / (n1 * n2))

    return results

def create_visualizations(user_stats, df):
    """Create plots for Experiment 1."""

    # Plot 1: Distribution of length preference ratios across users
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1a: Histogram of length preference ratios
    ax = axes[0, 0]
    ratios = user_stats["length_pref_ratio"].clip(0, 5)
    ax.hist(ratios, bins=40, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(1.0, color="red", linestyle="--", linewidth=2, label="No preference")
    ax.axvline(ratios.median(), color="orange", linestyle="-", linewidth=2, label=f"Median={ratios.median():.2f}")
    ax.set_xlabel("Length Preference Ratio (chosen/rejected)")
    ax.set_ylabel("Number of Users")
    ax.set_title("Per-User Length Preference Ratios")
    ax.legend()

    # 1b: Per-user length-score correlations
    ax = axes[0, 1]
    valid = user_stats.dropna(subset=["length_score_corr"])
    ax.hist(valid["length_score_corr"], bins=40, edgecolor="black", alpha=0.7, color="coral")
    ax.axvline(0, color="red", linestyle="--", linewidth=2)
    ax.axvline(valid["length_score_corr"].mean(), color="orange", linestyle="-", linewidth=2,
               label=f"Mean={valid['length_score_corr'].mean():.3f}")
    ax.set_xlabel("Spearman Correlation (score vs length)")
    ax.set_ylabel("Number of Users")
    ax.set_title("Per-User Score-Length Correlation")
    ax.legend()

    # 1c: Chosen vs rejected length by user (scatter)
    ax = axes[1, 0]
    ax.scatter(user_stats["mean_rejected_length"], user_stats["mean_chosen_length"],
               alpha=0.3, s=15, color="steelblue")
    max_len = max(user_stats["mean_chosen_length"].max(), user_stats["mean_rejected_length"].max())
    ax.plot([0, max_len], [0, max_len], "r--", linewidth=1, label="Equal length")
    ax.set_xlabel("Mean Rejected Response Length (words)")
    ax.set_ylabel("Mean Chosen Response Length (words)")
    ax.set_title("Chosen vs Rejected Length by User")
    ax.legend()

    # 1d: Distribution of chosen lengths across all users
    ax = axes[1, 1]
    chosen = df[df["if_chosen"] == True]["response_length"]
    rejected = df[df["if_chosen"] == False]["response_length"]
    ax.hist(chosen.clip(0, 500), bins=50, alpha=0.6, label=f"Chosen (mean={chosen.mean():.0f})",
            color="green", density=True)
    ax.hist(rejected.clip(0, 500), bins=50, alpha=0.6, label=f"Rejected (mean={rejected.mean():.0f})",
            color="red", density=True)
    ax.set_xlabel("Response Length (words)")
    ax.set_ylabel("Density")
    ax.set_title("Aggregate Chosen vs Rejected Length Distributions")
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "exp1_prism_length_preferences.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: Variance across users - how much do individuals differ?
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Top 30 users by data volume, showing their preference ratios
    top_users = user_stats.nlargest(30, "n_chosen").sort_values("length_pref_ratio")
    ax = axes[0]
    colors = ["green" if r > 1 else "red" for r in top_users["length_pref_ratio"]]
    ax.barh(range(len(top_users)), top_users["length_pref_ratio"], color=colors, alpha=0.7)
    ax.axvline(1.0, color="black", linestyle="--")
    ax.set_yticks(range(len(top_users)))
    ax.set_yticklabels(top_users["user_id"], fontsize=7)
    ax.set_xlabel("Length Preference Ratio")
    ax.set_title("Top 30 Users: Length Preference (green=prefers longer)")

    # Boxplot of per-user length-score correlations by preference direction
    ax = axes[1]
    user_stats_valid = user_stats.dropna(subset=["length_score_corr"])
    groups = [
        user_stats_valid[user_stats_valid["prefers_longer"]]["length_score_corr"],
        user_stats_valid[~user_stats_valid["prefers_longer"]]["length_score_corr"]
    ]
    bp = ax.boxplot(groups, labels=["Prefers Longer", "Prefers Shorter"], patch_artist=True)
    bp["boxes"][0].set_facecolor("lightgreen")
    bp["boxes"][1].set_facecolor("lightcoral")
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_ylabel("Score-Length Correlation")
    ax.set_title("Score-Length Correlation by Preference Direction")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "exp1_user_variation.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Experiment 1 visualizations saved.")

def main():
    print("=" * 60)
    print("Experiment 1: PRISM Per-User Preference Analysis")
    print("=" * 60)

    # Load data
    print("\nLoading PRISM utterances...")
    df = load_prism_utterances()
    print(f"Loaded {len(df)} utterances from {df['user_id'].nunique()} users")

    # Compute per-user stats
    print("\nComputing per-user preference statistics...")
    user_stats = compute_per_user_preferences(df)
    print(f"Analyzed {len(user_stats)} users with sufficient data")

    # Test H1
    print("\nTesting H1: Do individual users differ in length preferences?")
    h1_results = test_h1_user_differences(df, user_stats)

    # Print key results
    print(f"\n--- Key Results ---")
    print(f"Users preferring longer responses: {h1_results['pct_prefer_longer']:.1f}%")
    print(f"Users preferring shorter responses: {h1_results['pct_prefer_shorter']:.1f}%")
    print(f"Mean length preference ratio: {h1_results['mean_length_pref_ratio']:.3f}")
    print(f"Std of length preference ratio: {h1_results['std_length_pref_ratio']:.3f}")
    print(f"Aggregate chosen mean length: {h1_results['aggregate_chosen_mean_length']:.1f} words")
    print(f"Aggregate rejected mean length: {h1_results['aggregate_rejected_mean_length']:.1f} words")
    print(f"Kruskal-Wallis H={h1_results.get('kruskal_wallis_stat', 'N/A'):.2f}, p={h1_results.get('kruskal_wallis_p', 'N/A'):.4e}")
    print(f"Mean per-user score-length correlation: {h1_results['mean_length_score_corr']:.3f}")
    print(f"Users with positive score-length correlation: {h1_results['pct_positive_corr']:.1f}%")
    print(f"Users with significant positive correlation: {h1_results['pct_sig_positive_corr']:.1f}%")

    # Save results
    with open(RESULTS_DIR / "exp1_results.json", "w") as f:
        json.dump(h1_results, f, indent=2)
    user_stats.to_csv(RESULTS_DIR / "exp1_user_stats.csv", index=False)

    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(user_stats, df)

    print("\nExperiment 1 complete. Results saved to results/exp1_results.json")
    return h1_results, user_stats

if __name__ == "__main__":
    main()
