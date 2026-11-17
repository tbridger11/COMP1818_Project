from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_FILE = PROJECT_ROOT / "results" / "evaluations" / "evaluation_results.csv"

MODEL_FILTERS = [
    ("ppo_plr_inspired_250k", "PPO with PLR-inspired selection (250k steps)"),
    ("ppo_mixed_optuna_250k", "Optuna-tuned PPO, mixed sizes (250k steps)"),
    ("ppo_mixed_250k", "PPO, mixed sizes (250k steps)"),
    ("ppo_15x15_optuna_250k", "Optuna-tuned PPO, fixed 15x15 (250k steps)"),
    ("ppo_20x20_baseline_250k", "PPO, fixed 20x20 (250k steps)"),
    ("ppo_15x15_baseline_250k", "PPO, fixed 15x15 (250k steps)"),
    ("ppo_10x10_baseline_250k", "PPO, fixed 10x10 (250k steps)"),
    ("ppo_5x5_baseline_250k", "PPO, fixed 5x5 (250k steps)"),
    ("random_action_seed", "Random action agent"),
    ("random_nav_only_seed", "Random navigation-only agent"),
]

COMPARISON_500K_FILTERS = [
    ("ppo_15x15_baseline_500k", "Fixed 15x15 PPO (500k steps)"),
    ("ppo_15x15_optuna_500k", "Optuna-tuned fixed 15x15 PPO (500k steps)"),
    ("ppo_mixed_optuna_500k", "Optuna-tuned mixed PPO (500k steps)"),
    ("ppo_plr_inspired_500k", "PLR-inspired PPO (500k steps)"),
]
COMPARISON_TEST_SIZES = {5, 10, 15, 20, 30, 40}
COMPARISON_TRAIN_SEEDS = {42, 123, 999}

PLOTS_DIR = PROJECT_ROOT / "plots_all_models"
TABLES_DIR = PROJECT_ROOT / "tables_all_models"
SUMMARY_FILE = TABLES_DIR / "summary_statistics.csv"
REQUIRED_COLUMNS = [
    "algorithm",
    "model_name",
    "train_size",
    "test_size",
    "timesteps",
    "seed",
    "episodes",
    "success_rate",
    "timeout_rate",
    "average_reward",
    "average_steps",
    "evaluation_time_seconds",
]

def validate_evaluation_data(df):
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("No evaluation results found.")

    print(f"Loaded {len(df)} records.")


def build_summary(df):
    seed_summary = (
        df.groupby(["method", "test_size", "seed"], as_index=False)
        .agg(success_rate=("success_rate", "mean"))
    )
    summary = seed_summary.groupby(["method", "test_size"]).agg(
        success_mean=("success_rate", "mean"),
        success_std=("success_rate", "std"),
        seed_count=("seed", "nunique"),
    )
    return summary.reset_index()


def create_generalisation_table(summary):
    means = summary.pivot(
        index="method",
        columns="test_size",
        values="success_mean",
    )
    standard_deviations = summary.pivot(
        index="method",
        columns="test_size",
        values="success_std",
    )
    results = means.copy()
    for test_size in means.columns:
        results[test_size] = [
            f"{mean * 100:.1f} ± {std * 100:.1f}"
            for mean, std in zip(
                means[test_size],
                standard_deviations[test_size],
            )
        ]

    results.index.name = "Training configuration"
    results.columns = [
        f"{test_size}x{test_size} grid"
        for test_size in results.columns
    ]
    results.columns.name = None

    results.to_csv(TABLES_DIR / "generalisation_results.csv")
    results.to_latex(TABLES_DIR / "generalisation_results.tex")


def create_overall_table(summary):
    overall = (
        summary.rename(
            columns={
                "method": "training_configuration",
                "test_size": "evaluation_size",
                "success_mean": "mean_success_rate",
                "success_std": "std_success_rate",
                "seed_count": "training_seed_count",
            }
        )
        .loc[
            :,
            [
                "training_configuration",
                "evaluation_size",
                "mean_success_rate",
                "std_success_rate",
                "training_seed_count",
            ],
        ]
        .sort_values(["training_configuration", "evaluation_size"])
    )

    overall.to_csv(TABLES_DIR / "overall_results.csv", index=False)
    overall.round(3).to_latex(
        TABLES_DIR / "overall_results.tex",
        index=False,
    )


def plot_generalisation(summary):
    fig, ax = plt.subplots(figsize=(7.2, 5))
    test_sizes = sorted(summary["test_size"].unique())
    x_positions = {
        test_size: position
        for position, test_size in enumerate(test_sizes)
    }

    for _, method in MODEL_FILTERS:
        if method not in summary["method"].unique():
            continue

        subset = summary[summary["method"] == method].copy()
        subset = subset.sort_values("test_size")

        ax.plot(
            subset["test_size"].map(x_positions),
            subset["success_mean"],
            marker="o",
            linewidth=2,
            label=method,
        )

    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_xticks(
        range(len(test_sizes)),
        [f"{size}x{size}" for size in test_sizes],
    )
    ax.set_xlabel("Evaluation grid size (width x height)")
    ax.set_ylabel("Mean success rate (%)")
    ax.set_title("PPO performance across evaluation grid sizes")
    ax.legend(
        title="Training configuration",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=2,
        fontsize=8,
        title_fontsize=9,
    )

    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        top=0.88,
        bottom=0.38,
    )

    plt.savefig(
        PLOTS_DIR / "generalisation_comparison.pdf",
        bbox_inches="tight",
    )

    plt.close()

def plot_unseen_sizes(summary):
    unseen = summary[summary["test_size"].isin([30, 40])]

    fig, ax = plt.subplots(figsize=(7.2, 5))

    sns.barplot(
        data=unseen,
        x="test_size",
        y="success_mean",
        hue="method",
        hue_order=[
            label
            for _, label in MODEL_FILTERS
            if label in unseen["method"].unique()
        ],
        ax=ax,
        errorbar=None,
    )

    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    test_sizes = sorted(unseen["test_size"].unique())
    ax.set_xticks(
        range(len(test_sizes)),
        [f"{size}x{size}" for size in test_sizes],
    )
    ax.set_title("PPO performance on grid sizes excluded from training")
    ax.set_xlabel("Evaluation grid size (width x height)")
    ax.set_ylabel("Mean success rate (%)")
    ax.legend(
        title="Training configuration",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=2,
        fontsize=8,
        title_fontsize=9,
    )

    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        top=0.88,
        bottom=0.38,
    )

    plt.savefig(
        PLOTS_DIR / "unseen_sizes_comparison.pdf",
        bbox_inches="tight",
    )

    plt.close()


def plot_500k_comparison(df):
    comparison = df.copy()
    comparison["method"] = pd.NA

    for model_name_filter, label in COMPARISON_500K_FILTERS:
        matches = comparison["model_name"].str.contains(
            model_name_filter,
            regex=False,
            na=False,
        )
        comparison.loc[matches, "method"] = label

    comparison = comparison[comparison["method"].notna()].copy()
    labels = [label for _, label in COMPARISON_500K_FILTERS]
    if not set(labels).issubset(set(comparison["method"])):
        print(
            "Skipping 500k comparison chart: evaluate all 500k models first."
        )
        return None

    for label in labels:
        sizes = set(comparison.loc[
            comparison["method"] == label,
            "test_size",
        ])
        if sizes != COMPARISON_TEST_SIZES:
            print(
                "Skipping 500k comparison chart: "
                f"{label} needs results for all evaluation sizes."
            )
            return None

        for test_size in sorted(COMPARISON_TEST_SIZES):
            seeds = set(comparison.loc[
                (comparison["method"] == label)
                & (comparison["test_size"] == test_size),
                "seed",
            ])
            if seeds != COMPARISON_TRAIN_SEEDS:
                print(
                    "Skipping 500k comparison chart: "
                    f"{label} needs seeds "
                    f"{sorted(COMPARISON_TRAIN_SEEDS)} for "
                    f"{test_size}x{test_size}; found {sorted(seeds)}."
                )
                return None

    summary = build_summary(comparison)
    fig, ax = plt.subplots(figsize=(7.2, 5))

    sns.barplot(
        data=summary,
        x="test_size",
        y="success_mean",
        hue="method",
        hue_order=labels,
        ax=ax,
        errorbar=None,
    )

    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_xticks(
        range(len(COMPARISON_TEST_SIZES)),
        [
            f"{size}x{size}"
            for size in sorted(COMPARISON_TEST_SIZES)
        ],
    )
    ax.set_xlabel("Evaluation grid size (width x height)")
    ax.set_ylabel("Mean success rate (%)")
    ax.set_title("500k PPO model comparison across grid sizes")
    ax.legend(
        title="Training configuration",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        fontsize=8,
        title_fontsize=9,
    )

    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        top=0.88,
        bottom=0.28,
    )

    plt.savefig(
        PLOTS_DIR / "500k_model_comparison.pdf",
        bbox_inches="tight",
    )

    plt.close()
    return summary


def main():
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(RESULTS_FILE)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    df = pd.read_csv(RESULTS_FILE)

    validate_evaluation_data(df)
    comparison_500k_summary = plot_500k_comparison(df)

    df["method"] = pd.NA
    for model_name_filter, label in MODEL_FILTERS:
        matches = df["model_name"].str.contains(
            model_name_filter,
            regex=False,
            na=False,
        )
        df.loc[matches, "method"] = label

    df = df[df["method"].notna()].copy()
    if df.empty:
        raise ValueError("No evaluation results matched MODEL_FILTERS.")

    plot_summary = build_summary(df)
    table_summary = plot_summary
    if comparison_500k_summary is not None:
        table_summary = pd.concat(
            [plot_summary, comparison_500k_summary],
            ignore_index=True,
        )

    table_summary.to_csv(SUMMARY_FILE, index=False)

    create_generalisation_table(table_summary)
    create_overall_table(table_summary)
    plot_generalisation(plot_summary)
    plot_unseen_sizes(plot_summary)

    print(f"Plots saved to: {PLOTS_DIR}")
    print(f"Tables saved to: {TABLES_DIR}")


if __name__ == "__main__":
    main()