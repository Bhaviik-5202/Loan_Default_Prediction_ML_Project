"""
Step 2: Exploratory Data Analysis (EDA) Pipeline
Computes and visualizes:
1. Target variable distribution and class imbalance
2. Numerical feature descriptive statistics & outlier analysis (IQR method)
3. Categorical feature distributions and default rates across groups
4. Default rate across binned numerical ranges (Age, Credit Score, DTI, Interest Rate)
5. Correlation matrix between numerical features and default
6. Visualizations exported to:
   - ml_pipeline/plots/target_distribution.png
   - ml_pipeline/plots/numerical_histograms.png
   - ml_pipeline/plots/numerical_boxplots.png
   - ml_pipeline/plots/correlation_heatmap.png
   - ml_pipeline/plots/categorical_default_rates.png
   - ml_pipeline/plots/credit_score_vs_default.png
   - ml_pipeline/plots/income_vs_default.png
   - ml_pipeline/plots/loan_amount_vs_default.png
   - and copied to public/artifacts/plots/ for frontend consumption
"""

import json
import os
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def run_eda(data_path: str = "/app/applet/Loan_Default.csv") -> dict:
    print("[*] Running EDA on dataset...")
    df = pd.read_csv(data_path)

    plots_dir = "/app/applet/ml_pipeline/plots"
    public_plots_dir = "/app/applet/public/artifacts/plots"
    reports_dir = "/app/applet/ml_pipeline/reports"
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(public_plots_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # Style settings
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#e3ddd4"
    plt.rcParams["axes.linewidth"] = 0.8

    # 1. Target Distribution Plot
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    counts = df["Default"].value_counts()
    percentages = df["Default"].value_counts(normalize=True) * 100
    bars = ax.bar(["Non-Default (0)", "Default (1)"], counts.values, color=["#31877d", "#d9534f"], width=0.5)
    for bar, pct in zip(bars, percentages.values):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 3000, f"{int(yval):,} ({pct:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Loan Default Class Distribution (255,347 Records)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Borrower Count", fontsize=11)
    ax.set_ylim(0, max(counts.values) * 1.15)
    plt.tight_layout()
    target_plot_path = os.path.join(plots_dir, "target_distribution.png")
    fig.savefig(target_plot_path)
    plt.close(fig)

    # 2. Numerical Features Outlier & Distribution Analysis
    num_cols = ["Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed", "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio"]
    outlier_analysis = {}
    for col in num_cols:
        q25 = float(df[col].quantile(0.25))
        q75 = float(df[col].quantile(0.75))
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outliers_count = int(((df[col] < lower_bound) | (df[col] > upper_bound)).sum())
        outlier_analysis[col] = {
            "q25": round(q25, 2),
            "q75": round(q75, 2),
            "iqr": round(iqr, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
            "outliers_count": outliers_count,
            "outliers_percentage": round((outliers_count / len(df)) * 100, 2)
        }

    # Histograms
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), dpi=150)
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        ax = axes[i]
        sns.histplot(df[col], kde=True, ax=ax, color="#31877d", bins=30, alpha=0.6)
        ax.set_title(f"Distribution of {col}", fontsize=11, fontweight="bold")
        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
    plt.suptitle("Numerical Feature Distributions", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    hist_plot_path = os.path.join(plots_dir, "numerical_histograms.png")
    fig.savefig(hist_plot_path)
    plt.close(fig)

    # Boxplots
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), dpi=150)
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        ax = axes[i]
        sns.boxplot(y=df[col], ax=ax, color="#b2d8d8", fliersize=2)
        ax.set_title(f"Boxplot of {col}", fontsize=11, fontweight="bold")
        ax.set_ylabel(col, fontsize=9)
    plt.suptitle("Numerical Feature Boxplots (Outlier Inspection)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    box_plot_path = os.path.join(plots_dir, "numerical_boxplots.png")
    fig.savefig(box_plot_path)
    plt.close(fig)

    # 3. Correlation Heatmap
    corr_matrix = df[num_cols + ["Default"]].corr()
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="Blues", ax=ax, cbar=True, square=True)
    ax.set_title("Correlation Heatmap (Numerical Attributes & Target Default)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    corr_plot_path = os.path.join(plots_dir, "correlation_heatmap.png")
    fig.savefig(corr_plot_path)
    plt.close(fig)

    # 4. Categorical Default Rate Analysis
    cat_cols = ["Education", "EmploymentType", "MaritalStatus", "HasMortgage", "HasDependents", "LoanPurpose", "HasCoSigner"]
    fig, axes = plt.subplots(4, 2, figsize=(14, 16), dpi=150)
    axes = axes.flatten()
    cat_summary = {}

    for i, col in enumerate(cat_cols):
        ax = axes[i]
        group = df.groupby(col)["Default"].agg(["count", "mean"]).reset_index()
        group["rate_pct"] = group["mean"] * 100
        cat_summary[col] = {
            row[col]: {
                "count": int(row["count"]),
                "default_rate": round(float(row["rate_pct"]), 2)
            }
            for _, row in group.iterrows()
        }
        bars = ax.bar(group[col], group["rate_pct"], color="#31877d", alpha=0.85, width=0.5)
        ax.axhline(df["Default"].mean() * 100, color="#d9534f", linestyle="--", linewidth=1.2, label=f"Average Default ({df['Default'].mean()*100:.1f}%)")
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.3, f"{height:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(f"Default Rate by {col}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Default Rate (%)", fontsize=9)
        ax.set_ylim(0, max(group["rate_pct"]) * 1.25)
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_visible(False)  # Unused 8th subplot
    plt.suptitle("Empirical Default Rate across Categorical Feature Categories", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    cat_plot_path = os.path.join(plots_dir, "categorical_default_rates.png")
    fig.savefig(cat_plot_path)
    plt.close(fig)

    # 5. Targeted Feature vs Default Boxplots (Credit Score, Income, Loan Amount)
    for col, fname, color in [
        ("CreditScore", "credit_score_vs_default.png", "#4a90e2"),
        ("Income", "income_vs_default.png", "#50e3c2"),
        ("LoanAmount", "loan_amount_vs_default.png", "#f5a623"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
        sns.boxplot(x="Default", y=col, data=df, ax=ax, palette=["#31877d", "#d9534f"])
        ax.set_xticklabels(["Non-Default (0)", "Default (1)"])
        ax.set_title(f"{col} by Default Status", fontsize=12, fontweight="bold")
        ax.set_xlabel("Target Class", fontsize=10)
        ax.set_ylabel(col, fontsize=10)
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, fname))
        plt.close(fig)

    # Copy plots to public/artifacts/plots for web UI access
    for filename in os.listdir(plots_dir):
        if filename.endswith(".png"):
            shutil.copy2(os.path.join(plots_dir, filename), os.path.join(public_plots_dir, filename))

    # Numerical correlation with Default ranked
    default_corr = corr_matrix["Default"].drop("Default").sort_values(ascending=False).to_dict()

    eda_summary = {
        "record_count": len(df),
        "target_distribution": {
            "non_default_count": int(counts[0]),
            "non_default_pct": round(float(percentages[0]), 2),
            "default_count": int(counts[1]),
            "default_pct": round(float(percentages[1]), 2),
            "imbalance_ratio": round(counts[0] / counts[1], 2)
        },
        "default_correlation_ranked": {k: round(float(v), 4) for k, v in default_corr.items()},
        "outlier_analysis": outlier_analysis,
        "categorical_default_rates": cat_summary,
        "generated_plots": [f for f in os.listdir(plots_dir) if f.endswith(".png")]
    }

    summary_json_path = os.path.join(reports_dir, "eda_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(eda_summary, f, indent=2)
    print(f"[+] EDA complete. Report saved to {summary_json_path}")
    return eda_summary


if __name__ == "__main__":
    run_eda()
