"""Generate EDA artifacts from the canonical Parquet dataset."""
from __future__ import annotations

import argparse
import base64
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq


DEFAULT_DATASET = Path("data/processed/bughubs_canonical.parquet")
TABLES_DIR = Path("outputs/tables")
FIGURES_DIR = Path("outputs/figures")
REPORT_PATH = Path("outputs/reports/EDA_Report.html")

IDENTITY_COLUMNS = ["source", "project", "issue_id"]
MISSINGNESS_COLUMNS = [
    "title",
    "description",
    "component",
    "severity",
    "priority",
    "status",
    "resolution",
    "created_at",
    "updated_at",
]


def read_columns(dataset_path: Path, columns: list[str]) -> pd.DataFrame:
    return pd.read_parquet(dataset_path, columns=columns)


def write_table(df: pd.DataFrame, name: str) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / name, index=False)


def html_table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, classes="data-table", border=0)


def image_src(path: Path) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def compute_overview(dataset_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pq.ParquetFile(dataset_path).metadata
    labels = read_columns(dataset_path, ["original_label"])
    identity = read_columns(dataset_path, IDENTITY_COLUMNS)

    label_counts = labels["original_label"].value_counts(dropna=False)
    total_records = metadata.num_rows
    bugs = int(label_counts.get("bug", 0))
    non_bugs = int(label_counts.get("non-bug", 0))

    overview = pd.DataFrame(
        [
            {
                "dataset_path": dataset_path.as_posix(),
                "total_records": total_records,
                "unique_composite_issues": int(
                    identity.drop_duplicates(IDENTITY_COLUMNS).shape[0]
                ),
                "duplicate_composite_identities": int(
                    identity.duplicated(IDENTITY_COLUMNS).sum()
                ),
                "sources": int(identity["source"].nunique(dropna=True)),
                "projects": int(identity["project"].nunique(dropna=True)),
                "bugs": bugs,
                "non_bugs": non_bugs,
                "bug_percentage": round(bugs / total_records * 100, 2),
                "non_bug_percentage": round(non_bugs / total_records * 100, 2),
                "parquet_row_groups": metadata.num_row_groups,
            }
        ]
    )

    label_distribution = (
        labels["original_label"]
        .value_counts(dropna=False)
        .rename_axis("original_label")
        .reset_index(name="records")
    )
    label_distribution["percentage"] = (
        label_distribution["records"] / total_records * 100
    ).round(2)

    return overview, label_distribution


def compute_group_distribution(dataset_path: Path, group_column: str) -> pd.DataFrame:
    df = read_columns(dataset_path, [group_column, "original_label"])
    result = (
        df.groupby(group_column, dropna=False)["original_label"]
        .value_counts(dropna=False)
        .unstack(fill_value=0)
        .reset_index()
    )
    for label in ("bug", "non-bug"):
        if label not in result.columns:
            result[label] = 0
    result = result.rename(columns={"bug": "bugs", "non-bug": "non_bugs"})
    result["records"] = result["bugs"] + result["non_bugs"]
    result["bug_percentage"] = (result["bugs"] / result["records"] * 100).round(2)
    return result[
        [group_column, "records", "bugs", "non_bugs", "bug_percentage"]
    ].sort_values("records", ascending=False)


def compute_text_statistics(dataset_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = read_columns(dataset_path, ["model_text"])
    text_lengths = df["model_text"].fillna("").str.len()
    stats = pd.DataFrame(
        [
            {
                "records": int(text_lengths.shape[0]),
                "mean_chars": round(float(text_lengths.mean()), 2),
                "std_chars": round(float(text_lengths.std()), 2),
                "min_chars": int(text_lengths.min()),
                "p25_chars": float(text_lengths.quantile(0.25)),
                "median_chars": float(text_lengths.quantile(0.50)),
                "p75_chars": float(text_lengths.quantile(0.75)),
                "max_chars": int(text_lengths.max()),
            }
        ]
    )

    bins = [0, 100, 250, 500, 1000, 2500, 5000, 10000, float("inf")]
    labels = [
        "0-99",
        "100-249",
        "250-499",
        "500-999",
        "1000-2499",
        "2500-4999",
        "5000-9999",
        "10000+",
    ]
    distribution = (
        pd.cut(text_lengths, bins=bins, labels=labels, right=False)
        .value_counts()
        .sort_index()
        .rename_axis("text_length_chars")
        .reset_index(name="records")
    )
    distribution["percentage"] = (
        distribution["records"] / text_lengths.shape[0] * 100
    ).round(2)
    return stats, distribution


def compute_missingness(dataset_path: Path) -> pd.DataFrame:
    df = read_columns(dataset_path, MISSINGNESS_COLUMNS)
    total = len(df)
    rows = []
    for column in MISSINGNESS_COLUMNS:
        values = df[column]
        missing = values.isna() | values.astype("string").str.strip().eq("")
        rows.append(
            {
                "field": column,
                "missing_count": int(missing.sum()),
                "missing_percentage": round(float(missing.mean() * 100), 2),
                "present_count": int(total - missing.sum()),
            }
        )
    return pd.DataFrame(rows)


def compute_temporal_distribution(dataset_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = read_columns(dataset_path, ["created_at", "original_label"])
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["created_at"]).copy()
    df["year"] = df["created_at"].dt.year
    df["month"] = df["created_at"].dt.to_period("M").astype(str)

    yearly = make_temporal_table(df, "year")
    monthly = make_temporal_table(df, "month")
    return yearly, monthly


def make_temporal_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    result = (
        df.groupby(column)["original_label"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
        .sort_values(column)
    )
    for label in ("bug", "non-bug"):
        if label not in result.columns:
            result[label] = 0
    result = result.rename(columns={"bug": "bugs", "non-bug": "non_bugs"})
    result["records"] = result["bugs"] + result["non_bugs"]
    result["bug_percentage"] = (result["bugs"] / result["records"] * 100).round(2)
    return result[[column, "records", "bugs", "non_bugs", "bug_percentage"]]


def save_figures(
    projects: pd.DataFrame,
    missingness: pd.DataFrame,
    yearly: pd.DataFrame,
    text_distribution: pd.DataFrame,
) -> dict[str, Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "projects": FIGURES_DIR / "top_20_projects.png",
        "missingness": FIGURES_DIR / "missingness.png",
        "temporal": FIGURES_DIR / "temporal_distribution.png",
        "text": FIGURES_DIR / "text_length_distribution.png",
    }

    top_projects = projects.head(20).sort_values("records")
    plt.figure(figsize=(10, 8))
    plt.barh(top_projects["project"].astype(str), top_projects["records"])
    plt.xlabel("Issues")
    plt.ylabel("Project")
    plt.title("Top 20 Projects by Issue Count")
    plt.tight_layout()
    plt.savefig(paths["projects"], dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(missingness["field"], missingness["missing_percentage"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Missing values (%)")
    plt.title("Missingness by Canonical Field")
    plt.tight_layout()
    plt.savefig(paths["missingness"], dpi=300)
    plt.close()

    valid_years = yearly[yearly["year"] >= 1990]
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(valid_years["year"], valid_years["records"], alpha=0.7)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Issues")
    ax2 = ax1.twinx()
    ax2.plot(valid_years["year"], valid_years["bug_percentage"], marker="o")
    ax2.set_ylabel("Bug percentage")
    plt.title("Issue Volume and Bug Percentage by Year")
    fig.tight_layout()
    plt.savefig(paths["temporal"], dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(text_distribution["text_length_chars"].astype(str), text_distribution["records"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Records")
    plt.title("model_text Length Distribution")
    plt.tight_layout()
    plt.savefig(paths["text"], dpi=300)
    plt.close()
    return paths


def write_report(
    overview: pd.DataFrame,
    labels: pd.DataFrame,
    sources: pd.DataFrame,
    projects: pd.DataFrame,
    text_stats: pd.DataFrame,
    text_distribution: pd.DataFrame,
    missingness: pd.DataFrame,
    yearly: pd.DataFrame,
    figure_paths: dict[str, Path],
) -> None:
    o = overview.iloc[0]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>EDA Report - Multi-Agent-SDP</title>
<style>
body {{ font-family: Arial, sans-serif; color: #1f2933; margin: 36px; line-height: 1.5; }}
h1, h2 {{ color: #12355b; }}
.summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }}
.card {{ border: 1px solid #d9e2ec; border-radius: 6px; padding: 14px; background: #f8fafc; }}
.label {{ color: #52606d; font-size: 0.85rem; }}
.value {{ font-size: 1.35rem; font-weight: 700; margin-top: 4px; }}
.data-table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 0.95rem; }}
.data-table th, .data-table td {{ border: 1px solid #d9e2ec; padding: 8px; text-align: left; }}
.data-table th {{ background: #eef2f7; }}
.figure {{ margin: 16px 0 28px; }}
.figure img {{ max-width: 100%; height: auto; border: 1px solid #d9e2ec; }}
code {{ background: #eef2f7; padding: 2px 4px; border-radius: 4px; }}
</style>
</head>
<body>
<h1>Exploratory Data Analysis Report</h1>
<p><strong>Project:</strong> Multi-Agent Software Defect Prediction</p>
<p><strong>Input artifact:</strong> <code>{o.dataset_path}</code></p>
<p>This report is generated from the canonical Parquet dataset described in Stage 1 of the architecture. The original BugHub source label is preserved as <code>original_label</code>.</p>

<div class="summary">
  <div class="card"><div class="label">Total Records</div><div class="value">{o.total_records:,}</div></div>
  <div class="card"><div class="label">Unique Composite Issues</div><div class="value">{o.unique_composite_issues:,}</div></div>
  <div class="card"><div class="label">Duplicate Composite IDs</div><div class="value">{o.duplicate_composite_identities:,}</div></div>
  <div class="card"><div class="label">Sources</div><div class="value">{o.sources}</div></div>
  <div class="card"><div class="label">Projects</div><div class="value">{o.projects:,}</div></div>
  <div class="card"><div class="label">Class Split</div><div class="value">BUG {o.bug_percentage}% / NON-BUG {o.non_bug_percentage}%</div></div>
</div>

<h2>Label Distribution</h2>
{html_table(labels)}

<h2>Source Distribution</h2>
{html_table(sources)}

<h2>Project Distribution</h2>
<div class="figure"><img src="{image_src(figure_paths["projects"])}" alt="Top 20 projects"></div>
{html_table(projects.head(25))}

<h2>Text Statistics</h2>
<p><code>model_text</code> is the model-ready text formed from title and description only.</p>
{html_table(text_stats)}
<div class="figure"><img src="{image_src(figure_paths["text"])}" alt="Text length distribution"></div>
{html_table(text_distribution)}

<h2>Missingness</h2>
<p>Metadata missingness is source-specific and is not automatically treated as data corruption.</p>
<div class="figure"><img src="{image_src(figure_paths["missingness"])}" alt="Missingness"></div>
{html_table(missingness)}

<h2>Temporal Distribution</h2>
<div class="figure"><img src="{image_src(figure_paths["temporal"])}" alt="Temporal distribution"></div>
{html_table(yearly)}

<h2>Stage 2 Implications</h2>
<p>The Parquet artifact is suitable for Stage 2 preprocessing validation: schema checks, composite identity validation, label validation, timestamp parsing, missingness analysis, and model text validation. It does not create agent-audited labels; that belongs to later architecture stages.</p>
</body>
</html>
"""
    REPORT_PATH.write_text(html, encoding="utf-8")


def run_pipeline(dataset_path: Path) -> None:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Parquet dataset not found: {dataset_path}")

    print(f"Generating Parquet-based EDA from {dataset_path}")
    overview, labels = compute_overview(dataset_path)
    sources = compute_group_distribution(dataset_path, "source")
    projects = compute_group_distribution(dataset_path, "project")
    text_stats, text_distribution = compute_text_statistics(dataset_path)
    missingness = compute_missingness(dataset_path)
    yearly, monthly = compute_temporal_distribution(dataset_path)

    write_table(overview, "basic_overview.csv")
    write_table(labels, "label_distribution.csv")
    write_table(sources, "source_distribution.csv")
    write_table(projects, "project_distribution.csv")
    write_table(text_stats, "text_statistics.csv")
    write_table(text_distribution, "text_length_distribution.csv")
    write_table(missingness, "missingness.csv")
    write_table(yearly, "temporal_distribution_yearly.csv")
    write_table(monthly, "temporal_distribution_monthly.csv")

    figure_paths = save_figures(projects, missingness, yearly, text_distribution)
    write_report(
        overview,
        labels,
        sources,
        projects,
        text_stats,
        text_distribution,
        missingness,
        yearly,
        figure_paths,
    )
    print(f"EDA report written to {REPORT_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Parquet-based EDA.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Canonical Parquet dataset. Default: {DEFAULT_DATASET}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.dataset)
