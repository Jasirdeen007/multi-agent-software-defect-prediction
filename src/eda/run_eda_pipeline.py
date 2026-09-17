"""
EDA Pipeline Runner & Comprehensive Report Generator
Multi-Agent-SDP Project

Runs all EDA analyses against agenttriage.canonical_dataset:
  1. Overview & Label Distribution
  2. Text Statistics (model_text)
  3. Project Distribution (Top projects & plot)
  4. Missingness Analysis (Missing rates & plot)
  5. Temporal Distribution (Yearly & Monthly trends & plot)
  6. Compiles and generates the complete HTML EDA report

Outputs:
  - outputs/tables/*.csv
  - outputs/figures/*.png
  - outputs/reports/EDA_Report.html
"""

import base64
import sys
from pathlib import Path

# Add project root to sys.path so it can run both directly and as module
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import matplotlib.pyplot as plt
from src.database.connection import get_engine

REPORT_PATH = Path("outputs/reports/EDA_Report.html")
TABLES_DIR = Path("outputs/tables")
FIGURES_DIR = Path("outputs/figures")


def df_to_html_table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, classes="data-table", border=0)


def img_to_base64_src(img_path: Path) -> str:
    if not img_path.exists():
        return ""
    encoded = base64.b64encode(img_path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def run_pipeline():
    print("=" * 60)
    print("STARTING FULL EDA PIPELINE & REPORT GENERATION")
    print("=" * 60)

    # Ensure output directories exist
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    engine = get_engine()

    # -------------------------------------------------------------
    # 1. Dataset Overview & Label Distribution
    # -------------------------------------------------------------
    print("\n[1/5] Extracting Dataset Overview & Label Distribution...")
    overview_query = """
        SELECT
            COUNT(*) AS total_records,
            COUNT(DISTINCT (source, project, issue_id)) AS unique_issues,
            COUNT(DISTINCT source) AS sources,
            COUNT(DISTINCT project) AS projects,
            COUNT(*) FILTER (WHERE original_label = 'bug') AS bugs,
            COUNT(*) FILTER (WHERE original_label = 'non-bug') AS non_bugs,
            ROUND(100.0 * COUNT(*) FILTER (WHERE original_label = 'bug') / COUNT(*), 2) AS bug_percentage,
            ROUND(100.0 * COUNT(*) FILTER (WHERE original_label = 'non-bug') / COUNT(*), 2) AS non_bug_percentage
        FROM agenttriage.canonical_dataset;
    """
    df_overview = pd.read_sql(overview_query, engine)
    df_overview.to_csv(TABLES_DIR / "basic_overview.csv", index=False)
    o = df_overview.iloc[0]

    # Source breakdown
    source_query = """
        SELECT
            source,
            COUNT(*) AS records,
            COUNT(*) FILTER (WHERE original_label = 'bug') AS bugs,
            COUNT(*) FILTER (WHERE original_label = 'non-bug') AS non_bugs,
            ROUND(100.0 * COUNT(*) FILTER (WHERE original_label = 'bug') / COUNT(*), 2) AS bug_percentage
        FROM agenttriage.canonical_dataset
        GROUP BY source
        ORDER BY records DESC;
    """
    df_sources = pd.read_sql(source_query, engine)
    df_sources.to_csv(TABLES_DIR / "source_distribution.csv", index=False)

    # -------------------------------------------------------------
    # 2. Text Statistics (model_text = title + description)
    # -------------------------------------------------------------
    print("[2/5] Computing Text Length Statistics...")
    text_query = """
        SELECT
            COUNT(*) AS records,
            ROUND(AVG(LENGTH(model_text)), 2) AS avg_chars,
            ROUND(STDDEV(LENGTH(model_text)), 2) AS std_chars,
            MIN(LENGTH(model_text)) AS min_chars,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY LENGTH(model_text)) AS p25_chars,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY LENGTH(model_text)) AS median_chars,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY LENGTH(model_text)) AS p75_chars,
            MAX(LENGTH(model_text)) AS max_chars
        FROM agenttriage.canonical_dataset
        WHERE model_text IS NOT NULL;
    """
    df_text = pd.read_sql(text_query, engine)
    df_text.to_csv(TABLES_DIR / "text_statistics.csv", index=False)

    # -------------------------------------------------------------
    # 3. Project Distribution & Chart
    # -------------------------------------------------------------
    print("[3/5] Computing Project Distribution & Visualizations...")
    project_query = """
        SELECT
            project,
            COUNT(*) AS records,
            COUNT(*) FILTER (WHERE original_label = 'bug') AS bugs,
            COUNT(*) FILTER (WHERE original_label = 'non-bug') AS non_bugs,
            ROUND(100.0 * COUNT(*) FILTER (WHERE original_label = 'bug') / COUNT(*), 2) AS bug_percentage
        FROM agenttriage.canonical_dataset
        GROUP BY project
        ORDER BY records DESC;
    """
    df_projects = pd.read_sql(project_query, engine)
    df_projects.to_csv(TABLES_DIR / "project_distribution.csv", index=False)

    # Plot top 20 projects
    top_projects = df_projects.head(20).sort_values("records")
    plt.figure(figsize=(10, 8))
    plt.barh(top_projects["project"], top_projects["records"], color="#2b6cb0")
    plt.xlabel("Number of Issues")
    plt.ylabel("Project")
    plt.title("Top 20 Projects by Total Issues")
    plt.tight_layout()
    proj_img_path = FIGURES_DIR / "top_20_projects.png"
    plt.savefig(proj_img_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 4. Missingness Analysis & Chart
    # -------------------------------------------------------------
    print("[4/5] Analyzing Missingness Across Canonical Fields...")
    missing_query = """
        SELECT
            COUNT(*) AS total_records,
            COUNT(*) FILTER (WHERE title IS NULL OR BTRIM(title) = '') AS missing_title,
            COUNT(*) FILTER (WHERE description IS NULL OR BTRIM(description) = '') AS missing_description,
            COUNT(*) FILTER (WHERE component IS NULL OR BTRIM(component) = '') AS missing_component,
            COUNT(*) FILTER (WHERE severity IS NULL OR BTRIM(severity) = '') AS missing_severity,
            COUNT(*) FILTER (WHERE priority IS NULL OR BTRIM(priority) = '') AS missing_priority,
            COUNT(*) FILTER (WHERE status IS NULL OR BTRIM(status) = '') AS missing_status,
            COUNT(*) FILTER (WHERE resolution IS NULL OR BTRIM(resolution) = '') AS missing_resolution,
            COUNT(*) FILTER (WHERE created_at IS NULL) AS missing_created_at,
            COUNT(*) FILTER (WHERE updated_at IS NULL) AS missing_updated_at
        FROM agenttriage.canonical_dataset;
    """
    df_missing_raw = pd.read_sql(missing_query, engine)
    missing_long = df_missing_raw.T.reset_index()
    missing_long.columns = ["field", "missing_count"]
    missing_long = missing_long[missing_long["field"] != "total_records"].copy()
    missing_long["missing_percentage"] = (
        missing_long["missing_count"] / o.total_records * 100
    ).round(2)
    missing_long.to_csv(TABLES_DIR / "missingness.csv", index=False)

    # Plot missingness
    plt.figure(figsize=(10, 5))
    plt.bar(missing_long["field"], missing_long["missing_percentage"], color="#c53030")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Missing Values (%)")
    plt.title("Missing Values by Canonical Field")
    plt.tight_layout()
    missing_img_path = FIGURES_DIR / "missingness.png"
    plt.savefig(missing_img_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 5. Temporal Distribution & Chart
    # -------------------------------------------------------------
    print("[5/5] Analyzing Temporal Trends...")
    temporal_year_query = """
        SELECT
            EXTRACT(YEAR FROM created_at)::int AS year,
            COUNT(*) AS records,
            COUNT(*) FILTER (WHERE original_label = 'bug') AS bugs,
            COUNT(*) FILTER (WHERE original_label = 'non-bug') AS non_bugs,
            ROUND(100.0 * COUNT(*) FILTER (WHERE original_label = 'bug') / COUNT(*), 2) AS bug_percentage
        FROM agenttriage.canonical_dataset
        WHERE created_at IS NOT NULL
        GROUP BY year
        ORDER BY year;
    """
    df_temporal_year = pd.read_sql(temporal_year_query, engine)
    df_temporal_year.to_csv(TABLES_DIR / "temporal_distribution.csv", index=False)

    # Plot yearly trends (filtering out anomalous two-digit year parse artifacts < 1990 for clear timeline)
    valid_years = df_temporal_year[df_temporal_year["year"] >= 1990]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    color_bar = "#3182ce"
    color_line = "#e53e3e"

    ax1.set_xlabel("Year")
    ax1.set_ylabel("Total Issues", color=color_bar)
    ax1.bar(valid_years["year"], valid_years["records"], color=color_bar, alpha=0.7, label="Total Records")
    ax1.tick_params(axis="y", labelcolor=color_bar)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Bug Percentage (%)", color=color_line)
    ax2.plot(valid_years["year"], valid_years["bug_percentage"], color=color_line, marker="o", linewidth=2, label="Bug %")
    ax2.tick_params(axis="y", labelcolor=color_line)

    plt.title("Issue Volume and Bug Percentage by Year (1997 - 2023)")
    fig.tight_layout()
    temporal_img_path = FIGURES_DIR / "temporal_distribution.png"
    plt.savefig(temporal_img_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # Generate HTML Report
    # -------------------------------------------------------------
    print("\nGenerating final self-contained HTML EDA report...")

    proj_b64 = img_to_base64_src(proj_img_path)
    missing_b64 = img_to_base64_src(missing_img_path)
    temporal_b64 = img_to_base64_src(temporal_img_path)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EDA Report — Multi-Agent Software Defect Prediction</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        line-height: 1.6;
        color: #1a202c;
        max-width: 1200px;
        margin: 0 auto;
        padding: 30px 20px;
        background-color: #f7fafc;
    }}
    h1 {{
        color: #2b6cb0;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 12px;
    }}
    h2 {{
        color: #2d3748;
        margin-top: 35px;
        border-bottom: 1px solid #edf2f7;
        padding-bottom: 8px;
    }}
    .stat-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 16px;
        margin: 25px 0;
    }}
    .card {{
        background: #ffffff;
        padding: 18px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-top: 4px solid #3182ce;
    }}
    .card-title {{
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #718096;
    }}
    .card-value {{
        font-size: 1.8rem;
        font-weight: 700;
        color: #2d3748;
        margin-top: 6px;
    }}
    .data-table {{
        width: 100%;
        border-collapse: collapse;
        background: #ffffff;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin: 15px 0;
    }}
    .data-table th, .data-table td {{
        padding: 12px 16px;
        text-align: left;
        border-bottom: 1px solid #edf2f7;
    }}
    .data-table th {{
        background-color: #ebf8ff;
        color: #2b6cb0;
        font-weight: 600;
    }}
    .data-table tr:hover {{
        background-color: #f7fafc;
    }}
    .img-container {{
        text-align: center;
        margin: 20px 0;
        background: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}
    .img-container img {{
        max-width: 100%;
        height: auto;
        border-radius: 4px;
    }}
    .badge {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }}
    .badge-bug {{ background: #fed7d7; color: #9b2c2c; }}
    .badge-non-bug {{ background: #c6f6d5; color: #22543d; }}
    .info-box {{
        background-color: #ebf8ff;
        border-left: 4px solid #3182ce;
        padding: 15px;
        margin: 20px 0;
        border-radius: 4px;
    }}
</style>
</head>
<body>

<h1>Exploratory Data Analysis Report</h1>
<p><strong>Project:</strong> Multi-Agent Software Defect Prediction (Multi-Agent-SDP)</p>
<p><strong>Dataset Table:</strong> <code>agenttriage.canonical_dataset</code></p>

<div class="stat-grid">
    <div class="card">
        <div class="card-title">Total Canonical Records</div>
        <div class="card-value">{o.total_records:,}</div>
    </div>
    <div class="card">
        <div class="card-title">Unique Issues</div>
        <div class="card-value">{o.unique_issues:,}</div>
    </div>
    <div class="card">
        <div class="card-title">Bug Count</div>
        <div class="card-value"><span class="badge badge-bug">{o.bugs:,} ({o.bug_percentage}%)</span></div>
    </div>
    <div class="card">
        <div class="card-title">Non-Bug Count</div>
        <div class="card-value"><span class="badge badge-non-bug">{o.non_bugs:,} ({o.non_bug_percentage}%)</span></div>
    </div>
    <div class="card">
        <div class="card-title">Sources & Projects</div>
        <div class="card-value">{o.sources} Sources / {o.projects} Projs</div>
    </div>
</div>

<h2>1. Source Distribution</h2>
<p>Breakdown across BugHub source repositories (Bugzilla, GitHub, JIRA):</p>
{df_to_html_table(df_sources)}

<h2>2. Text Length Statistics (model_text)</h2>
<p>Statistics for <code>model_text</code> (Title + Description) measured in character counts:</p>
{df_to_html_table(df_text)}

<h2>3. Project Distribution (Top 25)</h2>
<div class="img-container">
    <img src="{proj_b64}" alt="Top 20 Projects">
</div>
{df_to_html_table(df_projects.head(25))}

<h2>4. Missingness Analysis Across Canonical Fields</h2>
<p>Note: Source-specific missing values are expected (e.g. GitHub does not provide component/severity/priority/resolution fields).</p>
<div class="img-container">
    <img src="{missing_b64}" alt="Missingness Rates">
</div>
{df_to_html_table(missing_long)}

<h2>5. Temporal Distribution</h2>
<div class="img-container">
    <img src="{temporal_b64}" alt="Temporal Distribution">
</div>
{df_to_html_table(df_temporal_year)}

<h2>6. Methodological Context & Next Research Phase</h2>
<div class="info-box">
    <p><strong>Canonical Table:</strong> <code>agenttriage.canonical_dataset</code> (926,461 records)</p>
    <p><strong>Primary Pipeline:</strong> Canonical Dataset &rarr; Knowledge Graph Construction &rarr; Multi-Agent Auditing (Policy, Data, Pattern, Judge) &rarr; DeBERTa-v3 &rarr; XGBoost &rarr; SHAP</p>
</div>

</body>
</html>
"""

    REPORT_PATH.write_text(html_content, encoding="utf-8")
    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETE! Report generated at:\n  {REPORT_PATH.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()

