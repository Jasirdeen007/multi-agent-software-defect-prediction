from pathlib import Path
import pandas as pd
from src.database.connection import get_engine

REPORT = Path("outputs/reports/EDA_Report.html")

def table(df):
    return df.to_html(
        index=False,
        classes="data-table",
        border=0
    )

def main():
    engine = get_engine()

    # -------------------------
    # 1. Overview
    # -------------------------
    overview = pd.read_sql("""
        SELECT
            COUNT(*) AS total_records,
            COUNT(DISTINCT (source, project, issue_id)) AS unique_issues,
            COUNT(DISTINCT source) AS sources,
            COUNT(DISTINCT project) AS projects,
            COUNT(*) FILTER (
                WHERE original_label = 'bug'
            ) AS bugs,
            COUNT(*) FILTER (
                WHERE original_label = 'non-bug'
            ) AS non_bugs
        FROM agenttriage.canonical_dataset;
    """, engine)

    # -------------------------
    # 2. Source distribution
    # -------------------------
    source = pd.read_sql("""
        SELECT
            source,
            COUNT(*) AS records,
            COUNT(*) FILTER (
                WHERE original_label = 'bug'
            ) AS bugs,
            COUNT(*) FILTER (
                WHERE original_label = 'non-bug'
            ) AS non_bugs
        FROM agenttriage.canonical_dataset
        GROUP BY source
        ORDER BY records DESC;
    """, engine)

    # -------------------------
    # 3. Project distribution
    # -------------------------
    projects = pd.read_sql("""
        SELECT
            project,
            COUNT(*) AS records,
            COUNT(*) FILTER (
                WHERE original_label = 'bug'
            ) AS bugs,
            COUNT(*) FILTER (
                WHERE original_label = 'non-bug'
            ) AS non_bugs
        FROM agenttriage.canonical_dataset
        GROUP BY project
        ORDER BY records DESC;
    """, engine)

    # -------------------------
    # 4. Text statistics
    # -------------------------
    text = pd.read_sql("""
        SELECT
            COUNT(*) AS records,
            ROUND(AVG(LENGTH(model_text)), 2) AS avg_chars,
            ROUND(STDDEV(LENGTH(model_text)), 2) AS std_chars,
            MIN(LENGTH(model_text)) AS min_chars,
            PERCENTILE_CONT(0.25)
                WITHIN GROUP (
                    ORDER BY LENGTH(model_text)
                ) AS p25_chars,
            PERCENTILE_CONT(0.50)
                WITHIN GROUP (
                    ORDER BY LENGTH(model_text)
                ) AS median_chars,
            PERCENTILE_CONT(0.75)
                WITHIN GROUP (
                    ORDER BY LENGTH(model_text)
                ) AS p75_chars,
            MAX(LENGTH(model_text)) AS max_chars
        FROM agenttriage.canonical_dataset
        WHERE model_text IS NOT NULL;
    """, engine)

    # -------------------------
    # 5. Missingness
    # -------------------------
    missing = pd.read_sql("""
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
    """, engine)

    # -------------------------
    # 6. Temporal distribution
    # -------------------------
    temporal = pd.read_sql("""
        SELECT
            EXTRACT(YEAR FROM created_at)::int AS year,
            COUNT(*) AS records,
            COUNT(*) FILTER (WHERE original_label = 'bug') AS bugs,
            COUNT(*) FILTER (WHERE original_label = 'non-bug') AS non_bugs
        FROM agenttriage.canonical_dataset
        WHERE created_at IS NOT NULL
        GROUP BY year
        ORDER BY year;
    """, engine)

    o = overview.iloc[0]
    bug_pct = o.bugs / o.total_records * 100
    non_bug_pct = o.non_bugs / o.total_records * 100

    missing_long = missing.T.reset_index()
    missing_long.columns = ["field", "missing_count"]
    missing_long = missing_long[missing_long["field"] != "total_records"]
    missing_long["missing_percentage"] = (
        missing_long["missing_count"] / o.total_records * 100
    ).round(2)

    # Build HTML Report (queries run directly against PostgreSQL via engine)
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>EDA Report — Multi-Agent Software Defect Prediction</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 40px; color: #222; }}
h1, h2, h3 {{ color: #1a365d; }}
.data-table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
.data-table th, .data-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
.data-table th {{ background-color: #f2f4f8; font-weight: bold; }}
.stat-box {{ background: #f7fafc; border-left: 4px solid #3182ce; padding: 15px; margin: 15px 0; }}
</style>
</head>
<body>
<h1>EDA Report — Multi-Agent Software Defect Prediction</h1>
<div class="stat-box">
  <p><strong>Total Canonical Records:</strong> {o.total_records:,}</p>
  <p><strong>Unique Issues:</strong> {o.unique_issues:,}</p>
  <p><strong>Sources:</strong> {o.sources} (Bugzilla, GitHub, JIRA)</p>
  <p><strong>Projects:</strong> {o.projects}</p>
  <p><strong>Class Split:</strong> BUG = {o.bugs:,} ({bug_pct:.2f}%) | NON-BUG = {o.non_bugs:,} ({non_bug_pct:.2f}%)</p>
</div>

<h2>1. Source Distribution</h2>
{table(source)}

<h2>2. Text Length Statistics (model_text)</h2>
{table(text)}

<h2>3. Missingness by Canonical Field</h2>
{table(missing_long)}

<h2>4. Temporal Distribution (by Year)</h2>
{table(temporal)}

<h2>5. Project Distribution (Top 25)</h2>
{table(projects.head(25))}

<h2>6. Next Research Phase</h2>
<p>Cleaned canonical dataset in <code>agenttriage.canonical_dataset</code> feeds into Knowledge Graph construction and the Multi-Agent Label Auditing workflow.</p>
</body>
</html>
"""

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(html, encoding="utf-8")
    print(f"\nEDA report generated: {REPORT}")

if __name__ == "__main__":
    main()