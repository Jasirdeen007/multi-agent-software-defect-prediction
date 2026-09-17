import pandas as pd
from src.database.connection import get_engine

engine = get_engine()

query = """
SELECT
    COUNT(*) AS total_records,
    COUNT(DISTINCT (source, project, issue_id)) AS unique_issues,
    COUNT(DISTINCT source) AS sources,
    COUNT(DISTINCT project) AS projects,
    COUNT(*) FILTER (WHERE original_label = 'bug') AS bugs,
    COUNT(*) FILTER (WHERE original_label = 'non-bug') AS non_bugs,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE original_label = 'bug') / COUNT(*), 2
    ) AS bug_percentage,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE original_label = 'non-bug') / COUNT(*), 2
    ) AS non_bug_percentage
FROM agenttriage.canonical_dataset;
"""

df = pd.read_sql(query, engine)

print("\n===== BASIC DATASET OVERVIEW =====")
print(df.to_string(index=False))

df.to_csv(
    "outputs/tables/basic_overview.csv",
    index=False
)