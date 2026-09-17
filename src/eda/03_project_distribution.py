import pandas as pd
import matplotlib.pyplot as plt
from src.database.connection import get_engine

engine = get_engine()

query = """
SELECT
    project,
    COUNT(*) AS records,
    COUNT(*) FILTER (WHERE original_label = 'bug') AS bugs,
    COUNT(*) FILTER (WHERE original_label = 'non-bug') AS non_bugs,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE original_label = 'bug')
        / COUNT(*), 2
    ) AS bug_percentage
FROM agenttriage.canonical_dataset
GROUP BY project
ORDER BY records DESC;
"""

df = pd.read_sql(query, engine)

print("\n===== PROJECT DISTRIBUTION =====")
print(df.to_string(index=False))

df.to_csv(
    "outputs/tables/project_distribution.csv",
    index=False
)

# Top 20 projects
top = df.head(20).sort_values("records")

plt.figure(figsize=(10, 8))

plt.barh(
    top["project"],
    top["records"]
)

plt.xlabel("Number of Issues")
plt.ylabel("Project")
plt.title("Top 20 Projects by Number of Issues")

plt.tight_layout()
plt.savefig(
    "outputs/figures/top_20_projects.png",
    dpi=300
)

plt.close()