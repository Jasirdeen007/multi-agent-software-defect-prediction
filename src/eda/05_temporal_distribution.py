import pandas as pd
import matplotlib.pyplot as plt
from src.database.connection import get_engine

engine = get_engine()

query = """
SELECT
    DATE_TRUNC('month', created_at) AS month,
    COUNT(*) AS records,
    COUNT(*) FILTER (
        WHERE original_label = 'bug'
    ) AS bugs,
    COUNT(*) FILTER (
        WHERE original_label = 'non-bug'
    ) AS non_bugs
FROM agenttriage.canonical_dataset
WHERE created_at IS NOT NULL
GROUP BY month
ORDER BY month;
"""

df = pd.read_sql(query, engine)

print("\n===== TEMPORAL DISTRIBUTION =====")
print(df.to_string(index=False))

df.to_csv(
    "outputs/tables/monthly_distribution.csv",
    index=False
)

plt.figure(figsize=(11, 5))

plt.plot(
    df["month"],
    df["records"]
)

plt.xlabel("Month")
plt.ylabel("Number of Issues")
plt.title("Issue Creation Over Time")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "outputs/figures/temporal_distribution.png",
    dpi=300
)

plt.close()