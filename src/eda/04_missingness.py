import pandas as pd
import matplotlib.pyplot as plt
from src.database.connection import get_engine

engine = get_engine()

query = """
SELECT
    COUNT(*) AS total_records,

    COUNT(*) FILTER (
        WHERE title IS NULL OR BTRIM(title) = ''
    ) AS missing_title,

    COUNT(*) FILTER (
        WHERE description IS NULL OR BTRIM(description) = ''
    ) AS missing_description,

    COUNT(*) FILTER (
        WHERE component IS NULL OR BTRIM(component) = ''
    ) AS missing_component,

    COUNT(*) FILTER (
        WHERE severity IS NULL OR BTRIM(severity) = ''
    ) AS missing_severity,

    COUNT(*) FILTER (
        WHERE priority IS NULL OR BTRIM(priority) = ''
    ) AS missing_priority,

    COUNT(*) FILTER (
        WHERE status IS NULL OR BTRIM(status) = ''
    ) AS missing_status,

    COUNT(*) FILTER (
        WHERE resolution IS NULL OR BTRIM(resolution) = ''
    ) AS missing_resolution,

    COUNT(*) FILTER (
        WHERE created_at IS NULL
    ) AS missing_created_at,

    COUNT(*) FILTER (
        WHERE updated_at IS NULL
    ) AS missing_updated_at

FROM agenttriage.canonical_dataset;
"""

df = pd.read_sql(query, engine)

print("\n===== MISSINGNESS =====")
print(df.to_string(index=False))

# Convert to long format
missing = df.T.reset_index()
missing.columns = ["field", "missing_count"]

missing = missing[
    missing["field"] != "total_records"
]

total = int(df.loc[0, "total_records"])

missing["missing_percentage"] = (
    missing["missing_count"] / total * 100
).round(2)

missing.to_csv(
    "outputs/tables/missingness.csv",
    index=False
)

# Plot
plt.figure(figsize=(10, 5))

plt.bar(
    missing["field"],
    missing["missing_percentage"]
)

plt.xticks(rotation=45, ha="right")
plt.ylabel("Missing Values (%)")
plt.title("Missing Values by Canonical Field")

plt.tight_layout()
plt.savefig(
    "outputs/figures/missingness.png",
    dpi=300
)

plt.close()