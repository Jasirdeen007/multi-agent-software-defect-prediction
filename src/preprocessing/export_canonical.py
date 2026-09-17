"""Export the PostgreSQL canonical dataset to Parquet/CSV for reproducible EDA."""
import os
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine

OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

url = os.environ.get("BUGHUB_DATABASE_URL")
if not url:
    raise RuntimeError("Set BUGHUB_DATABASE_URL, e.g. postgresql+psycopg2://user:password@localhost:5432/bughub")

engine = create_engine(url)
df = pd.read_sql("SELECT * FROM agenttriage.canonical_dataset", engine)
df.to_parquet(OUT / "canonical_dataset.parquet", index=False)
df.to_csv(OUT / "canonical_dataset.csv", index=False)
print(f"Exported {len(df):,} records")
