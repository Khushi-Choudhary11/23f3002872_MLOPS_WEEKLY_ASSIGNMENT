"""
Load the four poisoned training-set variants (produced by poison_data.py)
into an isolated BigQuery table and expose them to Feast as timestamped
snapshots, so training can retrieve poisoned features through the same
Feast offline-store path used by the existing pipeline (train.py).

This NEVER touches the existing production table
(mlops-project-500020.iris_feast.iris_features) used by prior weeks.
It writes to a new table: iris_feast.iris_features_week8_poison.

Mechanism: each poisoning level gets its own event_timestamp "snapshot"
(base_ts + level_index hours). All 150 iris_ids are written at every
snapshot -- test rows and un-poisoned train rows carry their original
clean feature values, poisoned train rows carry the corrupted values.
Feast's point-in-time join then returns exactly the right snapshot when
queried with that level's timestamp.
"""

import pandas as pd
from pathlib import Path
from google.cloud import bigquery

PROJECT = "mlops-project-500020"
DATASET = "iris_feast"
TABLE = "iris_features_week8_poison"
FULL_TABLE_ID = f"{PROJECT}.{DATASET}.{TABLE}"

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
POISON_LEVELS = [0, 5, 10, 50]
BASE_TS = pd.Timestamp("2026-08-09T00:00:00")

RAW_PATH = Path("data/raw/iris.csv")
DATA_DIR = Path("data")


def snapshot_ts(level):
    idx = POISON_LEVELS.index(level)
    return BASE_TS + pd.Timedelta(hours=idx)


def load_full_clean():
    df = pd.read_csv(RAW_PATH, encoding="utf-8-sig").reset_index(drop=True)
    df.insert(0, "iris_id", range(len(df)))
    return df


def train_variant_path(level):
    return DATA_DIR / "clean" / "train_0pct.csv" if level == 0 else DATA_DIR / "poisoned" / f"train_{level}pct.csv"


def build_snapshot_rows(full_clean, level):
    """Full 150-row feature table for this level: poisoned values for the
    poisoned train rows, original clean values everywhere else."""
    snap = full_clean[["iris_id"] + FEATURES].copy()
    train_variant = pd.read_csv(train_variant_path(level))
    for feat in FEATURES:
        snap = snap.set_index("iris_id")
        snap.loc[train_variant["iris_id"], feat] = train_variant.set_index("iris_id")[feat]
        snap = snap.reset_index()
    snap["event_timestamp"] = snapshot_ts(level)
    snap["poisoning_level"] = level
    return snap


def main():
    full_clean = load_full_clean()

    frames = [build_snapshot_rows(full_clean, level) for level in POISON_LEVELS]
    combined = pd.concat(frames, ignore_index=True)
    combined["iris_id"] = combined["iris_id"].astype("int64")
    combined["event_timestamp"] = pd.to_datetime(combined["event_timestamp"], utc=True)
    print(f"Prepared {len(combined)} rows ({len(POISON_LEVELS)} snapshots x {len(full_clean)} entities)")
    print(combined.groupby("poisoning_level").size())

    client = bigquery.Client(project=PROJECT)
    dataset_ref = bigquery.DatasetReference(PROJECT, DATASET)
    client.get_dataset(dataset_ref)  # confirms iris_feast dataset already exists, raises if not

    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    job = client.load_table_from_dataframe(combined, FULL_TABLE_ID, job_config=job_config)
    job.result()
    print(f"Loaded {job.output_rows} rows into {FULL_TABLE_ID}")

    table = client.get_table(FULL_TABLE_ID)
    print(f"Table now has {table.num_rows} rows, schema: {[f.name for f in table.schema]}")

    for level in POISON_LEVELS:
        print(f"  level={level:>3}% snapshot_ts={snapshot_ts(level)}")


if __name__ == "__main__":
    main()
