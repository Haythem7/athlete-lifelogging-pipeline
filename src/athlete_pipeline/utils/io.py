import json
from pathlib import Path

import pandas as pd


def load_json(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path, **kwargs):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def load_participants(path):
    df = pd.read_excel(path, header=1)
    df.columns = [
        "participant_id",
        "age",
        "height_cm",
        "gender",
        "group",
        "max_hr",
        "date_5km",
        "min_5km",
        "sec_5km",
        "stride_walk",
        "stride_run",
    ]
    df = df[df["participant_id"].str.startswith("p", na=False)].copy()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["max_hr"] = pd.to_numeric(df["max_hr"], errors="coerce")
    df["max_hr"] = df.apply(
        lambda row: row["max_hr"] if pd.notna(row["max_hr"]) else 220 - row["age"], axis=1
    )
    return df.set_index("participant_id")
