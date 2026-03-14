import json

import numpy as np
import pandas as pd


def parse_minute_series(records, col):
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["dateTime"]).dt.date
    df[col] = pd.to_numeric(df["value"], errors="coerce")
    return df.groupby("date")[col].sum().reset_index()


def parse_resting_hr(records):
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "date": pd.to_datetime(record["dateTime"]).date(),
                "resting_hr": record["value"]["value"],
            }
            for record in records
        ]
    )


def parse_hr_zones(records):
    if not records:
        return pd.DataFrame()

    rows = []
    for record in records:
        zones = record["value"]["valuesInZones"]
        rows.append(
            {
                "date": pd.to_datetime(record["dateTime"]).date(),
                "hr_fat_burn_min": zones.get("IN_DEFAULT_ZONE_1", 0),
                "hr_cardio_min": zones.get("IN_DEFAULT_ZONE_2", 0),
                "hr_peak_min": zones.get("IN_DEFAULT_ZONE_3", 0),
            }
        )
    return pd.DataFrame(rows)


def parse_sleep(records):
    if not records:
        return pd.DataFrame()

    rows = []
    for record in records:
        summary = record.get("levels", {}).get("summary", {})
        rows.append(
            {
                "date": pd.to_datetime(record["dateOfSleep"]).date(),
                "sleep_minutes": record.get("minutesAsleep", np.nan),
                "sleep_efficiency": record.get("efficiency", np.nan),
                "sleep_deep_min": summary.get("deep", {}).get("minutes", 0),
                "sleep_rem_min": summary.get("rem", {}).get("minutes", 0),
            }
        )
    return pd.DataFrame(rows).groupby("date").first().reset_index()


def parse_exercise(records):
    if not records:
        return pd.DataFrame()

    rows = [
        {
            "date": pd.to_datetime(record["startTime"]).date(),
            "activity": record.get("activityName", ""),
            "cal_exercise": record.get("calories", 0),
            "duration_min": round(record.get("duration", 0) / 60000, 1),
            "avg_hr": record.get("averageHeartRate", np.nan),
        }
        for record in records
    ]
    return pd.DataFrame(rows).groupby("date").agg(
        n_sessions=("activity", "count"),
        cal_exercise=("cal_exercise", "sum"),
        duration_min=("duration_min", "sum"),
        avg_hr=("avg_hr", "mean"),
    ).reset_index()


def parse_injury(df_raw):
    if df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["effective_time_frame"], utc=True, errors="coerce").dt.date

    def has_injury(raw_value):
        try:
            injury = json.loads(str(raw_value).replace("'", '"'))
            if injury:
                is_major = int("major" in str(injury.get("severity", "")).lower())
                return 1, is_major
            return 0, 0
        except Exception:
            return 0, 0

    df[["has_injury", "is_major"]] = df["injuries"].apply(lambda value: pd.Series(has_injury(value)))
    return df.groupby("date").agg(
        has_injury=("has_injury", "max"),
        is_major=("is_major", "max"),
    ).reset_index()


def parse_srpe(df_raw):
    if df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["end_date_time"], utc=True, errors="coerce").dt.date
    df["rpe"] = pd.to_numeric(df["perceived_exertion"], errors="coerce")
    df["dur"] = pd.to_numeric(df["duration_min"], errors="coerce")
    df["load"] = df["rpe"] * df["dur"]

    return df.groupby("date").agg(
        mean_rpe=("rpe", "mean"),
        srpe_load=("load", "sum"),
        srpe_duration=("dur", "sum"),
    ).reset_index()


def parse_wellness(df_raw):
    if df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["effective_time_frame"], utc=True, errors="coerce").dt.date

    for col in ["fatigue", "mood", "readiness", "sleep_quality", "soreness", "stress"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["wellness_score"] = (
        (6 - df["fatigue"])
        + df["mood"]
        + (df["readiness"] / 2)
        + df["sleep_quality"]
        + (6 - df["soreness"])
        + (6 - df["stress"])
    )

    keep = [
        "date",
        "fatigue",
        "mood",
        "readiness",
        "sleep_quality",
        "soreness",
        "stress",
        "wellness_score",
    ]
    return df[[col for col in keep if col in df.columns]].drop_duplicates("date")


def parse_reporting(df_raw):
    if df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce").dt.date
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df["fluids"] = pd.to_numeric(df["glasses_of_fluid"], errors="coerce")
    df["alcohol"] = df["alcohol_consumed"].str.lower().map({"yes": 1, "no": 0})
    df["n_meals"] = df["meals"].str.count(",").fillna(0).astype(int) + 1
    return df[["date", "weight", "fluids", "alcohol", "n_meals"]].dropna(subset=["date"])


def parse_sleep_score(df_raw):
    if df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["date"] = df["timestamp"].dt.date
    if "overall_score" in df.columns:
        df = df.rename(columns={"overall_score": "sleep_score"})

    needed = [col for col in ["date", "sleep_score"] if col in df.columns]
    if set(needed) != {"date", "sleep_score"}:
        return pd.DataFrame()
    return df[needed].dropna(subset=["date"])
