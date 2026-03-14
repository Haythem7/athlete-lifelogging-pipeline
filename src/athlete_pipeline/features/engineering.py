import numpy as np
import pandas as pd


def winsorize(series, low=0.01, high=0.99):
    return series.clip(lower=series.quantile(low), upper=series.quantile(high))


def norm(series):
    mn = series.min()
    mx = series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


def add_acwr(df):
    parts = []
    for _, group in df.groupby("participant_id"):
        group = group.sort_values("date").copy()
        load = group["srpe_load"].fillna(0) if "srpe_load" in group.columns else pd.Series(0, index=group.index)
        group["acwr"] = (
            load.rolling(7, min_periods=1).mean()
            / load.rolling(28, min_periods=1).mean().replace(0, np.nan)
        )
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def add_temporal_features(df, rolling_cols=None):
    if rolling_cols is None:
        rolling_cols = [
            "srpe_load",
            "total_steps",
            "resting_hr",
            "wellness_score",
            "sleep_minutes",
        ]

    parts = []
    for _, group in df.groupby("participant_id"):
        group = group.sort_values("date").copy()
        for col in [c for c in rolling_cols if c in group.columns]:
            group[f"{col}_r7"] = group[col].rolling(7, min_periods=1).mean()
            group[f"{col}_r28"] = group[col].rolling(28, min_periods=1).mean()
            group[f"{col}_lag1"] = group[col].shift(1)
        parts.append(group)

    out = pd.concat(parts, ignore_index=True)
    out["day_of_week"] = pd.to_datetime(out["date"]).dt.dayofweek
    num_cols = out.select_dtypes(include=np.number).columns
    out[num_cols] = out[num_cols].fillna(out[num_cols].median())
    return out


def build_perf_index(df_clean):
    sleep_score = norm(df_clean["sleep_score"]) if "sleep_score" in df_clean.columns else norm(
        df_clean.get("sleep_minutes", pd.Series(0, index=df_clean.index))
    )
    rhr_score = 1 - norm(df_clean["resting_hr"]) if "resting_hr" in df_clean.columns else pd.Series(
        0.5, index=df_clean.index
    )
    recovery = 0.6 * sleep_score + 0.4 * rhr_score

    wellness = norm(df_clean["wellness_score"]) if "wellness_score" in df_clean.columns else pd.Series(
        0.5, index=df_clean.index
    )

    acwr = df_clean.get("acwr", pd.Series(1.0, index=df_clean.index))
    load = acwr.apply(
        lambda x: 1.0 if pd.notna(x) and 0.8 <= x <= 1.3
        else max(0, 1 - abs(x - 1.05) / 1.05) if pd.notna(x)
        else 0.5
    )

    activity = norm(df_clean["total_steps"].clip(0, 25000)) if "total_steps" in df_clean.columns else pd.Series(
        0.5, index=df_clean.index
    )

    df_clean = df_clean.copy()
    df_clean["perf_index"] = (0.30 * recovery + 0.30 * wellness + 0.25 * load + 0.15 * activity) * 100
    return df_clean
