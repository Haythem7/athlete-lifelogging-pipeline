import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import RobustScaler


def add_perf_target(df):
    parts = []
    for _, group in df.groupby("participant_id"):
        group = group.sort_values("date").copy()
        group["perf_target"] = group["perf_index"].shift(-1)
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def train_perf_model(df_clean, exclude=None, test_size=0.2, random_state=42):
    if exclude is None:
        exclude = [
            "date",
            "participant_id",
            "gender",
            "group",
            "activity",
            "has_injury",
            "is_major",
            "injury_next7d",
            "perf_index",
            "perf_target",
        ]

    df_model = add_perf_target(df_clean)

    numeric_cols = df_model.select_dtypes(include=np.number).columns
    features = [c for c in numeric_cols if c not in exclude]

    X = df_model[features].fillna(0)
    y = df_model["perf_target"].fillna(df_model["perf_index"])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=random_state,
    )
    model.fit(X_tr_s, y_tr)

    y_pred = model.predict(X_te_s)

    cv_r2 = cross_val_score(model, scaler.fit_transform(X), y, cv=5, scoring="r2")

    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae = mean_absolute_error(y_te, y_pred)
    r2 = r2_score(y_te, y_pred)

    return {
        "df_model": df_model,
        "features": features,
        "model": model,
        "scaler": scaler,
        "y_true": y_te,
        "y_pred": y_pred,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "cv_r2_mean": cv_r2.mean(),
        "cv_r2_std": cv_r2.std(),
    }
