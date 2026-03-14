import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import RobustScaler


EXCLUDE = [
    "date",
    "participant_id",
    "gender",
    "group",
    "activity",
    "has_injury",
    "is_major",
    "injury_next7d",
    "perf_index",
]


def add_injury_target(df):
    parts = []
    for _, group in df.groupby("participant_id"):
        group = group.sort_values("date").copy()
        group["injury_next7d"] = (
            group["has_injury"].shift(-7).rolling(7, min_periods=1).max().fillna(0).astype(int)
        )
        parts.append(group)
    return np.array(parts, dtype=object), parts


def train_injury_model(df_clean, test_size=0.2, random_state=42, threshold=0.40):
    parts_arr, parts = add_injury_target(df_clean)
    _ = parts_arr  # keep compatibility if this helper evolves
    df_model = pd.concat(parts, ignore_index=True)

    feats = [
        col
        for col in df_model.select_dtypes(include=np.number).columns
        if col not in EXCLUDE
    ]

    X = df_model[feats].fillna(0)
    y = df_model["injury_next7d"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    k_neighbors = min(5, max(1, int(y_tr.sum()) - 1))
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_tr_s, y_tr = smote.fit_resample(X_tr_s, y_tr)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_tr_s, y_tr)

    y_prob = model.predict_proba(X_te_s)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    cv_scores = cross_val_score(
        model,
        scaler.fit_transform(X),
        y,
        cv=StratifiedKFold(5, shuffle=True, random_state=random_state),
        scoring="roc_auc",
    )

    return {
        "df_model": df_model,
        "features": feats,
        "model": model,
        "scaler": scaler,
        "y_true": y_te,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "classification_report": classification_report(
            y_te, y_pred, target_names=["No injury", "Injury"], zero_division=0
        ),
        "roc_auc": roc_auc_score(y_te, y_prob),
        "cv_roc_auc_mean": cv_scores.mean(),
        "cv_roc_auc_std": cv_scores.std(),
    }
