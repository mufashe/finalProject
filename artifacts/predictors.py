# artifacts/predictors.py
import numpy as np
import pandas as pd
from .ml_registry import load_pipeline, load_lstm

SUBJECTS = ["Kinyarwanda", "English", "Mathematics", "Science", "Social_Studies", "Creative_Arts"]
GRADES_ALL = ["P1", "P2", "P3", "P4", "P5", "P6"]
GRADES_SEQ = ["P1", "P2", "P3", "P4", "P5"]

SUBJECT_COLS_ALL = [f"{s}_{g}" for g in GRADES_ALL for s in SUBJECTS]
CAT_COLS = ["Gender", "School_Location", "Residence_Location", "Has_Electricity", "Parental_Education_Level"]
REQ_PIPE_COLS = SUBJECT_COLS_ALL + CAT_COLS

SEQ_ORDER = [f"{s}_{g}" for g in GRADES_SEQ for s in SUBJECTS]  # exact order for LSTM


def _require_columns(df: pd.DataFrame, required):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def predict_with_pipeline(df: pd.DataFrame, which: str):
    """
    which: 'primary_rf' or 'primary_logreg'
    Returns: df with Pred, Prob columns
    """
    _require_columns(df, REQ_PIPE_COLS)
    pipe = load_pipeline(which)
    proba = pipe.predict_proba(df[REQ_PIPE_COLS])[:, 1]
    pred = (proba >= 0.5).astype(int)
    out = df.copy()
    out["Pred"] = pred
    out["Prob"] = proba
    return out


def predict_with_lstm(df: pd.DataFrame):
    """
    Uses saved Keras model + saved preprocessor pack.
    Requires P1–P5 subject columns in SEQ_ORDER.
    """
    _require_columns(df, SEQ_ORDER)
    model, pack = load_lstm()
    imputer = pack["imputer"]
    scaler = pack["scaler"]
    timesteps = pack["timesteps"]
    n_subjects = pack["n_subjects"]

    M = df[SEQ_ORDER].values  # shape (N, 5*6)
    M_imp = imputer.transform(M)
    M_scaled = scaler.transform(M_imp)
    X = M_scaled.reshape(len(df), timesteps, n_subjects)

    proba = model.predict(X, verbose=0).ravel()
    pred = (proba >= 0.5).astype(int)

    out = df.copy()
    out["Pred"] = pred
    out["Prob"] = proba
    return out
