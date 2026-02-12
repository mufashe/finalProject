# artifacts/o_level_predictors.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from django.conf import settings


# -------------------- Locate artifacts --------------------
def _model_dir() -> Path:
    # Put O-Level models under BASE_DIR/models/olevel (same as training script)
    base = Path(getattr(settings, "BASE_DIR", ".")) / "models" / "O-Level"
    if not base.exists():
        raise FileNotFoundError(f"O-Level model directory not found: {base}")
    return base


def _load_meta() -> Dict:
    p = _model_dir() / "olevel_meta.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


META = _load_meta()
SUBJECTS: List[str] = META["subjects"]
YEARS: List[str] = META["years"]
REQ_COLS: List[str] = META["required_columns"]
CLASSES: List[str] = META["classes"]

# Lazy singletons
_SCALER = None
_LE = None
_RF = None
_LSTM = None


def scaler():
    global _SCALER
    if _SCALER is None:
        _SCALER = joblib.load(_model_dir() / "olevel_scaler.pkl")
    return _SCALER


def label_encoder():
    global _LE
    if _LE is None:
        _LE = joblib.load(_model_dir() / "olevel_label_encoder.pkl")
    return _LE


def rf_model():
    global _RF
    if _RF is None:
        _RF = joblib.load(_model_dir() / "olevel_rf.pkl")
    return _RF


def lstm_model():
    global _LSTM
    if _LSTM is None:
        _LSTM = tf.keras.models.load_model(_model_dir() / "olevel_lstm.keras")
    return _LSTM


# -------------------- Preprocessing --------------------
def _verify_columns(df: pd.DataFrame):
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _scale_inplace(df: pd.DataFrame):
    df[REQ_COLS] = scaler().transform(df[REQ_COLS])
    return df


def make_sequences(df: pd.DataFrame) -> np.ndarray:
    seqs = []
    for _, row in df.iterrows():
        steps = []
        for y in YEARS:
            steps.append([row[f"{y}_{s}"] for s in SUBJECTS])
        seqs.append(steps)
    return np.asarray(seqs, dtype="float32")  # (N, 3, 10)


# -------------------- Predictors --------------------
def predict_with_olevel_rf(df: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """
    Returns columns:
      Pred_Stream, Pred_Label, TopK_Streams, TopK_Probs, plus per-class columns Prob_<cls>
    """
    _verify_columns(df)
    _scale_inplace(df.copy(deep=False))
    X = make_sequences(df).reshape((len(df), -1))

    probs = rf_model().predict_proba(X)  # (N, C)
    top_idx = np.argsort(-probs, axis=1)[:, :top_k]
    pred_idx = top_idx[:, 0]
    pred_streams = label_encoder().inverse_transform(pred_idx)

    out = pd.DataFrame({
        "Pred_Label": pred_idx,
        "Pred_Stream": pred_streams,
        "TopK_Streams": [[CLASSES[i] for i in row] for row in top_idx],
        "TopK_Probs": [[float(probs[i, j]) for j in row] for i, row in enumerate(top_idx)],
    })
    # append per-class probability columns
    for j, cls in enumerate(CLASSES):
        out[f"Prob_{cls}"] = probs[:, j]
    return out


def predict_with_olevel_lstm(df: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    _verify_columns(df)
    _scale_inplace(df.copy(deep=False))
    X = make_sequences(df)

    probs = lstm_model().predict(X)  # (N, C)
    top_idx = np.argsort(-probs, axis=1)[:, :top_k]
    pred_idx = top_idx[:, 0]
    pred_streams = label_encoder().inverse_transform(pred_idx)

    out = pd.DataFrame({
        "Pred_Label": pred_idx,
        "Pred_Stream": pred_streams,
        "TopK_Streams": [[CLASSES[i] for i in row] for row in top_idx],
        "TopK_Probs": [[float(probs[i, j]) for j in row] for i, row in enumerate(top_idx)],
    })
    for j, cls in enumerate(CLASSES):
        out[f"Prob_{cls}"] = probs[:, j]
    return out


def predict_with_olevel_ensemble(df: pd.DataFrame, alpha: float = 0.5, top_k: int = 3) -> pd.DataFrame:
    """
    alpha = weight on LSTM (0..1). Ensemble = alpha*LSTM + (1-alpha)*RF
    """
    _verify_columns(df)
    _scale_inplace(df.copy(deep=False))
    X_seq = make_sequences(df)
    X_rf = X_seq.reshape((len(df), -1))

    p_lstm = lstm_model().predict(X_seq)
    p_rf = rf_model().predict_proba(X_rf)
    probs = alpha * p_lstm + (1.0 - alpha) * p_rf

    top_idx = np.argsort(-probs, axis=1)[:, :top_k]
    pred_idx = top_idx[:, 0]
    pred_streams = label_encoder().inverse_transform(pred_idx)

    out = pd.DataFrame({
        "Pred_Label": pred_idx,
        "Pred_Stream": pred_streams,
        "TopK_Streams": [[CLASSES[i] for i in row] for row in top_idx],
        "TopK_Probs": [[float(probs[i, j]) for j in row] for i, row in enumerate(top_idx)],
    })
    for j, cls in enumerate(CLASSES):
        out[f"Prob_{cls}"] = probs[:, j]
    return out


# ------------- Required columns helper for UI/API -------------
def required_columns() -> List[str]:
    return list(REQ_COLS)
