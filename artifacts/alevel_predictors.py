# artifacts/alevel_predictors.py
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import joblib
from django.conf import settings


# ---------- Load artifacts ----------
def _dir() -> Path:
    base = Path(getattr(settings, "BASE_DIR", ".")) / "models" / "A-Level"
    if not base.exists():
        raise FileNotFoundError(f"A-Level model dir not found: {base}")
    return base


with open(_dir() / "alevel_meta.json", "r", encoding="utf-8") as f:
    META = json.load(f)

FEATURE_COLS: List[str] = META["feature_cols"]
CLASSES: List[str] = META["classes"]
TARGET: str = META["target"]

_PIPE = None
_LE = None


def pipe():
    global _PIPE
    if _PIPE is None:
        _PIPE = joblib.load(_dir() / "alevel_pipeline.pkl")
    return _PIPE


def label_encoder():
    global _LE
    if _LE is None:
        _LE = joblib.load(_dir() / "alevel_label_encoder.pkl")
    return _LE


# ---------- Core ----------
def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # keep only feature cols, in correct order
    return df.loc[:, FEATURE_COLS].copy()


def predict_alevel_programs(df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """
    Returns:
      Pred_Label, Pred_Program, TopK_Programs, TopK_Probs, and per-class probabilities Prob_<cls>
    """
    X = _prepare(df)
    probs = pipe().predict_proba(X)  # (N, C)
    top_idx = np.argsort(-probs, axis=1)[:, :top_k]
    pred_idx = top_idx[:, 0]
    pred_names = label_encoder().inverse_transform(pred_idx)

    out = pd.DataFrame({
        "Pred_Label": pred_idx,
        "Pred_Program": pred_names,
        "TopK_Programs": [[CLASSES[j] for j in row] for row in top_idx],
        "TopK_Probs": [[float(probs[i, j]) for j in row] for i, row in enumerate(top_idx)],
    })
    # per-class cols
    for j, cls in enumerate(CLASSES):
        out[f"Prob_{cls}"] = probs[:, j]
    return out


def required_columns() -> List[str]:
    return list(FEATURE_COLS)
