import io
import os
import pickle
import pandas as pd
import pickle
from functools import lru_cache
from django.conf import settings
from pathlib import Path

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def _coerce_estimator(obj):
    # Accept estimator/pipeline
    if hasattr(obj, "predict"):
        return obj
    # Accept CV objects
    if hasattr(obj, "best_estimator_") and hasattr(obj.best_estimator_, "predict"):
        return obj.best_estimator_
    # Accept common wrappers
    if isinstance(obj, dict):
        for k in ("model", "pipeline", "clf", "estimator", "best_estimator_"):
            if k in obj and hasattr(obj[k], "predict"):
                return obj[k]
    raise TypeError(f"Loaded object is {type(obj)}; it has no .predict(). Re-save a trained sklearn Pipeline.")


@lru_cache(maxsize=8)
def load_model(model_key: str):
    candidates = []
    reg = getattr(settings, "MODEL_REGISTRY", {}) or {}
    p = reg.get(model_key)
    if p:
        candidates.append(Path(p))

    model_dir = getattr(settings, "MODEL_DIR", None)
    if model_dir:
        md = Path(model_dir)
        candidates += [md / f"{model_key}.pkl", md / f"{model_key}_model.pkl"]

    here = Path(__file__).resolve().parent
    app_ml = here / "ml_models"
    candidates += [app_ml / f"{model_key}.pkl", app_ml / f"{model_key}_model.pkl"]

    last_err = None
    for cand in candidates:
        try:
            if cand.exists():
                try:
                    import joblib
                    obj = joblib.load(cand)
                except Exception:
                    with open(cand, "rb") as f:
                        obj = pickle.load(f)
                return _coerce_estimator(obj)
        except Exception as e:
            last_err = e

    raise FileNotFoundError(
        f"Model '{model_key}' not found or not a valid estimator.\n"
        f"Checked:\n  - " + "\n  - ".join(str(c) for c in candidates) +
        (f"\nLast load error: {last_err}" if last_err else "")
    )


def read_dataframe(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload CSV or Excel.")


def get_feature_frame(df, model):
    if hasattr(model, "feature_names_in_"):
        need = list(model.feature_names_in_)
        miss = [c for c in need if c not in df.columns]
        if miss:
            raise ValueError(f"Missing required feature columns: {miss}")
        return df[need].copy()
    return df.copy()


def predict_with_model(model, X: pd.DataFrame):
    """
    Returns (pred_labels, proba_dict_list, classes_list)
    proba_dict_list: list[dict] like [{"ClassA":0.2, "ClassB":0.8}, ...] or None if not available
    """
    y_pred = model.predict(X)
    proba_list = None
    classes = None

    if hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(X)
            if probas is not None:
                classes = getattr(model, "classes_", None)
                if classes is not None:
                    classes = list(map(str, classes))
                    proba_list = []
                    for row in probas:
                        proba_list.append({c: float(p) for c, p in zip(classes, row)})
        except Exception:
            # silently ignore probability failures
            proba_list = None
            classes = None

    # Coerce labels to string for JSON safe storage
    y_pred = [str(v) for v in y_pred]
    return y_pred, proba_list, classes


# *********************************************************************************************************************
def _get_feature_names_from_ct(ct):
    """Extract final output feature names from a ColumnTransformer."""
    output_features = []
    for name, transformer, cols in getattr(ct, "transformers_", []):
        if transformer in ("drop", None):
            continue
        if transformer == "passthrough":
            # passthrough keeps original col names
            if isinstance(cols, slice):
                raise ValueError("Slice columns not supported in this helper.")
            output_features.extend(list(cols))
        else:
            # If this is a Pipeline, get the last step
            if hasattr(transformer, "get_feature_names_out"):
                feat = transformer.get_feature_names_out(cols)
                output_features.extend([str(f) for f in feat])
            elif hasattr(transformer, "named_steps"):
                last = list(transformer.named_steps.values())[-1]
                if hasattr(last, "get_feature_names_out"):
                    feat = last.get_feature_names_out(cols)
                    output_features.extend([str(f) for f in feat])
                else:
                    output_features.extend(list(cols))
            else:
                output_features.extend(list(cols))
    return output_features


def get_pipeline_feature_names(pipeline):
    """Return the expanded feature names for a Pipeline(pre→clf) if possible."""
    try:
        pre = pipeline.named_steps.get("pre")
    except Exception:
        pre = None
    if pre is None:
        # Fallback: some sklearn models expose feature_names_in_
        return list(getattr(pipeline, "feature_names_in_", []))
    try:
        return _get_feature_names_from_ct(pre)
    except Exception:
        return list(getattr(pipeline, "feature_names_in_", []))


def top_feature_importances(pipeline, top_n=12):
    """Return [(feature, importance)] for RF-like models with feature_importances_."""
    clf = None
    try:
        clf = pipeline.named_steps.get("clf", None) or pipeline
    except Exception:
        clf = pipeline
    if not hasattr(clf, "feature_importances_"):
        return []
    names = get_pipeline_feature_names(pipeline)
    imps = list(clf.feature_importances_)
    # pad/truncate names to match importances length
    if names and len(names) != len(imps):
        names = names[:len(imps)] if len(names) > len(imps) else names + [f"f_{i}" for i in
                                                                          range(len(imps) - len(names))]
    pairs = sorted(zip(names or [f"f_{i}" for i in range(len(imps))], imps), key=lambda x: x[1], reverse=True)[:top_n]
    return pairs


import numpy as np


def top_linear_coefficients(pipeline, top_n=12):
    # works for LogisticRegression, LinearSVM (has coef_), etc.
    try:
        clf = pipeline.named_steps.get("clf", pipeline)
    except Exception:
        clf = pipeline
    if not hasattr(clf, "coef_"):
        return []

    names = []
    try:
        names = get_pipeline_feature_names(pipeline)
    except Exception:
        names = []

    coef = np.asarray(clf.coef_, dtype=float)
    # Handle binary/multiclass: pick absolute weights of class 1 if available
    if coef.ndim == 2 and coef.shape[0] > 1:
        weights = np.abs(coef[1])
    else:
        weights = np.abs(coef.ravel())

    if not names or len(names) != len(weights):
        names = [f"f_{i}" for i in range(len(weights))]

    pairs = sorted(zip(names, weights), key=lambda x: x[1], reverse=True)[:top_n]
    return pairs
