# artifacts/utils.py
import os
import io
import json
import joblib
import numpy as np
import pandas as pd


def _try_import(modname):
    try:
        return __import__(modname)
    except Exception:
        return None


_tf = _try_import("tensorflow")
_xgb = _try_import("xgboost")
_lgb = _try_import("lightgbm")


def load_model(artifact_path, model_format):
    """Load model object based on format + file extension."""
    ext = os.path.splitext(artifact_path)[1].lower()
    if model_format == "keras":
        if _tf is None:
            raise RuntimeError("TensorFlow/Keras not installed.")
        # Supports .h5 or SavedModel dir path
        return _tf.keras.models.load_model(artifact_path)
    elif model_format in ("sklearn", "statsmodels", "xgboost", "lightgbm"):
        # Most of these are stored as joblib/pickle
        with open(artifact_path, "rb") as f:
            return joblib.load(f)
    elif model_format == "onnx":
        ort = _try_import("onnxruntime")
        if ort is None:
            raise RuntimeError("onnxruntime not installed.")
        # Return a small wrapper to run session
        return ("onnx", ort.InferenceSession(artifact_path))
    else:
        raise ValueError(f"Unsupported model_format: {model_format}")


def _ensure_columns(df, needed):
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def run_classification(model, df, feature_list=None, want_proba=True, top_k=3):
    """Return DataFrame with predictions (and optional top-k)."""
    X = df[feature_list] if feature_list else df.select_dtypes(include=[np.number])

    # ONNX special-case
    if isinstance(model, tuple) and model[0] == "onnx":
        session = model[1]
        input_name = session.get_inputs()[0].name
        pred = session.run(None, {input_name: X.values.astype(np.float32)})
        # pred may vary by model; here we assume argmax classification
        y_pred = np.argmax(pred[0], axis=1)
        proba = pred[0]
    else:
        # If the saved object is a sklearn Pipeline, this will include preprocessing/encoders
        if hasattr(model, "predict_proba") and want_proba:
            proba = model.predict_proba(X)
        else:
            proba = None
        y_pred = model.predict(X)

    out = pd.DataFrame(index=df.index)
    out["prediction"] = y_pred

    if proba is not None:
        proba = np.asarray(proba)
        # Top-k labels (if estimator has classes_)
        classes_ = getattr(model, "classes_", None)
        if classes_ is None and hasattr(getattr(model, "named_steps", {}), "classifier"):
            classes_ = model.named_steps["classifier"].classes_
        if classes_ is not None:
            classes_ = np.array(classes_)
        # compute top-k
        if proba.ndim == 2 and proba.shape[1] > 1:
            top_k = min(top_k, proba.shape[1])
            top_idx = np.argsort(-proba, axis=1)[:, :top_k]
            for k in range(top_k):
                col = f"top{k + 1}_label"
                proba_col = f"top{k + 1}_proba"
                if classes_ is not None:
                    out[col] = classes_[top_idx[:, k]]
                else:
                    out[col] = top_idx[:, k]
                out[proba_col] = proba[np.arange(proba.shape[0]), top_idx[:, k]]
        else:
            # binary case with single prob column sometimes
            if proba.ndim == 1:
                out["proba"] = proba
            elif proba.ndim == 2 and proba.shape[1] == 1:
                out["proba"] = proba[:, 0]

    return out


def run_forecasting(model, df, horizon=12, time_col=None, target_col=None):
    """
    Generic forecast:
      - If model has .forecast(h), use it.
      - Else if .predict, try .predict on the next h steps (no exog).
    df may be ignored if the model contains state (ARIMA) — we keep it for future extensions.
    """
    # Best effort defaults:
    if time_col is None:
        for c in ["ds", "date", "time", "year", "period"]:
            if c in df.columns:
                time_col = c
                break
    if target_col is None:
        # first numeric column as target fallback
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            target_col = num_cols[0]

    # Try statsmodels/pmdarima style
    if hasattr(model, "forecast"):
        yhat = model.forecast(horizon)
    elif hasattr(model, "predict"):
        try:
            yhat = model.predict(horizon)
        except TypeError:
            # some statsmodels require start/end
            start = len(df)
            end = start + horizon - 1
            yhat = model.predict(start=start, end=end)
    else:
        raise ValueError("Model does not support forecast/predict for time series.")

    # Build future index
    if time_col and pd.api.types.is_datetime64_any_dtype(df[time_col]):
        last = df[time_col].max()
        # naive frequency guess (monthly)
        freq = pd.infer_freq(df[time_col].sort_values()) or "MS"
        future_idx = pd.date_range(last, periods=horizon + 1, freq=freq, inclusive="right")
    else:
        future_idx = pd.RangeIndex(start=1, stop=horizon + 1)

    out = pd.DataFrame({"forecast": np.array(yhat).reshape(-1)}, index=future_idx)
    out.index.name = "future"
    return out
