# artifacts/ml_registry.py
from pathlib import Path
from django.conf import settings
import threading
import joblib

# TensorFlow only needed for LSTM
try:
    import tensorflow as tf
except Exception:
    tf = None


class _ModelCache:
    _models = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, key, loader):
        if key not in cls._models:
            with cls._lock:
                if key not in cls._models:
                    cls._models[key] = loader()
        return cls._models[key]


def load_pipeline(key: str):
    """Load a saved sklearn pipeline (preprocessor + clf)."""
    path = Path(settings.MODEL_REGISTRY[key])
    return _ModelCache.get(key, lambda: joblib.load(path))


def load_lstm():
    """Load Keras LSTM model + its preprocessor pack."""
    if tf is None:
        raise RuntimeError("TensorFlow is not installed; cannot load LSTM.")
    model_path = Path(settings.MODEL_REGISTRY["primary_lstm_model"])
    preproc_path = Path(settings.MODEL_REGISTRY["primary_lstm_preproc"])

    model = _ModelCache.get("primary_lstm_model", lambda: tf.keras.models.load_model(model_path))
    preproc = _ModelCache.get("primary_lstm_preproc", lambda: joblib.load(preproc_path))
    return model, preproc
