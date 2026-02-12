# artifacts/models.py
from django.db import models
from django.conf import settings

from configurations.models import EducationLevel


def model_upload_path(instance, filename):
    return f"models/{instance.level.slug}/{filename}"


def inference_output_path(instance, filename):
    return f"inference/{instance.artifact.level.slug}/{filename}"


class ModelArtifact(models.Model):
    TASK_CHOICES = [
        ('classification', 'Classification'),
        ('forecasting', 'Forecasting'),
    ]
    FORMAT_CHOICES = [
        ('sklearn', 'scikit-learn/joblib/pickle'),
        ('keras', 'Keras/TensorFlow (.h5 or SavedModel)'),
        ('statsmodels', 'statsmodels/pmdarima'),
        ('xgboost', 'XGBoost'),
        ('lightgbm', 'LightGBM'),
        ('onnx', 'ONNX (optional)'),
    ]

    name = models.CharField(max_length=120, unique=True)
    level = models.ForeignKey(EducationLevel, on_delete=models.PROTECT, related_name='artifacts')
    task = models.CharField(max_length=20, choices=TASK_CHOICES)
    model_format = models.CharField(max_length=20, choices=FORMAT_CHOICES, default='sklearn')
    file = models.FileField(upload_to=model_upload_path)
    # Optional: store expected inputs
    feature_list = models.JSONField(blank=True, null=True,
                                    help_text="Optional: list of feature names expected by the model.")
    target_name = models.CharField(max_length=80, blank=True, help_text="Optional: target column name (for docs).")
    notes = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    active = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.name} [{self.level.name} • {self.task}]"


class InferenceJob(models.Model):
    artifact = models.ForeignKey(ModelArtifact, on_delete=models.CASCADE, related_name='jobs')
    dataset_label = models.CharField(max_length=200)  # Human label (original file name or chosen label)
    dataset_path = models.TextField()  # Absolute path to the dataset file used
    params = models.JSONField(blank=True, null=True)  # e.g., {"top_k":3} or {"horizon":12}
    started_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(auto_now=True)
    ok = models.BooleanField(default=False)
    message = models.TextField(blank=True)
    output_csv = models.FileField(upload_to=inference_output_path, blank=True, null=True)

    def __str__(self):
        return f"Job#{self.id} - {self.artifact.name} on {self.dataset_label}"
