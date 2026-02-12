from django.db import models

from configurations.models import EducationLevel


def primary_upload_path(instance, filename):
    return f'primary/uploads/{filename}'


class PrimaryUpload(models.Model):
    level = models.ForeignKey(EducationLevel, on_delete=models.PROTECT, related_name='primary_uploads')
    file = models.FileField(upload_to=primary_upload_path)
    original_name = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    notes = models.TextField(blank=True)

    def __str__(self):
        return f"{self.original_name} ({self.uploaded_at:%Y-%m-%d %H:%M})"


# **********************************************************************************************************************

from django.db import models
from django.utils import timezone
from django.contrib.postgres.fields import ArrayField  # Optional; only if using Postgres
from django.core.serializers.json import DjangoJSONEncoder


class UploadedDataset(models.Model):
    LEVEL_CHOICES = [
        ("primary", "Primary"),
        ("olevel", "O-Level"),
        ("alevel", "A-Level"),
    ]
    level = models.CharField(max_length=20, choices=LEVEL_CHOICES, default="primary")
    file = models.FileField(upload_to="uploads/%Y/%m/%d/")
    original_name = models.CharField(max_length=255)
    rows = models.PositiveIntegerField(default=0)
    uploaded_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.original_name} ({self.level})"


class PredictionBatch(models.Model):
    """
    One batch per upload+model combination.
    Stores a companion CSV of predictions for quick download as well.
    """
    dataset = models.ForeignKey(UploadedDataset, on_delete=models.CASCADE, related_name="batches")
    model_key = models.CharField(max_length=100)  # e.g., "primary_logreg", "primary_rf"
    classes = models.JSONField(encoder=DjangoJSONEncoder, default=list, blank=True)
    predictions_csv = models.FileField(upload_to="predictions/%Y/%m/%d/", blank=True, null=True)
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"Batch #{self.id} - {self.model_key} - {self.dataset.original_name}"


class PredictionRow(models.Model):
    """
    Store per-row predictions for display. Keep it generic:
    - external_id: optional column from the uploaded data (e.g., student ID)
    - label: predicted class
    - proba: per-class probabilities when available
    """
    batch = models.ForeignKey(PredictionBatch, on_delete=models.CASCADE, related_name="rows")
    index_in_file = models.PositiveIntegerField()
    external_id = models.CharField(max_length=255, blank=True, null=True)
    label = models.CharField(max_length=255)
    proba = models.JSONField(encoder=DjangoJSONEncoder, blank=True, null=True)

    def __str__(self):
        return f"Row {self.index_in_file} -> {self.label}"
