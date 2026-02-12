from django.db import models


# Create your models here.


class UploadedDataset(models.Model):
    original_file = models.FileField(upload_to="datasets/")
    original_filename = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    # Optional: processed artifact
    processed_file = models.FileField(upload_to="processed/", null=True, blank=True)

    # Simple metadata
    n_rows = models.IntegerField(default=0)
    n_cols = models.IntegerField(default=0)

    def __str__(self):
        return f"{self.original_filename} ({self.uploaded_at:%Y-%m-%d %H:%M})"
