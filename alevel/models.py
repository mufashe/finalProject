from django.db import models

from configurations.models import EducationLevel


def a_level_upload_path(instance, filename):
    return f'primary/uploads/{filename}'


class AlevelDataset(models.Model):
    class ALevelUpload(models.Model):
        level = models.ForeignKey(
            EducationLevel, on_delete=models.PROTECT, related_name='a_level_uploads'
        )

    file = models.FileField(upload_to=a_level_upload_path)
    original_name = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    notes = models.TextField(blank=True)

    def __str__(self):
        return f"{self.original_name} ({self.uploaded_at:%Y-%m-%d %H:%M})"
