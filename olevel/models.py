from django.db import models

from configurations.models import EducationLevel


def primary_upload_path(instance, filename):
    return f'olevel/oleveldataset/{filename}'


class OlevelDataset(models.Model):
    level = models.ForeignKey(
        EducationLevel, on_delete=models.PROTECT, related_name='o_level_uploads'
    )
    file = models.FileField(upload_to=primary_upload_path)
    original_name = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)  # flip to True after parsing if you want
    notes = models.TextField(blank=True)

    def __str__(self):
        return f"{self.original_name} ({self.uploaded_at:%Y-%m-%d %H:%M})"
