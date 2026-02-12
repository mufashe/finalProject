# levels/models.py
from django.db import models


class EducationLevel(models.Model):
    slug = models.SlugField(max_length=20, unique=True)
    name = models.CharField(max_length=50, unique=True)
    description = models.TextField(blank=True)

    class Meta:
        ordering = ['id']

    def __str__(self):
        return self.name
