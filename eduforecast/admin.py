from django.contrib import admin

# Register your models here.

# uploader/admin.py
from django.contrib import admin
from .models import UploadedDataset


@admin.register(UploadedDataset)
class UploadedDatasetAdmin(admin.ModelAdmin):
    list_display = ("id", "original_filename", "uploaded_at", "n_rows", "n_cols", "processed_file")
    readonly_fields = ("uploaded_at",)
