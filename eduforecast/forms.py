from django import forms
from .models import UploadedDataset


class UploadForm(forms.ModelForm):
    class Meta:
        model = UploadedDataset
        fields = ["original_file"]

    def clean_original_file(self):
        f = self.cleaned_data["original_file"]
        allowed = (".xlsx", ".xls", ".csv")
        name = f.name.lower()
        if not name.endswith(allowed):
            raise forms.ValidationError("Please upload an Excel (.xlsx/.xls) or CSV file.")
        return f
