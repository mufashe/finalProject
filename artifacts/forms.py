# artifacts/forms.py
from django import forms

from configurations.models import EducationLevel
from .models import ModelArtifact


class ModelArtifactForm(forms.ModelForm):
    class Meta:
        model = ModelArtifact
        fields = ["name", "level", "task", "model_format", "file", "feature_list", "target_name", "notes"]
        widgets = {"notes": forms.Textarea(attrs={"rows": 3})}


class InferenceForm(forms.Form):
    level = forms.ModelChoiceField(
        queryset=EducationLevel.objects.all().order_by("id"),
        required=False, empty_label="All levels"
    )
    artifact = forms.ModelChoiceField(
        queryset=ModelArtifact.objects.filter(active=True).order_by("-created_at"),
        required=True
    )
    dataset = forms.ChoiceField(choices=[], required=True, help_text="Choose an uploaded dataset file")
    top_k = forms.IntegerField(min_value=1, max_value=10, initial=3, required=False)
    horizon = forms.IntegerField(min_value=1, max_value=120, initial=12, required=False)

    def set_dataset_choices(self, rows):
        choices = [(row["token"], f'{row["label"]} — {row["level"]} — {row["source"]}') for row in rows]
        self.fields["dataset"].choices = choices or [("", "No datasets found for the selected level")]


# *************************************************************************************************************
# artifacts/forms.py
from django import forms

MODEL_CHOICES = [
    ("primary_rf", "Primary — Random Forest"),
    ("primary_logreg", "Primary — Logistic Regression"),
    ("primary_lstm", "Primary — LSTM (P1–P5)"),
]


class DatasetUploadForm(forms.Form):
    dataset = forms.FileField(help_text="CSV with required columns.")
    model = forms.ChoiceField(choices=MODEL_CHOICES)
    save_as = forms.CharField(
        required=False,
        max_length=120,
        help_text="Optional: custom name (no extension). Letters/numbers/spaces only; we'll sanitize."
    )
