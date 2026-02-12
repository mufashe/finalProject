from django import forms

from configurations.models import EducationLevel
from primary.models import PrimaryUpload

ALLOWED_EXTS = ('.xlsx', '.xls', '.csv')


class PrimaryUploadForm(forms.ModelForm):
    level = forms.ModelChoiceField(queryset=EducationLevel.objects.all().order_by('id'),
                                   empty_label=None, label="Level",
                                   help_text="Select the education level this file belongs to."
                                   )

    class Meta:
        model = PrimaryUpload
        fields = ['level', 'file', 'notes']
        widgets = {'notes': forms.Textarea(attrs={'rows': 3})}

    def clean_file(self):
        f = self.cleaned_data['file']
        name = f.name.lower()
        if not any(name.endswith(ext) for ext in ALLOWED_EXTS):
            raise forms.ValidationError("Please upload an Excel file (.xlsx or .xls).")
        if f.size > 10 * 1024 * 1024:  # 10 MB limit (adjust as needed)
            raise forms.ValidationError("File too large. Max size is 10 MB.")
        return f


# ********************************************************************************************************************
from django import forms

MODEL_CHOICES = [
    ("primary_logreg", "Primary – Logistic Regression"),
    ("primary_rf", "Primary – Random Forest"),
]

LEVEL_CHOICES = [
    ("primary", "Primary"),
    ("olevel", "O-Level"),
    ("alevel", "A-Level"),
]


class UploadPredictForm(forms.Form):
    level = forms.ChoiceField(choices=LEVEL_CHOICES, initial="primary")
    model_key = forms.ChoiceField(choices=MODEL_CHOICES, initial="primary_rf")
    dataset = forms.FileField(
        help_text="CSV or Excel (.xlsx/.xls). Ensure column names align with the model’s training features."
    )
    id_column = forms.CharField(
        required=False,
        help_text="Optional column name to treat as external ID (e.g., student_id)."
    )
