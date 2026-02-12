from django import forms

from alevel.models import AlevelDataset
from configurations.models import EducationLevel

ALLOWED_EXTS = ('.xlsx', '.xls', '.csv')


class AlevelDatasetForm(forms.ModelForm):
    level = forms.ModelChoiceField(  # <-- NEW
        queryset=EducationLevel.objects.all().order_by('id'),
        empty_label=None, label="Level", help_text="Select the education level this file belongs to."
    )

    class Meta:
        model = AlevelDataset
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
