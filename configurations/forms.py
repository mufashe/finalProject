# levels/forms.py
from django import forms
from django.utils.text import slugify
from .models import EducationLevel

ALLOWED_EXTS = ('.xlsx', '.xls')


class EducationLevelForm(forms.ModelForm):
    class Meta:
        model = EducationLevel
        fields = ["name", "slug", "description"]
        widgets = {
            "description": forms.Textarea(attrs={"rows": 3, "placeholder": "Optional"}),
        }

    def clean(self):
        cleaned = super().clean()
        name = cleaned.get("name")
        slug = cleaned.get("slug")

        if name and not slug:
            cleaned["slug"] = slugify(name)

        # Enforce case-insensitive uniqueness for name and slug
        qs = EducationLevel.objects.all()
        if self.instance.pk:
            qs = qs.exclude(pk=self.instance.pk)

        if name and qs.filter(name__iexact=name).exists():
            self.add_error("name", "A level with this name already exists.")

        if cleaned.get("slug") and qs.filter(slug__iexact=cleaned["slug"]).exists():
            self.add_error("slug", "A level with this slug already exists.")

        return cleaned
