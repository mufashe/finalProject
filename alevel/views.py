from django.shortcuts import render, redirect
from django.contrib import messages
from django.urls import reverse
from django.views.decorators.http import require_http_methods
import pandas as pd

from alevel.forms import AlevelDatasetForm
from alevel.models import AlevelDataset


@require_http_methods(["GET", "POST"])
def upload_a_level_dataset(request):
    if request.method == "POST":
        form = AlevelDatasetForm(request.POST, request.FILES)
        if form.is_valid():
            upload = form.save(commit=False)
            upload.original_name = request.FILES['file'].name
            upload.save()

            # OPTIONAL: parse with pandas immediately
            try:
                df = pd.read_excel(upload.file.path)  # requires openpyxl
                # TODO: validate columns & optionally insert into DB
                # Example: just preview first rows in message (keep short)
                preview_cols = list(df.columns[:6])
                upload.processed = True
                upload.save(update_fields=["processed"])
                messages.success(
                    request,
                    f"Upload successful. Detected columns: {preview_cols} (showing up to 6)."
                )
            except Exception as e:
                messages.warning(request, f"File saved but not parsed: {e}")

            return redirect(reverse('alevel:viewdataset'))
        else:
            messages.error(request, "Please fix the errors below.")
    else:
        form = AlevelDatasetForm()
    return render(request, 'uploadalevel.html', {'form': form})


def viewUploadedDataset(request):
    alevedatasets = AlevelDataset.objects.order_by('-uploaded_at')
    return render(request, 'uploads_list.html', {'alevedatasets': alevedatasets})
