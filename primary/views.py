import json
from collections import Counter
from django.db.models import Count
from .utils import load_model, top_feature_importances
from django.views.decorators.http import require_http_methods
import pandas as pd

from configurations.models import EducationLevel
from primary.forms import PrimaryUploadForm
from primary.models import PrimaryUpload


@require_http_methods(["GET", "POST"])
def upload_primary_excel(request):
    if request.method == "POST":
        form = PrimaryUploadForm(request.POST, request.FILES)
        if form.is_valid():
            upload = form.save(commit=False)
            upload.original_name = request.FILES['file'].name
            upload.save()
            try:
                pd.read_excel(upload.file.path)  # optional parse
                upload.processed = True
                upload.save(update_fields=["processed"])
                messages.success(request, "Upload successful and parsed.")
            except Exception as e:
                messages.warning(request, f"File saved but not parsed: {e}")
            return redirect(reverse('primary:uploads_list'))
        messages.error(request, "Please fix the errors below.")
    else:
        # Preselect 'primary' if available
        initial_level = EducationLevel.objects.filter(slug='primary').first()
        form = PrimaryUploadForm(initial={'level': initial_level})

    return render(request, 'upload.html', {'form': form})


def uploads_list(request):
    uploads = PrimaryUpload.objects.order_by('-uploaded_at')
    return render(request, 'uploads_list.html', {'uploads': uploads})


# **********************************************************************************************************************
import os
import pandas as pd
from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.core.files.base import ContentFile
from django.urls import reverse
from .forms import UploadPredictForm
from .models import UploadedDataset, PredictionBatch, PredictionRow
from .utils import load_model, read_dataframe, get_feature_frame, predict_with_model


def upload_and_predict(request):
    """
    Page 1: upload dataset + pick model, run prediction, store results.
    Redirect to a detail page to view predictions.
    """
    if request.method == "POST":
        form = UploadPredictForm(request.POST, request.FILES)
        if form.is_valid():
            level = form.cleaned_data["level"]
            model_key = form.cleaned_data["model_key"]
            upfile = request.FILES["dataset"]
            id_column = form.cleaned_data.get("id_column") or ""

            # Read the dataframe
            try:
                df = read_dataframe(upfile)
            except Exception as e:
                messages.error(request, f"Failed to read file: {e}")
                return render(request, "primary/upload_predict.html", {"form": form})

            # Persist the uploaded file
            uploaded = UploadedDataset.objects.create(
                level=level,
                file=upfile,
                original_name=upfile.name,
                rows=len(df),
            )

            # Load model and align features
            try:
                model = load_model(model_key)
                X = get_feature_frame(df, model)
            except Exception as e:
                messages.error(request, f"Model/feature error: {e}")
                uploaded.delete()  # clean up if desired
                return render(request, "primary/upload_predict.html", {"form": form})

            # Run predictions
            try:
                y_pred, proba_list, classes = predict_with_model(model, X)
            except Exception as e:
                messages.error(request, f"Prediction error: {e}")
                uploaded.delete()
                return render(request, "primary/upload_predict.html", {"form": form})

            # Create batch
            batch = PredictionBatch.objects.create(
                dataset=uploaded,
                model_key=model_key,
                classes=classes or [],
            )

            # Build per-row objects
            rows_to_create = []
            ext_ids = df[id_column].astype(str).tolist() if id_column and id_column in df.columns else [None] * len(df)
            for i, (pred, ext) in enumerate(zip(y_pred, ext_ids), start=1):
                proba = proba_list[i - 1] if proba_list else None
                rows_to_create.append(PredictionRow(
                    batch=batch,
                    index_in_file=i,
                    external_id=ext,
                    label=pred,
                    proba=proba,
                ))
            PredictionRow.objects.bulk_create(rows_to_create, batch_size=1000)

            # Also save a CSV for download
            out_df = pd.DataFrame({
                "index_in_file": range(1, len(y_pred) + 1),
                "external_id": ext_ids,
                "prediction": y_pred,
            })
            if proba_list and classes:
                for c in classes:
                    out_df[f"proba_{c}"] = [d.get(c, None) for d in proba_list]

            csv_bytes = out_df.to_csv(index=False).encode("utf-8")
            csv_name = f"pred_{batch.id}_{os.path.splitext(uploaded.original_name)[0]}.csv"
            batch.predictions_csv.save(csv_name, ContentFile(csv_bytes))
            batch.save()

            messages.success(request, "Predictions generated successfully.")
            return redirect(reverse("prediction_detail", args=[batch.id]))
        else:
            messages.error(request, "Please correct the errors below.")
    else:
        form = UploadPredictForm()

    return render(request, "primary/upload_predict.html", {"form": form})


def predictions_list(request):
    """
    Page 2 (list): all batches with links to detail pages.
    """
    batches = (PredictionBatch.objects
               .select_related("dataset")
               .order_by("-created_at"))
    return render(request, "primary/predictions_list.html", {"batches": batches})


def prediction_detail(request, batch_id: int):
    batch = get_object_or_404(PredictionBatch.objects.select_related("dataset"), pk=batch_id)
    rows_qs = batch.rows.all().order_by("index_in_file")
    total_rows = rows_qs.count()
    rows = list(rows_qs[:1000])  # cap for chart payload

    # 1) Predicted class distribution
    label_counts = Counter([r.label for r in rows])
    dist_labels = list(label_counts.keys())
    dist_values = list(label_counts.values())

    # 2) Probability histogram (binary only, if proba saved)
    proba_vals = []
    if batch.classes and len(batch.classes) == 2:
        pos_label = batch.classes[1]  # assuming classes = [0,1] or ['No','Yes'] in training order
        for r in rows:
            if r.proba and pos_label in r.proba:
                proba_vals.append(float(r.proba[pos_label]))
    # small 10-bin histogram
    hist_bins = [i / 10 for i in range(11)]
    hist_counts = [0] * 10
    for p in proba_vals:
        idx = min(int(p * 10), 9)
        hist_counts[idx] += 1

    # 3) Feature importances for RF
    feat_labels, feat_values = [], []
    try:
        model = load_model(batch.model_key)
        pairs = top_feature_importances(model, top_n=12)
        if not pairs:
            # fall back to linear coefficients (e.g., Logistic Regression)
            from .utils import top_linear_coefficients
            pairs = top_linear_coefficients(model, top_n=12)
        if pairs:
            feat_labels = [a for a, _ in pairs]
            feat_values = [float(b) for _, b in pairs]
    except Exception:
        pass

    # Serialize for Chart.js
    chart_data = {
        "predDist": {"labels": dist_labels, "values": dist_values},
        "probaHist": {"bins": [f"{hist_bins[i]:.1f}-{hist_bins[i + 1]:.1f}" for i in range(10)], "values": hist_counts},
        "featImp": {"labels": feat_labels, "values": feat_values},
        "meta": {"total_rows": total_rows, "model": batch.model_key}
    }

    rows_preview = rows[:50]
    more = total_rows - len(rows_preview)
    return render(request, "primary/prediction_detail.html", {
        "batch": batch,
        "rows": rows_preview,
        "more_count": more if more > 0 else 0,
        "chart_json": json.dumps(chart_data),
    })
