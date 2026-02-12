import json
import os.path

import joblib
import numpy as np
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
import tensorflow as tf
from django.views.decorators.csrf import csrf_exempt
import pandas as pd

# Create your views here.
from configurations.models import EducationLevel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'ml_models')

randomForest = joblib.load(os.path.join(MODEL_DIR, 'rf_model.pkl'))
metaModel = joblib.load(os.path.join(MODEL_DIR, 'meta_model.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
lstmModel = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'lstm_model.h5'))
targetEncoder = joblib.load(os.path.join(MODEL_DIR, 'target_encoder.pkl'))
labels = targetEncoder.inverse_transform([0, 2, 1])
print(labels)

SUBJECTS = ["Math", "Physics", "Chemistry", "Biology", "Geography", "Economics", "Computer", "History"]


@csrf_exempt
def predictALevel(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        X_tab = [data[feat] for feat in data if not any(s in feat for s in ['S4', 'S5', 'S6'])]
        X_tab = np.array(X_tab).reshape(1, -1)
        X_tab = scaler.transform(X_tab)

        seq = []
        for year in ['S4', 'S5', 'S6']:
            seq.append([data[f"{year}_{sub}"] for sub in SUBJECTS])

        X_seq = np.array(seq).reshape(1, 3, len(SUBJECTS))

        randomForestPrediction = randomForest.predict(X_tab)[0]
        lstmModelPrediction = lstmModel.predict(X_seq)
        lstmModelPrediction_cls = int(np.argmax(lstmModelPrediction, axis=1)[0])

        # Ensemble meta-prediction
        metaX = np.array([[lstmModelPrediction_cls, randomForestPrediction]])
        metaPrediction = metaModel.predict(metaX)[0]

        return JsonResponse({
            'randomForestPrediction': int(randomForestPrediction),
            'lstmPrediction': int(lstmModelPrediction_cls),
            'ensemblePrediction': int(metaPrediction)
        })
    else:
        return JsonResponse({"error": "POST request required"}, status=400)


# ************************************************************************************************************


def batch_predict(file_path):
    df = pd.read_excel(file_path)
    results = []

    # You may want to map categorical variables if not already encoded!
    cat_cols = ["gender", "school_location", "residence_location", "school_type",
                "is_boarding", "has_electricity", "parent_status", "extracurricular"]
    for col in cat_cols:
        if col in df.columns and not np.issubdtype(df[col].dtype, np.number):
            df[col] = pd.factorize(df[col])[0]

    # Prepare features for models
    X_tab = df.drop(columns=["student_id", "predicted_university_subject"], errors='ignore')
    X_tab_scaled = scaler.transform(X_tab)

    # Prepare sequence input for LSTM
    X_seq = []
    for i, row in df.iterrows():
        seq = []
        for year in ["S4", "S5", "S6"]:
            seq.append([row.get(f"{year}_{sub}", 0) for sub in SUBJECTS])
        X_seq.append(seq)
    X_seq = np.array(X_seq)

    # Predictions
    rf_preds = randomForest.predict(X_tab_scaled)
    lstm_preds = lstmModel.predict(X_seq)
    lstm_preds_cls = np.argmax(lstm_preds, axis=1)
    meta_X = np.vstack([lstm_preds_cls, rf_preds]).T
    meta_preds = metaModel.predict(meta_X)

    # Save to result DataFrame
    df['rf_prediction'] = targetEncoder.inverse_transform(rf_preds)
    df['lstm_prediction'] = targetEncoder.inverse_transform(lstm_preds_cls)
    df['ensemble_prediction'] = targetEncoder.inverse_transform(meta_preds)

    return df


@csrf_exempt
def upload_excel(request):
    if request.method == "POST" and request.FILES.get('file'):
        excel_file = request.FILES['file']
        df_result = batch_predict(excel_file)

        # Return as downloadable Excel
        response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        response['Content-Disposition'] = 'attachment; filename=predictions.xlsx'
        with pd.ExcelWriter(response, engine='openpyxl') as writer:
            df_result.to_excel(writer, index=False)
        return response

    # Simple HTML upload form
    return render(request, 'eduforecast/upload.html')


# *********************************************************************************************************************
import io
import pandas as pd
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from django.contrib import messages
from django.core.files.base import ContentFile

from .forms import UploadForm
from .models import UploadedDataset
from .processing import process_dataframe


def _read_df(file_field):
    name = file_field.name.lower()
    path = file_field.path
    if name.endswith(".csv"):
        return pd.read_csv(path)
    # Excel
    return pd.read_excel(path, engine="openpyxl")


def list_datasets(request):
    qs = UploadedDataset.objects.order_by("-uploaded_at")
    return render(request, "eduforecast/list.html", {"datasets": qs})


def upload_dataset(request):
    if request.method == "POST":
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            obj = form.save(commit=False)
            obj.original_filename = request.FILES["original_file"].name
            obj.save()  # file is saved to MEDIA_ROOT

            # compute basic metadata
            try:
                df = _read_df(obj.original_file)
                obj.n_rows, obj.n_cols = df.shape
                obj.save(update_fields=["n_rows", "n_cols"])
                messages.success(request, "File uploaded successfully.")
                return redirect("uploader:detail", pk=obj.pk)
            except Exception as e:
                obj.delete()
                messages.error(request, f"Failed to read file: {e}")
                return redirect("uploader:upload")
    else:
        form = UploadForm()
    return render(request, "eduforecast/upload.html", {"form": form})


def dataset_detail(request, pk):
    obj = get_object_or_404(UploadedDataset, pk=pk)
    try:
        df = _read_df(obj.original_file)
        preview_html = df.head(100).to_html(classes="table table-striped table-sm", index=False, border=0)
    except Exception as e:
        preview_html = f"<p class='text-danger'>Error reading file: {e}</p>"
    return render(request, "eduforecast/detail.html", {"obj": obj, "preview_html": preview_html})


def process_dataset(request, pk):
    obj = get_object_or_404(UploadedDataset, pk=pk)

    try:
        df = _read_df(obj.original_file)
        df_proc = process_dataframe(df)

        csv_bytes = df_proc.to_csv(index=False).encode("utf-8")
        fname = obj.original_filename.rsplit(".", 1)[0] + "_processed.csv"
        obj.processed_file.save(fname, ContentFile(csv_bytes), save=True)

        messages.success(request, "Processing complete.")
        return redirect("uploader:processed_detail", pk=obj.pk)
    except Exception as e:
        messages.error(request, f"Processing failed: {e}")
        return redirect("uploader:detail", pk=obj.pk)


def processed_detail(request, pk):
    obj = get_object_or_404(UploadedDataset, pk=pk)
    if not obj.processed_file:
        messages.info(request, "This dataset has not been processed yet.")
        return redirect("uploader:detail", pk=obj.pk)

    try:
        df = pd.read_csv(obj.processed_file.path)
        preview_html = df.head(100).to_html(classes="table table-striped table-sm", index=False, border=0)
    except Exception as e:
        preview_html = f"<p class='text-danger'>Error reading processed file: {e}</p>"

    return render(request, "eduforecast/processed_detail.html", {"obj": obj, "preview_html": preview_html})


# ************************************************************************************************************
# primary/views.py
from itertools import chain
from django.apps import apps
from django.core.paginator import Paginator
from django.shortcuts import render
from django.contrib import messages


def all_datasets(request):
    """
    Aggregated list of uploaded datasets across apps with a level filter.
    It looks for these models if present:
      - primary.PrimaryUpload
      - o_level.OLevelUpload
      - a_level.ALevelUpload
    """
    level_slug = request.GET.get("level") or ""  # e.g., 'primary', 'o-level', 'a-level'
    page = int(request.GET.get("page", 1))
    page_size = int(request.GET.get("page_size", 20))

    # Level filter object (optional)
    level_obj = None
    if level_slug:
        level_obj = EducationLevel.objects.filter(slug=level_slug).first()
        if not level_obj:
            messages.warning(request, f"Level '{level_slug}' not found; showing all.")
            level_slug = ""

    # Helper: collect rows from a model if it exists
    def collect(model_label: str, source_name: str):
        try:
            Model = apps.get_model(model_label)
        except LookupError:
            return []
        qs = Model.objects.select_related("level").all()
        if level_obj:
            qs = qs.filter(level=level_obj)
        # Build a uniform dict for the template
        rows = []
        for obj in qs:
            try:
                file_url = obj.file.url if getattr(obj, "file", None) else ""
            except Exception:
                file_url = ""
            rows.append({
                "source": source_name,
                "original_name": getattr(obj, "original_name", str(obj)),
                "level_name": getattr(obj.level, "name", "—") if hasattr(obj, "level") else "—",
                "uploaded_at": getattr(obj, "uploaded_at", None),
                "processed": getattr(obj, "processed", False),
                "file_url": file_url,
            })
        return rows

    # Aggregate from the three apps (robust to missing apps)
    combined = list(chain(
        collect("primary.PrimaryUpload", "Primary"),
        collect("o_level.OLevelUpload", "O-Level"),
        collect("a_level.ALevelUpload", "A-Level"),
    ))

    # Sort by uploaded_at desc (None-safe)
    combined.sort(key=lambda r: (r["uploaded_at"] is None, r["uploaded_at"]), reverse=True)

    # Pagination
    paginator = Paginator(combined, page_size)
    page_obj = paginator.get_page(page)

    # Levels for filter dropdown
    levels = EducationLevel.objects.order_by("id")

    context = {
        "levels": levels,
        "selected_level_slug": level_slug,
        "page_obj": page_obj,
        "total_count": paginator.count,
        "page_size": page_size,
    }
    return render(request, "eduforecast/allDatasets.html", context)
