from django.shortcuts import render, redirect
from django.contrib import messages
from django.urls import reverse
from django.views.decorators.http import require_http_methods
import pandas as pd

from olevel.forms import OlevelDatasetForm
from olevel.models import OlevelDataset


@require_http_methods(["GET", "POST"])
def upload_o_level_dataset(request):
    if request.method == "POST":
        form = OlevelDatasetForm(request.POST, request.FILES)
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

            return redirect(reverse('olevel:viewdataset'))
        else:
            messages.error(request, "Please fix the errors below.")
    else:
        form = OlevelDatasetForm()
    return render(request, 'uploadolevel.html', {'form': form})


def viewUploadedDataset(request):
    olevedatasets = OlevelDataset.objects.order_by('-uploaded_at')
    return render(request, 'viewolevels.html', {'olevedatasets': olevedatasets})


# *********************************************************************************************************************

# artifacts/views_olevel.py
from pathlib import Path
import json
import pandas as pd
from django.utils import timezone
from django.utils.text import slugify
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.http import JsonResponse, HttpResponseBadRequest
from django.shortcuts import render

from artifacts.o_level_predictors import (
    predict_with_olevel_rf,
    predict_with_olevel_lstm,
    predict_with_olevel_ensemble,
    required_columns as olevel_req_cols,
)


# ---------- Upload a CSV, run chosen model, save predictions ----------
@require_http_methods(["GET", "POST"])
def upload_and_predict_olevel(request):
    if request.method == "GET":
        return render(request, "olevel/olevel_upload.html", {
            "models": [("olevel_rf", "Random Forest"),
                       ("olevel_lstm", "LSTM"),
                       ("olevel_ens", "Ensemble (avg)")]
        })

    # POST
    f = request.FILES.get("dataset")
    model_choice = (request.POST.get("model") or "").strip()
    custom_label = (request.POST.get("save_as") or "").strip()

    if not f:
        return render(request, "olevel/olevel_upload.html", {"error": "Please choose a CSV."}, status=400)
    if model_choice not in {"olevel_rf", "olevel_lstm", "olevel_ens"}:
        return render(request, "olevel/olevel_upload.html", {"error": "Invalid model."}, status=400)

    try:
        df = pd.read_csv(f)
        if model_choice == "olevel_rf":
            out = predict_with_olevel_rf(df, top_k=3)
        elif model_choice == "olevel_lstm":
            out = predict_with_olevel_lstm(df, top_k=3)
        else:
            out = predict_with_olevel_ensemble(df, alpha=0.5, top_k=3)

        present = [c for c in PASSTHRU_COLS if c in df.columns]
        if present:
            out = pd.concat([df[present].reset_index(drop=True), out.reset_index(drop=True)], axis=1)

        # Save predictions
        stem = slugify(custom_label) or Path(getattr(f, "name", "dataset")).stem
        tag = model_choice
        ts = timezone.now().strftime("%Y%m%d_%H%M%S")
        rel_name = f"predictions/olevel_preds_{stem}_{tag}_{ts}.csv"
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        rel_path = default_storage.save(rel_name, ContentFile(csv_bytes))

        preview_html = out.head(60).to_html(classes="table table-striped table-sm", index=False)
        return render(request, "olevel/olevel_result.html", {
            "saved_name": Path(rel_path).name,
            "download_url": default_storage.url(rel_path),
            "preview_table": preview_html,
        })

    except Exception as e:
        return render(request, "olevel/olevel_upload.html", {"error": str(e)}, status=400)


# ---------- Predict-one (JSON) ----------
@csrf_exempt
@require_http_methods(["GET", "POST"])
def predict_one_olevel(request):
    """
    GET  -> schema (required columns)
    POST -> JSON with fields S1_*, S2_*, S3_* for all subjects
            { "model":"olevel_ens", "alpha":0.5, "scores": { "S1_Mathematics": 77, ... } }
    """
    if request.method == "GET":
        return JsonResponse({
            "ok": True,
            "usage": "POST JSON: {'model': 'olevel_ens|olevel_rf|olevel_lstm', 'alpha':0.5, <columns...>}",
            "required_columns": olevel_req_cols(),
            "subjects": list(o for o in olevel_req_cols() if o.startswith("S1_")),
            "models": ["olevel_ens", "olevel_rf", "olevel_lstm"],
            "example": {"model": "olevel_ens", "S1_Mathematics": 77, "S1_Physics": 75, "S1_Chemistry": 92, "...": "..."}
        })

    # POST
    try:
        if request.META.get("CONTENT_TYPE", "").startswith("application/json"):
            payload = json.loads(request.body.decode("utf-8"))
        else:
            payload = request.POST.dict()

        model_choice = payload.pop("model", "olevel_ens")
        alpha = float(payload.pop("alpha", 0.5))
        alpha = max(0.0, min(1.0, alpha))

        # Build single-row df from remaining numeric fields
        df = pd.DataFrame([payload]).apply(pd.to_numeric, errors="ignore")

        if model_choice == "olevel_rf":
            out = predict_with_olevel_rf(df, top_k=3)
        elif model_choice == "olevel_lstm":
            out = predict_with_olevel_lstm(df, top_k=3)
        else:
            out = predict_with_olevel_ensemble(df, alpha=alpha, top_k=3)

        row = out.iloc[0]
        return JsonResponse({
            "ok": True,
            "model": model_choice,
            "pred_stream": row["Pred_Stream"],
            "pred_label": int(row["Pred_Label"]),
            "topk_streams": row["TopK_Streams"],
            "topk_probs": [round(float(x), 4) for x in row["TopK_Probs"]],
        })
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=400)


# artifacts/views_olevel.py (append)
from django.shortcuts import render
from django.views.decorators.http import require_GET


@require_GET
def predict_one_olevel_ui(request):
    """Simple page to predict A-Level stream for one student."""
    return render(request, "olevel/predict_one_olevel.html")


# ***************************************************DASHBOARD**********************************************************
# ---------- O-Level Dashboard: helpers, APIs, page ----------
from typing import Any, Dict, List, Optional
import re
import numpy as np
import pandas as pd
from pathlib import Path
from django.conf import settings
from django.views.decorators.http import require_GET
from django.http import JsonResponse
from django.shortcuts import render

# Reuse O-Level predictors / metadata
from artifacts.o_level_predictors import (rf_model, required_columns as olevel_required_cols)

from artifacts.o_level_predictors import META as O_META, SUBJECTS as O_SUBJECTS, YEARS as O_YEARS, CLASSES as O_CLASSES


# --- Where O-Level prediction CSVs live (same /media/predictions dir as primary) ---
def _pred_dir() -> Path:
    p = Path(settings.MEDIA_ROOT) / "predictions"
    p.mkdir(parents=True, exist_ok=True)
    return p


# --- We’ll pass these through to saved predictions (see patch below) ---
PASSTHRU_COLS = ["Student_ID", "Province", "District", "Gender", "School_Location", "Academic_Year"]

# ----------------- Year parsing (robust) -----------------
YEAR_ALIAS_CANDS = [
    "Academic_Year", "academic_year", "Year", "year",
    "Exam_Year", "exam_year", "AcademicYear", "AY", "ay", "Year_Range", "ExamYear"
]
YEAR_REGEX = re.compile(r"(19\d{2}|20\d{2})")


def _find_year_col(df: pd.DataFrame) -> Optional[str]:
    for c in YEAR_ALIAS_CANDS:
        if c in df.columns:
            return c
    for c in df.columns:
        if "year" in c.lower():
            return c
    return None


def _year_numeric(s: pd.Series) -> pd.Series:
    y = pd.to_numeric(s, errors="coerce")
    if y.isna().mean() > 0.3:
        y = s.astype(str).str.extract(YEAR_REGEX, expand=False).pipe(pd.to_numeric, errors="coerce")
    return y


# ----------------- Filters helpers -----------------
FILTER_CANDS = {
    "Province": ["Province", "province"],
    "District": ["District", "district"],
    "Gender": ["Gender", "gender"],
    "School_Location": ["School_Location", "school_location"],
    "Academic_Year": ["Academic_Year", "academic_year", "Year", "year"],
}


def _first_col(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    return None


def _list_filter_options(df: pd.DataFrame) -> Dict[str, List[Any]]:
    out: Dict[str, List[Any]] = {}
    for key, cands in FILTER_CANDS.items():
        col = _first_col(df, cands)
        if not col:
            continue
        vals = sorted([v for v in df[col].dropna().unique().tolist() if str(v) != ""])
        out[key] = vals
    return out


def _apply_filters(df: pd.DataFrame, params) -> pd.DataFrame:
    work = df.copy()
    # Province, District, Gender, School_Location
    for key in ["Province", "District", "Gender", "School_Location"]:
        col = _first_col(work, FILTER_CANDS[key])
        if not col:
            continue
        vals = params.getlist(key.lower()) if hasattr(params, "getlist") else []
        if vals:
            work = work[work[col].isin(vals)]

    # Year range
    ycol = _find_year_col(work)
    if ycol and ycol in work.columns:
        yn = _year_numeric(work[ycol])
        y_min = (params.get("year_min") or "").strip()
        y_max = (params.get("year_max") or "").strip()
        mask = pd.Series(True, index=work.index)
        if y_min: mask &= yn >= pd.to_numeric(y_min, errors="coerce")
        if y_max: mask &= yn <= pd.to_numeric(y_max, errors="coerce")
        if y_min or y_max: mask &= yn.notna()
        work = work[mask]
    return work


# ----------------- Aggregations for multi-class -----------------
def _overall_stream_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    # Expected: "Pred_Stream" & per-class prob columns "Prob_<cls>"
    if "Pred_Stream" not in df.columns:
        return {"labels": [], "counts": [], "share": []}
    counts = df["Pred_Stream"].value_counts().reindex(O_CLASSES, fill_value=0)
    total = int(counts.sum()) or 1
    # Average confidence per predicted stream (use its own Prob_<cls>)
    avg_conf = []
    for cls in O_CLASSES:
        col = f"Prob_{cls}"
        sub = df[df["Pred_Stream"] == cls]
        if col in df.columns and len(sub):
            avg_conf.append(float(sub[col].mean()))
        else:
            avg_conf.append(0.0)
    return {
        "labels": O_CLASSES,
        "counts": counts.astype(int).tolist(),
        "share": [round(c / total, 4) for c in counts.tolist()],
        "avg_confidence": [round(x, 4) for x in avg_conf],
        "total": total
    }


def _grouped_stream_stack(df: pd.DataFrame, group_key: str, top_n: int = 12) -> Dict[str, Any]:
    """
    Returns stacked data to plot distribution of predicted streams across a grouping (e.g., Gender/Province).
    """
    col = _first_col(df, FILTER_CANDS[group_key])
    if not col or "Pred_Stream" not in df.columns:
        return {"groups": [], "series": []}
    # Pick top groups by size
    sizes = df[col].value_counts().head(top_n)
    labels = sizes.index.tolist()
    mat = []
    for cls in O_CLASSES:
        counts = (df[df[col].isin(labels)]["Pred_Stream"] == cls).groupby(df[col]).sum().reindex(labels, fill_value=0)
        mat.append(counts.astype(int).tolist())
    return {"groups": labels, "streams": O_CLASSES, "series": mat}


def _yearly_stream_series(df: pd.DataFrame) -> Dict[str, Any]:
    ycol = _find_year_col(df)
    if not ycol or "Pred_Stream" not in df.columns:
        return {}
    yn = _year_numeric(df[ycol])
    work = df.assign(_Year=yn).dropna(subset=["_Year"]).copy()
    work["_Year"] = work["_Year"].astype(int)
    # counts per class per year (sorted by year)
    years = sorted(work["_Year"].unique().tolist())
    series = {cls: [] for cls in O_CLASSES}
    for y in years:
        sub = work[work["_Year"] == y]
        cts = sub["Pred_Stream"].value_counts().reindex(O_CLASSES, fill_value=0)
        for cls in O_CLASSES:
            series[cls].append(int(cts[cls]))
    return {"years": years, "series": series}


# ----------------- RF subject importance (global & per year) -----------------
def _rf_subject_importance() -> Dict[str, Any]:
    """
    Map RF feature_importances_ (length = len(REQ_COLS)) to:
      - subject totals across S1..S3
      - per-year breakdown for each subject
    """
    rf = rf_model()  # trained on flattened [S1_* ... S2_* ... S3_*] in this exact order
    feats = olevel_required_cols()  # required columns order used at training time
    imps = getattr(rf, "feature_importances_", None)
    if imps is None or len(imps) != len(feats):
        return {"subjects": O_SUBJECTS, "total": [0.0] * len(O_SUBJECTS),
                "per_year": {y: [0.0] * len(O_SUBJECTS) for y in O_YEARS}}

    # Build per (year,subject)
    per_year = {y: {s: 0.0 for s in O_SUBJECTS} for y in O_YEARS}
    for name, val in zip(feats, imps):
        # name like "S2_Chemistry"
        try:
            y, subj = name.split("_", 1)
        except ValueError:
            continue
        if y in per_year and subj in per_year[y]:
            per_year[y][subj] += float(val)

    # Subject totals across years
    totals = {s: 0.0 for s in O_SUBJECTS}
    for y in O_YEARS:
        for s in O_SUBJECTS:
            totals[s] += per_year[y][s]

    # Pack arrays in subject order
    total_arr = [round(totals[s], 6) for s in O_SUBJECTS]
    per_year_arr = {y: [round(per_year[y][s], 6) for s in O_SUBJECTS] for y in O_YEARS}

    return {"subjects": O_SUBJECTS, "total": total_arr, "per_year": per_year_arr}


# ----------------- APIs -----------------
@require_GET
def olevel_prediction_files_api(request):
    """List O-Level prediction CSVs (newest first)."""
    pred_dir = _pred_dir()
    files = []
    for f in pred_dir.glob("*.csv"):
        # conventionally we used olevel_preds_* in the upload view
        if not f.name.startswith("olevel_preds_"):
            continue
        st = f.stat()
        files.append({"name": f.name, "size": st.st_size, "modified": st.st_mtime})
    files.sort(key=lambda x: x["modified"], reverse=True)
    return JsonResponse({"ok": True, "files": files})


@require_GET
def olevel_filters_api(request):
    """Return dropdown options (and year list/range) for the chosen file."""
    name = (request.GET.get("file") or "").strip()
    if not name.endswith(".csv"):
        return JsonResponse({"ok": False, "error": "Missing/invalid ?file"}, status=400)
    p = _pred_dir() / Path(name).name
    if not p.exists():
        return JsonResponse({"ok": False, "error": "File not found"}, status=404)

    df = pd.read_csv(p)
    opts = _list_filter_options(df)
    ycol = _find_year_col(df)
    years, year_range = None, None
    if ycol:
        yv = _year_numeric(df[ycol]).dropna().astype(int)
        if not yv.empty:
            years = sorted(yv.unique().tolist())
            year_range = {"min": int(yv.min()), "max": int(yv.max())}
    return JsonResponse({"ok": True, "filters": opts, "years": years, "year_range": year_range, "classes": O_CLASSES})


@require_GET
def olevel_dashboard_data_api(request):
    """
    Query args:
      file=<csv>  [required]
      province=.. (multi), district=.. (multi), gender=.. (multi), school_location=.. (multi)
      year_min=YYYY, year_max=YYYY
    """
    name = (request.GET.get("file") or "").strip()
    if not name.endswith(".csv"):
        return JsonResponse({"ok": False, "error": "Missing/invalid ?file"}, status=400)
    p = _pred_dir() / Path(name).name
    if not p.exists():
        return JsonResponse({"ok": False, "error": "File not found"}, status=404)

    df = pd.read_csv(p)
    if "Pred_Stream" not in df.columns:
        return JsonResponse({"ok": False, "error": "Pred_Stream column missing in file."}, status=400)

    fdf = _apply_filters(df, request.GET)

    payload: Dict[str, Any] = {"ok": True, "file": p.name}
    payload["overall"] = _overall_stream_distribution(fdf)
    payload["by_gender"] = _grouped_stream_stack(fdf, "Gender", top_n=3)
    payload["by_province"] = _grouped_stream_stack(fdf, "Province", top_n=10)
    payload["by_district"] = _grouped_stream_stack(fdf, "District", top_n=12)
    payload["yearly"] = _yearly_stream_series(fdf)

    # RF subject importance (global/per-year)
    try:
        payload["rf_importance"] = _rf_subject_importance()
    except Exception as e:
        payload["rf_importance_error"] = str(e)

    return JsonResponse(payload)


# ----------------- Page -----------------
def olevel_dashboard_page(request):
    return render(request, "olevel/olevel_dashboard.html")
