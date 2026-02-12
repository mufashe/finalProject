from __future__ import annotations
# # artifacts/views.py
# import os, io
# import pandas as pd
# from itertools import chain
# from django.apps import apps
# from django.contrib import messages
# from django.shortcuts import render
# from django.views.decorators.http import require_http_methods
# from django.conf import settings
# from django.utils import timezone
#
# from configurations.models import EducationLevel
# from .forms import InferenceForm
# from .models import ModelArtifact, InferenceJob
# from .utils import load_model, run_classification, run_forecasting
#
# TPL_RUN = "run.html"  # or "artifacts/run.html" if you use subfolders
# TPL_JOB = "artifacts_job_detail.html"  # or "artifacts/job_detail.html"
#
#
# def _collect_uploads(level=None):
#     def rows_for(model_label, source_name):
#         try:
#             Model = apps.get_model(model_label)
#         except LookupError:
#             return []
#         qs = Model.objects.select_related("level").all()
#         if level:
#             qs = qs.filter(level=level)
#         out = []
#         for obj in qs:
#             f = getattr(obj, "file", None)
#             if not f:
#                 continue
#             label = getattr(obj, "original_name",
#                             os.path.basename(getattr(f, "name", "")) or f"{source_name} #{obj.pk}")
#             out.append({
#                 "token": f"{model_label}:{obj.pk}",  # we POST this token, not a path
#                 "label": label,
#                 "level": getattr(obj.level, "name", "—"),
#                 "source": source_name,
#             })
#         return out
#
#     return list(chain(
#         rows_for("primary.PrimaryUpload", "Primary"),
#         rows_for("o_level.OLevelUpload", "O-Level"),
#         rows_for("a_level.ALevelUpload", "A-Level"),
#     ))
#
#
# @require_http_methods(["GET", "POST"])
# def run_inference(request):
#     form = InferenceForm(request.POST or None)
#
#     # Level filter (support GET or POST). Never touch cleaned_data on GET.
#     level_id = (request.POST.get("level") or request.GET.get("level") or "").strip()
#     level_obj = None
#     if level_id:
#         try:
#             level_obj = EducationLevel.objects.get(pk=int(level_id))
#         except (EducationLevel.DoesNotExist, ValueError, TypeError):
#             level_obj = None
#
#     # Filter artifact choices by level
#     art_qs = ModelArtifact.objects.filter(active=True).order_by("-created_at")
#     if level_obj:
#         art_qs = art_qs.filter(level=level_obj)
#         if request.method == "GET":
#             form.initial["level"] = level_obj.pk
#     form.fields["artifact"].queryset = art_qs
#
#     # Populate dataset choices (token-based)
#     dataset_rows = _collect_uploads(level=level_obj)
#     form.set_dataset_choices(dataset_rows)
#
#     # On GET, just render the form. ALWAYS return.
#     if request.method == "GET":
#         # UX: nudge when nothing to run
#         if not art_qs.exists():
#             messages.warning(request, "No saved models for this level yet. Add one on the Saved Models page.")
#         if not dataset_rows:
#             messages.warning(request, "No uploaded datasets found for this level.")
#         return render(request, TPL_RUN, {"form": form})
#
#     # POST: validate first. If invalid, re-render with errors (ALWAYS return).
#     if not form.is_valid():
#         return render(request, TPL_RUN, {"form": form})
#
#     # Safe to read cleaned_data now
#     artifact = form.cleaned_data["artifact"]
#     dataset_token = form.cleaned_data["dataset"]
#     top_k = form.cleaned_data.get("top_k") or 3
#     horizon = form.cleaned_data.get("horizon") or 12
#
#     # Resolve token -> object -> file
#     try:
#         model_label, pk_str = dataset_token.split(":", 1)
#         Model = apps.get_model(model_label)
#         ds_obj = Model.objects.get(pk=int(pk_str))
#         f = getattr(ds_obj, "file", None)
#         if not f:
#             raise ValueError("Selected dataset has no file attached.")
#     except Exception as e:
#         messages.error(request, f"Could not resolve dataset: {e}")
#         return render(request, TPL_RUN, {"form": form})
#
#     # Read dataset (path-safe / storage-safe)
#     try:
#         df = None
#         dataset_path = None
#         try:
#             dataset_path = f.path  # may not exist on non-local storage
#         except Exception:
#             dataset_path = None
#
#         name_lower = (getattr(f, "name", "") or "").lower()
#         if dataset_path and os.path.exists(dataset_path):
#             lower = dataset_path.lower()
#             if lower.endswith((".xlsx", ".xls")):
#                 df = pd.read_excel(dataset_path)
#             elif lower.endswith(".csv"):
#                 df = pd.read_csv(dataset_path)
#
#         if df is None:
#             f.open("rb")
#             data = f.read()
#             f.close()
#             bio = io.BytesIO(data)
#             if name_lower.endswith((".xlsx", ".xls")):
#                 df = pd.read_excel(bio)
#             elif name_lower.endswith(".csv"):
#                 df = pd.read_csv(bio)
#             else:
#                 df = pd.read_excel(bio)  # default to Excel parser
#     except Exception as e:
#         messages.error(request, f"Failed to read dataset: {e}")
#         return render(request, TPL_RUN, {"form": form})
#
#     # Load model
#     try:
#         model = load_model(artifact.file.path, artifact.model_format)
#     except Exception as e:
#         messages.error(request, f"Failed to load model: {e}")
#         return render(request, TPL_RUN, {"form": form})
#
#     # Create job record
#     job = InferenceJob.objects.create(
#         artifact=artifact,
#         dataset_label=os.path.basename(getattr(f, "name", "dataset")),
#         dataset_path=getattr(f, "name", ""),
#         params={"top_k": top_k, "horizon": horizon},
#     )
#
#     # Run inference
#     try:
#         if artifact.task == "classification":
#             out = run_classification(
#                 model, df,
#                 feature_list=artifact.feature_list, want_proba=True, top_k=top_k
#             )
#             id_cols = [c for c in ["Student_ID", "student_id", "id"] if c in df.columns]
#             res = pd.concat([df[id_cols], out], axis=1) if id_cols else out
#
#         elif artifact.task == "forecasting":
#             out = run_forecasting(model, df, horizon=horizon)
#             res = out.reset_index()
#
#         else:
#             raise ValueError(f"Unknown task: {artifact.task}")
#
#         # Save CSV under MEDIA/inference/<level>/
#         ts = timezone.now().strftime("%Y%m%d_%H%M%S")
#         out_name = f"{artifact.name.replace(' ', '_').lower()}__{ts}.csv"
#         out_dir = os.path.join(settings.MEDIA_ROOT, "inference", artifact.level.slug)
#         os.makedirs(out_dir, exist_ok=True)
#         out_path = os.path.join(out_dir, out_name)
#         res.to_csv(out_path, index=False)
#
#         rel_path = os.path.relpath(out_path, settings.MEDIA_ROOT)
#         job.output_csv.name = rel_path
#         job.ok = True
#         job.message = f"OK: {len(res)} rows."
#         job.save(update_fields=["output_csv", "ok", "message", "finished_at"])
#
#         preview = res.head(30).to_html(index=False, border=0)
#         messages.success(request, f"Inference completed. {len(res)} rows. Download below.")
#         return render(request, TPL_JOB, {"job": job, "preview_html": preview})
#
#     except Exception as e:
#         job.ok = False
#         job.message = f"Failed: {e}"
#         job.save(update_fields=["ok", "message", "finished_at"])
#         messages.error(request, f"Inference failed: {e}")
#         return render(request, TPL_RUN, {"form": form})
#
#
# # *********************************************************************************************************************
# # artifacts/views.py
# import io
# import pandas as pd
# from django.shortcuts import render
# from django.http import HttpResponse, HttpResponseBadRequest
# from django.core.files.base import ContentFile
# from django.core.files.storage import default_storage
#
# from .forms import DatasetUploadForm
# from .predictors import predict_with_pipeline, predict_with_lstm
#
# # artifacts/views.py
# import io
# from pathlib import Path
# from django.shortcuts import render
# from django.http import HttpResponseBadRequest
# from django.core.files.base import ContentFile
# from django.core.files.storage import default_storage
# from django.utils.text import slugify
# from django.utils import timezone
# import pandas as pd
#
# from .forms import DatasetUploadForm
# from .predictors import predict_with_pipeline, predict_with_lstm
#
#
# def upload_and_predict(request):
#     if request.method == "POST":
#         form = DatasetUploadForm(request.POST, request.FILES)
#         if not form.is_valid():
#             return render(request, "artifacts/upload.html", {"form": form}, status=400)
#
#         f = form.cleaned_data["dataset"]
#         model_choice = form.cleaned_data["model"]
#         custom_label = form.cleaned_data.get("save_as") or ""
#
#         try:
#             df = pd.read_csv(f)
#
#             if model_choice in ("primary_rf", "primary_logreg"):
#                 out = predict_with_pipeline(df, which=model_choice)
#             elif model_choice == "primary_lstm":
#                 out = predict_with_lstm(df)
#             else:
#                 return HttpResponseBadRequest("Unknown model selection.")
#
#             # --- Build nice filename ---
#             # Base priority: custom label (if given) → uploaded file stem → 'preds'
#             uploaded_stem = Path(getattr(f, "name", "dataset")).stem
#             base = slugify(custom_label) or slugify(uploaded_stem) or "preds"
#             tag = slugify(model_choice)  # e.g., primary-rf
#             ts = timezone.now().strftime("%Y%m%d_%H%M%S")
#             # Final name: preds_<base>_<tag>_<ts>.csv
#             rel_name = f"predictions/preds_{base}_{tag}_{ts}.csv"
#
#             csv_bytes = out.to_csv(index=False).encode("utf-8")
#             rel_path = default_storage.save(rel_name, ContentFile(csv_bytes))
#             download_url = default_storage.url(rel_path)
#
#             preview_html = out.head(50).to_html(classes="table table-striped table-sm", index=False)
#
#             return render(
#                 request,
#                 "artifacts/prediction_result.html",
#                 {
#                     "preview_table": preview_html,
#                     "download_url": download_url,
#                     "saved_name": Path(rel_path).name,
#                 },
#             )
#         except Exception as e:
#             return render(request, "artifacts/upload.html", {"form": form, "error": str(e)}, status=400)
#
#     # GET
#     form = DatasetUploadForm()
#     return render(request, "artifacts/upload.html", {"form": form})
#
#
# # **********************************************************************************************************************
#
#
# # artifacts/views.py
# import json
# import pandas as pd
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# from django.views.decorators.http import require_http_methods
# from .predictors import (
#     predict_with_pipeline, predict_with_lstm,
#     REQ_PIPE_COLS, SEQ_ORDER
# )
#
#
# @csrf_exempt
# @require_http_methods(["GET", "POST"])
# def predict_one(request):
#     # GET -> show usage so browser GET doesn’t 405
#     if request.method == "GET":
#         return JsonResponse({
#             "ok": True,
#             "usage": "POST JSON with 'model' and required fields.",
#             "models": ["primary_rf", "primary_logreg", "primary_lstm"],
#             "required_columns": {
#                 "primary_rf": REQ_PIPE_COLS,
#                 "primary_logreg": REQ_PIPE_COLS,
#                 "primary_lstm": SEQ_ORDER
#             },
#             "note": "LSTM requires only P1–P5 subject scores (no demographics)."
#         })
#
#     # POST -> run prediction
#     try:
#         if request.META.get("CONTENT_TYPE", "").startswith("application/json"):
#             payload = json.loads(request.body.decode("utf-8"))
#         else:
#             payload = request.POST.dict()
#
#         model_choice = payload.pop("model", "primary_rf")
#         threshold = float(payload.pop("threshold", 0.5))  # <— NEW: optional
#         threshold = max(0.0, min(1.0, threshold))  # clamp to [0,1]
#
#         df = pd.DataFrame([payload]).apply(pd.to_numeric, errors="ignore")
#
#         if model_choice in ("primary_rf", "primary_logreg"):
#             out = predict_with_pipeline(df, which=model_choice)
#         elif model_choice == "primary_lstm":
#             out = predict_with_lstm(df)
#         else:
#             return JsonResponse({"ok": False, "error": "Unknown model"}, status=400)
#
#         row = out.iloc[0]
#         prob = float(row["Prob"])
#         pred = int(prob >= threshold)  # <— use custom threshold
#         label = "Pass" if pred == 1 else "Not Pass"
#
#         return JsonResponse({
#             "ok": True,
#             "model": model_choice,
#             "threshold": threshold,
#             "pred": pred,
#             "label": label,
#             "prob": round(prob, 4),
#         })
#     except Exception as e:
#         return JsonResponse({"ok": False, "error": str(e)}, status=500)
#
#
# def predict_one_ui(request):
#     return render(request, "artifacts/predict_one_ui.html")
#
#
# # ****************************************************************************************************************
# # artifacts/views.py
# from pathlib import Path
# import pandas as pd
# import numpy as np
# from django.http import JsonResponse
# from django.shortcuts import render
# from django.views.decorators.http import require_GET
# from django.conf import settings
#
# # If you want subject importance charts from saved pipelines:
# from .ml_registry import load_pipeline
#
# # ---------- Constants ----------
# SUBJECTS = ["Kinyarwanda", "English", "Mathematics", "Science", "Social_Studies", "Creative_Arts"]
# GRADES_ALL = ["P1", "P2", "P3", "P4", "P5", "P6"]
# SUBJECT_COLS_ALL = [f"{s}_{g}" for g in GRADES_ALL for s in SUBJECTS]
# CAT_COLS = ["Gender", "School_Location", "Residence_Location", "Has_Electricity", "Parental_Education_Level"]
#
# FILTER_COLS = {
#     "Province": ["Province", "province"],
#     "District": ["District", "district"],
#     "Gender": ["Gender", "gender"],
#     "School_Location": ["School_Location", "school_location"],
#     "Academic_Year": ["Academic_Year", "academic_year", "Year", "year"],
# }
#
#
# # ---------- Helpers ----------
# def _predictions_dir() -> Path:
#     p = Path(settings.MEDIA_ROOT) / "predictions"
#     p.mkdir(parents=True, exist_ok=True)
#     return p
#
#
# def _first_col(df, candidates):
#     for c in candidates:
#         if c in df.columns:
#             return c
#     return None
#
#
# def _list_filter_options(df: pd.DataFrame):
#     opts = {}
#     for key, cands in FILTER_COLS.items():
#         col = _first_col(df, cands)
#         if not col:
#             continue
#         vals = sorted([v for v in df[col].dropna().unique().tolist() if str(v) != ""])
#         opts[key] = vals
#     return opts
#
#
# def _apply_filters(df: pd.DataFrame, params):
#     work = df.copy()
#
#     # Province & District
#     prov_col = _first_col(work, FILTER_COLS["Province"])
#     dist_col = _first_col(work, FILTER_COLS["District"])
#     if prov_col:
#         prov = params.getlist("province") if hasattr(params, "getlist") else []
#         if prov:
#             work = work[work[prov_col].isin(prov)]
#     if dist_col:
#         dist = params.getlist("district") if hasattr(params, "getlist") else []
#         if dist:
#             work = work[work[dist_col].isin(dist)]
#
#     # Gender
#     gen_col = _first_col(work, FILTER_COLS["Gender"])
#     if gen_col:
#         g = params.getlist("gender") if hasattr(params, "getlist") else []
#         if g:
#             work = work[work[gen_col].isin(g)]
#
#     # School Location
#     loc_col = _first_col(work, FILTER_COLS["School_Location"])
#     if loc_col:
#         loc = params.getlist("school_location") if hasattr(params, "getlist") else []
#         if loc:
#             work = work[work[loc_col].isin(loc)]
#
#     # Academic Year range
#     yr_col = _first_col(work, FILTER_COLS["Academic_Year"])
#     if yr_col and yr_col in work.columns:
#         try:
#             work[yr_col] = pd.to_numeric(work[yr_col], errors="coerce")
#         except Exception:
#             pass
#         y_min = params.get("year_min")
#         y_max = params.get("year_max")
#         if y_min:
#             work = work[work[yr_col] >= pd.to_numeric(y_min, errors="coerce")]
#         if y_max:
#             work = work[work[yr_col] <= pd.to_numeric(y_max, errors="coerce")]
#
#     return work
#
#
# def _aggregate_pass_rates(df):
#     out = {}
#
#     # Probability histogram (10 bins)
#     bins = np.linspace(0, 1, 11)
#     hist, edges = np.histogram(df["Prob"].values, bins=bins)
#     out["prob_hist"] = {"edges": edges.tolist(), "counts": hist.tolist()}
#
#     # Province aggregates + map dict
#     prov_col = _first_col(df, FILTER_COLS["Province"])
#     if prov_col:
#         g = df.groupby(prov_col)["Pred"].agg(["mean", "size"]).sort_values("mean", ascending=False)
#         counts = df.groupby([prov_col, "Pred"]).size().unstack(fill_value=0)
#         out["province"] = {
#             "labels": g.index.tolist(),
#             "pass_rate": g["mean"].round(4).tolist(),
#             "passed": counts.get(1, pd.Series(0, index=counts.index)).tolist(),
#             "failed": counts.get(0, pd.Series(0, index=counts.index)).tolist(),
#         }
#         out["province_map"] = {k: float(v) for k, v in df.groupby(prov_col)["Pred"].mean().to_dict().items()}
#
#     # District aggregates + map dict
#     dist_col = _first_col(df, FILTER_COLS["District"])
#     if dist_col:
#         totals = df.groupby(dist_col).size().sort_values(ascending=False)
#         top_labels = totals.head(15).index
#         sub = df[df[dist_col].isin(top_labels)]
#         g = sub.groupby(dist_col)["Pred"].mean().loc[top_labels]
#         counts = sub.groupby([dist_col, "Pred"]).size().unstack(fill_value=0).loc[top_labels]
#         out["district_top"] = {
#             "labels": top_labels.tolist(),
#             "pass_rate": g.round(4).tolist(),
#             "passed": counts.get(1, pd.Series(0, index=counts.index)).tolist(),
#             "failed": counts.get(0, pd.Series(0, index=counts.index)).tolist(),
#         }
#         out["district_map"] = {k: float(v) for k, v in df.groupby(dist_col)["Pred"].mean().to_dict().items()}
#
#     # Gender
#     gen_col = _first_col(df, FILTER_COLS["Gender"])
#     if gen_col:
#         g = df.groupby(gen_col)["Pred"].mean()
#         out["gender"] = {
#             "labels": g.index.astype(str).tolist(),
#             "pass_rate": g.round(4).tolist(),
#         }
#
#     # School location
#     loc_col = _first_col(df, FILTER_COLS["School_Location"])
#     if loc_col:
#         g = df.groupby(loc_col)["Pred"].mean()
#         out["location"] = {
#             "labels": g.index.astype(str).tolist(),
#             "pass_rate": g.round(4).tolist(),
#         }
#
#     return out
#
#
# def _timeseries_blocks(df):
#     res = {}
#     yr_col = _first_col(df, FILTER_COLS["Academic_Year"])
#     if not yr_col:
#         return res
#
#     series_df = df.copy()
#     series_df[yr_col] = pd.to_numeric(series_df[yr_col], errors="coerce")
#     series_df = series_df.dropna(subset=[yr_col])
#
#     g = series_df.groupby(yr_col)["Pred"].mean().sort_index()
#     res["overall_yearly"] = [{"year": int(k), "pass_rate": float(v)} for k, v in g.items()]
#
#     prov_col = _first_col(series_df, FILTER_COLS["Province"])
#     if prov_col:
#         remaining = series_df[prov_col].dropna().unique().tolist()
#         if len(remaining) == 1:
#             sel = remaining[0]
#             gp = series_df.groupby([yr_col, prov_col])["Pred"].mean().unstack()
#             if sel in gp.columns:
#                 res["province_yearly"] = [{"year": int(y), "pass_rate": float(r)} for y, r in gp[sel].dropna().items()]
#                 res["province_name"] = sel
#
#     dist_col = _first_col(series_df, FILTER_COLS["District"])
#     if dist_col:
#         remaining = series_df[dist_col].dropna().unique().tolist()
#         if len(remaining) == 1:
#             sel = remaining[0]
#             gd = series_df.groupby([yr_col, dist_col])["Pred"].mean().unstack()
#             if sel in gd.columns:
#                 res["district_yearly"] = [{"year": int(y), "pass_rate": float(r)} for y, r in gd[sel].dropna().items()]
#                 res["district_name"] = sel
#
#     return res
#
#
# # Robust subject-importance extractor (RF/LR)
# def _subject_importance_from_pipeline(which="primary_rf"):
#     import numpy as np
#     pipe = load_pipeline(which)
#     pre = pipe.named_steps.get("pre")
#     clf = pipe.named_steps.get("clf")
#     if pre is None or clf is None:
#         raise RuntimeError("Pipeline must have steps named 'pre' (ColumnTransformer) and 'clf'.")
#
#     try:
#         feat_names = pre.get_feature_names_out()
#     except Exception:
#         # Fallback names if very old sklearn
#         feat_names = np.array([f"num__{c}" for c in SUBJECT_COLS_ALL])
#
#     feat_names = np.array(feat_names)
#
#     if hasattr(clf, "feature_importances_"):
#         imp_all = np.asarray(clf.feature_importances_, dtype=float)
#     elif hasattr(clf, "coef_"):
#         imp_all = np.abs(np.asarray(clf.coef_).ravel())
#     else:
#         return {"labels": SUBJECTS, "scores": [0.0] * len(SUBJECTS)}
#
#     if imp_all.shape[0] != feat_names.shape[0]:
#         raise RuntimeError(f"Importance length {imp_all.shape[0]} != features {feat_names.shape[0]}")
#
#     num_mask = np.array([name.startswith("num__") for name in feat_names])
#     imp_num = imp_all[num_mask]
#     num_names = feat_names[num_mask]  # e.g., num__Kinyarwanda_P3
#
#     subj_scores = {s: 0.0 for s in SUBJECTS}
#     for name, val in zip(num_names, imp_num):
#         col = name.split("__", 1)[1]  # Kinyarwanda_P3
#         subj = col.split("_", 1)[0]
#         subj_scores[subj] += float(val)
#
#     items = sorted(subj_scores.items(), key=lambda x: x[1], reverse=True)
#     return {"labels": [k for k, _ in items], "scores": [round(v, 6) for _, v in items]}
#
#
# # ---------- APIs ----------
# @require_GET
# def prediction_files_api(request):
#     pred_dir = _predictions_dir()
#     files = []
#     for f in pred_dir.glob("*.csv"):
#         stat = f.stat()
#         files.append({"name": f.name, "size": stat.st_size, "modified": stat.st_mtime})
#     files.sort(key=lambda x: x["modified"], reverse=True)
#     return JsonResponse({"ok": True, "files": files})
#
#
# @require_GET
# def filters_api(request):
#     pred_dir = _predictions_dir()
#     requested = request.GET.get("file", "").strip()
#     if not requested:
#         return JsonResponse({"ok": False, "error": "Missing ?file parameter."}, status=400)
#
#     name = Path(requested).name
#     if name != requested or not name.endswith(".csv"):
#         return JsonResponse({"ok": False, "error": "Invalid filename."}, status=400)
#
#     chosen = pred_dir / name
#     if not chosen.exists():
#         return JsonResponse({"ok": False, "error": f"File not found: {name}"}, status=404)
#
#     df = pd.read_csv(chosen)
#     if "Pred" not in df.columns or "Prob" not in df.columns:
#         return JsonResponse({"ok": False, "error": "'Pred' or 'Prob' missing."}, status=400)
#
#     opts = _list_filter_options(df)
#     yr_col = _first_col(df, FILTER_COLS["Academic_Year"])
#     year_range = None
#     if yr_col:
#         yrs = pd.to_numeric(df[yr_col], errors="coerce").dropna().astype(int)
#         if not yrs.empty:
#             year_range = {"min": int(yrs.min()), "max": int(yrs.max())}
#
#     return JsonResponse({"ok": True, "filters": opts, "year_range": year_range})
#
#
# @require_GET
# def dashboard_data_api(request):
#     """
#     GET:
#       file=<filename.csv>  (required)
#       province=.. (multi)  district=.. (multi)  gender=.. (multi)
#       school_location=.. (multi)  year_min=YYYY  year_max=YYYY
#     """
#     pred_dir = _predictions_dir()
#     requested = request.GET.get("file", "").strip()
#     if not requested:
#         return JsonResponse({"ok": False, "error": "Missing ?file parameter."}, status=400)
#
#     name = Path(requested).name
#     if name != requested or not name.endswith(".csv"):
#         return JsonResponse({"ok": False, "error": "Invalid filename."}, status=400)
#
#     chosen = pred_dir / name
#     if not (chosen.exists() and chosen.is_file()):
#         return JsonResponse({"ok": False, "error": f"File not found: {name}"}, status=404)
#
#     df = pd.read_csv(chosen)
#     if "Pred" not in df.columns or "Prob" not in df.columns:
#         return JsonResponse({"ok": False, "error": f"'Pred' or 'Prob' missing in {chosen.name}."}, status=400)
#
#     fdf = _apply_filters(df, request.GET)
#
#     payload = {"ok": True, "file": chosen.name}
#     payload["overall"] = {
#         "total": int(len(fdf)),
#         "pass_rate": round(float(fdf["Pred"].mean()) if len(fdf) else 0.0, 4),
#         "avg_prob": round(float(fdf["Prob"].mean()) if len(fdf) else 0.0, 4)
#     }
#     payload["groups"] = _aggregate_pass_rates(fdf)
#
#     try:
#         payload["importance_rf"] = _subject_importance_from_pipeline("primary_rf")
#     except Exception as e:
#         payload["importance_rf_error"] = str(e)
#     try:
#         payload["importance_lr"] = _subject_importance_from_pipeline("primary_logreg")
#     except Exception as e:
#         payload["importance_lr_error"] = str(e)
#
#     payload["timeseries"] = _timeseries_blocks(fdf)
#     return JsonResponse(payload)
#
#
# # ---------- Page ----------
# def dashboard_page(request):
#     return render(request, "artifacts/dashboard.html")
#
#
# # **********************************************************************************************************************
# # artifacts/views.py  (new helpers & listing API)
#
# # def _predictions_dir() -> Path:
# #     from django.conf import settings
# #     p = Path(settings.MEDIA_ROOT) / "predictions"
# #     p.mkdir(parents=True, exist_ok=True)
# #     return p
#
#
# @require_GET
# def prediction_files_api(request):
#     """
#     Return available prediction CSV files under /media/predictions/ (newest first).
#     """
#     pred_dir = _predictions_dir()
#     files = []
#     for f in pred_dir.glob("*.csv"):
#         stat = f.stat()
#         files.append({
#             "name": f.name,
#             "size": stat.st_size,
#             "modified": stat.st_mtime,  # epoch seconds
#         })
#     files.sort(key=lambda x: x["modified"], reverse=True)
#     return JsonResponse({"ok": True, "files": files})
#
#
# # *****************************************************************************************************************
# from django.views.decorators.http import require_GET
# from django.http import JsonResponse
# from pathlib import Path
# import pandas as pd
# import numpy as np
#
# # Reuse your SUBJECTS/GRADES_* if already declared
# SUBJECTS = ["Kinyarwanda", "English", "Mathematics", "Science", "Social_Studies", "Creative_Arts"]
# GRADES_ALL = ["P1", "P2", "P3", "P4", "P5", "P6"]
# SUBJECT_COLS_ALL = [f"{s}_{g}" for g in GRADES_ALL for s in SUBJECTS]
#
# FILTER_COLS = {
#     "Province": ["Province", "province"],
#     "District": ["District", "district"],
#     "Gender": ["Gender", "gender"],
#     "School_Location": ["School_Location", "school_location"],
#     "Academic_Year": ["Academic_Year", "academic_year", "Year", "year"],
# }
#
#
# def _first_col(df, candidates):
#     for c in candidates:
#         if c in df.columns:
#             return c
#     return None
#
#
# def _list_filter_options(df: pd.DataFrame):
#     """Return unique values (sorted) for each filterable column in df."""
#     opts = {}
#     for key, cands in FILTER_COLS.items():
#         col = _first_col(df, cands)
#         if not col:
#             continue
#         vals = sorted([v for v in df[col].dropna().unique().tolist() if str(v) != ""])
#         opts[key] = vals
#     return opts
#
#
# def _apply_filters(df: pd.DataFrame, params):
#     """Apply query param filters to df and return filtered df."""
#     work = df.copy()
#
#     # Province & District
#     prov_col = _first_col(work, FILTER_COLS["Province"])
#     dist_col = _first_col(work, FILTER_COLS["District"])
#     if prov_col:
#         prov = params.getlist("province") if hasattr(params, "getlist") else []
#         if prov:
#             work = work[work[prov_col].isin(prov)]
#     if dist_col:
#         dist = params.getlist("district") if hasattr(params, "getlist") else []
#         if dist:
#             work = work[work[dist_col].isin(dist)]
#
#     # Gender
#     gen_col = _first_col(work, FILTER_COLS["Gender"])
#     if gen_col:
#         g = params.getlist("gender") if hasattr(params, "getlist") else []
#         if g:
#             work = work[work[gen_col].isin(g)]
#
#     # School Location
#     loc_col = _first_col(work, FILTER_COLS["School_Location"])
#     if loc_col:
#         loc = params.getlist("school_location") if hasattr(params, "getlist") else []
#         if loc:
#             work = work[work[loc_col].isin(loc)]
#
#     # Academic Year range
#     yr_col = _first_col(work, FILTER_COLS["Academic_Year"])
#     if yr_col and yr_col in work.columns:
#         try:
#             work[yr_col] = pd.to_numeric(work[yr_col], errors="coerce")
#         except Exception:
#             pass
#         y_min = params.get("year_min")
#         y_max = params.get("year_max")
#         if y_min:
#             work = work[work[yr_col] >= pd.to_numeric(y_min, errors="coerce")]
#         if y_max:
#             work = work[work[yr_col] <= pd.to_numeric(y_max, errors="coerce")]
#
#     return work
#
#
# def _timeseries_blocks(df):
#     """
#     Build time series if Academic_Year exists.
#     Returns overall yearly, plus per-selected province/district if filters choose one.
#     """
#     res = {}
#     yr_col = _first_col(df, FILTER_COLS["Academic_Year"])
#     if not yr_col:
#         return res
#
#     # Ensure numeric year for sorting
#     series_df = df.copy()
#     series_df[yr_col] = pd.to_numeric(series_df[yr_col], errors="coerce")
#     series_df = series_df.dropna(subset=[yr_col])
#
#     # overall
#     g = series_df.groupby(yr_col)["Pred"].mean().sort_index()
#     res["overall_yearly"] = [{"year": int(k), "pass_rate": float(v)} for k, v in g.items()]
#
#     # if a single province is selected, add its trend
#     prov_col = _first_col(series_df, FILTER_COLS["Province"])
#     if prov_col:
#         sel_prov = None
#         # We infer selection if only one province remains in df
#         remaining = series_df[prov_col].dropna().unique().tolist()
#         if len(remaining) == 1:
#             sel_prov = remaining[0]
#             gp = series_df.groupby([yr_col, prov_col])["Pred"].mean().unstack()
#             if sel_prov in gp.columns:
#                 res["province_yearly"] = [
#                     {"year": int(y), "pass_rate": float(r)} for y, r in gp[sel_prov].dropna().items()
#                 ]
#                 res["province_name"] = sel_prov
#
#     # if a single district is selected, add its trend
#     dist_col = _first_col(series_df, FILTER_COLS["District"])
#     if dist_col:
#         remaining = series_df[dist_col].dropna().unique().tolist()
#         if len(remaining) == 1:
#             sel_dist = remaining[0]
#             gd = series_df.groupby([yr_col, dist_col])["Pred"].mean().unstack()
#             if sel_dist in gd.columns:
#                 res["district_yearly"] = [
#                     {"year": int(y), "pass_rate": float(r)} for y, r in gd[sel_dist].dropna().items()
#                 ]
#                 res["district_name"] = sel_dist
#
#     return res
#
#
# @require_GET
# def filters_api(request):
#     """
#     Return filter choices for a given file (so the UI can populate dropdowns).
#     GET: ?file=<filename.csv>
#     """
#     pred_dir = _predictions_dir()
#     requested = request.GET.get("file", "").strip()
#     if not requested:
#         return JsonResponse({"ok": False, "error": "Missing ?file parameter."}, status=400)
#
#     name = Path(requested).name
#     if name != requested or not name.endswith(".csv"):
#         return JsonResponse({"ok": False, "error": "Invalid filename."}, status=400)
#
#     chosen = pred_dir / name
#     if not chosen.exists():
#         return JsonResponse({"ok": False, "error": f"File not found: {name}"}, status=404)
#
#     df = pd.read_csv(chosen)
#     if "Pred" not in df.columns or "Prob" not in df.columns:
#         return JsonResponse({"ok": False, "error": "'Pred' or 'Prob' missing."}, status=400)
#
#     opts = _list_filter_options(df)
#     # Provide min/max year if Academic_Year exists
#     yr_col = _first_col(df, FILTER_COLS["Academic_Year"])
#     year_range = None
#     if yr_col:
#         yrs = pd.to_numeric(df[yr_col], errors="coerce").dropna().astype(int)
#         if not yrs.empty:
#             year_range = {"min": int(yrs.min()), "max": int(yrs.max())}
#
#     return JsonResponse({"ok": True, "filters": opts, "year_range": year_range})


# artifacts/views.py
from __future__ import annotations

import io
import os
import json
import re
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from django.apps import apps
from django.conf import settings
from django.contrib import messages
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.http import JsonResponse, HttpResponseBadRequest
from django.shortcuts import render
from django.utils import timezone
from django.utils.text import slugify
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods

from configurations.models import EducationLevel
from .forms import InferenceForm, DatasetUploadForm
from .models import ModelArtifact, InferenceJob
from .utils import load_model, run_classification, run_forecasting
from .predictors import predict_with_pipeline, predict_with_lstm
from .ml_registry import load_pipeline  # for subject-importance charts

# --------------------------------------------------------------------------------------
# Templates
# --------------------------------------------------------------------------------------
TPL_RUN = "run.html"  # or "artifacts/run.html"
TPL_JOB = "artifacts_job_detail.html"  # or "artifacts/job_detail.html"

# --------------------------------------------------------------------------------------
# Constants (features & filters used by dashboard and importance charts)
# --------------------------------------------------------------------------------------
SUBJECTS = ["Kinyarwanda", "English", "Mathematics", "Science", "Social_Studies", "Creative_Arts"]
GRADES_ALL = ["P1", "P2", "P3", "P4", "P5", "P6"]
SUBJECT_COLS_ALL = [f"{s}_{g}" for g in GRADES_ALL for s in SUBJECTS]
CAT_COLS = ["Gender", "School_Location", "Residence_Location", "Has_Electricity", "Parental_Education_Level"]

FILTER_COLS = {
    "Province": ["Province", "province"],
    "District": ["District", "district"],
    "Gender": ["Gender", "gender"],
    "School_Location": ["School_Location", "school_location"],
    # We keep Academic_Year list for legacy; robust find happens via _find_year_column
    "Academic_Year": ["Academic_Year", "academic_year", "Year", "year"],
}

# Extra aliases for robust year detection
YEAR_ALIAS_CANDS = [
    "Academic_Year", "academic_year", "Year", "year",
    "Exam_Year", "exam_year", "AcademicYear",
    "AY", "ay", "Year_Range", "ExamYear"
]
YEAR_REGEX = re.compile(r"(19\d{2}|20\d{2})")  # first 4-digit year ~ 1900–2099


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _predictions_dir() -> Path:
    p = Path(settings.MEDIA_ROOT) / "predictions"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _find_year_column(df: pd.DataFrame) -> Optional[str]:
    # First: explicit aliases
    for c in YEAR_ALIAS_CANDS:
        if c in df.columns:
            return c
    # Fallback: any column containing "year"
    for c in df.columns:
        if "year" in c.lower():
            return c
    return None


def _parse_year_series(s: pd.Series) -> pd.Series:
    """
    Return a numeric year series:
      - If numeric-like, use it.
      - Otherwise, extract the FIRST 4-digit year present in the string.
        '2022/23' -> 2022, '2019-2020' -> 2019, 'AY 2018' -> 2018
    """
    y = pd.to_numeric(s, errors="coerce")
    # If many are NaN, try string extraction
    if y.isna().mean() > 0.3:
        y = (
            s.astype(str)
            .str.extract(YEAR_REGEX, expand=False)
            .pipe(pd.to_numeric, errors="coerce")
        )
    return y


def _list_filter_options(df: pd.DataFrame) -> Dict[str, List[Any]]:
    opts: Dict[str, List[Any]] = {}
    for key, cands in FILTER_COLS.items():
        col = _first_col(df, cands)
        if not col:
            continue
        vals = sorted([v for v in df[col].dropna().unique().tolist() if str(v) != ""])
        opts[key] = vals
    return opts


def _apply_filters(df: pd.DataFrame, params) -> pd.DataFrame:
    """Apply query param filters to df and return filtered df."""
    work = df.copy()

    # Province & District
    prov_col = _first_col(work, FILTER_COLS["Province"])
    dist_col = _first_col(work, FILTER_COLS["District"])
    if prov_col:
        sel = params.getlist("province") if hasattr(params, "getlist") else []
        if sel:
            work = work[work[prov_col].isin(sel)]
    if dist_col:
        sel = params.getlist("district") if hasattr(params, "getlist") else []
        if sel:
            work = work[work[dist_col].isin(sel)]

    # Gender
    gen_col = _first_col(work, FILTER_COLS["Gender"])
    if gen_col:
        sel = params.getlist("gender") if hasattr(params, "getlist") else []
        if sel:
            work = work[work[gen_col].isin(sel)]

    # School Location
    loc_col = _first_col(work, FILTER_COLS["School_Location"])
    if loc_col:
        sel = params.getlist("school_location") if hasattr(params, "getlist") else []
        if sel:
            work = work[work[loc_col].isin(sel)]

    # Academic Year range (robust parsing)
    yr_col = _find_year_column(work)
    if yr_col and yr_col in work.columns:
        yr_num = _parse_year_series(work[yr_col])
        req_min = (params.get("year_min") or "").strip()
        req_max = (params.get("year_max") or "").strip()
        mask = pd.Series(True, index=work.index)
        if req_min:
            mask &= yr_num >= pd.to_numeric(req_min, errors="coerce")
        if req_max:
            mask &= yr_num <= pd.to_numeric(req_max, errors="coerce")
        if req_min or req_max:
            mask &= yr_num.notna()  # drop rows with unknown year only when filtering is requested
        work = work[mask]

    return work


def _aggregate_pass_rates(df: pd.DataFrame) -> Dict[str, Any]:
    """Aggregates for charts + map join dicts."""
    out: Dict[str, Any] = {}

    # Probability histogram (10 bins)
    bins = np.linspace(0, 1, 11)
    hist, edges = np.histogram(df["Prob"].values, bins=bins)
    out["prob_hist"] = {"edges": edges.tolist(), "counts": hist.tolist()}

    # Province aggregates + map dict
    prov_col = _first_col(df, FILTER_COLS["Province"])
    if prov_col:
        g = df.groupby(prov_col)["Pred"].agg(["mean", "size"]).sort_values("mean", ascending=False)
        counts = df.groupby([prov_col, "Pred"]).size().unstack(fill_value=0)
        out["province"] = {
            "labels": g.index.tolist(),
            "pass_rate": g["mean"].round(4).tolist(),
            "passed": counts.get(1, pd.Series(0, index=counts.index)).tolist(),
            "failed": counts.get(0, pd.Series(0, index=counts.index)).tolist(),
        }
        out["province_map"] = {k: float(v) for k, v in df.groupby(prov_col)["Pred"].mean().to_dict().items()}

    # District aggregates + map dict
    dist_col = _first_col(df, FILTER_COLS["District"])
    if dist_col:
        totals = df.groupby(dist_col).size().sort_values(ascending=False)
        top_labels = totals.head(15).index
        sub = df[df[dist_col].isin(top_labels)]
        g = sub.groupby(dist_col)["Pred"].mean().loc[top_labels]
        counts = sub.groupby([dist_col, "Pred"]).size().unstack(fill_value=0).loc[top_labels]
        out["district_top"] = {
            "labels": top_labels.tolist(),
            "pass_rate": g.round(4).tolist(),
            "passed": counts.get(1, pd.Series(0, index=counts.index)).tolist(),
            "failed": counts.get(0, pd.Series(0, index=counts.index)).tolist(),
        }
        out["district_map"] = {k: float(v) for k, v in df.groupby(dist_col)["Pred"].mean().to_dict().items()}

    # Gender
    gen_col = _first_col(df, FILTER_COLS["Gender"])
    if gen_col:
        gg = df.groupby(gen_col)["Pred"].mean()
        out["gender"] = {"labels": gg.index.astype(str).tolist(), "pass_rate": gg.round(4).tolist()}

    # School location
    loc_col = _first_col(df, FILTER_COLS["School_Location"])
    if loc_col:
        gl = df.groupby(loc_col)["Pred"].mean()
        out["location"] = {"labels": gl.index.astype(str).tolist(), "pass_rate": gl.round(4).tolist()}

    return out


def _timeseries_blocks(df: pd.DataFrame) -> Dict[str, Any]:
    """Build time series if a year-ish column exists (robust)."""
    res: Dict[str, Any] = {}
    yr_col = _find_year_column(df)
    if not yr_col:
        return res

    series_df = df.copy()
    yr_num = _parse_year_series(series_df[yr_col])
    series_df = series_df.assign(_Year=yr_num).dropna(subset=["_Year"])
    series_df["_Year"] = series_df["_Year"].astype(int)

    g = series_df.groupby("_Year")["Pred"].mean().sort_index()
    res["overall_yearly"] = [{"year": int(k), "pass_rate": float(v)} for k, v in g.items()]

    prov_col = _first_col(series_df, FILTER_COLS["Province"])
    if prov_col:
        remaining = series_df[prov_col].dropna().unique().tolist()
        if len(remaining) == 1:
            sel = remaining[0]
            gp = series_df.groupby(["_Year", prov_col])["Pred"].mean().unstack()
            if sel in gp.columns:
                res["province_yearly"] = [{"year": int(y), "pass_rate": float(r)} for y, r in gp[sel].dropna().items()]
                res["province_name"] = sel

    dist_col = _first_col(series_df, FILTER_COLS["District"])
    if dist_col:
        remaining = series_df[dist_col].dropna().unique().tolist()
        if len(remaining) == 1:
            sel = remaining[0]
            gd = series_df.groupby(["_Year", dist_col])["Pred"].mean().unstack()
            if sel in gd.columns:
                res["district_yearly"] = [{"year": int(y), "pass_rate": float(r)} for y, r in gd[sel].dropna().items()]
                res["district_name"] = sel

    return res


def _subject_importance_from_pipeline(which: str = "primary_rf") -> Dict[str, Any]:
    """
    Compute subject-level importance from a saved sklearn pipeline (RF or LR).
    - Works when ColumnTransformer names numeric features like 'num__Kinyarwanda_P3'
    - Aggregates per subject by splitting on the *last* underscore
      ('Social_Studies_P3' -> 'Social_Studies').
    - Tolerant if a subject key doesn't exist in SUBJECTS (won't crash).
    """
    import numpy as np
    pipe = load_pipeline(which)
    pre = pipe.named_steps.get("pre")
    clf = pipe.named_steps.get("clf")
    if pre is None or clf is None:
        raise RuntimeError("Pipeline must have steps named 'pre' (ColumnTransformer) and 'clf'.")

    # Feature names out of the preprocessor
    try:
        feat_names = pre.get_feature_names_out()
    except Exception:
        # Fallback (older sklearn): assume ColumnTransformer prefixes 'num__'
        feat_names = np.array([f"num__{c}" for c in SUBJECT_COLS_ALL])

    feat_names = np.array(feat_names)

    # Model-level importances
    if hasattr(clf, "feature_importances_"):  # RandomForest
        imp_all = np.asarray(clf.feature_importances_, dtype=float)
    elif hasattr(clf, "coef_"):  # LogisticRegression
        imp_all = np.abs(np.asarray(clf.coef_).ravel())  # use |coef|
    else:
        return {"labels": SUBJECTS, "scores": [0.0] * len(SUBJECTS)}

    if imp_all.shape[0] != feat_names.shape[0]:
        raise RuntimeError(f"Importance length {imp_all.shape[0]} != features {feat_names.shape[0]}")

    # Keep only numeric features from the ColumnTransformer (avoid OHE cats)
    num_mask = np.array([name.startswith("num__") for name in feat_names])
    imp_num = imp_all[num_mask]
    num_names = feat_names[num_mask]  # e.g., 'num__Kinyarwanda_P3', 'num__Social_Studies_P3'

    # Aggregate by subject (split at the LAST underscore)
    subj_scores: Dict[str, float] = {}
    for name, val in zip(num_names, imp_num):
        col = name.split("__", 1)[1] if "__" in name else name  # -> 'Social_Studies_P3'
        subj = col.rsplit("_", 1)[0]  # -> 'Social_Studies'
        subj_scores[subj] = subj_scores.get(subj, 0.0) + float(val)

    # Ensure all expected subjects appear (even if 0)
    for s in SUBJECTS:
        subj_scores.setdefault(s, 0.0)

    items = sorted(subj_scores.items(), key=lambda x: x[1], reverse=True)
    return {"labels": [k for k, _ in items], "scores": [round(v, 6) for _, v in items]}


# --------------------------------------------------------------------------------------
# Inference (artifact + token-based dataset selector)
# --------------------------------------------------------------------------------------
def _collect_uploads(level: Optional[EducationLevel] = None) -> List[Dict[str, Any]]:
    def rows_for(model_label: str, source_name: str) -> List[Dict[str, Any]]:
        try:
            Model = apps.get_model(model_label)
        except LookupError:
            return []
        qs = Model.objects.select_related("level").all()
        if level:
            qs = qs.filter(level=level)
        out = []
        for obj in qs:
            f = getattr(obj, "file", None)
            if not f:
                continue
            label = getattr(
                obj, "original_name",
                os.path.basename(getattr(f, "name", "")) or f"{source_name} #{obj.pk}",
            )
            out.append({
                "token": f"{model_label}:{obj.pk}",
                "label": label,
                "level": getattr(obj.level, "name", "—"),
                "source": source_name,
            })
        return out

    return list(chain(
        rows_for("primary.PrimaryUpload", "Primary"),
        rows_for("o_level.OLevelUpload", "O-Level"),
        rows_for("a_level.ALevelUpload", "A-Level"),
    ))


@require_http_methods(["GET", "POST"])
def run_inference(request):
    form = InferenceForm(request.POST or None)

    # Level filter (support GET or POST)
    level_id = (request.POST.get("level") or request.GET.get("level") or "").strip()
    level_obj = None
    if level_id:
        try:
            level_obj = EducationLevel.objects.get(pk=int(level_id))
        except (EducationLevel.DoesNotExist, ValueError, TypeError):
            level_obj = None

    # Filter artifact choices by level
    art_qs = ModelArtifact.objects.filter(active=True).order_by("-created_at")
    if level_obj:
        art_qs = art_qs.filter(level=level_obj)
        if request.method == "GET":
            form.initial["level"] = level_obj.pk
    form.fields["artifact"].queryset = art_qs

    # Populate dataset choices (token-based)
    dataset_rows = _collect_uploads(level=level_obj)
    form.set_dataset_choices(dataset_rows)

    if request.method == "GET":
        if not art_qs.exists():
            messages.warning(request, "No saved models for this level yet. Add one on the Saved Models page.")
        if not dataset_rows:
            messages.warning(request, "No uploaded datasets found for this level.")
        return render(request, TPL_RUN, {"form": form})

    # POST: validate
    if not form.is_valid():
        return render(request, TPL_RUN, {"form": form})

    artifact = form.cleaned_data["artifact"]
    dataset_token = form.cleaned_data["dataset"]
    top_k = form.cleaned_data.get("top_k") or 3
    horizon = form.cleaned_data.get("horizon") or 12

    # Resolve token -> object -> file
    try:
        model_label, pk_str = dataset_token.split(":", 1)
        Model = apps.get_model(model_label)
        ds_obj = Model.objects.get(pk=int(pk_str))
        f = getattr(ds_obj, "file", None)
        if not f:
            raise ValueError("Selected dataset has no file attached.")
    except Exception as e:
        messages.error(request, f"Could not resolve dataset: {e}")
        return render(request, TPL_RUN, {"form": form})

    # Read dataset (path-safe / storage-safe)
    try:
        df = None
        dataset_path = getattr(f, "path", None)
        name_lower = (getattr(f, "name", "") or "").lower()
        if dataset_path and os.path.exists(dataset_path):
            lower = dataset_path.lower()
            if lower.endswith((".xlsx", ".xls")):
                df = pd.read_excel(dataset_path)
            elif lower.endswith(".csv"):
                df = pd.read_csv(dataset_path)

        if df is None:
            f.open("rb")
            data = f.read()
            f.close()
            bio = io.BytesIO(data)
            if name_lower.endswith((".xlsx", ".xls")):
                df = pd.read_excel(bio)
            elif name_lower.endswith(".csv"):
                df = pd.read_csv(bio)
            else:
                df = pd.read_excel(bio)  # default to Excel parser
    except Exception as e:
        messages.error(request, f"Failed to read dataset: {e}")
        return render(request, TPL_RUN, {"form": form})

    # Load model
    try:
        model = load_model(artifact.file.path, artifact.model_format)
    except Exception as e:
        messages.error(request, f"Failed to load model: {e}")
        return render(request, TPL_RUN, {"form": form})

    # Create job record
    job = InferenceJob.objects.create(
        artifact=artifact,
        dataset_label=os.path.basename(getattr(f, "name", "dataset")),
        dataset_path=getattr(f, "name", ""),
        params={"top_k": top_k, "horizon": horizon},
    )

    # Run inference
    try:
        if artifact.task == "classification":
            out = run_classification(
                model, df,
                feature_list=artifact.feature_list, want_proba=True, top_k=top_k
            )
            id_cols = [c for c in ["Student_ID", "student_id", "id"] if c in df.columns]
            res = pd.concat([df[id_cols], out], axis=1) if id_cols else out

        elif artifact.task == "forecasting":
            out = run_forecasting(model, df, horizon=horizon)
            res = out.reset_index()

        else:
            raise ValueError(f"Unknown task: {artifact.task}")

        # Save CSV under MEDIA/inference/<level>/
        ts = timezone.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"{artifact.name.replace(' ', '_').lower()}__{ts}.csv"
        out_dir = os.path.join(settings.MEDIA_ROOT, "inference", artifact.level.slug)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, out_name)
        res.to_csv(out_path, index=False)

        rel_path = os.path.relpath(out_path, settings.MEDIA_ROOT)
        job.output_csv.name = rel_path
        job.ok = True
        job.message = f"OK: {len(res)} rows."
        job.save(update_fields=["output_csv", "ok", "message", "finished_at"])

        preview = res.head(30).to_html(index=False, border=0)
        messages.success(request, f"Inference completed. {len(res)} rows. Download below.")
        return render(request, TPL_JOB, {"job": job, "preview_html": preview})

    except Exception as e:
        job.ok = False
        job.message = f"Failed: {e}"
        job.save(update_fields=["ok", "message", "finished_at"])
        messages.error(request, f"Inference failed: {e}")
        return render(request, TPL_RUN, {"form": form})


# --------------------------------------------------------------------------------------
# Upload & Predict (saves CSV into MEDIA/predictions/ with custom name)
# --------------------------------------------------------------------------------------
def upload_and_predict(request):
    if request.method == "POST":
        form = DatasetUploadForm(request.POST, request.FILES)
        if not form.is_valid():
            return render(request, "artifacts/upload.html", {"form": form}, status=400)

        f = form.cleaned_data["dataset"]
        model_choice = form.cleaned_data["model"]
        custom_label = form.cleaned_data.get("save_as") or ""

        try:
            df = pd.read_csv(f)

            if model_choice in ("primary_rf", "primary_logreg"):
                out = predict_with_pipeline(df, which=model_choice)
            elif model_choice == "primary_lstm":
                out = predict_with_lstm(df)
            else:
                return HttpResponseBadRequest("Unknown model selection.")

            # Build filename: preds_<label>_<model>_<timestamp>.csv
            uploaded_stem = Path(getattr(f, "name", "dataset")).stem
            base = slugify(custom_label) or slugify(uploaded_stem) or "preds"
            tag = slugify(model_choice)
            ts = timezone.now().strftime("%Y%m%d_%H%M%S")
            rel_name = f"predictions/preds_{base}_{tag}_{ts}.csv"

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            rel_path = default_storage.save(rel_name, ContentFile(csv_bytes))
            download_url = default_storage.url(rel_path)

            preview_html = out.head(50).to_html(classes="table table-striped table-sm", index=False)
            return render(
                request,
                "artifacts/prediction_result.html",
                {"preview_table": preview_html, "download_url": download_url, "saved_name": Path(rel_path).name},
            )
        except Exception as e:
            return render(request, "artifacts/upload.html", {"form": form, "error": str(e)}, status=400)

    # GET
    form = DatasetUploadForm()
    return render(request, "artifacts/upload.html", {"form": form})


# --------------------------------------------------------------------------------------
# Predict-one API + simple UI
# --------------------------------------------------------------------------------------
@csrf_exempt
@require_http_methods(["GET", "POST"])
def predict_one(request):
    # GET -> usage
    if request.method == "GET":
        from .predictors import REQ_PIPE_COLS, SEQ_ORDER
        return JsonResponse({
            "ok": True,
            "usage": "POST JSON with 'model' and required fields.",
            "models": ["primary_rf", "primary_logreg", "primary_lstm"],
            "required_columns": {
                "primary_rf": REQ_PIPE_COLS,
                "primary_logreg": REQ_PIPE_COLS,
                "primary_lstm": SEQ_ORDER
            },
            "note": "LSTM requires only P1–P5 subject scores (no demographics)."
        })

    # POST -> infer
    try:
        if request.META.get("CONTENT_TYPE", "").startswith("application/json"):
            payload = json.loads(request.body.decode("utf-8"))
        else:
            payload = request.POST.dict()

        model_choice = payload.pop("model", "primary_rf")
        threshold = float(payload.pop("threshold", 0.5))
        threshold = max(0.0, min(1.0, threshold))  # clamp

        df = pd.DataFrame([payload]).apply(pd.to_numeric, errors="ignore")

        if model_choice in ("primary_rf", "primary_logreg"):
            out = predict_with_pipeline(df, which=model_choice)
        elif model_choice == "primary_lstm":
            out = predict_with_lstm(df)
        else:
            return JsonResponse({"ok": False, "error": "Unknown model"}, status=400)

        row = out.iloc[0]
        prob = float(row["Prob"])
        pred = int(prob >= threshold)
        label = "Pass" if pred == 1 else "Not Pass"

        return JsonResponse({"ok": True, "model": model_choice, "threshold": threshold,
                             "pred": pred, "label": label, "prob": round(prob, 4)})
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=500)


def predict_one_ui(request):
    return render(request, "artifacts/predict_one_ui.html")


# --------------------------------------------------------------------------------------
# Dashboard APIs (files list, filters, data) + page
# --------------------------------------------------------------------------------------
@require_GET
def prediction_files_api(request):
    pred_dir = _predictions_dir()
    files = []
    for f in pred_dir.glob("*.csv"):
        stat = f.stat()
        files.append({"name": f.name, "size": stat.st_size, "modified": stat.st_mtime})
    files.sort(key=lambda x: x["modified"], reverse=True)
    return JsonResponse({"ok": True, "files": files})


@require_GET
def filters_api(request):
    """Return filter choices + years for a given predictions file. GET: ?file=<filename.csv>"""
    pred_dir = _predictions_dir()
    requested = request.GET.get("file", "").strip()
    if not requested:
        return JsonResponse({"ok": False, "error": "Missing ?file parameter."}, status=400)

    name = Path(requested).name
    if name != requested or not name.endswith(".csv"):
        return JsonResponse({"ok": False, "error": "Invalid filename."}, status=400)

    chosen = pred_dir / name
    if not chosen.exists():
        return JsonResponse({"ok": False, "error": f"File not found: {name}"}, status=404)

    df = pd.read_csv(chosen)
    if "Pred" not in df.columns or "Prob" not in df.columns:
        return JsonResponse({"ok": False, "error": "'Pred' or 'Prob' missing."}, status=400)

    opts = _list_filter_options(df)

    # Build distinct year list + min/max using robust parsers
    yr_col = _find_year_column(df)
    years, year_range = None, None
    if yr_col:
        yr_vals = _parse_year_series(df[yr_col]).dropna().astype(int)
        if not yr_vals.empty:
            years = sorted(yr_vals.unique().tolist())
            year_range = {"min": int(yr_vals.min()), "max": int(yr_vals.max())}

    return JsonResponse({"ok": True, "filters": opts, "years": years, "year_range": year_range})


@require_GET
def dashboard_data_api(request):
    """
    GET:
      file=<filename.csv> (required)
      province=.. (multi)  district=.. (multi)  gender=.. (multi)
      school_location=.. (multi)  year_min=YYYY  year_max=YYYY
    """
    pred_dir = _predictions_dir()
    requested = request.GET.get("file", "").strip()
    if not requested:
        return JsonResponse({"ok": False, "error": "Missing ?file parameter."}, status=400)

    name = Path(requested).name
    if name != requested or not name.endswith(".csv"):
        return JsonResponse({"ok": False, "error": "Invalid filename."}, status=400)

    chosen = pred_dir / name
    if not (chosen.exists() and chosen.is_file()):
        return JsonResponse({"ok": False, "error": f"File not found: {name}"}, status=404)

    df = pd.read_csv(chosen)
    if "Pred" not in df.columns or "Prob" not in df.columns:
        return JsonResponse({"ok": False, "error": f"'Pred' or 'Prob' missing in {chosen.name}."}, status=400)

    fdf = _apply_filters(df, request.GET)

    payload: Dict[str, Any] = {"ok": True, "file": chosen.name}
    payload["overall"] = {
        "total": int(len(fdf)),
        "pass_rate": round(float(fdf["Pred"].mean()) if len(fdf) else 0.0, 4),
        "avg_prob": round(float(fdf["Prob"].mean()) if len(fdf) else 0.0, 4),
    }
    payload["groups"] = _aggregate_pass_rates(fdf)

    try:
        payload["importance_rf"] = _subject_importance_from_pipeline("primary_rf")
    except Exception as e:
        payload["importance_rf_error"] = str(e)
    try:
        payload["importance_lr"] = _subject_importance_from_pipeline("primary_logreg")
    except Exception as e:
        payload["importance_lr_error"] = str(e)

    payload["timeseries"] = _timeseries_blocks(fdf)
    return JsonResponse(payload)


def dashboard_page(request):
    return render(request, "artifacts/dashboard.html")


# **************************************A-Level-Views*******************************************************************
# artifacts/views_alevel.py

import json
from pathlib import Path
import pandas as pd
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.http import JsonResponse, HttpResponseBadRequest
from django.shortcuts import render
from django.utils import timezone
from django.utils.text import slugify
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods, require_GET

from .alevel_predictors import predict_alevel_programs, required_columns

# carry-through columns if present
_CTX_CANDS = [
    "Province", "province", "District", "district",
    "Gender", "gender", "School_Location", "school_location",
    "Academic_Year", "academic_year", "Year", "year",
    "Student_ID", "student_id", "School_Name", "school_name",
    "university_program", "University_Program", "Program"
]


@require_http_methods(["GET", "POST"])
def upload_and_predict_alevel(request):
    if request.method == "GET":
        return render(request, "alevel/alevel_upload.html")

    f = request.FILES.get("dataset")
    custom_label = (request.POST.get("save_as") or "").strip()
    if not f:
        return render(request, "alevel/alevel_upload.html", {"error": "Please choose a CSV."}, status=400)
    try:
        raw = pd.read_csv(f)
        out = predict_alevel_programs(raw, top_k=5)
        keep = [c for c in _CTX_CANDS if c in raw.columns]
        res = pd.concat([raw[keep], out], axis=1) if keep else out

        base = slugify(custom_label) or Path(getattr(f, "name", "dataset")).stem
        ts = timezone.now().strftime("%Y%m%d_%H%M%S")
        rel = f"predictions/alevel_preds_{base}_{ts}.csv"
        rel_path = default_storage.save(rel, ContentFile(res.to_csv(index=False).encode("utf-8")))
        return render(request, "alevel/alevel_result.html", {
            "saved_name": Path(rel_path).name,
            "download_url": default_storage.url(rel_path),
            "preview_table": res.head(60).to_html(classes="table table-striped table-sm", index=False),
        })
    except Exception as e:
        return render(request, "alevel/alevel_upload.html", {"error": str(e)}, status=400)


# ---------- Predict-one API + UI ----------
@csrf_exempt
@require_http_methods(["GET", "POST"])
def predict_one_alevel(request):
    if request.method == "GET":
        return JsonResponse({
            "ok": True,
            "usage": "POST JSON with required feature columns to get Top-K programs.",
            "required_columns": required_columns()
        })
    try:
        if request.META.get("CONTENT_TYPE", "").startswith("application/json"):
            payload = json.loads(request.body.decode("utf-8"))
        else:
            payload = request.POST.dict()
        df = pd.DataFrame([payload]).apply(pd.to_numeric, errors="ignore")
        out = predict_alevel_programs(df, top_k=5)
        row = out.iloc[0]
        return JsonResponse({
            "ok": True,
            "pred_program": row["Pred_Program"],
            "pred_label": int(row["Pred_Label"]),
            "topk_programs": row["TopK_Programs"],
            "topk_probs": [round(float(x), 4) for x in row["TopK_Probs"]],
        })
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=400)


@require_GET
def predict_one_alevel_ui(request):
    return render(request, "alevel/predict_one_alevel.html")


# **************************************A-Level-Dashboard View**********************************************************
import re, json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET


# ------------- Where predictions/models live -------------
def _predictions_dir() -> Path:
    p = Path(settings.MEDIA_ROOT) / "predictions"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _models_dir() -> Path:
    p = Path(getattr(settings, "BASE_DIR", ".")) / "models" / "A-Level"
    if not p.exists():
        raise FileNotFoundError(f"A-Level models dir not found: {p}")
    return p


# ------------- Filters / years helpers -------------
FILTER_COLS = {
    "Province": ["Province", "province"],
    "District": ["District", "district"],
    "Gender": ["Gender", "gender"],
    "School_Location": ["School_Location", "school_location"],
    "Academic_Year": ["Academic_Year", "academic_year", "Year", "year", "Exam_Year", "exam_year"],
}

YEAR_ALIAS_CANDS = [
    "Academic_Year", "academic_year", "Year", "year", "Exam_Year", "exam_year", "Cohort_Year", "cohort_year"
]
YEAR_RE = re.compile(r"(19|20)\d{2}")


def _first_col(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    for c in cands:
        if c in df.columns: return c
    return None


def _find_year_col(df: pd.DataFrame) -> Optional[str]:
    for c in YEAR_ALIAS_CANDS:
        if c in df.columns: return c
    for c in df.columns:
        if "year" in c.lower(): return c
    return None


def _to_year_series(s: pd.Series) -> pd.Series:
    y = pd.to_numeric(s, errors="coerce")
    if y.isna().mean() > 0.3:
        y = s.astype(str).str.extract(YEAR_RE, expand=False).pipe(pd.to_numeric, errors="coerce")
    return y


def _list_filter_options(df: pd.DataFrame) -> Dict[str, List[Any]]:
    out: Dict[str, List[Any]] = {}
    for key, cands in FILTER_COLS.items():
        col = _first_col(df, cands)
        if not col: continue
        vals = sorted([v for v in df[col].dropna().unique().tolist() if str(v) != ""])
        out[key] = vals
    return out


def _apply_filters(df: pd.DataFrame, params) -> pd.DataFrame:
    w = df.copy()
    for key in ["Province", "District", "Gender", "School_Location"]:
        col = _first_col(w, FILTER_COLS[key])
        if not col: continue
        vals = params.getlist(key.lower()) if hasattr(params, "getlist") else []
        if vals: w = w[w[col].isin(vals)]
    ycol = _find_year_col(w)
    if ycol and ycol in w.columns:
        y = _to_year_series(w[ycol])
        y_min = params.get("year_min");
        y_max = params.get("year_max")
        mask = pd.Series(True, index=w.index)
        if y_min: mask &= y >= pd.to_numeric(y_min, errors="coerce")
        if y_max: mask &= y <= pd.to_numeric(y_max, errors="coerce")
        if y_min or y_max: mask &= y.notna()
        w = w[mask]
    return w


# ------------- Stream/Program aggregations -------------
def _program_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    col = "Pred_Program" if "Pred_Program" in df.columns else ("Pred_Stream" if "Pred_Stream" in df.columns else None)
    if not col: return {"labels": [], "counts": []}
    vc = df[col].value_counts()
    return {"labels": vc.index.astype(str).tolist(), "counts": vc.values.astype(int).tolist()}


def _program_by_gender(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    gcol = _first_col(df, FILTER_COLS["Gender"])
    pcol = "Pred_Program" if "Pred_Program" in df.columns else None
    if not (gcol and pcol): return None
    t = pd.crosstab(df[gcol], df[pcol])
    return {"genders": t.index.astype(str).tolist(),
            "programs": t.columns.astype(str).tolist(),
            "matrix": t.values.tolist()}


def _confidence_hist(df: pd.DataFrame) -> Dict[str, Any]:
    # Use per-class probability columns Prob_<cls> if present, else TopK_Probs[0]
    prob_cols = [c for c in df.columns if c.startswith("Prob_")]
    if prob_cols:
        conf = df[prob_cols].max(axis=1).fillna(0.0).clip(0, 1).to_numpy()
    else:
        if "TopK_Probs" in df.columns and df["TopK_Probs"].dtype == object:
            conf = df["TopK_Probs"].apply(
                lambda xs: (xs[0] if isinstance(xs, (list, tuple)) and xs else np.nan)
            ).fillna(0.0).to_numpy()
        else:
            conf = np.zeros(len(df))
    bins = np.linspace(0, 1, 11)
    hist, edges = np.histogram(conf, bins=bins)
    return {"edges": edges.tolist(), "counts": hist.astype(int).tolist()}


def _alignment_if_gt(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    # If ground truth University_Program present, compute accuracy per program
    gt_col = None
    for c in ["University_Program", "university_program", "Program", "program", "Target_Program", "target_program"]:
        if c in df.columns: gt_col = c; break
    if not gt_col or "Pred_Program" not in df.columns: return None
    ok = (df[gt_col].astype(str) == df["Pred_Program"].astype(str))
    overall = float(ok.mean()) if len(df) else 0.0
    by_prog = df.groupby(gt_col)["Pred_Program"].apply(lambda s: np.mean(s.values == s.name))
    return {
        "overall_acc": round(overall, 4),
        "labels": by_prog.index.astype(str).tolist(),
        "acc": [round(float(v), 4) for v in by_prog.values]
    }


# ------------- Importance: subject & subject×year from alevel_importances.csv -------------
SUBJECT_MAP = {
    # normalize common variants to a canonical subject label
    "math": "Mathematics", "mathematics": "Mathematics",
    "physics": "Physics", "chemistry": "Chemistry", "biology": "Biology",
    "english": "English", "kinyarwanda": "Kinyarwanda", "ict": "ICT",
    "geography": "Geography", "history": "History", "economics": "Economics"
}
YEAR_TAGS = ("S4", "S5", "S6")


def _detect_year_and_subject(feat: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (year_tag, subject_name) for features like:
      'S4_Mathematics', 'Mathematics_S4', 'S5-Physics', 'English S6'
    If not a subject score, returns (None, None).
    """
    f = feat.replace("-", "_").replace(" ", "_")
    up = f.upper()
    # prefix
    for y in YEAR_TAGS:
        if up.startswith(y + "_"):
            subj = f[len(y) + 1:]
            return y, subj
    # suffix
    for y in YEAR_TAGS:
        if up.endswith("_" + y):
            subj = f[:-(len(y) + 1)]
            return y, subj
    return None, None


def _canonical_subject(name: str) -> Optional[str]:
    if not name: return None
    key = re.sub(r"[^a-z]", "", name.lower())
    return SUBJECT_MAP.get(key)


def _load_importances() -> pd.DataFrame:
    p = _models_dir() / "alevel_importances.csv"
    if not p.exists():
        # Fallback: empty
        return pd.DataFrame(columns=["feature", "importance"])
    df = pd.read_csv(p)
    if "feature" not in df.columns or "importance" not in df.columns:
        return pd.DataFrame(columns=["feature", "importance"])
    # coerce importance to float
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce").fillna(0.0)
    return df


def _importance_buckets() -> Dict[str, Any]:
    imp = _load_importances()
    if imp.empty:
        return {
            "subject": {"labels": [], "scores": []},
            "subject_year": {"years": list(YEAR_TAGS), "subjects": [], "matrix": []},
            "top_other": {"labels": [], "scores": []}
        }

    # Aggregate to subject and subject×year
    subj_year = {(y, s): 0.0 for y in YEAR_TAGS for s in SUBJECT_MAP.values()}
    other_feats: Dict[str, float] = {}

    for _, row in imp.iterrows():
        f = str(row["feature"])
        val = float(row["importance"])
        year, subj_raw = _detect_year_and_subject(f)
        subj = _canonical_subject(subj_raw) if subj_raw else None
        if year and subj:
            subj_year[(year, subj)] += val
        else:
            # Other features (could be numeric or OHE categories like Gender_Female)
            other_feats[f] = other_feats.get(f, 0.0) + val

    # subject totals (sum S4..S6)
    subjects = list(SUBJECT_MAP.values())
    subj_totals = {s: 0.0 for s in subjects}
    for (y, s), v in subj_year.items():
        subj_totals[s] += v

    # sort subjects by total importance
    ordered = sorted(subj_totals.items(), key=lambda x: x[1], reverse=True)
    subj_labels = [k for k, _ in ordered]
    subj_scores = [float(v) for _, v in ordered]

    # build subject×year matrix aligned to subjects order
    mat = []
    for y in YEAR_TAGS:
        row = [float(subj_year[(y, s)]) for s in subj_labels]
        mat.append(row)

    # top other features (non-subjects)
    top_other = sorted(other_feats.items(), key=lambda x: x[1], reverse=True)[:15]
    return {
        "subject": {"labels": subj_labels, "scores": subj_scores},
        "subject_year": {"years": list(YEAR_TAGS), "subjects": subj_labels, "matrix": mat},
        "top_other": {"labels": [k for k, _ in top_other], "scores": [float(v) for _, v in top_other]}
    }


# ------------- Utilities -------------
def _maybe_eval_lists(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("TopK_Programs", "TopK_Probs"):
        if c in df.columns and df[c].dtype == object:
            try:
                df[c] = df[c].apply(lambda x: json.loads(x) if isinstance(x, str) and x.strip().startswith("[") else x)
            except Exception:
                # try literal eval fallback
                import ast
                try:
                    df[c] = df[c].apply(
                        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip().startswith("[") else x)
                except Exception:
                    pass
    return df


# ------------- APIs -------------
@require_GET
def alevel_prediction_files_api(request):
    """List available A-Level prediction CSVs under /media/predictions/ (newest first)."""
    pred_dir = _predictions_dir()
    files = []
    # Prefer A-Level files
    for f in pred_dir.glob("alevel_*.csv"):
        stat = f.stat()
        files.append({"name": f.name, "size": stat.st_size, "modified": stat.st_mtime})
    # fallback: any predictions file
    if not files:
        for f in pred_dir.glob("*.csv"):
            stat = f.stat()
            files.append({"name": f.name, "size": stat.st_size, "modified": stat.st_mtime})
    files.sort(key=lambda x: x["modified"], reverse=True)
    return JsonResponse({"ok": True, "files": files})


@require_GET
def alevel_filters_api(request):
    pred_dir = _predictions_dir()
    name = (request.GET.get("file") or "").strip()
    p = pred_dir / Path(name).name
    if not (name and p.exists()):
        return JsonResponse({"ok": False, "error": "File not found."}, status=404)

    df = pd.read_csv(p)
    opts = _list_filter_options(df)

    ycol = _find_year_col(df)
    years = None;
    year_range = None
    if ycol:
        y = _to_year_series(df[ycol]).dropna().astype(int)
        if not y.empty:
            years = sorted(y.unique().tolist())
            year_range = {"min": int(y.min()), "max": int(y.max())}

    return JsonResponse({"ok": True, "filters": opts, "years": years, "year_range": year_range})


@require_GET
def alevel_dashboard_data_api(request):
    """
    GET:
      file=<csv>  + filters: province=.. district=.. gender=.. school_location=.. year_min/year_max
    """
    pred_dir = _predictions_dir()
    name = (request.GET.get("file") or "").strip()
    p = pred_dir / Path(name).name
    if not (name and p.exists()):
        return JsonResponse({"ok": False, "error": "File not found."}, status=404)

    df = _maybe_eval_lists(pd.read_csv(p))
    fdf = _apply_filters(df, request.GET)

    payload: Dict[str, Any] = {
        "ok": True,
        "file": p.name,
        "overall": {"total": int(len(fdf))}
    }

    # Distribution of predicted programs
    payload["program_dist"] = _program_distribution(fdf)

    # Programs by gender (stacked)
    pbg = _program_by_gender(fdf)
    if pbg: payload["program_by_gender"] = pbg

    # Confidence histogram
    payload["confidence"] = _confidence_hist(fdf)

    # Alignment vs GT (if present)
    align = _alignment_if_gt(fdf)
    if align: payload["alignment"] = align

    # Feature importance (from saved importances csv)
    payload["importance"] = _importance_buckets()

    return JsonResponse(payload)


# ------------- Page -------------
def alevel_dashboard_page(request):
    return render(request, "alevel/alevel_dashboard.html")
