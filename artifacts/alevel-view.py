# artifacts/views_alevel_dashboard.py
from __future__ import annotations
import re, json, os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache

import numpy as np
import pandas as pd
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET


# -------------------- Locations --------------------
def _predictions_dir() -> Path:
    p = Path(settings.MEDIA_ROOT) / "predictions"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _models_dir() -> Path:
    p = Path(getattr(settings, "BASE_DIR", ".")) / "models" / "alevel"
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    return p


# -------------------- CSV cache --------------------
def _cached_read_csv(path: Path) -> pd.DataFrame:
    mtime = path.stat().st_mtime if path.exists() else 0.0
    df = _read_csv_cached(str(path), mtime)
    return df.copy()


@lru_cache(maxsize=16)
def _read_csv_cached(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


# -------------------- Filters / year helpers --------------------
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


# -------------------- Predicted program column detection --------------------
PRED_PROG_CANDS = [
    "Pred_Program", "pred_program", "Predicted_Program", "Pred_Prog",
    "Pred_Stream", "pred_stream", "Predicted_Stream"
]


def _find_pred_program_col(df: pd.DataFrame) -> Optional[str]:
    for c in PRED_PROG_CANDS:
        if c in df.columns:
            return c
    return None


# -------------------- Aggregations --------------------
def _program_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    pcol = _find_pred_program_col(df)
    if not pcol:
        return {"labels": [], "counts": []}
    vc = df[pcol].value_counts()
    return {"labels": vc.index.astype(str).tolist(), "counts": vc.values.astype(int).tolist()}


def _program_by_gender(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    gcol = _first_col(df, FILTER_COLS["Gender"])
    pcol = _find_pred_program_col(df)
    if not (gcol and pcol):
        return None
    t = pd.crosstab(df[gcol], df[pcol])
    return {"genders": t.index.astype(str).tolist(),
            "programs": t.columns.astype(str).tolist(),
            "matrix": t.values.tolist()}


def _confidence_hist(df: pd.DataFrame) -> Dict[str, Any]:
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
    pcol = _find_pred_program_col(df)
    if not pcol:
        return None
    gt_col = None
    for c in ["University_Program", "university_program", "Program", "program", "Target_Program", "target_program"]:
        if c in df.columns: gt_col = c; break
    if not gt_col:
        return None

    ok = (df[gt_col].astype(str) == df[pcol].astype(str))
    overall = float(ok.mean()) if len(df) else 0.0
    by_prog = df.groupby(gt_col)[pcol].apply(lambda s: np.mean(s.values == s.name))
    return {
        "overall_acc": round(overall, 4),
        "labels": by_prog.index.astype(str).tolist(),
        "acc": [round(float(v), 4) for v in by_prog.values]
    }


# -------------------- Importances --------------------
SUBJECT_MAP = {
    "math": "Mathematics", "mathematics": "Mathematics",
    "physics": "Physics", "chemistry": "Chemistry", "biology": "Biology",
    "english": "English", "kinyarwanda": "Kinyarwanda", "ict": "ICT",
    "geography": "Geography", "history": "History", "economics": "Economics"
}
YEAR_TAGS = ("S4", "S5", "S6")


def _detect_year_and_subject(feat: str) -> Tuple[Optional[str], Optional[str]]:
    f = feat.replace("-", "_").replace(" ", "_")
    up = f.upper()
    for y in YEAR_TAGS:
        if up.startswith(y + "_"):
            subj = f[len(y) + 1:]
            return y, subj
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
        return pd.DataFrame(columns=["feature", "importance"])
    df = pd.read_csv(p)
    if "feature" not in df.columns or "importance" not in df.columns:
        return pd.DataFrame(columns=["feature", "importance"])
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
            other_feats[f] = other_feats.get(f, 0.0) + val

    subjects = list(SUBJECT_MAP.values())
    subj_totals = {s: 0.0 for s in subjects}
    for (y, s), v in subj_year.items():
        subj_totals[s] += v

    ordered = sorted(subj_totals.items(), key=lambda x: x[1], reverse=True)
    subj_labels = [k for k, _ in ordered]
    subj_scores = [float(v) for _, v in ordered]

    mat = []
    for y in YEAR_TAGS:
        row = [float(subj_year[(y, s)]) for s in subj_labels]
        mat.append(row)

    top_other = sorted(other_feats.items(), key=lambda x: x[1], reverse=True)[:15]
    return {
        "subject": {"labels": subj_labels, "scores": subj_scores},
        "subject_year": {"years": list(YEAR_TAGS), "subjects": subj_labels, "matrix": mat},
        "top_other": {"labels": [k for k, _ in top_other], "scores": [float(v) for _, v in top_other]}
    }


# -------------------- Time-series --------------------
def _program_share_timeseries(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    ycol = _find_year_col(df)
    pcol = _find_pred_program_col(df)
    if not ycol or not pcol:
        return None
    work = df.copy()
    work["_Y"] = _to_year_series(work[ycol])
    work = work.dropna(subset=["_Y"])
    if work.empty:
        return None
    work["_Y"] = work["_Y"].astype(int)
    t = pd.crosstab(work["_Y"], work[pcol]).sort_index()
    if t.empty:
        return None
    denom = t.sum(axis=1).replace(0, np.nan)
    share = (t.T / denom).T.fillna(0.0)
    years = t.index.astype(int).tolist()
    progs = t.columns.astype(str).tolist()
    return {"years": years, "programs": progs, "matrix": share.values.tolist(), "counts": t.values.tolist()}


# -------------------- Safe list parsing --------------------
def _maybe_eval_lists(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("TopK_Programs", "TopK_Probs"):
        if c in df.columns and df[c].dtype == object:
            try:
                df[c] = df[c].apply(lambda x: json.loads(x) if isinstance(x, str) and x.strip().startswith("[") else x)
            except Exception:
                import ast
                try:
                    df[c] = df[c].apply(
                        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip().startswith("[") else x)
                except Exception:
                    pass
    return df


# -------------------- APIs --------------------
@require_GET
def alevel_prediction_files_api(request):
    pred_dir = _predictions_dir()
    files = []
    for f in pred_dir.glob("alevel_*.csv"):
        stat = f.stat()
        files.append({"name": f.name, "size": stat.st_size, "modified": stat.st_mtime})
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

    df = _cached_read_csv(p)
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
    pred_dir = _predictions_dir()
    name = (request.GET.get("file") or "").strip()
    p = pred_dir / Path(name).name
    if not (name and p.exists()):
        return JsonResponse({"ok": False, "error": "File not found."}, status=404)

    df = _maybe_eval_lists(_cached_read_csv(p))
    fdf = _apply_filters(df, request.GET)

    payload: Dict[str, Any] = {"ok": True, "file": p.name, "overall": {"total": int(len(fdf))}}

    payload["program_dist"] = _program_distribution(fdf)
    pbg = _program_by_gender(fdf)
    if pbg: payload["program_by_gender"] = pbg
    payload["confidence"] = _confidence_hist(fdf)
    align = _alignment_if_gt(fdf)
    if align: payload["alignment"] = align

    payload["importance"] = _importance_buckets()

    ts = _program_share_timeseries(fdf)
    if ts:
        payload["timeseries"] = ts

    return JsonResponse(payload)


# -------------------- Page --------------------
def alevel_dashboard_page(request):
    return render(request, "alevel/alevel_dashboard.html")
