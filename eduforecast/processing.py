# uploader/processing.py
from __future__ import annotations
import re
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Canonical subject names
SUBJECTS = ["Kinyarwanda", "English", "Mathematics", "Science", "Social_Studies", "Creative_Arts"]
GRADES = [f"P{i}" for i in range(1, 7)]

# Token -> canonical subject (add your own aliases here if needed)
SUBJECT_TOKENS: Dict[str, str] = {
    # Kinyarwanda
    "kinyarwanda": "Kinyarwanda", "kiny": "Kinyarwanda", "kinya": "Kinyarwanda",
    # English
    "english": "English", "eng": "English",
    # Mathematics
    "mathematics": "Mathematics", "math": "Mathematics", "maths": "Mathematics", "mat": "Mathematics",
    # Science
    "science": "Science", "sci": "Science",
    # Social Studies
    "socialstudies": "Social_Studies", "social_studies": "Social_Studies", "social": "Social_Studies",
    "sst": "Social_Studies", "socstudies": "Social_Studies", "soc_studies": "Social_Studies",
    # Creative Arts
    "creativearts": "Creative_Arts", "creative_arts": "Creative_Arts", "creative": "Creative_Arts",
    "arts": "Creative_Arts", "art": "Creative_Arts", "ca": "Creative_Arts",
}

# Words to ignore if they appear in header names
NOISE = {"score", "marks", "mark", "avg", "average", "mean", "total", "subject", "paper"}

# Accept many grade shapes -> canonical P1..P6
GRADE_PATTERNS = [
    r"p\.?\s*([1-6])",  # P1, P.1, P 1
    r"grade\.?\s*([1-6])",  # Grade1, Grade 1
    r"g\.?\s*([1-6])",  # G1
    r"class\.?\s*([1-6])",  # Class1
    r"primary\.?\s*([1-6])",  # Primary1
    r"std\.?\s*([1-6])",  # Std1
    r"year\.?\s*([1-6])",  # Year1
    r"y\.?\s*([1-6])",  # Y1
    r"([1-6])",  # bare digit, used only when subject is present
]

SPLIT_RE = re.compile(r"[^A-Za-z0-9]+")


def _flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = ["_".join([str(x) for x in tup if str(x).strip() != ""]).strip() for tup in df.columns.values]
    return df


def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to ensure first real row is header and headers are clean strings."""
    df = _flatten_multiindex_columns(df)
    # If columns look unnamed or numeric, try promoting first row to header
    col_strs = [str(c) for c in df.columns]
    unnamed_ratio = sum(1 for c in col_strs if c.lower().startswith("unnamed") or c.isdigit()) / max(1, len(col_strs))
    if unnamed_ratio > 0.5 and df.shape[0] > 0:
        candidate = df.iloc[0].astype(str).tolist()
        # Only promote if candidate has at least a couple of non-empty labels
        if sum(1 for x in candidate if x.strip()) >= max(2, len(candidate) // 3):
            df = df.copy()
            df.columns = candidate
            df = df.iloc[1:].reset_index(drop=True)

    # Final clean-up: ensure string headers, strip spaces
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _canon_text(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("’", "'")
    return s


def _tokens(s: str) -> List[str]:
    s = _canon_text(s)
    toks = [t for t in SPLIT_RE.split(s) if t]
    return toks


def _subject_from_tokens(tokens: List[str]) -> Optional[str]:
    # try single tokens and collapsed pairs for things like "social studies"
    collapsed = set()
    for i in range(len(tokens) - 1):
        collapsed.add(tokens[i] + tokens[i + 1])
    all_forms = set(tokens) | collapsed
    # remove noise words
    all_forms = {t for t in all_forms if t not in NOISE}
    for t in all_forms:
        # direct match
        if t in SUBJECT_TOKENS:
            return SUBJECT_TOKENS[t]
        # try replacing common plurals or underscores
        t2 = t.replace("_", "")
        if t2 in SUBJECT_TOKENS:
            return SUBJECT_TOKENS[t2]
    return None


def _grade_from_tokens_or_text(tokens: List[str], text: str, require_grade: bool = True) -> Optional[str]:
    txt = " ".join(tokens)
    for pat in GRADE_PATTERNS:
        m = re.search(pat, txt, flags=re.I)
        if not m:
            m = re.search(pat, text, flags=re.I)
        if m:
            g = int(m.group(1))
            if 1 <= g <= 6:
                return f"P{g}"
    return None if require_grade else "P1"  # fallback not usually used


def _infer_pairs(cols: List[str]) -> Tuple[List[str], List[str], List[Tuple[str, str, str]]]:
    pairs: List[Tuple[str, str, str]] = []
    subj_seen, grade_seen = set(), set()

    for col in cols:
        tks = _tokens(col)
        subj = _subject_from_tokens(tks)
        grade = _grade_from_tokens_or_text(tks, col, require_grade=True)

        # If grade not found but column ends with 1..6 joined to subject (e.g., Maths1)
        if subj and not grade:
            tail_digit = re.search(r"([1-6])$", _canon_text(col))
            if tail_digit:
                grade = f"P{tail_digit.group(1)}"

        if subj and grade:
            pairs.append((subj, grade, col))
            subj_seen.add(subj);
            grade_seen.add(grade)

    if not pairs:
        raise ValueError("Could not infer Subject_Grade columns after header normalization.")

    subjects = [s for s in SUBJECTS if s in subj_seen] + sorted(list(subj_seen - set(SUBJECTS)))
    grades = [g for g in GRADES if g in grade_seen] + sorted(list(grade_seen - set(GRADES)), key=lambda x: int(x[1:]))

    return subjects, grades, pairs


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Normalizes headers (flattens, promotes first row if needed).
    - Infers Subject/Grade from many naming styles.
    - Computes per-subject averages and an Overall_Average.
    """
    df = _normalize_headers(df)
    cols = df.columns.tolist()

    try:
        subjects, grades, pairs = _infer_pairs(cols)
    except ValueError as e:
        # Attach diagnostics: show the first few column names so you can see what's coming in
        sample = ", ".join([repr(c) for c in cols[:8]])
        raise ValueError(
            "Could not infer Subject_Grade columns. "
            "Examples of accepted forms: 'English_P1', 'P1 English', 'ENG-P.1', 'Maths1', 'P-2_History'. "
            f"First columns detected: {sample}"
        ) from e

    # Map found (subject, grade) -> actual column
    colmap = {(s, g): c for (s, g, c) in pairs}

    proc = df.copy()

    # Per-subject averages
    for s in subjects:
        gcols = [colmap[(s, g)] for g in grades if (s, g) in colmap]
        if gcols:
            proc[f"Avg_{s}"] = proc[gcols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)

    # Overall average
    all_cols = [colmap[(s, g)] for s in subjects for g in grades if (s, g) in colmap]
    if all_cols:
        proc["Overall_Average"] = proc[all_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)

    return proc
