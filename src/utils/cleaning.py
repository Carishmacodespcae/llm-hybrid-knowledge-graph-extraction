import pandas as pd
import numpy as np
import re
import ftfy


from difflib import SequenceMatcher


def clean_pid(x):
    if pd.isna(x):
        return ""
    return str(x).replace("/", "").replace("-", "").strip()


def norm(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def normalize_pipe_field(x):
    """Sort pipe-separated segments so order doesn't affect sim score."""
    if pd.isna(x) or str(x).strip() == "":
        return ""
    parts = [p.strip() for p in str(x).split("|")]
    return " | ".join(sorted(parts))


def normalize_location(loc):
    """Split by comma, sort parts, rejoin — order-independent comparison."""
    if pd.isna(loc) or str(loc).strip().lower() in {"", "nan", "none"}:
        return ""
    parts = sorted([p.strip().lower() for p in str(loc).split(",")])
    return ", ".join(parts)


def normalize_location_col(df, col):
    """Apply normalize_location to a DataFrame column if it exists."""
    if col in df.columns:
        df = df.copy()
        df[col] = df[col].apply(normalize_location)
    return df


def extract_abbreviation(text):
    """Extract abbreviation from parentheses e.g. 'University of X (UESTC)' → 'UESTC'"""
    if not text:
        return None
    match = re.search(r'\(([A-Z]{2,10})\)', str(text))
    return match.group(1) if match else None


def institutions_match(manual, llm):
    """
    Returns True if manual and llm refer to the same institution.
    Handles: exact match, containment, abbreviation in parentheses,
    and generated abbreviation (first letters of major words).
    """
    if not manual or not llm:
        return False

    m = str(manual).strip()
    l = str(llm).strip()
    m_low = m.lower()
    l_low = l.lower()

    # Exact match
    if m_low == l_low:
        return True

    # Containment both ways
    if m_low in l_low or l_low in m_low:
        return True

    # Abbreviation in parentheses e.g. "Full Name (UESTC)"
    llm_abbr = extract_abbreviation(l)
    if llm_abbr and m.upper() == llm_abbr:
        return True

    manual_abbr = extract_abbreviation(m)
    if manual_abbr and l.upper() == manual_abbr:
        return True

    # Generated abbreviation e.g. "University of Electronic Science..." → "UESTC"
    def make_abbr(text):
        skip = {"of", "the", "and", "for", "in", "at", "de"}
        words = re.sub(r'[^a-zA-Z\s]', '', text).split()
        return "".join(w[0].upper() for w in words if w.lower() not in skip)

    if make_abbr(m) == l.upper():
        return True
    if make_abbr(l) == m.upper():
        return True

    return False

def normalize_degree(text):
    if not text or not isinstance(text, str):
        return text
    return ftfy.fix_text(text).strip()

def sim(a, b):
    a = norm(a)
    b = norm(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def join_list(v):
    if isinstance(v, list):
        return "; ".join([str(x) for x in v if x is not None and str(x).strip() != ""])
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
