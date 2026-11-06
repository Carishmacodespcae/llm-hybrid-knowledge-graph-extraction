import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from src.utils.cleaning import sim, institutions_match
from src.config_noapi_mode_TN import SIM_THRESHOLD as _DEFAULT_THRESHOLD, PRIMARY_ID


def outcome_label(y_true, y_pred):
    if y_true == 1 and y_pred == 1:
        return "TP"
    elif y_true == 0 and y_pred == 0:
        return "TN"
    elif y_true == 0 and y_pred == 1:
        return "FP"
    elif y_true == 1 and y_pred == 0:
        return "FN"
    return ""


def build_position_presence_metrics(presence_df):
    vc = presence_df["outcome"].value_counts().to_dict()
    tp = vc.get("TP", 0)
    tn = vc.get("TN", 0)
    fp = vc.get("FP", 0)
    fn = vc.get("FN", 0)
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return pd.DataFrame([{
        "entity": "Positions",
        "field": "position_exists",
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }])


def _normalize_bool(series):
    return (
        series.astype(str)
        .str.lower()
        .str.strip()
        .replace({"nan": "", "<na>": "", "none": ""})
    )


def _to_ybool(val):
    v = str(val).lower().strip() if pd.notna(val) else ""
    return "" if v in ("nan", "<na>", "none") else v


def _normalize_location_string(x):
    x = "" if pd.isna(x) else str(x).strip().lower()
    if not x:
        return ""
    parts = [p.strip() for p in x.split(",") if p.strip()]
    if len(parts) == 2:
        a, b = parts
        return " | ".join(sorted([a, b]))
    return x


def _ypred_text(a, b, sim_threshold, field=None):
    a = "" if pd.isna(a) else str(a).strip()
    b = "" if pd.isna(b) else str(b).strip()

    if a == "":
        return 1 if b != "" else 0

    if field == "location":
        na = _normalize_location_string(a)
        nb = _normalize_location_string(b)
        return 1 if (nb != "" and na == nb) else 0

    if field in ("institution", "university"):
        if b != "":
            if institutions_match(a, b):
                return 1
            if PRIMARY_ID == "orcid" and sim(a, b) >= sim_threshold:
                return 1
        return 0

    return 1 if (b != "" and sim(a, b) >= sim_threshold) else 0


def _ypred_num(mv, lv):
    if pd.isna(mv):
        return 1 if pd.notna(lv) else 0
    return 1 if (pd.notna(lv) and mv == lv) else 0


def _ypred_bool(mv, lv):
    if mv == "":
        return 1 if lv != "" else 0
    return 1 if (lv != "" and mv == lv) else 0


def compute_field_outcomes(merged_df, manual_col, llm_col, field_type, sim_threshold=None):
    sim_threshold = sim_threshold if sim_threshold is not None else _DEFAULT_THRESHOLD
    field = manual_col.replace("_manual", "") if manual_col.endswith("_manual") else None

    outcomes = []
    m_series = merged_df.get(manual_col, pd.Series([pd.NA] * len(merged_df)))
    l_series = merged_df.get(llm_col, pd.Series([pd.NA] * len(merged_df)))

    if field_type == "text":
        m = m_series.fillna("").astype(str).str.strip()
        l = l_series.fillna("").astype(str).str.strip()
        for a, b in zip(m, l):
            y_true = 1 if a != "" else 0
            y_pred = _ypred_text(a, b, sim_threshold, field=field)
            outcomes.append(outcome_label(y_true, y_pred))

    elif field_type == "num":
        m = pd.to_numeric(m_series, errors="coerce")
        l = pd.to_numeric(l_series, errors="coerce")
        for mv, lv in zip(m, l):
            y_true = 1 if pd.notna(mv) else 0
            y_pred = _ypred_num(mv, lv)
            outcomes.append(outcome_label(y_true, y_pred))

    elif field_type == "bool":
        m = _normalize_bool(m_series)
        l = _normalize_bool(l_series)
        for mv, lv in zip(m, l):
            y_true = 1 if mv != "" else 0
            y_pred = _ypred_bool(mv, lv)
            outcomes.append(outcome_label(y_true, y_pred))
    else:
        outcomes = [""] * len(merged_df)

    return outcomes


def compare_rows(df, text_fields=None, num_fields=None, bool_fields=None, sim_threshold=None, entity_name=""):
    sim_threshold = sim_threshold if sim_threshold is not None else _DEFAULT_THRESHOLD
    text_fields = text_fields or []
    num_fields = num_fields or []
    bool_fields = bool_fields or []

    matched = df[df["_merge"] == "both"].copy()
    results = []

    def field_metrics(mcol, lcol, field_type, label):
        if mcol not in matched.columns or lcol not in matched.columns:
            return None

        if matched.empty:
            return {
                "entity": entity_name,
                "field": label,
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "TN": 0,
                "FP": 0,
                "FN": 0,
                "TP": 0,
            }

        if field_type == "text":
            m = matched[mcol].fillna("").astype(str).str.strip()
            l = matched[lcol].fillna("").astype(str).str.strip()
            y_true = (m != "").astype(int).to_numpy()
            y_pred = np.array(
                [_ypred_text(a, b, sim_threshold, field=label) for a, b in zip(m, l)],
                dtype=int,
            )

        elif field_type == "num":
            m = pd.to_numeric(matched[mcol], errors="coerce")
            l = pd.to_numeric(matched[lcol], errors="coerce")
            y_true = pd.notna(m).astype(int).to_numpy()
            y_pred = np.array([_ypred_num(mv, lv) for mv, lv in zip(m, l)], dtype=int)

        elif field_type == "bool":
            m = _normalize_bool(matched[mcol])
            l = _normalize_bool(matched[lcol])
            y_true = (m != "").astype(int).to_numpy()
            y_pred = np.array([_ypred_bool(mv, lv) for mv, lv in zip(m, l)], dtype=int)

        else:
            return None

        acc = accuracy_score(y_true, y_pred) if len(y_true) else 0
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        return {
            "entity": entity_name,
            "field": label,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        }

    for f in text_fields:
        results.append(field_metrics(f"{f}_manual", f"{f}_llm", "text", f))
    for f in num_fields:
        results.append(field_metrics(f"{f}_manual", f"{f}_llm", "num", f))
    for f in bool_fields:
        results.append(field_metrics(f"{f}_manual", f"{f}_llm", "bool", f))

    return pd.DataFrame([r for r in results if r is not None])


def add_outcome_columns(llm_df, manual_df, key_cols, text_fields=None, num_fields=None, bool_fields=None, sim_threshold=None):
    sim_threshold = sim_threshold if sim_threshold is not None else _DEFAULT_THRESHOLD
    text_fields = text_fields or []
    num_fields = num_fields or []
    bool_fields = bool_fields or []

    merged = pd.merge(llm_df, manual_df, on=key_cols, how="left")

    for field in text_fields:
        merged[f"outcome_{field}"] = compute_field_outcomes(
            merged, f"{field}_manual", f"{field}_llm", "text", sim_threshold
        )
    for field in num_fields:
        merged[f"outcome_{field}"] = compute_field_outcomes(
            merged, f"{field}_manual", f"{field}_llm", "num", sim_threshold
        )
    for field in bool_fields:
        merged[f"outcome_{field}"] = compute_field_outcomes(
            merged, f"{field}_manual", f"{field}_llm", "bool", sim_threshold
        )

    llm_cols = [c for c in merged.columns if c.endswith("_llm") or c in key_cols or c == "dblp_pid"]
    outcome_cols = [c for c in merged.columns if c.startswith("outcome_")]
    final_df = merged[llm_cols + outcome_cols].copy()
    rename_back = {c: c.replace("_llm", "") for c in final_df.columns if c.endswith("_llm")}
    return final_df.rename(columns=rename_back)


def classify_error_type(row):
    field = str(row.get("field", "")).lower()
    outcome = row.get("outcome", "")
    manual = "" if pd.isna(row.get("manual")) else str(row.get("manual")).strip()
    llm = "" if pd.isna(row.get("llm")) else str(row.get("llm")).strip()

    if outcome not in ["FP", "FN"]:
        return ""

    if field == "location":
        if outcome == "FP" and manual == "" and llm != "":
            return "Location hallucination"
        if outcome == "FN" and manual != "" and llm == "":
            return "Location omission"
        return "Other / miscellaneous"

    if field in ["institution", "university"]:
        return "Institution name mismatch"

    if field == "department":
        return "Department boundary ambiguity"

    if field in ["raw_facts", "activities", "projects"]:
        return "Open-ended field error"

    if field == "nationality":
        return "Nationality confusion"

    return "Other / miscellaneous"


def build_error_category_summary(error_df, source_name):
    error_df = error_df.copy()
    error_df["error_type"] = error_df.apply(classify_error_type, axis=1)
    errors = error_df[error_df["error_type"] != ""].copy()

    error_counts = (
        errors.groupby(["error_type", "outcome"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for col in ["FP", "FN"]:
        if col not in error_counts.columns:
            error_counts[col] = 0

    result = error_counts[["error_type", "FP", "FN"]].copy()
    result["source"] = source_name
    return result


def count_multi_entry_merging(llm_df, manual_df, key_col):
    llm_counts = llm_df.groupby(key_col).size().reset_index(name="llm_count")
    manual_counts = manual_df.groupby(key_col).size().reset_index(name="manual_count")
    merged = llm_counts.merge(manual_counts, on=key_col, how="outer").fillna(0)

    fp_count = 0
    fn_count = 0
    for _, row in merged.iterrows():
        llm_count = int(row["llm_count"])
        manual_count = int(row["manual_count"])
        if llm_count < manual_count:
            fn_count += (manual_count - llm_count)
        elif llm_count > manual_count:
            fp_count += (llm_count - manual_count)
    return fp_count, fn_count


def build_full_error_table(
    source_error_df,
    source_name,
    llm_edu_df=None,
    manual_edu_df=None,
    llm_pos_df=None,
    manual_pos_df=None,
    key_col=None,
):
    summary = build_error_category_summary(source_error_df, source_name)

    merge_fp = 0
    merge_fn = 0

    if key_col and llm_edu_df is not None and manual_edu_df is not None:
        fp, fn = count_multi_entry_merging(llm_edu_df, manual_edu_df, key_col)
        merge_fp += fp
        merge_fn += fn

    if key_col and llm_pos_df is not None and manual_pos_df is not None:
        fp, fn = count_multi_entry_merging(llm_pos_df, manual_pos_df, key_col)
        merge_fp += fp
        merge_fn += fn

    has_merging = (summary["error_type"] == "Multi-entry merging").any()
    if not has_merging:
        summary = pd.concat([
            summary,
            pd.DataFrame([{
                "error_type": "Multi-entry merging",
                "FP": merge_fp,
                "FN": merge_fn,
                "source": source_name,
            }])
        ], ignore_index=True)

    return summary.sort_values(["FP", "FN"], ascending=False).reset_index(drop=True)


def build_error_analysis_df(
    matched,
    key_cols,
    text_fields,
    num_fields,
    bool_fields,
    entity_name,
    sim_threshold=None,
):
    sim_threshold = sim_threshold if sim_threshold is not None else _DEFAULT_THRESHOLD
    error_rows = []

    def get_outcome_and_score(m, l, field_type, field=None):
        if field_type == "text":
            a = str(m).strip() if pd.notna(m) else ""
            b = str(l).strip() if pd.notna(l) else ""
            y_true = 1 if a != "" else 0
            y_pred = _ypred_text(a, b, sim_threshold, field=field)
            if field in ("institution", "university"):
                score = 1.0 if institutions_match(a, b) else sim(a, b)
            elif field == "location":
                score = 1.0 if _normalize_location_string(a) == _normalize_location_string(b) and a != "" and b != "" else sim(a, b)
            else:
                score = sim(a, b)

        elif field_type == "num":
            mv = pd.to_numeric(m, errors="coerce")
            lv = pd.to_numeric(l, errors="coerce")
            y_true = 1 if pd.notna(mv) else 0
            y_pred = _ypred_num(mv, lv)
            score = 1.0 if y_pred == 1 else 0.0

        elif field_type == "bool":
            mv = _to_ybool(m)
            lv = _to_ybool(l)
            y_true = 1 if mv != "" else 0
            y_pred = _ypred_bool(mv, lv)
            score = 1.0 if y_pred == 1 else 0.0
        else:
            return "", 0.0

        return outcome_label(y_true, y_pred), score

    for _, row in matched.iterrows():
        base = {k: row[k] for k in matched.columns if k in key_cols}
        base["entity"] = entity_name

        for ft in text_fields + num_fields + bool_fields:
            mcol = f"{ft}_manual"
            lcol = f"{ft}_llm"
            if mcol not in matched.columns or lcol not in matched.columns:
                continue

            field_type = "text" if ft in text_fields else "num" if ft in num_fields else "bool"
            outcome, score = get_outcome_and_score(row[mcol], row[lcol], field_type, field=ft)

            error_rows.append({
                "entity": base["entity"],
                "field": ft,
                "manual": row[mcol],
                "llm": row[lcol],
                "outcome": outcome,
                "sim_score": float(score),
                **{k: base[k] for k in base if k not in ["field", "entity"]},
            })

    return pd.DataFrame(error_rows)