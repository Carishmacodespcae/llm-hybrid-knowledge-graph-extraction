import numpy as np
import pandas as pd

from src.config import ENTITY_CONFIG, JOIN_KEYS, SIM_THRESHOLD, PRIMARY_ID
from src.utils.metrics import (
    compare_rows,
    add_outcome_columns,
    build_error_analysis_df,
)


def rename_with_suffix(df, fields, suffix):
    rename_map = {field: f"{field}_{suffix}" for field in fields}
    return df.rename(columns=rename_map)


def add_concurrent_job_flag(edu_df, pos_df):
    """
    Adds a 'has_concurrent_job' boolean column to the education dataframe.
    True if the author has is_current=True in education AND is_current=True in any position.
    """
    edu = edu_df.copy()
    pos = pos_df.copy()

    if "is_current" in edu.columns:
        edu["is_current_bool"] = (
            edu["is_current"]
            .fillna(False)
            .astype(str)
            .str.lower()
            .map({"true": True, "1": True, "1.0": True})
            .fillna(False)
        )
    else:
        edu["is_current_bool"] = False

    if "is_current" in pos.columns:
        pos["is_current_bool"] = (
            pos["is_current"]
            .fillna(False)
            .astype(str)
            .str.lower()
            .map({"true": True, "1": True, "1.0": True})
            .fillna(False)
        )
    else:
        pos["is_current_bool"] = False

    group_keys = [k for k in JOIN_KEYS if k in pos.columns and k in edu.columns]
    if group_keys:
        current_jobs = pos.groupby(group_keys)["is_current_bool"].any().reset_index()
        current_jobs.rename(columns={"is_current_bool": "has_any_current_job"}, inplace=True)
        edu = pd.merge(edu, current_jobs, on=group_keys, how="left")
    else:
        edu["has_any_current_job"] = False

    edu["has_any_current_job"] = edu["has_any_current_job"].fillna(False)
    edu["has_concurrent_job"] = edu["is_current_bool"] & edu["has_any_current_job"]
    edu = edu.drop(columns=["is_current_bool", "has_any_current_job"])
    return edu


def prepare_education_eval_dfs(manual_edu_df, llm_edu_df):
    cfg = ENTITY_CONFIG["education"]

    manual_edu_eval = manual_edu_df[cfg["manual_cols"]].copy()
    manual_edu_eval = rename_with_suffix(
        manual_edu_eval,
        cfg["rename_fields"],
        "manual",
    )

    llm_edu_eval = rename_with_suffix(
        llm_edu_df.copy(),
        cfg["rename_fields"],
        "llm",
    )

    return manual_edu_eval, llm_edu_eval


def merge_education_data(manual_edu_eval, llm_edu_eval):
    manual = manual_edu_eval.copy()
    llm = llm_edu_eval.copy()

    sort_cols_manual = [
        PRIMARY_ID,
        "dblp_pid_clean",
        "degree_manual",
        "university_manual",
        "location_manual",
    ]
    sort_cols_llm = [
        PRIMARY_ID,
        "dblp_pid_clean",
        "degree_llm",
        "university_llm",
        "location_llm",
    ]

    sort_cols_manual = [c for c in sort_cols_manual if c in manual.columns]
    sort_cols_llm = [c for c in sort_cols_llm if c in llm.columns]

    manual = manual.sort_values(sort_cols_manual).copy()
    llm = llm.sort_values(sort_cols_llm).copy()

    manual["_pair_id"] = manual.groupby(JOIN_KEYS).cumcount()
    llm["_pair_id"] = llm.groupby(JOIN_KEYS).cumcount()

    merged = pd.merge(
        manual,
        llm,
        on=JOIN_KEYS + ["_pair_id"],
        how="outer",
        indicator=True,
    )

    return merged, manual, llm


def build_base_researcher_df(manual_auth_df, llm_auth_df):
    base_df = pd.concat(
        [
            manual_auth_df[JOIN_KEYS].copy(),
            llm_auth_df[JOIN_KEYS].copy(),
        ],
        ignore_index=True,
    ).drop_duplicates()

    return base_df


def build_education_presence_df(manual_auth_df, llm_auth_df, manual_edu_df, llm_edu_df):
    base_df = build_base_researcher_df(manual_auth_df, llm_auth_df)

    edu_content_cols = ["degree", "field", "university", "location", "edu_year"]

    def normalize_cell(val):
        if pd.isna(val):
            return ""
        v = str(val).strip()
        return "" if v.lower() in {"", "nan", "none", "<na>"} else v

    def has_real_education(df):
        df = df.copy()
        existing_cols = [c for c in edu_content_cols if c in df.columns]

        if not existing_cols:
            return pd.DataFrame(columns=JOIN_KEYS + ["has_education"])

        for col in existing_cols:
            df[col] = df[col].apply(normalize_cell)

        df["has_education"] = df[existing_cols].apply(
            lambda row: any(v != "" for v in row),
            axis=1,
        )

        df = df[df["has_education"]].copy()
        return df[JOIN_KEYS].drop_duplicates().assign(has_education=1)

    manual_presence = has_real_education(manual_edu_df).rename(
        columns={"has_education": "manual_has_education"}
    )
    llm_presence = has_real_education(llm_edu_df).rename(
        columns={"has_education": "llm_has_education"}
    )

    df = base_df.merge(manual_presence, on=JOIN_KEYS, how="left")
    df = df.merge(llm_presence, on=JOIN_KEYS, how="left")

    df["manual_has_education"] = df["manual_has_education"].fillna(0).astype(int)
    df["llm_has_education"] = df["llm_has_education"].fillna(0).astype(int)

    def get_outcome(row):
        y_true = row["manual_has_education"]
        y_pred = row["llm_has_education"]
        if y_true == 1 and y_pred == 1:
            return "TP"
        elif y_true == 0 and y_pred == 0:
            return "TN"
        elif y_true == 0 and y_pred == 1:
            return "FP"
        else:
            return "FN"

    df["entity"] = "Education"
    df["field"] = "education_exists"
    df["manual"] = df["manual_has_education"]
    df["llm"] = df["llm_has_education"]
    df["outcome"] = df.apply(get_outcome, axis=1)
    df["sim_score"] = 1.0

    return df[["entity", "field", "manual", "llm", "outcome", "sim_score"] + JOIN_KEYS].copy()


def build_education_presence_metrics(presence_df):
    vc = presence_df["outcome"].value_counts().to_dict()
    tp = vc.get("TP", 0)
    tn = vc.get("TN", 0)
    fp = vc.get("FP", 0)
    fn = vc.get("FN", 0)
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return pd.DataFrame(
        [
            {
                "entity": "Education",
                "field": "education_exists",
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
            }
        ]
    )


def evaluate_education(
    manual_auth_df,
    llm_auth_df,
    manual_edu_df,
    llm_edu_df,
    manual_pos_df,
    llm_pos_df,
):
    cfg = ENTITY_CONFIG["education"]

    manual_edu_df = add_concurrent_job_flag(manual_edu_df, manual_pos_df)
    llm_edu_df = add_concurrent_job_flag(llm_edu_df, llm_pos_df)

    manual_edu_eval, llm_edu_eval = prepare_education_eval_dfs(
        manual_edu_df,
        llm_edu_df,
    )

    edu_merged, manual_edu_eval, llm_edu_eval = merge_education_data(
        manual_edu_eval,
        llm_edu_eval,
    )

    presence_error_df = build_education_presence_df(
        manual_auth_df,
        llm_auth_df,
        manual_edu_df,
        llm_edu_df,
    )
    presence_metrics = build_education_presence_metrics(presence_error_df)

    edu_metrics = compare_rows(
        edu_merged,
        text_fields=cfg["text_fields"],
        num_fields=cfg["num_fields"],
        bool_fields=cfg["bool_fields"],
        sim_threshold=SIM_THRESHOLD,
        entity_name=cfg["entity_name"],
    )

    edu_metrics = pd.concat([presence_metrics, edu_metrics], ignore_index=True)

    llm_edu_out = add_outcome_columns(
        llm_edu_eval,
        manual_edu_eval,
        key_cols=JOIN_KEYS + ["_pair_id"],
        text_fields=cfg["text_fields"],
        num_fields=cfg["num_fields"],
        bool_fields=cfg["bool_fields"],
        sim_threshold=SIM_THRESHOLD,
    )

    edu_error_df = build_error_analysis_df(
        edu_merged[edu_merged["_merge"] == "both"],
        key_cols=JOIN_KEYS + ["_pair_id"],
        text_fields=cfg["text_fields"],
        num_fields=cfg["num_fields"],
        bool_fields=cfg["bool_fields"],
        entity_name="Education",
    )

    edu_error_df = pd.concat([presence_error_df, edu_error_df], ignore_index=True)

    return {
        "merged": edu_merged,
        "metrics": edu_metrics,
        "llm_output": llm_edu_out,
        "error_output": edu_error_df,
    }