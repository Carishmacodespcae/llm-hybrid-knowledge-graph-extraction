import pandas as pd

from src.config import ENTITY_CONFIG, JOIN_KEYS, SIM_THRESHOLD
from src.utils.metrics import (
    compare_rows,
    add_outcome_columns,
    build_error_analysis_df,
)


def rename_with_suffix(df, fields, suffix):
    rename_map = {field: f"{field}_{suffix}" for field in fields}
    return df.rename(columns=rename_map)


def prepare_author_eval_dfs(manual_auth_df, llm_auth_df):
    cfg = ENTITY_CONFIG["authors"]

    # Combine JOIN_KEYS and config columns, but use a set to REMOVE DUPLICATES,
    # then convert back to list so the order is maintained/valid for pandas.
    # We use a dictionary fromkeys() trick to remove duplicates while keeping the original order.
    manual_keep_cols = list(dict.fromkeys(JOIN_KEYS + cfg["manual_cols"]))
    llm_keep_cols = list(dict.fromkeys(JOIN_KEYS + cfg["rename_fields"]))

    # Safely select columns
    manual_auth_eval = manual_auth_df.reindex(columns=manual_keep_cols).copy()

    # Fill missing columns with empty strings
    for col in cfg["manual_cols"]:
        if manual_auth_eval[col].isna().all():
            manual_auth_eval[col] = ""

    manual_auth_eval = rename_with_suffix(
        manual_auth_eval,
        cfg["rename_fields"],
        "manual",
    )

    # Do the same for LLM data
    llm_auth_eval = llm_auth_df.reindex(columns=llm_keep_cols).copy()
    for col in cfg["rename_fields"]:
        if llm_auth_eval[col].isna().all():
            llm_auth_eval[col] = ""

    llm_auth_eval = rename_with_suffix(
        llm_auth_eval,
        cfg["rename_fields"],
        "llm",
    )

    return manual_auth_eval, llm_auth_eval


def merge_author_data(manual_auth_eval, llm_auth_eval):
    return pd.merge(
        manual_auth_eval,
        llm_auth_eval,
        on=JOIN_KEYS,
        how="outer",
        indicator=True,
    )


def evaluate_authors(manual_auth_df, llm_auth_df):
    cfg = ENTITY_CONFIG["authors"]

    manual_auth_eval, llm_auth_eval = prepare_author_eval_dfs(
        manual_auth_df,
        llm_auth_df,
    )

    auth_merged = merge_author_data(manual_auth_eval, llm_auth_eval)

    auth_metrics = compare_rows(
        auth_merged,
        text_fields=cfg["text_fields"],
        num_fields=cfg["num_fields"],
        bool_fields=cfg["bool_fields"],
        sim_threshold=SIM_THRESHOLD,
        entity_name=cfg["entity_name"],
    )

    llm_auth_out = add_outcome_columns(
        llm_auth_eval,
        manual_auth_eval,
        key_cols=JOIN_KEYS,
        text_fields=cfg["text_fields"],
        num_fields=cfg["num_fields"],
        bool_fields=cfg["bool_fields"],
        sim_threshold=SIM_THRESHOLD,
    )

    auth_error_df = build_error_analysis_df(
        auth_merged[auth_merged["_merge"] == "both"],
        key_cols=JOIN_KEYS,
        text_fields=cfg["text_fields"],
        num_fields=cfg["num_fields"],
        bool_fields=cfg["bool_fields"],
        entity_name="Author",
    )

    return {
        "merged": auth_merged,
        "metrics": auth_metrics,
        "llm_output": llm_auth_out,
        "error_output": auth_error_df,
    }
