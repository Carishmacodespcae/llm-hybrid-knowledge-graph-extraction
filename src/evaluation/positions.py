import pandas as pd

from src.config import ENTITY_CONFIG, JOIN_KEYS, SIM_THRESHOLD, PRIMARY_ID
from src.utils.metric import (
    compare_rows,
    compute_field_outcomes,
    build_error_analysis_df,
)
from src.utils.cleaning import sim, institutions_match


def rename_with_suffix(df, fields, suffix):
    rename_map = {field: f"{field}_{suffix}" for field in fields}
    return df.rename(columns=rename_map)


def prepare_position_eval_dfs(manual_pos_df, llm_pos_df):
    cfg = ENTITY_CONFIG["positions"]

    manual_pos_eval = manual_pos_df[cfg["manual_cols"]].copy()
    manual_pos_eval = rename_with_suffix(
        manual_pos_eval, cfg["rename_fields"], "manual"
    )

    llm_pos_eval = rename_with_suffix(llm_pos_df.copy(), cfg["rename_fields"], "llm")

    return manual_pos_eval, llm_pos_eval


def merge_position_data(manual_pos_eval, llm_pos_eval):
    """
    Similarity-based best-match pairing.
    role (60%) + institution (40%) similarity determines the best LLM match per manual row.

    _merge values:
      "both"       -> matched pair             (evaluate field-by-field)
      "left_only"  -> manual only, LLM missed (FN)
      "right_only" -> LLM only, not in manual (FP)
    """
    manual = manual_pos_eval.copy().reset_index(drop=True)
    llm = llm_pos_eval.copy().reset_index(drop=True)

    all_manual_cols = [c for c in manual.columns if c not in JOIN_KEYS]
    all_llm_cols = [c for c in llm.columns if c not in JOIN_KEYS]

    rows = []
    used_llm_idx = set()

    for _, m_row in manual.iterrows():
        mask = pd.Series([True] * len(llm), index=llm.index)
        for k in JOIN_KEYS:
            mask &= llm[k].astype(str).str.strip() == str(m_row[k]).strip()
        candidates = llm[mask]

        best_score, best_idx = -1.0, None
        for llm_idx, l_row in candidates.iterrows():
            if llm_idx in used_llm_idx:
                continue

            role_sim = sim(
                str(m_row.get("role_manual", "") or ""),
                str(l_row.get("role_llm", "") or ""),
            )

            inst_m = str(m_row.get("institution_manual", "") or "")
            inst_l = str(l_row.get("institution_llm", "") or "")
            inst_sim = 1.0 if institutions_match(inst_m, inst_l) else sim(inst_m, inst_l)

            score = role_sim * 0.6 + inst_sim * 0.4

            if score > best_score:
                best_score, best_idx = score, llm_idx

        row = {k: m_row[k] for k in JOIN_KEYS}
        for col in all_manual_cols:
            row[col] = m_row.get(col)

        if best_idx is not None:
            used_llm_idx.add(best_idx)
            for col in all_llm_cols:
                row[col] = llm.loc[best_idx].get(col)
            row["_merge"] = "both"
        else:
            for col in all_llm_cols:
                row[col] = None
            row["_merge"] = "left_only"

        rows.append(row)

    for llm_idx, l_row in llm.iterrows():
        if llm_idx in used_llm_idx:
            continue

        mask = pd.Series([True] * len(manual), index=manual.index)
        for k in JOIN_KEYS:
            mask &= manual[k].astype(str).str.strip() == str(l_row[k]).strip()

        if not manual[mask].empty:
            row = {k: l_row[k] for k in JOIN_KEYS}
            for col in all_manual_cols:
                row[col] = None
            for col in all_llm_cols:
                row[col] = l_row.get(col)
            row["_merge"] = "right_only"
            rows.append(row)

    merged = pd.DataFrame(rows) if rows else pd.DataFrame()
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


def build_position_presence_df(manual_auth_df, llm_auth_df, manual_pos_df, llm_pos_df):
    base_df = build_base_researcher_df(manual_auth_df, llm_auth_df)

    pos_content_cols = [
        "role",
        "department",
        "institution",
        "location",
        "start_year",
        "end_year",
    ]

    def normalize_cell(val):
        if pd.isna(val):
            return ""
        v = str(val).strip()
        return "" if v.lower() in {"", "nan", "none", "<na>"} else v

    def has_real_position(df):
        df = df.copy()
        existing_cols = [c for c in pos_content_cols if c in df.columns]

        if not existing_cols:
            return pd.DataFrame(columns=JOIN_KEYS + ["has_position"])

        for col in existing_cols:
            df[col] = df[col].apply(normalize_cell)

        df["has_position"] = df[existing_cols].apply(
            lambda row: any(v != "" for v in row),
            axis=1,
        )

        df = df[df["has_position"]].copy()
        return df[JOIN_KEYS].drop_duplicates().assign(has_position=1)

    manual_presence = has_real_position(manual_pos_df).rename(
        columns={"has_position": "manual_has_position"}
    )
    llm_presence = has_real_position(llm_pos_df).rename(
        columns={"has_position": "llm_has_position"}
    )

    df = base_df.merge(manual_presence, on=JOIN_KEYS, how="left")
    df = df.merge(llm_presence, on=JOIN_KEYS, how="left")

    df["manual_has_position"] = df["manual_has_position"].fillna(0).astype(int)
    df["llm_has_position"] = df["llm_has_position"].fillna(0).astype(int)

    def get_outcome(row):
        y_true = row["manual_has_position"]
        y_pred = row["llm_has_position"]
        if y_true == 1 and y_pred == 1:
            return "TP"
        elif y_true == 0 and y_pred == 0:
            return "TN"
        elif y_true == 0 and y_pred == 1:
            return "FP"
        else:
            return "FN"

    df["entity"] = "Position"
    df["field"] = "position_exists"
    df["manual"] = df["manual_has_position"]
    df["llm"] = df["llm_has_position"]
    df["outcome"] = df.apply(get_outcome, axis=1)
    df["sim_score"] = 1.0

    return df[["entity", "field", "manual", "llm", "outcome", "sim_score"] + JOIN_KEYS].copy()


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
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return pd.DataFrame(
        [
            {
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
            }
        ]
    )


def evaluate_positions(manual_auth_df, llm_auth_df, manual_pos_df, llm_pos_df):
    cfg = ENTITY_CONFIG["positions"]

    # Optional debug
    if "full_name" in manual_pos_df.columns:
        print("=== MANUAL POS rows for Fatiha ===")
        bl = manual_pos_df[manual_pos_df["full_name"].str.contains("Fatiha", na=False)]
        debug_cols = [
            c
            for c in [PRIMARY_ID, "dblp_pid", "dblp_pid_clean", "role", "institution"]
            if c in bl.columns
        ]
        if debug_cols:
            print(bl[debug_cols].to_string())

    if "full_name" in llm_pos_df.columns:
        print("\n=== LLM POS rows for Fatiha ===")
        bl2 = llm_pos_df[llm_pos_df["full_name"].str.contains("Fatiha", na=False)]
        debug_cols2 = [
            c
            for c in [PRIMARY_ID, "dblp_pid", "dblp_pid_clean", "role", "institution"]
            if c in bl2.columns
        ]
        if debug_cols2:
            print(bl2[debug_cols2].to_string())

    manual_pos_eval, llm_pos_eval = prepare_position_eval_dfs(manual_pos_df, llm_pos_df)

    print("\n=== DEBUG: positions ===")
    print("manual_pos_df:", len(manual_pos_df))
    print("llm_pos_df:", len(llm_pos_df))
    print("manual_pos_eval:", len(manual_pos_eval))
    print("llm_pos_eval:", len(llm_pos_eval))

    pos_merged, manual_pos_eval, llm_pos_eval = merge_position_data(
        manual_pos_eval, llm_pos_eval
    )

    print("pos_merged:", len(pos_merged))
    if not pos_merged.empty and "_merge" in pos_merged.columns:
        print(pos_merged["_merge"].value_counts(dropna=False).to_string())

    presence_error_df = build_position_presence_df(
        manual_auth_df,
        llm_auth_df,
        manual_pos_df,
        llm_pos_df,
    )
    presence_metrics = build_position_presence_metrics(presence_error_df)

    pos_metrics = compare_rows(
        pos_merged,
        text_fields=cfg["text_fields"],
        num_fields=cfg["num_fields"],
        bool_fields=cfg["bool_fields"],
        sim_threshold=SIM_THRESHOLD,
        entity_name=cfg["entity_name"],
    )

    pos_metrics = pd.concat([presence_metrics, pos_metrics], ignore_index=True)

    llm_pos_out = pos_merged[pos_merged["_merge"].isin(["both", "right_only"])].copy()

    for field in cfg["text_fields"]:
        llm_pos_out[f"outcome_{field}"] = compute_field_outcomes(
            llm_pos_out, f"{field}_manual", f"{field}_llm", "text", SIM_THRESHOLD
        )

    for field in cfg["num_fields"]:
        llm_pos_out[f"outcome_{field}"] = compute_field_outcomes(
            llm_pos_out, f"{field}_manual", f"{field}_llm", "num", SIM_THRESHOLD
        )

    for field in cfg["bool_fields"]:
        llm_pos_out[f"outcome_{field}"] = compute_field_outcomes(
            llm_pos_out, f"{field}_manual", f"{field}_llm", "bool", SIM_THRESHOLD
        )

    llm_cols = [c for c in llm_pos_out.columns if c.endswith("_llm") or c in JOIN_KEYS]
    outcome_cols = [c for c in llm_pos_out.columns if c.startswith("outcome_")]
    llm_pos_out = llm_pos_out[llm_cols + outcome_cols].copy()
    llm_pos_out = llm_pos_out.rename(
        columns={c: c.replace("_llm", "") for c in llm_cols if c.endswith("_llm")}
    )
    
    print("llm_pos_out final:", len(llm_pos_out))


    pos_error_df = build_error_analysis_df(
        pos_merged[pos_merged["_merge"] == "both"] if not pos_merged.empty else pos_merged,
        key_cols=JOIN_KEYS,
        text_fields=cfg["text_fields"],
        num_fields=cfg["num_fields"],
        bool_fields=cfg["bool_fields"],
        entity_name="Position",
    )

    print("pos_error_df before concat:", len(pos_error_df))



    pos_error_df = pd.concat([presence_error_df, pos_error_df], ignore_index=True)

    print("presence_error_df:", len(presence_error_df))
    print("pos_error_df final:", len(pos_error_df))
    print("=== END DEBUG ===\n")
    
    return {
        "merged": pos_merged,
        "metrics": pos_metrics,
        "llm_output": llm_pos_out,
        "error_output": pos_error_df,
    }