import pandas as pd
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
pd.set_option("future.no_silent_downcasting", True)

from src.config import (
    OUT_LLM_AUTHORS,
    OUT_LLM_EDU,
    OUT_LLM_POS,
    OUT_METRICS_CSV,
    OUT_AUTHOR_REVIEW_XLSX,
    OUT_EDU_REVIEW_XLSX,
    OUT_POSITION_REVIEW_XLSX,
)
from src.io.loaders import load_all_data
from src.evaluation.authors import evaluate_authors
from src.evaluation.education import evaluate_education
from src.evaluation.positions import evaluate_positions


def build_count_sheet(error_df, entity_name):
    if error_df.empty or "outcome" not in error_df.columns:
        return pd.DataFrame(
            [
                {
                    "entity": entity_name,
                    "field": "N/A",
                    "TP": 0,
                    "TN": 0,
                    "FP": 0,
                    "FN": 0,
                    "total": 0,
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                }
            ]
        )

    rows = []
    for field, grp in error_df.groupby("field"):
        vc = grp["outcome"].value_counts().to_dict()
        tp = vc.get("TP", 0)
        tn = vc.get("TN", 0)
        fp = vc.get("FP", 0)
        fn = vc.get("FN", 0)
        tot = tp + tn + fp + fn

        accuracy = (tp + tn) / tot if tot else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        rows.append(
            {
                "entity": entity_name,
                "field": field,
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "total": tot,
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        )

    all_tp = sum(r["TP"] for r in rows)
    all_tn = sum(r["TN"] for r in rows)
    all_fp = sum(r["FP"] for r in rows)
    all_fn = sum(r["FN"] for r in rows)
    all_tot = all_tp + all_tn + all_fp + all_fn

    all_acc = (all_tp + all_tn) / all_tot if all_tot else 0.0
    all_prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    all_rec = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    all_f1 = (
        (2 * all_prec * all_rec) / (all_prec + all_rec)
        if (all_prec + all_rec) > 0
        else 0.0
    )

    rows.append(
        {
            "entity": entity_name,
            "field": "** OVERALL **",
            "TP": all_tp,
            "TN": all_tn,
            "FP": all_fp,
            "FN": all_fn,
            "total": all_tot,
            "accuracy": round(all_acc, 4),
            "precision": round(all_prec, 4),
            "recall": round(all_rec, 4),
            "f1": round(all_f1, 4),
        }
    )

    return pd.DataFrame(rows)


def with_error_flag(df):
    df = df.copy()
    df["has_error"] = df["outcome"].apply(
        lambda x: "Error" if x in ("FP", "FN") else "OK"
    )
    return df


def save_outputs(author_results, education_results, position_results):
    author_results["llm_output"].to_csv(OUT_LLM_AUTHORS, index=False, encoding="utf-8")
    education_results["llm_output"].to_csv(OUT_LLM_EDU, index=False, encoding="utf-8")
    position_results["llm_output"].to_csv(OUT_LLM_POS, index=False, encoding="utf-8")

    all_metrics = pd.concat(
        [
            author_results["metrics"],
            education_results["metrics"],
            position_results["metrics"],
        ],
        ignore_index=True,
    )
    all_metrics.to_csv(OUT_METRICS_CSV, index=False, encoding="utf-8")

    author_count_sheet = build_count_sheet(author_results["error_output"], "Authors")
    edu_count_sheet = build_count_sheet(education_results["error_output"], "Education")
    position_count_sheet = build_count_sheet(
        position_results["error_output"], "Positions"
    )

    author_review = with_error_flag(author_results["error_output"])
    edu_review = with_error_flag(education_results["error_output"])
    pos_review = with_error_flag(position_results["error_output"])

    def sort_review_df(llm_out, review_df):
        if review_df.empty or llm_out.empty:
            return review_df

        id_col = None
        for col in ["orcid", "doi", "dblp_pid", "dblp_pid_clean"]:
            if col in llm_out.columns and col in review_df.columns:
                id_col = col
                break

        if not id_col:
            return review_df

        correct_order = llm_out[id_col].drop_duplicates().tolist()
        review_df[id_col] = pd.Categorical(
            review_df[id_col],
            categories=correct_order,
            ordered=True,
        )
        return review_df.sort_values(id_col, kind="stable")

    author_review = sort_review_df(author_results["llm_output"], author_review)
    edu_review = sort_review_df(education_results["llm_output"], edu_review)
    pos_review = sort_review_df(position_results["llm_output"], pos_review)

    with pd.ExcelWriter(OUT_AUTHOR_REVIEW_XLSX, engine="openpyxl") as writer:
        author_review.to_excel(writer, sheet_name="manual_vs_llm", index=False)
        author_count_sheet.to_excel(writer, sheet_name="count_sheet", index=False)

    with pd.ExcelWriter(OUT_EDU_REVIEW_XLSX, engine="openpyxl") as writer:
        edu_review.to_excel(writer, sheet_name="manual_vs_llm", index=False)
        edu_count_sheet.to_excel(writer, sheet_name="count_sheet", index=False)

    with pd.ExcelWriter(OUT_POSITION_REVIEW_XLSX, engine="openpyxl") as writer:
        pos_review.to_excel(writer, sheet_name="manual_vs_llm", index=False)
        position_count_sheet.to_excel(writer, sheet_name="count_sheet", index=False)

    return all_metrics


def main():
    manual_data, llm_data = load_all_data()

    author_results = evaluate_authors(
        manual_data["authors"],
        llm_data["authors"],
    )

    education_results = evaluate_education(
        manual_data["authors"],
        llm_data["authors"],
        manual_data["education"],
        llm_data["education"],
        manual_data["positions"],
        llm_data["positions"],
    )

    position_results = evaluate_positions(
        manual_data["authors"],
        llm_data["authors"],
        manual_data["positions"],
        llm_data["positions"],
    )

    save_outputs(author_results, education_results, position_results)

    print("Done.")
    print("Saved:")
    print(OUT_LLM_AUTHORS)
    print(OUT_LLM_EDU)
    print(OUT_LLM_POS)
    print(OUT_METRICS_CSV)
    print(OUT_AUTHOR_REVIEW_XLSX)
    print(OUT_EDU_REVIEW_XLSX)
    print(OUT_POSITION_REVIEW_XLSX)


if __name__ == "__main__":
    main()