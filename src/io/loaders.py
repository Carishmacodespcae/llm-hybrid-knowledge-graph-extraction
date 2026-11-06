import json
import pandas as pd
import numpy as np

from src.config import (
    MANUAL_AUTHORS_PATH,
    MANUAL_EDU_PATH,
    MANUAL_POS_PATH,
    LLM_JSONL_PATH,
    PRIMARY_ID,            
    ENCODING_CSV,
    ENCODING_JSON
)
from src.utils.cleaning import join_list, normalize_location_col, normalize_degree  
from src.utils.postprocess import clean_entity

def read_csv_flexible(path):
    encodings = [ENCODING_CSV, "utf-8-sig", "utf-8", "cp1252", "latin1"]
    last_error = None

    for enc in dict.fromkeys(encodings):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_error = e

    raise last_error

def format_awards(awards_list):
    if not isinstance(awards_list, list):
        return ""
    formatted = []
    for a in awards_list:
        if not isinstance(a, dict):
            continue
        name = a.get("name", "")
        year = a.get("year")
        
        # Convert list to string if the LLM hallucinated, and safely check it
        year_str = str(year).strip()
        if year is not None and year_str not in ["", "nan", "[]", "None"]:
            # Use year_str here so it prints safely!
            formatted.append(f"{name} ({year_str})")
        else:
            formatted.append(str(name))
    return " | ".join(formatted)


def format_metrics(metrics_list):
    if not isinstance(metrics_list, list):
        return ""
    formatted = []
    for m in metrics_list:
        if not isinstance(m, dict):
            continue
        m_type = m.get("metric_type", "")
        count = m.get("count", "")
        desc = m.get("description", "")
        formatted.append(f"{m_type}: {count} ({desc})")
    return " | ".join(formatted)


def flatten_llm_jsonl(path):
    author_rows = []
    edu_rows = []
    pos_rows = []

    with open(path, "r", encoding=ENCODING_JSON) as f:
        for line in f:
            obj = json.loads(line)

            # ✅ Dynamically fetch the primary ID (either "orcid" or "doi") based on config
            primary_val = obj.get(PRIMARY_ID, "") 
            
            pid = str(obj.get("dblp_pid", "") or "").strip()
            ent = obj.get("entities", {}) or {}
            ent = clean_entity(ent)
            full_name = ent.get("full_name", "")

            personal = (
                ent.get("personal", {}) if isinstance(ent.get("personal"), dict) else {}
            )

            author_rows.append(
                {
                    PRIMARY_ID: primary_val,  # ✅ Flexible ID injection
                    "dblp_pid": pid,
                    "full_name": full_name,
                    "birth_year": personal.get("birth_year"),
                    "birth_location": personal.get("birth_location"),
                    "nationality": personal.get("nationality"),
                    "research_areas": join_list(ent.get("research_areas", [])),
                    "awards": format_awards(ent.get("awards", [])),
                    "scholarly_metrics": format_metrics(ent.get("scholarly_metrics", [])),
                    "projects": join_list(ent.get("projects", [])),
                    "memberships": join_list(ent.get("memberships", [])),
                    "activities": join_list(ent.get("activities", [])),
                    "raw_facts": join_list(ent.get("raw_facts", [])),
                }
            )

            for edu in ent.get("education", []) or []:
                edu_rows.append(
                    {
                        PRIMARY_ID: primary_val,  # ✅ Flexible ID injection
                        "dblp_pid": pid,
                        "full_name": full_name,
                        "degree": normalize_degree(edu.get("degree")), 
                        "field": edu.get("field"),
                        "university": edu.get("university"),
                        "location": edu.get("location"),
                        "edu_year": edu.get("year"),
                        "is_current": edu.get("is_current"),
                    }
                )

            for pos in ent.get("positions", []) or []:
                pos_rows.append(
                    {
                        PRIMARY_ID: primary_val,  # ✅ Flexible ID injection
                        "dblp_pid": pid,
                        "full_name": full_name,
                        "role": pos.get("role"),
                        "department": pos.get("department"),
                        "institution": pos.get("institution"),
                        "location": pos.get("location"),
                        "start_year": pos.get("start_year"),
                        "end_year": pos.get("end_year"),
                        "is_current": pos.get("is_current"),
                        "experience_years": pos.get("experience_years"),
                        "description": pos.get("description"),
                    }
                )

    return (
        pd.DataFrame(author_rows),
        pd.DataFrame(edu_rows),
        pd.DataFrame(pos_rows),
    )


def prepare_manual_authors(path=MANUAL_AUTHORS_PATH):
    df = read_csv_flexible(path)
    df["dblp_pid_clean"] = df["dblp_pid"].astype(str).str.strip().str.lower()
    df = normalize_location_col(df, "birth_location") 
#    df["_original_order"] = range(len(df)) # <--- ADD THIS LINE

    return df


def prepare_manual_edu(path=MANUAL_EDU_PATH):
    df = read_csv_flexible(path)
    df["dblp_pid_clean"] = df["dblp_pid"].astype(str).str.strip().str.lower()
    if "degree" in df.columns:
        df["degree"] = df["degree"].apply(normalize_degree) 
    return df


def prepare_manual_pos(path=MANUAL_POS_PATH):
    df = read_csv_flexible(path)
    df["dblp_pid_clean"] = df["dblp_pid"].astype(str).str.strip().str.lower()
    return df


def prepare_llm_data(path=LLM_JSONL_PATH):
    llm_auth, llm_edu, llm_pos = flatten_llm_jsonl(path)

    llm_auth["dblp_pid_clean"] = llm_auth["dblp_pid"].astype(str).str.strip().str.lower()
    llm_edu["dblp_pid_clean"] = llm_edu["dblp_pid"].astype(str).str.strip().str.lower()
    llm_pos["dblp_pid_clean"] = llm_pos["dblp_pid"].astype(str).str.strip().str.lower()

    for df in [llm_auth, llm_edu, llm_pos]:
        if PRIMARY_ID in df.columns:
            df[PRIMARY_ID] = df[PRIMARY_ID].astype(str).str.strip()

    llm_auth = normalize_location_col(llm_auth, "birth_location")
    return llm_auth, llm_edu, llm_pos


def load_all_data():
    manual_auth = prepare_manual_authors()
    manual_edu = prepare_manual_edu()
    manual_pos = prepare_manual_pos()

    llm_auth, llm_edu, llm_pos = prepare_llm_data()

    manual_data = {
        "authors": manual_auth,
        "education": manual_edu,
        "positions": manual_pos,
    }
    llm_data = {"authors": llm_auth, "education": llm_edu, "positions": llm_pos}

    return manual_data, llm_data