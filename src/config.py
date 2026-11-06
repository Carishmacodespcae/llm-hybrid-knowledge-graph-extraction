from pathlib import Path

# =========================================================
# Base paths
# =========================================================
BASE_DIR = Path(r"C:\Carishma\Thesis Prep\Prep 2\Implementation\05_Evaluation")
DATA_DIR = BASE_DIR
OUTPUT_DIR = BASE_DIR / "output" / "Verification"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# =========================================================

# ── MODE: "ieee" or "orcid" ───────────────────────────────
DATA_SOURCE = "orcid"   # ← change to "ieee" for IEEE, "orcid" for ORCID
# ─────────────────────────────────────────────────────────

# =========================================================
# Dynamic File Paths Based on Mode
# =========================================================
if DATA_SOURCE == "ieee":
    # Inputs
    LLM_JSONL_PATH      = DATA_DIR / "Entire Run" / "ieee_complete.jsonl"
   # LLM_JSONL_PATH      = DATA_DIR / "Entire Run" / "ieee_prod_output_390k_FINAL.jsonl"
    MANUAL_AUTHORS_PATH = DATA_DIR / "Manual" / "ieee_manual_authors.csv"
    MANUAL_EDU_PATH     = DATA_DIR / "Manual" / "ieee_manual_education.csv"
    MANUAL_POS_PATH     = DATA_DIR / "Manual" / "ieee_manual_positions.csv"
    
    # Keys
    JOIN_KEYS           = ["doi", "dblp_pid_clean"]
    PRIMARY_ID          = "doi"
     # Outputs
    OUT_LLM_AUTHORS = OUTPUT_DIR / "ieee_authors_llm.csv"
    OUT_LLM_EDU = OUTPUT_DIR / "ieee_education_llm.csv"
    OUT_LLM_POS = OUTPUT_DIR / "ieee_positions_llm.csv"

    OUT_METRICS_CSV = OUTPUT_DIR / "ieee_metrics.csv"
    OUT_AUTHOR_REVIEW_XLSX = OUTPUT_DIR / "ieee_authors_review.xlsx"
    OUT_EDU_REVIEW_XLSX = OUTPUT_DIR / "ieee_education_review.xlsx"
    OUT_POSITION_REVIEW_XLSX = OUTPUT_DIR / "ieee_positions_review.xlsx"

elif DATA_SOURCE == "orcid":
    # Inputs
    LLM_JSONL_PATH      = DATA_DIR / "orcid" / "orcid_complete.jsonl"
    MANUAL_AUTHORS_PATH = DATA_DIR / "Manual" / "orcid_manual_authors.csv"
    MANUAL_EDU_PATH     = DATA_DIR / "Manual" / "orcid_manual_education.csv"
    MANUAL_POS_PATH     = DATA_DIR / "Manual" / "orcid_manual_positions.csv"
    
    # Keys
    JOIN_KEYS           = ["orcid", "dblp_pid_clean"]
    PRIMARY_ID          = "orcid"
    
    # Outputs - LLM Extracted CSVs
    #OUT_LLM_AUTHORS     = OUTPUT_DIR / "llm_extracted_authors_orcid.csv"
    #OUT_LLM_EDU         = OUTPUT_DIR / "llm_extracted_education_orcid.csv"
    #OUT_LLM_POS         = OUTPUT_DIR / "llm_extracted_positions_orcid.csv"
    
    # Outputs - Metrics and Review
    #OUT_METRICS_CSV          = OUTPUT_DIR / "evaluation_metrics_orcid.csv"
    #OUT_AUTHOR_REVIEW_XLSX   = OUTPUT_DIR / "review_authors_orcid.xlsx"
    #OUT_EDU_REVIEW_XLSX      = OUTPUT_DIR / "review_education_orcid.xlsx"
    #OUT_POSITION_REVIEW_XLSX = OUTPUT_DIR / "review_positions_orcid.xlsx"

    # Outputs
    OUT_LLM_AUTHORS = OUTPUT_DIR / "orcid_authors_llm.csv"
    OUT_LLM_EDU = OUTPUT_DIR / "orcid_education_llm.csv"
    OUT_LLM_POS = OUTPUT_DIR / "orcid_positions_llm.csv"

    OUT_METRICS_CSV = OUTPUT_DIR / "orcid_metrics.csv"
    OUT_AUTHOR_REVIEW_XLSX = OUTPUT_DIR / "orcid_authors_review.xlsx"
    OUT_EDU_REVIEW_XLSX = OUTPUT_DIR / "orcid_education_review.xlsx"
    OUT_POSITION_REVIEW_XLSX = OUTPUT_DIR / "orcid_positions_review.xlsx"
else:
    raise ValueError("DATA_SOURCE must be either 'ieee' or 'orcid'")


# =========================================================
# Global settings
# =========================================================
SIM_THRESHOLD = 0.75
#ENCODING_CSV = "cp1252"  # manual CSV files saved from Excel
ENCODING_CSV = "utf-8-sig" 
ENCODING_JSON = "utf-8"  # LLM JSONL output

# =========================================================
# Entity field definitions
# =========================================================
ENTITY_CONFIG = {
    "authors": {
        "entity_name": "Authors",
        "manual_cols": [
            PRIMARY_ID,
            "dblp_pid_clean",
            "full_name",
            "birth_year",
            "birth_location",
            "nationality",
            "research_areas",
            "awards",
            "scholarly_metrics",
            "projects",
            "memberships",
            "activities",
            "raw_facts",
        ],
        "text_fields": [
            "full_name",
            "birth_location",
            "nationality",
            "research_areas",
            "awards",
            "scholarly_metrics",
            "projects",
            "memberships",
            "activities",
            "raw_facts",
        ],
        "num_fields": ["birth_year"],
        "bool_fields": [],
        "rename_fields": [
            "full_name",
            "birth_year",
            "birth_location",
            "nationality",
            "research_areas",
            "awards",
            "scholarly_metrics",
            "projects",
            "memberships",
            "activities",
            "raw_facts",
        ],
    },
    "education": {
        "entity_name": "Education",
        "manual_cols": [
            PRIMARY_ID,
            "dblp_pid_clean",
            "full_name",
            "degree",
            "field",
            "university",
            "location",
            "edu_year",
            "is_current",
            "has_concurrent_job",
        ],
        "text_fields": [
            "full_name",
            "degree",
            "field",
            "university",
            "location",
        ],
        "num_fields": ["edu_year"],
        "bool_fields": ["is_current", "has_concurrent_job"],
        "rename_fields": [
            "full_name",
            "degree",
            "field",
            "university",
            "location",
            "edu_year",
            "is_current",
            "has_concurrent_job",
        ],
    },
    "positions": {
        "entity_name": "Positions",
        "manual_cols": [
            PRIMARY_ID,
            "dblp_pid_clean",
            "full_name",
            "role",
            "department",
            "institution",
            "location",
            "start_year",
            "end_year",
            "is_current",
            "experience_years",
            "description",
        ],
        "text_fields": [
            "full_name",
            "role",
            "department",
            "institution",
            "location",
            "description",
        ],
        "num_fields": ["start_year", "end_year", "experience_years"],
        "bool_fields": ["is_current"],
        "rename_fields": [
            "full_name",
            "role",
            "department",
            "institution",
            "location",
            "start_year",
            "end_year",
            "is_current",
            "experience_years",
            "description",
        ],
    },
}

# =========================================================
# Helper maps for output file routing
# =========================================================
ENTITY_OUTPUTS = {
    "authors": {
        "llm_output": OUT_LLM_AUTHORS,
    },
    "education": {
        "llm_output": OUT_LLM_EDU,
    },
    "positions": {
        "llm_output": OUT_LLM_POS,
    },
}