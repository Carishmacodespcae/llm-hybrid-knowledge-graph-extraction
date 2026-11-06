# LLM-Based Researcher Career Information Extraction

Schema-driven extraction of structured researcher career information from
academic biographies using a self-hosted large language model.

This repository contains implementation artifacts accompanying my master's
thesis at the University of Trier:

> Hybrid Knowledge Graph Construction from Structured Metadata and Biography Text  
> Carishmarahamathi Ahamed Khan  
> Master of Science, Computer Science — 2026

## Overview

Academic biographies contain valuable information about researchers'
education, professional experience, research interests, and achievements.
However, this information is usually written as free text, making it
difficult to query and analyze systematically.

This project investigates how a schema-driven, prompt-based large language
model pipeline can transform academic biographies into structured,
machine-readable career records without task-specific model fine-tuning.

The work focuses on the extraction and evaluation layer that can support
future scholarly knowledge graph enrichment. It does not implement a
complete knowledge graph or merge IEEE and ORCID researcher records.

## My Contribution

The thesis work included:

- Designing a common extraction structure for heterogeneous biographies.
- Developing and refining the extraction prompt.
- Running large-scale LLM inference using HPC resources.
- Organizing batch execution with SLURM job arrays.
- Processing IEEE and ORCID biographies separately.
- Merging extraction outputs from batch jobs.
- Evaluating author, education, and position information against manual
  annotations.
- Analyzing extraction errors and limitations.

## Extracted Information

The output organizes biography information into three main groups:

| Group | Examples |
|---|---|
| Author-level information | Name, personal details, research areas, awards, memberships, activities, and scholarly metrics |
| Education | Degree, field of study, university, location, year, and ongoing-study status |
| Professional positions | Role, department, institution, location, start/end years, and current-position status |

Researcher identifiers are retained by the surrounding pipeline for
traceability. The biography text provides the evidence for extraction.

## Workflow

```text
Biography records
        |
        v
Schema-driven prompting
        |
        v
LLM inference on HPC
        |
        v
Structured extraction outputs
        |
        v
Output processing and batch merging
        |
        v
Author, education, and position tables
        |
        v
Field-level evaluation against manual annotations
        |
        v
Metrics and diagnostic error analysis
```

The thesis workflow also includes preprocessing, output validation, and
tabular export. Coverage of these stages in the retained implementation
has not yet been fully verified.

## Technologies

| Area | Technologies |
|---|---|
| Programming | Python |
| Language model | LLaMA 3.1 SauerkrautLM 70B Instruct |
| Inference | vLLM |
| Computing | KISSKI/GWDG HPC environment |
| Job scheduling | SLURM job arrays |
| Data processing | pandas, NumPy |
| Data formats | JSONL, JSON, CSV |
| Evaluation review | XLSX spreadsheets |

The production model was used without task-specific fine-tuning for this
extraction task.

## Repository Structure

```text
.
├── doc/                 Extraction prompt and documentation
├── hpc/                 Inference, batch merging, and SLURM files
├── src/
│   ├── evaluation/      Author, education, position, and shared evaluation logic
│   ├── io/              Data-loading utilities
│   ├── utils/           Cleaning and metric utilities
│   ├── config.py        Workflow configuration
│   └── main.py          Main entry point
├── requirements.txt     Dependency list
└── README.md
```

### HPC implementation

The `hpc` directory contains source-specific inference scripts, merge
scripts, and SLURM submission files for IEEE and ORCID processing.

### Evaluation implementation

The `src` directory contains evaluation modules for author, education,
and position records, together with shared loading, cleaning, and
metric utilities.

## Thesis-Reported Dataset Scale

| Source | Extraction-ready biography records |
|---|---:|
| IEEE | 390,920 |
| ORCID | 25,759 |
| Total across both sources | 416,679 |

For IEEE, the longest available biography was selected for each
`dblp_pid` before the final length filtering.

IEEE and ORCID were processed separately. The combined total counts
biography records across sources, not deduplicated researchers across
both sources.

## Thesis-Reported Evaluation Results

Evaluation used a manually annotated gold standard of:

- 100 IEEE biographies.
- 50 ORCID biographies.

The evaluation scope was English-language biography text.

| Source | Output group | Precision | Recall | Micro-F1 |
|---|---|---:|---:|---:|
| IEEE | Author | 0.974 | 0.963 | 0.969 |
| IEEE | Education | 0.998 | 0.991 | 0.995 |
| IEEE | Position | 0.945 | 0.953 | 0.949 |
| ORCID | Author | 0.963 | 0.854 | 0.905 |
| ORCID | Education | 0.986 | 0.946 | 0.966 |
| ORCID | Position | 0.942 | 0.886 | 0.913 |

These figures are transcribed from the thesis. They are field-level
micro-averaged scores under the thesis matching and counting rules,
not whole-record accuracy.

They have not yet been reproduced by rerunning this public repository.

## Key Findings

- Education information achieved the strongest extraction results.
- Position extraction was more challenging because role, department,
  institution, location, and dates often appear together.
- Open-ended fields were harder to assign consistently.
- Location hallucination remained an issue despite explicit prompt rules.
- Schema guidance supported structured output, but did not eliminate
  the need for validation and error analysis.

## Data Availability

Source datasets, real biography samples, manual annotations, and
researcher-level input and output files are not included in this public
repository because they may contain sensitive or restricted information.

The repository shares the available source code, extraction prompt,
HPC scripts, and evaluation implementation.

Running the workflows requires suitable local data and configuration
of input/output paths.

## Execution and Reproducibility Status

This repository preserves available implementation artifacts from the
thesis project.

The public copy has not yet been verified end-to-end in a clean environment.
It should not currently be treated as a fully reproducible release of
the original production pipeline.

Before running the code:

1. Review the dependencies in `requirements.txt`.
2. Configure local input and output paths.
3. Provide input files in the formats expected by the scripts.
4. For inference, configure model access and a compatible GPU environment.
5. Adapt SLURM account, resource, and filesystem settings to your cluster.

Exact execution commands will be documented after verification against
the retained scripts.

## Limitations

- The evaluation gold standard is small relative to the processed collection.
- Evaluation covers English-language biographies.
- String-based comparison does not establish external entity identity.
- Institution and department boundaries can be ambiguous.
- Location hallucination and multi-entry merging remain challenges.
- Selecting the longest biography does not guarantee the most recent
  career information.
- Full knowledge graph integration is future work.

## Author

Carishmarahamathi Ahamed Khan  
M.Sc. Data Science, University of Trier

## Third-Party Materials

References to datasets, models, and software do not grant redistribution
rights to those materials.

Any code license added to this repository will apply only within its
stated scope and will not automatically cover third-party data or
model weights.
