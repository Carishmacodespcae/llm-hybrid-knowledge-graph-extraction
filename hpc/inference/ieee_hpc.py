#!/usr/bin/env python3
"""
Production extraction — 390k+ IEEE records
4-GPU tensor parallel + batch inference
Accepts --start and --end to process a specific line range
(enables parallel execution across multiple SLURM jobs)
"""
import json, time, re, os, argparse
from datetime import datetime
from pathlib import Path

os.environ["TMPDIR"] = "/mnt/vast-kisski/home/c.ahamedkhan/u25954/tmp"
os.environ["VLLM_WORKER_MULTIPROC_TMPDIR"] = "/mnt/vast-kisski/home/c.ahamedkhan/u25954/tmp"
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=1,    help="First line to process (1-indexed, inclusive)")
parser.add_argument("--end",   type=int, default=None, help="Last line to process (1-indexed, inclusive)")
args = parser.parse_args()

# ── config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "/mnt/vast-kisski/projects/kisski_metadata_graph_constr/models/sauerkrautlm-70b"
INPUT_FILE  = "ieee_complete.jsonl"
PROMPT_FILE = "prompt.txt"
MAX_TOKENS  = 2048
TEMPERATURE = 0.0
BATCH_SIZE  = 64
OUTPUT_FILE = f"/llm_bio_extraction/ieee_prod_output_part_{args.start}_{args.end}.jsonl"

# ── load vLLM ─────────────────────────────────────────────────────────────────
from vllm import LLM, SamplingParams

print(f"Loading model | chunk {args.start}–{args.end or 'end'} → {OUTPUT_FILE}")
llm = LLM(
    model=MODEL_PATH,
    dtype="bfloat16",
    max_model_len=8192,
    tensor_parallel_size=4,
    gpu_memory_utilization=0.92,
    max_num_batched_tokens=16384,
    disable_log_stats=True,
)
sampling_params = SamplingParams(
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
    min_tokens=10,
    stop=None,
)
print("Model loaded!")


def clean_extraction(raw: str) -> dict:
    if not raw or not raw.strip():
        return {"error": "parse_failed", "detail": "empty response from model", "raw": ""}
    cleaned = raw.strip()
    cleaned = re.sub(r'<<SYS>>.*?<</SYS>>', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'```json\s*', '', cleaned)
    cleaned = re.sub(r'\s*```', '', cleaned)
    cleaned = re.sub(r'\\n', ' ', cleaned)
    cleaned = cleaned.strip()
    if not cleaned.startswith('{'):
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)
    cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        if "Extra data" in str(e):
            try:
                return json.loads(cleaned[:e.pos])
            except Exception:
                pass
        try:
            return json.loads(cleaned.replace("'", '"'))
        except Exception:
            return {"error": "parse_failed", "detail": str(e), "raw": raw[:300]}


def build_prompt(system_prompt: str, bio: str) -> str:
    return (
        f"[INST] <<SYS>>\n"
        f"{system_prompt.strip()}\n"
        f"<</SYS>>\n\n"
        f'biography: "{bio}" [/INST]'
    )


def normalise_experience(entities: dict, warnings: list):
    exp_keys = [k for k in entities if "experience" in k.lower()]
    if not exp_keys:
        return
    raw_exp = None
    for k in exp_keys:
        raw_exp = entities.pop(k)
        warnings.append(f"NORMALISED: '{k}' → experience_in / year_of_experience")
    exp_in = None
    exp_year = None
    if isinstance(raw_exp, int):
        exp_year = raw_exp
    elif isinstance(raw_exp, str):
        m = re.search(r'(\d+)\s*\+?\s*years?', raw_exp, re.IGNORECASE)
        if m:
            exp_year = int(m.group(1))
        exp_in = re.sub(r'\d+\+?\s*years?\s*(of|in)?\s*', '', raw_exp, flags=re.IGNORECASE).strip(" ,.")
        if not exp_in:
            exp_in = None
    elif isinstance(raw_exp, dict):
        exp_year = raw_exp.get("years") or raw_exp.get("year_of_experience") or raw_exp.get("total_years")
        exp_in   = raw_exp.get("field") or raw_exp.get("domain") or raw_exp.get("experience_in") or raw_exp.get("area")
        if isinstance(exp_year, str):
            m = re.search(r'(\d+)', exp_year)
            exp_year = int(m.group(1)) if m else None
    elif isinstance(raw_exp, list) and raw_exp:
        first = raw_exp[0]
        if isinstance(first, str):
            m = re.search(r'(\d+)\s*\+?\s*years?', first, re.IGNORECASE)
            if m:
                exp_year = int(m.group(1))
            exp_in = re.sub(r'\d+\+?\s*years?\s*(of|in)?\s*', '', first, flags=re.IGNORECASE).strip(" ,.")
    entities["experience_in"]      = exp_in   if exp_in   else None
    entities["year_of_experience"] = exp_year if exp_year else None


def validate_and_fix(entities: dict) -> tuple:
    warnings = []
    normalise_experience(entities, warnings)
    for pos in entities.get("positions") or []:
        if pos.get("is_current") is None:
            pos["is_current"] = pos.get("end_year") is None
            warnings.append("FIXED: is_current null")
    for field in ("research_areas", "awards", "projects", "memberships", "activities", "raw_facts"):
        val = entities.get(field)
        if isinstance(val, str):
            entities[field] = [val]
            warnings.append(f"FIXED: {field} string→list")
    if not entities.get("full_name"):    warnings.append("WARN: missing full_name")
    if not entities.get("positions"):    warnings.append("WARN: no positions")
    if not entities.get("education"):    warnings.append("WARN: no education")
    return entities, warnings


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for _ in f)


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    if h > 0:   return f"{h}h {m}m {s}s"
    elif m > 0: return f"{m}m {s}s"
    else:       return f"{s}s"


def main():
    system_prompt = Path(PROMPT_FILE).read_text(encoding="utf-8")
    output_path   = Path(OUTPUT_FILE)

    # Resume: count how many lines already written in THIS chunk's output file
    already_done  = count_lines(output_path)
    # start_line is 1-indexed input line; skip already-done records in this chunk
    chunk_start   = args.start + already_done
    chunk_end     = args.end  # None = run to end of file

    run_start = time.time()
    print("=" * 60)
    print(f"START  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CHUNK  : lines {args.start}–{chunk_end or 'end'} | resuming at {chunk_start}")
    print(f"OUTPUT : {OUTPUT_FILE}")
    print(f"MODEL  : {MODEL_PATH}")
    print("=" * 60)

    processed = ok_count = warn_count = error_count = skip_count = 0

    with output_path.open("a", encoding="utf-8") as out:
        with open(INPUT_FILE, encoding="utf-8") as f:
            all_lines = f.readlines()

        total_input = len(all_lines)
        effective_end = min(chunk_end, total_input) if chunk_end else total_input
        chunk_total   = effective_end - args.start + 1
        print(f"This chunk covers {chunk_total:,} records ({args.start}–{effective_end})")

        i = chunk_start - 1  # 0-indexed
        while i < effective_end:
            batch_end_idx = min(i + BATCH_SIZE, effective_end)
            batch_records = []
            batch_meta    = []
            batch_prompts = []

            for j in range(i, batch_end_idx):
                record   = json.loads(all_lines[j].strip())
                bio      = (record.get("biographies") or [""])[0]
                doi      = record.get("doi",      f"L{j+1}")
                dblp_pid = record.get("dblp_pid", "N/A")
                line_num = j + 1

                if not bio.strip():
                    skip_count += 1
                    result = {
                        "doi": doi, "dblp_pid": dblp_pid,
                        "entities": {"error": "no_biography"},
                        "_meta": {
                            "line": line_num, "has_error": True,
                            "warnings": ["SKIP: empty biography"],
                            "needs_review": False,
                            "model": MODEL_PATH, "prompt_file": PROMPT_FILE,
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        },
                    }
                    print(json.dumps(result, ensure_ascii=False), file=out, flush=True)
                    print(f"[{line_num}] {doi} ... ⏭ skipped")
                else:
                    batch_records.append(record)
                    batch_meta.append({"doi": doi, "dblp_pid": dblp_pid, "line_num": line_num})
                    batch_prompts.append(build_prompt(system_prompt, bio))

            i = batch_end_idx

            if not batch_prompts:
                continue

            batch_start_t = time.time()
            print(f"\n→ Batch lines {batch_meta[0]['line_num']}–{batch_meta[-1]['line_num']} ({len(batch_prompts)} records)...", flush=True)

            try:
                outputs = llm.generate(batch_prompts, sampling_params)
            except Exception as e:
                for meta in batch_meta:
                    error_count += 1
                    result = {
                        "doi": meta["doi"], "dblp_pid": meta["dblp_pid"],
                        "entities": {"error": "batch_exception", "detail": str(e)},
                        "_meta": {
                            "line": meta["line_num"], "has_error": True,
                            "warnings": [f"BATCH_EXCEPTION: {e}"],
                            "needs_review": True,
                            "model": MODEL_PATH, "prompt_file": PROMPT_FILE,
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        },
                    }
                    print(json.dumps(result, ensure_ascii=False), file=out, flush=True)
                print(f"✗ BATCH EXCEPTION: {e}")
                continue

            batch_elapsed = time.time() - batch_start_t
            run_elapsed   = time.time() - run_start

            for k, output in enumerate(outputs):
                raw      = output.outputs[0].text
                meta     = batch_meta[k]
                entities        = clean_extraction(raw)
                entities, warns = validate_and_fix(entities)
                result = {
                    "doi": meta["doi"], "dblp_pid": meta["dblp_pid"],
                    "entities": entities,
                    "_meta": {
                        "line": meta["line_num"], "has_error": "error" in entities,
                        "warnings": warns, "needs_review": len(warns) > 0,
                        "model": MODEL_PATH, "prompt_file": PROMPT_FILE,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    },
                }
                print(json.dumps(result, ensure_ascii=False), file=out, flush=True)
                processed += 1
                if "error" in entities:   error_count += 1; status = "✗"
                elif warns:               warn_count  += 1; status = "⚠"
                else:                     ok_count    += 1; status = "✓"
                print(f"  {status} [{meta['line_num']}] {meta['doi']} — {entities.get('full_name','NO_NAME')}" +
                      (f" [{len(warns)} fixes]" if warns else ""))

            avg_per_rec   = batch_elapsed / len(batch_prompts)
            chunk_done    = processed + skip_count + already_done
            chunk_remain  = chunk_total - chunk_done
            eta_sec       = chunk_remain * avg_per_rec
            print(f"  ↳ {batch_elapsed:.1f}s ({avg_per_rec:.1f}s/rec) | total: {format_duration(run_elapsed)} | "
                  f"ETA: {format_duration(eta_sec)} | chunk done: {chunk_done:,}/{chunk_total:,}")

    print(f"\n✅ Chunk done → {OUTPUT_FILE}")
    print(f"   OK: {ok_count} | Warns: {warn_count} | Errors: {error_count} | Skipped: {skip_count}")


if __name__ == "__main__":
    main()
