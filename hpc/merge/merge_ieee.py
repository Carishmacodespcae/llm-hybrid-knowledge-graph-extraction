#!/usr/bin/env python3
"""
Merge 4 chunk output files into a single final JSONL.
Run AFTER all 4 jobs complete.
"""
from pathlib import Path
import json

CHUNKS = [
    "ieee_prod_output_part_1_97732.jsonl",
    "ieee_prod_output_part_97733_195464.jsonl",
    "ieee_prod_output_part_195465_293196.jsonl",
    "ieee_prod_output_part_293197_390928.jsonl",
]
OUTPUT = "ieee_complete.jsonl"

total = 0
with open(OUTPUT, "w", encoding="utf-8") as out:
    for chunk in CHUNKS:
        p = Path(chunk)
        if not p.exists():
            print(f"⚠ Missing: {chunk}")
            continue
        count = 0
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    print(line, file=out)
                    count += 1
        total += count
        print(f"✅ {chunk}: {count:,} records")

print(f"\n✅ Final merged file: {OUTPUT}")
print(f"   Total records: {total:,}")