#!/usr/bin/env python3
"""
eval/make_manifest.py
Computes SHA-256 hashes and byte counts for all core evaluation and data files
using hashlib, writes eval/MANIFEST.json, and prints the result.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TRACKED_FILES = [
    "data/invoices.csv",
    "data/policies/expense_policy.pdf",
    "data/policies/finance_compliance_policy.pdf",
    "data/policies/invoice_terms_policy.pdf",
    "data/policies/payment_policy.pdf",
    "data/policies/vendor_policy.pdf",
    "eval/judge_holdout.json",
    "eval/planner_stress.json",
    "eval/questions.json",
    "eval/questions_heldout.json",
]

def hash_file(path: Path):
    h = hashlib.sha256()
    content = path.read_bytes()
    h.update(content)
    return h.hexdigest(), len(content)

def compute_combined_dataset_hash():
    h = hashlib.sha256()
    csv_p = ROOT / "data" / "invoices.csv"
    if csv_p.exists():
        h.update(csv_p.read_bytes())
    pol_dir = ROOT / "data" / "policies"
    for fname in sorted(pol_dir.glob("*.pdf")):
        h.update(fname.read_bytes())
    return h.hexdigest()

def main():
    manifest = {}
    for rel_path in TRACKED_FILES:
        full_path = ROOT / rel_path
        if not full_path.exists():
            raise FileNotFoundError(f"Missing file: {full_path}")
        sha, size = hash_file(full_path)
        manifest[rel_path] = {
            "sha256": sha,
            "bytes": size
        }

    manifest["_combined_hash"] = compute_combined_dataset_hash()

    out_file = ROOT / "eval" / "MANIFEST.json"
    out_json = json.dumps(manifest, indent=2)
    out_file.write_text(out_json + "\n", encoding="utf-8")
    print(out_json)

if __name__ == "__main__":
    main()
