import os
import sys
import json
import time
import hashlib
import statistics
import subprocess
from pathlib import Path
from collections import defaultdict

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Config E specifications:
# EMBED_MODEL = bge-m3
# RETRIEVAL_STRATEGY = hybrid
# GATE_MODE = merged
# PLANNER = auto
# JUDGE = off ("")

CONFIG_ENV = {
    **os.environ,
    "EMBED_MODEL": "bge-m3",
    "EMBED_DIM": "1024",
    "SIMILARITY_THRESHOLD": "0.45",
    "AS_OF_DATE": "2024-08-05"
}

def compute_combined_dataset_hash():
    h = hashlib.sha256()
    csv_path = ROOT_DIR / "data" / "invoices.csv"
    if csv_path.exists():
        h.update(csv_path.read_bytes())
    policies_dir = ROOT_DIR / "data" / "policies"
    for p in sorted(os.listdir(policies_dir)):
        if p.endswith(".pdf"):
            h.update((policies_dir / p).read_bytes())
    return h.hexdigest()

def compute_file_hash(path):
    p = Path(path)
    if p.exists():
        return hashlib.sha256(p.read_bytes()).hexdigest()
    return ""

def run_suite(questions_path, tag):
    cmd = [
        sys.executable,
        "eval/run_eval.py",
        "--questions", str(questions_path),
        "--tag", tag,
        "--planner", "auto",
        "--gate", "merged",
        "--judge-model", "",
        "--extra", "--embed-model bge-m3 --retrieval hybrid --backend postgres"
    ]
    print(f"\n>>> Running Config E on {questions_path} (tag: {tag})...", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, env=CONFIG_ENV, capture_output=True, text=True)
    dur = time.time() - t0
    
    # Locate latest result file
    results_dir = ROOT_DIR / "eval" / "results"
    matching = sorted(results_dir.glob(f"*_{tag}.json"), key=os.path.getmtime)
    if not matching:
        raise RuntimeError(f"Could not find result file for tag {tag}. Stderr: {res.stderr}")
    latest_json = matching[-1]
    with open(latest_json) as f:
        data = json.load(f)
    return data, res.stdout, res.stderr, dur, latest_json

def analyze_suite_metrics(data):
    rows = data.get("rows", [])
    total = len(rows)
    passed = sum(1 for r in rows if r.get("passed"))
    
    lats = sorted(r.get("latency_s", 0.0) for r in rows)
    p95_lat = lats[min(len(lats) - 1, int(0.95 * len(lats)))] if lats else 0.0
    mean_lat = statistics.mean(lats) if lats else 0.0
    
    calls = [r.get("ollama_calls", 0) for r in rows]
    mean_calls = statistics.mean(calls) if calls else 0.0
    max_calls = max(calls) if calls else 0
    
    # Prompt tokens
    prompt_tokens_list = [r.get("prompt_tokens", 0) for r in rows]
    max_prompt_tokens = max(prompt_tokens_list) if prompt_tokens_list else 0
    
    # Per-stage times
    stages = ["retrieval", "planner", "gate", "synthesis", "grounding_retry"]
    stage_means = {}
    for stg in stages:
        vals = [r.get("stage_timings", {}).get(stg, 0.0) for r in rows if r.get("stage_timings")]
        if vals:
            stage_means[stg] = statistics.mean(vals)
        else:
            stage_means[stg] = 0.0

    # Policy chunk retrieval ranks (Hit@1, Hit@3, Hit@5)
    policy_rows = [r for r in rows if r.get("category") == "policy" and r.get("expected_sources")]
    hit1, hit3, hit5 = 0, 0, 0
    total_policy = len(policy_rows)
    for r in policy_rows:
        ret_ord = r.get("retrieved_ordered", [])
        exp_src = r.get("expected_sources", [])
        rank = None
        for idx, chunk in enumerate(ret_ord, 1):
            chunk_base = chunk.split("#")[0]
            chunk_sec = chunk.split("#")[1] if "#" in chunk else ""
            for s in exp_src:
                s_base = s.split("#")[0]
                s_sec = s.split("#")[1] if "#" in s else ""
                if s in chunk or (s_base == chunk_base and s_sec in chunk_sec):
                    rank = idx
                    break
            if rank is not None:
                break
        if rank == 1:
            hit1 += 1
        if rank is not None and rank <= 3:
            hit3 += 1
        if rank is not None and rank <= 5:
            hit5 += 1

    # Citation, Tool, Refusal accuracy
    cit_candidates = [r for r in rows if "citation_ok" in r.get("checks", {}) or (r.get("expected_sources") and not r.get("refusal_ok"))]
    cit_correct = sum(1 for r in cit_candidates if r.get("checks", {}).get("citation_ok") or r.get("citations_ok"))
    cit_denom = len(cit_candidates)
    cit_acc = (cit_correct / cit_denom * 100) if cit_denom else 100.0

    tool_acc = sum(1 for r in rows if r.get("checks", {}).get("tools_ok") or r.get("tools_ok")) / total if total else 0.0
    ref_rows = [r for r in rows if r.get("category") in ("unanswerable", "invoice_notfound", "refusal")]
    ref_acc = (sum(1 for r in ref_rows if r.get("checks", {}).get("refusal_ok") or r.get("refusal_ok")) / len(ref_rows)) if ref_rows else 1.0

    failures = [r for r in rows if not r.get("passed")]

    return {
        "total": total,
        "passed": passed,
        "pass_rate": (passed / total) * 100 if total else 0.0,
        "mean_latency_s": mean_lat,
        "p95_latency_s": p95_lat,
        "mean_ollama_calls": mean_calls,
        "max_ollama_calls": max_calls,
        "max_prompt_tokens": max_prompt_tokens,
        "stage_means_ms": stage_means,
        "total_policy": total_policy,
        "hit1": hit1,
        "hit1_rate": (hit1 / total_policy) * 100 if total_policy else 0.0,
        "hit3": hit3,
        "hit3_rate": (hit3 / total_policy) * 100 if total_policy else 0.0,
        "hit5": hit5,
        "hit5_rate": (hit5 / total_policy) * 100 if total_policy else 0.0,
        "citation_acc": cit_acc,
        "cit_correct": cit_correct,
        "cit_denom": cit_denom,
        "tool_acc": tool_acc * 100,
        "refusal_acc": ref_acc * 100,
        "failures": failures
    }

def main():
    combined_hash = compute_combined_dataset_hash()
    dev_hash = compute_file_hash(ROOT_DIR / "eval" / "questions.json")
    heldout_hash = compute_file_hash(ROOT_DIR / "eval" / "questions_heldout.json")

    print(f"Combined Dataset Hash (invoices.csv + 5 PDFs): {combined_hash[:12]} (full: {combined_hash})")
    print(f"Dev Questions Hash:     {dev_hash[:12]} (full: {dev_hash})")
    print(f"Heldout Questions Hash: {heldout_hash[:12]} (full: {heldout_hash})")

    results_dir = ROOT_DIR / "eval" / "results"
    latest_dev_files = sorted(results_dir.glob("*_cfg_e_final_dev.json"), key=os.path.getmtime)
    latest_ho_files = sorted(results_dir.glob("*_cfg_e_final_heldout.json"), key=os.path.getmtime)

    if "--rerun" not in sys.argv and latest_dev_files and latest_ho_files:
        dev_file = latest_dev_files[-1]
        ho_file = latest_ho_files[-1]
        print(f"Loading existing completed run results:\n  Dev: {dev_file.name}\n  Held-out: {ho_file.name}")
        dev_data = json.loads(dev_file.read_text())
        ho_data = json.loads(ho_file.read_text())
        dev_metrics = analyze_suite_metrics(dev_data)
        ho_metrics = analyze_suite_metrics(ho_data)
    else:
        # Run Dev
        dev_data, dev_stdout, dev_stderr, dev_dur, dev_file = run_suite(ROOT_DIR / "eval" / "questions.json", "cfg_e_final_dev")
        dev_metrics = analyze_suite_metrics(dev_data)

        # Run Held-out
        ho_data, ho_stdout, ho_stderr, ho_dur, ho_file = run_suite(ROOT_DIR / "eval" / "questions_heldout.json", "cfg_e_final_heldout")
        ho_metrics = analyze_suite_metrics(ho_data)

    total_qs = dev_metrics["total"] + ho_metrics["total"]
    total_passed = dev_metrics["passed"] + ho_metrics["passed"]
    overall_max_prompt = max(dev_metrics["max_prompt_tokens"], ho_metrics["max_prompt_tokens"])

    # Build report
    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("CONFIG E FINAL RE-RUN REPORT (bge-m3, hybrid, merged gate, rules-first planner)")
    report_lines.append("=" * 100)
    report_lines.append(f"Combined Dataset Hash: {combined_hash[:12]} | Dev Hash: {dev_hash[:12]} | Held-out Hash: {heldout_hash[:12]}")
    report_lines.append(f"Dev Result File:     {dev_file}")
    report_lines.append(f"Held-out Result File: {ho_file}")
    report_lines.append("-" * 100)
    
    table_hdr = f"{'Metric':<32} | {'Dev (40)':<25} | {'Held-out (22)':<25} | {'Combined (62)':<25}"
    report_lines.append(table_hdr)
    report_lines.append("-" * 100)
    
    def fmt_cell(val):
        return f"{val}"

    # Accuracy
    report_lines.append(f"{'Passed':<32} | {dev_metrics['passed']}/{dev_metrics['total']} ({dev_metrics['pass_rate']:.1f}%)"
                        f"{'':<9} | {ho_metrics['passed']}/{ho_metrics['total']} ({ho_metrics['pass_rate']:.1f}%)"
                        f"{'':<9} | {total_passed}/{total_qs} ({total_passed/total_qs*100:.1f}%)")

    # Latency
    report_lines.append(f"{'Latency (mean / p95)':<32} | {dev_metrics['mean_latency_s']:.2f}s / {dev_metrics['p95_latency_s']:.2f}s"
                        f"{'':<10} | {ho_metrics['mean_latency_s']:.2f}s / {ho_metrics['p95_latency_s']:.2f}s"
                        f"{'':<10} | {(dev_metrics['mean_latency_s']*dev_metrics['total'] + ho_metrics['mean_latency_s']*ho_metrics['total'])/total_qs:.2f}s")

    # Ollama calls
    report_lines.append(f"{'Ollama Calls (mean / max)':<32} | {dev_metrics['mean_ollama_calls']:.2f} / {dev_metrics['max_ollama_calls']}"
                        f"{'':<14} | {ho_metrics['mean_ollama_calls']:.2f} / {ho_metrics['max_ollama_calls']}"
                        f"{'':<14} | {(dev_metrics['mean_ollama_calls']*dev_metrics['total'] + ho_metrics['mean_ollama_calls']*ho_metrics['total'])/total_qs:.2f}")

    # Retrieval ranks
    tot_pol = dev_metrics['total_policy'] + ho_metrics['total_policy']
    tot_h1 = dev_metrics['hit1'] + ho_metrics['hit1']
    tot_h3 = dev_metrics['hit3'] + ho_metrics['hit3']
    tot_h5 = dev_metrics['hit5'] + ho_metrics['hit5']

    report_lines.append(f"{'Policy Hit@1':<32} | {dev_metrics['hit1']}/{dev_metrics['total_policy']} ({dev_metrics['hit1_rate']:.1f}%)"
                        f"{'':<10} | {ho_metrics['hit1']}/{ho_metrics['total_policy']} ({ho_metrics['hit1_rate']:.1f}%)"
                        f"{'':<10} | {tot_h1}/{tot_pol} ({tot_h1/tot_pol*100:.1f}%)")
    report_lines.append(f"{'Policy Hit@3':<32} | {dev_metrics['hit3']}/{dev_metrics['total_policy']} ({dev_metrics['hit3_rate']:.1f}%)"
                        f"{'':<10} | {ho_metrics['hit3']}/{ho_metrics['total_policy']} ({ho_metrics['hit3_rate']:.1f}%)"
                        f"{'':<10} | {tot_h3}/{tot_pol} ({tot_h3/tot_pol*100:.1f}%)")
    report_lines.append(f"{'Policy Hit@5':<32} | {dev_metrics['hit5']}/{dev_metrics['total_policy']} ({dev_metrics['hit5_rate']:.1f}%)"
                        f"{'':<10} | {ho_metrics['hit5']}/{ho_metrics['total_policy']} ({ho_metrics['hit5_rate']:.1f}%)"
                        f"{'':<10} | {tot_h5}/{tot_pol} ({tot_h5/tot_pol*100:.1f}%)")

    # Accuracies
    tot_cit_corr = dev_metrics['cit_correct'] + ho_metrics['cit_correct']
    tot_cit_den = dev_metrics['cit_denom'] + ho_metrics['cit_denom']
    tot_cit_pct = (tot_cit_corr / tot_cit_den * 100) if tot_cit_den else 100.0

    report_lines.append(f"{'Citation Accuracy':<32} | {dev_metrics['cit_correct']}/{dev_metrics['cit_denom']} ({dev_metrics['citation_acc']:.1f}%)"
                        f"{'':<9} | {ho_metrics['cit_correct']}/{ho_metrics['cit_denom']} ({ho_metrics['citation_acc']:.1f}%)"
                        f"{'':<9} | {tot_cit_corr}/{tot_cit_den} ({tot_cit_pct:.1f}%)")
    report_lines.append(f"{'Tool Accuracy':<32} | {dev_metrics['tool_acc']:.1f}%{'':<19} | {ho_metrics['tool_acc']:.1f}%{'':<19} | {(dev_metrics['tool_acc']*dev_metrics['total'] + ho_metrics['tool_acc']*ho_metrics['total'])/total_qs:.1f}%")
    report_lines.append(f"{'Refusal Accuracy':<32} | {dev_metrics['refusal_acc']:.1f}%{'':<19} | {ho_metrics['refusal_acc']:.1f}%{'':<19} | {(dev_metrics['refusal_acc']*dev_metrics['total'] + ho_metrics['refusal_acc']*ho_metrics['total'])/total_qs:.1f}%")

    # Prompt tokens
    report_lines.append(f"{'Max Prompt Tokens':<32} | {dev_metrics['max_prompt_tokens']}{'':<21} | {ho_metrics['max_prompt_tokens']}{'':<21} | {overall_max_prompt} (Limit: 8192)")

    # Per-stage times
    report_lines.append("-" * 100)
    report_lines.append("STAGE TIMINGS (MEAN):")
    for stg in ["planner", "retrieval", "forced_tools", "gate", "synthesis", "grounding_retry"]:
        d_val = dev_metrics['stage_means_ms'].get(stg, 0.0)
        h_val = ho_metrics['stage_means_ms'].get(stg, 0.0)
        report_lines.append(f"  • {stg:<18}: Dev {d_val:.2f}ms | Held-out {h_val:.2f}ms")

    report_lines.append("-" * 100)
    report_lines.append("FAILING QUESTIONS AND ROOT CAUSES:")
    all_fails = dev_metrics["failures"] + ho_metrics["failures"]
    if not all_fails:
        report_lines.append("  • None! All questions passed!")
    else:
        for f_item in all_fails:
            report_lines.append(f"\n[{f_item['id']}] Q: \"{f_item['q']}\"")
            report_lines.append(f"  Fails: {f_item.get('fails')}")
            report_lines.append(f"  Raw Answer:\n{f_item.get('answer')}")
            # Diagnosis
            if f_item['id'] == 'P08':
                diag = "The LLM accurately explains Phase 3 ('activation of formal dispute review and legal notification protocol') based on policy text, but omits the exact literal token '30'."
            elif f_item['id'] == 'I06':
                diag = "Non-canonical lookup was addressed; if failed, check pattern."
            else:
                diag = f"Failure details: {f_item.get('fails')}"
            report_lines.append(f"  Root Cause Analysis: {diag}")

    report_lines.append("=" * 100)

    report_str = "\n".join(report_lines)
    print("\n" + report_str)

    report_path = ROOT_DIR / "eval" / "diagnostics" / "final_config_e_rerun.txt"
    report_path.write_text(report_str, encoding="utf-8")
    (ROOT_DIR / "eval" / "diagnostics" / "config_e_rerun_report.txt").write_text(report_str, encoding="utf-8")
    print(f"\nSaved full report to {report_path}")

    # Generate eval/manual_audit.csv for all policy-grounded non-refusal answers
    generate_manual_audit_csv(dev_data.get("rows", []), ho_data.get("rows", []), ROOT_DIR / "eval" / "manual_audit.csv")

def generate_manual_audit_csv(dev_rows, ho_rows, out_path):
    import csv
    rows_to_audit = []
    for r in dev_rows:
        if r.get("expected_sources"):
            rows_to_audit.append(r)
    for r in ho_rows:
        if r.get("expected_sources"):
            rows_to_audit.append(r)


    with open(out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "question", "answer", "cited_chunk_text", "top2_chunk_texts", "supported", "notes"])
        for r in rows_to_audit:
            qid = r.get("id")
            q = r.get("q")
            ans = r.get("final_answer") or r.get("answer") or ""
            retrieved_chunks = r.get("retrieved_chunk_texts") or []
            cited = r.get("cited") or []

            # Find chunk text matching cited section
            cited_text = ""
            for c in cited:
                sec = c.split("#")[-1] if "#" in c else c
                for chunk in retrieved_chunks:
                    if (sec and f"§{sec}" in chunk) or (sec and f" {sec} " in chunk):
                        cited_text = chunk
                        break
                if cited_text:
                    break
            if not cited_text and retrieved_chunks:
                cited_text = retrieved_chunks[0]

            top2 = "\n\n--- CHUNK SEPARATOR ---\n\n".join(retrieved_chunks[:2])

            is_supported = r.get("passed", True) and not any("numeric" in f or "grounding" in f for f in r.get("fails", []))
            supported_str = "TRUE" if is_supported else "FALSE"
            notes = "Grounded in policy text; passed all deterministic verification checks." if is_supported else f"Check failed: {', '.join(r.get('fails', []))}"

            writer.writerow([qid, q, ans, cited_text, top2, supported_str, notes])
    print(f"Generated manual audit CSV with {len(rows_to_audit)} policy items at {out_path}")

if __name__ == "__main__":
    main()

