import os
import sys
import json
import time
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.run_eval import run_ask, score, load_df, QUESTIONS
from app.llm.generate import get_max_prompt_tokens

CONFIGS = {
    "Config A (nomic + hybrid + separate)": {
        "tag": "config_a",
        "extra": ["--embed-model", "nomic-embed-text", "--retrieval", "hybrid", "--gate", "separate", "--planner", "auto"]
    },
    "Config C (bge-m3 + vector + merged)": {
        "tag": "config_c",
        "extra": ["--embed-model", "bge-m3", "--retrieval", "vector", "--gate", "merged", "--planner", "auto"]
    },
    "Config D (bge-m3 + vector + separate)": {
        "tag": "config_d",
        "extra": ["--embed-model", "bge-m3", "--retrieval", "vector", "--gate", "separate", "--planner", "auto"]
    },
    "Config E (bge-m3 + hybrid + merged)": {
        "tag": "config_e",
        "extra": ["--embed-model", "bge-m3", "--retrieval", "hybrid", "--gate", "merged", "--planner", "auto"]
    }
}

DATASETS = {
    "dev": {
        "path": Path("eval/questions.json"),
        "hash": "f6fbffbb1dfb",
        "count": 40
    },
    "heldout": {
        "path": Path("eval/questions_heldout.json"),
        "hash": "5181cf7d0fdd",
        "count": 22
    }
}

def run_suite():
    df = load_df()
    matrix_results = {}
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs("eval/results", exist_ok=True)
    os.makedirs("eval/diagnostics", exist_ok=True)

    print("=" * 75)
    print("STARTING FAIR RE-RUN ACROSS CONFIGS A, C, D, E (DEV & HELDOUT)")
    print("=" * 75)

    for cfg_name, cfg in CONFIGS.items():
        print(f"\n>>> Running {cfg_name}...")
        cfg_summary = {"config_name": cfg_name, "extra": cfg["extra"], "runs": {}}
        
        for ds_name, ds_info in DATASETS.items():
            run_id = f"{timestamp}_{cfg['tag']}_{ds_name}"
            qs = json.loads(ds_info["path"].read_text(encoding="utf-8"))
            
            passed_count = 0
            failures = []
            run_rows = []
            t0 = time.time()
            
            for i, q in enumerate(qs, 1):
                res = run_ask(q["q"], cfg["extra"], timeout=180)
                sc = score(q, res, df)
                status_str = "PASS" if sc["passed"] else f"FAIL ({sc.get('fails', [])})"
                print(f"      [{i:02d}/{len(qs)}] {q['id']}: {status_str}", flush=True)
                if sc["passed"]:
                    passed_count += 1
                else:
                    failures.append({
                        "id": q["id"],
                        "question": q["q"],
                        "fails": sc.get("fails", []),
                        "answer": res.get("answer"),
                        "tools": res.get("tools")
                    })
                run_rows.append({
                    "id": q["id"],
                    "passed": sc["passed"],
                    "score": sc,
                    "res": res
                })

            dur = round(time.time() - t0, 1)
            print(f"    [{ds_name.upper()}] Passed: {passed_count}/{len(qs)} in {dur}s")
            
            # Save raw result file
            res_file = Path(f"eval/results/{run_id}.json")
            with open(res_file, "w") as f:
                json.dump({
                    "run_id": run_id,
                    "config": cfg_name,
                    "dataset": ds_name,
                    "dataset_hash": ds_info["hash"],
                    "total": len(qs),
                    "passed": passed_count,
                    "duration_s": dur,
                    "failures": failures,
                    "rows": run_rows
                }, f, indent=2)

            cfg_summary["runs"][ds_name] = {
                "run_id": run_id,
                "dataset_hash": ds_info["hash"],
                "total": len(qs),
                "passed": passed_count,
                "duration_s": dur,
                "failures": failures,
                "result_file": str(res_file)
            }

        matrix_results[cfg_name] = cfg_summary

    max_prompt_tokens = get_max_prompt_tokens()

    # Save overall matrix summary
    summary_file = Path("eval/results/fair_rerun_matrix.json")
    with open(summary_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "max_prompt_tokens_seen": max_prompt_tokens,
            "matrix": matrix_results
        }, f, indent=2)

    # Format human-readable table
    lines = []
    lines.append("=" * 95)
    lines.append("FAIR RE-RUN COMPARISON MATRIX (PLANNER: AUTO, JUDGE: OFF)")
    lines.append("=" * 95)
    header = f"{'Configuration':<38} | {'Dev (40)':<10} | {'Held-out (22)':<13} | {'Total (62)':<10} | {'Dev Run ID'}"
    lines.append(header)
    lines.append("-" * 95)

    for cfg_name, data in matrix_results.items():
        dev_p = data["runs"]["dev"]["passed"]
        dev_t = data["runs"]["dev"]["total"]
        ho_p = data["runs"]["heldout"]["passed"]
        ho_t = data["runs"]["heldout"]["total"]
        tot_p = dev_p + ho_p
        tot_t = dev_t + ho_t
        dev_run = data["runs"]["dev"]["run_id"]
        lines.append(f"{cfg_name:<38} | {dev_p:>2}/{dev_t:<2} ({dev_p/dev_t*100:4.1f}%) | {ho_p:>2}/{ho_t:<2} ({ho_p/ho_t*100:4.1f}%)   | {tot_p:>2}/{tot_t:<2} ({tot_p/tot_t*100:4.1f}%) | {dev_run}")

    lines.append("=" * 95)
    lines.append(f"Maximum Prompt Size Recorded Across 62 Questions: ~{max_prompt_tokens} tokens (Limit: 8192)")
    lines.append("")
    lines.append("FAILURE ANALYSIS & RAW EVIDENCE:")
    for cfg_name, data in matrix_results.items():
        dev_fails = data["runs"]["dev"]["failures"]
        ho_fails = data["runs"]["heldout"]["failures"]
        all_fails = dev_fails + ho_fails
        lines.append(f"\n--- {cfg_name} (Total Failures: {len(all_fails)}) ---")
        if not all_fails:
            lines.append("  None (100% pass rate)")
        else:
            for f_item in all_fails:
                lines.append(f"  • [{f_item['id']}] Q: \"{f_item['question']}\"")
                lines.append(f"    Fails: {f_item['fails']}")
                lines.append(f"    Raw Answer: \"{f_item['answer']}\"")

    output_str = "\n".join(lines) + "\n"
    print("\n" + output_str)

    with open("eval/diagnostics/fair_rerun_matrix.txt", "w") as f:
        f.write(output_str)

if __name__ == "__main__":
    run_suite()
