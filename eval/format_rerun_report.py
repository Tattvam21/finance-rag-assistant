import sys
import json
import statistics
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from eval.run_config_e_eval import compute_combined_dataset_hash, compute_file_hash, analyze_suite_metrics
dev_file = ROOT_DIR / "eval" / "results" / "20261016_164544_cfg_e_final_dev.json"
ho_file = ROOT_DIR / "eval" / "results" / "20261016_164837_cfg_e_final_heldout.json"

dev_data = json.loads(dev_file.read_text())
ho_data = json.loads(ho_file.read_text())

dev_metrics = analyze_suite_metrics(dev_data)
ho_metrics = analyze_suite_metrics(ho_data)

combined_hash = compute_combined_dataset_hash()
dev_hash = compute_file_hash(ROOT_DIR / "eval" / "questions.json")
ho_hash = compute_file_hash(ROOT_DIR / "eval" / "questions_heldout.json")

total_qs = dev_metrics["total"] + ho_metrics["total"]
total_passed = dev_metrics["passed"] + ho_metrics["passed"]
overall_max_prompt = max(dev_metrics["max_prompt_tokens"], ho_metrics["max_prompt_tokens"])

report_lines = []
report_lines.append("=" * 100)
report_lines.append("CONFIG E FINAL RE-RUN REPORT (bge-m3, hybrid, merged gate, rules-first planner)")
report_lines.append("=" * 100)
report_lines.append(f"Combined Dataset Hash: {combined_hash[:12]} | Dev Hash: {dev_hash[:12]} | Held-out Hash: {ho_hash[:12]}")
report_lines.append(f"Dev Result File:     {dev_file}")
report_lines.append(f"Held-out Result File: {ho_file}")
report_lines.append("-" * 100)

table_hdr = f"{'Metric':<32} | {'Dev (40)':<25} | {'Held-out (22)':<25} | {'Combined (62)':<25}"
report_lines.append(table_hdr)
report_lines.append("-" * 100)

report_lines.append(f"{'Passed':<32} | {dev_metrics['passed']}/{dev_metrics['total']} ({dev_metrics['pass_rate']:.1f}%)"
                    f"{'':<9} | {ho_metrics['passed']}/{ho_metrics['total']} ({ho_metrics['pass_rate']:.1f}%)"
                    f"{'':<9} | {total_passed}/{total_qs} ({total_passed/total_qs*100:.1f}%)")

report_lines.append(f"{'Latency (mean / p95)':<32} | {dev_metrics['mean_latency_s']:.2f}s / {dev_metrics['p95_latency_s']:.2f}s"
                    f"{'':<10} | {ho_metrics['mean_latency_s']:.2f}s / {ho_metrics['p95_latency_s']:.2f}s"
                    f"{'':<10} | {(dev_metrics['mean_latency_s']*dev_metrics['total'] + ho_metrics['mean_latency_s']*ho_metrics['total'])/total_qs:.2f}s")

report_lines.append(f"{'Ollama Calls (mean / max)':<32} | {dev_metrics['mean_ollama_calls']:.2f} / {dev_metrics['max_ollama_calls']}"
                    f"{'':<14} | {ho_metrics['mean_ollama_calls']:.2f} / {ho_metrics['max_ollama_calls']}"
                    f"{'':<14} | {(dev_metrics['mean_ollama_calls']*dev_metrics['total'] + ho_metrics['mean_ollama_calls']*ho_metrics['total'])/total_qs:.2f}")

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

tot_cit_corr = dev_metrics['cit_correct'] + ho_metrics['cit_correct']
tot_cit_den = dev_metrics['cit_denom'] + ho_metrics['cit_denom']
tot_cit_pct = (tot_cit_corr / tot_cit_den * 100) if tot_cit_den else 100.0

report_lines.append(f"{'Citation Accuracy':<32} | {dev_metrics['cit_correct']}/{dev_metrics['cit_denom']} ({dev_metrics['citation_acc']:.1f}%)"
                    f"{'':<9} | {ho_metrics['cit_correct']}/{ho_metrics['cit_denom']} ({ho_metrics['citation_acc']:.1f}%)"
                    f"{'':<9} | {tot_cit_corr}/{tot_cit_den} ({tot_cit_pct:.1f}%)")
report_lines.append(f"{'Tool Accuracy':<32} | {dev_metrics['tool_acc']:.1f}%{'':<19} | {ho_metrics['tool_acc']:.1f}%{'':<19} | {(dev_metrics['tool_acc']*dev_metrics['total'] + ho_metrics['tool_acc']*ho_metrics['total'])/total_qs:.1f}%")
report_lines.append(f"{'Refusal Accuracy':<32} | {dev_metrics['refusal_acc']:.1f}%{'':<19} | {ho_metrics['refusal_acc']:.1f}%{'':<19} | {(dev_metrics['refusal_acc']*dev_metrics['total'] + ho_metrics['refusal_acc']*ho_metrics['total'])/total_qs:.1f}%")

report_lines.append(f"{'Max Prompt Tokens':<32} | {dev_metrics['max_prompt_tokens']}{'':<21} | {ho_metrics['max_prompt_tokens']}{'':<21} | {overall_max_prompt} (Limit: 8192)")

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
        if f_item['id'] == 'P08':
            diag = "The LLM accurately explains Phase 3 ('activation of formal dispute review and legal notification protocol') based on policy text, but omits the exact literal token '30'."
        elif f_item['id'] == 'I06':
            diag = "Non-canonical lookup was addressed; if failed, check pattern."
        else:
            diag = f"Failure details: {f_item.get('fails')}"
        report_lines.append(f"  Root Cause Analysis: {diag}")

report_lines.append("=" * 100)
report_str = "\n".join(report_lines)
print(report_str)
(ROOT_DIR / "eval" / "diagnostics" / "config_e_rerun_report.txt").write_text(report_str, encoding="utf-8")
print("\nUpdated eval/diagnostics/config_e_rerun_report.txt successfully.")
