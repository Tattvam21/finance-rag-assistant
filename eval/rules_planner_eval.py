import json
from eval.test_planner_rules import all_q, rule_planner

# Load LLM plans from comparison run A
dev_res = json.load(open("eval/results/20261016_113122_comp_Config_A_dev.json"))
held_res = json.load(open("eval/results/20261016_113709_comp_Config_A_heldout.json"))
res_rows = {r["id"]: r for r in dev_res["rows"] + held_res["rows"]}

rule_handled = 0
agreed = 0
disagreed = []
latencies_saved = []

for q in all_q:
    qid = q["id"]
    qtext = q["q"]
    r_plan, confident = rule_planner(qtext)
    
    r_row = res_rows.get(qid, {})
    llm_plan_raw = r_row.get("plan")
    llm_dur = r_row.get("stage_timings", {}).get("planner", 0.0)
    
    if confident:
        rule_handled += 1
        latencies_saved.append(llm_dur)
        
        # Compare tools chosen by rule vs LLM executed tools
        r_tools = set()
        if r_plan.needs_policy: r_tools.add("search_policies")
        if r_plan.invoice: r_tools.add("query_invoices")
        if r_plan.late_fee_invoice_id: r_tools.add("calculate_late_fee")
        
        llm_tools = set(r_row.get("tools", []))
            
        if r_tools == llm_tools:
            agreed += 1
        else:
            disagreed.append({
                "id": qid,
                "q": qtext,
                "rule_tools": sorted(list(r_tools)),
                "llm_tools": sorted(list(llm_tools))
            })

print(f"Total Questions: {len(all_q)}")
print(f"Handled by rules: {rule_handled}/{len(all_q)} ({rule_handled/len(all_q)*100:.1f}%)")
print(f"Agreement on handled: {agreed}/{rule_handled} ({agreed/rule_handled*100:.1f}%)")
print(f"Total Disagreements: {len(disagreed)}")
for d in disagreed:
    print(f"  [{d['id']}] {d['q']}")
    print(f"      Rule tools: {d['rule_tools']} vs LLM tools: {d['llm_tools']}")

mean_saved = sum(latencies_saved) / len(latencies_saved) if latencies_saved else 0
print(f"\nAverage planner latency saved per rule-handled question: {mean_saved:.1f} ms")
print(f"Total latency saved across benchmark: {sum(latencies_saved)/1000:.2f} s")
