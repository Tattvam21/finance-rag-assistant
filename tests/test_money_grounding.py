import pytest
from app.tools.late_fee import calculate_late_fee, format_late_fee_template
from app.llm.generate import check_numeric_grounding

def test_late_fee_template_output():
    sample_fee_data = {
        "found": True,
        "invoice_id": "INV-2024-036",
        "principal": 849465.44,
        "principal_formatted": "₹8,49,465.44",
        "days_overdue": 966,
        "fee": 547055.74,
        "fee_formatted": "₹5,47,055.74",
        "total_due": 1396521.18,
        "total_due_formatted": "₹13,96,521.18",
        "conditions": "Interest charges will only apply if the delay is solely attributable to internal processing and not vendor billing errors."
    }
    rendered = format_late_fee_template(sample_fee_data)
    assert "Late fee on INV-2024-036: ₹5,47,055.74" in rendered
    assert "principal ₹8,49,465.44 × 2.0%/30 × 966 days" in rendered
    assert "Total due ₹13,96,521.18" in rendered
    assert "Condition: Interest charges will only apply if the delay is solely attributable" in rendered
    assert "The policy defines no cap or grace period." in rendered

def test_numeric_grounding_success():
    question = "How much late fee is owed on INV-2024-036?"
    tool_results = [{
        "invoice_id": "INV-2024-036",
        "fee": 547055.74,
        "fee_formatted": "₹5,47,055.74",
        "principal": 849465.44,
        "total_due": 1396521.18,
        "days_overdue": 966
    }]
    chunks = [{"content": "Net 30 payment terms apply."}]
    answer = "Late fee on INV-2024-036: ₹5,47,055.74 (principal ₹8,49,465.44 × 2.0%/30 × 966 days). Total due ₹13,96,521.18."
    ok, offending = check_numeric_grounding(answer, tool_results, chunks, question)
    assert ok is True
    assert offending == []

def test_numeric_grounding_detects_hallucination():
    question = "How much late fee is owed on INV-2024-036?"
    tool_results = [{
        "fee": 547055.74,
        "principal": 849465.44
    }]
    chunks = []
    # Hallucinated number 54,705.74 instead of 5,47,055.74
    bad_answer = "The late fee owed on invoice INV-2024-036 is ₹54,705.74."
    ok, offending = check_numeric_grounding(bad_answer, tool_results, chunks, question)
    assert ok is False
    assert "54705.74" in offending
