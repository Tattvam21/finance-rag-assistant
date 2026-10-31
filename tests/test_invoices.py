import os
import pytest
import pandas as pd
from datetime import date
from app.tools.invoice_queries import query_invoices
from app.tools.late_fee import calculate_late_fee

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "invoices.csv")

@pytest.fixture(scope="module")
def df_ground_truth():
    assert os.path.exists(CSV_PATH), f"CSV not found at {CSV_PATH}"
    df = pd.read_csv(CSV_PATH)
    df["amount_inr"] = df["amount_inr"].astype(float)
    return df

def test_unpaid_above_1_lakh(df_ground_truth):
    """
    Test 1: Unpaid (Pending or Overdue) with amount_inr >= 100,000.
    Compares count and total amount against pandas.
    """
    tool_res = query_invoices(status="unpaid", min_amount=100000.0, limit=200)
    
    unpaid_mask = df_ground_truth["status"].isin(["Pending", "Overdue"]) & (df_ground_truth["amount_inr"] >= 100000.0)
    expected_df = df_ground_truth[unpaid_mask]
    expected_count = len(expected_df)
    expected_total = round(expected_df["amount_inr"].sum(), 2)

    assert tool_res["count"] == expected_count, f"Count mismatch: got {tool_res['count']}, expected {expected_count}"
    assert round(tool_res["total_amount"], 2) == expected_total, f"Total mismatch: got {tool_res['total_amount']}, expected {expected_total}"
    assert "₹" in tool_res["total_formatted"]

def test_overdue_only(df_ground_truth):
    """
    Test 2: Status = 'Overdue' only.
    Compares count and total amount against pandas.
    """
    tool_res = query_invoices(status="Overdue", limit=200)
    
    expected_df = df_ground_truth[df_ground_truth["status"] == "Overdue"]
    expected_count = len(expected_df)
    expected_total = round(expected_df["amount_inr"].sum(), 2)

    assert tool_res["count"] == expected_count
    assert round(tool_res["total_amount"], 2) == expected_total
    for inv in tool_res["invoices"]:
        assert inv["status"].lower() == "overdue"

def test_single_vendor(df_ground_truth):
    """
    Test 3: Filter by single vendor (e.g., 'Blue Dart Express').
    Compares count and IDs against pandas.
    """
    vendor = "Blue Dart Express"
    tool_res = query_invoices(vendor=vendor, limit=200)
    
    expected_df = df_ground_truth[df_ground_truth["vendor_name"].str.contains(vendor, case=False)]
    expected_count = len(expected_df)
    expected_ids = set(expected_df["invoice_id"].tolist())

    assert tool_res["count"] == expected_count
    tool_ids = set(inv["invoice_id"] for inv in tool_res["invoices"])
    assert tool_ids == expected_ids

def test_missing_invoice_id(df_ground_truth):
    """
    Test 4: Lookup for a non-existent invoice ID (e.g., 'INV001').
    Must return found=False, error message, and suggestions of existing IDs.
    """
    missing_id = "INV001"
    tool_res = query_invoices(invoice_id=missing_id)

    assert tool_res["found"] is False
    assert tool_res["count"] == 0
    assert tool_res["invoices"] == []
    assert "suggested_ids" in tool_res
    assert len(tool_res["suggested_ids"]) > 0
    assert "not found" in tool_res["error"].lower()
    csv_ids = set(df_ground_truth["invoice_id"].tolist())
    for sug in tool_res["suggested_ids"]:
        assert sug in csv_ids

def test_unpaid_synonyms(df_ground_truth):
    """
    Test 5: Checks synonyms for unpaid: 'not paid', 'outstanding', 'pending or overdue'.
    """
    expected_count = len(df_ground_truth[df_ground_truth["status"].isin(["Pending", "Overdue"])])

    for syn in ["not paid", "outstanding", "pending or overdue"]:
        res = query_invoices(status=syn)
        assert res["count"] == expected_count, f"Synonym '{syn}' yielded {res['count']} instead of {expected_count}"

def test_late_fee_calculation():
    """
    Test 6: Late fee calculator tool correctness.
    - Paid invoice has 0 fee.
    - Overdue invoice computes daily pro-rata fee at 2.0% per month.
    """
    # Case A: Paid invoice INV-2024-001
    paid_res = calculate_late_fee("INV-2024-001")
    assert paid_res["found"] is True
    assert paid_res["status"].lower() == "paid"
    assert paid_res["fee"] == 0.0

    # Case B: Overdue invoice INV-2024-036
    # Principal: 849465.44, Due: 2024-02-17
    overdue_res = calculate_late_fee("INV-2024-036", as_of_date="2024-03-18")
    assert overdue_res["found"] is True
    assert overdue_res["status"].lower() == "overdue"
    assert overdue_res["days_overdue"] == 30
    # Expected fee for 30 days: principal * (0.02 / 30) * 30 = principal * 0.02 = 16989.31
    expected_fee = round(849465.44 * 0.02, 2)
    assert overdue_res["fee"] == expected_fee
    assert overdue_res["total_due"] == round(849465.44 + expected_fee, 2)
def test_invoice_due_date_and_status_derivation(df_ground_truth):
    """
    Test 7: Validates that every invoice's due_date strictly conforms to vendor policy
    (Net 45 for Vendor ABC Corp, Net 30 for all other vendors) and that status is strictly
    derived: 'Paid' if payment_date is set, else 'Overdue' if due_date < AS_OF_DATE, else 'Pending'.
    """
    from datetime import datetime, timedelta
    from app.config import AS_OF_DATE

    as_of_dt = datetime.strptime(AS_OF_DATE.strip(), "%Y-%m-%d").date()

    # Strict dataset boundary checks
    assert len(df_ground_truth) == 90, f"Expected 90 invoices, found {len(df_ground_truth)}"
    all_ids = set(df_ground_truth["invoice_id"].tolist())
    expected_ids = {f"INV-2024-{i:03d}" for i in range(1, 91)}
    assert all_ids == expected_ids, f"Invoice ID set mismatch: diff={all_ids ^ expected_ids}"
    assert "INV-2024-100" not in all_ids, "INV-2024-100 must remain non-existent"
    assert "INV-2024-999" not in all_ids, "INV-2024-999 must remain non-existent"

    # Coverage rows (081-090) checks
    cov_df = df_ground_truth[df_ground_truth["invoice_id"].isin([f"INV-2024-{i:03d}" for i in range(81, 91)])]
    assert len(cov_df) == 10
    assert (cov_df["status"] == "Pending").all(), "All coverage rows (81-90) must be Pending"
    assert cov_df["payment_date"].isna().all() or (cov_df["payment_date"] == "").all()

    due_dates = set(cov_df["due_date"].astype(str).tolist())
    assert "2024-08-05" in due_dates, "Must include invoice due exactly on AS_OF_DATE (2024-08-05)"
    assert "2024-08-06" in due_dates, "Must include invoice due exactly on 2024-08-06"
    assert (cov_df["amount_inr"] > 200000).any(), "Must include amounts above ₹2,00,000"
    assert (cov_df["amount_inr"] < 100000).any(), "Must include amounts below ₹1,00,000"

    for _, row in df_ground_truth.iterrows():
        inv_id = row["invoice_id"]
        vendor = row["vendor_name"]
        inv_date = datetime.strptime(str(row["invoice_date"]), "%Y-%m-%d").date()
        due_date = datetime.strptime(str(row["due_date"]), "%Y-%m-%d").date()
        pmt_str = str(row["payment_date"]).strip() if pd.notna(row["payment_date"]) else ""

        # Vendor policy term check
        expected_net_days = 45 if "vendor abc corp" in vendor.lower() else 30
        expected_due = inv_date + timedelta(days=expected_net_days)
        assert due_date == expected_due, (
            f"Invoice {inv_id} for vendor '{vendor}' has due date {due_date}, "
            f"expected {expected_due} (Net {expected_net_days})"
        )

        # Status derivation check
        if pmt_str:
            pmt_date = datetime.strptime(pmt_str, "%Y-%m-%d").date()
            assert row["status"] == "Paid", f"Invoice {inv_id} has payment_date {pmt_str} but status '{row['status']}'"
            assert pmt_date <= as_of_dt, f"Invoice {inv_id} payment_date {pmt_date} is after AS_OF_DATE {as_of_dt}"
            assert pmt_date >= inv_date, f"Invoice {inv_id} payment_date {pmt_date} is before invoice_date {inv_date}"
        else:
            expected_status = "Overdue" if due_date < as_of_dt else "Pending"
            assert row["status"] == expected_status, (
                f"Invoice {inv_id} (due {due_date}) has status '{row['status']}', expected '{expected_status}'"
            )

def test_invoice_id_canonical_and_suggestions():
    """
    Task 2 tests:
    - Canonical ID (INV-\\d{4}-\\d{3}) case-insensitive queries database normally.
    - Non-canonical forms (INV001, INV 036, invoice number 36, #36) do NOT query guessed ID.
    - Returns not-found with suggestion ('... was not found. Did you mean ...?').
    - Trailing punctuation handled.
    - Late-fee calculation with non-canonical ID also returns not-found + suggestion without answering.
    """
    # 1. Non-canonical INV001
    r_inv001 = query_invoices(invoice_id="INV001")
    assert r_inv001["found"] is False
    assert "INV001 was not found" in r_inv001["error"]
    assert "Did you mean INV-2024-001?" in r_inv001["error"]
    assert len(r_inv001["invoices"]) == 0

    # 2. Non-canonical INV 036
    r_inv036 = query_invoices(invoice_id="INV 036")
    assert r_inv036["found"] is False
    assert "INV 036 was not found" in r_inv036["error"]
    assert "Did you mean INV-2024-036?" in r_inv036["error"]

    # 3. Non-canonical invoice number 36
    r_num36 = query_invoices(invoice_id="invoice number 36")
    assert r_num36["found"] is False
    assert "invoice number 36 was not found" in r_num36["error"]
    assert "Did you mean INV-2024-036?" in r_num36["error"]

    # 4. Non-canonical #36
    r_hash36 = query_invoices(invoice_id="#36")
    assert r_hash36["found"] is False
    assert "#36 was not found" in r_hash36["error"]
    assert "Did you mean INV-2024-036?" in r_hash36["error"]

    # 5. Canonical lowercase inv-2024-036
    r_lower = query_invoices(invoice_id="inv-2024-036")
    assert r_lower["found"] is True
    assert r_lower["invoices"][0]["invoice_id"] == "INV-2024-036"

    # 6. Canonical with trailing punctuation (from query parsing)
    from app.llm.generate import parse_invoice_reference
    c_id, non_c, _ = parse_invoice_reference("What is the status of INV-2024-036?")
    assert c_id == "INV-2024-036"
    assert non_c is None

    # 7. Late fee non-canonical check
    r_lf = calculate_late_fee(invoice_id="INV 036")
    assert r_lf["found"] is False
    assert "INV 036 was not found" in r_lf["error"]
    assert "Did you mean INV-2024-036?" in r_lf["error"]

def test_no_phase_wording_in_invoice_tool():
    """
    Task 3: query_invoices / calculate_late_fee must return days_overdue only,
    without any hard-coded policy phase text.
    """
    r_overdue = query_invoices(invoice_id="INV-2024-048")
    assert r_overdue["found"] is True
    days_str = r_overdue["model_payload"]["days_overdue"]
    assert days_str is not None
    assert "days overdue" in days_str
    # Crucial assertion: no policy phase wording in invoice tool output!
    assert "Phase" not in days_str
    assert "Day 30+" not in days_str
    assert "escalation" not in days_str

if __name__ == "__main__":
    pytest.main(["-v", __file__])

