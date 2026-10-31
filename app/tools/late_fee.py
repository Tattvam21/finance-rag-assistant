from datetime import datetime, date
from typing import Dict, Any, Optional
from app.tools.invoice_queries import get_invoice_by_id
from app.utils.formatters import format_inr
from app.config import LATE_FEE_MONTHLY_RATE, LATE_FEE_GRACE_PERIOD_DAYS, AS_OF_DATE

def calculate_late_fee(
    invoice_id: str,
    as_of_date: Optional[str] = None,
    backend: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculates late interest fees for an invoice according to corporate policy:
    - Monthly rate configured in app.config (LATE_FEE_MONTHLY_RATE).
    - Interest calculated strictly on days overdue past due_date.
    - If Paid or not yet overdue, fee is 0 with explanation.
    """
    clean_id = invoice_id.strip()
    import re
    from app.tools.invoice_queries import is_canonical_invoice_id
    if not is_canonical_invoice_id(clean_id):
        digits = re.findall(r'\d+', clean_id)
        sug_id = f"INV-2024-{int(digits[-1]):03d}" if digits else "INV-2024-001"
        return {
            "found": False,
            "invoice_id": clean_id,
            "error": f"{clean_id} was not found. Did you mean {sug_id}?"
        }

    inv = get_invoice_by_id(clean_id, backend=backend)

    if not inv:
        return {
            "found": False,
            "invoice_id": clean_id,
            "error": f"Invoice '{clean_id}' was not found in records."
        }

    status = inv["status"]
    principal = float(inv["amount_inr"])
    due_date_str = str(inv["due_date"])
    vendor = inv["vendor_name"]
    payment_date_str = str(inv["payment_date"]) if inv.get("payment_date") else None

    # Parse target calculation date (defaults to configured AS_OF_DATE)
    default_calc_date = datetime.strptime(AS_OF_DATE.strip(), "%Y-%m-%d").date()
    if as_of_date:
        try:
            calc_date = datetime.strptime(as_of_date.strip(), "%Y-%m-%d").date()
        except ValueError:
            calc_date = default_calc_date
    else:
        calc_date = default_calc_date

    due_date = datetime.strptime(due_date_str, "%Y-%m-%d").date()

    policy_condition = "Interest charges will only apply if the delay is solely attributable to internal processing and not vendor billing errors."

    # Case 1: Already paid
    if status.lower() == "paid":
        paid_desc = f" on {payment_date_str}" if payment_date_str else ""
        return {
            "found": True,
            "invoice_id": clean_id,
            "vendor": vendor,
            "status": "Paid",
            "principal": principal,
            "principal_formatted": format_inr(principal),
            "due_date": due_date_str,
            "days_overdue": 0,
            "monthly_rate_pct": LATE_FEE_MONTHLY_RATE * 100,
            "fee": 0.0,
            "fee_formatted": "₹0.00",
            "total_due": 0.0,
            "total_due_formatted": "₹0.00",
            "formula": "Invoice is already settled; late fee is ₹0.00.",
            "reason": f"Invoice was marked as Paid{paid_desc}.",
            "conditions": policy_condition,
            "no_cap_or_grace_period_defined_in_policy": True
        }

    # Case 2: Not yet due
    if calc_date <= due_date:
        return {
            "found": True,
            "invoice_id": clean_id,
            "vendor": vendor,
            "status": status,
            "principal": principal,
            "principal_formatted": format_inr(principal),
            "due_date": due_date_str,
            "as_of_date": str(calc_date),
            "days_overdue": 0,
            "monthly_rate_pct": LATE_FEE_MONTHLY_RATE * 100,
            "fee": 0.0,
            "fee_formatted": "₹0.00",
            "total_due": principal,
            "total_due_formatted": format_inr(principal),
            "formula": "Invoice is not yet overdue.",
            "reason": f"Due date is {due_date_str}; as of {calc_date}, 0 days overdue.",
            "conditions": policy_condition,
            "no_cap_or_grace_period_defined_in_policy": True
        }

    # Case 3: Overdue
    days_overdue = (calc_date - due_date).days
    effective_days = max(0, days_overdue - LATE_FEE_GRACE_PERIOD_DAYS)

    # Pro-rata daily calculation: (monthly_rate / 30 days) * effective_days * principal
    daily_rate = LATE_FEE_MONTHLY_RATE / 30.0
    fee = round(principal * daily_rate * effective_days, 2)
    total_due = round(principal + fee, 2)

    formula = (
        f"principal ({format_inr(principal)}) × (monthly_rate {LATE_FEE_MONTHLY_RATE*100}% / 30 days) "
        f"× {effective_days} days overdue = {format_inr(fee)}"
    )

    return {
        "found": True,
        "invoice_id": clean_id,
        "vendor": vendor,
        "status": status,
        "principal": principal,
        "principal_formatted": format_inr(principal),
        "due_date": due_date_str,
        "as_of_date": str(calc_date),
        "days_overdue": days_overdue,
        "effective_days_overdue": effective_days,
        "monthly_rate_pct": LATE_FEE_MONTHLY_RATE * 100,
        "annual_rate_pct": LATE_FEE_MONTHLY_RATE * 12 * 100,
        "fee": fee,
        "fee_formatted": format_inr(fee),
        "total_due": total_due,
        "total_due_formatted": format_inr(total_due),
        "formula": formula,
        "reason": f"Invoice is {days_overdue} days past contractual due date {due_date_str}.",
        "conditions": policy_condition,
        "no_cap_or_grace_period_defined_in_policy": True
    }


def format_late_fee_template(fee_data: Dict[str, Any]) -> str:
    """
    Renders calculate_late_fee answers deterministically using the required Python template:
    Late fee on <id>: ₹<fee> (principal ₹<p> × <rate>/30 × <days> days). Total due ₹<total>.
    Condition: <policy condition>. The policy defines no cap or grace period.
    """
    inv_id = fee_data.get("invoice_id", "")
    fee = fee_data.get("fee_formatted", "").replace("₹", "").strip()
    p = fee_data.get("principal_formatted", "").replace("₹", "").strip()
    days = fee_data.get("days_overdue", 0)
    total = fee_data.get("total_due_formatted", "").replace("₹", "").strip()
    cond = fee_data.get("conditions", "Interest charges will only apply if the delay is solely attributable to internal processing and not vendor billing errors.")
    rate_pct = f"{LATE_FEE_MONTHLY_RATE * 100:.1f}%"
    return (
        f"Late fee on {inv_id}: ₹{fee} (principal ₹{p} × {rate_pct}/30 × {days} days). Total due ₹{total}.\n"
        f"Condition: {cond} The policy defines no cap or grace period."
    )
