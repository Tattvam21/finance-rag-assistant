import os
import re
from typing import List, Dict, Any, Optional
from app.db import get_connection
from app.utils.formatters import format_inr, format_inr_lakh_crore
from app.config import AS_OF_DATE

# ---------------------------------------------------------------------------
# Distinct Statuses in Ground Truth Data:
#   ['Paid', 'Pending', 'Overdue']
# Explicit Policy Status Mapping:
#   - 'unpaid', 'not paid', 'outstanding' -> ('Pending', 'Overdue')
#   - 'paid', 'cleared', 'settled'        -> ('Paid',)
#   - 'overdue'                           -> ('Overdue',)
#   - 'pending'                           -> ('Pending',)
#   - 'disputed'                          -> ('Disputed',)
#   - 'partially paid', 'partial'         -> ('Partially Paid',)
# ---------------------------------------------------------------------------

CANONICAL_INVOICE_ID_RE = re.compile(r"^INV-\d{4}-\d{3}$", re.IGNORECASE)

def is_canonical_invoice_id(inv_id: Optional[str]) -> bool:
    """Checks whether an invoice ID string is in canonical form (e.g. 'INV-2024-001', case-insensitive)."""
    if not inv_id:
        return False
    return bool(CANONICAL_INVOICE_ID_RE.match(str(inv_id).strip()))

def query_invoices(
    status: Optional[Any] = None,
    min_amount: Optional[float] = None,
    max_amount: Optional[float] = None,
    vendor: Optional[str] = None,
    due_before: Optional[str] = None,
    invoice_id: Optional[str] = None,
    department: Optional[str] = None,
    limit: int = 100,
    backend: Optional[str] = None
) -> Dict[str, Any]:
    """
    Safely queries invoices using parameterized SQL.
    Count and total amounts are aggregated directly via SQL.
    When > 5 invoices match, prepares a condensed model payload containing:
    count, total, top 3, and status breakdown, advising the LLM to summarize
    concisely instead of enumerating all rows.
    """
    conn, engine = get_connection(backend=backend)
    cur = conn.cursor()
    placeholder = "%s" if engine == "postgres" else "?"

    # 1. Exact invoice_id lookup
    if invoice_id:
        clean_id = invoice_id.strip()

        # Invoice ID Policy: Use an invoice ID only if written in canonical form.
        # For non-canonical forms (e.g. INV001, INV 036, #36), do NOT query the guessed ID.
        # Return not-found for user's text and suggest the closest canonical ID.
        if not is_canonical_invoice_id(clean_id):
            digits = re.findall(r'\d+', clean_id)
            if digits:
                num = int(digits[-1])
                sug_id = f"INV-2024-{num:03d}"
            else:
                sug_id = "INV-2024-001"
            conn.close()
            return {
                "found": False,
                "count": 0,
                "total_amount": 0.0,
                "total_formatted": "₹0.00",
                "searched_id": clean_id,
                "suggested_ids": [sug_id],
                "invoices": [],
                "model_payload": {
                    "error": f"{clean_id} was not found. Did you mean {sug_id}?",
                    "suggested_ids": [sug_id]
                },
                "error": f"{clean_id} was not found. Did you mean {sug_id}?"
            }

        check_sql = f"""
            SELECT invoice_id, vendor_name, invoice_date, due_date, amount_inr, status, payment_date, department, description
            FROM invoices
            WHERE LOWER(invoice_id) = LOWER({placeholder});
        """
        cur.execute(check_sql, (clean_id,))
        rows = cur.fetchall()

        if rows:
            cols = ["invoice_id", "vendor_name", "invoice_date", "due_date", "amount_inr", "status", "payment_date", "department", "description"]
            inv = dict(zip(cols, rows[0]))
            inv["amount_inr"] = float(inv["amount_inr"])
            inv["amount_formatted"] = format_inr(inv["amount_inr"])
            inv["invoice_date"] = str(inv["invoice_date"])
            inv["due_date"] = str(inv["due_date"])
            inv["payment_date"] = str(inv["payment_date"]) if inv.get("payment_date") else None
            conn.close()
            days_overdue_str = None
            if str(inv.get("status", "")).lower() == "overdue" and inv.get("due_date"):
                try:
                    import datetime
                    d_due = datetime.date.fromisoformat(str(inv["due_date"]))
                    d_as_of = datetime.date.fromisoformat(AS_OF_DATE)
                    d_diff = max(0, (d_as_of - d_due).days)
                    # Return days_overdue only, without hard-coded policy phase text
                    days_overdue_str = f"{d_diff} days overdue"
                except Exception:
                    pass

            return {
                "found": True,
                "count": 1,
                "total_amount": inv["amount_inr"],
                "total_formatted": format_inr(inv["amount_inr"]),
                "invoices": [inv],
                "model_payload": {
                    "invoice_id": inv["invoice_id"],
                    "vendor": inv["vendor_name"],
                    "status": inv["status"],
                    "amount": f"{inv['amount_formatted']} ({inv['amount_inr']/1e5:.2f} lakh, amount_inr={inv['amount_inr']})",
                    "due_date": inv["due_date"],
                    "days_overdue": days_overdue_str,
                    "payment_date": inv["payment_date"]
                },
                "message": f"Found invoice {inv['invoice_id']} for {inv['vendor_name']}."
            }
        else:
            # Search for candidate suggestions for canonical ID not in DB
            digits = re.findall(r'\d+', clean_id)
            if digits:
                num = digits[-1].zfill(3)
                like_pat = f"%{num}%"
                cur.execute(f"SELECT invoice_id FROM invoices WHERE invoice_id LIKE {placeholder} ORDER BY invoice_id ASC LIMIT 3;", (like_pat,))
                sug_rows = cur.fetchall()
                suggestions = [r[0] for r in sug_rows] if sug_rows else []
            else:
                cur.execute("SELECT invoice_id FROM invoices ORDER BY invoice_id ASC LIMIT 3;")
                suggestions = [r[0] for r in cur.fetchall()]

            if not suggestions:
                cur.execute("SELECT invoice_id FROM invoices ORDER BY invoice_id ASC LIMIT 3;")
                suggestions = [r[0] for r in cur.fetchall()]

            conn.close()
            return {
                "found": False,
                "count": 0,
                "total_amount": 0.0,
                "total_formatted": "₹0.00",
                "searched_id": clean_id,
                "suggested_ids": suggestions,
                "invoices": [],
                "model_payload": {
                    "error": f"Invoice '{clean_id}' was not found in records.",
                    "suggested_ids": suggestions
                },
                "error": f"Invoice '{clean_id}' was not found in the records."
            }

    # 2. Multi-parameter query with SQL aggregates
    conditions = []
    params = []

    if status:
        if isinstance(status, list):
            clean_list = [str(s).strip().lower() for s in status if s]
            matched_statuses = set()
            for s in clean_list:
                if s in ("unpaid", "not paid", "outstanding", "pending or overdue"):
                    matched_statuses.add("Pending")
                    matched_statuses.add("Overdue")
                elif s in ("paid", "cleared", "settled"):
                    matched_statuses.add("Paid")
                elif s == "overdue":
                    matched_statuses.add("Overdue")
                elif s == "pending":
                    matched_statuses.add("Pending")
                elif s == "disputed":
                    matched_statuses.add("Disputed")
                elif s in ("partially paid", "partially_paid", "partial"):
                    matched_statuses.add("Partially Paid")
                else:
                    matched_statuses.add(s.capitalize())
            if matched_statuses:
                status_list = list(matched_statuses)
                placeholders = ", ".join([placeholder] * len(status_list))
                conditions.append(f"status IN ({placeholders})")
                params.extend(status_list)
        else:
            st_clean = status.strip().lower()
            if st_clean in ("unpaid", "not paid", "outstanding", "pending or overdue"):
                conditions.append("status IN ('Pending', 'Overdue')")
            elif st_clean in ("paid", "cleared", "settled"):
                conditions.append("status = 'Paid'")
            elif st_clean == "overdue":
                conditions.append("status = 'Overdue'")
            elif st_clean == "pending":
                conditions.append("status = 'Pending'")
            elif st_clean == "disputed":
                conditions.append("status = 'Disputed'")
            elif st_clean in ("partially paid", "partially_paid", "partial"):
                conditions.append("status IN ('Partially Paid', 'Partially_Paid')")
            else:
                conditions.append(f"LOWER(status) = LOWER({placeholder})")
                params.append(status.strip())

    if min_amount is not None:
        conditions.append(f"amount_inr >= {placeholder}")
        params.append(float(min_amount))

    if max_amount is not None:
        conditions.append(f"amount_inr <= {placeholder}")
        params.append(float(max_amount))

    if vendor:
        conditions.append(f"LOWER(vendor_name) LIKE LOWER({placeholder})")
        params.append(f"%{vendor.strip()}%")

    if due_before:
        conditions.append(f"due_date <= {placeholder}")
        params.append(due_before.strip())

    if department:
        conditions.append(f"LOWER(department) = LOWER({placeholder})")
        params.append(department.strip())

    where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

    # SQL aggregation for count and total amount
    count_sql = f"SELECT COUNT(*), COALESCE(SUM(amount_inr), 0) FROM invoices {where_clause};"
    cur.execute(count_sql, tuple(params))
    count_row = cur.fetchone()
    total_count = int(count_row[0])
    total_amount = round(float(count_row[1]), 2)

    # Status breakdown aggregation via SQL
    status_sql = f"SELECT status, COUNT(*) FROM invoices {where_clause} GROUP BY status;"
    cur.execute(status_sql, tuple(params))
    status_breakdown = {r[0]: int(r[1]) for r in cur.fetchall()}

    # Fetch rows
    rows_sql = f"""
        SELECT invoice_id, vendor_name, invoice_date, due_date, amount_inr, status, payment_date, department, description
        FROM invoices
        {where_clause}
        ORDER BY amount_inr DESC
        LIMIT {int(limit)};
    """
    cur.execute(rows_sql, tuple(params))
    rows = cur.fetchall()

    cols = ["invoice_id", "vendor_name", "invoice_date", "due_date", "amount_inr", "status", "payment_date", "department", "description"]
    results = []
    for r in rows:
        d = dict(zip(cols, r))
        d["amount_inr"] = float(d["amount_inr"])
        d["amount_formatted"] = format_inr(d["amount_inr"])
        d["invoice_date"] = str(d["invoice_date"])
        d["due_date"] = str(d["due_date"])
        d["payment_date"] = str(d["payment_date"]) if d.get("payment_date") else None
        results.append(d)

    conn.close()

    # Task 5: If > 5 rows, condense payload for model to prevent 70s listing latency
    if total_count > 5:
        top_3 = [
            {
                "invoice_id": r["invoice_id"],
                "vendor": r["vendor_name"],
                "amount": r["amount_formatted"],
                "status": r["status"]
            }
            for r in results[:3]
        ]
        model_payload = {
            "total_count": total_count,
            "total_amount_formatted": format_inr(total_amount),
            "total_lakh_crore": format_inr_lakh_crore(total_amount),
            "status_breakdown": status_breakdown,
            "top_3_by_amount": top_3,
            "instruction": (
                f"There are {total_count} matching invoices. DO NOT list individual invoice rows or create a table; "
                "the Python CLI will display the full table. Write a concise 2-3 sentence financial summary highlighting "
                f"the total count ({total_count}), total value ({format_inr(total_amount)}), status breakdown, and top vendors."
            )
        }
    else:
        import datetime
        inv_payload_list = []
        for r in results:
            item = {
                "invoice_id": r["invoice_id"],
                "vendor": r["vendor_name"],
                "amount": r["amount_formatted"],
                "due_date": r["due_date"],
                "status": r["status"],
                "payment_date": r["payment_date"]
            }
            if str(r.get("status", "")).lower() == "overdue" and r.get("due_date"):
                try:
                    d_due = datetime.date.fromisoformat(str(r["due_date"]))
                    d_as_of = datetime.date.fromisoformat(AS_OF_DATE)
                    d_diff = (d_as_of - d_due).days
                    item["days_overdue"] = f"{d_diff} days (over 30 days overdue)" if d_diff >= 30 else f"{d_diff} days overdue"
                except Exception:
                    pass
            inv_payload_list.append(item)

        model_payload = {
            "total_count": total_count,
            "total_amount_formatted": format_inr(total_amount),
            "invoices": inv_payload_list
        }

    return {
        "found": total_count > 0,
        "count": total_count,
        "total_amount": total_amount,
        "total_formatted": format_inr(total_amount),
        "total_lakh_crore": format_inr_lakh_crore(total_amount),
        "status_breakdown": status_breakdown,
        "invoices": results,
        "model_payload": model_payload
    }

def get_invoice_by_id(invoice_id: str, backend: Optional[str] = None) -> Optional[Dict[str, Any]]:
    res = query_invoices(invoice_id=invoice_id, limit=1, backend=backend)
    if res.get("invoices"):
        return res["invoices"][0]
    return None

def get_distinct_vendors(backend: Optional[str] = None) -> List[str]:
    """Fetches unique vendor names from the invoices database."""
    try:
        conn, _ = get_connection(backend=backend)
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT vendor_name FROM invoices WHERE vendor_name IS NOT NULL;")
        vendors = [r[0] for r in cur.fetchall()]
        conn.close()
        return vendors
    except Exception:
        return [
            "Apex Facility Management", "AWS Cloud India", "Blue Dart Express",
            "Dell Technologies India", "Deloitte Advisory", "Infosys Cloud Services",
            "Microsoft India", "Reliance Office Supplies", "Tata Tech Logistics",
            "Vanguard Legal Partners", "Vendor ABC Corp", "Zomato Corporate Dining"
        ]
