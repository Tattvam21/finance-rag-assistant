import os
import csv
import random
from datetime import datetime, timedelta
import matplotlib
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Register Unicode fonts (DejaVu Sans has full support for ₹ U+20B9)
_font_dir = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf')
pdfmetrics.registerFont(TTFont('DejaVuSans', os.path.join(_font_dir, 'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DejaVuSans-Bold', os.path.join(_font_dir, 'DejaVuSans-Bold.ttf')))
pdfmetrics.registerFont(TTFont('DejaVuSans-Oblique', os.path.join(_font_dir, 'DejaVuSans-Oblique.ttf')))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
POLICIES_DIR = os.path.join(DATA_DIR, "policies")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(POLICIES_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Generate invoices.csv
# -------------------------------------------------------------
def generate_invoices():
    csv_path = os.path.join(DATA_DIR, "invoices.csv")
    if os.path.exists(csv_path):
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            if len(reader) == 90 and reader[-1]["invoice_id"] == "INV-2024-090":
                print(f"invoices.csv already contains 90 canonical invoices up to INV-2024-090. Preserved without drift.")
                return

    # If generating from scratch, build 90 invoices deterministically
    random.seed(42)
    vendors = [
        ("Vendor ABC Corp", 45, "IT"),
        ("Tata Tech Logistics", 30, "Operations"),
        ("Reliance Office Supplies", 30, "Operations"),
        ("Infosys Cloud Services", 30, "IT"),
        ("Apex Facility Management", 30, "HR"),
        ("Microsoft India", 30, "IT"),
        ("AWS Cloud India", 30, "IT"),
        ("Dell Technologies India", 30, "IT"),
        ("Deloitte Advisory", 30, "Finance"),
        ("Zomato Corporate Dining", 30, "HR"),
        ("Blue Dart Express", 30, "Operations"),
        ("Vanguard Legal Partners", 30, "Legal"),
    ]

    descriptions = {
        "IT": ["Cloud infrastructure hosting", "Enterprise software licenses", "Server maintenance", "Workstation upgrades", "Network security audit"],
        "Operations": ["Warehousing logistics", "Fleet transport charges", "Office stationery bulk pack", "Packaging material supply", "Courier services"],
        "HR": ["Employee cafeteria catering", "Recruitment portal subscription", "Corporate wellness program", "Offsite event management"],
        "Finance": ["Quarterly internal audit", "Statutory tax compliance advisory", "Accounting software subscription"],
        "Legal": ["Contract review retainership", "Regulatory filing consultation"]
    }

    base_date = datetime(2024, 1, 15)
    as_of_date = datetime(2024, 8, 5)
    invoices = []

    for i in range(1, 81):
        inv_id = f"INV-2024-{i:03d}"
        vendor_name, net_days, default_dept = random.choice(vendors)
        desc = random.choice(descriptions[default_dept])
        
        if random.random() < 0.45:
            amount = round(random.uniform(105000, 850000), 2)
        else:
            amount = round(random.uniform(12000, 95000), 2)

        invoice_offset = random.randint(0, 180)
        inv_date = base_date + timedelta(days=invoice_offset)
        due_date = inv_date + timedelta(days=net_days)

        status_roll = random.random()
        if status_roll < 0.40:
            status = "Paid"
            payment_date = (inv_date + timedelta(days=random.randint(5, net_days))).strftime("%Y-%m-%d")
        else:
            payment_date = ""
            status = "Overdue" if due_date < as_of_date else "Pending"

        invoices.append({
            "invoice_id": inv_id,
            "vendor_name": vendor_name,
            "invoice_date": inv_date.strftime("%Y-%m-%d"),
            "due_date": due_date.strftime("%Y-%m-%d"),
            "amount_inr": amount,
            "status": status,
            "payment_date": payment_date,
            "department": default_dept,
            "description": desc
        })

    # Coverage rows INV-2024-081 to INV-2024-090
    coverage_rows = [
        {"invoice_id": "INV-2024-081", "vendor_name": "Tata Tech Logistics", "invoice_date": "2024-07-06", "due_date": "2024-08-05", "amount_inr": 45000.00, "status": "Pending", "payment_date": "", "department": "Operations", "description": "Fleet transport charges"},
        {"invoice_id": "INV-2024-082", "vendor_name": "Reliance Office Supplies", "invoice_date": "2024-07-07", "due_date": "2024-08-06", "amount_inr": 28500.00, "status": "Pending", "payment_date": "", "department": "Operations", "description": "Office stationery bulk pack"},
        {"invoice_id": "INV-2024-083", "vendor_name": "Vendor ABC Corp", "invoice_date": "2024-07-12", "due_date": "2024-08-26", "amount_inr": 480000.00, "status": "Pending", "payment_date": "", "department": "IT", "description": "Cloud infrastructure hosting"},
        {"invoice_id": "INV-2024-084", "vendor_name": "Microsoft India", "invoice_date": "2024-07-15", "due_date": "2024-08-14", "amount_inr": 310000.00, "status": "Pending", "payment_date": "", "department": "IT", "description": "Enterprise software licenses"},
        {"invoice_id": "INV-2024-085", "vendor_name": "Deloitte Advisory", "invoice_date": "2024-07-18", "due_date": "2024-08-17", "amount_inr": 75000.00, "status": "Pending", "payment_date": "", "department": "Finance", "description": "Statutory tax compliance advisory"},
        {"invoice_id": "INV-2024-086", "vendor_name": "AWS Cloud India", "invoice_date": "2024-07-20", "due_date": "2024-08-19", "amount_inr": 540000.00, "status": "Pending", "payment_date": "", "department": "IT", "description": "Server maintenance"},
        {"invoice_id": "INV-2024-087", "vendor_name": "Blue Dart Express", "invoice_date": "2024-07-22", "due_date": "2024-08-21", "amount_inr": 18500.00, "status": "Pending", "payment_date": "", "department": "Operations", "description": "Courier services"},
        {"invoice_id": "INV-2024-088", "vendor_name": "Infosys Cloud Services", "invoice_date": "2024-07-25", "due_date": "2024-08-24", "amount_inr": 225000.00, "status": "Pending", "payment_date": "", "department": "IT", "description": "Workstation upgrades"},
        {"invoice_id": "INV-2024-089", "vendor_name": "Apex Facility Management", "invoice_date": "2024-07-28", "due_date": "2024-08-27", "amount_inr": 62000.00, "status": "Pending", "payment_date": "", "department": "HR", "description": "Corporate wellness program"},
        {"invoice_id": "INV-2024-090", "vendor_name": "Zomato Corporate Dining", "invoice_date": "2024-08-02", "due_date": "2024-09-01", "amount_inr": 89000.00, "status": "Pending", "payment_date": "", "department": "HR", "description": "Employee cafeteria catering"},
    ]
    invoices.extend(coverage_rows)

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["invoice_id", "vendor_name", "invoice_date", "due_date", "amount_inr", "status", "payment_date", "department", "description"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(invoices)

    print(f"Generated {len(invoices)} invoices at {csv_path}")

# -------------------------------------------------------------
# 2. Generate Policy PDFs using reportlab
# -------------------------------------------------------------
def build_pdf(filename, title, sections):
    pdf_path = os.path.join(POLICIES_DIR, filename)
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='DejaVuSans-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=12
    )

    section_style = ParagraphStyle(
        'SecTitle',
        parent=styles['Heading2'],
        fontName='DejaVuSans-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#2563EB'),
        spaceBefore=14,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='DejaVuSans',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155'),
        spaceAfter=6
    )

    sub_style = ParagraphStyle(
        'SubHeader',
        parent=styles['Italic'],
        fontName='DejaVuSans-Oblique',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#64748B')
    )

    story = [
        Paragraph(title, title_style),
        Paragraph("Corporate Finance & Operations Policy Manual | Version 2.4", sub_style),
        Spacer(1, 10),
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#CBD5E1'), spaceAfter=14)
    ]

    for sec_num, sec_title, sec_paragraphs in sections:
        header_text = f"<b>{sec_num} {sec_title}</b>"
        story.append(Paragraph(header_text, section_style))
        for p in sec_paragraphs:
            story.append(Paragraph(p, body_style))
        story.append(Spacer(1, 4))

    doc.build(story)
    print(f"Built PDF policy: {pdf_path}")

def generate_policies():
    # Policy 1: Payment Policy
    build_pdf(
        "payment_policy.pdf",
        "Corporate Payment & Disbursement Policy",
        [
            ("1.1", "Standard Payment Terms", [
                "The organization operates strictly under standard Net 30 payment terms for all regular vendors, meaning settlement is scheduled thirty (30) calendar days from receipt of a verified, undisputed tax invoice.",
                "Invoices received after 5:00 PM IST on any business day are registered on the next business day."
            ]),
            ("1.2", "Disbursement Schedules and Cycles", [
                "Accounts Payable executes bulk disbursements twice monthly: on the 1st and the 15th calendar day (or the next immediate business day if falling on a banking holiday or weekend).",
                "Emergency or out-of-cycle payments require explicit written sign-off from the Corporate Controller or Chief Financial Officer."
            ]),
            ("1.3", "Payment Methods and Routing", [
                "All domestic vendor disbursements exceeding ₹2,00,000 must be routed via Real Time Gross Settlement (RTGS).",
                "Payments between ₹10,000 and ₹2,00,000 are processed via National Electronic Funds Transfer (NEFT) or Immediate Payment Service (IMPS). Physical cheque issuance is prohibited except for court-mandated statutory deposits."
            ]),
            ("1.4", "Early Settlement Discounts", [
                "Vendors offering a prompt payment discount of 1.5% or greater for settlement within ten (10) calendar days will be prioritized in the weekly payment run."
            ])
        ]
    )

    # Policy 2: Vendor Policy
    build_pdf(
        "vendor_policy.pdf",
        "Vendor Management & SLA Policy",
        [
            ("2.1", "Vendor Onboarding and Due Diligence", [
                "Prior to issuing any purchase orders, prospective suppliers must complete Level-1 KYC, submit GSTIN registration certificates, verified PAN cards, and cancelled cheques.",
                "Suppliers without active GST registration cannot be boarded for contracts exceeding ₹50,000 per annum."
            ]),
            ("2.2", "Special Vendor Agreements (Vendor ABC Corp)", [
                "By virtue of Master Services Agreement MSA-2023-ABC, Vendor ABC Corp is granted customized Net 45 payment terms rather than standard Net 30 terms.",
                "All hardware and cloud procurement from Vendor ABC Corp requires three-way matching against warehouse delivery receipts before invoice approval."
            ]),
            ("2.3", "Dispute Resolution and Hold Protocol", [
                "If an invoice contains discrepancies, Accounts Payable must notify the vendor within seven (7) business days of invoice receipt.",
                "The disputed portion will be placed on administrative hold, while undisputed amounts under the same invoice must be paid per schedule."
            ]),
            ("2.4", "Vendor Blacklisting Criteria", [
                "Vendors failing to fulfill statutory GST filing (GSTR-1 / GSTR-3B) for two consecutive quarters will have all pending payouts blocked until reconciliation is finalized."
            ])
        ]
    )

    # Policy 3: Expense Policy
    build_pdf(
        "expense_policy.pdf",
        "Employee Travel & Business Expense Policy",
        [
            ("3.1", "Approval Limits and Hierarchy", [
                "Tier 1 (Self-Approval): Incidental business expenses up to ₹5,000 do not require prior management approval if accompanied by valid itemized GST receipts.",
                "Tier 2 (Manager Approval): Expenses between ₹5,001 and ₹25,000 require written approval from the direct Reporting Manager.",
                "Tier 3 (Department Head Approval): Outlays between ₹25,001 and ₹1,00,000 require Department Head (VP/Director) authorization.",
                "Tier 4 (Executive Approval): Any expenditure exceeding ₹1,00,000 mandates explicit sign-off from the Chief Financial Officer (CFO)."
            ]),
            ("3.2", "Daily Travel and Per Diem Rates", [
                "For Tier-1 metropolitan cities (Mumbai, Delhi NCR, Bengaluru, Hyderabad), the maximum lodging reimbursement is ₹6,500 per night, with a daily per diem food allowance of ₹1,500.",
                "For Tier-2 and other municipal locations, lodging ceiling is ₹4,000 per night, with daily food allowance capped at ₹1,000."
            ]),
            ("3.3", "Reimbursement Timelines", [
                "All expense reports must be submitted within thirty (30) days of incurrence. Claims submitted after 60 days will be forfeited unless approved by the VP of People."
            ])
        ]
    )

    # Policy 4: Invoice Terms Policy
    build_pdf(
        "invoice_terms_policy.pdf",
        "Commercial Invoice Terms & Overdue Policies",
        [
            ("4.1", "Invoice Submission Requirements", [
                "Invoices must be submitted electronically to ap@company.com with purchase order (PO) reference numbers clearly stated on the invoice header.",
                "The monthly billing cutoff is the 25th calendar day of each month. Invoices submitted after the 25th are processed in the subsequent month's first cycle."
            ]),
            ("4.2", "Overdue Invoices and Late Fee Structure", [
                "Invoices remaining unpaid past their contractual due date (Net 30 or Net 45) will accrue a late interest fee of 2.0% per month (or 24% per annum pro-rata) on the outstanding principal.",
                "Interest charges will only apply if the delay is solely attributable to internal processing and not vendor billing errors."
            ]),
            ("4.3", "Overdue Escalation Workflow", [
                "Phase 1 (Day 1 to 14 Overdue): Automated reminder notification generated by ERP and dispatched to vendor relationship manager.",
                "Phase 2 (Day 15 Overdue): Mandatory escalation meeting between Finance Controller and Vendor Account Lead. Accounts Payable freezes new purchase order issuance to the vendor.",
                "Phase 3 (Day 30+ Overdue): Formal dispute review and legal notification protocol activated."
            ]),
            ("4.4", "Three-Way Matching Rule", [
                "Payment disbursement requires verification of three documents: Purchase Order (PO), Goods Received Note (GRN) or Service Completion Certificate, and the Vendor Tax Invoice. Discrepancies exceeding 0.5% require re-authorization."
            ])
        ]
    )

    # Policy 5: Finance Compliance Policy
    build_pdf(
        "finance_compliance_policy.pdf",
        "Finance Compliance, TDS & Statutory Withholding",
        [
            ("5.1", "Tax Deducted at Source (TDS) Withholding Rates", [
                "Section 194C (Contractors & Sub-contractors): 1% TDS for individual/HUF contractors, 2% TDS for corporate entities.",
                "Section 194J (Professional & Technical Services): 10% TDS for professional fee disbursements, 2% TDS for technical call center or software maintenance services.",
                "Section 194I (Rent & Property Lease): 10% TDS on land/building leases, 2% on plant and machinery lease."
            ]),
            ("5.2", "Goods and Services Tax (GST) Verification", [
                "No invoice will be cleared for payment if the vendor's GSTIN status shows cancelled or suspended on the GST Portal.",
                "Input Tax Credit (ITC) eligibility must match data reflected in Form GSTR-2B before finalizing monthly vendor balances."
            ]),
            ("5.3", "Statutory Audit Records Retention", [
                "All tax invoices, debit notes, credit notes, and payment remittance slips must be retained in digital storage for a statutory minimum of eight (8) consecutive fiscal years."
            ])
        ]
    )

if __name__ == "__main__":
    generate_invoices()
    generate_policies()
    print("All synthetic data generated successfully!")
