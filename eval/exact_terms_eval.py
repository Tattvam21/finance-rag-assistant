#!/usr/bin/env python3
"""
eval/exact_terms_eval.py
Tests 6 exact-term queries on vector vs hybrid retrieval across embedders:
- nomic-embed-text
- bge-m3
- mxbai-embed-large

Target exact terms and ground-truth chunks:
- "RTGS": payment_policy.pdf §1.3 Payment Methods and Routing
- "GSTR-2B": finance_compliance_policy.pdf §5.2 GST Invoicing and Input Tax Credit
- "MSA-2023-ABC": vendor_policy.pdf §2.2 Special Vendor Agreements (Vendor ABC Corp)
- "194J": finance_compliance_policy.pdf §5.1 TDS (Tax Deducted at Source) Guidelines
- "ap@company.com": invoice_terms_policy.pdf §4.1 Invoice Submission and Processing Window
- "PO number": invoice_terms_policy.pdf §4.4 Three-Way Matching Rule
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.retrieval.search import search_chunks_with_timing

EXACT_TERMS = [
    {
        "term": "RTGS",
        "expected_source": "payment_policy.pdf",
        "expected_section": "1.3",
        "doc_hint": "Disbursements exceeding ₹2,00,000 must be routed via RTGS"
    },
    {
        "term": "GSTR-2B",
        "expected_source": "finance_compliance_policy.pdf",
        "expected_section": "5.2",
        "doc_hint": "Vendors must upload invoices to GSTN to appear in GSTR-2B"
    },
    {
        "term": "MSA-2023-ABC",
        "expected_source": "vendor_policy.pdf",
        "expected_section": "2.2",
        "doc_hint": "Master Services Agreement MSA-2023-ABC"
    },
    {
        "term": "194J",
        "expected_source": "finance_compliance_policy.pdf",
        "expected_section": "5.1",
        "doc_hint": "Section 194J (Professional / Technical Fees)"
    },
    {
        "term": "ap@company.com",
        "expected_source": "invoice_terms_policy.pdf",
        "expected_section": "4.1",
        "doc_hint": "All vendor invoices must be submitted electronically to ap@company.com"
    },
    {
        "term": "PO number",
        "expected_source": "invoice_terms_policy.pdf",
        "expected_section": "4.4",
        "doc_hint": "Purchase Order (PO), Goods Received Note (GRN)... Three-Way Matching"
    }
]

MODELS = ["nomic-embed-text", "bge-m3", "mxbai-embed-large"]

def find_rank(chunks: List[Dict[str, Any]], expected_src: str, expected_sec: str) -> Optional[int]:
    for idx, c in enumerate(chunks, 1):
        src = c.get("source_file", "")
        sec = str(c.get("section", ""))
        if expected_src in src and expected_sec in sec:
            return idx
    return None

def run_exact_terms_eval():
    print("=" * 90)
    print("EXACT-TERM RETRIEVAL EVALUATION: VECTOR vs HYBRID ACROSS EMBEDDERS")
    print("=" * 90)

    results = []

    for model in MODELS:
        print(f"\nEvaluating Embedder: {model}")
        print(f"{'Term':<15} | {'Vector Rank':<13} | {'Vector Sim':<11} | {'Hybrid Rank':<13} | {'Hybrid Score':<13} | Beat?")
        print("-" * 90)

        for item in EXACT_TERMS:
            term = item["term"]
            exp_src = item["expected_source"]
            exp_sec = item["expected_section"]

            # Vector only
            v_chunks, _, _ = search_chunks_with_timing(
                query=term,
                limit=5,
                retrieval_mode="vector",
                backend="postgres",
                embed_model=model
            )
            v_rank = find_rank(v_chunks, exp_src, exp_sec)
            v_score = v_chunks[0]["similarity"] if v_chunks else 0.0
            if v_rank and v_rank <= len(v_chunks):
                v_score = v_chunks[v_rank - 1]["similarity"]

            # Hybrid (Vector + Postgres FTS via RRF)
            h_chunks, _, _ = search_chunks_with_timing(
                query=term,
                limit=5,
                retrieval_mode="hybrid",
                backend="postgres",
                embed_model=model
            )
            h_rank = find_rank(h_chunks, exp_src, exp_sec)
            h_score = h_chunks[0].get("rrf_score", 0.0) if h_chunks else 0.0
            if h_rank and h_rank <= len(h_chunks):
                h_score = h_chunks[h_rank - 1].get("rrf_score", 0.0)

            v_rank_str = f"Rank {v_rank}" if v_rank else "Not in top-5"
            h_rank_str = f"Rank {h_rank}" if h_rank else "Not in top-5"

            # Does hybrid beat vector?
            # Beat means hybrid rank is strictly better (smaller integer, or found when vector missed)
            if v_rank is None and h_rank is not None:
                beat = "YES (H found, V missed)"
            elif v_rank is not None and h_rank is not None and h_rank < v_rank:
                beat = f"YES (R{h_rank} vs R{v_rank})"
            elif v_rank is not None and h_rank is not None and h_rank == v_rank:
                beat = f"TIE (both R{v_rank})"
            elif v_rank is not None and h_rank is None:
                beat = "NO (V beat H)"
            else:
                beat = "TIE (both missed)"

            print(f"{term:<15} | {v_rank_str:<13} | {v_score:<11.4f} | {h_rank_str:<13} | {h_score:<13.6f} | {beat}")

            results.append({
                "model": model,
                "term": term,
                "vector_rank": v_rank,
                "hybrid_rank": h_rank,
                "beat": beat
            })

    print("\n" + "=" * 90)
    print("ANALYSIS SUMMARY: DOES HYBRID EVER BEAT VECTOR?")
    print("-" * 90)
    for model in MODELS:
        m_res = [r for r in results if r["model"] == model]
        beats = [r for r in m_res if "YES" in r["beat"]]
        ties = [r for r in m_res if "TIE" in r["beat"]]
        losses = [r for r in m_res if "NO" in r["beat"]]
        print(f"[{model}]")
        print(f"  • Hybrid Beats Vector: {len(beats)}/6 terms")
        for b in beats:
            print(f"    - {b['term']}: {b['beat']}")
        print(f"  • Ties: {len(ties)}/6 terms")
        print(f"  • Vector Beats Hybrid: {len(losses)}/6 terms")

if __name__ == "__main__":
    run_exact_terms_eval()
