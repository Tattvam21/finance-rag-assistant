#!/usr/bin/env python3
"""Build eval/manual_audit.csv from saved results JSON + the policy PDFs (no LLM, no agent text)."""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFUSAL = "i couldn't find that in the documents"
SEC_RE = re.compile(r"^(\d+\.\d+)\s+[A-Za-z]", re.M)
CITE_RE = re.compile(r"\[\s*([\w\-]+\.pdf)\s*,\s*p\.\s*\d+\s*,\s*§\s*(\d+\.\d+)")


def pdf_text(path: Path) -> str:
    try:
        import pymupdf
        return "\n".join(p.get_text() for p in pymupdf.open(str(path)))
    except ImportError:
        try:
            import fitz
            return "\n".join(p.get_text() for p in fitz.open(str(path)))
        except ImportError:
            from pypdf import PdfReader
            return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)



def split_sections(text: str) -> dict:
    marks = [(m.start(), m.group(1)) for m in SEC_RE.finditer(text)]
    out = {}
    for i, (pos, num) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        out[num] = re.sub(r"[ \t]+", " ", text[pos:end]).strip()
    return out


def find_rows(obj):
    """Return the first list of dicts that look like per-question result rows."""
    if isinstance(obj, list):
        if obj and isinstance(obj[0], dict) and "id" in obj[0] and ("answer" in obj[0] or "final_answer" in obj[0]):
            return obj
        for v in obj:
            r = find_rows(v)
            if r:
                return r
    elif isinstance(obj, dict):
        for v in obj.values():
            r = find_rows(v)
            if r:
                return r
    return None


def tool_blob(row: dict) -> str:
    parts = []
    for k, v in row.items():
        if "tool" in k.lower() and v and k.lower() not in ("tools", "tools_ok"):
            s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
            parts.append(f"{k}: {s[:1500]}")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True, help="results JSON files (dev, held-out)")
    ap.add_argument("--questions", nargs="+",
                    default=[str(HERE / "questions.json"), str(HERE / "questions_heldout.json")])
    ap.add_argument("--pdf-dir", default="data/policies")
    ap.add_argument("--out", default=str(HERE / "manual_audit.csv"))
    args = ap.parse_args()

    secs = {}
    for pdf in sorted(Path(args.pdf_dir).glob("*.pdf")):
        secs[pdf.name] = split_sections(pdf_text(pdf))
    if not secs:
        sys.exit(f"No PDFs found in {args.pdf_dir}")
    print("Sections parsed from PDFs:", {k: sorted(v) for k, v in secs.items()})

    qmeta = {}
    for qf in args.questions:
        for q in json.loads(Path(qf).read_text(encoding="utf-8")):
            qmeta[q["id"]] = q

    def wanted(q):
        if q.get("expect_refusal"):
            return False
        return bool(q.get("expected_sources")) or q.get("category", "").startswith("trap")

    rows = {}
    for rf in args.results:
        data = json.loads(Path(rf).read_text(encoding="utf-8"))
        found = find_rows(data)
        if not found:
            sys.exit(f"Could not find per-question rows in {rf}")
        for r in found:
            rows[r["id"]] = r

    out_rows, skipped = [], []
    for qid, q in qmeta.items():
        if not wanted(q):
            continue
        r = rows.get(qid)
        if not r:
            skipped.append((qid, "not in results"))
            continue
        answer = (r.get("answer") or r.get("final_answer") or "").strip()
        if REFUSAL in answer.lower():
            skipped.append((qid, "refusal"))
            continue
        cites = CITE_RE.findall(answer)
        note = ""
        if not cites:
            exp = [s.split("#") for s in q.get("expected_sources", [])]
            cites = [(f, n) for f, n in exp]
            note = "NO CITATION IN ANSWER; evidence below is the expected section. "
        ev = []
        for f, n in dict.fromkeys(cites):
            ev.append(f"[{f} §{n}]\n{secs.get(f, {}).get(n, '!! SECTION NOT FOUND IN PDF !!')}")
        out_rows.append({
            "id": qid,
            "question": q.get("q", ""),
            "answer": answer,
            "evidence_from_pdf": "\n\n---\n\n".join(ev),
            "tool_results": tool_blob(r),
            "supported": "",
            "notes": note.strip(),
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print(f"\nWrote {len(out_rows)} rows to {args.out}")
    if skipped:
        print("Skipped:", skipped)
    not_found = [r["id"] for r in out_rows if "NOT FOUND" in r["evidence_from_pdf"]]
    if not_found:
        print("WARNING: evidence section not found for:", not_found)


if __name__ == "__main__":
    main()
