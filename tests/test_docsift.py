import csv
import json
from pathlib import Path

import nuqta as nq
import pytest

import docsift
from docsift import DocsiftError
from docsift.core import count_tokens
from docsift.llm import PROMPT, RETRY_PROMPT, parse_response
from docsift.pdf import _clean

PDF = Path(__file__).parent.parent / "fixtures" / "pdf"
WITH_TRUTH = sorted(p.stem for p in PDF.glob("*.pdf") if (PDF / f"{p.stem}.table1.csv").exists())


def truth(name: str) -> list[list[list[str]]]:
    return [list(csv.reader(p.open(encoding="utf-8"))) for p in sorted(PDF.glob(f"{name}.table*.csv"))]


# ── local PDF backend against ground truth ─────────────────────────────────────

@pytest.mark.parametrize("name", WITH_TRUTH)
def test_pdf_tables_match_ground_truth(name):
    doc = docsift.read(PDF / f"{name}.pdf")
    expected = truth(name)
    assert len(doc.tables) == len(expected)
    for table, want in zip(doc.tables, expected):
        got = [table.columns, *table.rows]
        assert [[nq.normalize_key(c) for c in r] for r in got] == [[nq.normalize_key(c) for c in r] for r in want]


def test_multipage_table_is_merged():
    doc = docsift.read(PDF / "multipage_en.pdf")
    (t,) = doc.tables
    assert t.pages == [1, 2, 3, 4] and len(t.rows) == 90
    assert "Date" not in [r[0] for r in t.rows]  # repeated headers aren't data


def test_arabic_document_direction_and_text():
    doc = docsift.read(PDF / "ruled_ar.pdf")
    assert doc.direction == "rtl"
    assert doc.text.splitlines()[0] == "سجل الطلبات"
    assert doc.tables[0].columns[0] == "التاريخ"  # rightmost column comes first


def test_mixed_line_reads_in_logical_order():
    doc = docsift.read(PDF / "invoice_mixed.pdf")
    assert "Date / التاريخ: 2025-03-14 (١٣ رمضان ١٤٤٦)" in doc.text.splitlines()


def test_narrative_text_excludes_tables():
    doc = docsift.read(PDF / "ruled_en.pdf")
    assert "Order ledger" in doc.text and "Figures in USD." in doc.text
    assert "ORD-1000" not in doc.text


def test_scanned_pdf_raises_stable_code():
    with pytest.raises(DocsiftError) as e:
        docsift.read(PDF / "scanned.pdf")
    assert e.value.code == "SCANNED_PDF"


def test_pages_argument():
    doc = docsift.read(PDF / "multipage_en.pdf", pages="2-3")
    assert doc.tables[0].pages == [2, 3]
    for bad in ("9", "x", [0]):
        with pytest.raises(DocsiftError) as e:
            docsift.read(PDF / "multipage_en.pdf", pages=bad)
        assert e.value.code == "INVALID_PAGE_RANGE"


def test_profile_numbers_include_arabic_digits():
    doc = docsift.read(PDF / "ruled_ar.pdf")
    amount = doc.tables[0].profile()[4]
    assert amount["type"] == "number"
    want = sum(nq.to_number(r[4]) for r in truth("ruled_ar")[0][1:])
    assert amount["sum"] == pytest.approx(want)


def test_unmapped_glyphs_become_replacement_char():
    assert _clean("Total (cid:12)(cid:7)", "ltr") == "Total ��"


# ── CSV / Excel ────────────────────────────────────────────────────────────────

def test_csv_windows_1256_semicolons(tmp_path):
    p = tmp_path / "export.csv"
    p.write_bytes("التاريخ;المبلغ;‏البيان\n2025-01-01;1250;قلم\n2025-01-02;300;دفتر\n".encode("cp1256"))
    doc = docsift.read(p)
    assert doc.stats["encoding"] == "cp1256" and doc.stats["delimiter"] == ";"
    t = doc.tables[0]
    assert t.columns == ["التاريخ", "المبلغ", "البيان"]
    assert t.profile()[1]["sum"] == 1550


def test_xlsx_title_row_and_merged_two_row_header(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Quarterly summary"])
    ws.merge_cells("A1:E1")
    ws.append(["Region", "Q1", None, "Q2", None])
    ws.merge_cells("B2:C2")
    ws.merge_cells("D2:E2")
    ws.append([None, "Debit", "Credit", "Debit", "Credit"])
    ws.append(["North", 10, 5.5, 7, 0.1 + 0.2])
    p = tmp_path / "q.xlsx"
    wb.save(p)
    (t,) = docsift.read(p).tables
    assert t.columns == ["Region", "Q1 Debit", "Q1 Credit", "Q2 Debit", "Q2 Credit"]
    assert t.rows == [["North", "10", "5.5", "7", "0.3"]]


def test_unsupported_and_missing(tmp_path):
    (tmp_path / "a.docx").write_text("x")
    with pytest.raises(DocsiftError) as e:
        docsift.read(tmp_path / "a.docx")
    assert e.value.code == "UNSUPPORTED_FILE"
    with pytest.raises(DocsiftError) as e:
        docsift.read(tmp_path / "nope.pdf")
    assert e.value.code == "FILE_NOT_FOUND"


# ── rendering ──────────────────────────────────────────────────────────────────

def test_render_formats():
    doc = docsift.read(PDF / "ruled_en.pdf")
    md, tsv = doc.render(), doc.render("tsv")
    assert "| Date | Reference |" in md and "(7 more rows)" in md
    assert "```tsv\nDate\tReference" in tsv
    data = json.loads(doc.render("json", rows=None))
    assert data["tables"][0]["row_count"] == 12 and len(data["tables"][0]["rows"]) == 12
    with pytest.raises(ValueError):
        doc.render("xml")


def test_max_tokens_trims_rows_but_keeps_profile():
    doc = docsift.read(PDF / "multipage_en.pdf")
    out = doc.render(rows=None, max_tokens=400)
    assert count_tokens(out) <= 400
    assert "- Amount: number" in out


def test_token_report_orders_formats():
    r = docsift.read(PDF / "multipage_en.pdf").token_report()
    assert r["all_rows_tsv"] < r["all_rows_markdown"] < r["all_rows_json"]
    assert r["context_default"] < r["all_rows_tsv"]


# ── LLM backend: wiring only (no paid calls) ───────────────────────────────────

CANNED = """# Order ledger
Branch: North.

```tsv table=orders page_start=1 page_end=1
Date\tReference\tAmount
2025-07-21\tORD-1000\t550.41
2025-02-12\tORD-1001
```
"""


def test_llm_backend_gets_pdf_and_prompt_and_retries_once():
    calls = []

    def fake(data: bytes, prompt: str) -> str:
        calls.append((data, prompt))
        return "" if len(calls) == 1 else CANNED  # first answer unparseable

    doc = docsift.read(PDF / "ruled_en.pdf", llm=fake)
    assert [p for _, p in calls] == [PROMPT, RETRY_PROMPT]
    assert calls[0][0] == (PDF / "ruled_en.pdf").read_bytes() and calls[0][0].startswith(b"%PDF")
    (t,) = doc.tables
    assert (t.name, t.method, t.pages) == ("orders", "llm", [1])
    assert t.rows == [["2025-07-21", "ORD-1000", "550.41"], ["2025-02-12", "ORD-1001", None]]
    assert doc.text == "# Order ledger\nBranch: North." and doc.stats["backend"] == "llm"
    assert doc.pages == 1 and doc.stats["attempts"] == 2


def test_llm_backend_reports_failure_with_code():
    def broken(data, prompt):
        raise TimeoutError("upstream timeout")

    with pytest.raises(DocsiftError) as e:
        docsift.read(PDF / "ruled_en.pdf", llm=broken)
    assert e.value.code == "LLM_EXTRACTION_FAILED" and len(e.value.details["errors"]) == 2


def test_parse_response_edge_cases():
    raw = ("```tsv table=جدول\nالبند\\tالمبلغ\nقلم\\t5\n```\n"
           "```tsv table=جدول\na\tb\n1\t2\n```\n"
           "```tsv\nonly\theader\n```")
    tables, text = parse_response(raw)
    assert [t.name for t in tables] == ["جدول", "جدول_2"]  # Unicode names kept, deduped
    assert tables[0].rows == [["قلم", "5"]]                 # literal "\t" accepted
    assert text == ""                                       # header-only block dropped, no narrative
