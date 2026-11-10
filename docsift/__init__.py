"""docsift: turn PDFs, CSVs and spreadsheets into compact, LLM-ready context.

    import docsift
    doc = docsift.read("report.pdf")
    print(doc.render())          # profile + 5 sample rows per table, markdown
    doc.tables[0].to_pandas()    # all rows
"""
from __future__ import annotations

from pathlib import Path

from .core import Document, DocsiftError, Table, count_tokens, profile_column
from .llm import PROMPT, parse_response

__all__ = ["read", "Document", "Table", "DocsiftError", "count_tokens", "profile_column",
           "PROMPT", "parse_response"]
__version__ = "0.1.0"


def read(path, *, pages=None, direction: str = "auto", llm=None) -> Document:
    """Read a .pdf, .csv, .xlsx/.xlsm or .xls file.

    pages:     PDF only. "1-3,5" or [1, 2]; default all.
    direction: PDF only. "auto" (majority of letters), "rtl" or "ltr". Sets how
               visual-order text is put back into reading order and whether
               table columns are read right-to-left.
    llm:       PDF only. A callable `llm(pdf_bytes, prompt) -> str`; when given,
               the PDF is extracted by that model instead of locally.
    """
    path = Path(path)
    if not path.is_file():
        raise DocsiftError("FILE_NOT_FOUND", str(path))
    if direction not in ("auto", "rtl", "ltr"):
        raise ValueError(f"direction must be 'auto', 'rtl' or 'ltr', got {direction!r}")
    ext = path.suffix.lower()
    if ext == ".pdf":
        if llm is not None:
            from .llm import read_pdf_llm
            return read_pdf_llm(path, llm)
        from .pdf import read_pdf
        return read_pdf(path, pages=pages, direction=direction)
    if ext in (".csv", ".tsv", ".txt"):
        from .tabular import read_csv
        return read_csv(path)
    if ext in (".xlsx", ".xlsm", ".xls"):
        from .tabular import read_excel
        return read_excel(path)
    raise DocsiftError("UNSUPPORTED_FILE", f"don't know how to read {ext or 'files without an extension'}",
                       file=str(path))
