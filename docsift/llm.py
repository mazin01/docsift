"""Optional LLM backend: send the PDF to a vision model, get markdown + fenced TSV back.

Provider-agnostic: you pass `llm(pdf_bytes, prompt) -> str`. Asking for TSV
instead of JSON keeps the *output* tokens low (no repeated keys, no quoting),
and fenced blocks parse reliably even when the model adds prose around them.
"""
from __future__ import annotations

import re
import time
from collections import Counter
from pathlib import Path
from typing import Callable

import pdfplumber

from .core import DocsiftError, Document, Table
from .pdf import _direction

LLM = Callable[[bytes, str], str]

PROMPT = "\n".join([
    "You are extracting structured data from a PDF document.",
    "",
    "Return markdown, not JSON.",
    "",
    "First write a concise markdown rendering of the document narrative, headings, and key-value facts.",
    "Then export every detected table as its own fenced TSV block.",
    "",
    "Use this exact table block format:",
    "```tsv table=<short_slug> page_start=<number_or_blank> page_end=<number_or_blank>",
    "<column 1>\t<column 2>\t<column 3>",
    "<row 1 col 1>\t<row 1 col 2>\t<row 1 col 3>",
    "```",
    "",
    "Rules:",
    "  - Do not invent or summarise table values. If a table cell is blank, leave the TSV cell empty.",
    "  - Preserve non-English text (e.g. Arabic) verbatim. Do not translate headers or values.",
    "  - Preserve numbers exactly as written, including commas, signs, decimals, and currency symbols.",
    "  - Preserve the original column order. Keep original headers unless they are visually blank.",
    "  - TSV rows must have the same number of columns as the header. "
    "If a value contains a tab or newline, replace it with a single space.",
    "  - For multi-page tables that continue with the same header, merge them into ONE entry with the full page range.",
    "  - If no tables are present, return only the markdown narrative and no TSV blocks.",
    "  - Keep the narrative markdown under 12000 characters; prioritize the tables if output is long.",
])

RETRY_PROMPT = "\n".join([
    PROMPT,
    "",
    "The previous response could not be parsed. Retry with only markdown plus fenced TSV blocks.",
    "Do not use JSON. Do not use markdown tables for extracted data tables; use fenced TSV blocks.",
])

_FENCE_RE = re.compile(r"```(?:tsv|tab-separated-values)\b([^\n`]*)\n(.*?)```", re.IGNORECASE | re.DOTALL)
_ATTR_RE = re.compile(r"([A-Za-z_][\w-]*)=(?:\"([^\"]*)\"|'([^']*)'|(\S+))")


def _split(line: str) -> list[str]:
    # Models sometimes write a literal backslash-t instead of a tab.
    return line.split("\t") if "\t" in line else line.split("\\t")


def parse_response(raw: str) -> tuple[list[Table], str]:
    """Fenced TSV blocks → Tables; everything else → narrative markdown."""
    tables: list[Table] = []
    names = Counter()
    for info, body in _FENCE_RE.findall(raw):
        lines = [l.rstrip() for l in body.replace("\r\n", "\n").replace("\r", "\n").split("\n") if l.strip()]
        if not lines:
            continue
        columns = [c.strip() for c in _split(lines[0])]
        while columns and not columns[-1]:
            columns.pop()
        if not columns:
            continue
        rows = []
        for line in lines[1:]:
            cells = _split(line)
            rows.append([(cells[i].strip() or None) if i < len(cells) else None for i in range(len(columns))])
        if not rows:
            continue
        meta = {m[0]: (m[1] or m[2] or m[3]).strip() for m in _ATTR_RE.findall(info)}
        name = re.sub(r"\W+", "_", meta.get("table") or meta.get("name") or "").strip("_") or f"table_{len(tables) + 1}"
        names[name] += 1
        if names[name] > 1:
            name = f"{name}_{names[name]}"
        pages = [int(meta[k]) for k in ("page_start", "page_end") if meta.get(k, "").isdigit()]
        pages = list(range(pages[0], pages[-1] + 1)) if pages else []
        tables.append(Table(name, columns, rows, "llm", pages))
    text = re.sub(r"\n{3,}", "\n\n", _FENCE_RE.sub("", raw)).strip()
    return tables, text


def read_pdf_llm(path, llm: LLM) -> Document:
    """Extract a PDF with `llm`. Retries once with a stricter prompt if nothing parses."""
    started = time.perf_counter()
    path = Path(path)
    data = path.read_bytes()
    with pdfplumber.open(path) as pdf:
        total = len(pdf.pages)
    tables, text, errors = [], "", []
    for attempt, prompt in enumerate((PROMPT, RETRY_PROMPT), 1):
        try:
            tables, text = parse_response(llm(data, prompt))
        except Exception as exc:  # the caller's client failed; try once more, then report
            errors.append(f"{type(exc).__name__}: {exc}")
            continue
        if tables or text:
            break
    else:
        raise DocsiftError("LLM_EXTRACTION_FAILED", "no parseable markdown/TSV after 2 attempts",
                           errors=errors, file=str(path))
    sample = text + "".join(str(c) for t in tables for c in t.columns)
    return Document(path.name, "pdf", tables, text=text, pages=total, direction=_direction(sample),
                    stats={"backend": "llm", "attempts": attempt, "tables": len(tables),
                           "rows": sum(len(t.rows) for t in tables),
                           "seconds": round(time.perf_counter() - started, 3)})
