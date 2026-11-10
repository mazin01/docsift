"""CSV and Excel readers. Cells come back as cleaned strings; typing is the profile's job."""
from __future__ import annotations

import csv
import io
import time
from collections import Counter
from datetime import date, datetime, time as dtime
from pathlib import Path

import nuqta as nq

from .core import DocsiftError, Document, Table

SNIFF_BYTES = 1_000_000


def _cell(v):
    """Python/Excel value → clean string or None."""
    if v is None:
        return None
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else f"{v:.15g}"
    if isinstance(v, datetime):
        return v.date().isoformat() if v.time() == dtime(0) else v.isoformat(sep=" ")
    if isinstance(v, (date, dtime)):
        return v.isoformat()
    s = " ".join(nq.canonical_spaces(nq.strip_invisible(str(v))).split())
    return s or None


def _table(name: str, grid: list[list], method: str) -> Table | None:
    """First non-title row is the header; empty rows/columns dropped; header names deduped."""
    grid = [[_cell(v) for v in r] for r in grid]
    grid = [r for r in grid if any(v is not None for v in r)]
    if not grid:
        return None
    width = max(len(r) for r in grid)
    grid = [r + [None] * (width - len(r)) for r in grid]
    keep = [i for i in range(width) if any(r[i] is not None for r in grid)]
    grid = [[r[i] for i in keep] for r in grid]
    # Title rows ("Quarterly sales report", merged across the top) have one value; skip them.
    while len(grid) > 1 and len(keep) > 1 and sum(v is not None for v in grid[0]) == 1:
        grid.pop(0)
    seen, header = Counter(), []
    for i, v in enumerate(grid[0]):
        name_ = v or f"col_{i + 1}"
        seen[name_] += 1
        header.append(name_ if seen[name_] == 1 else f"{name_}_{seen[name_]}")
    return Table(name, header, grid[1:], method)


def read_csv(path) -> Document:
    started = time.perf_counter()
    path = Path(path)
    raw = path.read_bytes()
    if not raw.strip():
        raise DocsiftError("EMPTY_FILE", "file is empty", file=str(path))
    encoding = nq.sniff_encoding(raw[:SNIFF_BYTES])
    text = raw.decode(encoding, errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text[:50_000], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    table = _table(path.stem, list(csv.reader(io.StringIO(text), dialect)), "csv")
    tables = [table] if table else []
    return Document(path.name, "csv", tables,
                    stats={"backend": "local", "encoding": encoding, "delimiter": dialect.delimiter,
                           "tables": len(tables), "rows": sum(len(t.rows) for t in tables),
                           "seconds": round(time.perf_counter() - started, 3)})


def _merged_header(ws, grid: list[list]) -> list[list]:
    """Fill merged ranges with their top-left value, and fold a two-row header
    ("Q1" merged over "Debit | Credit") into one row ("Q1 Debit", "Q1 Credit")."""
    spans = []
    single = {i for i, r in enumerate(grid) if sum(v is not None for v in r) == 1}  # titles, before filling
    for rng in ws.merged_cells.ranges:
        if rng.min_row - 1 in single:
            continue  # a merged title stays one value so _table() can skip it
        value = grid[rng.min_row - 1][rng.min_col - 1]
        for r in range(rng.min_row - 1, rng.max_row):
            for c in range(rng.min_col - 1, rng.max_col):
                grid[r][c] = value
        if rng.max_col > rng.min_col:
            spans.append(rng.min_row - 1)
    rows = [i for i, r in enumerate(grid) if sum(v is not None for v in r) > 1]
    if len(rows) >= 2:
        top, sub = rows[0], rows[1]
        if top in spans and all(not isinstance(v, (int, float)) for v in grid[sub] if v is not None):
            grid[sub] = [f"{a} {b}" if a is not None and b is not None and a != b else (b if b is not None else a)
                         for a, b in zip(grid[top], grid[sub])]
            del grid[top]
    return grid


def read_excel(path) -> Document:
    started = time.perf_counter()
    path = Path(path)
    tables = []
    if path.suffix.lower() == ".xls":
        import xlrd

        book = xlrd.open_workbook(str(path))
        for sh in book.sheets():
            grid = [[xlrd.xldate_as_datetime(c.value, book.datemode) if c.ctype == xlrd.XL_CELL_DATE else
                     (None if c.ctype == xlrd.XL_CELL_EMPTY else c.value) for c in sh.row(r)] for r in range(sh.nrows)]
            if (t := _table(sh.name, grid, "excel")):
                tables.append(t)
        kind = "xls"
    else:
        import openpyxl

        # ponytail: full (not read-only) load so merged cells are visible; huge sheets
        # would want read_only=True and no merged-header support.
        book = openpyxl.load_workbook(path, data_only=True)
        for ws in book.worksheets:
            grid = [list(r) for r in ws.iter_rows(min_row=1, min_col=1, values_only=True)]
            if grid and (t := _table(ws.title, _merged_header(ws, grid), "excel")):
                tables.append(t)
        kind = "xlsx"
    return Document(path.name, kind, tables,
                    stats={"backend": "local", "tables": len(tables), "rows": sum(len(t.rows) for t in tables),
                           "seconds": round(time.perf_counter() - started, 3)})
