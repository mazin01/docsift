"""Local PDF backend: pdfplumber + bidi-aware cleanup. No network, no API keys."""
from __future__ import annotations

import re
import statistics
import time
import unicodedata
from collections import Counter
from pathlib import Path

import nuqta as nq
import pdfplumber

from .core import DocsiftError, Document, Table

_CID_RE = re.compile(r"\(cid:\d+\)")          # pdfminer's placeholder for unmapped glyphs
_BAD_CHAR_RE = re.compile("[\x01-\x08\x0b\x0c\x0e-\x1f�]")
MIN_ALIGNED_COLUMNS = 3
MIN_ALIGNED_ROWS = 3                            # header + 2 data rows
LINE_TOLERANCE = 3                              # pt: words this close vertically share a line
PHRASE_GAP = 0.8                                # × median word height: wider gaps split cells


def parse_pages(pages, total: int) -> list[int]:
    """None → all; "1-3,5" or [1, 2] → sorted 1-based page numbers, validated."""
    if pages is None:
        return list(range(1, total + 1))
    try:
        if isinstance(pages, str):
            out = []
            for part in pages.split(","):
                a, _, b = part.strip().partition("-")
                out += range(int(a), int(b or a) + 1)
        else:
            out = [int(p) for p in pages]
    except ValueError:
        raise DocsiftError("INVALID_PAGE_RANGE", f"can't parse pages={pages!r}", total_pages=total) from None
    if not out or any(not 1 <= p <= total for p in out):
        raise DocsiftError("INVALID_PAGE_RANGE", f"pages={pages!r} outside 1-{total}", total_pages=total)
    return sorted(set(out))


def _direction(text: str) -> str:
    rtl = sum(unicodedata.bidirectional(c) in ("R", "AL") for c in text)
    ltr = sum(unicodedata.bidirectional(c) == "L" for c in text)
    return "rtl" if rtl > ltr else "ltr"


def _clean(text, base: str):
    """Extracted (visual-order) text → logical, single-line, NFKC'd if shaped glyphs."""
    if text is None:
        return None
    text = _CID_RE.sub("�", text)
    s = " ".join(line.strip() for line in nq.visual_to_logical(text, base).split("\n") if line.strip())
    if nq.has_presentation_forms(s):
        s = unicodedata.normalize("NFKC", s)
    return s or None


# ── table finders: both return [(bbox, raw_rows)] in visual (left→right) order ───

def _ruled_tables(page) -> list:
    out = []
    for t in page.find_tables():
        raw = t.extract()
        if len(raw) >= 2 and len(raw[0]) >= 2:
            out.append((t.bbox, raw))
    return out


def _aligned_tables(page) -> list:
    """Borderless tables: runs of lines whose words form ≥3 gap-separated phrases."""
    words = page.extract_words()
    if not words:
        return []
    gap = PHRASE_GAP * statistics.median(w["bottom"] - w["top"] for w in words)
    lines: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if lines and abs(lines[-1][0]["top"] - w["top"]) <= LINE_TOLERANCE:
            lines[-1].append(w)
        else:
            lines.append([w])
    phrased = []  # per line: [(x0, x1, top, bottom, text)]
    for line in lines:
        phrases: list[list[dict]] = []
        for w in sorted(line, key=lambda w: w["x0"]):
            if phrases and w["x0"] - phrases[-1][-1]["x1"] < gap:
                phrases[-1].append(w)
            else:
                phrases.append([w])
        phrased.append([(p[0]["x0"], p[-1]["x1"], min(x["top"] for x in p), max(x["bottom"] for x in p),
                         " ".join(x["text"] for x in p)) for p in phrases])

    out, block = [], []
    for ph in phrased + [[]]:  # sentinel flushes the last block
        if len(ph) >= MIN_ALIGNED_COLUMNS:
            block.append(ph)
            continue
        if len(block) >= MIN_ALIGNED_ROWS:
            # Columns = x ranges covered by phrases; gaps nobody crosses split them.
            spans = sorted((p[0], p[1]) for row in block for p in row)
            cols = [list(spans[0])]
            for a, b in spans[1:]:
                if a <= cols[-1][1]:
                    cols[-1][1] = max(cols[-1][1], b)
                else:
                    cols.append([a, b])
            # A left-aligned header over right-aligned numbers leaves two non-overlapping
            # ranges for one column: merge the closest neighbours down to the usual width.
            width = Counter(len(row) for row in block).most_common(1)[0][0]
            while len(cols) > width:
                i = min(range(len(cols) - 1), key=lambda i: cols[i + 1][0] - cols[i][1])
                cols[i:i + 2] = [[cols[i][0], cols[i + 1][1]]]
            raw = []
            for row in block:
                cells = [None] * len(cols)
                for x0, x1, _t, _b, text in row:
                    i = next(i for i, (a, b) in enumerate(cols) if a <= (x0 + x1) / 2 <= b)
                    cells[i] = text if cells[i] is None else f"{cells[i]} {text}"
                raw.append(cells)
            bbox = (cols[0][0], min(p[2] for p in block[0]), cols[-1][1], max(p[3] for p in block[-1]))
            out.append((bbox, raw))
        block = []
    return out


# ── assembly ───────────────────────────────────────────────────────────────────

def _header(raw_row, base) -> list[str]:
    cols, seen = [], Counter()
    for i, c in enumerate(raw_row):
        name = _clean(c, base) or f"col_{i + 1}"
        seen[name] += 1
        cols.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return cols


def _strip_repeated_lines(pages_lines: list[list[str]]) -> list[list[str]]:
    """Drop running headers/footers: lines on ≥2 pages and ≥half of them (first copy kept)."""
    if len(pages_lines) < 2:
        return pages_lines
    counts = Counter(line for lines in pages_lines for line in set(lines))
    repeated = {l for l, n in counts.items() if n >= 2 and n >= len(pages_lines) / 2}
    seen, out = set(), []
    for lines in pages_lines:
        keep = []
        for line in lines:
            if line in repeated:
                if line in seen:
                    continue
                seen.add(line)
            keep.append(line)
        out.append(keep)
    return out


def read_pdf(path, pages=None, direction: str = "auto") -> Document:
    started = time.perf_counter()
    path = Path(path)
    try:
        pdf = pdfplumber.open(path)
    except Exception as exc:  # pdfminer raises a zoo of types; classify by name
        code = "ENCRYPTED_PDF" if "password" in type(exc).__name__.lower() else "UNREADABLE_PDF"
        raise DocsiftError(code, str(exc) or type(exc).__name__, file=str(path)) from exc
    with pdf:
        total = len(pdf.pages)
        wanted = parse_pages(pages, total)
        plumbed = [pdf.pages[p - 1] for p in wanted]
        all_text = "".join(c["text"] for pg in plumbed for c in pg.chars)
        if not all_text.strip():
            raise DocsiftError("SCANNED_PDF", "no text layer on the requested pages; OCR it first",
                               pages=wanted, images=sum(len(pg.images) for pg in plumbed))
        base = _direction(all_text) if direction == "auto" else direction

        found, page_lines, scanned, broken = [], [], [], Counter()
        for pno, page in zip(wanted, plumbed):
            chars = page.chars
            if not "".join(c["text"] for c in chars).strip():
                scanned.append(pno)
                page_lines.append([])
                continue
            bad = sum(bool(_CID_RE.fullmatch(c["text"]) or _BAD_CHAR_RE.search(c["text"])) for c in chars)
            if bad:
                broken[pno] = bad
            tables = [(b, r, "ruled") for b, r in _ruled_tables(page)]
            rest = page
            for bbox, _r, _m in tables:
                rest = rest.outside_bbox(bbox)
            aligned = [(b, r, "aligned") for b, r in _aligned_tables(rest)]
            for bbox, _r, _m in aligned:
                rest = rest.outside_bbox(bbox)
            page_tables = sorted(tables + aligned, key=lambda t: t[0][1])  # top to bottom
            for i, (_bbox, raw, method) in enumerate(page_tables):
                found.append((pno, i == 0, i == len(page_tables) - 1, raw, method))
            text = rest.extract_text() or ""
            page_lines.append([c for c in (_clean(line, base) for line in text.split("\n")) if c])

        result: list[Table] = []
        last_was_last = False
        for pno, first_on_page, last_on_page, raw, method in found:
            if base == "rtl":
                raw = [list(reversed(r)) for r in raw]  # columns were read left→right
            header = _header(raw[0], base)
            rows = [[_clean(c, base) for c in r] for r in raw[1:]]
            prev = result[-1] if result else None
            if prev and prev.pages[-1] == pno - 1 and len(header) == len(prev.columns) and prev.method == method:
                if header == prev.columns:  # repeated header on the next page
                    prev.rows += rows
                    prev.pages.append(pno)
                    last_was_last = last_on_page
                    continue
                if first_on_page and last_was_last:
                    # ponytail: headerless continuation = same width, same method, last table of the
                    # previous page and first of this one. Two different same-width tables split by a
                    # page break get merged; pass pages= to read them separately.
                    prev.rows += [[_clean(c, base) for c in raw[0]], *rows]
                    prev.pages.append(pno)
                    last_was_last = last_on_page
                    continue
            result.append(Table(f"table_{len(result) + 1}", header, rows, method, [pno]))
            last_was_last = last_on_page
        for t in result:
            t.rows = [r for r in t.rows if any(c is not None for c in r)]

    warnings = []
    if scanned:
        warnings.append({"code": "SCANNED_PAGES", "message": f"no text layer on pages {scanned}; OCR needed",
                         "pages": scanned})
    if broken:
        warnings.append({"code": "BROKEN_FONT_MAPPING",
                         "message": f"{sum(broken.values())} glyphs without a Unicode mapping (shown as �) "
                                    f"on pages {sorted(broken)}; re-export the PDF with embedded Unicode fonts",
                         "pages": sorted(broken)})
    text = "\n".join(line for lines in _strip_repeated_lines(page_lines) for line in lines)
    return Document(
        source=path.name, kind="pdf", tables=result, text=text, pages=total, direction=base, warnings=warnings,
        stats={"backend": "local", "pages_read": len(wanted), "tables": len(result),
               "rows": sum(len(t.rows) for t in result), "methods": dict(Counter(t.method for t in result)),
               "seconds": round(time.perf_counter() - started, 3)},
    )
