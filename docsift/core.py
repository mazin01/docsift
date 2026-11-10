"""Document model, column profiling, rendering and token counting."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime

import nuqta as nq

FORMATS = ("markdown", "tsv", "json")


class DocsiftError(Exception):
    """Failure with a stable machine-readable `code` (e.g. SCANNED_PDF)."""

    def __init__(self, code: str, message: str, **details):
        super().__init__(f"{code}: {message}")
        self.code, self.details = code, details


# ── profiling ──────────────────────────────────────────────────────────────────

_DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
                 "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S")


def parse_date(value: str) -> date | None:
    """ISO or day-first dates ("14/03/2025"). Day-first is a deliberate choice: ambiguous
    US-style dates (03/14/2025) only parse when the day can't be a month."""
    s = nq.normalize_digits(value.strip())
    for fmt in _DATE_FORMATS + ("%m/%d/%Y",):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _num(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")


def profile_column(name: str, values: list) -> dict:
    """Type, null/unique counts and range/sum for one column of raw string cells."""
    present = [v for v in values if v not in (None, "")]
    out: dict = {"name": name, "type": "empty", "nulls": len(values) - len(present), "unique": len(set(present))}
    if not present:
        return out
    bools = [nq.parse_bool(v) for v in present]
    nums = [nq.to_number(v) for v in present]
    ok = [n for n in nums if n is not None]
    if all(b is not None for b in bools) and len(ok) < len(present):
        out["type"] = "boolean"
    elif len(ok) / len(present) >= 0.85:
        out.update(type="number", min=min(ok), max=max(ok), sum=round(math.fsum(ok), 6))
    elif sum(nq.looks_hijri(v) for v in present) / len(present) >= 0.75:
        out["type"] = "date_hijri"
    else:
        dates = [d for d in (parse_date(v) for v in present) if d]
        if len(dates) / len(present) >= 0.75:
            out.update(type="date", min=min(dates).isoformat(), max=max(dates).isoformat())
        else:
            out["type"] = "text"
    out["examples"] = list(dict.fromkeys(present))[:3]
    return out


def _profile_line(p: dict) -> str:
    bits = [p["type"]]
    if p["type"] == "number":
        bits.append(f"{_num(p['min'])}..{_num(p['max'])}, sum {_num(p['sum'])}")
    elif p["type"] == "date":
        bits.append(f"{p['min']}..{p['max']}")
    elif p["type"] in ("text", "date_hijri", "boolean"):
        bits.append(f"{p['unique']} unique, e.g. " + " | ".join(str(e)[:40] for e in p["examples"]))
    if p["nulls"]:
        bits.append(f"{p['nulls']} empty")
    return f"- {p['name']}: " + ", ".join(bits)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


# ── model ──────────────────────────────────────────────────────────────────────

@dataclass
class Table:
    name: str
    columns: list[str]
    rows: list[list]                  # cells as cleaned strings or None
    method: str                       # ruled | aligned | llm | csv | excel
    pages: list[int] = field(default_factory=list)

    def profile(self) -> list[dict]:
        return [profile_column(c, [r[i] for r in self.rows]) for i, c in enumerate(self.columns)]

    def records(self) -> list[dict]:
        return [dict(zip(self.columns, r)) for r in self.rows]

    def to_pandas(self, typed: bool = False):
        """DataFrame of the cells; typed=True converts number/date columns."""
        import pandas as pd

        df = pd.DataFrame(self.rows, columns=self.columns, dtype=object)
        if typed:
            for p in self.profile():
                col = p["name"]
                if p["type"] == "number":
                    df[col] = [nq.to_number(v) for v in df[col]]
                elif p["type"] == "date":
                    df[col] = pd.to_datetime([parse_date(v) if v else None for v in df[col]])
        return df

    def to_parquet(self, path) -> None:
        self.to_pandas(typed=True).to_parquet(path, index=False)

    def _rows_block(self, fmt: str, n: int | None) -> str:
        rows = self.rows if n is None else self.rows[:n]
        cell = lambda v: "" if v is None else str(v)  # noqa: E731
        if fmt == "tsv":
            clean = lambda v: cell(v).replace("\t", " ").replace("\n", " ")  # noqa: E731
            return "\n".join("\t".join(clean(v) for v in r) for r in [self.columns, *rows])
        esc = lambda v: cell(v).replace("|", "\\|").replace("\n", " ")  # noqa: E731
        lines = ["| " + " | ".join(esc(c) for c in self.columns) + " |",
                 "|" + "---|" * len(self.columns)]
        lines += ["| " + " | ".join(esc(v) for v in r) + " |" for r in rows]
        return "\n".join(lines)

    def render(self, fmt: str = "markdown", rows: int | None = 5, profile: bool = True) -> str:
        where = f" · pages {self.pages[0]}-{self.pages[-1]}" if len(self.pages) > 1 else \
            f" · page {self.pages[0]}" if self.pages else ""
        out = [f"## {self.name}{where} · {len(self.rows)} rows × {len(self.columns)} columns"]
        if profile:
            out += ["columns:", *(_profile_line(p) for p in self.profile())]
        if rows != 0:
            fence = "```tsv\n{}\n```" if fmt == "tsv" else "{}"
            out.append(fence.format(self._rows_block(fmt, rows)))
            shown = len(self.rows) if rows is None else min(rows, len(self.rows))
            if shown < len(self.rows):
                out.append(f"({len(self.rows) - shown} more rows)")
        return "\n".join(out)


@dataclass
class Document:
    source: str
    kind: str                          # pdf | csv | xlsx | xls
    tables: list[Table]
    text: str = ""                     # narrative text outside tables (PDF)
    pages: int | None = None
    direction: str = "ltr"
    warnings: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def to_dict(self, rows: int | None = None, profile: bool = True) -> dict:
        return {
            "source": self.source, "kind": self.kind, "pages": self.pages, "direction": self.direction,
            "text": self.text, "warnings": self.warnings, "stats": self.stats,
            "tables": [{
                "name": t.name, "method": t.method, "pages": t.pages, "row_count": len(t.rows),
                **({"profile": t.profile()} if profile else {}),
                "rows": t.records() if rows is None else t.records()[:rows],
            } for t in self.tables],
        }

    def render(self, fmt: str = "markdown", rows: int | None = 5, profile: bool = True,
               text: bool = True, max_tokens: int | None = None) -> str:
        """LLM-ready context. rows=None means all rows; max_tokens trims rows, then text."""
        if fmt not in FORMATS:
            raise ValueError(f"fmt must be one of {FORMATS}, got {fmt!r}")
        out = self._render(fmt, rows, profile, text)
        if max_tokens is None or count_tokens(out) <= max_tokens:
            return out
        most = max((len(t.rows) for t in self.tables), default=0) if rows is None else rows
        lo, hi = 0, most  # largest row count that fits, by bisection
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if count_tokens(self._render(fmt, mid, profile, text)) <= max_tokens:
                lo = mid
            else:
                hi = mid - 1
        out = self._render(fmt, lo, profile, text)
        if count_tokens(out) > max_tokens and text and self.text:
            keep = self.text
            while keep and count_tokens(self._render(fmt, lo, profile, text, keep + " [truncated]")) > max_tokens:
                keep = keep[: int(len(keep) * 0.8)]
            out = self._render(fmt, lo, profile, text, (keep + " [truncated]") if keep else "")
        return out

    def _render(self, fmt, rows, profile, text, text_override=None) -> str:
        if fmt == "json":
            d = self.to_dict(rows=rows, profile=profile)
            d.pop("stats")
            if not text:
                d.pop("text")
            elif text_override is not None:
                d["text"] = text_override
            return json.dumps(d, ensure_ascii=False)
        head = [f"# {self.source}",
                " · ".join(filter(None, [self.kind, _plural(self.pages, "page") if self.pages else "",
                                         _plural(len(self.tables), "table"), self.direction]))]
        for w in self.warnings:
            head.append(f"warning {w['code']}: {w['message']}")
        body = text_override if text_override is not None else self.text
        parts = ["\n".join(head)]
        if text and body:
            parts.append(body)
        parts += [t.render(fmt, rows, profile) for t in self.tables]
        return "\n\n".join(parts)

    def token_report(self) -> dict:
        """Tokens needed for the full content in each format, plus the default context."""
        report = {f"all_rows_{f}": count_tokens(self.render(f, rows=None, profile=False)) for f in FORMATS}
        report["context_default"] = count_tokens(self.render())
        report["estimator"] = TOKEN_ESTIMATOR
        return report


# ── tokens ─────────────────────────────────────────────────────────────────────

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("o200k_base")
    TOKEN_ESTIMATOR = "tiktoken o200k_base"

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text, disallowed_special=()))
except ImportError:  # ponytail: rough fallback; Arabic tokenizes worse than 4 chars/token
    TOKEN_ESTIMATOR = "chars/4 estimate"

    def count_tokens(text: str) -> int:
        return math.ceil(len(text) / 4)
