"""Benchmark docsift on the synthetic fixtures. Writes bench/results.md.

    python bench/run.py

Tokens: tiktoken o200k_base (an OpenAI tokenizer; other models differ, the
ratios hold roughly). Accuracy: share of ground-truth cells (header included)
reproduced exactly, after normalize_key on both sides.
"""
import csv
import json
import time
from pathlib import Path

import nuqta as nq
import pdfplumber

import docsift
from docsift import count_tokens

PDF = Path(__file__).parent.parent / "fixtures" / "pdf"
OUT = Path(__file__).parent / "results.md"


def cell_accuracy(tables: list[list[list]], truth: list[list[list[str]]]) -> float:
    total = sum(len(r) for t in truth for r in t)
    ok = 0
    for got, want in zip(tables, truth):
        for gr, wr in zip(got, want):
            ok += sum(nq.normalize_key(a) == nq.normalize_key(b) for a, b in zip(gr, wr))
    return ok / total


def baseline(path: Path) -> tuple[str, list[list[list]]]:
    """What you get from pdfplumber out of the box: extract_text() + extract_tables()."""
    with pdfplumber.open(path) as pdf:
        text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        tables = [t for p in pdf.pages for t in p.extract_tables()]
    if len(tables) > 1 and all(len(t[0]) == len(tables[0][0]) for t in tables):
        tables = [tables[0] + [r for t in tables[1:] for r in t[1:]]]  # be fair: stitch pages
    return text, tables


rows = []
for pdf in sorted(PDF.glob("*.pdf")):
    truth = [list(csv.reader(p.open(encoding="utf-8"))) for p in sorted(PDF.glob(f"{pdf.stem}.table*.csv"))]
    if not truth:
        continue
    raw_text, raw_tables = baseline(pdf)
    t0 = time.perf_counter()
    doc = docsift.read(pdf)
    secs = time.perf_counter() - t0
    records = [dict(zip(t.columns, r)) for t in doc.tables for r in t.rows]
    rows.append({
        "file": pdf.name,
        "pages": doc.pages,
        "cells": sum(len(r) for t in truth for r in t),
        "acc_baseline": cell_accuracy(raw_tables, truth),
        "acc_docsift": cell_accuracy([[t.columns, *t.rows] for t in doc.tables], truth),
        "tok_raw_text": count_tokens(raw_text),
        "tok_json_records": count_tokens(json.dumps(records, ensure_ascii=False)),
        "tok_markdown": count_tokens(doc.render("markdown", rows=None, profile=False)),
        "tok_tsv": count_tokens(doc.render("tsv", rows=None, profile=False)),
        "tok_context": count_tokens(doc.render()),
        "seconds": secs,
    })

pct = lambda x: f"{x:.0%}"  # noqa: E731
lines = [
    "| fixture | pages | cells | pdfplumber accuracy | docsift accuracy | seconds |",
    "|---|---:|---:|---:|---:|---:|",
    *(f"| {r['file']} | {r['pages']} | {r['cells']} | {pct(r['acc_baseline'])} | {pct(r['acc_docsift'])} | "
      f"{r['seconds']:.2f} |" for r in rows),
    "",
    "| fixture | raw text | JSON records | markdown (all rows) | TSV (all rows) | default context |",
    "|---|---:|---:|---:|---:|---:|",
    *(f"| {r['file']} | {r['tok_raw_text']} | {r['tok_json_records']} | {r['tok_markdown']} | {r['tok_tsv']} | "
      f"{r['tok_context']} |" for r in rows),
]
tot = {k: sum(r[k] for r in rows) for k in rows[0] if k.startswith("tok_")}
lines.append(f"| **total** | {tot['tok_raw_text']} | {tot['tok_json_records']} | {tot['tok_markdown']} | "
             f"{tot['tok_tsv']} | {tot['tok_context']} |")
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
