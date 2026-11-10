"""docsift FILE [--format markdown|tsv|json] [--rows N|all] [--max-tokens N] [--pages 1-3] [--stats]"""
import argparse
import json
import sys

from . import DocsiftError, read


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="docsift", description="PDF/CSV/Excel → compact LLM-ready context")
    ap.add_argument("file")
    ap.add_argument("--format", default="markdown", choices=["markdown", "tsv", "json"])
    ap.add_argument("--rows", default="5", help="sample rows per table, or 'all' (default 5)")
    ap.add_argument("--max-tokens", type=int, help="trim rows, then text, to fit this budget")
    ap.add_argument("--pages", help='PDF pages, e.g. "1-3,5"')
    ap.add_argument("--direction", default="auto", choices=["auto", "rtl", "ltr"])
    ap.add_argument("--no-profile", action="store_true", help="omit the per-column profile")
    ap.add_argument("--stats", action="store_true", help="print extraction stats + token report to stderr")
    a = ap.parse_args(argv)
    try:
        doc = read(a.file, pages=a.pages, direction=a.direction)
    except DocsiftError as e:
        print(json.dumps({"error": e.code, "message": str(e), **e.details}, ensure_ascii=False), file=sys.stderr)
        return 2
    rows = None if a.rows == "all" else int(a.rows)
    sys.stdout.reconfigure(encoding="utf-8")
    print(doc.render(a.format, rows=rows, profile=not a.no_profile, max_tokens=a.max_tokens))
    if a.stats:
        sys.stderr.reconfigure(encoding="utf-8")
        print(json.dumps({**doc.stats, "tokens": doc.token_report()}, ensure_ascii=False, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
