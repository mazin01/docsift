"""Generate the synthetic test PDFs and their ground-truth CSVs.

All data is made up and deterministic. HTML fixtures are printed to PDF with a
headless Chromium browser (Edge or Chrome), which writes proper Unicode text
for Arabic. The "visual order" fixture is drawn with reportlab from pre-shaped
glyphs to reproduce PDFs from older producers. Needs: reportlab,
arabic-reshaper, python-bidi, and Edge/Chrome. The Amiri font (OFL) is
downloaded into fixtures/_fonts on first run.

    python fixtures/generate.py
"""
from __future__ import annotations

import csv
import io
import random
import shutil
import subprocess
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "pdf"
FONTS = HERE / "_fonts"
FONT_URL = "https://github.com/google/fonts/raw/main/ofl/amiri/{}"
BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "google-chrome", "chromium", "chromium-browser", "msedge",
]

rng = random.Random(7)

EN_ITEMS = ["Printer paper A4", "Blue ink cartridge", "Stapler", "Desk lamp", "USB cable 2m",
            "Notebook pack", "Whiteboard markers", "Office chair", "Monitor stand", "Paper clips"]
AR_ITEMS = ["ورق طباعة", "حبر أزرق", "دباسة", "مصباح مكتب", "كابل يو إس بي",
            "دفاتر", "أقلام سبورة", "كرسي مكتب", "حامل شاشة", "مشابك ورق"]
AR_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


def font(name: str) -> Path:
    FONTS.mkdir(exist_ok=True)
    path = FONTS / name
    if not path.exists():
        urllib.request.urlretrieve(FONT_URL.format(name), path)
    return path


def browser() -> str:
    for b in BROWSERS:
        if Path(b).exists() or shutil.which(b):
            return b
    raise SystemExit("Need Edge or Chrome to print HTML fixtures to PDF.")


def write_truth(name: str, tables: list[list[list[str]]]) -> None:
    """Ground truth: one CSV per table, header row first."""
    for i, table in enumerate(tables, 1):
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerows(table)
        (OUT / f"{name}.table{i}.csv").write_text(buf.getvalue(), encoding="utf-8")


def html_to_pdf(name: str, body: str, rtl: bool = False) -> None:
    regular, bold = font("Amiri-Regular.ttf").as_uri(), font("Amiri-Bold.ttf").as_uri()
    html = f"""<!doctype html><html dir="{'rtl' if rtl else 'ltr'}"><head><meta charset="utf-8"><style>
@font-face {{ font-family: Amiri; src: url("{regular}"); }}
@font-face {{ font-family: Amiri; font-weight: bold; src: url("{bold}"); }}
@page {{ size: A4; margin: 18mm; }}
body {{ font-family: Amiri; font-size: 11pt; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
.ruled td, .ruled th {{ border: 1px solid #333; padding: 3px 6px; }}
.plain td, .plain th {{ padding: 2px 6px; }}
th {{ text-align: start; }} td.n {{ text-align: end; }}
thead {{ display: table-header-group; }} tr {{ break-inside: avoid; }}
</style></head><body>{body}</body></html>"""
    src = OUT / f"_{name}.html"
    src.write_text(html, encoding="utf-8")
    subprocess.run([browser(), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    "--print-to-pdf-no-header", f"--print-to-pdf={OUT / (name + '.pdf')}", src.as_uri()],
                   check=True, capture_output=True, timeout=120)
    src.unlink()


def html_table(rows: list[list[str]], cls: str, numeric_cols: set[int]) -> str:
    head = "".join(f"<th>{h}</th>" for h in rows[0])
    body = "".join(
        "<tr>" + "".join(f'<td class="{"n" if i in numeric_cols else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>"
        for r in rows[1:]
    )
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def ledger(n: int, items: list[str], header: list[str], arabic_digits: bool = False) -> list[list[str]]:
    rows = [header]
    for i in range(n):
        qty = rng.randint(1, 40)
        price = rng.randint(150, 9_000) / 100
        amount = f"{qty * price:,.2f}"
        date = f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        ref = f"ORD-{1000 + i}"
        if arabic_digits:
            amount = amount.translate(AR_DIGITS)
        rows.append([date, ref, rng.choice(items), str(qty), amount])
    return rows


EN_HEADER = ["Date", "Reference", "Description", "Qty", "Amount"]
AR_HEADER = ["التاريخ", "المرجع", "البيان", "الكمية", "المبلغ"]


def ruled_en() -> None:
    t = ledger(12, EN_ITEMS, EN_HEADER)
    html_to_pdf("ruled_en", "<h2>Order ledger</h2><p>Branch: North. Period: 2025.</p>"
                + html_table(t, "ruled", {3, 4}) + "<p>Prepared automatically. Figures in USD.</p>")
    write_truth("ruled_en", [t])


def borderless_en() -> None:
    t = ledger(15, EN_ITEMS, EN_HEADER)
    html_to_pdf("borderless_en", "<h2>Account activity</h2><p>Statement for customer 4471.</p>"
                + html_table(t, "plain", {3, 4}))
    write_truth("borderless_en", [t])


def multipage_en() -> None:
    t = ledger(90, EN_ITEMS, EN_HEADER)
    html_to_pdf("multipage_en", "<h2>Annual order register</h2>" + html_table(t, "ruled", {3, 4}))
    write_truth("multipage_en", [t])


def ruled_ar() -> None:
    t = ledger(12, AR_ITEMS, AR_HEADER, arabic_digits=True)
    html_to_pdf("ruled_ar", "<h2>سجل الطلبات</h2><p>الفرع: الشمال. الفترة: ٢٠٢٥.</p>"
                + html_table(t, "ruled", {3, 4}), rtl=True)
    write_truth("ruled_ar", [t])


def invoice_mixed() -> None:
    items = [["Item / الصنف", "Qty / الكمية", "Unit price / سعر الوحدة", "Total / الإجمالي"]]
    for en, ar in rng.sample(list(zip(EN_ITEMS, AR_ITEMS)), 5):
        qty, price = rng.randint(1, 10), rng.randint(500, 20_000) / 100
        items.append([f"{en} - {ar}", str(qty), f"{price:,.2f}", f"{qty * price:,.2f}"])
    meta = (
        "<h2>Invoice / فاتورة</h2>"
        "<p>Invoice no. / رقم الفاتورة: INV-2025-0042</p>"
        "<p>Date / التاريخ: 2025-03-14 (١٣ رمضان ١٤٤٦)</p>"
        "<p>Customer / العميل: Example Trading Co. / شركة المثال التجارية</p>"
    )
    html_to_pdf("invoice_mixed", meta + html_table(items, "ruled", {1, 2, 3})
                + "<p>Thank you for your business. شكراً لتعاملكم معنا.</p>")
    write_truth("invoice_mixed", [items])


def visual_order_ar() -> None:
    """Shaped glyphs in visual order + ruled grid, like older PDF producers emit."""
    import arabic_reshaper
    from bidi.algorithm import get_display
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    pdfmetrics.registerFont(TTFont("Amiri", str(font("Amiri-Regular.ttf"))))
    t = ledger(8, AR_ITEMS, AR_HEADER)
    c = canvas.Canvas(str(OUT / "visual_order_ar.pdf"), pagesize=A4, invariant=1)
    w, h = A4
    visual = lambda s: get_display(arabic_reshaper.reshape(s))  # noqa: E731
    c.setFont("Amiri", 16)
    c.drawRightString(w - 50, h - 60, visual("كشف الطلبات"))
    c.setFont("Amiri", 11)
    col_w, row_h, top = (w - 100) / 5, 22, h - 100
    for r, row in enumerate(t):
        y = top - r * row_h
        for i, cell in enumerate(row):
            x_right = w - 50 - i * col_w  # RTL: first column on the right
            c.rect(x_right - col_w, y - row_h, col_w, row_h)
            c.drawRightString(x_right - 5, y - 15, visual(cell))
    c.save()
    write_truth("visual_order_ar", [t])


def scanned() -> None:
    """Image-only page: no text layer at all."""
    from PIL import Image, ImageDraw
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    img = Image.new("RGB", (1240, 1754), "white")
    d = ImageDraw.Draw(img)
    for i in range(12):
        d.text((100, 120 + i * 40), f"Scanned line {i + 1}   2025-01-{i + 1:02d}   {100 * (i + 1)}.00", fill="black")
    c = canvas.Canvas(str(OUT / "scanned.pdf"), pagesize=A4, invariant=1)
    c.drawImage(ImageReader(img), 0, 0, *A4)
    c.save()


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for make in (ruled_en, borderless_en, multipage_en, ruled_ar, invoice_mixed, visual_order_ar, scanned):
        make()
        print("ok", make.__name__)
