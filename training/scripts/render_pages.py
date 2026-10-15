"""Render sArabodhinI PDFs to data volumes (GT deferred).

Inputs:
  /media/.../sArabodhinI/01.pdf (630 pp) = Vol.1 (00-22)
  /media/.../sArabodhinI/02.pdf (616 pp) = Vol.2 (23-32)

Outputs (canonical, GT later):
  data/RTS_sb_01/pages_raw/pXXXX.jpg + data/RTS_sb_01/metadata/pages.csv
  data/RTS_sb_02/pages_raw/pXXXX.jpg + data/RTS_sb_02/metadata/pages.csv
Image paths in CSV are repo-relative (e.g. data/RTS_sb_01/pages_raw/p0031.jpg).
Printed-page offsets drift after unnumbered plates; stored printed is best-effort
(pdf_index - OFFSET), footer digits win when available. Front matter (i < OFFSET) blank.

Use --vol 01|02|both with --subset 0:70 for chunks (full render ~3s/pg; run chunks to avoid timeouts).
DPI 150 pilot; 200-300 for final training.
"""
import argparse
import csv
import pathlib
import pymupdf

BASE = pathlib.Path(__file__).resolve().parents[1]
REPO = BASE.parent if BASE.name == "training" else BASE
PDF_01 = "/media/vvasuki/vData/text/granthasangrahaH/AgamaH/vaiShNavaH/shrIvaiShNavaH/venkaTanAtha/rts/sArabodhinI/01.pdf"
PDF_02 = "/media/vvasuki/vData/text/granthasangrahaH/AgamaH/vaiShNavaH/shrIvaiShNavaH/venkaTanAtha/rts/sArabodhinI/02.pdf"
OFFSET_01 = 30  # pdf_index - OFFSET = printed (tune after visual check; front matter < OFFSET)
OFFSET_02 = None  # to be probed; 02.pdf continues pagination - detect from footer OCR or manual

def render_one(pdf_path, out_dir, offset, dpi, subset=None, prefix="", fmt="jpg", jpg_quality=85):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        repo_root = pathlib.Path(__file__).resolve().parents[2]  # training/scripts -> repo
    except Exception:
        repo_root = out_dir
    doc = pymupdf.open(pdf_path)
    n = len(doc)
    if subset:
        lo, hi = subset
        indices = range(max(0, lo), min(n, hi))
    else:
        indices = range(n)
    rows = []
    for i in indices:
        page = doc[i]
        pix = page.get_pixmap(dpi=dpi)
        # Save JPEG (small, shareable) - standard for OCR training; keep RGB
        fname = f"{prefix}p{i:04d}.{fmt}"
        out_path = out_dir / fname
        if not out_path.exists():
            if fmt == "jpg":
                pix.save(str(out_path), jpg_quality=jpg_quality)
            else:
                pix.save(str(out_path))
        printed = "" if (offset is None or i < offset) else str(i - offset)
        try:
            rel = str(out_path.relative_to(repo_root))
        except Exception:
            rel = str(out_path)
        rows.append({"pdf_index": i, "printed_page": printed, "image": rel, "w": pix.width, "h": pix.height})
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--vol", choices=["01", "02", "both"], default="both")
    ap.add_argument("--subset", type=str, default="", help="e.g. 0:70 (applies to chosen vol(s))")
    ap.add_argument("--subset01", type=str, default="", help="legacy; prefer --vol 01 --subset A:B")
    ap.add_argument("--subset02", type=str, default="", help="legacy; prefer --vol 02 --subset A:B")
    ap.add_argument("--offset01", type=int, default=OFFSET_01)
    ap.add_argument("--offset02", type=int, default=-1, help="-1 = unknown, leave printed blank")
    args = ap.parse_args()

    def parse_subset(s):
        if not s:
            return None
        a, b = s.split(":")
        return (int(a), int(b))

    vols = ["01", "02"] if args.vol == "both" else [args.vol]
    for vol in vols:
        pdf = PDF_01 if vol == "01" else PDF_02
        offset = args.offset01 if vol == "01" else (None if args.offset02 < 0 else args.offset02)
        subset = parse_subset(args.subset) or parse_subset(args.subset01 if vol == "01" else args.subset02)
        out_dir = REPO / "data" / f"RTS_sb_{vol}" / "pages_raw"
        rows = render_one(pdf, out_dir, offset, args.dpi, subset)
        fixed = [{"pdf_index": r["pdf_index"], "printed_page": r["printed_page"],
                  "image": r["image"], "w": r["w"], "h": r["h"]} for r in rows]
        meta = REPO / "data" / f"RTS_sb_{vol}" / "metadata" / "pages.csv"
        meta.parent.mkdir(parents=True, exist_ok=True)
        # merge with existing rows (chunked runs): keep max set by pdf_index
        existing = {}
        if meta.exists():
            import csv as _csv
            with open(meta, encoding="utf-8") as f:
                for row in _csv.DictReader(f):
                    existing[int(row["pdf_index"])] = row
        for r in fixed:
            existing[r["pdf_index"]] = {k: str(v) for k, v in r.items()}
        with open(meta, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["pdf_index", "printed_page", "image", "w", "h"])
            w.writeheader()
            for i in sorted(existing):
                w.writerow(existing[i])
        print(f"{vol}: {len(fixed)} pages this chunk, {len(existing)} total in {meta} (dpi={args.dpi})")

if __name__ == "__main__":
    main()
