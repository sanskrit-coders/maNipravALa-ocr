#!/usr/bin/env python3
"""Audit the line segmentation used by apply_ocr_pdf/pipeline (segment_lines).

For each page: run segment_lines, then report
  - n_segs, pitch
  - containment of Tesseract level-4 line boxes in segments (center and fully-inside)
  - ink coverage: fraction of dark pixels that fall inside some segment
  - suspicious segments: tall enough to be a merged pair, or x-projection with a
    wide interior empty gap (possible two-column lines)
Writes a numbered overlay PNG per page (segment boxes) for visual inspection.

Usage:
  python3 training/scripts/script_cls/audit_lines.py \
      --pdf /path.pdf --pages 100 --out-dir /tmp/opencode/line_audit
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from apply_ocr_pdf import segment_lines, tesseract_boxes  # noqa: E402


def audit_page(img, psm=6):
    from PIL import ImageDraw

    segs = segment_lines(img, psm)
    a = np.asarray(img)
    dark = a < 128
    H, W = dark.shape

    hs = sorted(b[3] - b[1] for b in (tesseract_boxes_from(img, psm) or [[0, 0, 0, 1]]))
    pitch = hs[len(hs) // 2] if hs else 40

    boxes = tesseract_boxes_from(img, psm)
    inside_center = inside_full = 0
    for x0, y0, x1, y1 in boxes:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        c = any(sx0 <= cx <= sx1 and sy0 <= cy <= sy1 for sx0, sy0, sx1, sy1 in segs)
        f = any(sx0 <= x0 and sx1 >= x1 and sy0 <= y0 and sy1 >= y1 for sx0, sy0, sx1, sy1 in segs)
        inside_center += int(c)
        inside_full += int(f)

    covered = np.zeros(H, dtype=bool)
    for x0, y0, x1, y1 in segs:
        covered[y0:y1] = True
    ink_total = int(dark.sum())
    ink_cov = int(dark[covered].sum())
    ink_missed_rows = int(dark[~covered].sum())

    suspicious_tall, suspicious_gap = [], []
    for i, (x0, y0, x1, y1) in enumerate(segs):
        h = y1 - y0
        if h > 1.5 * pitch:
            sub = dark[y0:y1, x0:x1].sum(axis=1)
            floor = np.percentile(sub, 10)
            peak = np.percentile(sub, 80)
            if peak > 0 and floor < 0.35 * peak:
                suspicious_tall.append((i, h, round(h / pitch, 2)))
        prof = dark[y0:y1, x0:x1].sum(axis=0)
        xs = np.nonzero(prof)[0]
        if len(xs) > 0.5 * (x1 - x0):
            gaps = np.nonzero(prof[xs[0]:xs[-1]] == 0)[0]
            if len(gaps):
                runs = np.split(gaps, np.nonzero(np.diff(gaps) > 1)[0] + 1)
                runs = [r for r in runs if len(r) > 0.06 * W]
                if runs and xs[-1] - xs[0] > 0.5 * W:
                    suspicious_gap.append((i, max(len(r) for r in runs)))

    return {
        "n_segs": len(segs),
        "pitch": pitch,
        "n_tess_boxes": len(boxes),
        "tess_center_in_seg": inside_center,
        "tess_full_in_seg": inside_full,
        "ink_total": ink_total,
        "ink_covered_frac": ink_cov / ink_total if ink_total else 1.0,
        "ink_missed": ink_missed_rows,
        "suspicious_merged": suspicious_tall,
        "suspicious_twocol": suspicious_gap,
        "segs": segs,
    }


def tesseract_boxes_from(img, psm):
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        img.save(f.name)
        return tesseract_boxes(f.name, psm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--pages", default="100")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out-dir", default="/tmp/opencode/line_audit")
    args = ap.parse_args()

    import json

    import pymupdf
    from PIL import Image, ImageDraw

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(args.pdf)
    indices = []
    for part in args.pages.split(","):
        if ":" in part:
            lo, hi = (int(x) if x else None for x in part.split(":"))
            indices += list(range(lo or 0, hi if hi is not None else len(doc)))
        else:
            indices.append(int(part))

    for i in indices:
        page = doc[i]
        pix = page.get_pixmap(dpi=args.dpi, colorspace=pymupdf.csGRAY)
        img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
        res = audit_page(img)
        segs = res.pop("segs")
        print(f"page {i}: {json.dumps(res, ensure_ascii=False)}", flush=True)
        ov = img.convert("RGB")
        dr = ImageDraw.Draw(ov)
        for k, (x0, y0, x1, y1) in enumerate(segs):
            color = (255, 0, 0)
            if res["suspicious_merged"] and k in {s[0] for s in res["suspicious_merged"]}:
                color = (255, 128, 0)
            if res["suspicious_twocol"] and k in {s[0] for s in res["suspicious_twocol"]}:
                color = (0, 0, 255)
            dr.rectangle([x0, y0, x1, y1], outline=color, width=3)
            dr.text((max(0, x0 + 4), max(0, y0 - 14)), str(k), fill=(0, 160, 0))
        ov.save(out / f"p{i:04d}.png")


if __name__ == "__main__":
    main()
