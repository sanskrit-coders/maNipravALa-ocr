#!/usr/bin/env python3
"""Apply the trained line recognizer to a target PDF (e.g. ra-tra-sA_5-commentaries).

Pipeline per page: pymupdf render (grayscale) -> line segmentation from Tesseract
TSV anchors (level-4 line boxes, deduplicated; boxes taller than 1.5x the line pitch
recursively split at row-darkness local minima) -> crop -> kraken
ImageInputTransforms(1,64,0,1,(16,0)) dewarp -> LineNet inference -> CTC greedy
decode with the TRAIN codec.

Outputs:
  {out}/{pdf_stem}_p{pdf_index:04d}.txt   one line of text per detected line
  {out}/lines.jsonl                       {pdf, page, line_idx, x0,y0,x1,y1, text}

Usage:
  python3 training/scripts/apply_ocr_pdf.py --pdf /path/to.pdf --pages 100:110 \
      --out evaluation/ocr_5comm/pdf1 --crop-dir /tmp/opencode/crops_pdf1 --overlay-dir /tmp/opencode/ovl
"""
import argparse
import csv
import io
import json
import os
import pathlib
import statistics
import subprocess
import sys
import tempfile

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from train_lines2 import SUBSAMPLE, ids_to_text  # noqa: E402

BASE = pathlib.Path(__file__).resolve().parents[1]
REPO = BASE.parent if BASE.name == "training" else BASE

TESS = "/tmp/tess-root/usr/bin/tesseract"
TESSDATA = "/tmp/tess-root/usr/share/tesseract-ocr/5/tessdata/"


def tesseract_boxes(img_path, psm):
    env = dict(os.environ, TESSDATA_PREFIX=TESSDATA, OMP_THREAD_LIMIT="1")
    r = subprocess.run([TESS, str(img_path), "stdout", "-l", "tam+san",
                        "--psm", str(psm), "tsv"],
                       capture_output=True, env=env)
    if r.returncode != 0:
        return []
    boxes = []
    for row in csv.reader(io.StringIO(r.stdout.decode()), delimiter="\t"):
        if len(row) < 12 or row[0] != "4":
            continue
        try:
            left, top, w, h = int(row[6]), int(row[7]), int(row[8]), int(row[9])
        except ValueError:
            continue
        if h >= 25 and w >= 40:
            boxes.append([left, top, left + w, top + h])
    return boxes


def segment_lines(img, psm=6):
    a = np.asarray(img)
    dark = a < 128
    row_dark = dark.sum(axis=1)

    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        img.save(f.name)
        boxes = tesseract_boxes(f.name, psm)
    if not boxes:
        return []

    hs = sorted(b[3] - b[1] for b in boxes)
    med = hs[len(hs) // 2]
    pitch = statistics.median([h for h in hs if h <= 1.3 * med])

    def split_band(b, depth=0):
        x0, y0, x1, y1 = b
        if y1 - y0 <= 1.5 * pitch or depth >= 4:
            return [b]
        sub = row_dark[y0:y1]
        thr = 0.18 * np.percentile(sub, 80)
        cuts = [y0 + k for k in range(2, len(sub) - 2)
                if sub[k] <= thr and sub[k] < sub[k - 1] and sub[k] <= sub[k + 1]]
        out, prev = [], y0
        for c in cuts:
            if c - prev >= 0.7 * pitch and y1 - c >= 0.7 * pitch:
                out.append([x0, prev, x1, c])
                prev = c
        out.append([x0, prev, x1, y1])
        if len(out) <= 1:
            return [b]
        res = []
        for p in out:
            res += split_band(p, depth + 1)
        return res

    boxes.sort(key=lambda b: b[1])
    kept = []
    for b in boxes:
        dup = False
        for k in kept:
            oy = min(k[3], b[3]) - max(k[1], b[1])
            ox = min(k[2], b[2]) - max(k[0], b[0])
            if oy > 0.5 * min(k[3] - k[1], b[3] - b[1]) and \
               ox > 0.5 * min(k[2] - k[0], b[2] - b[0]):
                dup = True
                break
        if not dup:
            kept.append(b)
    segs_tess = []
    for b in kept:
        segs_tess += split_band(b)

    bands = []
    y = 0
    H = a.shape[0]
    while y < H:
        if row_dark[y] <= 2:
            y += 1
            continue
        y0 = y
        while y < H and row_dark[y] > 2:
            y += 1
        bands.append([0, y0, a.shape[1], y])
    segs_proj = []
    for b in bands:
        segs_proj += split_band(b)
    segs_proj = [s for s in segs_proj if s[3] - s[1] >= 0.5 * pitch]

    def coverage(segs):
        return sum(int(dark[y0:y1, x0:x1].sum()) for x0, y0, x1, y1 in segs)

    cov_t, cov_p = coverage(segs_tess), coverage(segs_proj)
    segs = segs_tess if cov_t >= 0.85 * cov_p else segs_proj
    segs.sort(key=lambda b: b[1])
    return [(x0, y0, x1, y1) for x0, y0, x1, y1 in segs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--ckpt", default="training/local/kaggle_v10_best.pt")
    ap.add_argument("--pages", default="", help="pdf_index range lo:hi (python slice); default all")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--psm", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop-dir", default="")
    ap.add_argument("--overlay-dir", default="")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import pymupdf
    from kraken.lib.vgsl import TorchVGSLModel
    from kraken.lib.dataset.utils import ImageInputTransforms

    ckpt = torch.load(REPO / args.ckpt, map_location="cpu", weights_only=False)
    model = TorchVGSLModel(vgsl=ckpt["spec"], mode="recognition")
    net = model.nn
    net.load_state_dict(ckpt["state_dict"])
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    net.to(dev).eval()

    train_meta = json.load(open(REPO / "training/local/lines_cache/train/meta.json"))
    rev = {v: ch for ch, vs in train_meta["c2l"].items() for v in vs}
    n_classes = int(ckpt["spec"].rsplit("O1c", 1)[1].rstrip("]"))
    assert n_classes == max(rev.keys()) + 1, (n_classes, max(rev.keys()))

    trans = ImageInputTransforms(1, 64, 0, 1, (16, 0), valid_norm=True)
    out_dir = REPO / args.out if not pathlib.Path(args.out).is_absolute() else pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = pathlib.Path(args.crop_dir) if args.crop_dir else None
    if crop_dir:
        crop_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(args.pdf)
    if args.pages:
        lo, hi = (int(x) if x else None for x in args.pages.split(":"))
        indices = range(lo or 0, hi if hi is not None else len(doc))
    else:
        indices = range(len(doc))
    stem = pathlib.Path(args.pdf).stem
    overlay_dir = pathlib.Path(args.overlay_dir) if args.overlay_dir else None
    if overlay_dir:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    dump = open(out_dir / "lines.jsonl", "w", encoding="utf-8")

    for i in indices:
        page = doc[i]
        pix = page.get_pixmap(dpi=args.dpi, colorspace=pymupdf.csGRAY)
        img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
        boxes = segment_lines(img, args.psm)
        if overlay_dir:
            ov = img.convert("RGB")
            dr = ImageDraw.Draw(ov)
            for (x0, y0, x1, y1) in boxes:
                dr.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)
            ov.save(overlay_dir / f"p{i:04d}.png")
        if not boxes:
            print(f"page {i}: 0 lines", flush=True)
            continue
        items, metas = [], []
        for k, (x0, y0, x1, y1) in enumerate(boxes):
            X0, Y0 = max(0, x0 - 4), max(0, y0 - 2)
            X1, Y1 = min(img.width, x1 + 4), min(img.height, y1 + 2)
            if X1 - X0 < 20 or Y1 - Y0 < 8:
                continue
            crop = img.crop((X0, Y0, X1, Y1))
            if crop_dir:
                crop.save(crop_dir / f"p{i:04d}_L{k:02d}.png")
            h2 = 128
            s = h2 / crop.height
            im2 = crop.resize((max(1, round(crop.width * s)), h2), Image.LANCZOS)
            x = trans(im2)
            items.append(x)
            metas.append((X0, Y0, X1, Y1, int(x.shape[2])))
        order = sorted(range(len(items)), key=lambda j: metas[j][4])
        texts = [None] * len(items)
        with torch.no_grad():
            for b in range(0, len(order), args.batch):
                idxs = order[b:b + args.batch]
                W = max(items[j].shape[2] for j in idxs)
                stack = torch.zeros(len(idxs), 1, 64, W)
                for r, j in enumerate(idxs):
                    stack[r, 0, :, :items[j].shape[2]] = items[j][0]
                out = net(stack.to(dev))
                lg = (out[0] if isinstance(out, tuple) else out).squeeze(2).permute(2, 0, 1)
                preds = lg.argmax(2)
                for r, j in enumerate(idxs):
                    tlen = max(1, min(preds.shape[0], metas[j][4] // SUBSAMPLE))
                    texts[j] = ids_to_text(rev, preds[:tlen, r].tolist())
        with open(out_dir / f"{stem}_p{i:04d}.txt", "w", encoding="utf-8") as f:
            for j, (x0, y0, x1, y1, w) in enumerate(metas):
                f.write(texts[j] + "\n")
                dump.write(json.dumps({"pdf": stem, "page": i, "line_idx": j,
                                       "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                                       "text": texts[j]}, ensure_ascii=False) + "\n")
        print(f"page {i}: {len(metas)} lines -> {stem}_p{i:04d}.txt", flush=True)
    dump.close()


if __name__ == "__main__":
    main()
