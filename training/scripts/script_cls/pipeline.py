#!/usr/bin/env python3
"""Script-run boxing + faux-page OCR pipeline.

Per target PDF page:
  1. line segmentation (same heuristic as apply_ocr_pdf.py),
  2. word segmentation (ink-gap projection inside each line),
  3. patch-level script classification dev/tam/other (sliding 48px patches),
     splitting mixed words at script transitions,
  4. compose script-pure faux pages (one run per line, wide spacing),
  5. OCR faux pages with Google Drive API (via the doc_curation venv helper),
  6. map OCR lines back to runs and reassemble per original line.

Outputs:
  {out}/{stem}_p{page:04d}.txt    reassembled page text
  {out}/runs.jsonl                per-run records (box, script, pre_space, text)
  {workdir}/faux/*.jpg            faux pages
  {workdir}/overlay/*.jpg         colored box overlay per page

Usage:
  python3 training/scripts/script_cls/pipeline.py --pdf ... --pages 100:102 \
      --out data/RTS_5c_01/ocr_scriptcls --workdir /tmp/opencode/scriptcls_run
"""
import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from apply_ocr_pdf import segment_lines  # noqa: E402

BASE = pathlib.Path(__file__).resolve().parents[2]
REPO = BASE.parent if BASE.name == "training" else BASE

VENV_PY = "/home/vvasuki/gitland/python_venv/bin/python"
DRIVE_HELPER = pathlib.Path(__file__).resolve().parent / "drive_ocr_images.py"
GOOGLE_KEY = "/home/vvasuki/gitland/vvasuki-git/sysconf/kunchikA/google/profing/ocr_app_oauth.json"
GOOGLE_KEY = "/home/vvasuki/gitland/vvasuki-git/sysconf/kunchikA/google/proofing/ocr_app_oauth.json"

SCRIPT_COLORS = {"dev": (200, 0, 0), "tam": (0, 120, 0), "other": (0, 0, 220)}
GAP_WORD = 10
STRIDE = 8
MIN_RUN = 20


def load_classifier(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    from train_classifier import Net
    net = Net()
    net.load_state_dict(ck["state_dict"])
    net.to(device).eval()
    return net, ck["classes"]


def word_segments(dark, x0, x1, y0, y1):
    sub = dark[y0:y1, x0:x1]
    col = sub.any(axis=0)
    segs = []
    x = 0
    W = len(col)
    while x < W:
        if not col[x]:
            x += 1
            continue
        a = x
        while x < W and col[x]:
            x += 1
        while x < W:
            g, y = 0, x
            while y < W and not col[y]:
                g += 1
                y += 1
            if y < W and g < GAP_WORD:
                while y < W and col[y]:
                    y += 1
                x = y
            else:
                break
        if x - a >= 6:
            segs.append((x0 + a, x0 + x))
    return segs


def classify_word(net, classes, arr, x0, x1, y0, y1, dev):
    sub = (arr[y0:y1, x0:x1] < 128)
    rows = np.where(sub.any(axis=1))[0]
    if len(rows) == 0:
        return None
    cy = y0 + (int(rows[0]) + int(rows[-1])) // 2
    xs = list(range(x0 + 20, max(x0 + 21, x1 - 20), STRIDE))
    if not xs:
        return None
    patches = []
    for x in xs:
        px0 = x - 24
        py0 = cy - 24
        patch = np.full((48, 48), 255, dtype=np.uint8)
        sx0, sy0 = max(0, px0), max(0, py0)
        sx1 = min(arr.shape[1], px0 + 48)
        sy1 = min(arr.shape[0], py0 + 48)
        if sx1 <= sx0 or sy1 <= sy0:
            patches.append(patch)
            continue
        patch[sy0 - py0:sy1 - py0, sx0 - px0:sx1 - px0] = arr[sy0:sy1, sx0:sx1]
        patches.append(patch)
    xb = torch.from_numpy(np.stack(patches)).unsqueeze(1).float().to(dev)
    xb = (1.0 - xb / 255.0)
    with torch.no_grad():
        probs = net(xb).softmax(1).cpu().numpy()
    labels = []
    for p in probs:
        di, ti, oi = classes.index("dev"), classes.index("tam"), classes.index("other")
        m = max((p[di], "dev"), (p[ti], "tam"), (p[oi], "other"))
        labels.append(m[1] if m[0] > 0.45 else "null")
    runs = []
    a = 0
    for k in range(1, len(labels) + 1):
        if k == len(labels) or labels[k] != labels[a]:
            runs.append([labels[a], xs[a], xs[k - 1] + 1])
            a = k
    merged = []
    for lb, rx0, rx1 in runs:
        if lb == "null":
            continue
        if merged and merged[-1][0] == lb:
            merged[-1][2] = rx1
        else:
            merged.append([lb, rx0, rx1])
    out = []
    for k, (lb, rx0, rx1) in enumerate(merged):
        lo = x0 if k == 0 else (rx0 + merged[k - 1][2]) // 2
        hi = x1 if k == len(merged) - 1 else (rx1 + merged[k + 1][1]) // 2
        if hi - lo < MIN_RUN:
            continue
        rs = (arr[y0:y1, lo:hi] < 128)
        rws = np.where(rs.any(axis=1))[0]
        cols = np.where(rs.any(axis=0))[0]
        if len(rws) == 0 or len(cols) == 0:
            continue
        out.append((lb, lo + int(cols[0]), lo + int(cols[-1]) + 1,
                    y0 + int(rws[0]), y0 + int(rws[-1]) + 1))
    return out


def compose_faux(arr, runs, path, prefix):
    """runs: list of (idx, x0,x1,y0,y1). Stacks crops one per line, height 90."""
    crops = []
    for r in runs:
        _, x0, x1, y0, y1 = r
        c = arr[max(0, y0 - 6):y1 + 6, max(0, x0 - 6):x1 + 6]
        h = c.shape[0]
        s = 90.0 / h
        s = min(s, 1780.0 / max(c.shape[1], 1))
        s = max(s, 0.6)
        im = Image.fromarray(c).resize((max(1, int(round(c.shape[1] * s))), max(1, int(round(h * s)))), Image.LANCZOS)
        crops.append(im)
    W = max(600, max(im.width for im in crops) + 120)
    gap = 130
    H = 40 + sum(im.height + gap for im in crops)
    page = Image.new("L", (W, H), 255)
    y = 20
    for im in crops:
        page.paste(im, (60, y))
        y += im.height + gap
    page.convert("RGB").save(path, quality=92)
    return len(crops)


def parse_drive_lines(txt):
    out = []
    import re
    for line in txt.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("![") or s.startswith("[image") or s.startswith("<data:"):
            continue
        if s == "---":
            continue
        s = s.replace("**", "")
        s = re.sub(r"\\([\[\]\\_\-])", r"\1", s)
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--pages", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--classifier", default="training/local/script_cls.pt")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--overlay", action="store_true")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    net, classes = load_classifier(REPO / args.classifier, dev)
    import pymupdf
    doc = pymupdf.open(args.pdf)
    lo, hi = (int(x) if x else None for x in args.pages.split(":"))
    indices = range(lo or 0, hi if hi is not None else len(doc))
    out_dir = REPO / args.out if not pathlib.Path(args.out).is_absolute() else pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    workdir = pathlib.Path(args.workdir)
    faux_dir = workdir / "faux"
    faux_dir.mkdir(parents=True, exist_ok=True)
    if args.overlay:
        (workdir / "overlay").mkdir(parents=True, exist_ok=True)
    runs_dump = open(out_dir / "runs.jsonl", "w", encoding="utf-8")
    stem = pathlib.Path(args.pdf).stem

    for pi in indices:
        pix = doc[pi].get_pixmap(dpi=args.dpi, colorspace=pymupdf.csGRAY)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        dark = arr < 128
        lines = segment_lines(Image.fromarray(arr), 6)
        page_runs = []
        for li, (lx0, ly0, lx1, ly1) in enumerate(lines):
            words = word_segments(dark, lx0, lx1, ly0, ly1)
            for wi, (wx0, wx1) in enumerate(words):
                rr = classify_word(net, classes, arr, wx0, wx1, ly0, ly1, dev)
                if not rr:
                    continue
                pre_space = wi == 0 or (wx0 - words[wi - 1][1]) >= GAP_WORD
                for k, (script, rx0, rx1, ry0, ry1) in enumerate(rr):
                    page_runs.append({
                        "page": pi, "line": li, "word": wi, "seq": len(page_runs),
                        "script": script, "x0": rx0, "y0": ry0, "x1": rx1, "y1": ry1,
                        "pre_space": pre_space if k == 0 else False, "text": None})
        print(f"page {pi}: {len(lines)} lines, {len(page_runs)} script runs", flush=True)

        if args.overlay:
            ov = Image.fromarray(arr).convert("RGB")
            dr = ImageDraw.Draw(ov)
            for r in page_runs:
                dr.rectangle([r["x0"] - 3, r["y0"] - 3, r["x1"] + 3, r["y1"] + 3],
                             outline=SCRIPT_COLORS[r["script"]], width=3)
            ov.save(workdir / "overlay" / f"p{pi:04d}.jpg", quality=90)

        faux_files = []
        for script in ("dev", "tam", "other"):
            subset = [r for r in page_runs if r["script"] == script]
            if not subset:
                continue
            for ci in range(0, len(subset), args.chunk):
                chunk = subset[ci:ci + args.chunk]
                idxs = [page_runs.index(r) for r in chunk]
                path = faux_dir / f"p{pi:04d}_{script}_{ci:03d}.jpg"
                n = compose_faux(arr, [(k, r["x0"], r["x1"], r["y0"], r["y1"]) for k, r in zip(idxs, chunk)], path, script)
                assert n == len(chunk)
                faux_files.append((path, chunk))
        if faux_files:
            subprocess.run([VENV_PY, str(DRIVE_HELPER), "--key", GOOGLE_KEY] + [str(p) for p, _ in faux_files],
                           check=True, capture_output=True)
        for path, chunk in faux_files:
            txt_path = pathlib.Path(str(path) + ".txt")
            ols = parse_drive_lines(txt_path.read_text(encoding="utf-8")) if txt_path.exists() else []
            if len(ols) == len(chunk):
                for r, t in zip(chunk, ols):
                    r["text"] = t
                continue
            print(f"  WARN {path.name}: {len(ols)} ocr lines vs {len(chunk)} runs -> per-run retry", flush=True)
            singles = []
            for k, r in enumerate(chunk):
                sp = faux_dir / f"{path.stem}_s{k:02d}.jpg"
                compose_faux(arr, [(-1, r["x0"], r["x1"], r["y0"], r["y1"])], sp, r["script"])
                singles.append((sp, r))
            subprocess.run([VENV_PY, str(DRIVE_HELPER), "--key", GOOGLE_KEY] + [str(p) for p, _ in singles],
                           check=True, capture_output=True)
            for sp, r in singles:
                st = pathlib.Path(str(sp) + ".txt")
                sl = parse_drive_lines(st.read_text(encoding="utf-8")) if st.exists() else []
                r["text"] = " ".join(sl) if sl else None

        by_line = {}
        for r in page_runs:
            by_line.setdefault(r["line"], []).append(r)
        with open(out_dir / f"{stem}_p{pi:04d}.txt", "w", encoding="utf-8") as f:
            for li in sorted(by_line):
                parts = []
                for k, r in enumerate(by_line[li]):
                    t = r["text"] or ""
                    if k > 0 and r["pre_space"]:
                        parts.append(" ")
                    parts.append(t)
                f.write("".join(parts) + "\n")
        for r in page_runs:
            runs_dump.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"page {pi}: done", flush=True)
    runs_dump.close()


if __name__ == "__main__":
    main()
