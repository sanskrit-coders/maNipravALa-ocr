#!/usr/bin/env python3
"""OCR the four RTS 5-commentary volumes page by page.

For each page: deskew, detect every horizontal section rule with
detect_blocks.find_section_rules, segment lines/words, classify script with the
new 48x48 CNN, OCR script segments with Tesseract (san/tam), and rebuild the
page with Markdown horizontal rules between sections.

The heavy loop is intended for Kaggle.  Locally, point --pdf-dir at the PDF
directory and --outdir at the raw target.
"""
import argparse
import collections
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool

import cv2
import numpy as np
import torch
from PIL import Image

try:
    import pymupdf
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pymupdf"], check=True)
    import pymupdf

HERE = pathlib.Path(__file__).resolve().parent
for cand in (HERE, HERE.parent,
             pathlib.Path("/home/vvasuki/gitland/sanskrit-coders/doc_curation/doc_curation/image/checker")):
    if str(cand) not in sys.path:
        sys.path.insert(0, str(cand))
if pathlib.Path("/kaggle/input").exists():
    for p in pathlib.Path("/kaggle/input").rglob("pipeline.py"):
        if str(p.parent) not in sys.path:
            sys.path.insert(0, str(p.parent))

import types
if "train_lines2" not in sys.modules:
    _tl = types.ModuleType("train_lines2")
    _tl.SUBSAMPLE = 4
    _tl.ids_to_text = lambda rev, ids: ""
    sys.modules["train_lines2"] = _tl

import pipeline as P
from train_classifier import Net, CLASSES
from detect_blocks import estimate_skew, deskew_image, find_section_rules

DPI = 150
OTHER_OK = re.compile(r"^[\s।॥|.,;:!?()\[\]{}'\"“”‘’\-–—0-9]+$")
DOC = None
NET = None
DPI_ACTIVE = DPI
TESS_BIN = "/tmp/tess-root/usr/bin/tesseract" if pathlib.Path("/tmp/tess-root/usr/bin/tesseract").exists() else "/usr/bin/tesseract"


def resolve_input(name):
    if pathlib.Path(name).exists():
        return pathlib.Path(name)
    if pathlib.Path("/kaggle/input").exists():
        hits = sorted(pathlib.Path("/kaggle/input").rglob(pathlib.Path(name).name))
        if hits:
            return hits[0]
    raise SystemExit(f"missing input: {name}")


def resolve_pdf_dir(pdf_dir):
    if pdf_dir:
        p = pathlib.Path(pdf_dir)
        if p.exists():
            return p
    if pathlib.Path("/kaggle/input").exists():
        hits = sorted(pathlib.Path("/kaggle/input").rglob("ra-tra-sA_5-commentaries_1.pdf"))
        if hits:
            return hits[0].parent
    return pathlib.Path("/media/vvasuki/vData/text/granthasangrahaH/AgamaH/vaiShNavaH/shrIvaiShNavaH/venkaTanAtha/rts/5-comm")


def tess_env(tessdata):
    return dict(os.environ, TESSDATA_PREFIX=str(tessdata), OMP_THREAD_LIMIT="1")


def tess_text(img, lang, tessdata, psm=7):
    if img is None or img.size == 0:
        return ""
    im = Image.fromarray(img)
    if im.height < 90:
        s = max(1.0, 90.0 / max(1, im.height))
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        im.save(path)
        r = subprocess.run([TESS_BIN, path, "stdout", "-l", lang,
                            "--psm", str(psm)],
                           capture_output=True, text=True, env=tess_env(tessdata))
        return " ".join(r.stdout.split())
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def init_worker(pdf_path, tessdata):
    global DOC, NET, DPI_ACTIVE
    DOC = pymupdf.open(pdf_path)
    NET = Net()
    try:
        ck_path = resolve_input("script_cls.pt")
    except SystemExit:
        ck_path = pathlib.Path("/home/vvasuki/gitland/sanskrit-coders/maNipravALa-ocr/training/local/script_cls.pt")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    NET.load_state_dict(ck["state_dict"])
    NET.to(torch.device("cpu")).eval()
    DPI_ACTIVE = DPI
TESS_BIN = "/tmp/tess-root/usr/bin/tesseract" if pathlib.Path("/tmp/tess-root/usr/bin/tesseract").exists() else "/usr/bin/tesseract"


def assemble_line(runs, segs, texts):
    parts = []
    first = True
    for s in segs:
        if s["script"] == "other" and not texts.get(s["idx"]):
            continue
        t = (texts.get(s["idx"]) or "").strip()
        if not t:
            continue
        if not first and runs[s["indices"][0]]["pre_space"]:
            parts.append(" ")
        parts.append(t)
        first = False
    return "".join(parts)


def insert_section_rules(lines, rule_ys):
    out = []
    ri = 0
    for y, text in lines:
        while ri < len(rule_ys) and rule_ys[ri] < y - 2:
            if out and out[-1] != "---":
                out.append("---")
            ri += 1
        if text.strip():
            out.append(text.strip())
    return out


def process_page(pi):
    page = DOC[pi]
    pix = page.get_pixmap(dpi=DPI_ACTIVE, colorspace=pymupdf.csGRAY)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    gray = arr
    angle, _ = estimate_skew(gray)
    if abs(angle) >= 0.05:
        arr = deskew_image(arr, angle)
        gray = arr
    meta = find_section_rules(arr, angle=angle)
    hb = int(meta["header_bottom"])
    rule_ys = [float(L["yc"]) for L in meta["rules"] if L["yc"] > hb + 2]

    lines = P.segment_lines(Image.fromarray(arr), 6)
    dark = arr < 128
    runs = []
    for li, (x0, y0, x1, y1) in enumerate(lines):
        words = P.word_segments(dark, x0, x1, y0, y1)
        for wi, (wx0, wx1) in enumerate(words):
            rr = P.classify_word(NET, CLASSES, arr, wx0, wx1, y0, y1, torch.device("cpu"))
            if not rr:
                continue
            pre_space = wi == 0 or (wx0 - words[wi - 1][1]) >= P.GAP_WORD
            for k, (script, rx0, rx1, ry0, ry1) in enumerate(rr):
                runs.append({"line": li, "script": script, "x0": rx0, "y0": ry0,
                             "x1": rx1, "y1": ry1,
                             "pre_space": pre_space if k == 0 else False})
    segs = []
    by_line = collections.defaultdict(list)
    for ri, r in enumerate(runs):
        by_line[r["line"]].append(ri)
    for li in sorted(by_line):
        cur = []
        for ri in by_line[li]:
            if cur and runs[ri]["script"] != runs[cur[-1]]["script"]:
                segs.append({"line": li, "indices": cur,
                             "script": runs[cur[0]]["script"]})
                cur = []
            cur.append(ri)
        if cur:
            segs.append({"line": li, "indices": cur,
                         "script": runs[cur[0]]["script"]})
    for k, s in enumerate(segs):
        rs = [runs[i] for i in s["indices"]]
        s["idx"] = k
        s["box"] = (min(r["x0"] for r in rs), min(r["y0"] for r in rs),
                    max(r["x1"] for r in rs), max(r["y1"] for r in rs))
    texts = {}
    for s in segs:
        x0, y0, x1, y1 = s["box"]
        im = arr[max(0, y0 - 5):y1 + 5, max(0, x0 - 5):x1 + 5]
        if im.size == 0:
            continue
        if s["script"] in ("dev", "tam"):
            texts[s["idx"]] = tess_text(im, "san" if s["script"] == "dev" else "tam", TESSDATA)
        elif s["script"] == "other":
            t = tess_text(im, "tam+san", TESSDATA)
            if t and OTHER_OK.match(t):
                texts[s["idx"]] = t
    lineout = []
    for li in sorted(by_line):
        if lines[li][1] < hb:
            continue
        ss = [s for s in segs if s["line"] == li]
        lineout.append((lines[li][1], assemble_line(runs, ss, texts)))
    body = insert_section_rules(lineout, rule_ys)
    return body


def header_number(pi, tessdata):
    page = DOC[pi]
    pix = page.get_pixmap(dpi=DPI_ACTIVE, colorspace=pymupdf.csGRAY)
    arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
    im = Image.fromarray(arr[: max(20, int(arr.shape[0] * 0.11))])
    im = im.resize((im.width * 2, im.height * 2), Image.LANCZOS)
    t = tess_text(np.asarray(im), "san", tessdata, psm=6)
    nums = [int(x) for x in re.findall(r"\d{1,4}", t)]
    for n in reversed(nums):
        if 0 < n < 2000:
            return n
    return None


def volume_offset(doc, tessdata):
    vals = []
    for pi in range(min(len(doc), 90), min(len(doc), 140)):
        n = header_number(pi, tessdata)
        if n:
            vals.append((pi + 1) - n)
    if not vals:
        return 0
    c = collections.Counter(vals)
    return c.most_common(1)[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default="")
    ap.add_argument("--outdir", default="")
    ap.add_argument("--tessdata", default="")
    ap.add_argument("--volumes", default="1,2,3,4")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pages", default="")
    args = ap.parse_args()

    global TESSDATA
    if args.tessdata:
        TESSDATA = pathlib.Path(args.tessdata)
    else:
        try:
            TESSDATA = resolve_input("tam.traineddata").parent
        except SystemExit:
            TESSDATA = pathlib.Path("/tmp/tess-root/usr/share/tesseract-ocr/5/tessdata")
    pdf_dir = resolve_pdf_dir(args.pdf_dir)
    outdir = pathlib.Path(args.outdir) if args.outdir else (
        pathlib.Path("/kaggle/working/raw") if pathlib.Path("/kaggle/working").exists()
        else pathlib.Path("/home/vvasuki/gitland/vishvAsa/rAmAnujIyam/content/tattvam/venkaTa-nAtha-shAkhA/venkaTanAthaH/rahasya-traya-sAraH/5-tIkAH/raw"))
    outdir.mkdir(parents=True, exist_ok=True)
    print("pdf_dir", pdf_dir, "outdir", outdir, "tessdata", TESSDATA, flush=True)

    for v in [int(x) for x in args.volumes.split(",") if x.strip()]:
        pdf = pdf_dir / f"ra-tra-sA_5-commentaries_{v}.pdf"
        if not pdf.exists():
            print("missing", pdf, flush=True)
            continue
        doc = pymupdf.open(pdf)
        global DOC
        DOC = doc
        offset = volume_offset(doc, TESSDATA)
        print(f"v{v}: {len(doc)} pages offset {offset}", flush=True)
        out = outdir / f"v{v}.md"
        with out.open("w", encoding="utf-8") as f:
            pool = Pool(args.workers, initializer=init_worker, initargs=(str(pdf), str(TESSDATA)))
            pages = range(len(doc))
            if args.pages:
                a, b = (int(x) if x else None for x in args.pages.split(":"))
                pages = range(a or 0, b if b is not None else len(doc))
            for k, (pi, body) in enumerate(zip(pages, pool.imap(process_page, pages, chunksize=1))):
                printed = (pi + 1) - offset if offset else pi + 1
                if printed <= 0:
                    printed = pi + 1
                f.write(f"[[P{printed}]]\n")
                if body:
                    f.write("\n".join(body) + "\n")
                f.write("\n")
                if k % 10 == 0:
                    print(f"v{v} p{pi}: {len(body)} blocks", flush=True)
                    f.flush()
            pool.close()
            pool.join()
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
