#!/usr/bin/env python3
"""Combine scriptSepDriveOCR (script-pure faux pages + Drive OCR) with
devTamCharsOCR (our line recognizer) at the run level.

Re-runs the deterministic parts of pipeline.py (line/word segmentation, script
classification), recovers the Drive per-run texts from the stored faux-page
outputs, and adds devTamCharsOCR predictions for the same run crops.

Outputs per page:
  {outdir}/runs_{page}.jsonl         per-run records (box, script, drive_text, dev_text)
  {outdir}/{stem}_p{page}_sepdrive.txt   page assembled from Drive texts
  {outdir}/{stem}_p{page}_devruns.txt    page assembled from devTam texts
  {outdir}/{stem}_p{page}_hybrid.txt     per-run heuristic mix

Usage:
  python3 training/scripts/script_cls/combine_eval.py --pdf ... --page 100 \
      --workdir /tmp/opencode/scriptcls_v2 --outdir /tmp/opencode/combined
"""
import argparse
import json
import pathlib
import re
import sys

import numpy as np
import torch
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pipeline as P  # noqa: E402
from train_lines2 import SUBSAMPLE, ids_to_text  # noqa: E402

REPO = P.REPO
SUSPECT_RE = re.compile(r"[A-Za-z\\|*#\[\]{}<>]")


def assemble(page_runs, key):
    by_line = {}
    for r in page_runs:
        by_line.setdefault(r["line"], []).append(r)
    out = []
    for li in sorted(by_line):
        parts = []
        for k, r in enumerate(by_line[li]):
            t = r[key] or ""
            if k > 0 and r["pre_space"]:
                parts.append(" ")
            parts.append(t)
        out.append("".join(parts))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--page", type=int, required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--classifier", default="training/local/script_cls.pt")
    ap.add_argument("--ckpt", default="training/local/kaggle_v10_best.pt")
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    workdir = pathlib.Path(args.workdir)
    faux_dir = workdir / "faux"

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cls_net, classes = P.load_classifier(REPO / args.classifier, dev)

    import pymupdf
    doc = pymupdf.open(args.pdf)
    pi = args.page
    pix = doc[pi].get_pixmap(dpi=args.dpi, colorspace=pymupdf.csGRAY)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    dark = arr < 128
    lines = P.segment_lines(Image.fromarray(arr), 6)

    page_runs = []
    for li, (lx0, ly0, lx1, ly1) in enumerate(lines):
        words = P.word_segments(dark, lx0, lx1, ly0, ly1)
        for wi, (wx0, wx1) in enumerate(words):
            rr = P.classify_word(cls_net, classes, arr, wx0, wx1, ly0, ly1, dev)
            if not rr:
                continue
            pre_space = wi == 0 or (wx0 - words[wi - 1][1]) >= P.GAP_WORD
            for k, (script, rx0, rx1, ry0, ry1) in enumerate(rr):
                page_runs.append({
                    "page": pi, "line": li, "word": wi, "seq": len(page_runs),
                    "script": script, "x0": rx0, "y0": ry0, "x1": rx1, "y1": ry1,
                    "pre_space": pre_space if k == 0 else False,
                    "drive_text": None, "dev_text": None})
    print(f"page {pi}: {len(lines)} lines, {len(page_runs)} runs", flush=True)

    # --- recover Drive texts from the stored faux pages (same naming as pipeline) ---
    for script in ("dev", "tam", "other"):
        subset = [r for r in page_runs if r["script"] == script]
        if not subset:
            continue
        for ci in range(0, len(subset), args.chunk):
            chunk = subset[ci:ci + args.chunk]
            path = faux_dir / f"p{pi:04d}_{script}_{ci:03d}.jpg"
            txt_path = pathlib.Path(str(path) + ".txt")
            if not txt_path.exists():
                print(f"  WARN missing {txt_path}", flush=True)
                continue
            texts = P.parse_tagged(txt_path.read_text(encoding="utf-8"), len(chunk))
            missing = [k for k, t in enumerate(texts) if t is None]
            if missing:
                retry = faux_dir / f"{path.stem}_retry.jpg"
                rt = pathlib.Path(str(retry) + ".txt")
                if rt.exists():
                    rtexts = P.parse_tagged(rt.read_text(encoding="utf-8"), len(missing))
                    for m, t in zip(missing, rtexts):
                        if t is not None:
                            texts[m] = t
            for r, t in zip(chunk, texts):
                r["drive_text"] = t

    # --- devTamCharsOCR on the same run crops ---
    from kraken.lib.vgsl import TorchVGSLModel
    from kraken.lib.dataset.utils import ImageInputTransforms

    ckpt = torch.load(REPO / args.ckpt, map_location="cpu", weights_only=False)
    model = TorchVGSLModel(vgsl=ckpt["spec"], mode="recognition")
    net = model.nn
    net.load_state_dict(ckpt["state_dict"])
    net.to(dev).eval()
    train_meta = json.load(open(REPO / "training/local/lines_cache/train/meta.json"))
    rev = {v: ch for ch, vs in train_meta["c2l"].items() for v in vs}
    trans = ImageInputTransforms(1, 64, 0, 1, (16, 0), valid_norm=True)

    items, metas = [], []
    for r in page_runs:
        c = arr[max(0, r["y0"] - 6):r["y1"] + 6, max(0, r["x0"] - 6):r["x1"] + 6]
        if c.size == 0:
            items.append(None)
            metas.append(0)
            continue
        im = Image.fromarray(c)
        s = 128 / im.height
        im2 = im.resize((max(1, round(im.width * s)), 128), Image.LANCZOS)
        x = trans(im2)
        items.append(x)
        metas.append(int(x.shape[2]))
    order = sorted([j for j, it in enumerate(items) if it is not None], key=lambda j: metas[j])
    B = 20
    with torch.no_grad():
        for b in range(0, len(order), B):
            idxs = order[b:b + B]
            W = max(items[j].shape[2] for j in idxs)
            stack = torch.zeros(len(idxs), 1, 64, W)
            for r, j in enumerate(idxs):
                stack[r, 0, :, :items[j].shape[2]] = items[j][0]
            out = net(stack.to(dev))
            lg = (out[0] if isinstance(out, tuple) else out).squeeze(2).permute(2, 0, 1)
            preds = lg.argmax(2)
            for r, j in enumerate(idxs):
                tlen = max(1, min(preds.shape[0], metas[j] // SUBSAMPLE))
                page_runs[j]["dev_text"] = ids_to_text(rev, preds[:tlen, r].tolist())

    for r in page_runs:
        dt, vt = r["drive_text"], r["dev_text"]
        if dt and SUSPECT_RE.search(dt) and vt:
            r["hybrid_text"] = vt
            r["hybrid_src"] = "dev"
        else:
            r["hybrid_text"] = dt
            r["hybrid_src"] = "drive"
    n_dev = sum(1 for r in page_runs if r["hybrid_src"] == "dev")
    n_null = sum(1 for r in page_runs if r["drive_text"] is None)
    print(f"  hybrid: {n_dev} runs from devTam, {len(page_runs)-n_dev} from Drive "
          f"({n_null} empty Drive runs)", flush=True)

    stem = pathlib.Path(args.pdf).stem
    with open(outdir / f"runs_{pi}.jsonl", "w", encoding="utf-8") as f:
        for r in page_runs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for key, suffix in (("drive_text", "sepdrive"), ("dev_text", "devruns"),
                        ("hybrid_text", "hybrid")):
        lines_out = assemble(page_runs, key)
        with open(outdir / f"{stem}_p{pi:04d}_{suffix}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines_out) + "\n")
        print(f"  wrote {suffix}: {len(lines_out)} lines", flush=True)


if __name__ == "__main__":
    main()
