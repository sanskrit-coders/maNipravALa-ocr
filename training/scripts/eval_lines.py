#!/usr/bin/env python3
"""Line-level CER/WER on the eval split, for Tesseract baseline or a Kraken model.

Usage:
  python3 training/scripts/eval_lines.py --engine tess
  python3 training/scripts/eval_lines.py --engine kraken --model training/local/kraken_model/best.safetensors
Metrics: jiwer CER/WER on raw text and on whitespace-stripped text (scripts with
uncertain spacing); printed per line plus aggregate.
"""
import argparse
import csv
import json
import os
import pathlib
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

BASE = pathlib.Path(__file__).resolve().parents[1]
REPO = BASE.parent if BASE.name == "training" else BASE
TESS = "/tmp/tess-root/usr/bin/tesseract"
TESSDATA = "/tmp/tess-root/usr/share/tesseract-ocr/5/tessdata/"
LINES = REPO / "training/local/lines"


def strip_ws(s):
    return re.sub(r"\s+", "", s)


def metrics(pairs):
    import jiwer
    raw_t = [t for t, _ in pairs]
    raw_h = [h for _, h in pairs]
    nows_t = [strip_ws(t) for t in raw_t]
    nows_h = [strip_ws(h) for h in raw_h]
    out = {
        "lines": len(pairs),
        "cer": jiwer.cer(raw_t, raw_h),
        "wer": jiwer.wer(raw_t, raw_h),
        "cer_nows": jiwer.cer(nows_t, nows_h),
        "wer_nows": jiwer.wer(nows_t, nows_h),
    }
    return out


def tess_pred(path):
    env = dict(os.environ, TESSDATA_PREFIX=TESSDATA, OMP_THREAD_LIMIT="1")
    r = subprocess.run([TESS, str(path), "stdout", "-l", "tam+san", "--psm", "7"],
                       capture_output=True, text=True, env=env)
    return re.sub(r"\s+", " ", r.stdout).strip()


def engine_tess(rows):
    with ThreadPoolExecutor(max_workers=6) as ex:
        preds = list(ex.map(lambda r: tess_pred(LINES / "eval" / (r["name"] + ".png")), rows))
    return [{"name": r["name"], "gt": r["gt"], "pred": p} for r, p in zip(rows, preds)]


def engine_kraken(rows, model_path):
    from kraken.containers import Segmentation, BBoxLine
    from kraken.models import load_models
    from kraken import rpred
    from PIL import Image
    models = load_models(str(model_path))
    net = models[0]
    out = []
    for r in rows:
        p = LINES / "eval" / (r["name"] + ".png")
        im = Image.open(p)
        seg = Segmentation(type="bbox", imagename=str(p), text_direction="horizontal-lr",
                           script_detection=False,
                           lines=[BBoxLine(id=r["name"], bbox=(0, 0, im.size[0], im.size[1]))])
        preds = list(rpred.rpred(net, im, seg))
        out.append({"name": r["name"], "gt": r["gt"],
                    "pred": preds[0].prediction.strip() if preds else ""})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["tess", "kraken"], required=True)
    ap.add_argument("--model")
    ap.add_argument("--dump", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = []
    for line in open(LINES / "manifest.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["split"] == "eval" and (LINES / "eval" / (r["name"] + ".png")).exists():
            rows.append(r)
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} eval lines")

    if args.engine == "tess":
        preds = engine_tess(rows)
    else:
        preds = engine_kraken(rows, args.model)

    pairs = [(p["gt"], p["pred"]) for p in preds]
    res = metrics(pairs)
    per_block = Counter()
    block_pairs = {}
    for r, p in zip(rows, preds):
        block_pairs.setdefault((r["vol"], r["page"], r["block"]), []).append((p["gt"], p["pred"]))
    blocks = {f"{v}:{pg}:{b}": metrics(pr) for (v, pg, b), pr in block_pairs.items()}
    print(json.dumps({"engine": args.engine, "model": args.model or "tesseract-tam+san-psm7",
                      "overall": res}, indent=1))
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        json.dump(blocks, open(args.dump + ".blocks.json", "w"), indent=1)
        print("dumped", args.dump)
    low = sorted(preds, key=lambda p: __import__("jiwer").cer(p["gt"], p["pred"]), reverse=True)[:5]
    print("worst lines:", [(p["name"], round(__import__("jiwer").cer(p["gt"], p["pred"]), 2)) for p in low])


if __name__ == "__main__":
    main()
