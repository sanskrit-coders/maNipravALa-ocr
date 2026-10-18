#!/usr/bin/env python3
"""Cache word-level TSV for block crops (line segmentation + rough line text).

For every row in the given split files, runs Tesseract (tam+san, PSM 6, tsv config)
on the crop image and caches the TSV under training/metadata/tsv_{vol}/.

Usage:
  python3 training/scripts/ocr_tsv.py training/split_train.jsonl evaluation/split_eval.jsonl
"""
import argparse
import json
import os
import pathlib
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

BASE = pathlib.Path(__file__).resolve().parents[1]
REPO = BASE.parent if BASE.name == "training" else BASE
TESS = "/tmp/tess-root/usr/bin/tesseract"
TESSDATA = "/tmp/tess-root/usr/share/tesseract-ocr/5/tessdata/"


def run_one(row):
    vol = re.search(r"RTS_sb_(\d+)", row["image"]).group(1)
    out = REPO / f"training/metadata/tsv_{vol}/{row['page']}_{row['block']}.tsv"
    if out.exists() and out.stat().st_size > 10:
        return "cached"
    out.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TESSDATA_PREFIX=TESSDATA, OMP_THREAD_LIMIT="1")
    with open(out, "w") as f:
        r = subprocess.run([TESS, str(REPO / row["image"]), "stdout", "-l", "tam+san",
                            "--psm", "6", "tsv"], stdout=f, stderr=subprocess.DEVNULL, env=env)
    if r.returncode != 0:
        out.unlink(missing_ok=True)
        return f"FAIL {row['image']}"
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("splits", nargs="+")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    rows = []
    for sp in args.splits:
        for line in open(sp):
            rows.append(json.loads(line))
    todo = [r for r in rows if "_doubtful" not in r["image"]]
    print(f"{len(todo)} crops")
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(run_one, todo):
            done += 1
            if done % 100 == 0:
                print(f"  tsv {done}/{len(todo)}", flush=True)
    print("tsv done")


if __name__ == "__main__":
    main()
