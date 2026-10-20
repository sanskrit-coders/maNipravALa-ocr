#!/usr/bin/env python3
"""Segmentation control for the RTS_sb block evaluation.

Reuses the run segmentation stored by eval_sb.py and recognizes the same
script segments with the mixed `tam+san` Tesseract model. Comparing this with
tess_seg (per-script `san`/`tam`) separates the gain from shorter units from
the gain from per-script model routing.

Usage:
  python3 training/scripts/script_cls/control_seg.py \
      --indir evaluation/sb_eval --out evaluation/sb_eval/tess_seg_mixed.json
"""
import argparse
import json
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_sb as E  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="evaluation/sb_eval")
    ap.add_argument("--out", default="evaluation/sb_eval/tess_seg_mixed.json")
    args = ap.parse_args()

    indir = E.REPO / args.indir
    blocks = {}
    jobs = []
    for name, img, gt_path in E.BLOCKS:
        runs = [json.loads(line) for line in
                open(indir / f"runs_{name}.jsonl", encoding="utf-8")]
        arr = E.load_gray(img, 2)
        segs = E.make_segments(runs)
        gt = (E.REPO / gt_path).read_text(encoding="utf-8")
        blocks[name] = {"runs": runs, "segs": segs, "gt": gt}
        for s in segs:
            jobs.append((name, s["idx"], E.crop(arr, s["box"])))

    def work(job):
        name, idx, im = job
        return name, idx, E.tess_text(im, "tam+san")

    texts = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for name, idx, text in pool.map(work, jobs):
            texts[(name, idx)] = text

    result, total_d, total_n = {}, 0, 0
    for name, b in blocks.items():
        pred = "\n".join(E.assemble_segments(
            b["runs"], b["segs"], lambda s: texts[(name, s["idx"])]))
        g = "".join(b["gt"].split())
        q = "".join(pred.split())
        dist = Levenshtein.distance(g, q)
        result[name] = {"cer": dist / len(g), "dist": dist, "n": len(g)}
        total_d += dist
        total_n += len(g)

    result["micro"] = {"cer": total_d / total_n, "dist": total_d, "n": total_n}
    out = E.REPO / args.out
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"tess_seg_mixed micro: {result['micro']['cer']:.4f}")


if __name__ == "__main__":
    main()
