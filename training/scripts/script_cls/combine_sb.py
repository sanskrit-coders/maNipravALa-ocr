#!/usr/bin/env python3
"""Combination variants for the RTS_sb block evaluation.

Reuses the per-run texts stored by eval_sb.py (runs_{block}.jsonl) and scores
run-level combinations of scriptSepDriveOCR and devTamCharsOCR against the
block GT (whitespace-stripped CER, micro-averaged).

Usage:
  python3 training/scripts/script_cls/combine_sb.py \
      --indir evaluation/sb_eval --out evaluation/sb_eval/combine.json
"""
import argparse
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_sb as E  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402

SUSPECT = re.compile(r"[A-Za-z\\|*#\[\]{}<>]")


def assemble(runs, text_of):
    by_line = {}
    for r in runs:
        by_line.setdefault(r["line"], []).append(r)
    out = []
    for li in sorted(by_line):
        parts = []
        for k, r in enumerate(by_line[li]):
            t = text_of(r) or ""
            if k > 0 and r["pre_space"]:
                parts.append(" ")
            parts.append(t)
        out.append("".join(parts))
    return "\n".join(out)


VARIANTS = {
    "drive_sep": lambda r: r.get("drive_sep"),
    "dev_run": lambda r: r.get("dev_run"),
    "dev_runs_for_dev": lambda r: r.get("dev_run") if r["script"] == "dev" else r.get("drive_sep"),
    "dev_runs_for_tam": lambda r: r.get("dev_run") if r["script"] == "tam" else r.get("drive_sep"),
    "empty_fill": lambda r: r.get("drive_sep") or r.get("dev_run"),
    "suspect_hybrid": lambda r: (
        r.get("dev_run")
        if (r.get("drive_sep") and SUSPECT.search(r["drive_sep"]) and r.get("dev_run"))
        else r.get("drive_sep")
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="evaluation/sb_eval")
    ap.add_argument("--out", default="evaluation/sb_eval/combine.json")
    args = ap.parse_args()

    indir = E.REPO / args.indir
    rows, totals = {}, {}
    for name, _img, gt_path in E.BLOCKS:
        runs = [json.loads(line) for line in
                open(indir / f"runs_{name}.jsonl", encoding="utf-8")]
        gt = (E.REPO / gt_path).read_text(encoding="utf-8")
        g = "".join(gt.split())
        rows[name] = {}
        for key, fn in VARIANTS.items():
            pred = assemble(runs, fn)
            q = "".join(pred.split())
            dist = Levenshtein.distance(g, q)
            rows[name][key] = {"cer": dist / len(g), "dist": dist, "n": len(g)}
            t = totals.setdefault(key, [0, 0])
            t[0] += dist
            t[1] += len(g)

    result = {"blocks": rows,
              "micro": {k: {"cer": d / n, "dist": d, "n": n}
                        for k, (d, n) in totals.items()}}
    out = E.REPO / args.out
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    for k, v in result["micro"].items():
        print(f"{k}: {v['cer']:.4f}")


if __name__ == "__main__":
    main()
