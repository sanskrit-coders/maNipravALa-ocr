#!/usr/bin/env python3
"""Compare OCR text candidates against a small line-level ground truth.

Usage:
  eval_lines.py --gt gt.txt "script-cls=pred1.txt" "linenet=pred2.txt" ...
All files are read as UTF-8; whitespace is normalized for CER, line breaks
are kept for WER at the whitespace-token level.
"""
import argparse
import json
import re

import jiwer


def strip_ws(s):
    return re.sub(r"\s+", "", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("preds", nargs="+", help="name=path")
    args = ap.parse_args()
    gt = open(args.gt, encoding="utf-8").read().strip()
    out = {}
    for spec in args.preds:
        name, path = spec.split("=", 1)
        pr = open(path, encoding="utf-8").read().strip()
        out[name] = {
            "cer": round(jiwer.cer(gt, pr), 4),
            "cer_nows": round(jiwer.cer(strip_ws(gt), strip_ws(pr)), 4),
            "wer": round(jiwer.wer(gt, pr), 4),
            "wer_nows": round(jiwer.wer(strip_ws(gt), strip_ws(pr)), 4),
            "n_chars": len(strip_ws(gt)),
        }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
