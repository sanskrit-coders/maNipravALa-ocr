#!/usr/bin/env python3
"""Compare line-level eval dumps (from eval_lines.py --dump) into a markdown table.

Usage: python3 training/scripts/compare_line_evals.py \
   --baseline evaluation/line_eval_baseline.jsonl --kraken evaluation/line_eval_kraken.jsonl
"""
import argparse
import json
import pathlib
import re

import jiwer


def strip_ws(s):
    return re.sub(r"\s+", "", s)


def metrics(pairs):
    raw_t = [t for t, _ in pairs]
    raw_h = [h for _, h in pairs]
    nows_t = [strip_ws(t) for t in raw_t]
    nows_h = [strip_ws(h) for h in raw_h]
    return {
        "n": len(pairs),
        "cer": jiwer.cer(raw_t, raw_h),
        "wer": jiwer.wer(raw_t, raw_h),
        "cer_nows": jiwer.cer(nows_t, nows_h),
    }


def load(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return {r["name"]: r for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="evaluation/line_eval_baseline.jsonl")
    ap.add_argument("--kraken", default="evaluation/line_eval_kraken.jsonl")
    ap.add_argument("--out", default="evaluation/COMPARE_LINES.md")
    args = ap.parse_args()

    base = load(args.baseline)
    kr = load(args.kraken)
    names = sorted(set(base) & set(kr))
    pairs_b = [(base[n]["gt"], base[n]["pred"]) for n in names]
    pairs_k = [(kr[n]["gt"], kr[n]["pred"]) for n in names]
    mb, mk = metrics(pairs_b), metrics(pairs_k)

    lines = ["# Line-level OCR comparison (eval split)", "",
             f"Lines: {len(names)}", "",
             "| engine | CER | CER (no-space) | WER |", "|---|---|---|---|",
             f"| Tesseract tam+san psm7 | {mb['cer']:.3f} | {mb['cer_nows']:.3f} | {mb['wer']:.3f} |",
             f"| Kraken model | {mk['cer']:.3f} | {mk['cer_nows']:.3f} | {mk['wer']:.3f} |", ""]
    pathlib.Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("wrote", args.out)
    free = sum(1 for n in names if kr[n]["pred"] == kr[n]["gt"])
    print(f"kraken exact-match lines: {free}/{len(names)}")


if __name__ == "__main__":
    main()
