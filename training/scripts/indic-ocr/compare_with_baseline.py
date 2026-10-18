#!/usr/bin/env python3
"""Compare Tesseract baseline vs PaddleOCR (Indic-OCR engine) dumps on the eval split.

Dumps (from eval_lines.py and indic-ocr/eval_paddleocr.py):
  evaluation/line_eval_baseline.jsonl     {name, gt, pred}
  evaluation/line_eval_paddleocr.jsonl    {name, gt, pred, pred_ta, pred_hi, best_lang, ...} (rec)
  evaluation/line_eval_paddleocr_det.jsonl  same, det+rec mode

Reports metrics on the CTC-feasible subset (the 445 lines used for kraken
training/eval; the 89 excluded lines carry broken build_lines alignments) and,
separately, on all 534 lines.

Usage: python3 training/scripts/indic-ocr/compare_with_baseline.py [--markdown OUT.md]
"""
import argparse
import json
import pathlib
import re

import jiwer

REPO = pathlib.Path(__file__).resolve().parents[3]
EVAL = REPO / "evaluation"


def strip_ws(s):
    return re.sub(r"\s+", "", s)


def metrics(pairs):
    raw_t = [t for t, _ in pairs]
    raw_h = [h for _, h in pairs]
    return {
        "n": len(pairs),
        "cer": jiwer.cer(raw_t, raw_h),
        "wer": jiwer.wer(raw_t, raw_h),
        "cer_nows": jiwer.cer([strip_ws(t) for t in raw_t], [strip_ws(h) for h in raw_h]),
        "wer_nows": jiwer.wer([strip_ws(t) for t in raw_t], [strip_ws(h) for h in raw_h]),
    }


def feasible_subset():
    meta = json.load(open(REPO / "training/local/lines_cache/eval/meta.json"))
    return {pathlib.Path(meta["files"][i]).stem
            for i in range(meta["n"]) if meta["ylens"][i] * 4 <= meta["widths"][i]}


def script_of(gt):
    deva = sum(1 for c in gt if "\u0900" <= c <= "\u097f")
    taml = sum(1 for c in gt if "\u0b80" <= c <= "\u0bff")
    tot = deva + taml
    if tot == 0:
        return "other"
    if taml / tot >= 0.8:
        return "tamil-dominant"
    if deva / tot >= 0.8:
        return "devanagari-dominant"
    return "mixed"


def load(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return {r["name"]: r for r in rows}


def row(fmt, tag, m):
    return (f"| {tag} | {m['n']} | {m['cer']:.3f} | {m['cer_nows']:.3f} | "
            f"{m['wer']:.3f} | {m['wer_nows']:.3f} |")


def engine_rows(rows, names, key):
    return [(rows[n]["gt"], rows[n][key]) for n in names if n in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", default="")
    args = ap.parse_args()

    tess = load(EVAL / "line_eval_baseline.jsonl")
    rec = load(EVAL / "line_eval_paddleocr.jsonl")
    detf = EVAL / "line_eval_paddleocr_det.jsonl"
    det = load(detf) if detf.exists() else None
    feas = feasible_subset()

    all_names = sorted(set(tess) & set(rec))
    sub_names = [n for n in all_names if n in feas]
    print(f"all lines {len(all_names)}, feasible subset {len(sub_names)}")

    out = []

    def emit(s=""):
        print(s)
        out.append(s)

    emit("### Feasible subset (CTC-valid lines, used for kraken training/eval)")
    emit()
    emit("| engine | n | CER | CER no-space | WER | WER no-space |")
    emit("|---|---|---|---|---|---|")
    emit(row("", "Tesseract tam+san psm7", metrics(engine_rows(tess, sub_names, "pred"))))
    emit(row("", "PaddleOCR rec (ta)", metrics(engine_rows(rec, sub_names, "pred_ta"))))
    emit(row("", "PaddleOCR rec (hi)", metrics(engine_rows(rec, sub_names, "pred_hi"))))
    emit(row("", "PaddleOCR rec (best-of)", metrics(engine_rows(rec, sub_names, "pred"))))
    if det:
        emit(row("", "PaddleOCR det+rec (ta)", metrics(engine_rows(det, sub_names, "pred_ta"))))
        emit(row("", "PaddleOCR det+rec (best-of)", metrics(engine_rows(det, sub_names, "pred"))))

    emit()
    emit("### All 534 lines")
    emit()
    emit("| engine | n | CER | CER no-space | WER | WER no-space |")
    emit("|---|---|---|---|---|---|")
    emit(row("", "Tesseract tam+san psm7", metrics(engine_rows(tess, all_names, "pred"))))
    emit(row("", "PaddleOCR rec (best-of)", metrics(engine_rows(rec, all_names, "pred"))))
    if det:
        emit(row("", "PaddleOCR det+rec (best-of)", metrics(engine_rows(det, all_names, "pred"))))

    emit()
    emit("### By script composition (feasible subset, GT-based)")
    emit()
    buckets = {}
    for n in sub_names:
        buckets.setdefault(script_of(tess[n]["gt"]), []).append(n)
    emit("| script | n | tess CER | paddle rec ta CER | paddle rec best CER |")
    emit("|---|---|---|---|---|")
    for b in sorted(buckets):
        ns = buckets[b]
        mt = metrics(engine_rows(tess, ns, "pred"))
        mta = metrics(engine_rows(rec, ns, "pred_ta"))
        mb = metrics(engine_rows(rec, ns, "pred"))
        emit(f"| {b} | {len(ns)} | {mt['cer']:.3f} | {mta['cer']:.3f} | {mb['cer']:.3f} |")

    if args.markdown:
        pathlib.Path(args.markdown).write_text("\n".join(out) + "\n", encoding="utf-8")
        print("wrote", args.markdown)


if __name__ == "__main__":
    main()
