#!/usr/bin/env python3
"""Evaluate PaddleOCR -- the recognition engine behind BytesByJay/Indic-OCR -- on
the maNipravALa line eval split.

Context: Indic-OCR ships no trained weights (models/ is empty; the CNN script
classifier is randomly initialized in the repo; the TrOCR backend is the
English-only microsoft/trocr-base-handwritten). Its accuracy on Tamil/Devanagari
is therefore exactly PaddleOCR's pretrained accuracy, optionally wrapped in the
project's OpenCV preprocessing and script selection. This script measures that
directly per language (ta, hi); the "best" column gives the system an ORACLE
script selector (upper bound for Indic-OCR's untrained classifier).

Modes:
  rec : recognition only (det=False) on the line crop. Fair comparison to our
        line-level Tesseract/kraken evals (inputs are pre-segmented lines).
  det : exact Indic-OCR call path: PaddleOCR.ocr(img, cls=True) (det+rec).
  both: run both (dumps written for each).

Run inside the pinned venv (see setup_env.sh):
  /tmp/opencode/indicocr-venv/bin/python training/scripts/indic-ocr/eval_paddleocr.py \
      --mode both --langs ta,hi --dump evaluation/line_eval_paddleocr.jsonl
"""
import argparse
import json
import os
import pathlib
import re
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[3]
LINES = REPO / "training/local/lines"


def strip_ws(s):
    return re.sub(r"\s+", "", s)


def metrics(pairs):
    import jiwer
    raw_t = [t for t, _ in pairs]
    raw_h = [h for _, h in pairs]
    out = {
        "lines": len(pairs),
        "cer": jiwer.cer(raw_t, raw_h),
        "wer": jiwer.wer(raw_t, raw_h),
        "cer_nows": jiwer.cer([strip_ws(t) for t in raw_t], [strip_ws(h) for h in raw_h]),
        "wer_nows": jiwer.wer([strip_ws(t) for t in raw_t], [strip_ws(h) for h in raw_h]),
    }
    return out


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def parse_rec(out):
    """PaddleOCR .ocr(det=False) -> (text, mean_conf)."""
    texts, confs = [], []
    if not out:
        return "", 0.0
    for line in out:
        if not line:
            continue
        for seg in line:
            if seg and len(seg) >= 2:
                texts.append(str(seg[0]))
                confs.append(float(seg[1]))
    return norm(" ".join(texts)), (sum(confs) / len(confs) if confs else 0.0)


def parse_det(out):
    """PaddleOCR .ocr() (det+rec) -> (text, mean_conf)."""
    texts, confs = [], []
    if not out:
        return "", 0.0
    for page in out:
        if not page:
            continue
        for item in page:
            if item and len(item) >= 2 and item[1]:
                texts.append(str(item[1][0]))
                confs.append(float(item[1][1]))
    return norm(" ".join(texts)), (sum(confs) / len(confs) if confs else 0.0)


def load_rows(limit=0):
    rows = []
    for line in open(LINES / "manifest.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["split"] == "eval" and (LINES / "eval" / (r["name"] + ".png")).exists():
            rows.append(r)
    return rows[:limit] if limit else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rec", "det", "both"], default="rec")
    ap.add_argument("--langs", default="ta,hi")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dump", default="evaluation/line_eval_paddleocr.jsonl")
    ap.add_argument("--indic-repo", default="/tmp/opencode/indic-ocr")
    ap.add_argument("--preprocess", choices=["none", "repo"], default="none",
                    help="'repo' = Indic-OCR's ImagePreprocessor.preprocess_for_ocr")
    ap.add_argument("--cpu-threads", type=int, default=8)
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.cpu_threads))
    import jiwer
    import paddle
    import paddleocr
    from paddleocr import PaddleOCR
    print(f"paddleocr {paddleocr.__version__} / paddle {paddle.__version__} | mode={args.mode} "
          f"preprocess={args.preprocess} device={args.device}", flush=True)

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    engines = {}
    for lang in langs:
        engines[lang] = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=args.device == "gpu",
                                  show_log=False, cpu_threads=args.cpu_threads)
        print(f"engine ready: {lang}", flush=True)

    pp = None
    if args.preprocess == "repo":
        import importlib.util
        pp_file = pathlib.Path(args.indic_repo) / "src" / "preprocessing.py"
        spec_ = importlib.util.spec_from_file_location("indic_preprocessing", pp_file)
        mod = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(mod)
        pp = mod.ImagePreprocessor()
        print("using Indic-OCR preprocessing (preprocess_for_ocr)", flush=True)

    rows = load_rows(args.limit)
    print(f"{len(rows)} eval lines", flush=True)

    modes = ["rec", "det"] if args.mode == "both" else [args.mode]
    for mode in modes:
        results = []
        t0 = time.time()
        for k, r in enumerate(rows):
            p = LINES / "eval" / (r["name"] + ".png")
            img = str(p)
            if pp is not None:
                import cv2
                img = pp.preprocess_for_ocr(cv2.imread(str(p)))
            row = {"name": r["name"], "gt": r["gt"], "vol": r["vol"], "page": r["page"],
                   "block": r["block"], "unit": r.get("unit")}
            for lang in langs:
                if mode == "rec":
                    out = engines[lang].ocr(img, det=False, rec=True, cls=True)
                    text, conf = parse_rec(out)
                else:
                    out = engines[lang].ocr(img, cls=True)
                    text, conf = parse_det(out)
                row[f"pred_{lang}"] = text
                row[f"conf_{lang}"] = round(conf, 4)
            cers = {lang: jiwer.cer(row["gt"], row[f"pred_{lang}"]) for lang in langs}
            best = min(cers, key=cers.get)
            row["best_lang"] = best
            row["pred"] = row[f"pred_{best}"]
            results.append(row)
            if (k + 1) % 50 == 0:
                rate = (k + 1) / (time.time() - t0)
                print(f"  {mode} {k+1}/{len(rows)} ({rate:.1f}/s)", flush=True)

        summary = {"mode": mode, "preprocess": args.preprocess, "n": len(results)}
        for lang in langs:
            summary[lang] = metrics([(r["gt"], r[f"pred_{lang}"]) for r in results])
        summary["best_of"] = metrics([(r["gt"], r["pred"]) for r in results])
        summary["best_lang_counts"] = {lang: sum(1 for r in results if r["best_lang"] == lang)
                                       for lang in langs}
        print(json.dumps(summary, indent=1), flush=True)

        if args.dump:
            dump = pathlib.Path(args.dump)
            if mode == "det":
                dump = dump.with_name(dump.stem + "_det" + dump.suffix)
            dump.parent.mkdir(parents=True, exist_ok=True)
            with open(dump, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            json.dump(summary, open(str(dump) + ".summary.json", "w"), indent=1)
            print(f"dumped {dump}", flush=True)

        worst = sorted(results, key=lambda r: jiwer.cer(r["gt"], r["pred"]), reverse=True)[:5]
        for r in worst:
            print(f"  worst {r['name']} cer={jiwer.cer(r['gt'], r['pred']):.2f} best={r['best_lang']}",
                  flush=True)


if __name__ == "__main__":
    main()
