"""Evaluate block-separated OCR (core vs commentary) on held-out eval set.

Eval layout (no ambiguity):
  evaluation/images/core/{page}.jpg  + evaluation/gt/core/{page}.txt   (only pages with core; p0042 has none)
  evaluation/images/comm/{page}.jpg  + evaluation/gt/comm/{page}.txt   (page-part GT, not full unit)
  evaluation/eval_index.jsonl lists explicit core/comm paths (core null when absent).
Preds (same split):
  evaluation/pred/core/{page}.txt, evaluation/pred/comm/{page}.txt
  (Tesseract tam+san PSM6 on crops; other engines write same paths).

Scores core and comm separately (jiwer CER/WER + rapidfuzz token_set on Deva->Tamil normalized text).
Core lines are short verses (100-418 chars); comm are pages-parts (1900-2330 chars).

Usage:
  python3 training/scripts/evaluate.py --pred evaluation/pred --out evaluation/results.json
  (--pred must contain core/ and comm/ subfolders with {page}.txt; reads GT via eval_index.jsonl)
"""
import argparse
import json
import pathlib
import re

BASE = pathlib.Path(__file__).resolve().parents[1]
REPO = BASE.parent if BASE.name == "training" else BASE

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None
try:
    import jiwer
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    HAS_INDIC = True
except ImportError:
    HAS_INDIC = False

DEV_RE = re.compile(r"[\u0900-\u097F]+")

def deva_to_tamil(s):
    if not HAS_INDIC:
        return s
    return DEV_RE.sub(lambda m: transliterate(m.group(0), sanscript.DEVANAGARI, sanscript.TAMIL), s)

def normalize(s):
    return re.sub(r"\s+", " ", deva_to_tamil(s)).strip()

def score_pair(pred, ref):
    pn, gn = normalize(pred), normalize(ref)
    r = {"pred_chars": len(pred), "ref_chars": len(ref)}
    if HAS_JIWER and pn and gn:
        try:
            r["cer"] = round(jiwer.cer(gn, pn), 4)
            r["wer"] = round(jiwer.wer(gn, pn), 4)
        except Exception as ex:
            r["cer"] = r["wer"] = None
            r["err"] = str(ex)
    if fuzz and pn and gn:
        r["fuzz_token_set"] = round(fuzz.token_set_ratio(pn[:8000], gn[:8000]), 1)
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="evaluation/pred")
    ap.add_argument("--eval_index", default="evaluation/eval_index.jsonl")
    ap.add_argument("--out", default="evaluation/results.json")
    args = ap.parse_args()

    def R(p):
        pp = pathlib.Path(p)
        return pp if pp.is_absolute() else (REPO / p)

    pred_root, idx_path, out_path = R(args.pred), R(args.eval_index), R(args.out)
    rows = []
    for line in open(idx_path, encoding="utf-8"):
        e = json.loads(line)
        page = e["page"]
        for block in ["core", "comm"]:
            gt_rel = e.get(f"{block}_gt")
            if not gt_rel:
                continue  # e.g. p0042 has no core
            gt_p = R(gt_rel)
            pred_p = pred_root / block / f"{page}.txt"
            if not pred_p.exists():
                print(f"{page}/{block}: missing pred {pred_p}, skip")
                continue
            pred = pred_p.read_text(encoding="utf-8", errors="ignore")
            ref = gt_p.read_text(encoding="utf-8", errors="ignore")
            s = score_pair(pred, ref)
            s.update({"page": page, "block": block, "unit": e.get("unit")})
            rows.append(s)
            print(f"{page}/{block}: CER={s.get('cer')} WER={s.get('wer')} fuzz={s.get('fuzz_token_set')} (pred {s['pred_chars']} vs ref {s['ref_chars']})")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    if rows:
        import statistics
        for block in ["core", "comm"]:
            cers = [r["cer"] for r in rows if r["block"] == block and r.get("cer") is not None]
            if cers:
                print(f"MEAN {block}: CER={statistics.mean(cers):.4f} over {len(cers)} pages")
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
