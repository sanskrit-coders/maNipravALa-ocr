"""Fuzzy-align page OCR anchors to core/commentary MD units.

Problem: MDs (2-script) use Devanagari for Sanskrit stems (e.g. ईश्वरमुनि),
while scans sometimes use Tamil script for same stems (ஈச்வரமுனி).
Direct string match fails. Normalize by transliterating Devanagari runs -> Tamil,
then compare.

Also baseline Tesseract tam+san OCR is noisy (esp. mixed words like श्रीपादத்தை),
so use token-set fuzzy (rapidfuzz) on normalized text.

Inputs:
  metadata/units_core.jsonl, metadata/units_commentary.jsonl (from parse_mds.py)
  metadata/baseline_01/*.txt (from baseline_ocr.py)

Output:
  metadata/align_pilot.csv (page, best_core_unit, core_score, best_comm_unit, comm_score)
  metadata/align_details/<page>.json (top-5 candidates each)

Usage:
  python3 scripts/align.py --baseline metadata/baseline_01 --out metadata/align_pilot.csv --topk 5
"""
import argparse
import json
import pathlib
import re
import glob

BASE = pathlib.Path(__file__).resolve().parents[1]

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    HAS_INDIC = True
except ImportError:
    HAS_INDIC = False

DEV_RE = re.compile(r"[\u0900-\u097F]+")

def deva_to_tamil(s):
    """Transliterate Devanagari runs to Tamil, keep rest."""
    if not HAS_INDIC:
        return s
    def repl(m):
        try:
            return transliterate(m.group(0), sanscript.DEVANAGARI, sanscript.TAMIL)
        except Exception:
            return m.group(0)
    return DEV_RE.sub(repl, s)

def normalize(s):
    s = deva_to_tamil(s)
    # remove punctuation/symbols but keep Tamil, digits; collapse spaces
    s = re.sub(r"[॥।।०-९0-9॥।\-–—\"'“”‘’\(\)\[\]\{\}\|/\\@#\$%\^&\*\+=~`<>;:,\.!?।॥]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_units(jsonl_path):
    units = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            units.append(json.loads(line))
    return units

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="metadata/baseline_01")
    ap.add_argument("--out", default="metadata/align_pilot.csv")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--pages", default="", help="comma list of stems e.g. p0035,p0036 or empty=all")
    args = ap.parse_args()
    if fuzz is None:
        raise SystemExit("pip install rapidfuzz")

    core_units = load_units(BASE / "metadata" / "units_core.jsonl")
    comm_units = load_units(BASE / "metadata" / "units_commentary.jsonl")
    # pre-normalize units (cache)
    for u in core_units:
        u["_norm"] = normalize(u["body"])[:8000]
    for u in comm_units:
        u["_norm"] = normalize(u["body"])[:12000]

    base_dir = BASE / args.baseline if not pathlib.Path(args.baseline).is_absolute() else pathlib.Path(args.baseline)
    txts = sorted(glob.glob(str(base_dir / "*.txt")))
    if args.pages:
        want = set(args.pages.split(","))
        txts = [t for t in txts if pathlib.Path(t).stem in want]

    import csv
    out_path = BASE / args.out if not pathlib.Path(args.out).is_absolute() else pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    details_dir = out_path.parent / (out_path.stem + "_details")
    details_dir.mkdir(exist_ok=True)

    rows = []
    for txt_p in txts:
        stem = pathlib.Path(txt_p).stem
        ocr_raw = pathlib.Path(txt_p).read_text(encoding="utf-8", errors="ignore")
        ocr_norm = normalize(ocr_raw)[:8000]
        if len(ocr_norm) < 50:
            print(f"{stem}: OCR too short ({len(ocr_norm)}), skip")
            continue
        # score vs core (short units, use partial_ratio) and comm (long, token_set)
        core_scores = []
        for u in core_units:
            s = fuzz.token_set_ratio(ocr_norm[:4000], u["_norm"][:4000])
            core_scores.append((s, u["unit_id"]))
        core_scores.sort(reverse=True)
        comm_scores = []
        for u in comm_units:
            s = fuzz.token_set_ratio(ocr_norm, u["_norm"])
            comm_scores.append((s, u["unit_id"]))
        comm_scores.sort(reverse=True)
        rows.append({
            "page": stem,
            "best_core": core_scores[0][1], "core_score": round(core_scores[0][0], 1),
            "best_comm": comm_scores[0][1], "comm_score": round(comm_scores[0][0], 1),
            "ocr_chars": len(ocr_raw),
        })
        det = {
            "page": stem,
            "top_core": [{"unit": u, "score": round(s, 1)} for s, u in core_scores[:args.topk]],
            "top_comm": [{"unit": u, "score": round(s, 1)} for s, u in comm_scores[:args.topk]],
        }
        (details_dir / f"{stem}.json").write_text(json.dumps(det, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{stem}: core={core_scores[0][1]}:{core_scores[0][0]:.0f} comm={comm_scores[0][1]}:{comm_scores[0][0]:.0f}")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["page", "best_core", "core_score", "best_comm", "comm_score", "ocr_chars"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_path} + {details_dir}/")

if __name__ == "__main__":
    main()
