#!/usr/bin/env python3
"""Build line-level OCR train/eval data from block crops.

Per crop:
  - TSV from ocr_tsv.py -> line boxes + rough per-line OCR text.
  - GT = the crop's make_gt slice (read from the gt file, byte-exact body substring).
  - Per line GT: token stream of the slice assigned to lines by proportional init
    (OCR token counts) + rapidfuzz partial_ratio_alignment within a monotonic
    window per line; lines with < 2 tokens / score < 40 are dropped or fall back
    to proportional boundaries.

Writes (training/local/lines):
  {split}/{name}.png + {split}/{name}.gt.txt
  manifest.jsonl rows {name, split, vol, page, block, unit, line_idx, score}
  {split}_files.txt: image paths for ketos train/test
"""
import argparse
import csv
import json
import pathlib
import re
import sys
from bisect import bisect_right

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from make_gt import normalize, token_stream  # noqa: E402
from rapidfuzz import fuzz  # noqa: E402
from PIL import Image  # noqa: E402

BASE = pathlib.Path(__file__).resolve().parents[1]
REPO = BASE.parent if BASE.name == "training" else BASE
OUT = REPO / "training/local/lines"


def read_tsv_lines(path):
    lines = {}
    with open(path, encoding="utf-8") as f:
        rd = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(rd, None)
        for row in rd:
            if len(row) < 12:
                continue
            lvl, key = row[0], (row[2], row[3], row[4])
            try:
                left, top, w, h = int(row[6]), int(row[7]), int(row[8]), int(row[9])
            except ValueError:
                continue
            text = row[11].strip()
            if lvl == "4":
                lines[key] = {"x0": left, "y0": top, "x1": left + w, "y1": top + h, "words": []}
            elif lvl == "5" and key in lines and text:
                lines[key]["words"].append(text)
    out = [v for v in lines.values()
           if v["x1"] > v["x0"] and v["y1"] - v["y0"] >= 8]
    out.sort(key=lambda v: v["y0"])
    merged = []
    for ln in out:
        m = merged[-1] if merged else None
        if m and ln["y0"] < m["y1"] - 5 and ln["x0"] <= m["x1"] + 40:
            m["x0"] = min(m["x0"], ln["x0"])
            m["x1"] = max(m["x1"], ln["x1"])
            m["y0"] = min(m["y0"], ln["y0"])
            m["y1"] = max(m["y1"], ln["y1"])
            m["words"] += ln["words"]
        else:
            merged.append(dict(ln))
    return merged


def align_lines(gt_text, lines):
    toks, spans = token_stream(gt_text)
    T = len(toks)
    if T == 0:
        return []
    ocr_norms = [normalize(" ".join(l["words"])) for l in lines]
    wcs = [max(len(o.split()), 1) for o in ocr_norms]
    tot = sum(wcs)
    cum, ests = 0, []
    for wc in wcs:
        cum += wc
        ests.append(round(T * cum / tot))
    out, cursor = [], 0
    for k, ln in enumerate(lines):
        if cursor >= T:
            break
        est_end = max(min(ests[k], T), cursor + 1)
        win_hi = min(T, max(est_end + 60, cursor + 1))
        ocr_n = ocr_norms[k]
        s0 = e0 = None
        ntok, score = 0, -1
        if len(ocr_n) >= 4:
            win_toks = toks[cursor:win_hi]
            los, p = [], 0
            for t in win_toks:
                los.append(p)
                p += len(t) + 1
            res = fuzz.partial_ratio_alignment(ocr_n, " ".join(win_toks))
            if res is not None:
                i = max(0, bisect_right(los, res.dest_start) - 1)
                j = min(len(win_toks) - 1,
                        max(i, bisect_right(los, max(res.dest_start, res.dest_end - 1)) - 1))
                s0, e0 = spans[cursor + i][0], spans[cursor + j][1]
                ntok, score = j - i + 1, res.score
        adv = cursor + ntok
        if ntok < 2 or score < 40:
            lo = ests[k - 1] if k else 0
            if lo < T:
                s0 = spans[lo][0]
                e0 = spans[min(T - 1, est_end - 1)][1]
                ntok = est_end - lo
                score = -2
                adv = est_end
            else:
                continue
        if ntok >= 2 and s0 is not None and s0 < e0:
            out.append({"line": ln, "gt": gt_text[s0:e0], "score": score, "ntok": ntok})
        cursor = max(cursor + 1, adv)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max crops per split (debug)")
    args = ap.parse_args()

    report = {}
    for vol in ("01", "02"):
        for r in csv.DictReader(open(REPO / f"training/metadata/gt_report_{vol}.csv")):
            report[(vol, r["page"], r["block"])] = r

    stats = {}
    for split, split_file in (("train", REPO / "training/split_train.jsonl"),
                              ("eval", REPO / "evaluation/split_eval.jsonl")):
        rows = [json.loads(l) for l in open(split_file)]
        if args.limit:
            rows = rows[:args.limit]
        (OUT / split).mkdir(parents=True, exist_ok=True)
        manifest = open(OUT / "manifest.jsonl", "a", encoding="utf-8")
        files = open(OUT / f"{split}_files.txt", "w", encoding="utf-8")
        kept = dropped = crops_skipped = 0
        scores = []
        for ri, row in enumerate(rows):
            vol = re.search(r"RTS_sb_(\d+)", row["image"]).group(1)
            tsv = REPO / f"training/metadata/tsv_{vol}/{row['page']}_{row['block']}.tsv"
            if not tsv.exists():
                crops_skipped += 1
                continue
            gt_text = open(REPO / row["gt"], encoding="utf-8").read()
            if not gt_text.strip():
                crops_skipped += 1
                continue
            lines = read_tsv_lines(tsv)
            if not lines:
                crops_skipped += 1
                continue
            aligned = align_lines(gt_text, lines)
            if not aligned:
                crops_skipped += 1
                continue
            img = Image.open(REPO / row["image"]).convert("L")
            W, H = img.size
            for k, a in enumerate(aligned):
                ln = a["line"]
                name = f"{vol}_{row['page']}_{row['block']}_L{k:02d}"
                x0 = max(0, ln["x0"] - 8)
                x1 = min(W, ln["x1"] + 8)
                y0 = max(0, ln["y0"] - 3)
                y1 = min(H, ln["y1"] + 3)
                if x1 - x0 < 20 or y1 - y0 < 8:
                    continue
                img.crop((x0, y0, x1, y1)).save(OUT / split / f"{name}.png")
                (OUT / split / f"{name}.gt.txt").write_text(a["gt"], encoding="utf-8")
                files.write(f"training/local/lines/{split}/{name}.png\n")
                manifest.write(json.dumps({
                    "name": name, "split": split, "vol": vol, "page": row["page"],
                    "block": row["block"], "unit": row.get("unit"), "line_idx": k,
                    "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                    "score": a["score"], "gt": a["gt"]}, ensure_ascii=False) + "\n")
                kept += 1
                if a["score"] >= 0:
                    scores.append(a["score"])
            dropped += len(lines) - len(aligned)
            if (ri + 1) % 100 == 0:
                print(f"  {split} {ri+1}/{len(rows)} crops, {kept} lines", flush=True)
        manifest.close()
        files.close()
        stats[split] = {"crops": len(rows), "skipped": crops_skipped, "lines": kept,
                        "dropped_lines": dropped,
                        "avg_score": round(sum(scores) / len(scores), 1) if scores else 0}
        print(split, stats[split], flush=True)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
