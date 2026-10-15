"""Export common shareable training rows (page/block level).

Reads:
  metadata/pages_01.csv, pages_02.csv (from render_pages.py)
  metadata/toc_printed_mapping.csv (printed -> adhikara)
  metadata/units_core.jsonl, units_commentary.jsonl
  metadata/align_*.csv (optional, for candidate units)
  metadata/blocks_*/blocks_index.json (optional, for boxes)

Writes:
  metadata/training_index.jsonl  rows:
    {image, pdf, pdf_index, printed_page, adhikara, adhikara_file,
     core_box, comm_box, core_units, comm_units, status}
  Core/comm texts are referenced (not duplicated) via texts/core/*.txt + units_*.jsonl
  to keep the repo small; resolved text can be materialized with --materialize.

TODO (for Tesseract/Kraken/TrOCR training):
  - export_tesseract.py: line crops (from hOCR TSV word boxes grouped to lines)
    + .box/.lstmf + train/val splits.
  - export_kraken.py: PAGE-XML (image + polygon + unicode).
  - export_hf.py: HF datasets image-text pairs (line level).
Line segmentation + DP-align of GT lines to OCR lines still needed; pilot uses page/block level.
"""
import argparse
import csv
import json
import pathlib

BASE = pathlib.Path(__file__).resolve().parents[1]

def load_mapping():
    m = {}
    with open(BASE / "metadata" / "toc_printed_mapping.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pdf = r["pdf"]
            for p in range(int(r["printed_start"]), int(r["printed_end"]) + 1):
                m[(pdf, str(p))] = (r["adhikara_file"], r["adhikara_num"])
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="metadata/training_index.jsonl")
    args = ap.parse_args()
    mapping = load_mapping()
    rows = []
    for pdf in ["01", "02"]:
        pages_csv = BASE / "metadata" / f"pages_{pdf}.csv"
        if not pages_csv.exists():
            continue
        with open(pages_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                printed = r["printed_page"]
                key = (pdf, printed) if printed else None
                adh_file, adh_num = mapping.get(key, ("", ""))
                rows.append({
                    "image": r["image"],
                    "pdf": pdf,
                    "pdf_index": int(r["pdf_index"]),
                    "printed_page": printed,
                    "adhikara": adh_num,
                    "adhikara_file": adh_file,
                    "core_box": None,
                    "comm_box": None,
                    "core_units": [],
                    "comm_units": [],
                    "status": "auto" if adh_num else "front_matter_or_blank",
                })
    out = BASE / args.out
    with open(out, "w", encoding="utf-8") as fw:
        for r in rows:
            fw.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {out} (adhikara resolved via toc mapping; boxes/units to be filled by align+blocks)")

if __name__ == "__main__":
    main()
