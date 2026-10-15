"""Baseline OCR with local Tesseract (tam+san) for alignment anchors.

Uses Tesseract binary extracted to /tmp/tess-root (no sudo):
  /tmp/tess-root/usr/bin/tesseract with TESSDATA_PREFIX=/tmp/tess-root/usr/share/tesseract-ocr/5/tessdata/
Langs: tam+san (add hin/eng if needed). PSM 6 (uniform block).

Usage:
  python3 scripts/baseline_ocr.py --images images/sarabodhini-01 --glob "p00*.jpg" --out metadata/baseline_01 --psm 6 -l tam+san
Outputs per image: <name>.txt, <name>.hocr (with bboxes), <name>.tsv (word boxes+conf)
Also writes metadata/baseline_index.csv (image, txt, mean_conf, char_count).
"""
import argparse
import csv
import glob
import os
import pathlib
import subprocess

TESS_BIN = "/tmp/tess-root/usr/bin/tesseract"
TESSDATA = "/tmp/tess-root/usr/share/tesseract-ocr/5/tessdata"

BASE = pathlib.Path(__file__).resolve().parents[1]

def ocr_image(img_path, out_base, lang="tam+san", psm=6):
    env = dict(os.environ)
    env["TESSDATA_PREFIX"] = TESSDATA
    # txt
    cmd_txt = [TESS_BIN, str(img_path), str(out_base), "-l", lang, "--psm", str(psm)]
    r1 = subprocess.run(cmd_txt, capture_output=True, text=True, env=env)
    # hocr for boxes
    cmd_hocr = [TESS_BIN, str(img_path), str(out_base), "-l", lang, "--psm", str(psm), "hocr"]
    r2 = subprocess.run(cmd_hocr, capture_output=True, text=True, env=env)
    # tsv for conf
    cmd_tsv = [TESS_BIN, str(img_path), str(out_base), "-l", lang, "--psm", str(psm), "tsv"]
    r3 = subprocess.run(cmd_tsv, capture_output=True, text=True, env=env)
    return r1, r2, r3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--glob", default="*.jpg")
    ap.add_argument("--out", required=True)
    ap.add_argument("-l", default="tam+san")
    ap.add_argument("--psm", type=int, default=6)
    args = ap.parse_args()
    img_dir = BASE / args.images if not os.path.isabs(args.images) else pathlib.Path(args.images)
    out_dir = BASE / args.out if not os.path.isabs(args.out) else pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(img_dir / args.glob)))
    print(f"OCR {len(files)} images with -l {args.l} psm {args.psm}")
    rows = []
    for f in files:
        stem = pathlib.Path(f).stem
        out_base = out_dir / stem
        ocr_image(f, out_base, lang=args.l, psm=args.psm)
        txt_p = out_dir / (stem + ".txt")
        txt = txt_p.read_text(encoding="utf-8", errors="ignore") if txt_p.exists() else ""
        # mean conf from tsv
        tsv_p = out_dir / (stem + ".tsv")
        confs = []
        if tsv_p.exists():
            import csv as _csv
            with open(tsv_p, encoding="utf-8", errors="ignore") as fh:
                rd = _csv.DictReader(fh, delimiter="\t")
                for r in rd:
                    try:
                        c = float(r.get("conf", -1))
                        if c >= 0:
                            confs.append(c)
                    except Exception:
                        pass
        mean_conf = sum(confs) / len(confs) if confs else -1
        rows.append({"image": os.path.basename(f), "txt": os.path.basename(str(txt_p)), "chars": len(txt), "mean_conf": round(mean_conf, 1)})
        print(f"  {stem}: chars={len(txt)} mean_conf={mean_conf:.1f}")
    with open(out_dir.parent / (pathlib.Path(args.out).name + "_index.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "txt", "chars", "mean_conf"])
        w.writeheader()
        w.writerows(rows)

if __name__ == "__main__":
    main()
