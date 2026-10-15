"""Export Tesseract LSTM training data (synthetic + real line crops).

Two modes (pilot provides synthetic; real needs manual line GT):
- synthetic: render GT unit lines (from units_core/commentary, train split adhikara 00)
  with composite fonts (Devanagari runs -> NotoSansDevanagari, Tamil runs -> NotoSansTamilUI)
  to training/local/tessdata/synth/*.png + *.box (Tesseract box format, char-level approx via PIL per-char bboxes)
  + *.txt (line transcription). Shareable, no scan dependency, script matches MDs/targets.
- real: segment train page images into lines via OpenCV projection, pair with GT lines
  by order (requires manual verify; pilot outputs crops + placeholder pairing, status=auto).

Box format: `<char> <left> <bottom> <right> <top> <page>` (origin bottom-left).
We approximate char boxes with PIL getbbox per char (monospace-ish, sufficient for pilot;
for production use text2image --outputbase which gives exact boxes).

Usage:
  python3 training/scripts/export_tesseract.py --mode synth --out training/local/tessdata
  python3 training/scripts/export_tesseract.py --mode real --out training/local/tessdata_real
"""
import argparse
import pathlib
import re
import json

BASE = pathlib.Path(__file__).resolve().parents[1]
FONT_DEVA = "/usr/share/fonts/sanskrit-fonts/ttf-devanagari/noto/NotoSansDevanagari-Regular.ttf"
FONT_TAMIL = "/usr/share/fonts/truetype/noto/NotoSansTamilUI-Regular.ttf"
FONT_LATIN = FONT_DEVA  # fallback

DEV_RE = re.compile(r"[\u0900-\u097F]+")
TAM_RE = re.compile(r"[\u0B80-\u0BFF]+")

def pick_font(ch):
    o = ord(ch)
    if 0x0900 <= o <= 0x097F:
        return FONT_DEVA
    if 0x0B80 <= o <= 0x0BFF:
        return FONT_TAMIL
    return FONT_LATIN

def render_mixed_line(line, fontsize=32, pad=10):
    from PIL import Image, ImageDraw, ImageFont
    # split into runs by script to use correct font
    runs = []
    buf, cur = "", None
    for ch in line:
        kind = "deva" if 0x0900 <= ord(ch) <= 0x097F else ("tam" if 0x0B80 <= ord(ch) <= 0x0BFF else "lat")
        if cur is None:
            cur = kind
        if kind != cur:
            runs.append((buf, cur))
            buf, cur = "", kind
        buf += ch
    if buf:
        runs.append((buf, cur))
    fonts = {"deva": ImageFont.truetype(FONT_DEVA, fontsize),
             "tam": ImageFont.truetype(FONT_TAMIL, fontsize),
             "lat": ImageFont.truetype(FONT_LATIN, fontsize)}
    # measure
    widths = []
    for txt, kind in runs:
        f = fonts[kind]
        widths.append(f.getlength(txt))
    W = int(sum(widths) + 2 * pad)
    H = fontsize + 2 * pad + 10
    img = Image.new("RGB", (max(W, 10), H), "white")
    d = ImageDraw.Draw(img)
    x = pad
    # char boxes (Tesseract origin bottom-left)
    boxes = []
    for (txt, kind), _w in zip(runs, widths):
        f = fonts[kind]
        for ch in txt:
            cw = f.getlength(ch)
            # PIL anchor: draw at (x, pad); bbox approx
            d.text((x, pad), ch, font=f, fill="black")
            x0, y0 = int(x), pad
            x1, y1 = int(x + cw), pad + fontsize
            # convert to bottom-left origin
            boxes.append((ch, x0, H - y1, x1, H - y0))
            x += cw
    return img, boxes

def load_train_units():
    import json as js
    units = []
    for kind in ["core", "commentary"]:
        for line in open(BASE / "metadata" / f"units_{kind}.jsonl", encoding="utf-8"):
            u = js.loads(line)
            # pilot train: adhikara 00 only (printed 1-32); full run uses train_split
            if u["unit_id"].startswith("00-"):
                units.append((kind, u))
    return units

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synth", "real"], default="synth")
    ap.add_argument("--out", default="training/local/tessdata")
    ap.add_argument("--fontsize", type=int, default=32)
    ap.add_argument("--max_lines", type=int, default=200)
    args = ap.parse_args()
    # --out is relative to repo root (maNipravALa-ocr/), e.g. training/local/tessdata
    REPO = BASE.parent if BASE.name == "training" else BASE
    out_dir = (REPO / args.out) if not pathlib.Path(args.out).is_absolute() else pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "synth":
        units = load_train_units()
        n = 0
        for kind, u in units:
            # split body into lines: by newline, then chunk long paras to ~120 chars
            paras = [p.strip() for p in u["body"].split("\n") if p.strip()]
            for pi, para in enumerate(paras):
                # chunk
                while len(para) > 140:
                    cut = para.rfind(" ", 0, 140)
                    if cut < 40:
                        cut = 140
                    chunk, para = para[:cut], para[cut:].strip()
                    img, boxes = render_mixed_line(chunk, fontsize=args.fontsize)
                    stem = f"synth_{u['unit_id'].replace('-','_')}_{kind}_{pi}_{n}"
                    img.save(out_dir / f"{stem}.png")
                    (out_dir / f"{stem}.txt").write_text(chunk, encoding="utf-8")
                    with open(out_dir / f"{stem}.box", "w", encoding="utf-8") as f:
                        for ch, x0, y0, x1, y1 in boxes:
                            if ch.strip() == "":
                                continue
                            f.write(f"{ch} {x0} {y0} {x1} {y1} 0\n")
                    n += 1
                    if n >= args.max_lines:
                        break
                else:
                    img, boxes = render_mixed_line(para, fontsize=args.fontsize)
                    stem = f"synth_{u['unit_id'].replace('-','_')}_{kind}_{pi}_{n}"
                    img.save(out_dir / f"{stem}.png")
                    (out_dir / f"{stem}.txt").write_text(para, encoding="utf-8")
                    with open(out_dir / f"{stem}.box", "w", encoding="utf-8") as f:
                        for ch, x0, y0, x1, y1 in boxes:
                            if ch.strip() == "":
                                continue
                            f.write(f"{ch} {x0} {y0} {x1} {y1} 0\n")
                    n += 1
                if n >= args.max_lines:
                    break
            if n >= args.max_lines:
                break
        print(f"synth: wrote {n} lines to {out_dir} (train adhikara 00, composite Deva+Tamil fonts)")
        print("Next: box/TIF -> lstmf via tesstrain (tesseract ... lstm.train), then lstmtraining --model_output ...")
    else:
        print("real mode TODO: segment training/images pages to lines (OpenCV projection) + DP-align to GT lines.")
        print("Pilot provides page-level GT (units) + baseline OCR; manual line split required for production Tesseract training.")
        print("Use evaluation/ pages (p0042-44) only for eval, never for real training.")

if __name__ == "__main__":
    main()
