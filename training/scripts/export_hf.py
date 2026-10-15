"""Export HuggingFace datasets image-text pairs (TrOCR/Vision fine-tuning).

Pilot: synthetic line PNGs + metadata.jsonl {file_name, text} (HF `Image` + `Value`).
Real: line crops from training pages (OpenCV) + DP-aligned GT (TODO manual).

Layout (shareable, engine-agnostic):
  <out>/images/*.png + <out>/metadata.jsonl
HF load: `load_dataset('imagefolder', data_dir=..., split='train')` after adding metadata,
or custom script with `datasets.Features({'image': Image(), 'text': Value('string')})`.

Usage:
  python3 training/scripts/export_hf.py --out training/local/hf_trocr --max_lines 200
Never include evaluation/ images here; eval pairs go to evaluation/hf_eval (separate).
"""
import argparse
import json
import pathlib

BASE = pathlib.Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="training/local/hf_trocr")
    ap.add_argument("--max_lines", type=int, default=200)
    ap.add_argument("--fontsize", type=int, default=32)
    args = ap.parse_args()
    REPO = BASE.parent if BASE.name == "training" else BASE
    out = (REPO / args.out) if not pathlib.Path(args.out).is_absolute() else pathlib.Path(args.out)
    imgdir = out / "images"
    imgdir.mkdir(parents=True, exist_ok=True)

    import sys
    sys.path.insert(0, str(BASE / "scripts"))
    try:
        from export_tesseract import render_mixed_line, load_train_units
    except ImportError as e:
        print("import fail", e)
        return
    units = load_train_units()
    rows = []
    n = 0
    for kind, u in units:
        paras = [p.strip() for p in u["body"].split("\n") if p.strip()]
        for para in paras[:4]:
            chunk = para[:140]
            if len(chunk.strip()) < 5:
                continue
            img, _ = render_mixed_line(chunk, fontsize=args.fontsize)
            stem = f"hf_{u['unit_id'].replace('-','_')}_{n}.png"
            img.save(imgdir / stem)
            rows.append({"file_name": f"images/{stem}", "text": chunk})
            n += 1
            if n >= args.max_lines:
                break
        if n >= args.max_lines:
            break
    with open(out / "metadata.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # dataset script hint
    (out / "README.txt").write_text(
        "HF datasets: Features({'image': Image(), 'text': Value('string')}). "
        "TrOCR fine-tune: microsoft/trocr-base-printed + Seq2SeqTrainer, tokenizer handles Tamil+Deva as unicode. "
        "Needs GPU (Colab). Eval: evaluation/hf_eval (separate, never train on it).\n",
        encoding="utf-8")
    print(f"hf: {n} pairs in {out}")

if __name__ == "__main__":
    main()
