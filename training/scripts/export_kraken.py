"""Export Kraken PAGE-XML training data (line polygons + unicode).

Pilot: synthetic lines (same composite rendering as export_tesseract) + PAGE-XML,
plus real page crops with auto line polygons (OpenCV projection) awaiting manual transcription.
Kraken ketos train format: `ketos train -f page -t *.xml` (image + PAGE-XML with <TextLine> + <Unicode>).

Usage:
  python3 training/scripts/export_kraken.py --mode synth --out training/local/kraken --max_lines 200
PAGE-XML skeleton per line image: <PcGts><Page imageFilename><TextRegion><TextLine><Coords points><Baseline><TextEquiv><Unicode>GT</Unicode>
Real mode: segments training page images to line polygons (projection), writes XML with empty/OCR transcription
(status=auto, needs manual correction; never use evaluation/ images here).
"""
import argparse
import pathlib

BASE = pathlib.Path(__file__).resolve().parents[1]

def page_xml(image_name, w, h, lines):
    # lines: list of (polygon_points_str, baseline_points_str, text)
    tl = []
    for i, (poly, base, txt) in enumerate(lines):
        txt_esc = txt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        tl.append(f"""    <TextLine id="l{i}">
      <Coords points="{poly}"/>
      <Baseline points="{base}"/>
      <TextEquiv><Unicode>{txt_esc}</Unicode></TextEquiv>
    </TextLine>""")
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
  <Page imageFilename="{image_name}" imageWidth="{w}" imageHeight="{h}">
    <TextRegion id="r0">
""" + "\n".join(tl) + """
    </TextRegion>
  </Page>
</PcGts>
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synth", "real"], default="synth")
    ap.add_argument("--out", default="training/local/kraken")
    ap.add_argument("--max_lines", type=int, default=200)
    ap.add_argument("--fontsize", type=int, default=32)
    args = ap.parse_args()
    REPO = BASE.parent if BASE.name == "training" else BASE
    out = (REPO / args.out) if not pathlib.Path(args.out).is_absolute() else pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == "synth":
        import sys
        sys.path.insert(0, str(BASE / "scripts"))
        from export_tesseract import render_mixed_line, load_train_units
        # reimport via path (fallback: duplicate minimal render)
        n = 0
        units = load_train_units()
        for kind, u in units:
            paras = [p.strip() for p in u["body"].split("\n") if p.strip()]
            for para in paras[:4]:
                chunk = para[:140]
                try:
                    img, _ = render_mixed_line(chunk, fontsize=args.fontsize)
                except Exception as e:
                    print("render fail", e)
                    continue
                stem = f"kraken_synth_{u['unit_id'].replace('-','_')}_{n}"
                img.save(out / f"{stem}.png")
                w, h = img.size
                poly = f"0,0 {w},0 {w},{h} 0,{h}"
                base = f"0,{h-12} {w},{h-12}"
                xml = page_xml(f"{stem}.png", w, h, [(poly, base, chunk)])
                (out / f"{stem}.xml").write_text(xml, encoding="utf-8")
                n += 1
                if n >= args.max_lines:
                    break
            if n >= args.max_lines:
                break
        print(f"kraken synth: {n} pairs in {out}; train: ketos train -f page -t {out}/*.xml")
    else:
        print("kraken real TODO: OpenCV line polygons from training/images + DP-align GT; keep evaluation/ strictly separate.")

if __name__ == "__main__":
    main()
