"""Detect core (top) vs commentary (body) blocks; write canonical data volumes (GT later).

Layout: URL (<4%), header + double rules (~4-10%), [core + rule when present], comm, footer (>94%).
Method: OTSU inv + horizontal MORPH_OPEN (kw=0.3w), keep cw>0.3w, merge collinear segs (dy<=6),
header = lines yc<14%, separator = first merged line yc in [12%,92%] width>0.35w.
Verifed on eval: p0042 none; p0043 sep y=245 (yc .138); p0044 sep y=453 (yc .255).

Usage (canonical):
  python3 training/scripts/detect_blocks.py --vol 01 --subset 0:70
  python3 training/scripts/detect_blocks.py --vol 02 --subset 0:70
Writes data/RTS_sb_0V/images/{core,comm}/pXXXX.jpg + data/RTS_sb_0V/metadata/blocks.json (merged).
GT deferred (no text here). Legacy --images/--out still supported.
"""
import argparse
import json
import glob
import os
import pathlib
import cv2
import numpy as np

BASE = pathlib.Path(__file__).resolve().parents[1]
REPO = BASE.parent if BASE.name == "training" else BASE

def find_hlines(gray):
    h, w = gray.shape
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kw = max(80, int(w * 0.3))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(horiz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segs = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw > w * 0.15 and 1 <= ch <= 8:
            segs.append({"x": x, "y": y, "w": cw, "h": ch, "yc": y + ch / 2})
    # merge collinear segs (same line fragmented by scan skew/text): dy<=6 and x-overlap/gap small
    segs.sort(key=lambda d: (d["yc"], d["x"]))
    merged = []
    for s in segs:
        placed = False
        for m in merged:
            if abs(s["yc"] - m["yc"]) <= 6 and s["x"] <= m["x"] + m["w"] + 20 and m["x"] <= s["x"] + s["w"] + 20:
                x0 = min(m["x"], s["x"])
                x1 = max(m["x"] + m["w"], s["x"] + s["w"])
                m.update({"x": x0, "w": x1 - x0, "y": min(m["y"], s["y"]),
                          "h": max(m["y"] + m["h"], s["y"] + s["h"]) - min(m["y"], s["y"])})
                m["yc"] = m["y"] + m["h"] / 2
                placed = True
                break
        if not placed:
            merged.append(dict(s))
    lines = [m for m in merged if m["w"] > w * 0.3]
    lines.sort(key=lambda d: d["yc"])
    # raw fragments for faded-header fallback (e.g. vol02 p0057 header rules survive only
    # as ~200px segments invisible to the 0.3-width opening which erodes them away).
    # Needs its own opening with a smaller kernel, else short segs never become contours.
    kw2 = max(60, int(w * 0.12))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (kw2, 1))
    horiz2 = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel2)
    contours2, _ = cv2.findContours(horiz2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frags = []
    for cnt in contours2:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw > w * 0.12 and 1 <= ch <= 8:
            frags.append({"x": x, "y": y, "w": cw, "h": ch, "yc": y + ch / 2})
    return lines, frags, w, h

def split_page(img_path, out_dir, stem):
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lines, frags, w, h = find_hlines(gray)
    # Header bottom: below detected double-rules when present; else just below the
    # archive.org URL (~2%) with margin. Fixed 10% fallback CUT the first text line
    # on front-matter pages without running headers (text starts ~7-8%), so use 4.5%.
    # Faded header rules (e.g. vol02 p0057: only ~200px fragments survive) would still
    # leak into the crop, so honour any fragmentary header lines too (max bottom + 5).
    header_bottom = int(h * 0.045) + 8
    # Separator candidates first: header lines must lie strictly ABOVE the separator.
    # (A high separator, e.g. vol01 p0043 at yc=13.8%, would otherwise be counted as a
    # header line and swallow the core block.)
    seps = [L for L in lines if h * 0.12 < L["yc"] < h * 0.92 and L["w"] > w * 0.35]
    sep_yc = seps[0]["yc"] if seps else None
    # find header lines strictly above the separator (and <14%h)
    hlim = h * 0.14 if sep_yc is None else min(h * 0.14, sep_yc - 10)
    top_lines = [L for L in lines if L["yc"] < hlim]
    if len(top_lines) >= 2:
        header_bottom = max(L["y"] + L["h"] for L in top_lines[:3]) + 5
    elif len(top_lines) == 1:
        header_bottom = top_lines[0]["y"] + top_lines[0]["h"] + 5
    else:
        frag_tops = [f for f in frags if f["yc"] < h * 0.14]
        if frag_tops:
            header_bottom = max(header_bottom, max(f["y"] + f["h"] for f in frag_tops) + 5)
    footer_top = int(h * 0.94)
    # (seps precomputed above so header lines stay strictly above the separator)
    blocks = {"header_lines": top_lines, "all_lines": lines, "w": w, "h": h}
    core_box = None
    comm_box = None
    if seps:
        sep = seps[0]
        # Pad around the rule itself (scans show 2-4px halo beyond the detected edge;
        # without padding the rule remnant leaks into crops, e.g. vol02 p0057/p0082).
        # core = header_bottom .. sep top (if height > 30px, else no core)
        if sep["y"] - 3 - header_bottom > 40:
            core_box = {"x0": 0, "y0": header_bottom, "x1": w, "y1": sep["y"] - 3, "label": "core"}
        comm_box = {"x0": 0, "y0": sep["y"] + sep["h"] + 10, "x1": w, "y1": footer_top, "label": "commentary"}
        blocks["separator"] = sep
    else:
        comm_box = {"x0": 0, "y0": header_bottom, "x1": w, "y1": footer_top, "label": "commentary"}
        blocks["separator"] = None
    blocks["core_box"] = core_box
    blocks["comm_box"] = comm_box
    # save crops (skip when out_dir None: caller writes canonical crops itself)
    if out_dir is None:
        return blocks
    out_dir = pathlib.Path(out_dir)
    if core_box:
        crop = img[core_box["y0"]:core_box["y1"], core_box["x0"]:core_box["x1"]]
        cv2.imwrite(str(out_dir / f"{stem}_core.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if comm_box:
        crop = img[comm_box["y0"]:comm_box["y1"], comm_box["x0"]:comm_box["x1"]]
        cv2.imwrite(str(out_dir / f"{stem}_comm.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return blocks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol", choices=["01", "02"], default="")
    ap.add_argument("--subset", type=str, default="", help="e.g. 0:70")
    ap.add_argument("--images", default="", help="legacy input dir (repo-relative or abs)")
    ap.add_argument("--glob", default="*.jpg")
    ap.add_argument("--out", default="", help="legacy output dir")
    args = ap.parse_args()

    if args.vol:
        # canonical mode: pages_raw -> images/{core,comm} + metadata/blocks.json (GT later)
        from pathlib import Path as _P
        vol = args.vol
        pages_dir = REPO / "data" / f"RTS_sb_{vol}" / "pages_raw"
        core_dir = REPO / "data" / f"RTS_sb_{vol}" / "images" / "core"
        comm_dir = REPO / "data" / f"RTS_sb_{vol}" / "images" / "comm"
        core_dir.mkdir(parents=True, exist_ok=True)
        comm_dir.mkdir(parents=True, exist_ok=True)
        meta_path = REPO / "data" / f"RTS_sb_{vol}" / "metadata" / "blocks.json"
        existing = {}
        if meta_path.exists():
            try:
                existing = {f"{b['page']}_{b['block']}": b for b in json.loads(meta_path.read_text(encoding="utf-8")).get("blocks", [])}
            except Exception:
                existing = {}
        files = sorted(pages_dir.glob("p*.jpg"))
        if args.subset:
            a, b = args.subset.split(":")
            files = [f for f in files if int(f.stem[1:]) >= int(a) and int(f.stem[1:]) < int(b)]
        for f in files:
            stem = f.stem  # pXXXX
            blocks = split_page(str(f), None, stem)
            # write canonical crops
            cb, mb = blocks["core_box"], blocks["comm_box"]
            img = cv2.imread(str(f))
            if cb:
                crop = img[cb["y0"]:cb["y1"], cb["x0"]:cb["x1"]]
                cv2.imwrite(str(core_dir / f"{stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if mb:
                crop = img[mb["y0"]:mb["y1"], mb["x0"]:mb["x1"]]
                cv2.imwrite(str(comm_dir / f"{stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
            # Drop stale core entries (re-detect no longer finds a core: remove crop + record,
            # unless it is the curated absent-by-design marker which has no image/box).
            if not cb:
                prev = existing.get(f"{stem}_core")
                if prev and prev.get("image"):
                    try:
                        (REPO / prev["image"]).unlink(missing_ok=True)
                    except Exception:
                        pass
                    existing.pop(f"{stem}_core", None)
                    print(f"{vol}:{stem}: core dropped (stale crop removed)")
            for label, box in (("core", cb), ("comm", mb)):
                if not box:
                    continue
                # Preserve curated GT/status across re-runs (gold lives in splits + gt files;
                # re-detection must never silently wipe it back to auto-images-only).
                prev = existing.get(f"{stem}_{label}", {})
                entry = {"page": stem, "block": label,
                    "image": f"data/RTS_sb_{vol}/images/{label}/{stem}.jpg", "gt": prev.get("gt"),
                    "box": box, "separator": blocks.get("separator"),
                    "has_core": cb is not None, "status": prev.get("status", "auto-images-only"),
                    "note": prev.get("note", "images first, GT later")}
                if "unit" in prev:
                    entry["unit"] = prev["unit"]
                existing[f"{stem}_{label}"] = entry
            print(f"{vol}:{stem}: has_core={cb is not None} n_lines={len(blocks['all_lines'])}")
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps({"blocks": [existing[k] for k in sorted(existing)], "status": "images-only-GT-later"}, indent=1), encoding="utf-8")
        print(f"Wrote {meta_path} ({len(existing)} blocks)")
        return

    # legacy mode
    img_dir = REPO / args.images if args.images and not os.path.isabs(args.images) else pathlib.Path(args.images)
    out_dir = BASE / args.out if not os.path.isabs(args.out) else pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(exist_ok=True)
    files = sorted(glob.glob(str(img_dir / args.glob)))
    index = []
    for f in files:
        stem = pathlib.Path(f).stem
        blocks = split_page(f, crops_dir, stem)
        (out_dir / f"{stem}_blocks.json").write_text(json.dumps(blocks, indent=2), encoding="utf-8")
        index.append({"image": os.path.basename(f), "has_core": blocks["core_box"] is not None,
                      "core_box": blocks["core_box"], "comm_box": blocks["comm_box"]})
        print(f"{stem}: has_core={blocks['core_box'] is not None} n_lines={len(blocks['all_lines'])}")
    (out_dir / "blocks_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Wrote {len(index)} block JSONs to {out_dir}")

if __name__ == "__main__":
    main()
