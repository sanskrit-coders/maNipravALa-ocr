#!/usr/bin/env python3
"""Evaluate scriptSepDriveOCR, linewiseScriptSepDriveOCR and local baselines
on the verified RTS_sb evaluation blocks (block-level GT).

For each block crop (core/comm):
  * re-run line/word segmentation and script classification (pipeline.py),
  * compose script-pure faux pages (sep) and one faux page per printed line
    (linewise), OCR them with Drive in one batch,
  * OCR whole lines, script runs and script segments locally with Tesseract
    (`tam+san` for mixed lines, `tam`/`san` for pure units) and with
    devTamCharsOCR,
  * reassemble and score all variants against the block GT (whitespace-stripped
    character CER).

Outputs: evaluation/sb_eval/{metrics.json, *.pred_*.txt, runs_*.jsonl}
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image
from rapidfuzz.distance import Levenshtein

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import pipeline as P  # noqa: E402
from train_lines2 import SUBSAMPLE, ids_to_text  # noqa: E402

REPO = P.REPO
TESS = "/tmp/tess-root/usr/bin/tesseract"
TESSDATA = "/tmp/tess-root/usr/share/tesseract-ocr/5/tessdata"

BLOCKS = [
    ("sb01_comm_p0042", "data/RTS_sb_01/images/comm/p0042.jpg",
     "data/RTS_sb_01/gt/comm/p0042.txt"),
    ("sb01_core_p0043", "data/RTS_sb_01/images/core/p0043.jpg",
     "data/RTS_sb_01/gt/core/p0043.txt"),
    ("sb01_comm_p0043", "data/RTS_sb_01/images/comm/p0043.jpg",
     "data/RTS_sb_01/gt/comm/p0043.txt"),
    ("sb01_core_p0044", "data/RTS_sb_01/images/core/p0044.jpg",
     "data/RTS_sb_01/gt/core/p0044.txt"),
    ("sb01_comm_p0044", "data/RTS_sb_01/images/comm/p0044.jpg",
     "data/RTS_sb_01/gt/comm/p0044.txt"),
]

EXAMPLES = [
    "01_p0043_core_L00",
    "01_p0042_comm_L00",
    "01_p0042_comm_L03",
    "01_p0042_comm_L12",
    "01_p0043_comm_L05",
    "01_p0044_core_L01",
    "01_p0042_comm_L24",
]

PAD = 6
BORDER = 8


def load_gray(path, scale=2):
    im = Image.open(REPO / path if not pathlib.Path(path).is_absolute() else path).convert("L")
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.LANCZOS)
    return np.asarray(im)


def crop(arr, box, pad=PAD, border=0):
    x0, y0, x1, y1 = box
    c = arr[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
    if border:
        c = np.pad(c, border, constant_values=255)
    return c


def tess_text(img, lang, psm=7):
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        Image.fromarray(img).save(path)
        env = dict(os.environ, TESSDATA_PREFIX=TESSDATA, OMP_THREAD_LIMIT="1")
        r = subprocess.run([TESS, path, "stdout", "-l", lang, "--psm", str(psm)],
                           capture_output=True, text=True, env=env)
        return " ".join(r.stdout.split())
    finally:
        os.unlink(path)


def load_dev(ckpt, device):
    from kraken.lib.vgsl import TorchVGSLModel
    from kraken.lib.dataset.utils import ImageInputTransforms

    ck = torch.load(REPO / ckpt, map_location="cpu", weights_only=False)
    model = TorchVGSLModel(vgsl=ck["spec"], mode="recognition")
    net = model.nn
    net.load_state_dict(ck["state_dict"])
    net.to(device).eval()
    meta = json.load(open(REPO / "training/local/lines_cache/train/meta.json"))
    rev = {v: ch for ch, vs in meta["c2l"].items() for v in vs}
    trans = ImageInputTransforms(1, 64, 0, 1, (16, 0), valid_norm=True)
    return net, rev, trans


def dev_texts(images, net, rev, trans, device, batch=20):
    items, metas = [], []
    for img in images:
        im = Image.fromarray(img)
        s = 128 / im.height
        im2 = im.resize((max(1, round(im.width * s)), 128), Image.LANCZOS)
        x = trans(im2)
        items.append(x)
        metas.append(int(x.shape[2]))
    out = [None] * len(items)
    order = sorted(range(len(items)), key=lambda j: metas[j])
    with torch.no_grad():
        for b in range(0, len(order), batch):
            idxs = order[b:b + batch]
            W = max(items[j].shape[2] for j in idxs)
            stack = torch.zeros(len(idxs), 1, 64, W)
            for r, j in enumerate(idxs):
                stack[r, 0, :, :items[j].shape[2]] = items[j][0]
            res = net(stack.to(device))
            lg = (res[0] if isinstance(res, tuple) else res).squeeze(2).permute(2, 0, 1)
            preds = lg.argmax(2)
            for r, j in enumerate(idxs):
                tlen = max(1, min(preds.shape[0], metas[j] // SUBSAMPLE))
                out[j] = ids_to_text(rev, preds[:tlen, r].tolist())
    return out


def make_segments(runs):
    segs, by_line = [], {}
    for i, r in enumerate(runs):
        by_line.setdefault(r["line"], []).append(i)
    for li in sorted(by_line):
        cur = []
        for i in by_line[li]:
            if cur and runs[i]["script"] != runs[cur[-1]]["script"]:
                segs.append({"line": li, "indices": cur})
                cur = []
            cur.append(i)
        if cur:
            segs.append({"line": li, "indices": cur})
    for k, s in enumerate(segs):
        rs = [runs[i] for i in s["indices"]]
        s["box"] = (min(r["x0"] for r in rs), min(r["y0"] for r in rs),
                    max(r["x1"] for r in rs), max(r["y1"] for r in rs))
        s["script"] = rs[0]["script"]
        s["idx"] = k
    return segs


def line_map(runs):
    by = {}
    for i, r in enumerate(runs):
        by.setdefault(r["line"], []).append(i)
    return {li: v for li, v in sorted(by.items())}


def assemble_runs(runs, text_of):
    out = []
    for li, idxs in line_map(runs).items():
        parts = []
        for k, i in enumerate(idxs):
            t = text_of(i) or ""
            if k > 0 and runs[i]["pre_space"]:
                parts.append(" ")
            parts.append(t)
        out.append("".join(parts))
    return out


def assemble_segments(runs, segs, text_of):
    by_line = {}
    for s in segs:
        by_line.setdefault(s["line"], []).append(s)
    out = []
    for li in sorted(by_line):
        parts = []
        for k, s in enumerate(by_line[li]):
            t = text_of(s) or ""
            if k > 0 and runs[s["indices"][0]]["pre_space"]:
                parts.append(" ")
            parts.append(t)
        out.append("".join(parts))
    return out


def score(gt, pred):
    g = "".join(gt.split())
    p = "".join(pred.split())
    d = Levenshtein.distance(g, p)
    gc, pc = " ".join(gt.split()), " ".join(pred.split())
    return {"cer": d / max(1, len(g)), "dist": d, "n": len(g),
            "cer_raw": Levenshtein.distance(gc, pc) / max(1, len(gc))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evaluation/sb_eval")
    ap.add_argument("--workdir", default="/tmp/opencode/sb_eval")
    ap.add_argument("--classifier", default="training/local/script_cls.pt")
    ap.add_argument("--ckpt", default="training/local/kaggle_v10_best.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--blocks", default="")
    ap.add_argument("--no-examples", action="store_true")
    args = ap.parse_args()

    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    work = pathlib.Path(args.workdir)
    faux = work / "faux"
    faux.mkdir(parents=True, exist_ok=True)

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cls_net, classes = P.load_classifier(REPO / args.classifier, dev)

    chosen = set(args.blocks.split(",")) if args.blocks else None
    units = []
    for name, img_path, gt_path in BLOCKS:
        if chosen and name not in chosen:
            continue
        arr = load_gray(img_path, args.scale)
        gt = (REPO / gt_path).read_text(encoding="utf-8")
        lines = P.segment_lines(Image.fromarray(arr), 6)
        dark = arr < 128
        runs = []
        for li, (lx0, ly0, lx1, ly1) in enumerate(lines):
            words = P.word_segments(dark, lx0, lx1, ly0, ly1)
            for wi, (wx0, wx1) in enumerate(words):
                rr = P.classify_word(cls_net, classes, arr, wx0, wx1, ly0, ly1, dev)
                if not rr:
                    continue
                pre_space = wi == 0 or (wx0 - words[wi - 1][1]) >= P.GAP_WORD
                for k, (script, rx0, rx1, ry0, ry1) in enumerate(rr):
                    runs.append({"line": li, "seq": len(runs), "script": script,
                                 "x0": rx0, "y0": ry0, "x1": rx1, "y1": ry1,
                                 "pre_space": pre_space if k == 0 else False})
        segs = make_segments(runs)
        unit = {"name": name, "kind": "block", "gt": gt, "arr": arr,
                "lines": lines, "runs": runs, "segs": segs,
                "sep": [], "linewise": []}
        for script in ("dev", "tam", "other"):
            subset = [i for i, r in enumerate(runs) if r["script"] == script]
            for ci in range(0, len(subset), 20):
                chunk = subset[ci:ci + 20]
                path = faux / f"{name}_sep_{script}_{ci:03d}.jpg"
                P.compose_faux(arr, [(i, runs[i]["x0"], runs[i]["x1"],
                                      runs[i]["y0"], runs[i]["y1"]) for i in chunk], path)
                unit["sep"].append((path, chunk))
        for li, idxs in line_map(runs).items():
            for ci in range(0, len(idxs), 20):
                chunk = idxs[ci:ci + 20]
                path = faux / f"{name}_line_L{li:03d}_{ci:03d}.jpg"
                P.compose_faux(arr, [(i, runs[i]["x0"], runs[i]["x1"],
                                      runs[i]["y0"], runs[i]["y1"]) for i in chunk], path)
                unit["linewise"].append((path, chunk))
        units.append(unit)
        print(f"{name}: {len(lines)} lines, {len(runs)} runs, {len(segs)} segments", flush=True)

    for ex in ([] if args.no_examples else EXAMPLES):
        img_path = REPO / f"training/local/lines/eval/{ex}.png"
        if not img_path.exists():
            print("skip example", ex, flush=True)
            continue
        arr = load_gray(str(img_path), args.scale)
        gt = (REPO / f"training/local/lines/eval/{ex}.gt.txt").read_text(encoding="utf-8")
        lines = P.segment_lines(Image.fromarray(arr), 6)
        dark = arr < 128
        runs = []
        for li, (lx0, ly0, lx1, ly1) in enumerate(lines):
            words = P.word_segments(dark, lx0, lx1, ly0, ly1)
            for wi, (wx0, wx1) in enumerate(words):
                rr = P.classify_word(cls_net, classes, arr, wx0, wx1, ly0, ly1, dev)
                if not rr:
                    continue
                pre_space = wi == 0 or (wx0 - words[wi - 1][1]) >= P.GAP_WORD
                for k, (script, rx0, rx1, ry0, ry1) in enumerate(rr):
                    runs.append({"line": li, "seq": len(runs), "script": script,
                                 "x0": rx0, "y0": ry0, "x1": rx1, "y1": ry1,
                                 "pre_space": pre_space if k == 0 else False})
        segs = make_segments(runs)
        unit = {"name": ex, "kind": "example", "gt": gt, "arr": arr,
                "lines": lines, "runs": runs, "segs": segs,
                "sep": [], "linewise": []}
        for li, idxs in line_map(runs).items():
            for ci in range(0, len(idxs), 20):
                chunk = idxs[ci:ci + 20]
                path = faux / f"ex_{ex}_L{li:03d}_{ci:03d}.jpg"
                P.compose_faux(arr, [(i, runs[i]["x0"], runs[i]["x1"],
                                      runs[i]["y0"], runs[i]["y1"]) for i in chunk], path)
                unit["linewise"].append((path, chunk))
        units.append(unit)
        print(f"{ex}: {len(lines)} lines, {len(runs)} runs", flush=True)

    # --- Drive batch ---
    files = [p for u in units for p, _ in u["sep"] + u["linewise"]]
    if files:
        print(f"Drive OCR: {len(files)} faux pages", flush=True)
        subprocess.run([P.VENV_PY, str(P.DRIVE_HELPER), "--key", P.GOOGLE_KEY]
                       + [str(p) for p in files], check=True)
    retry = []
    for u in units:
        for kind in ("sep", "linewise"):
            for path, chunk in u[kind]:
                txt = pathlib.Path(str(path) + ".txt")
                texts = (P.parse_tagged(txt.read_text(encoding="utf-8"), len(chunk))
                         if txt.exists() else [None] * len(chunk))
                missing = [k for k, t in enumerate(texts) if t is None]
                if missing:
                    rpath = faux / f"{path.stem}_retry.jpg"
                    P.compose_faux(u["arr"], [(chunk[m], u["runs"][chunk[m]]["x0"],
                                               u["runs"][chunk[m]]["x1"],
                                               u["runs"][chunk[m]]["y0"],
                                               u["runs"][chunk[m]]["y1"]) for m in missing],
                                   rpath, tag_start=1)
                    retry.append((rpath, missing, texts))
                u.setdefault("drive", {}).setdefault(kind, []).append((chunk, texts))
    if retry:
        print(f"Drive retry: {len(retry)} pages", flush=True)
        subprocess.run([P.VENV_PY, str(P.DRIVE_HELPER), "--key", P.GOOGLE_KEY]
                       + [str(p) for p, _, _ in retry], check=True)
        for rpath, missing, texts in retry:
            txt = pathlib.Path(str(rpath) + ".txt")
            rtexts = (P.parse_tagged(txt.read_text(encoding="utf-8"), len(missing))
                      if txt.exists() else [])
            for m, t in zip(missing, rtexts):
                if t is not None:
                    texts[m] = t

    for u in units:
        u["drive_sep"] = [None] * len(u["runs"])
        u["drive_line"] = [None] * len(u["runs"])
        for chunk, texts in u.get("drive", {}).get("sep", []):
            for i, t in zip(chunk, texts):
                u["drive_sep"][i] = t
        for chunk, texts in u.get("drive", {}).get("linewise", []):
            for i, t in zip(chunk, texts):
                u["drive_line"][i] = t

    # --- local engines ---
    net, rev, trans = load_dev(args.ckpt, dev)
    jobs = []
    for u in units:
        arr, runs = u["arr"], u["runs"]
        u["line_boxes"] = [(lx0, ly0, lx1, ly1) for (lx0, ly0, lx1, ly1) in u["lines"]]
        u["line_imgs"] = [crop(arr, (x0, y0, x1, y1), pad=2) for x0, y0, x1, y1 in u["line_boxes"]]
        u["run_imgs"] = [crop(arr, (r["x0"], r["y0"], r["x1"], r["y1"])) for r in runs]
        u["seg_imgs"] = [crop(arr, s["box"]) for s in u["segs"]]
        jobs += [(u, "tess_line", i, im, "tam+san") for i, im in enumerate(u["line_imgs"])]
        jobs += [(u, "tess_run", i, im, "san" if runs[i]["script"] == "dev" else
                  "tam" if runs[i]["script"] == "tam" else "tam+san")
                 for i, im in enumerate(u["run_imgs"])]
        jobs += [(u, "tess_seg", i, im, "san" if s["script"] == "dev" else
                  "tam" if s["script"] == "tam" else "tam+san")
                 for i, (s, im) in enumerate(zip(u["segs"], u["seg_imgs"]))]

    def work_one(job):
        u, key, i, im, lang = job
        return u["name"], key, i, tess_text(im, lang)

    with ThreadPoolExecutor(max_workers=8) as ex:
        for name, key, i, text in ex.map(work_one, jobs):
            u = next(x for x in units if x["name"] == name)
            u.setdefault(key, {})[i] = text

    for u in units:
        u["dev_line"] = dev_texts(u["line_imgs"], net, rev, trans, dev)
        u["dev_run"] = dev_texts(u["run_imgs"], net, rev, trans, dev)
        u["dev_seg"] = dev_texts(u["seg_imgs"], net, rev, trans, dev)

    variants, rows = {}, []
    for u in units:
        runs, segs = u["runs"], u["segs"]
        preds = {
            "tess_line": "\n".join(u["tess_line"][i] for i in range(len(u["line_imgs"]))),
            "dev_line": "\n".join(u["dev_line"]),
            "tess_run": "\n".join(assemble_runs(runs, lambda i: u["tess_run"].get(i))),
            "tess_seg": "\n".join(assemble_segments(runs, segs,
                                                    lambda s: u["tess_seg"].get(s["idx"]))),
            "dev_run": "\n".join(assemble_runs(runs, lambda i: u["dev_run"][i])),
            "dev_seg": "\n".join(assemble_segments(runs, segs,
                                                   lambda s: u["dev_seg"][s["idx"]])),
            "drive_sep": "\n".join(assemble_runs(runs, lambda i: u["drive_sep"][i])),
            "drive_line": "\n".join(assemble_runs(runs, lambda i: u["drive_line"][i])),
        }
        if u["kind"] == "example":
            preds.pop("drive_sep", None)
        sc = {k: score(u["gt"], v) for k, v in preds.items()}
        variants[u["name"]] = sc
        rows.append((u["name"], u["kind"], len(u["gt"]), {k: v["cer"] for k, v in sc.items()}))
        if u["kind"] == "block":
            for k, v in preds.items():
                (out_dir / f"{u['name']}.pred_{k}.txt").write_text(v + "\n", encoding="utf-8")
            with open(out_dir / f"runs_{u['name']}.jsonl", "w", encoding="utf-8") as f:
                for i, r in enumerate(runs):
                    rec = dict(r, tess_run=u["tess_run"].get(i), dev_run=u["dev_run"][i],
                               drive_sep=u["drive_sep"][i], drive_line=u["drive_line"][i])
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            (out_dir / f"example_{u['name']}.json").write_text(json.dumps({
                "gt": u["gt"], "n_lines": len(u["lines"]), "n_runs": len(runs),
                "preds": preds, "scores": sc}, ensure_ascii=False, indent=1), encoding="utf-8")

    block_names = [n for n, k, _, _ in rows if k == "block"]
    keys = list(rows[0][3])
    micro = {}
    for k in keys:
        d = sum(variants[n][k]["dist"] for n in block_names)
        n = sum(variants[n][k]["n"] for n in block_names)
        micro[k] = {"cer": d / max(1, n), "dist": d, "n": n}
    result = {"blocks": {n: variants[n] for n in block_names},
              "micro": micro,
              "examples": {n: {"scores": {k: v["cer"] for k, v in variants[n].items()}}
                           for n in variants if n not in block_names}}
    (out_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print(json.dumps({"micro": {k: round(v["cer"], 4) for k, v in micro.items()}}, indent=1))
    for n, k, ln, sc in rows:
        print(n, k, ln, " ".join(f"{x}={y:.3f}" for x, y in sc.items()), flush=True)


if __name__ == "__main__":
    main()
