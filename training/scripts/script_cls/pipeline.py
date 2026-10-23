#!/usr/bin/env python3
"""Script-run boxing + faux-page OCR pipeline.

Per target PDF page:
  1. line segmentation (same heuristic as apply_ocr_pdf.py),
  2. word segmentation (ink-gap projection inside each line),
  3. patch-level script classification dev/tam/other (sliding 48px patches),
     splitting mixed words at script transitions,
  4. compose script-pure faux pages (one run per line, wide spacing),
  5. OCR faux pages with Google Drive API (via the doc_curation venv helper); faux pages
     from up to --page-batch PDF pages are sent in one batch (with --linewise, faux pages
     are composed per printed line instead of per page),
  6. map OCR lines back to runs and reassemble per original line.

Outputs:
  {out}/{stem}_p{page:04d}.txt    reassembled page text
  {out}/runs.jsonl                per-run records (box, script, pre_space, text)
  {workdir}/faux/*.jpg            faux pages
  {workdir}/overlay/*.jpg         colored box overlay per page

Usage:
  python3 training/scripts/script_cls/pipeline.py --pdf ... --pages 100:102 \
      --out data/RTS_sb_01/ocr_scriptcls --workdir /tmp/opencode/scriptcls_run
"""
import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from apply_ocr_pdf import segment_lines  # noqa: E402

BASE = pathlib.Path(__file__).resolve().parents[2]
REPO = BASE.parent if BASE.name == "training" else BASE

VENV_PY = os.environ.get("DRIVE_PY", "/home/vvasuki/gitland/python_venv/bin/python")
DRIVE_HELPER = pathlib.Path(os.environ.get("DRIVE_HELPER", str(pathlib.Path(__file__).resolve().parent / "drive_ocr_images.py")))
GOOGLE_KEY = os.environ.get("GOOGLE_KEY", "/home/vvasuki/gitland/vvasuki-git/sysconf/kunchikA/google/proofing/ocr_app_oauth.json")
TAG_FONT = os.environ.get("TAG_FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
if not pathlib.Path(TAG_FONT).exists():
    TAG_FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"

SCRIPT_COLORS = {"dev": (200, 0, 0), "tam": (0, 120, 0), "other": (0, 0, 220)}
GAP_WORD = 10
STRIDE = 8
MIN_RUN = 20
TAG_RE = re.compile(r"#T[O0]?(\d{1,2})")
MD_ESC_RE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|])")


def load_classifier(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    from train_classifier import Net
    net = Net()
    net.load_state_dict(ck["state_dict"])
    net.to(device).eval()
    return net, ck["classes"]


def word_segments(dark, x0, x1, y0, y1):
    sub = dark[y0:y1, x0:x1]
    col = sub.any(axis=0)
    segs = []
    x = 0
    W = len(col)
    while x < W:
        if not col[x]:
            x += 1
            continue
        a = x
        while x < W and col[x]:
            x += 1
        while x < W:
            g, y = 0, x
            while y < W and not col[y]:
                g += 1
                y += 1
            if y < W and g < GAP_WORD:
                while y < W and col[y]:
                    y += 1
                x = y
            else:
                break
        if x - a >= 6:
            segs.append((x0 + a, x0 + x))
    return segs


WORD_H, WORD_W = 64, 256


class WordNet(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        def block(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(),
                                 nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(),
                                 nn.MaxPool2d(2))
        self.f = nn.Sequential(block(1, 32), block(32, 64), block(64, 128),
                               nn.AdaptiveAvgPool2d(1))
        self.o = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.o(self.f(x))


def load_word_classifier(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    classes = ck.get("classes", ["dev", "tam", "other"])
    net = WordNet(len(classes))
    net.load_state_dict(ck["state_dict"])
    net.to(device).eval()
    return net, classes


def _word_tensor(arr, x0, x1, y0, y1, target_h):
    sub = arr[y0:y1, x0:x1]
    dark = sub < 128
    rows = np.where(dark.any(axis=1))[0]
    cols = np.where(dark.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    x0 = x0 + int(cols[0]); x1 = x0 + int(cols[-1]) - int(cols[0]) + 1
    y0 = y0 + int(rows[0]); y1 = y0 + int(rows[-1]) - int(rows[0]) + 1
    crop = Image.fromarray(arr[y0:y1, x0:x1])
    w, h = crop.size
    nw = max(1, int(round(w * target_h / max(1, h))))
    if nw > WORD_W - 8:
        nw = WORD_W - 8
        target_h = max(1, int(round(h * nw / max(1, w))))
    crop = crop.resize((nw, target_h), Image.BILINEAR)
    canvas = np.full((WORD_H, WORD_W), 255, dtype=np.uint8)
    x = (WORD_W - nw) // 2; y = (WORD_H - target_h) // 2
    canvas[y:y + target_h, x:x + nw] = np.asarray(crop)
    return torch.from_numpy(1.0 - canvas.astype(np.float32) / 255.0).unsqueeze(0)


def word_label(net, classes, arr, x0, x1, y0, y1, dev):
    xs = []
    for h in (28, 36, 44):
        t = _word_tensor(arr, x0, x1, y0, y1, h)
        if t is not None: xs.append(t)
    if not xs:
        return None, 0.0
    xb = torch.stack(xs).to(dev)
    with torch.no_grad():
        p = net(xb).softmax(1).mean(0).cpu().numpy()
    return classes[int(p.argmax())], float(p.max())


LINE_H, LINE_W = 64, 1024


class LineNet(nn.Module):
    def __init__(self):
        super().__init__()
        def block(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(),
                                 nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(),
                                 nn.MaxPool2d(2))
        self.f = nn.Sequential(block(1, 32), block(32, 64), block(64, 128),
                               block(128, 128), nn.AdaptiveAvgPool2d(1))
        self.o = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, 3))

    def forward(self, x):
        return self.o(self.f(x))


def load_line_classifier(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = LineNet()
    net.load_state_dict(ck["state_dict"])
    net.to(device).eval()
    return net, ck.get("classes", ["dev", "tam", "other"])


def _line_tensor(arr, x0, y0, x1, y1, target_h):
    sub = arr[y0:y1, x0:x1]
    dark = sub < 128
    rows = np.where(dark.any(axis=1))[0]
    cols = np.where(dark.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    sub = sub[int(rows[0]):int(rows[-1]) + 1, int(cols[0]):int(cols[-1]) + 1]
    im = Image.fromarray(sub)
    w, h = im.size
    nw = max(1, int(round(w * target_h / max(1, h))))
    if nw > LINE_W - 8:
        nw = LINE_W - 8
        target_h = max(1, int(round(h * nw / max(1, w))))
    im = im.resize((nw, target_h), Image.BILINEAR)
    canvas = np.full((LINE_H, LINE_W), 255, dtype=np.uint8)
    x = (LINE_W - nw) // 2; y = (LINE_H - target_h) // 2
    canvas[y:y + target_h, x:x + nw] = np.asarray(im)
    return torch.from_numpy(1.0 - canvas.astype(np.float32) / 255.0).unsqueeze(0)


def line_label(net, classes, arr, x0, y0, x1, y1, dev):
    xs = []
    for h in (28, 36, 44):
        t = _line_tensor(arr, x0, y0, x1, y1, h)
        if t is not None: xs.append(t)
    if not xs:
        return None, 0.0
    with torch.no_grad():
        p = net(torch.stack(xs).to(dev)).softmax(1).mean(0).cpu().numpy()
    return classes[int(p.argmax())], float(p.max())


COL_H, COL_W = 64, 256


class ColumnNet(nn.Module):
    def __init__(self):
        super().__init__()
        def block(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(),
                                 nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU())
        self.e = nn.Sequential(block(1, 32), nn.MaxPool2d((2, 1)), block(32, 64),
                               nn.MaxPool2d((2, 1)), block(64, 128), nn.MaxPool2d((2, 1)),
                               block(128, 128), nn.AdaptiveAvgPool2d((1, COL_W)))
        self.o = nn.Sequential(nn.Conv1d(128, 64, 5, padding=2), nn.ReLU(), nn.Conv1d(64, 3, 1))

    def forward(self, x):
        return self.o(self.e(x).squeeze(2))


def load_column_classifier(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = ColumnNet()
    net.load_state_dict(ck["state_dict"])
    net.to(device).eval()
    return net, ck.get("classes", ["dev", "tam", "other"])


def _snap_split(dark, y0, y1, x, lo, hi, rad):
    best = x; bestv = 10**9
    for c in range(max(lo, x - rad), min(hi, x + rad) + 1):
        v = int(dark[y0:y1, c].sum())
        if v < bestv:
            bestv = v; best = c
    return best


BND_H, BND_W = 64, 256


class BoundaryNet(nn.Module):
    def __init__(self):
        super().__init__()
        def b(a, c):
            return nn.Sequential(nn.Conv2d(a, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
                                 nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU())
        self.e = nn.Sequential(b(1, 32), nn.MaxPool2d((2, 1)), b(32, 64),
                               nn.MaxPool2d((2, 1)), b(64, 128), nn.MaxPool2d((2, 1)),
                               b(128, 128), nn.AdaptiveAvgPool2d((1, BND_W)))
        self.o = nn.Sequential(nn.Conv1d(128, 64, 5, padding=2), nn.ReLU(), nn.Conv1d(64, 1, 1))

    def forward(self, x):
        return self.o(self.e(x).squeeze(2)).squeeze(1)


def load_boundary_classifier(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = BoundaryNet()
    net.load_state_dict(ck["state_dict"])
    net.to(device).eval()
    return net


def split_mixed_word(boundary_net, word_net, word_classes, arr, x0, x1, y0, y1, dev):
    if word_net is not None and boundary_net is not None:
        lab, conf = word_label(word_net, word_classes, arr, x0, x1, y0, y1, dev)
        if lab in ("dev", "tam") and conf >= 0.9:
            return _whole_word_run(arr, x0, x1, y0, y1, [[lab, x0, x1]])
    crop = arr[y0:y1, x0:x1]
    im = Image.fromarray(crop).resize((BND_W, BND_H), Image.BILINEAR)
    xb = torch.from_numpy(1.0 - np.asarray(im, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(dev)
    with torch.no_grad():
        p = boundary_net(xb).argmax(1).item()
    f = p / max(1, BND_W - 1)
    x = x0 + int(round(f * (x1 - x0)))
    dark = arr < 128
    x = _snap_split(dark, y0, y1, x, x0, x1, max(2, int(0.04 * (x1 - x0))))
    # dev part earlier, Tamil later
    rs = dark[y0:y1, x0:x]
    cs = np.where(rs.any(axis=0))[0]
    if len(cs) == 0: return []
    dx1 = x0 + int(cs[-1]) + 1
    rs2 = dark[y0:y1, x:]
    cs2 = np.where(rs2.any(axis=0))[0]
    if len(cs2) == 0: return []
    tx0 = x + int(cs2[0])
    out = []
    for lb, bx0, bx1 in (("dev", x0, dx1), ("tam", tx0, x1)):
        rs = dark[y0:y1, bx0:bx1]
        rows = np.where(rs.any(axis=1))[0]
        cols = np.where(rs.any(axis=0))[0]
        if len(rows) and len(cols):
            out.append((lb, bx0 + int(cols[0]), bx0 + int(cols[-1]) + 1,
                        y0 + int(rows[0]), y0 + int(rows[-1]) + 1))
    return out


def classify_word_components(word_net, word_classes, arr, x0, x1, y0, y1, dev, dark=None):
    """Split a word at connected-component / script boundaries.

    The 4-class word model labels large components; tiny marks (repha, anusvara)
    are absorbed by the nearest large component.  Grouping is contiguous, so a
    script change is always at a real glyph gap, never inside a glyph.
    """
    if dark is None:
        dark = arr < 128
    sub = dark[y0:y1, x0:x1]
    lab, n = ndimage.label(sub, structure=np.ones((3, 3)))
    comps = []
    for k in range(1, n + 1):
        ys, xs = np.where(lab == k)
        if len(xs) < 4: continue
        bx0, bx1 = x0 + int(xs.min()), x0 + int(xs.max()) + 1
        by0, by1 = y0 + int(ys.min()), y0 + int(ys.max()) + 1
        label, conf = word_label(word_net, word_classes, arr, bx0, bx1, y0, y1, dev) if word_net is not None else (None, 0)
        if label not in ("dev", "tam"):
            # fall back to patch labels for this component
            rr = classify_word(None, [], arr, bx0, bx1, y0, y1, dev, dark) if False else []
            label = None
        comps.append({"k": k, "x0": bx0, "x1": bx1, "y0": by0, "y1": by1,
                      "w": bx1 - bx0, "label": label, "conf": conf})
    large = [c for c in comps if c["label"] in ("dev", "tam")]
    if not large: return []
    for c in comps:
        if c["label"] in ("dev", "tam"): continue
        # nearest large component by horizontal distance/overlap
        best = min(large, key=lambda q: 0 if not (q["x1"] < c["x0"] or q["x0"] > c["x1"])
                   else min(abs(q["x0"] - c["x1"]), abs(c["x0"] - q["x1"])))
        c["label"] = best["label"]
    comps.sort(key=lambda c: c["x0"])
    groups = []
    for c in comps:
        if groups and groups[-1][-1]["label"] == c["label"]:
            groups[-1].append(c)
        else:
            groups.append([c])
    out = []
    for g in groups:
        lb = g[0]["label"]
        if lb not in ("dev", "tam"): continue
        gx0 = min(c["x0"] for c in g); gx1 = max(c["x1"] for c in g)
        gy0 = min(c["y0"] for c in g); gy1 = max(c["y1"] for c in g)
        if gx1 - gx0 < MIN_RUN: continue
        out.append((lb, gx0, gx1, gy0, gy1))
    return out


def classify_word_columns(col_net, col_classes, arr, x0, x1, y0, y1, dev, dark=None):
    if dark is None:
        dark = arr < 128
    sub = arr[y0:y1, x0:x1]
    d = sub < 128
    rows = np.where(d.any(axis=1))[0]
    cols = np.where(d.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return []
    # Keep the full word box: training targets are fractions of this box.
    ix0, ix1 = x0, x1
    iy0, iy1 = y0, y1
    im = Image.fromarray(arr[iy0:iy1, ix0:ix1]).resize((COL_W, COL_H), Image.BILINEAR)
    xb = torch.from_numpy(1.0 - np.asarray(im, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(dev)
    with torch.no_grad():
        p = col_net(xb).argmax(1)[0].cpu().numpy()
    # median smooth to remove one-column flicker
    if len(p) >= 5:
        q = p.copy()
        for i in range(2, len(p) - 2):
            q[i] = np.bincount(p[i-2:i+3]).argmax()
        p = q
    raw = []
    a = 0
    for k in range(1, len(p) + 1):
        if k == len(p) or p[k] != p[a]:
            if k - a >= 2:
                raw.append((int(p[a]), a, k))
            a = k
    if not raw:
        return []
    bounds = [ix0]
    w = ix1 - ix0
    for k in range(1, len(raw)):
        c = int(round((raw[k-1][2] + raw[k][1]) / 2))
        x = ix0 + int(round(c * w / COL_W))
        x = _snap_split(dark, iy0, iy1, x, ix0, ix1, max(2, int(0.04 * w)))
        bounds.append(x)
    bounds.append(ix1)
    out = []
    for k, (lab, _, _) in enumerate(raw):
        lo, hi = bounds[k], bounds[k + 1]
        if hi - lo < MIN_RUN:
            continue
        script = col_classes[lab]
        if script not in ("dev", "tam"):
            continue
        rs = dark[iy0:iy1, lo:hi]
        rws = np.where(rs.any(axis=1))[0]
        ccs = np.where(rs.any(axis=0))[0]
        if len(rws) == 0 or len(ccs) == 0:
            continue
        out.append((script, lo + int(ccs[0]), lo + int(ccs[-1]) + 1,
                    iy0 + int(rws[0]), iy0 + int(rws[-1]) + 1))
    return out


def classify_word_hybrid(patch_net, patch_classes, word_net, word_classes, arr,
                         x0, x1, y0, y1, dev, dark=None):
    runs = classify_word(patch_net, patch_classes, arr, x0, x1, y0, y1, dev, dark)
    if not runs or word_net is None:
        return runs
    # Short tokens (page numbers, digits) are better left to the patch model.
    if (x1 - x0) < 2.5 * max(1, y1 - y0):
        return runs
    spans = {}
    for r in runs:
        spans[r[0]] = spans.get(r[0], 0) + (r[2] - r[1])
    total = max(1, sum(spans.values()))
    # A word with substantial ink in both scripts is genuinely mixed; keep the
    # safe patch partition.  A small edge run is usually a classifier error
    # (e.g. a Devanagari repha labelled Tamil), so let the word model override.
    mixed = (spans.get("dev", 0) >= 0.2 * total and
             spans.get("tam", 0) >= 0.2 * total)
    if not mixed:
        label, conf = word_label(word_net, word_classes, arr, x0, x1, y0, y1, dev)
        if label in ("dev", "tam") and conf >= 0.8:
            return _whole_word_run(arr, x0, x1, y0, y1, [[label, x0, x1]])
    # For genuinely mixed words, still verify each substantial script run with
    # the word model; patch edges are the usual source of misrouted marks.
    out = []
    for r in runs:
        if r[2] - r[1] >= 2.5 * max(1, y1 - y0):
            label, conf = word_label(word_net, word_classes, arr, r[1], r[2], y0, y1, dev)
            if label in ("dev", "tam") and conf >= 0.8:
                r = (label, r[1], r[2], r[3], r[4])
        out.append(r)
    return out


def _safe_split_x(dark, y0, y1, target, lo, hi, max_d=28):
    """Nearest column with no ink in the word's vertical band."""
    target = int(target)
    for d in range(max_d + 1):
        for c in (target - d, target + d):
            if lo <= c <= hi and not dark[y0:y1, c].any():
                return c
    return None


def _merge_short_runs(merged, y0, y1):
    """Fold tiny edge/mark runs into the larger neighbouring script run.

    Devanagari repha/anusvara marks are separate ink components and are easily
    labelled as the other script.  They are much narrower than a word, so
    absorbing them into the adjacent larger run keeps the glyph complete.
    """
    min_w = max(8, int(0.45 * max(1, y1 - y0)))
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for i, r in enumerate(merged):
            if r[2] - r[1] >= min_w:
                continue
            left = merged[i - 1] if i > 0 else None
            right = merged[i + 1] if i + 1 < len(merged) else None
            if left is None and right is None:
                break
            if right is None or (left is not None and (left[2] - left[1]) >= (right[2] - right[1])):
                left[2] = r[2]
            else:
                right[1] = r[1]
            del merged[i]
            changed = True
            break
    return merged


def _whole_word_run(arr, x0, x1, y0, y1, merged):
    spans = {}
    for lb, rx0, rx1 in merged:
        spans[lb] = spans.get(lb, 0) + (rx1 - rx0)
    lb = max(spans, key=spans.get)
    rs = (arr[y0:y1, x0:x1] < 128)
    rows = np.where(rs.any(axis=1))[0]
    cols = np.where(rs.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return []
    return [(lb, x0 + int(cols[0]), x0 + int(cols[-1]) + 1,
             y0 + int(rows[0]), y0 + int(rows[-1]) + 1)]


def classify_word(net, classes, arr, x0, x1, y0, y1, dev, dark=None):
    if dark is None:
        dark = arr < 128
    sub = dark[y0:y1, x0:x1]
    rows = np.where(sub.any(axis=1))[0]
    if len(rows) == 0:
        return None
    cy = y0 + (int(rows[0]) + int(rows[-1])) // 2
    xs = list(range(x0 + 20, max(x0 + 21, x1 - 20), STRIDE))
    if not xs:
        return None
    patches = []
    for x in xs:
        px0 = x - 24
        py0 = cy - 24
        patch = np.full((48, 48), 255, dtype=np.uint8)
        sx0, sy0 = max(0, px0), max(0, py0)
        sx1 = min(arr.shape[1], px0 + 48)
        sy1 = min(arr.shape[0], py0 + 48)
        if sx1 <= sx0 or sy1 <= sy0:
            patches.append(patch)
            continue
        patch[sy0 - py0:sy1 - py0, sx0 - px0:sx1 - px0] = arr[sy0:sy1, sx0:sx1]
        patches.append(patch)
    xb = torch.from_numpy(np.stack(patches)).unsqueeze(1).float().to(dev)
    xb = (1.0 - xb / 255.0)
    with torch.no_grad():
        probs = net(xb).softmax(1).cpu().numpy()
    labels = []
    for p in probs:
        di, ti, oi = classes.index("dev"), classes.index("tam"), classes.index("other")
        m = max((p[di], "dev"), (p[ti], "tam"), (p[oi], "other"))
        labels.append(m[1] if m[0] > 0.45 else "null")
    runs = []
    a = 0
    for k in range(1, len(labels) + 1):
        if k == len(labels) or labels[k] != labels[a]:
            runs.append([labels[a], xs[a], xs[k - 1] + 1])
            a = k
    merged = []
    for lb, rx0, rx1 in runs:
        if lb == "null":
            continue
        if merged and merged[-1][0] == lb:
            merged[-1][2] = rx1
        else:
            merged.append([lb, rx0, rx1])
    if not merged:
        return []
    merged = _merge_short_runs(merged, y0, y1)
    # First/last runs always use the full word bounds; internal splits must
    # fall on an ink-free column so no glyph is cut in half.
    bounds = [x0]
    for k in range(1, len(merged)):
        target = (merged[k - 1][2] + merged[k][1]) // 2
        sx = _safe_split_x(dark, y0, y1, target, x0, x1)
        if sx is None:
            return _whole_word_run(arr, x0, x1, y0, y1, merged)
        bounds.append(sx)
    bounds.append(x1)
    out = []
    for k, (lb, _, _) in enumerate(merged):
        lo, hi = bounds[k], bounds[k + 1]
        if hi - lo < MIN_RUN:
            continue
        rs = (arr[y0:y1, lo:hi] < 128)
        rws = np.where(rs.any(axis=1))[0]
        cols = np.where(rs.any(axis=0))[0]
        if len(rws) == 0 or len(cols) == 0:
            continue
        out.append((lb, lo + int(cols[0]), lo + int(cols[-1]) + 1,
                    y0 + int(rws[0]), y0 + int(rws[-1]) + 1))
    return out

def make_segments(runs):
    segs = []
    by_line = {}
    for i, r in enumerate(runs):
        by_line.setdefault(r["line"], []).append(i)
    for li in sorted(by_line):
        cur = []
        for i in by_line[li]:
            if cur and runs[i]["script"] != runs[cur[-1]]["script"]:
                segs.append({"line": li, "indices": cur, "script": runs[cur[0]]["script"]})
                cur = []
            cur.append(i)
        if cur:
            segs.append({"line": li, "indices": cur, "script": runs[cur[0]]["script"]})
    for k, s in enumerate(segs):
        rs = [runs[i] for i in s["indices"]]
        s["idx"] = k
        s["box"] = (min(r["x0"] for r in rs), min(r["y0"] for r in rs),
                    max(r["x1"] for r in rs), max(r["y1"] for r in rs))
    return segs


def compose_faux(arr, runs, path, tag_start=1):
    """Stack script-pure line strips on one page, one strip per row.

    runs: (idx, x0, x1, y0, y1).  Each strip is a whole line segment (not a
    single word), so pages are dense and OCR sees line context.
    """
    crops = []
    for r in runs:
        _, x0, x1, y0, y1 = r
        py = max(6, int(0.25 * max(1, y1 - y0)))
        c = arr[max(0, y0 - py):y1 + py, max(0, x0 - 6):x1 + 6]
        h = c.shape[0]
        s = 90.0 / max(1, h)
        s = max(s, 0.6)
        im = Image.fromarray(c).resize((max(1, int(round(c.shape[1] * s))),
                                       max(1, int(round(h * s)))), Image.LANCZOS)
        crops.append(im)
    W = min(1780, max(700, max(im.width for im in crops) + 210))
    H = 40 + len(crops) * 120
    page = Image.new("L", (W, H), 255)
    dr = ImageDraw.Draw(page)
    font = ImageFont.truetype(TAG_FONT, 44)
    for k, im in enumerate(crops):
        y = 20 + k * 120
        dr.text((20, y + 12), f"#T{tag_start + k:02d}", font=font, fill=0)
        if im.width > W - 190:
            s = (W - 190) / im.width
            im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
        page.paste(im, (180, y))
    page.convert("RGB").save(path, quality=92)
    return len(crops)

def parse_tagged(txt, n):
    """Map Drive OCR output lines to tag indices 1..n.

    Returns list of length n (text or None). Text after a tag on the same line,
    and all untagged continuation lines, belong to that tag; text before the
    first tag of a line also continues the previous tag."""
    texts = [None] * n
    cur = None
    for line in txt.splitlines():
        s = line.strip()
        if not s or s == "---" or s.startswith("![") or s.startswith("[image") or s.startswith("<data:"):
            continue
        s = s.replace("**", "")
        s = MD_ESC_RE.sub(r"\1", s)
        pieces = []
        pos = 0
        for m in TAG_RE.finditer(s):
            pieces.append((cur, s[pos:m.start()]))
            cur = int(m.group(1)) - 1
            pos = m.end()
        pieces.append((cur, s[pos:]))
        for idx, t in pieces:
            t = t.strip()
            if idx is None or not t or not (0 <= idx < n):
                continue
            texts[idx] = (texts[idx] + " " + t) if texts[idx] else t
    return texts


def insert_section_rules(lines, rule_ys):
    out = []
    ri = 0
    for y, text in lines:
        while ri < len(rule_ys) and rule_ys[ri] < y - 2:
            if out and out[-1] != "---":
                out.extend(["", "---", ""])
            ri += 1
        if text.strip():
            out.append(text.strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--pages", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--classifier", default="training/local/script_cls.pt")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--word-classifier", default="")
    ap.add_argument("--line-classifier", default="")
    ap.add_argument("--column-classifier", default="")
    ap.add_argument("--boundary-classifier", default="")
    ap.add_argument("--chunk", type=int, default=40)
    ap.add_argument("--page-batch", type=int, default=10,
                    help="pages to compose before one Drive API batch")
    ap.add_argument("--faux-batch", type=int, default=10,
                    help="faux pages sent to Drive per helper invocation")
    ap.add_argument("--md-dir", default="")
    ap.add_argument("--page-offset", type=int, default=0)
    ap.add_argument("--linewise", action="store_true",
                    help="compose one faux page per printed line (linewiseScriptSepDriveOCR)")
    ap.add_argument("--overlay", action="store_true")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    net, classes = load_classifier(REPO / args.classifier, dev)
    word_net, word_classes = (load_word_classifier(args.word_classifier, dev)
                              if args.word_classifier else (None, None))
    line_net, line_classes = (load_line_classifier(args.line_classifier, dev)
                              if args.line_classifier else (None, None))
    col_net, col_classes = (load_column_classifier(args.column_classifier, dev)
                            if args.column_classifier else (None, None))
    bnd_net = load_boundary_classifier(args.boundary_classifier, dev) if args.boundary_classifier else None
    import pymupdf
    doc = pymupdf.open(args.pdf)
    lo, hi = (int(x) if x else None for x in args.pages.split(":"))
    indices = range(lo or 0, hi if hi is not None else len(doc))
    out_dir = REPO / args.out if not pathlib.Path(args.out).is_absolute() else pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    workdir = pathlib.Path(args.workdir)
    faux_dir = workdir / "faux"
    faux_dir.mkdir(parents=True, exist_ok=True)
    if args.overlay:
        (workdir / "overlay").mkdir(parents=True, exist_ok=True)
    runs_dump = open(out_dir / "runs.jsonl", "w", encoding="utf-8")
    stem = pathlib.Path(args.pdf).stem

    def ocr_drive(files):
        for i in range(0, len(files), args.faux_batch):
            chunk = files[i:i + args.faux_batch]
            print(f"  Drive batch {i//args.faux_batch + 1}: {len(chunk)} faux pages", flush=True)
            r = subprocess.run([VENV_PY, str(DRIVE_HELPER), "--key", GOOGLE_KEY] + [str(p) for p in chunk],
                               capture_output=True, text=True)
            if r.returncode:
                print("DRIVE HELPER FAILED", r.returncode, flush=True)
                print(r.stdout[-4000:], flush=True)
                print(r.stderr[-8000:], flush=True)
                raise SystemExit(r.returncode)

    pending = []
    page_list = list(indices)

    def flush_batch(batch):
        ocr_drive([p for (_, _, _, fs) in batch for p, _ in fs])
        parsed = []
        retry_jobs = []
        for pi, arr, page_runs, faux_files in batch:
            chunks = []
            for path, chunk in faux_files:
                txt_path = pathlib.Path(str(path) + ".txt")
                texts = (parse_tagged(txt_path.read_text(encoding="utf-8"), len(chunk))
                         if txt_path.exists() else [None] * len(chunk))
                chunks.append((path, chunk, texts))
                missing = [k for k, t in enumerate(texts) if t is None]
                if missing:
                    print(f"  WARN {path.name}: {len(missing)}/{len(chunk)} tags missing -> retry page", flush=True)
                    retry_path = faux_dir / f"{path.stem}_retry.jpg"
                    compose_faux(arr, [(chunk[m]["idx"], chunk[m]["box"][0], chunk[m]["box"][2],
                                        chunk[m]["box"][1], chunk[m]["box"][3]) for m in missing],
                                   retry_path, tag_start=1)
                    retry_jobs.append((retry_path, missing, texts))
            parsed.append((pi, page_runs, chunks))
        ocr_drive([rp for rp, _, _ in retry_jobs])
        for retry_path, missing, texts in retry_jobs:
            rt = pathlib.Path(str(retry_path) + ".txt")
            rtexts = parse_tagged(rt.read_text(encoding="utf-8"), len(missing)) if rt.exists() else []
            for m, t in zip(missing, rtexts):
                if t is not None:
                    texts[m] = t
        for pi, page_runs, chunks in parsed:
            for _, chunk, texts in chunks:
                for s, t in zip(chunk, texts):
                    page_runs[s["indices"][0]]["text"] = t
            by_line = {}
            for r in page_runs:
                by_line.setdefault(r["line"], []).append(r)
            with open(out_dir / f"{stem}_p{pi:04d}.txt", "w", encoding="utf-8") as f:
                for li in sorted(by_line):
                    parts = []
                    first = True
                    for r in by_line[li]:
                        t = r["text"] or ""
                        if not t:
                            continue
                        if not first and r["pre_space"]:
                            parts.append(" ")
                        parts.append(t)
                        first = False
                    f.write("".join(parts) + "\n")
            if args.md_dir:
                try:
                    from detect_blocks import split_page_sections
                except ImportError:
                    split_page_sections = None
                rule_ys = []
                if split_page_sections is not None:
                    try:
                        meta = split_page_sections(arr)
                        rule_ys = [L["yc"] for L in meta["rules"]
                                   if L["yc"] > meta["header_bottom"] + 2]
                    except Exception:
                        rule_ys = []
                hb = int(meta.get("header_bottom", 0)) if split_page_sections is not None else 0
                entries = []
                for li in sorted(by_line):
                    texts = [r["text"] for r in by_line[li] if r["text"]]
                    if not texts:
                        continue
                    y = min(r["y0"] for r in by_line[li] if r["text"])
                    if y < hb:
                        continue
                    entries.append((y, " ".join(texts)))
                body = insert_section_rules(entries, rule_ys)
                printed = pi + 1 - args.page_offset
                if printed <= 0:
                    printed = pi + 1
                md_dir = pathlib.Path(args.md_dir)
                md_dir.mkdir(parents=True, exist_ok=True)
                (md_dir / f"p{pi:05d}.md").write_text(
                    f"[[P{printed}]]\n" + "\n".join(body) + "\n", encoding="utf-8")
            for r in page_runs:
                runs_dump.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"page {pi}: done", flush=True)

    for q, pi in enumerate(page_list):
        pix = doc[pi].get_pixmap(dpi=args.dpi, colorspace=pymupdf.csGRAY)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        dark = arr < 128
        lines = segment_lines(Image.fromarray(arr), 6)
        page_runs = []
        for li, (lx0, ly0, lx1, ly1) in enumerate(lines):
            words = word_segments(dark, lx0, lx1, ly0, ly1)
            for wi, (wx0, wx1) in enumerate(words):
                if bnd_net is not None:
                    rr = split_mixed_word(bnd_net, word_net, word_classes, arr, wx0, wx1, ly0, ly1, dev)
                elif col_net is not None:
                    rr = classify_word_columns(col_net, col_classes, arr, wx0, wx1, ly0, ly1, dev, dark)
                else:
                    rr = classify_word_hybrid(net, classes, word_net, word_classes, arr,
                                              wx0, wx1, ly0, ly1, dev, dark)
                if not rr:
                    continue
                pre_space = wi == 0 or (wx0 - words[wi - 1][1]) >= GAP_WORD
                for k, (script, rx0, rx1, ry0, ry1) in enumerate(rr):
                    page_runs.append({
                        "page": pi, "line": li, "word": wi, "seq": len(page_runs),
                        "script": script, "x0": rx0, "y0": ry0, "x1": rx1, "y1": ry1,
                        "pre_space": pre_space if k == 0 else False, "text": None})
        if line_net is not None:
            by_line = {}
            for r in page_runs:
                by_line.setdefault(r["line"], []).append(r)
            for li, rs in by_line.items():
                lx0 = min(r["x0"] for r in rs); lx1 = max(r["x1"] for r in rs)
                ly0 = min(r["y0"] for r in rs); ly1 = max(r["y1"] for r in rs)
                ll, lc = line_label(line_net, line_classes, arr, lx0, ly0, lx1, ly1, dev)
                if ll not in ("dev", "tam") or lc < 0.80:
                    continue
                h = max(1, ly1 - ly0)
                for r in rs:
                    if (r["script"] in ("dev", "tam") and r["script"] != ll
                            and (r["x1"] - r["x0"]) >= 0.3 * h):
                        # keep high-confidence inline words of the other script
                        wl, wc = word_label(word_net, word_classes, arr, r["x0"], r["x1"],
                                            r["y0"], r["y1"], dev) if word_net is not None else (None, 0)
                        if not (wl == r["script"] and wc >= 0.95):
                            r["script"] = ll
        segs = make_segments(page_runs)
        print(f"page {pi}: {len(lines)} lines, {len(page_runs)} runs, {len(segs)} segments", flush=True)

        if args.overlay:
            ov = Image.fromarray(arr).convert("RGB")
            dr = ImageDraw.Draw(ov)
            for r in page_runs:
                dr.rectangle([r["x0"] - 3, r["y0"] - 3, r["x1"] + 3, r["y1"] + 3],
                             outline=SCRIPT_COLORS[r["script"]], width=3)
            ov.save(workdir / "overlay" / f"p{pi:04d}.jpg", quality=90)

        faux_files = []
        for script in ("dev", "tam"):
            subset = [s for s in segs if s["script"] == script]
            if not subset:
                continue
            for ci in range(0, len(subset), args.chunk):
                chunk = subset[ci:ci + args.chunk]
                tag = "line" if args.linewise else "sep"
                path = faux_dir / f"p{pi:04d}_{tag}_{script}_{ci:03d}.jpg"
                n = compose_faux(arr, [(s["idx"], s["box"][0], s["box"][2],
                                        s["box"][1], s["box"][3]) for s in chunk], path)
                assert n == len(chunk)
                faux_files.append((path, chunk))
        pending.append((pi, arr, page_runs, faux_files))
        if len(pending) >= args.page_batch or q == len(page_list) - 1:
            flush_batch(pending)
            pending = []
    runs_dump.close()


if __name__ == "__main__":
    main()
