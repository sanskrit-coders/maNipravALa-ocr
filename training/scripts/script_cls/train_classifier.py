#!/usr/bin/env python3
"""Train a patch-level script classifier (devanagari / tamil / other / null).

Synthetic pages are rendered from the repo corpora with per-script fonts
(PIL + Raqm shaping).  Patches are labelled by the centre of the *script run*
they fall in, not by per-character advances (which are unreliable for Indic
shaping).  Includes scan-like augmentation, balanced classes, a larger
residual CNN, word-disjoint validation and Kaggle path auto-detection.

Usage local:
  python3 training/scripts/script_cls/train_classifier.py --steps 20000

Kaggle script kernel: place this file as train_classifier.py, GPU on; corpus
and fonts are found under /kaggle/input/*/{texts,fonts}.
"""
import argparse
import io
import json
import os
import pathlib
import random
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter, ImageFont

BASE = pathlib.Path(__file__).resolve().parents[2]
REPO = BASE.parent if BASE.name == "training" else BASE

PATCH = 48
CANVAS_W = 1600
MARGIN = 60

DEV_FONTS = [
    "/usr/share/fonts/sanskrit-fonts/ttf-devanagari/noto/NotoSansDevanagari-Regular.ttf",
    "/usr/share/fonts/sanskrit-fonts/ttf-devanagari/noto/NotoSansDevanagari-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerifDevanagari-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerifDevanagari-Bold.ttf",
    "/usr/share/fonts/sanskrit-fonts/ttf-devanagari/siddhanta.ttf",
    "/usr/share/fonts/sanskrit-fonts/ttf-devanagari/siddhanta-variations/siddhanta-calcutta.ttf",
    "/usr/share/fonts/sanskrit-fonts/ttf-devanagari/siddhanta-variations/siddhanta-nepali.ttf",
    "/usr/share/fonts/sanskrit-fonts/ttf-devanagari/Sanskrit2003.ttf",
    "/usr/local/share/fonts/t/TiroDevanagariSanskrit_Regular.ttf",
    "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
    "/usr/share/fonts/sanskrit-fonts/ttf-devanagari/Murty-Sanskrit-1.01.otf",
]
TAM_FONTS = [
    "/usr/share/fonts/sanskrit-fonts/ttf-tamil/NotoSansTamil-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansTamil-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerifTamil-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerifTamil-Bold.ttf",
    "/usr/share/fonts/sanskrit-fonts/ttf-tamil/TAMu_Kalyani.ttf",
    "/usr/share/fonts/sanskrit-fonts/ttf-tamil/TAMu_Maduram.ttf",
    "/usr/share/fonts/sanskrit-fonts/ttf-tamil/TSCu_Paranar.ttf",
    "/usr/share/fonts/sanskrit-fonts/ttf-tamil/TSCu_Times.ttf",
    "/usr/share/fonts/truetype/lohit-tamil-classical/Lohit-Tamil-Classical.ttf",
    "/usr/share/fonts/truetype/lohit-tamil/Lohit-Tamil.ttf",
]
OTHER_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
]

CLASSES = ["dev", "tam", "other", "null"]
CLS_ID = {c: i for i, c in enumerate(CLASSES)}
LATIN = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

FONTS_CACHE = {}


def get_font(path, size):
    key = (path, size)
    if key not in FONTS_CACHE:
        FONTS_CACHE[key] = ImageFont.truetype(path, size)
    return FONTS_CACHE[key]


def ch_script(ch):
    o = ord(ch)
    if 0x0900 <= o <= 0x097F:
        return "dev"
    if 0x0B80 <= o <= 0x0BFF:
        return "tam"
    if ch.isspace():
        return None
    return "other"


def load_words(texts_dir):
    words = []
    for sub in ("core", "commentary"):
        for f in sorted((pathlib.Path(texts_dir) / sub).glob("*.txt")):
            for w in f.read_text(encoding="utf-8").split():
                prof = [p for p in (ch_script(c) for c in w) if p]
                if 2 <= len(prof) <= 14 and any(p != "other" for p in prof):
                    words.append((w, prof))
    return words


def split_runs(chars, prof):
    runs = []
    for c, p in zip(chars, prof):
        if runs and runs[-1][1] == p:
            runs[-1][0] += c
        else:
            runs.append([c, p])
    return runs


def render_word(args):
    word, prof, seed = args
    rng = random.Random(seed)
    size = rng.randint(40, 64)
    fonts = {"dev": rng.choice(DEV_FONTS), "tam": rng.choice(TAM_FONTS),
             "other": rng.choice(OTHER_FONTS)}
    runs = split_runs(list(word), prof)
    if len(runs) > 5:
        return None
    img = Image.new("L", (CANVAS_W, 180), 255)
    d = ImageDraw.Draw(img)
    baseline = 120
    x = MARGIN
    out = []
    for text, script in runs:
        f = get_font(fonts[script], size)
        w = d.textlength(text, font=f)
        out.append({"x0": x, "x1": x + w, "script": script, "text": text, "size": size})
        d.text((x, baseline), text, font=f, fill=0, anchor="ls")
        x += w
    arr = np.asarray(img)
    cols = np.where((arr < 200).any(axis=0))[0]
    if len(cols) == 0:
        return None
    x0 = max(0, int(cols[0]) - MARGIN)
    x1 = min(CANVAS_W, int(cols[-1]) + MARGIN + 1)
    crop = np.ascontiguousarray(arr[:, x0:x1])
    for r in out:
        r["x0"] -= x0
        r["x1"] -= x0
    gaps = []
    prev = 0
    for r in out:
        if r["x0"] - prev >= 14:
            gaps.append((int(prev), int(r["x0"])))
        prev = max(prev, int(r["x1"]))
    if crop.shape[1] - prev >= 14:
        gaps.append((int(prev), int(crop.shape[1])))
    return {"img": crop, "runs": out, "gaps": gaps, "baseline": baseline, "size": size}


def render_other(seed):
    rng = random.Random(seed)
    if rng.random() < 0.65:
        n = rng.randint(2, 10)
        s = "".join(rng.choice(LATIN) for _ in range(n))
    else:
        s = rng.choice(["1924", "Page 12", "Vol. I", "cf. 1892", "No. 31", "xii",
                        "Introduction", "BORI", "CE", "Ed."])
    size = rng.randint(40, 64)
    f = get_font(rng.choice(OTHER_FONTS), size)
    img = Image.new("L", (1000, 180), 255)
    d = ImageDraw.Draw(img)
    baseline = 120
    x = MARGIN
    w = d.textlength(s, font=f)
    d.text((x, baseline), s, font=f, fill=0, anchor="ls")
    arr = np.asarray(img)
    cols = np.where((arr < 200).any(axis=0))[0]
    if len(cols) == 0:
        return None
    x0 = max(0, int(cols[0]) - MARGIN)
    x1 = min(1000, int(cols[-1]) + MARGIN + 1)
    crop = np.ascontiguousarray(arr[:, x0:x1])
    return {"img": crop, "runs": [{"x0": MARGIN - x0, "x1": MARGIN - x0 + w,
                                   "script": "other", "text": s, "size": size}],
            "gaps": [], "baseline": baseline, "size": size}


def sample_patch(arr, cx, cy):
    H, W = arr.shape
    x0 = int(round(cx - PATCH / 2))
    y0 = int(round(cy - PATCH / 2))
    out = np.full((PATCH, PATCH), 255, dtype=np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x0 + PATCH), min(H, y0 + PATCH)
    if sx1 <= sx0 or sy1 <= sy0:
        return None
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = arr[sy0:sy1, sx0:sx1]
    return out


def augment(p, rng):
    im = Image.fromarray(p)
    if rng.random() < 0.5:
        im = im.rotate(rng.uniform(-4, 4), resample=Image.BILINEAR, fillcolor=255)
    if rng.random() < 0.25:
        a = rng.uniform(-0.08, 0.08)
        im = im.transform(im.size, Image.AFFINE, (1, a, 0, 0, 1, 0),
                          resample=Image.BILINEAR, fillcolor=255)
    if rng.random() < 0.45:
        s = rng.uniform(0.6, 1.0)
        im = im.resize((max(1, int(PATCH * s)), max(1, int(PATCH * s))), Image.BILINEAR)
        im = im.resize((PATCH, PATCH), Image.BILINEAR)
    if rng.random() < 0.2:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=rng.randint(50, 95))
        buf.seek(0)
        im = Image.open(buf).convert("L")
    if rng.random() < 0.5:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.8)))
    if rng.random() < 0.15:
        im = im.filter(ImageFilter.MaxFilter(3))
    if rng.random() < 0.15:
        im = im.filter(ImageFilter.MinFilter(3))
    a = np.asarray(im).astype(np.float32) / 255.0
    if rng.random() < 0.3:
        yy, xx = np.mgrid[0:PATCH, 0:PATCH]
        a = a * (1.0 - 0.10 * np.sin(xx / rng.uniform(10, 40) + rng.uniform(0, 6))
                 * np.cos(yy / rng.uniform(10, 40)))
    if rng.random() < 0.7:
        a = np.clip(a * rng.uniform(0.55, 1.45) + rng.uniform(-0.1, 0.1), 0, 1)
    if rng.random() < 0.5:
        a = np.power(a, rng.uniform(0.65, 1.6))
    if rng.random() < 0.5:
        a = a + np.random.randn(*a.shape).astype(np.float32) * rng.uniform(0.005, 0.06)
    return np.clip(1.0 - a, 0, 1).astype(np.float32)


def build_index(mixed, others):
    by = {c: [] for c in ("dev", "tam", "other")}
    for ri, rec in enumerate(mixed):
        for run in rec["runs"]:
            w = run["x1"] - run["x0"]
            if run["script"] in ("dev", "tam") and w > 14:
                by[run["script"]].append((ri, run))
            elif run["script"] == "other" and w >= 24:
                by["other"].append((ri, run))
    for ri, rec in enumerate(others):
        for run in rec["runs"]:
            by["other"].append((len(mixed) + ri, run))
    return by


def make_sample(mixed, others, by, rng, augment_patch=True):
    u = rng.random()
    if u < 0.38:
        cls = "dev"
    elif u < 0.76:
        cls = "tam"
    elif u < 0.90:
        cls = "other"
    else:
        cls = "null"
    if cls == "null":
        rec = mixed[rng.randrange(len(mixed))]
        if rec["gaps"] and rng.random() < 0.75:
            a, b = rng.choice(rec["gaps"])
            cx = rng.uniform(a + 4, max(a + 5, b - 4))
            cy = rec["baseline"] - rng.uniform(-8, 48)
        elif rng.random() < 0.5:
            cx = rng.uniform(0, rec["img"].shape[1])
            cy = rng.uniform(0, max(1, rec["baseline"] - 58))
        else:
            cx = rng.uniform(0, rec["img"].shape[1])
            cy = rng.uniform(min(rec["img"].shape[0] - 1, rec["baseline"] + 18),
                             rec["img"].shape[0] - 1)
        lab = CLS_ID["null"]
    else:
        cands = by[cls]
        if not cands:
            return None
        ri, run = cands[rng.randrange(len(cands))]
        rec = mixed[ri] if ri < len(mixed) else others[ri - len(mixed)]
        w = run["x1"] - run["x0"]
        margin = min(10.0, w * 0.2)
        if cls in ("dev", "tam") and w > 30 and rng.random() < 0.15:
            # hard boundary patch: centre near a script transition
            if rng.random() < 0.5:
                cx = run["x0"] + rng.uniform(2, min(12, w / 2))
            else:
                cx = run["x1"] - rng.uniform(2, min(12, w / 2))
        else:
            cx = rng.uniform(run["x0"] + margin, max(run["x0"] + margin + 1, run["x1"] - margin))
        cy = rec["baseline"] - rng.uniform(2, min(40, rec["size"] * 0.8))
        lab = CLS_ID[cls]
    p = sample_patch(rec["img"], cx, cy)
    if p is None:
        return None
    return (augment(p, rng) if augment_patch else
            np.clip(1.0 - p.astype(np.float32) / 255.0, 0, 1)), lab


class SynthDS(torch.utils.data.Dataset):
    def __init__(self, mixed, others, by):
        self.mixed, self.others, self.by = mixed, others, by

    def __len__(self):
        return 1 << 40

    def __getitem__(self, i):
        rng = random.Random((i * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFF)
        for _ in range(8):
            s = make_sample(self.mixed, self.others, self.by, rng, True)
            if s is not None:
                return torch.from_numpy(s[0]).unsqueeze(0), s[1]
        return torch.zeros(1, PATCH, PATCH), CLS_ID["null"]


def collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                      nn.BatchNorm2d(cout))
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        y = F.silu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.silu(y + self.skip(x))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 32, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(32), nn.SiLU())
        self.l1 = ResBlock(32, 64, 2)
        self.l2 = ResBlock(64, 96, 2)
        self.l3 = ResBlock(96, 128, 2)
        self.l4 = ResBlock(128, 128, 1)
        self.f = nn.Sequential(nn.Linear(384, 192), nn.SiLU(), nn.Dropout(0.15),
                               nn.Linear(192, len(CLASSES)))

    def forward(self, x):
        h = self.l4(self.l3(self.l2(self.l1(self.stem(x)))))
        avg = h.mean((2, 3))
        mx = h.amax((2, 3))
        ctr = h[:, :, 1:5, 1:5].mean((2, 3))
        return self.f(torch.cat([avg, mx, ctr], 1))


@torch.no_grad()
def predict(net, x, dev, batch=256):
    net.eval()
    out = []
    for i in range(0, len(x), batch):
        out.append(net(x[i:i + batch].to(dev)).argmax(1).cpu())
    net.train()
    return torch.cat(out)


def gen_val(mixed, others, by, n=8000, seed=999):
    rng = random.Random(seed)
    xs, ys = [], []
    while len(xs) < n:
        s = make_sample(mixed, others, by, rng, False)
        if s is not None:
            xs.append(s[0])
            ys.append(s[1])
    return torch.from_numpy(np.stack(xs)).unsqueeze(1), torch.tensor(ys)


def build_pool(words, n, seed):
    rng = random.Random(seed)
    args = [(words[i][0], words[i][1], rng.randrange(1 << 30))
            for i in (rng.randrange(len(words)) for _ in range(n))]
    with Pool(min(12, os.cpu_count() or 4)) as pool:
        recs = [r for r in pool.map(render_word, args, chunksize=20) if r is not None]
    print(f"rendered {len(recs)} word images", flush=True)
    return recs


def build_other(n, seed):
    rng = random.Random(seed)
    recs = [render_other(rng.randrange(1 << 30)) for _ in range(n)]
    recs = [r for r in recs if r is not None]
    print(f"rendered {len(recs)} other images", flush=True)
    return recs


def localize_fonts(fonts, fonts_dir):
    if not fonts_dir:
        return [p for p in fonts if pathlib.Path(p).exists()]
    d = pathlib.Path(fonts_dir)
    out = [str(d / pathlib.Path(p).name) for p in fonts if (d / pathlib.Path(p).name).exists()]
    if not out:
        out = [str(p) for p in sorted(d.glob("*.ttf")) + sorted(d.glob("*.otf"))]
    return out


def default_input_dir(name, sub):
    if pathlib.Path("/kaggle/input").exists():
        hits = sorted(p for p in pathlib.Path("/kaggle/input").rglob(sub) if p.is_dir())
        if hits:
            return hits[0]
    return REPO / name / sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pool-size", type=int, default=12000)
    ap.add_argument("--val-pool-size", type=int, default=3000)
    ap.add_argument("--other-pool-size", type=int, default=3000)
    ap.add_argument("--val-other-size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--texts", default="")
    ap.add_argument("--fonts-dir", default="")
    ap.add_argument("--eval-every", type=int, default=500)
    args = ap.parse_args()

    if not args.fonts_dir and pathlib.Path("/kaggle/input").exists():
        hits = sorted(p for p in pathlib.Path("/kaggle/input").rglob("fonts") if p.is_dir())
        if hits:
            args.fonts_dir = str(hits[0])
    global DEV_FONTS, TAM_FONTS, OTHER_FONTS
    DEV_FONTS = localize_fonts(DEV_FONTS, args.fonts_dir)
    TAM_FONTS = localize_fonts(TAM_FONTS, args.fonts_dir)
    OTHER_FONTS = localize_fonts(OTHER_FONTS, args.fonts_dir)
    print(f"fonts dev={len(DEV_FONTS)} tam={len(TAM_FONTS)} other={len(OTHER_FONTS)}", flush=True)
    if not DEV_FONTS or not TAM_FONTS or not OTHER_FONTS:
        raise SystemExit("missing fonts; pass --fonts-dir")

    texts = pathlib.Path(args.texts) if args.texts else default_input_dir("data", "texts")
    words = load_words(texts)
    print(f"{len(words)} corpus words from {texts}", flush=True)

    rng = random.Random(args.seed)
    order = list(range(len(words)))
    rng.shuffle(order)
    nval = max(2000, len(order) // 20)
    val_words = [words[i] for i in order[:nval]]
    train_words = [words[i] for i in order[nval:]]
    print(f"train words {len(train_words)} val words {len(val_words)}", flush=True)

    mixed = build_pool(train_words, args.pool_size, args.seed + 1)
    val_mixed = build_pool(val_words, args.val_pool_size, args.seed + 2)
    others = build_other(args.other_pool_size, args.seed + 3)
    val_others = build_other(args.val_other_size, args.seed + 4)
    by = build_index(mixed, others)
    val_by = build_index(val_mixed, val_others)
    xv, yv = gen_val(val_mixed, val_others, val_by, n=8000, seed=args.seed + 5)
    counts = {c: int((yv == CLS_ID[c]).sum()) for c in CLASSES}
    print(f"val {tuple(xv.shape)} counts {counts}", flush=True)

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    net = Net().to(dev)
    print(f"params {sum(p.numel() for p in net.parameters())/1e6:.3f}M device {dev}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=args.lr_min)
    loader = torch.utils.data.DataLoader(
        SynthDS(mixed, others, by), batch_size=args.batch, num_workers=args.workers,
        collate_fn=collate, drop_last=True, persistent_workers=True,
        worker_init_fn=lambda _: torch.set_num_threads(1))
    it = iter(loader)
    best = 0.0
    t0 = time.time()
    out = pathlib.Path(args.out) if args.out else (
        pathlib.Path("/kaggle/working/script_cls.pt")
        if pathlib.Path("/kaggle/working").exists() else REPO / "training/local/script_cls_v2.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.steps + 1):
        x, y = next(it)
        loss = F.cross_entropy(net(x.to(dev)), y.to(dev), label_smoothing=0.02)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            pred = predict(net, xv, dev)
            acc = (pred == yv).float().mean().item()
            per = {}
            for c in CLASSES:
                m = yv == CLS_ID[c]
                per[c] = (pred[m] == yv[m]).float().mean().item() if m.any() else 0.0
            print(f"step {step} loss {loss.item():.3f} acc {acc:.4f} " +
                  " ".join(f"{c}:{per[c]:.4f}" for c in CLASSES) +
                  f" {time.time()-t0:.0f}s", flush=True)
            if acc > best:
                best = acc
                torch.save({"state_dict": net.state_dict(), "classes": CLASSES,
                            "patch": PATCH, "val_acc": acc, "per_class": per,
                            "arch": "resnet_v2", "seed": args.seed}, out)
    torch.save({"state_dict": net.state_dict(), "classes": CLASSES, "patch": PATCH,
                "val_acc": best, "arch": "resnet_v2", "seed": args.seed},
               out.with_name("last.pt"))
    pred = predict(net, xv, dev)
    acc = (pred == yv).float().mean().item()
    per = {c: ((pred[yv == CLS_ID[c]] == yv[yv == CLS_ID[c]]).float().mean().item()
               if (yv == CLS_ID[c]).any() else 0.0) for c in CLASSES}
    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(yv.numpy(), pred.numpy()):
        cm[t, p] += 1
    metrics = {"val_acc": acc, "best_val_acc": best, "per_class": per,
               "confusion": cm.tolist(), "classes": CLASSES, "steps": args.steps,
               "params_M": sum(p.numel() for p in net.parameters()) / 1e6,
               "train_words": len(train_words), "val_words": len(val_words),
               "counts": counts, "seed": args.seed,
               "font_counts": {"dev": len(DEV_FONTS), "tam": len(TAM_FONTS),
                               "other": len(OTHER_FONTS)}}
    (out.parent / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"FINAL acc {acc:.4f} best {best:.4f} " +
          " ".join(f"{c}:{per[c]:.4f}" for c in CLASSES), flush=True)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
