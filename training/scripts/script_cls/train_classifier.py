#!/usr/bin/env python3
"""Train a patch-level script classifier (devanagari / tamil / other / null) on
synthetic renders of words drawn from the repo corpora.

A pool of words is rendered once (per-script-run fonts, PIL+raqm shaping); 48x48
patches are then sampled in parallel workers, labeled by the script of the
character under the patch center ('null' from gaps/margins), with scan-like
augmentations. Writes training/local/script_cls.pt and prints val accuracy.

Usage:
  python3 training/scripts/script_cls/train_classifier.py [--steps 4000] [--device cuda:0]
"""
import argparse
import pathlib
import random
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

BASE = pathlib.Path(__file__).resolve().parents[2]
REPO = BASE.parent if BASE.name == "training" else BASE

PATCH = 48
CANVAS_W = 1600
MARGIN = 60

DEV_FONTS = [
    "/usr/share/fonts/sanskrit-fonts/ttf-devanagari/noto/NotoSansDevanagari-Regular.ttf",
    "/usr/share/fonts/sanskrit-fonts/ttf-devanagari/noto/NotoSansDevanagari-Bold.ttf",
    "/usr/share/fonts/sanskrit-fonts/ttf-devanagari/siddhanta.ttf",
    "/usr/share/fonts/sanskrit-fonts/ttf-devanagari/siddhanta-variations/siddhanta-calcutta.ttf",
    "/usr/share/fonts/sanskrit-fonts/ttf-devanagari/siddhanta-variations/siddhanta-nepali.ttf",
]
TAM_FONTS = [
    "/usr/share/fonts/sanskrit-fonts/ttf-tamil/NotoSansTamil-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansTamil-Bold.ttf",
    "/usr/share/fonts/sanskrit-fonts/ttf-tamil/TAMu_Kalyani.ttf",
    "/usr/share/fonts/sanskrit-fonts/ttf-tamil/TAMu_Maduram.ttf",
    "/usr/share/fonts/sanskrit-fonts/ttf-tamil/TSCu_Paranar.ttf",
    "/usr/share/fonts/truetype/lohit-tamil-classical/Lohit-Tamil-Classical.ttf",
]
OTHER_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
]

CLASSES = ["dev", "tam", "other", "null"]
CLS_ID = {c: i for i, c in enumerate(CLASSES)}

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


def load_words():
    words = []
    for sub in ("core", "commentary"):
        for f in sorted((REPO / "data/texts" / sub).glob("*.txt")):
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
    size = rng.randint(42, 62)
    fonts = {"dev": rng.choice(DEV_FONTS), "tam": rng.choice(TAM_FONTS),
             "other": rng.choice(OTHER_FONTS)}
    runs = split_runs(list(word), prof)
    if len(runs) > 4:
        return None
    img = Image.new("L", (CANVAS_W, 160), 255)
    d = ImageDraw.Draw(img)
    baseline = 110
    x = MARGIN
    char_pos = []
    for text, script in runs:
        f = get_font(fonts[script], size)
        for k in range(len(text)):
            pre = d.textlength(text[:k], font=f)
            adv = d.textlength(text[k:k + 1], font=f)
            char_pos.append({"x0": x + pre, "x1": x + pre + adv, "cls": script})
        d.text((x, baseline), text, font=f, fill=0, anchor="ls")
        x += d.textlength(text, font=f)
    arr = np.asarray(img)
    cols = np.where((arr < 200).any(axis=0))[0]
    if len(cols) == 0:
        return None
    x0 = max(0, int(cols[0]) - MARGIN)
    x1 = min(CANVAS_W, int(cols[-1]) + MARGIN + 1)
    crop = np.ascontiguousarray(arr[:, x0:x1])
    for cp in char_pos:
        cp["x0"] -= x0
        cp["x1"] -= x0
    return {"img": crop, "chars": char_pos, "baseline": baseline, "w": x1 - x0}


def build_pool(words, n=10000, seed=1234):
    rng = random.Random(seed)
    sample = [words[rng.randrange(len(words))] for _ in range(n)]
    args = [(w, p, rng.randrange(1 << 30)) for w, p in sample]
    with Pool(12) as pool:
        recs = [r for r in pool.map(render_word, args, chunksize=20) if r is not None]
    print(f"rendered {len(recs)} word images", flush=True)
    return recs


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
        im = im.rotate(rng.uniform(-2.5, 2.5), resample=Image.BILINEAR, fillcolor=255)
    if rng.random() < 0.3:
        s = rng.uniform(0.75, 1.0)
        im = im.resize((max(1, int(PATCH * s)), max(1, int(PATCH * s))), Image.BILINEAR)
        im = im.resize((PATCH, PATCH), Image.BILINEAR)
    if rng.random() < 0.4:
        from PIL import ImageFilter
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.4)))
    a = np.asarray(im).astype(np.float32) / 255.0
    if rng.random() < 0.7:
        a = np.clip(a * rng.uniform(0.6, 1.4) + rng.uniform(-0.08, 0.08), 0, 1)
    if rng.random() < 0.5:
        a = np.power(a, rng.uniform(0.7, 1.5))
    if rng.random() < 0.4:
        a = a + np.random.randn(*a.shape).astype(np.float32) * rng.uniform(0.005, 0.05)
    return np.clip(1.0 - a, 0, 1).astype(np.float32)


def make_sample(pool, rng):
    rec = pool[rng.randrange(len(pool))]
    img, chars, baseline = rec["img"], rec["chars"], rec["baseline"]
    u = rng.random()
    cls = "dev" if u < 0.42 else "tam" if u < 0.84 else "other" if u < 0.92 else "null"
    if cls == "null":
        gaps = []
        prev = 0
        for cp in chars:
            if cp["x0"] - prev > 26:
                gaps.append((prev + 10, cp["x0"] - 10))
            prev = max(prev, cp["x1"])
        if img.shape[1] - prev > 26:
            gaps.append((prev + 10, img.shape[1] - 10))
        if not gaps:
            return None
        gx0, gx1 = rng.choice(gaps)
        cx = rng.uniform(gx0, gx1)
        cy = baseline - rng.uniform(-10, 42)
        lab = CLS_ID["null"]
    else:
        cands = [cp for cp in chars if cp["cls"] == cls]
        if not cands:
            return None
        cp = rng.choice(cands)
        if rng.random() < 0.5:
            cx = (cp["x0"] + cp["x1"]) / 2 + rng.uniform(-0.3, 0.3) * max(cp["x1"] - cp["x0"], 6)
        else:
            cx = rng.uniform(cp["x0"], cp["x1"])
        cy = baseline - rng.uniform(2, 42)
        lab = CLS_ID[cls]
    p = sample_patch(img, cx, cy)
    if p is None:
        return None
    return augment(p, rng), lab


class SynthDS(torch.utils.data.Dataset):
    def __init__(self, pool):
        self.pool = pool

    def __len__(self):
        return 1 << 40

    def __getitem__(self, i):
        rng = random.Random((i * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFF)
        for _ in range(5):
            s = make_sample(self.pool, rng)
            if s is not None:
                return torch.from_numpy(s[0]).unsqueeze(0), s[1]
        return torch.zeros(1, PATCH, PATCH), CLS_ID["null"]


def collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(96, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.f = nn.Linear(128, len(CLASSES))

    def forward(self, x):
        return self.f(self.c(x).flatten(1))


def gen_val(pool, n=3000, seed=99):
    rng = random.Random(seed)
    xs, ys = [], []
    while len(xs) < n:
        s = make_sample(pool, rng)
        if s:
            xs.append(s[0])
            ys.append(s[1])
    return torch.from_numpy(np.stack(xs)).unsqueeze(1), torch.tensor(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="training/local/script_cls.pt")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--pool-size", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    words = load_words()
    print(f"{len(words)} corpus words", flush=True)
    pool = build_pool(words, n=args.pool_size, seed=args.seed)
    xv, yv = gen_val(pool)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=args.lr_min)
    ds = SynthDS(pool)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                                         collate_fn=collate, drop_last=True, persistent_workers=True)

    def val():
        net.eval()
        with torch.no_grad():
            pred = net(xv.to(dev)).argmax(1).cpu()
        acc = (pred == yv).float().mean().item()
        per = {}
        for ci in range(len(CLASSES)):
            m = yv == ci
            if m.sum() > 0:
                per[CLASSES[ci]] = (pred[m] == yv[m]).float().mean().item()
        net.train()
        return acc, per

    it = iter(loader)
    for step in range(1, args.steps + 1):
        x, y = next(it)
        loss = F.cross_entropy(net(x.to(dev)), y.to(dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 250 == 0:
            acc, per = val()
            print(f"step {step} loss {loss.item():.3f} val_acc {acc:.3f} " +
                  " ".join(f"{k}:{v:.3f}" for k, v in per.items()), flush=True)
    acc, per = val()
    print(f"FINAL val_acc {acc:.4f} " + " ".join(f"{k}:{v:.4f}" for k, v in per.items()))
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "classes": CLASSES, "patch": PATCH}, out)
    print("saved", out)


if __name__ == "__main__":
    main()
