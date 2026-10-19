#!/usr/bin/env python3
"""Minimal explicit line-recognition trainer (VGSL + CTC) for the maNipravALa data.

Uses the same kraken data pipeline (GroundTruthDataset + ImageInputTransforms,
dewarp + fixed-height resize) but an explicit training loop so loss/lr/eval
behavior is fully controlled and observable.

Usage:
  probe:  python3 training/scripts/train_lines2.py --probe 2000 --epochs 2 --lr 0.003
  full:   python3 training/scripts/train_lines2.py --epochs 10 --lr 0.002
"""
import argparse
import json
import math
import pathlib
import random
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from kraken.lib.util import parse_gt_path  # noqa: E402
from kraken.lib.dataset.recognition import GroundTruthDataset  # noqa: E402
from kraken.lib.dataset.utils import ImageInputTransforms  # noqa: E402
from kraken.lib.vgsl import TorchVGSLModel  # noqa: E402

BASE = pathlib.Path(__file__).resolve().parents[1]
REPO = BASE.parent if BASE.name == "training" else BASE

def make_spec(height, classes=None):
    spec = (f'[1,{height},0,1 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,32 Do0.1,2 Cr3,9,64 Do0.1,2 '
            'Mp2,2 Cr3,9,64 Do0.1,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do0.1,2 Lbx200 Do]')
    if classes:
        spec = f'[{spec[1:-1]} O1c{classes}]'
    return spec


SUBSAMPLE = 4


class LineDS(Dataset):
    def __init__(self, files, height=64):
        self.handle = GroundTruthDataset(normalization=None, whitespace_normalization=False,
                                         reorder=False, im_transforms=None)
        self.files = []
        for f in files:
            try:
                self.handle.add(line=parse_gt_path(f))
                self.files.append(f)
            except Exception as e:
                print("skip", f, e)
        self.handle.encode()
        self.handle.transforms = ImageInputTransforms(1, height, 0, 1, (16, 0), valid_norm=True)

    def __len__(self):
        return len(self.handle)

    def __getitem__(self, i):
        it = self.handle[i]
        return it['image'][0], it['target']


class CachedLineDS(Dataset):
    def __init__(self, cache_dir, augment=False):
        self.dir = pathlib.Path(cache_dir)
        meta = json.load(open(self.dir / 'meta.json'))
        self.files = [str(REPO / f) for f in meta['files']]
        self.widths = meta['widths']
        self.ylens = meta['ylens']
        self.height = meta['height']
        self.c2l = meta['c2l']
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = torch.load(self.dir / f'{i:06d}.pt')
        x, y = d['x'][0], d['y']
        if self.augment:
            x = augment_tensor(x)
        return x, y


def augment_tensor(x):
    import torch.nn.functional as F
    H, W = x.shape
    if random.random() < 0.8:
        ang = random.uniform(-1.0, 1.0) * math.pi / 180.0
        sc = random.uniform(0.92, 1.08)
        tx = random.uniform(-2.0, 2.0) * 2.0 / W
        ty = random.uniform(-1.5, 1.5) * 2.0 / H
        theta = torch.tensor([[sc * math.cos(ang), -sc * math.sin(ang), tx],
                              [sc * math.sin(ang), sc * math.cos(ang), ty]])
        grid = F.affine_grid(theta.unsqueeze(0), (1, 1, H, W), align_corners=False)
        x = F.grid_sample(x.unsqueeze(0).unsqueeze(0), grid, mode='bilinear',
                          padding_mode='zeros', align_corners=False)[0, 0]
    if random.random() < 0.5:
        x = (x * random.uniform(0.85, 1.15) + random.uniform(-0.1, 0.1)).clamp(0, 1)
    if random.random() < 0.3:
        x = (x + torch.randn_like(x) * random.uniform(0.01, 0.04)).clamp(0, 1)
    return x


def collate(batch):
    ims, tgts = zip(*batch)
    H = ims[0].shape[0]
    W = max(im.shape[1] for im in ims)
    out = torch.zeros(len(ims), 1, H, W)
    widths = []
    for i, im in enumerate(ims):
        out[i, 0, :, :im.shape[1]] = im
        widths.append(im.shape[1])
    return out, list(tgts), widths


class BucketSampler:
    def __init__(self, widths, batch, shuffle=True):
        self.batch, self.shuffle = batch, shuffle
        order = sorted(range(len(widths)), key=lambda i: widths[i])
        self.batches = [order[i:i + batch] for i in range(0, len(order), batch)]

    def __iter__(self):
        bs = list(self.batches)
        if self.shuffle:
            random.shuffle(bs)
        return iter(bs)

    def __len__(self):
        return len(self.batches)


def ids_to_text(rev, ids):
    out, prev = [], None
    for i in ids:
        if i == 0:
            prev = None
            continue
        if i != prev:
            out.append(rev.get(i, ''))
        prev = i
    return ''.join(out)


def greedy_text(net, ims):
    out = net(ims)
    lg = (out[0] if isinstance(out, tuple) else out).squeeze(2).permute(0, 2, 1)
    return lg.argmax(-1)


def batch_loss(net, criterion, ims, tgts, widths):
    out = net(ims)
    lg = (out[0] if isinstance(out, tuple) else out).squeeze(2).permute(2, 0, 1).log_softmax(2)
    T = lg.shape[0]
    lens = torch.tensor([max(1, min(T, w // SUBSAMPLE)) for w in widths], dtype=torch.long)
    flat = torch.cat([t.long() for t in tgts])
    tlens = torch.tensor([len(t) for t in tgts], dtype=torch.long)
    if not bool((tlens <= lens).any()):
        return None, 0
    return criterion(lg, flat, lens, tlens), int((tlens <= lens).sum())


def make_model(spec):
    return TorchVGSLModel(vgsl=spec, mode='recognition')


def evaluate(net, rev, ev_items, max_n=None):
    import jiwer
    pairs = []
    was_training = net.training
    net.eval()
    dev = next(net.parameters()).device
    with torch.no_grad():
        for (ims, tgts, gts, widths) in ev_items:
            ims = ims.to(dev)
            out = net(ims)
            lg = (out[0] if isinstance(out, tuple) else out).squeeze(2).permute(2, 0, 1)
            preds = lg.argmax(2)
            for b in range(ims.shape[0]):
                tlen = max(1, min(preds.shape[0], widths[b] // SUBSAMPLE))
                pred = ids_to_text(rev, preds[:tlen, b].tolist())
                pairs.append((gts[b], pred))
            if max_n and len(pairs) >= max_n:
                break
    if was_training:
        net.train()
    gt = [p[0] for p in pairs]
    pr = [p[1] for p in pairs]
    return {'n': len(pairs), 'cer': jiwer.cer(gt, pr), 'wer': jiwer.wer(gt, pr)}


def prep_eval_batches(ds, files, batch=16):
    gts = [open(f.replace('.png', '.gt.txt'), encoding='utf-8').read() for f in files]
    widths_all = [ds[i][0].shape[1] for i in range(len(ds))]
    order = sorted(range(len(ds)), key=lambda i: widths_all[i])
    bs = [order[i:i + batch] for i in range(0, len(order), batch)]
    out = []
    for b in bs:
        ims, tgts = zip(*[ds[i] for i in b])
        W = max(im.shape[1] for im in ims)
        H = ims[0].shape[0]
        stack = torch.zeros(len(ims), 1, H, W)
        widths = []
        for j, im in enumerate(ims):
            stack[j, 0, :, :im.shape[1]] = im
            widths.append(im.shape[1])
        out.append((stack, list(tgts), [gts[i] for i in b], widths))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=0.002)
    ap.add_argument('--min-lr', type=float, default=0.0002)
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--probe', type=int, default=0, help='use first N train lines')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--out', default='training/local/manual')
    ap.add_argument('--eval-n', type=int, default=150)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--height', type=int, default=64)
    ap.add_argument('--augment', action='store_true', help='random affine/contrast/noise on cached tensors')
    ap.add_argument('--cache', default='training/local/lines_cache',
                    help='dir with prebuilt line tensors (build_line_cache.py); empty to disable')
    args = ap.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)

    t0 = time.time()
    cache = REPO / args.cache if args.cache else None
    use_cache = bool(cache and (cache / 'train' / 'meta.json').exists() and not args.probe)
    if use_cache:
        train_ds = CachedLineDS(cache / 'train', augment=args.augment)
        eval_ds = CachedLineDS(cache / 'eval')
        if train_ds.height != args.height:
            raise SystemExit(f'cache height {train_ds.height} != --height {args.height}; '
                             'rebuild cache or change --height')
        rev = {v: ch for ch, vs in train_ds.c2l.items() for v in vs}
        widths = train_ds.widths
        print(f'cache loaded: train {len(train_ds)} eval {len(eval_ds)} '
              f'({time.time()-t0:.0f}s)', flush=True)
    else:
        train_files = [l.strip() for l in open(REPO / 'training/local/lines/train_files.txt')]
        eval_files = [l.strip() for l in open(REPO / 'training/local/lines/eval_files.txt')]
        if args.probe:
            train_files = train_files[:args.probe]
        print(f'train lines {len(train_files)} eval lines {len(eval_files)}', flush=True)
        train_ds = LineDS([str(REPO / f) for f in train_files], height=args.height)
        eval_ds = LineDS([str(REPO / f) for f in eval_files], height=args.height)
        print(f'datasets built in {time.time()-t0:.0f}s', flush=True)
        rev = {v: ch for ch, vs in train_ds.handle.codec.c2l.items() for v in vs}
        from PIL import Image as PILImage
        widths = [PILImage.open(f).size[0] for f in train_ds.files]

    # Drop samples whose encoded target cannot fit the CTC time steps
    # (T = width // SUBSAMPLE). Almost all of these are broken build_lines
    # alignments (e.g. a 60px sliver carrying a 1800-char commentary target).
    from torch.utils.data import Subset
    if use_cache:
        train_ylens, eval_ylens = train_ds.ylens, eval_ds.ylens
        train_widths, eval_widths = train_ds.widths, eval_ds.widths
    else:
        train_ylens = [int(t.shape[0]) for _, t in train_ds.handle.training_set]
        train_widths = widths
        eval_ylens = [int(t.shape[0]) for _, t in eval_ds.handle.training_set]
        eval_widths = [eval_ds[i][0].shape[1] for i in range(len(eval_ds))]
    keep = [i for i in range(len(train_ds)) if train_ylens[i] * SUBSAMPLE <= train_widths[i]]
    keep_e = [i for i in range(len(eval_ds)) if eval_ylens[i] * SUBSAMPLE <= eval_widths[i]]
    print(f'feasible at subsample {SUBSAMPLE}: train {len(keep)}/{len(train_ds)} '
          f'(dropped {len(train_ds)-len(keep)}), eval {len(keep_e)}/{len(eval_ds)} '
          f'(dropped {len(eval_ds)-len(keep_e)})', flush=True)
    widths = [train_widths[i] for i in keep]
    eval_files_kept = [eval_ds.files[i] for i in keep_e]
    train_ds = Subset(train_ds, keep)
    eval_ds = Subset(eval_ds, keep_e)

    sampler = BucketSampler(widths, args.batch)
    dl = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate, num_workers=args.workers,
                    pin_memory=True, worker_init_fn=lambda _: torch.set_num_threads(1))

    ev_items = prep_eval_batches(eval_ds, eval_files_kept, batch=args.batch)

    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    classes = max(rev.keys()) + 1
    spec = make_spec(args.height, classes)
    print(f'model: {classes} classes (blank + {classes-1} symbols), spec {spec}', flush=True)
    model = make_model(spec)
    net = model.nn.to(dev)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    crit = nn.CTCLoss(blank=0, zero_infinity=True)

    steps_per_epoch = len(dl)
    total = steps_per_epoch * args.epochs
    outdir = REPO / args.out
    outdir.mkdir(parents=True, exist_ok=True)

    step = 0
    best = 9.9
    for ep in range(args.epochs):
        t0 = time.time()
        run_loss, run_n, skipped = 0.0, 0, 0
        for ims, tgts, wlist in dl:
            ims = ims.to(dev)
            tgts = [t.to(dev) for t in tgts]
            for g in opt.param_groups:
                if step < args.warmup:
                    g['lr'] = args.lr * (step + 1) / args.warmup
                else:
                    frac = (step - args.warmup) / max(1, total - args.warmup)
                    g['lr'] = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * frac))
            opt.zero_grad()
            loss, n_ok = batch_loss(net, crit, ims, tgts, wlist)
            if loss is None or n_ok == 0:
                step += 1
                skipped += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            run_loss += float(loss)
            run_n += 1
            step += 1
            if step % 100 == 0:
                print(f'ep{ep} step{step}/{total} loss {run_loss/max(1,run_n):7.1f} '
                      f'lr {opt.param_groups[0]["lr"]:.5f} skip {skipped} {time.time()-t0:.0f}s', flush=True)
                run_loss, run_n = 0.0, 0
        ev = evaluate(net, rev, ev_items, max_n=args.eval_n)
        print(f'== epoch {ep} done {time.time()-t0:.0f}s val_cer {ev["cer"]:.3f} wer {ev["wer"]:.3f} n {ev["n"]}', flush=True)
        torch.save({'spec': spec, 'state_dict': net.state_dict(), 'epoch': ep,
                    'val_cer': ev['cer']}, outdir / 'last.pt')
        if ev['cer'] < best:
            best = ev['cer']
            torch.save({'spec': spec, 'state_dict': net.state_dict(), 'epoch': ep,
                        'val_cer': ev['cer']}, outdir / 'best.pt')
        with open(outdir / 'log.jsonl', 'a') as f:
            f.write(json.dumps({'epoch': ep, 'val_cer': ev['cer'], 'val_wer': ev['wer'],
                                'time': time.time() - t0}) + '\n')
    print('best val_cer', best, flush=True)


if __name__ == '__main__':
    main()
