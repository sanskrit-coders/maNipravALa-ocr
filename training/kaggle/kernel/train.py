#!/usr/bin/env python3
"""Kaggle GPU trainer: same pipeline as training/scripts/train_lines2.py but reads the
uint8 memmap cache (pack_cache_uint8.py) and supports fp16 AMP for T4.

Uploaded as a Kaggle script kernel; outputs last.pt/best.pt/log.jsonl to /kaggle/working.
"""
import argparse
import gzip
import json
import math
import pathlib
import random
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SUBSAMPLE = 4
OUT = pathlib.Path('/kaggle/working')


def make_spec(height, classes=None):
    spec = (f'[1,{height},0,1 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,32 Do0.1,2 Cr3,9,64 Do0.1,2 '
            'Mp2,2 Cr3,9,64 Do0.1,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do0.1,2 Lbx200 Do]')
    if classes:
        spec = f'[{spec[1:-1]} O1c{classes}]'
    return spec


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


class MemmapDS(Dataset):
    def __init__(self, bin_path, entries, augment=False):
        self.mm = np.memmap(bin_path, dtype=np.uint8, mode='r')
        self.entries = entries
        self.augment = augment

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        x = torch.from_numpy(self.mm[e['off']:e['off'] + e['h'] * e['w']]
                             .reshape(e['h'], e['w']).copy()).float() / 255.0
        if self.augment:
            x = augment_tensor(x)
        return x, torch.tensor(e['ids'], dtype=torch.int32)


def collate(batch):
    xs, ys = zip(*batch)
    H = xs[0].shape[0]
    W = max(x.shape[1] for x in xs)
    out = torch.zeros(len(xs), 1, H, W)
    widths = []
    for i, x in enumerate(xs):
        out[i, 0, :, :x.shape[1]] = x
        widths.append(x.shape[1])
    return out, list(ys), widths


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


def step_loss(net, criterion, ims, ys, widths, amp):
    with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
        out = net(ims)
    lg = (out[0] if isinstance(out, tuple) else out).squeeze(2).permute(2, 0, 1).float().log_softmax(2)
    T = lg.shape[0]
    lens = torch.tensor([max(1, min(T, w // SUBSAMPLE)) for w in widths], dtype=torch.long)
    flat = torch.cat([y.long() for y in ys])
    tlens = torch.tensor([len(y) for y in ys], dtype=torch.long)
    if not bool((tlens <= lens).any()):
        return None, 0
    return criterion(lg, flat, lens, tlens), int((tlens <= lens).sum())


def evaluate(net, rev, ev_batches, amp):
    import jiwer
    pairs = []
    was_training = net.training
    net.eval()
    with torch.no_grad():
        for ims, _ys, gts, widths in ev_batches:
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                out = net(ims)
            lg = (out[0] if isinstance(out, tuple) else out).squeeze(2).permute(2, 0, 1)
            preds = lg.argmax(2)
            for b in range(ims.shape[0]):
                tlen = max(1, min(preds.shape[0], widths[b] // SUBSAMPLE))
                pairs.append((gts[b], ids_to_text(rev, preds[:tlen, b].tolist())))
    if was_training:
        net.train()
    gt = [p[0] for p in pairs]
    pr = [p[1] for p in pairs]
    return {'n': len(pairs), 'cer': jiwer.cer(gt, pr), 'wer': jiwer.wer(gt, pr)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--lr', type=float, default=0.003)
    ap.add_argument('--min-lr', type=float, default=5e-5)
    ap.add_argument('--warmup', type=int, default=400)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--height', type=int, default=64)
    ap.add_argument('--no-augment', dest='augment', action='store_false')
    ap.add_argument('--no-amp', dest='amp', action='store_false')
    ap.add_argument('--time-limit-s', type=float, default=11 * 3600)
    ap.add_argument('--out', default=str(OUT / 'manual2'))
    args = ap.parse_args()
    args.augment = getattr(args, 'augment', True)
    args.amp = getattr(args, 'amp', True) and torch.cuda.is_available()

    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)

    dsdir = None
    for p in pathlib.Path('/kaggle/input').rglob('index.json'):
        dsdir = p.parent
        break
    if dsdir is None:
        raise SystemExit('index.json not found under /kaggle/input')
    print('dataset dir:', dsdir, flush=True)

    work = OUT / 'cache'
    work.mkdir(parents=True, exist_ok=True)
    for name in ('train.bin', 'eval.bin'):
        dst = work / name
        if not dst.exists():
            t0 = time.time()
            with gzip.open(dsdir / (name + '.gz'), 'rb') as fi, open(dst, 'wb') as fo:
                shutil.copyfileobj(fi, fo)
            print(f'decompressed {name} in {time.time()-t0:.0f}s', flush=True)

    index = json.load(open(dsdir / 'index.json'))
    assert index['height'] == args.height, (index['height'], args.height)
    c2l, height = index['c2l'], index['height']
    rev = {v: ch for ch, vs in c2l.items() for v in vs}
    classes = max(rev.keys()) + 1

    def feasible(e):
        return e['ylen'] * SUBSAMPLE <= e['w']

    train_entries = [e for e in index['train'] if feasible(e)]
    eval_entries = [e for e in index['eval'] if feasible(e)]
    print(f'feasible: train {len(train_entries)}/{len(index["train"])} '
          f'eval {len(eval_entries)}/{len(index["eval"])}', flush=True)

    train_ds = MemmapDS(work / 'train.bin', train_entries, augment=args.augment)
    sampler = BucketSampler([e['w'] for e in train_entries], args.batch)
    dl = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate, num_workers=args.workers,
                    pin_memory=True, worker_init_fn=lambda _: torch.set_num_threads(1))

    order = sorted(range(len(eval_entries)), key=lambda i: eval_entries[i]['w'])
    eval_ds = MemmapDS(work / 'eval.bin', eval_entries)
    ev_batches = []
    for i in range(0, len(order), args.batch):
        b = order[i:i + args.batch]
        xs, ys = zip(*[eval_ds[j] for j in b])
        H = xs[0].shape[0]
        W = max(x.shape[1] for x in xs)
        stack = torch.zeros(len(xs), 1, H, W)
        widths = []
        for k, x in enumerate(xs):
            stack[k, 0, :, :x.shape[1]] = x
            widths.append(x.shape[1])
        ev_batches.append((stack, list(ys), [eval_entries[j]['gt'] for j in b], widths))

    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    spec = make_spec(height, classes)
    print(f'model: {classes} classes, batch {args.batch}, amp {args.amp}, device {dev}', flush=True)
    from kraken.lib.vgsl import TorchVGSLModel
    model = TorchVGSLModel(vgsl=spec, mode='recognition')
    net = model.nn.to(dev)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    crit = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    steps_per_epoch = len(dl)
    total = steps_per_epoch * args.epochs
    t_start = time.time()
    step = 0
    best = 9.9
    for ep in range(args.epochs):
        t0 = time.time()
        run_loss, run_n, skipped = 0.0, 0, 0
        for ims, ys, wlist in dl:
            ims = ims.to(dev, non_blocking=True)
            ys = [y.to(dev) for y in ys]
            opt.zero_grad(set_to_none=True)
            loss, n_ok = step_loss(net, crit, ims, ys, wlist, args.amp)
            if loss is None or n_ok == 0:
                step += 1
                skipped += 1
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            for g in opt.param_groups:
                if step < args.warmup:
                    g['lr'] = args.lr * (step + 1) / args.warmup
                else:
                    frac = (step - args.warmup) / max(1, total - args.warmup)
                    g['lr'] = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * frac))
            run_loss += float(loss)
            run_n += 1
            step += 1
            if step % 50 == 0:
                el = time.time() - t_start
                print(f'ep{ep} step{step}/{total} loss {run_loss/max(1,run_n):7.1f} '
                      f'lr {opt.param_groups[0]["lr"]:.5f} skip {skipped} '
                      f'{time.time()-t0:.0f}s elapsed {el/60:.1f}m', flush=True)
                run_loss, run_n = 0.0, 0
        ev = evaluate(net, rev, ev_batches, args.amp)
        print(f'== epoch {ep} done {time.time()-t0:.0f}s val_cer {ev["cer"]:.3f} '
              f'wer {ev["wer"]:.3f} n {ev["n"]}', flush=True)
        torch.save({'spec': spec, 'state_dict': net.state_dict(), 'epoch': ep,
                    'val_cer': ev['cer']}, outdir / 'last.pt')
        if ev['cer'] < best:
            best = ev['cer']
            torch.save({'spec': spec, 'state_dict': net.state_dict(), 'epoch': ep,
                        'val_cer': ev['cer']}, outdir / 'best.pt')
        with open(outdir / 'log.jsonl', 'a') as f:
            f.write(json.dumps({'epoch': ep, 'val_cer': ev['cer'], 'val_wer': ev['wer'],
                                'time': time.time() - t0}) + '\n')
        if time.time() - t_start > args.time_limit_s:
            print('time limit reached, stopping', flush=True)
            break
    print('best val_cer', best, flush=True)


if __name__ == '__main__':
    main()
