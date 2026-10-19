#!/usr/bin/env python3
"""Kaggle GPU trainer: same pipeline as training/scripts/train_lines2.py but reads the
uint8 memmap cache (pack_cache_uint8.py) and supports fp16 AMP for T4.

Uploaded as a Kaggle script kernel; outputs last.pt/best.pt/log.jsonl to /kaggle/working.
"""
import argparse
import json
import math
import pathlib
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SUBSAMPLE = 4
OUT = pathlib.Path('/kaggle/working')


class ActConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        p = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
        self.co = nn.Conv2d(in_channels, out_channels, kernel_size, stride=(1, 1), padding=p)

    def forward(self, x):
        return torch.relu(self.co(x))


class Wrapped(nn.Module):
    def __init__(self, layer, kind):
        super().__init__()
        self.layer = layer
        self.kind = kind

    def forward(self, x):
        if self.kind == 'drop2d':
            return F.dropout2d(x, self.layer.p, self.training)
        if self.kind == 'drop':
            return F.dropout(x, self.layer.p, self.training)
        return self.layer(x)


class ReshapeS(nn.Module):
    def __init__(self, src_dim, part_a, part_b, high, low):
        super().__init__()
        self.src_dim, self.part_a, self.part_b = src_dim, part_a, part_b
        self.high, self.low = high, low

    def forward(self, x):
        x = x.reshape(x.shape[:self.src_dim] + (self.part_a, self.part_b) + x.shape[self.src_dim + 1:])
        dest = self.low
        src_dim = self.src_dim
        if self.high != src_dim:
            dest = self.high
        else:
            src_dim += 1
        perm = list(range(len(x.shape)))
        step = 1 if dest > src_dim else -1
        for k in range(src_dim, dest, step):
            perm[k], perm[k + step] = perm[k + step], perm[k]
        x = x.permute(perm)
        return x.reshape(x.shape[:dest] + (x.shape[dest] * x.shape[dest + 1],) + x.shape[dest + 2:])


class TransposedSummarizingRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layer = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.output_size = 2 * hidden_size

    def forward(self, x):
        x = x.permute(2, 0, 3, 1)
        siz = x.size()
        x = x.contiguous().view(-1, siz[2], siz[3])
        o, _ = self.layer(x)
        o = o.view(siz[0], siz[1], siz[2], self.output_size)
        return o.permute(1, 3, 0, 2)


class LinSoftmax(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.lin = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = x.transpose(1, 3)
        return self.lin(x).transpose(1, 3)


def init_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.LSTM):
            for p in m.parameters():
                if p.data.dim() == 2:
                    nn.init.orthogonal_(p.data)
                else:
                    nn.init.constant_(p.data[len(p) // 4:len(p) // 2], 1.0)
        elif isinstance(m, nn.Conv2d):
            for p in m.parameters():
                nn.init.uniform_(p.data, -0.1, 0.1)


class LineNet(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.C_0 = ActConv2D(1, 32, (3, 13))
        self.Do_1 = Wrapped(nn.Dropout2d(0.1), 'drop2d')
        self.Mp_2 = Wrapped(nn.MaxPool2d(2, 2), 'pool')
        self.C_3 = ActConv2D(32, 32, (3, 13))
        self.Do_4 = Wrapped(nn.Dropout2d(0.1), 'drop2d')
        self.C_5 = ActConv2D(32, 64, (3, 9))
        self.Do_6 = Wrapped(nn.Dropout2d(0.1), 'drop2d')
        self.Mp_7 = Wrapped(nn.MaxPool2d(2, 2), 'pool')
        self.C_8 = ActConv2D(64, 64, (3, 9))
        self.Do_9 = Wrapped(nn.Dropout2d(0.1), 'drop2d')
        self.S_10 = ReshapeS(2, 1, -1, 2, 1)
        self.L_11 = TransposedSummarizingRNN(1024, 200)
        self.Do_12 = Wrapped(nn.Dropout2d(0.1), 'drop2d')
        self.L_13 = TransposedSummarizingRNN(400, 200)
        self.Do_14 = Wrapped(nn.Dropout2d(0.1), 'drop2d')
        self.L_15 = TransposedSummarizingRNN(400, 200)
        self.Do_16 = Wrapped(nn.Dropout(0.5), 'drop')
        self.O_17 = LinSoftmax(400, classes)
        init_weights(self)

    def forward(self, x):
        for m in self.children():
            x = m(x)
        return x, None


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


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def evaluate(net, rev, ev_batches, amp):
    pairs = []
    was_training = net.training
    net.eval()
    with torch.no_grad():
        for ims, _ys, gts, widths in ev_batches:
            ims = ims.to(next(net.parameters()).device)
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
    num_c = sum(edit_distance(a, b) for a, b in zip(gt, pr))
    den_c = max(1, sum(len(a) for a in gt))
    gw = [a.split() for a in gt]
    pw = [b.split() for b in pr]
    num_w = sum(edit_distance(a, b) for a, b in zip(gw, pw))
    den_w = max(1, sum(len(a) for a in gw))
    return {'n': len(pairs), 'cer': num_c / den_c, 'wer': num_w / den_w}


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
    ap.add_argument('--augment', dest='augment', action='store_true')
    ap.add_argument('--amp', dest='amp', action='store_true')
    ap.add_argument('--time-limit-s', type=float, default=11 * 3600)
    ap.add_argument('--out', default=str(OUT / 'manual2'))
    ap.set_defaults(augment=False)
    args = ap.parse_args()
    args.amp = args.amp and torch.cuda.is_available()
    args.amp = getattr(args, 'amp', False) and torch.cuda.is_available()

    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)

    IN = pathlib.Path('/kaggle/input')
    found = next(IN.rglob('train.bin'), None)
    if found is not None:
        work = found.parent
        print('dataset dir:', work, flush=True)
    else:
        zips = list(IN.rglob('*.zip'))
        if not zips:
            raise SystemExit('no train.bin or dataset zip found under /kaggle/input')
        work = OUT / 'cache'
        work.mkdir(parents=True, exist_ok=True)
        import zipfile
        t0 = time.time()
        with zipfile.ZipFile(zips[0]) as z:
            z.extractall(work)
        print(f'extracted {zips[0]} in {time.time()-t0:.0f}s', flush=True)

    index = json.load(open(work / 'index.json'))
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
    net = LineNet(classes).to(dev)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    crit = nn.CTCLoss(blank=0, zero_infinity=True)
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

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
            if loss is None or n_ok == 0 or not torch.isfinite(loss):
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
            run_loss += float(loss.detach())
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
