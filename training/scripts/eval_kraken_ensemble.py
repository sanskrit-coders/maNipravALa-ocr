#!/usr/bin/env python3
"""Evaluate a probability-averaged ensemble of checkpoints on the full cached eval set.

Usage:
  python3 training/scripts/eval_kraken_ensemble.py \
      --ckpt training/local/kaggle_v10_best.pt \
      --ckpt training/local/kaggle_v13_best.pt \
      --dump evaluation/line_eval_kaggle_ens.jsonl
"""
import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from train_lines2 import CachedLineDS, SUBSAMPLE, ids_to_text  # noqa: E402
from eval_kraken_manual import metrics  # noqa: E402
from kraken.lib.vgsl import TorchVGSLModel  # noqa: E402

BASE = pathlib.Path(__file__).resolve().parents[1]
REPO = BASE.parent if BASE.name == "training" else BASE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--cache", default="training/local/lines_cache")
    ap.add_argument("--dump", default="")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    nets = []
    for c in args.ckpt:
        ckpt = torch.load(REPO / c, map_location="cpu", weights_only=False)
        m = TorchVGSLModel(vgsl=ckpt["spec"], mode="recognition")
        m.nn.load_state_dict(ckpt["state_dict"])
        nets.append(m.nn.to(dev).eval())
        print(f"loaded {c} (epoch {ckpt.get('epoch')}, val_cer {ckpt.get('val_cer'):.4f})")

    ds = CachedLineDS(REPO / args.cache / "eval")
    train_meta = json.load(open(REPO / args.cache / "train" / "meta.json"))
    rev = {v: ch for ch, vs in train_meta["c2l"].items() for v in vs}
    keep = [i for i in range(len(ds)) if ds.ylens[i] * SUBSAMPLE <= ds.widths[i]]

    order = sorted(keep, key=lambda i: ds.widths[i])
    batches = [order[i:i + args.batch] for i in range(0, len(order), args.batch)]
    pairs = []
    with torch.no_grad():
        for b in batches:
            items = [ds[i] for i in b]
            ims = [it[0] for it in items]
            H = ims[0].shape[0]
            W = max(im.shape[1] for im in ims)
            stack = torch.zeros(len(ims), 1, H, W)
            widths = []
            for j, im in enumerate(ims):
                stack[j, 0, :, :im.shape[1]] = im
                widths.append(im.shape[1])
            probs = None
            for net in nets:
                out = net(stack.to(dev))
                lg = (out[0] if isinstance(out, tuple) else out).squeeze(2).permute(2, 0, 1)
                p = lg.log_softmax(2).exp()
                probs = p if probs is None else probs + p
            probs /= len(nets)
            preds = probs.argmax(2)
            for j, i in enumerate(b):
                tlen = max(1, min(preds.shape[0], widths[j] // SUBSAMPLE))
                pred = ids_to_text(rev, preds[:tlen, j].tolist())
                name = pathlib.Path(ds.files[i]).stem
                gt = pathlib.Path(ds.files[i].replace(".png", ".gt.txt")).read_text(encoding="utf-8")
                pairs.append((name, gt, pred))

    m = metrics([(g, p) for _, g, p in pairs])
    m["ckpts"] = args.ckpt
    print(json.dumps(m, indent=1))
    if args.dump:
        with open(REPO / args.dump, "w", encoding="utf-8") as f:
            for name, gt, pred in pairs:
                f.write(json.dumps({"name": name, "gt": gt, "pred": pred}, ensure_ascii=False) + "\n")
        print("dumped", args.dump)


if __name__ == "__main__":
    main()
