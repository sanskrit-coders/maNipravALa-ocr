#!/usr/bin/env python3
"""Evaluate a checkpoint from train_lines2.py on the full cached eval set (all feasible lines).

Usage:
  python3 training/scripts/eval_kraken_manual.py --ckpt training/local/manual2/best.pt \
      --dump evaluation/line_eval_kraken.jsonl
"""
import argparse
import json
import pathlib
import re
import sys

import jiwer
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from train_lines2 import CachedLineDS, SUBSAMPLE, ids_to_text  # noqa: E402
from kraken.lib.vgsl import TorchVGSLModel  # noqa: E402

BASE = pathlib.Path(__file__).resolve().parents[1]
REPO = BASE.parent if BASE.name == "training" else BASE


def strip_ws(s):
    return re.sub(r"\s+", "", s)


def metrics(pairs):
    gt = [p[0] for p in pairs]
    pr = [p[1] for p in pairs]
    return {
        "n": len(pairs),
        "cer": jiwer.cer(gt, pr),
        "cer_nows": jiwer.cer([strip_ws(g) for g in gt], [strip_ws(p) for p in pr]),
        "wer": jiwer.wer(gt, pr),
        "wer_nows": jiwer.wer([strip_ws(g) for g in gt], [strip_ws(p) for p in pr]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default="training/local/lines_cache")
    ap.add_argument("--dump", default="")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--first-n", type=int, default=0,
                    help="evaluate only the first N lines in width-sorted order (matches in-training eval)")
    args = ap.parse_args()

    ckpt = torch.load(REPO / args.ckpt, map_location="cpu", weights_only=False)
    model = TorchVGSLModel(vgsl=ckpt["spec"], mode="recognition")
    net = model.nn
    net.load_state_dict(ckpt["state_dict"])
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    net.to(dev).eval()

    ds = CachedLineDS(REPO / args.cache / "eval")
    # decode with the TRAIN codec: the model's output layer was trained on the
    # train-split charset (the eval meta c2l is built from eval files only and
    # maps ids differently).
    train_meta = json.load(open(REPO / args.cache / "train" / "meta.json"))
    rev = {v: ch for ch, vs in train_meta["c2l"].items() for v in vs}
    n_classes = int(ckpt["spec"].rsplit("O1c", 1)[1].rstrip("]"))
    assert n_classes == max(rev.keys()) + 1, (n_classes, max(rev.keys()))
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
            out = net(stack.to(dev))
            lg = (out[0] if isinstance(out, tuple) else out).squeeze(2).permute(2, 0, 1)
            preds = lg.argmax(2)
            for j, i in enumerate(b):
                tlen = max(1, min(preds.shape[0], widths[j] // SUBSAMPLE))
                pred = ids_to_text(rev, preds[:tlen, j].tolist())
                name = pathlib.Path(ds.files[i]).stem
                gt = pathlib.Path(ds.files[i].replace(".png", ".gt.txt")).read_text(encoding="utf-8")
                pairs.append((name, gt, pred))

    if args.first_n:
        pairs = pairs[:args.first_n]
    m = metrics([(g, p) for _, g, p in pairs])
    m["ckpt"] = args.ckpt
    m["epoch"] = ckpt.get("epoch")
    print(json.dumps(m, indent=1))
    if args.dump:
        with open(REPO / args.dump, "w", encoding="utf-8") as f:
            for name, gt, pred in pairs:
                f.write(json.dumps({"name": name, "gt": gt, "pred": pred}, ensure_ascii=False) + "\n")
        print("dumped", args.dump)


if __name__ == "__main__":
    main()
