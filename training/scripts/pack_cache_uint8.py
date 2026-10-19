#!/usr/bin/env python3
"""Pack lines_cache float32 tensors into compact uint8 memmap files for upload to Kaggle.

Output in <out-dir>/:
  train.bin, eval.bin  - concatenated uint8 line images (H*W bytes each, row-major)
  index.json           - per-sample {off,h,w,ylen,file[,gt]} + c2l + height
"""
import argparse
import json
import pathlib
import time

import torch

BASE = pathlib.Path(__file__).resolve().parents[1]
REPO = BASE.parent if BASE.name == "training" else BASE


def pack_split(split, cache, out, index):
    meta = json.load(open(cache / split / "meta.json"))
    rev = {v: ch for ch, vs in meta["c2l"].items() for v in vs}
    entries = []
    t0 = time.time()
    off = 0
    with open(out / f"{split}.bin", "wb") as f:
        for i in range(meta["n"]):
            d = torch.load(cache / split / f"{i:06d}.pt")
            x = (d["x"][0] * 255).round().clamp(0, 255).to(torch.uint8)
            h, w = x.shape
            f.write(x.numpy().tobytes())
            e = {"off": off, "h": int(h), "w": int(w), "ylen": meta["ylens"][i],
                 "file": meta["files"][i], "ids": [int(t) for t in d["y"]]}
            if split == "eval":
                e["gt"] = "".join(rev.get(int(t), "") for t in d["y"])
            entries.append(e)
            off += h * w
            if i % 5000 == 0:
                print(f"{split} {i}/{meta['n']} ({time.time()-t0:.0f}s)", flush=True)
    index[split] = entries
    print(f"{split}: {len(entries)} samples, {off/1e6:.0f} MB ({time.time()-t0:.0f}s)", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(REPO / "training/local/lines_cache"))
    ap.add_argument("--out", default=str(REPO / "training/local/kaggle_pack"))
    args = ap.parse_args()
    cache = pathlib.Path(args.cache)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    index = {}
    meta = pack_split("train", cache, out, index)
    pack_split("eval", cache, out, index)
    index["height"] = meta["height"]
    index["c2l"] = meta["c2l"]
    json.dump(index, open(out / "index.json", "w"))
    print("wrote", out / "index.json")


if __name__ == "__main__":
    main()
