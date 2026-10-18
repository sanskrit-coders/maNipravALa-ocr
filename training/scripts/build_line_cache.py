#!/usr/bin/env python3
"""Precompute line image tensors for training, avoiding the per-epoch dewarp cost.

Same pipeline as kraken ImageInputTransforms(1, H, 0, 1, (16,0), valid_norm=True),
but the LANCZOS resize is applied BEFORE the center-line dewarp (which is
superlinear in width when run at original resolution; up to ~1.7s for wide
lines). The pre-resize target height is 2*H because CenterNormalizer outputs
images at H with a fixed ~0.5x relative width compression, so a 2*H input
yields output width ~= H/height_orig * width_orig (i.e. the plain resize).

Writes:
  training/local/lines_cache/{split}/{i:06d}.pt   {'x': tensor(1,H,W) f32, 'y': ids, 'w', 'file'}
  training/local/lines_cache/{split}/meta.json    {'height', 'n', 'widths', 'files', 'c2l', 'failed'}

Usage:
  python3 training/scripts/build_line_cache.py --height 64 --procs 12
"""
import argparse
import json
import pathlib
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

BASE = pathlib.Path(__file__).resolve().parents[1]
REPO = BASE.parent if BASE.name == "training" else BASE

_G = {}


def _init(files, targets, height, cache_dir):
    import torch
    from kraken.lib.dataset.utils import ImageInputTransforms
    torch.set_num_threads(1)
    _G['files'] = files
    _G['targets'] = targets
    _G['height'] = height
    _G['out'] = pathlib.Path(cache_dir)
    _G['trans'] = ImageInputTransforms(1, height, 0, 1, (16, 0), valid_norm=True)


def _work(i):
    from PIL import Image
    import torch
    try:
        p = _G['files'][i]
        h2 = 2 * _G['height']
        im = Image.open(p).convert('L')
        s = h2 / im.height
        if abs(s - 1.0) > 0.01:
            im = im.resize((max(1, round(im.width * s)), h2), Image.LANCZOS)
        exp = round(im.width / 2) + 32
        x = _G['trans'](im)
        if x.shape[1] != _G['height']:
            raise ValueError(f'dewarp output height {x.shape[1]} != {_G["height"]}')
        torch.save({'x': x, 'y': _G['targets'][i], 'w': int(x.shape[2]), 'file': p},
                   _G['out'] / f'{i:06d}.pt')
        return i, int(x.shape[2]), exp, None
    except Exception as e:
        return i, -1, -1, repr(e)


def build_split(split, files_rel, height, cache_dir, procs):
    from kraken.lib.dataset.recognition import GroundTruthDataset
    from kraken.lib.util import parse_gt_path
    files = [str(REPO / f) for f in files_rel]
    g = GroundTruthDataset(normalization=None, whitespace_normalization=False, reorder=False)
    kept = []
    for f in files:
        try:
            g.add(line=parse_gt_path(f))
            kept.append(f)
        except Exception as e:
            print('skip', f, e)
    g.encode()
    targets = [t for _, t in g.training_set]
    assert len(targets) == len(kept), (len(targets), len(kept))
    c2l = {ch: vs for ch, vs in g.codec.c2l.items()}

    out = pathlib.Path(cache_dir) / split
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob('*.pt'):
        old.unlink()

    t0 = time.time()
    widths = [0] * len(kept)
    ratios = []
    fails = []
    with Pool(procs, initializer=_init, initargs=(kept, targets, height, str(out))) as pool:
        for k, (i, w, exp, err) in enumerate(pool.imap_unordered(_work, range(len(kept)), chunksize=8)):
            if err:
                fails.append((i, err))
            else:
                widths[i] = w
                if exp > 0:
                    ratios.append(w / exp)
            if (k + 1) % 2000 == 0:
                print(f'{split} {k+1}/{len(kept)} {time.time()-t0:.0f}s', flush=True)
    ratios.sort()
    rmed = ratios[len(ratios) // 2] if ratios else 0.0
    ylens = [int(t.shape[0]) for t in targets]
    meta = {'split': split, 'height': height, 'n': len(kept), 'widths': widths,
            'ylens': ylens,
            'files': [str(pathlib.Path(f).relative_to(REPO)) for f in kept],
            'c2l': c2l, 'failed': fails, 'w_ratio_median': rmed}
    json.dump(meta, open(out / 'meta.json', 'w'))
    print(f'{split}: cached {len(kept)-len(fails)}/{len(kept)} in {time.time()-t0:.0f}s '
          f'fails={len(fails)} W/expected med={rmed:.2f} {fails[:3]}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--height', type=int, default=64)
    ap.add_argument('--procs', type=int, default=12)
    ap.add_argument('--out', default='training/local/lines_cache')
    args = ap.parse_args()
    for split in ('train', 'eval'):
        files = [l.strip() for l in open(REPO / f'training/local/lines/{split}_files.txt')]
        build_split(split, files, args.height, str(REPO / args.out), args.procs)


if __name__ == '__main__':
    main()
