#!/usr/bin/env python3
"""Verify that training/kaggle/kernel/arch.py exactly reproduces the kraken VGSL model
for our spec (same module names, same state_dict keys, same forward outputs)."""
import pathlib
import sys

import torch

BASE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))
sys.path.insert(0, str(BASE / "kaggle" / "kernel"))

from train_lines2 import make_spec  # noqa: E402
from kraken.lib.vgsl import TorchVGSLModel  # noqa: E402
from arch import LineNet  # noqa: E402
import importlib.util  # noqa: E402

spec_train = importlib.util.spec_from_file_location(
    'kernel_train', BASE / 'kaggle' / 'kernel' / 'train.py')
kernel_train = importlib.util.module_from_spec(spec_train)
spec_train.loader.exec_module(kernel_train)


def check(mine, name):
    spec = make_spec(64, 165)
    torch.manual_seed(1234)
    kraken_net = TorchVGSLModel(vgsl=spec, mode='recognition').nn
    torch.manual_seed(1234)
    mine_fresh = type(mine)(165)
    kd0 = kraken_net.state_dict()
    md0 = mine_fresh.state_dict()
    init_diff = max(float((kd0[k] - md0[k]).abs().max()) for k in kd0)
    print(f'{name}: init max abs diff vs kraken fresh init: {init_diff:.6f}')
    assert init_diff < 1e-6, init_diff
    kd, md = kraken_net.state_dict(), mine.state_dict()
    assert set(kd) == set(md), (set(kd) ^ set(md))
    for k in kd:
        assert kd[k].shape == md[k].shape, (k, kd[k].shape, md[k].shape)
    mine.load_state_dict(kd)
    kraken_net.eval()
    mine.eval()
    torch.manual_seed(0)
    x = torch.rand(3, 1, 64, 500)
    with torch.no_grad():
        yk = kraken_net(x)[0].detach()
        ym = mine(x)[0].detach()
    print(f'{name}: output shapes {tuple(yk.shape)} {tuple(ym.shape)}, max abs diff {float((yk - ym).abs().max())}')
    assert torch.allclose(yk, ym, atol=1e-5)

    torch.manual_seed(7)
    kraken_net.train()
    yk = kraken_net(x)[0].detach()
    torch.manual_seed(7)
    mine.train()
    ym = mine(x)[0].detach()
    print(f'{name}: train-mode max abs diff {float((yk - ym).abs().max())}')


def main():
    check(LineNet(165), 'arch.py')
    check(kernel_train.LineNet(165), 'kernel/train.py (inlined)')
    print('ARCH OK: identical modules, keys and outputs for both copies')


if __name__ == '__main__':
    main()
