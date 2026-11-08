#!/usr/bin/env python3
"""Parallel Kaggle driver for word-split Drive OCR over the 4 RTS 5-commentary volumes.

Pushes 4 parallel Kaggle kernels (one per volume, 2 CPU + 2 GPU to maximize concurrent
slots) and downloads each volume's assembled {vol}.md and per-page md_v{vol}.tar.gz into:
  raw/word-split-drive/{1,2,3,4}.md
  raw/.ocr_state/word_split_v{1,2,3,4}/p*.md

Usage:
  python3 training/scripts/script_cls/ocr_5comm_kaggle.py [--push] [--monitor]
"""
import argparse
import json
import pathlib
import shutil
import subprocess
import tarfile
import time

KAGGLE = "/home/vvasuki/.local/bin/kaggle"
TARGET_RAW = pathlib.Path(
    "/home/vvasuki/gitland/vishvAsa/rAmAnujIyam/content/tattvam/"
    "venkaTa-nAtha-shAkhA/venkaTanAthaH/rahasya-traya-sAraH/5-tIkAH/raw"
)
OUT_DIR = TARGET_RAW / "word-split-drive"
STATE_DIR = TARGET_RAW / ".ocr_state"

KERNELS = {
    1: ("vishvasvasuki/rts-5comm-ocr-run", "rts-5comm-ocr-run", 656, 16, True),
    2: ("vishvasvasuki/rts-5comm-chunk", "rts-5comm-chunk", 754, 16, True),
    3: ("vishvasvasuki/rts-5comm-chunk", "rts-5comm-chunk", 754, 16, True),
    4: ("vishvasvasuki/rts-5comm-ocr-run", "rts-5comm-ocr-run", 806, 10, True),
}

WATCH_KERNELS = {
    1: ["vishvasvasuki/rts-5comm-ocr-run", "vishvasvasuki/rts-5comm-ocr"],
    2: ["vishvasvasuki/rts-5comm-chunk", "vishvasvasuki/rts-5comm-ocr-2"],
    3: ["vishvasvasuki/rts-5comm-chunk"],
    4: ["vishvasvasuki/rts-5comm-ocr-run"],
}

DATASETS = [
    "vishvasvasuki/manipravala-rts-5comm",
    "vishvasvasuki/rts-ocr-code",
    "vishvasvasuki/rts-ocr-secret",
]

KERNEL_TEMPLATE = r'''import os, sys, subprocess, pathlib, tarfile, shutil

IN = pathlib.Path('/kaggle/input')
for mod, imp in [('pymupdf', 'pymupdf'), ('google-api-python-client', 'googleapiclient'),
                 ('oauth2client', 'oauth2client'), ('google-auth-oauthlib', 'google_auth_oauthlib')]:
    try:
        __import__(imp)
    except Exception:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', mod], check=False)

src_code_dir = next(h.parent for h in IN.rglob('DejaVuSans.ttf'))
code_dir = pathlib.Path('/tmp/code')
if code_dir.exists():
    shutil.rmtree(code_dir)
shutil.copytree(src_code_dir, code_dir)

# Ensure split_page_sections runs in a 4-thread pool concurrently with ocr_drive
p_file = code_dir / 'pipeline.py'
p_txt = p_file.read_text(encoding='utf-8')
if 'meta_pool = ThreadPoolExecutor' not in p_txt:
    p_txt = p_txt.replace(
        '        ocr_drive([p for p, _ in faux_chunks])',
        '        try:\n'
        '            from detect_blocks import split_page_sections\n'
        '        except ImportError:\n'
        '            split_page_sections = None\n'
        '        meta_futures = {}\n'
        '        meta_pool = None\n'
        '        if args.md_dir and split_page_sections is not None:\n'
        '            from concurrent.futures import ThreadPoolExecutor\n'
        '            meta_pool = ThreadPoolExecutor(max_workers=4)\n'
        '            for pi, arr, _, _ in batch:\n'
        '                meta_futures[pi] = meta_pool.submit(split_page_sections, arr)\n'
        '        ocr_drive([p for p, _ in faux_chunks])',
        1
    ).replace(
        '                try:\n'
        '                    from detect_blocks import split_page_sections\n'
        '                except ImportError:\n'
        '                    split_page_sections = None\n'
        '                rule_ys = []\n'
        '                if split_page_sections is not None:\n'
        '                    try:\n'
        '                        meta = split_page_sections(arr)\n'
        '                        rule_ys = [L["yc"] for L in meta["rules"]\n'
        '                                   if L["yc"] > meta["header_bottom"] + 2]\n'
        '                    except Exception:\n'
        '                        rule_ys = []\n'
        '                hb = int(meta.get("header_bottom", 0)) if split_page_sections is not None else 0',
        '                rule_ys = []\n'
        '                meta = {}\n'
        '                if pi in meta_futures:\n'
        '                    try:\n'
        '                        meta = meta_futures[pi].result()\n'
        '                        rule_ys = [L["yc"] for L in meta["rules"]\n'
        '                                   if L["yc"] > meta["header_bottom"] + 2]\n'
        '                    except Exception:\n'
        '                        rule_ys = []\n'
        '                hb = int(meta.get("header_bottom", 0)) if meta else 0',
        1
    )
    p_file.write_text(p_txt, encoding='utf-8')

sys.path.insert(0, str(code_dir))
os.environ['PYTHONPATH'] = str(code_dir) + ':' + os.environ.get('PYTHONPATH', '')
os.environ['TESS_BIN'] = '/nonexistent/tesseract'
os.environ['DRIVE_PY'] = sys.executable
os.environ['DRIVE_HELPER'] = str(code_dir / 'drive_ocr_images.py')
os.environ['TAG_FONT'] = str(code_dir / 'DejaVuSans.ttf')
os.environ['DRIVE_WORKERS'] = '5'

key = next(IN.rglob('ocr_app_oauth.json'))
sec = pathlib.Path('/tmp/secret')
sec.mkdir(parents=True, exist_ok=True)
for n in ['ocr_app_oauth.json', 'tokens.json']:
    s = key.parent / n
    if s.exists():
        (sec / n).write_bytes(s.read_bytes())
os.environ['GOOGLE_KEY'] = str(sec / 'ocr_app_oauth.json')

pdf = next(IN.rglob('ra-tra-sA_5-commentaries_@VOL@.pdf'))
total = @TOTAL@
offset = @OFFSET@
vol = @VOL@

md_dir = pathlib.Path('/tmp/md')
out_dir = pathlib.Path('/tmp/out')
work_dir = pathlib.Path('/tmp/work')
md_dir.mkdir(parents=True, exist_ok=True)

def save_outputs():
    parts = [f'+++\ntitle = "{vol}"\n+++\n']
    non_empty = 0
    for pi in range(total):
        f = md_dir / f'p{pi:05d}.md'
        if not f.exists():
            continue
        txt = f.read_text(encoding='utf-8').strip()
        lines = [ln for ln in txt.splitlines() if ln.strip()]
        if len(lines) <= 1:
            continue
        parts.append(txt + '\n')
        non_empty += 1
    out_md = pathlib.Path(f'/kaggle/working/{vol}.md')
    tmp_md = pathlib.Path(f'/kaggle/working/{vol}.md.tmp')
    tmp_md.write_text('\n'.join(parts), encoding='utf-8')
    tmp_md.replace(out_md)
    tar_tmp = pathlib.Path(f'/kaggle/working/md_v{vol}.tar.gz.tmp')
    tar_out = pathlib.Path(f'/kaggle/working/md_v{vol}.tar.gz')
    with tarfile.open(tar_tmp, 'w:gz') as tf:
        for f in sorted(md_dir.glob('p*.md')):
            tf.add(f, arcname=f.name)
    tar_tmp.replace(tar_out)
    print(f'SAVED /kaggle/working/{vol}.md ({non_empty} non-empty pages)', flush=True)

chunk_size = 100
for start in range(0, total, chunk_size):
    end = min(total, start + chunk_size)
    cmd = [
        sys.executable, str(code_dir / 'pipeline.py'),
        '--pdf', str(pdf),
        '--pages', f'{start}:{end}',
        '--out', str(out_dir),
        '--workdir', str(work_dir),
        '--classifier', str(code_dir / 'script_cls.pt'),
        '--page-batch', '25',
        '--faux-batch', '40',
        '--chunk', '40',
        '--dpi', '300',
        '--md-dir', str(md_dir),
        '--page-offset', str(offset),
    ]
    print('RUN', ' '.join(cmd), flush=True)
    r = subprocess.run(cmd)
    if (work_dir / 'faux').exists():
        shutil.rmtree(work_dir / 'faux', ignore_errors=True)
    save_outputs()
    if r.returncode != 0:
        raise SystemExit(r.returncode)
'''


def push_volume(vol):
    kid, title, total, offset, use_gpu = KERNELS[vol]
    wdir = pathlib.Path(f"/tmp/kpush/v{vol}")
    wdir.mkdir(parents=True, exist_ok=True)
    code = (KERNEL_TEMPLATE
            .replace("@VOL@", str(vol))
            .replace("@TOTAL@", str(total))
            .replace("@OFFSET@", str(offset)))
    (wdir / "run_vol.py").write_text(code, encoding="utf-8")
    meta = {
        "id": kid,
        "title": title,
        "code_file": "run_vol.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": use_gpu,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": DATASETS,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (wdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = subprocess.run([KAGGLE, "kernels", "push", "-p", str(wdir)],
                       capture_output=True, text=True)
    print(f"Pushed vol {vol} ({kid}, gpu={use_gpu}): rc={r.returncode} {r.stdout.strip()}", flush=True)
    return r.returncode == 0


def download_volume(vol, kid=None):
    if kid is None:
        kid, _, _, _, _ = KERNELS[vol]
    dl_dir = pathlib.Path(f"/tmp/kdl/v{vol}")
    shutil.rmtree(dl_dir, ignore_errors=True)
    dl_dir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([KAGGLE, "kernels", "output", kid, "-p", str(dl_dir)],
                       capture_output=True, text=True)
    print(f"Download vol {vol} ({kid}): rc={r.returncode}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    md_file = dl_dir / f"{vol}.md"
    tar_file = dl_dir / f"md_v{vol}.tar.gz"
    if tar_file.exists():
        v_state = STATE_DIR / f"word_split_v{vol}"
        v_state.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_file, "r:gz") as tf:
            tf.extractall(v_state)
        n_pages = len(list(v_state.glob("p*.md")))
        print(f"Vol {vol}: extracted {n_pages} page md files into {v_state}", flush=True)
    if md_file.exists():
        dst = OUT_DIR / f"{vol}.md"
        shutil.copy2(md_file, dst)
        print(f"Vol {vol}: saved {dst} ({dst.stat().st_size:,} bytes)", flush=True)
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volumes", default="1,2,3,4")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--monitor", action="store_true")
    args = ap.parse_args()
    vols = [int(x) for x in args.volumes.split(",") if x.strip()]
    if args.push:
        for v in vols:
            push_volume(v)
            time.sleep(2)
    if args.monitor:
        done = set()
        while len(done) < len(vols):
            for v in vols:
                if v in done:
                    continue
                for kid in WATCH_KERNELS.get(v, [KERNELS[v][0]]):
                    q = subprocess.run([KAGGLE, "kernels", "status", kid],
                                       capture_output=True, text=True)
                    st = q.stdout.strip().splitlines()[-1] if q.stdout.strip() else ""
                    print(f"[{time.strftime('%H:%M:%S')}] v{v} ({kid}): {st}", flush=True)
                    if "COMPLETE" in st:
                        if download_volume(v, kid=kid):
                            done.add(v)
                            break
            if len(done) < len(vols):
                time.sleep(60)


if __name__ == "__main__":
    main()

