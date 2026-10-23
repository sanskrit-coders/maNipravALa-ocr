#!/usr/bin/env python3
"""Resumable Kaggle driver for linewiseScriptSepDriveOCR over the 5-commentary books.

Each Kaggle run handles one contiguous page chunk of one volume, writes one md
file per A4 page, and exits.  The driver downloads completed pages immediately,
stores them under the target raw/.ocr_state/ tree, and rebuilds the volume md
after every chunk.  A failed or quota-stopped run loses at most one chunk.

Usage:
  python3 training/scripts/script_cls/ocr_5comm_kaggle.py \
      --volumes 1 --chunk 50 [--max-chunks 1]
"""
import argparse
import json
import pathlib
import re
import shutil
import subprocess
import time

TARGET = pathlib.Path(
    "/home/vvasuki/gitland/vishvAsa/rAmAnujIyam/content/tattvam/"
    "venkaTa-nAtha-shAkhA/venkaTanAthaH/rahasya-traya-sAraH/5-tIkAH/raw")
STATE = TARGET / ".ocr_state"
KERNEL_ID = "vishvasvasuki/rts-5comm-ocr-run"
KERNEL_TITLE = "rts-5comm-ocr-run"
DATASETS = ["vishvasvasuki/manipravala-rts-5comm",
            "vishvasvasuki/rts-ocr-code",
            "vishvasvasuki/rts-ocr-secret"]
VOLUMES = {1: (656, 16), 2: (754, 16), 3: (754, 16), 4: (806, 10)}

KERNEL_TEMPLATE = r'''import os,sys,subprocess,pathlib
IN=pathlib.Path('/kaggle/input')
def find(name):
    hits=sorted(IN.rglob(name))
    if not hits: raise SystemExit('missing '+name)
    return hits[0]
for mod in ['pymupdf','googleapiclient','oauth2client','google_auth_oauthlib']:
    try: __import__(mod)
    except Exception: subprocess.run([sys.executable,'-m','pip','install','-q',mod],check=False)
hits=sorted(IN.rglob('pipeline.py'))
code_dir=str([h for h in hits if 'rts-ocr-code' in str(h)][0].parent) if any('rts-ocr-code' in str(h) for h in hits) else str(hits[0].parent)
sys.path.insert(0,code_dir)
os.environ['PYTHONPATH']=code_dir+':'+os.environ.get('PYTHONPATH','')
src=find('tam.traineddata').parent
td=pathlib.Path('/kaggle/working/tessdata'); (td/'configs').mkdir(parents=True,exist_ok=True)
for n in ['tam.traineddata','san.traineddata']:
    if not (td/n).exists(): (td/n).symlink_to(src/n)
eng=pathlib.Path('/usr/share/tesseract-ocr/5/tessdata/eng.traineddata')
if eng.exists() and not (td/'eng.traineddata').exists(): (td/'eng.traineddata').symlink_to(eng)
(td/'configs'/'tsv').write_text('tessedit_create_tsv 1\n')
os.environ['TESS_BIN']='/usr/bin/tesseract'
os.environ['TESSDATA']=str(td)
os.environ['TESS_LINE_LANG']='eng'
os.environ['DRIVE_PY']=sys.executable
os.environ['DRIVE_HELPER']=str(pathlib.Path(code_dir)/'drive_ocr_images.py')
key=find('ocr_app_oauth.json')
sec=pathlib.Path('/kaggle/working/secret'); sec.mkdir(parents=True,exist_ok=True)
for n in ['ocr_app_oauth.json','tokens.json']:
    s=key.parent/n
    if s.exists(): (sec/n).write_bytes(s.read_bytes())
os.environ['GOOGLE_KEY']=str(sec/'ocr_app_oauth.json')
pdf=find('ra-tra-sA_5-commentaries_@VOL@.pdf')
cmd=[sys.executable,str(pathlib.Path(code_dir)/'pipeline.py'),
 '--pdf',str(pdf),'--pages','@START@:@END@','--out','/kaggle/working/out',
 '--workdir','/kaggle/working/work','--classifier',str(pathlib.Path(code_dir)/'script_cls.pt'),
 '--linewise','--page-batch','10','--faux-batch','10','--chunk','40','--dpi','300',
 '--md-dir','/kaggle/working/md','--page-offset','@OFFSET@']
print('RUN',' '.join(cmd),flush=True)
r=subprocess.run(cmd)
print('return',r.returncode,flush=True)
if r.returncode: raise SystemExit(r.returncode)
'''


def kernel_metadata():
    return {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": "run_chunk.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": DATASETS,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def run_kaggle(vol, start, end, offset, work):
    work.mkdir(parents=True, exist_ok=True)
    code = (KERNEL_TEMPLATE.replace("@VOL@", str(vol))
            .replace("@START@", str(start)).replace("@END@", str(end))
            .replace("@OFFSET@", str(offset)))
    (work / "run_chunk.py").write_text(code, encoding="utf-8")
    (work / "kernel-metadata.json").write_text(json.dumps(kernel_metadata(), indent=2),
                                               encoding="utf-8")
    print(f"push v{vol} {start}:{end}", flush=True)
    p = subprocess.run(["kaggle", "kernels", "push"], cwd=work,
                       capture_output=True, text=True)
    print(p.stdout[-1000:], p.stderr[-1000:], flush=True)
    if p.returncode:
        return False
    while True:
        q = subprocess.run(["kaggle", "kernels", "status", KERNEL_ID],
                           capture_output=True, text=True)
        st = q.stdout.strip().splitlines()[-1] if q.stdout.strip() else ""
        print(st, flush=True)
        if "COMPLETE" in st or "ERROR" in st or "CANCEL" in st:
            break
        time.sleep(60)
    out = work / "out"
    subprocess.run(["kaggle", "kernels", "output", KERNEL_ID, "-o", "-p", str(out),
                    "--file-pattern", r"md/.*\.md$"], capture_output=True, text=True)
    got = 0
    src = out / "md"
    page_dir = STATE / f"v{vol}"
    page_dir.mkdir(parents=True, exist_ok=True)
    if src.exists():
        for f in sorted(src.glob("p*.md")):
            shutil.copy2(f, page_dir / f.name)
            got += 1
    print(f"downloaded {got} pages for v{vol} {start}:{end}", flush=True)
    return "COMPLETE" in st


def rebuild(vol):
    total, _ = VOLUMES[vol]
    page_dir = STATE / f"v{vol}"
    parts = []
    for pi in range(total):
        f = page_dir / f"p{pi:05d}.md"
        if not f.exists():
            break
        parts.append(f.read_text(encoding="utf-8").rstrip() + "\n")
    if not parts:
        return 0
    out = TARGET / f"v{vol}.md"
    tmp = out.with_suffix(".md.tmp")
    tmp.write_text("\n".join(parts), encoding="utf-8")
    tmp.replace(out)
    return len(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volumes", default="1,2,3,4")
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--max-chunks", type=int, default=0)
    args = ap.parse_args()
    TARGET.mkdir(parents=True, exist_ok=True)
    STATE.mkdir(parents=True, exist_ok=True)
    for vol in VOLUMES:
        f = TARGET / f"v{vol}.md"
        bak = STATE / f"pre_layout_v{vol}.md"
        if f.exists() and not bak.exists():
            shutil.copy2(f, bak)
    # seed front matter already OCR'd in the pilot
    pilot = pathlib.Path("/tmp/opencode/kaggle_rts_chunk/md")
    seed = STATE / "v1"
    seed.mkdir(parents=True, exist_ok=True)
    if pilot.exists():
        for f in sorted(pilot.glob("p0000*.md")):
            if not (seed / f.name).exists():
                shutil.copy2(f, seed / f.name)
    done_chunks = 0
    for vol in [int(x) for x in args.volumes.split(",") if x.strip()]:
        total, offset = VOLUMES[vol]
        page_dir = STATE / f"v{vol}"
        page_dir.mkdir(parents=True, exist_ok=True)
        while True:
            present = {int(f.stem[1:]) for f in page_dir.glob("p*.md")}
            start = next((i for i in range(total) if i not in present), None)
            if start is None:
                break
            end = min(total, start + args.chunk)
            print(f"v{vol}: next chunk {start}:{end}", flush=True)
            ok = run_kaggle(vol, start, end, offset, pathlib.Path("/tmp/opencode/rts_kernel"))
            n = rebuild(vol)
            print(f"v{vol}: contiguous {n}/{total}", flush=True)
            done_chunks += 1
            if args.max_chunks and done_chunks >= args.max_chunks:
                return
            if not ok:
                print("chunk failed; resume with the same command", flush=True)
                return


if __name__ == "__main__":
    main()
