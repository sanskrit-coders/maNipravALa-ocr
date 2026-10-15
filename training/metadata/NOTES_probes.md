# Probe notes (how mapping was derived; one-off commands reproduced as scripts)

## Tesseract without sudo (Debian bookworm, no root)
- `libtesseract5` already installed, but no `tesseract` binary.
- `cd /tmp && apt-get download tesseract-ocr tesseract-ocr-tam tesseract-ocr-san tesseract-ocr-hin tesseract-ocr-eng tesseract-ocr-osd`
  (`apt download -o Dir=...` fails; `apt-get download` in `/tmp` works, no sudo needed).
- `dpkg -x tesseract-ocr_*.deb /tmp/tess-root` + same for lang debs.
- Binary: `/tmp/tess-root/usr/bin/tesseract`, data: `/tmp/tess-root/usr/share/tesseract-ocr/5/tessdata/`
  (`tam.traineddata san.traineddata hin eng osd`).
- Run: `TESSDATA_PREFIX=/tmp/tess-root/usr/share/tesseract-ocr/5/tessdata/ /tmp/tess-root/usr/bin/tesseract img.png stdout -l tam+san --psm 6`
- Baseline on `sara_p50.png` (printed 19, acharya list): Tamil OK, mixed words noisy
  (`श्रीपादத்தை` → `श्रीपाद्$@8@`, `नाथमुनि` Tamil → Devanagari confusion) but sufficient for fuzzy anchors.
- `scripts/baseline_ocr.py` wraps this (txt+hocr+tsv). TSV `mean_conf` currently -1 (level filtering bug — fix: average only level==5 word rows).

## PDF identity (critical correction)
- `01.pdf` (630 pp) and `02.pdf` (616 pp) are **Vol.1 (00–22)** and **Vol.2 (23–32)**, each with
  independent printed pagination from 1 (not two halves, not duplicates).
  Evidence: `02.pdf` TOC p10 lists II(23–26) III(27–29) IV(30–32) with printed 1–579;
  `02.pdf` idx35 → printed 4 (same front matter as Vol.1); `02.pdf` idx556 colophon ends 29th adhikara.
- `01.pdf` TOC pp (idx10–11): I: 00:1–32 … 10:303–312, then 11:313–350 … 22:565–591.
  Gaps (54,64,…) are blank/title/plate pages. Front matter ~30 pp with many unnumbered colour plates
  (acharyas), so `printed = pdf_index − offset` drifts 30→33; use footer digits for exact mapping.
- Last pages of both PDFs are blank (scan tail) — ignore.

## Core vs commentary split (user-confirmed + visual)
- Mixed pages (e.g. 01.pdf idx48/printed 17, idx60/printed 29): **bold top = core** (matches
  `mUlam/2-script` unit, e.g. `இவ்வாचार्यர்களில்; ईश्वरमुनि…` = core `00-0.9`), thin rule, then
  normal-weight commentary below (matches `maNipravALa-training-data` unit).
- Pure-core pages (e.g. idx32–33, mangala verses 2–11) and pure-commentary pages (e.g. idx42/printed 11)
  also occur (“core at top, when present”).
- Script mismatch is systematic, not content mismatch: MD `ईश्वरमुनिகள்` vs scan `ஈச்வரமுனிகள்`
  (Devanagari stem vs Tamil stem + same Tamil suffix). Verified: `grep` for `உருப்பட்டூர்` fails in
  comm MDs because that list page (printed 19) is **core** `00-0.11`
  (`नाथमुनिகள் श्रीपादத்தை…உய்யக்கொண்டார்…உருப்பட்டூர்…` — core MD has it, comm MD only comments on it).
  Earlier confusion resolved: core MD `00-0.11` len 1621 chars contains the full acharya list.

## Block detector status
- `scripts/detect_blocks.py` (OTSU + horizontal MORPH_OPEN `kw=w//6`) finds only fragments
  (e.g. p0036: 2 contours 642px+458px < 0.6w threshold → 0 lines) because scan lines are slightly
  skewed/curved and header title breaks continuity. Fix: threshold `cw > 0.3w` + merge collinear
  segments (Δy<6px), or HoughP, or use hOCR line heights/boldness instead. Pilot crops usable with
  fixed 10%/94% margins; mixed-page separator detection to be improved for full run.

## Alignment status
- `scripts/align.py` (Devanagari→Tamil norm + `rapidfuzz.token_set_ratio`) on 14 pilot pp:
  correct sequential hits for 00-section (p0039–40→00-0.2, p0041–42→00-0.3, p0043→00-0.4 core+comm,
  p0044→00-0.5) despite low absolute scores (50–67). Front-matter pp score <45/random — constrain
  candidates by `toc_printed_mapping.csv` adhikara in production.
- Verified gold: p0043 (idx43/printed ~13): top `“पापिष्ठः क्षत्रबन्धुश्च…”` = core `00-0.4`,
  body `இப்படி सर्वमुमुक्षु…` = comm `00-0.4` (baseline OCR vs MD match, see pilot CSV).
