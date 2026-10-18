# maNipravALa (mixed Devanagari-Tamil) OCR training — sArabodhinI

Parallel-text source for training an OCR that will be applied to
`ra-tra-sA_5-commentaries_{1,2,3,4}.pdf` (5 commentaries, same scripts).

## PDF layout (critical)

Each scan page has up to two parallel texts:

- **core (mUlam)** — at the top **when present**, often bold/centered
  (verses or bold prose), followed by a horizontal rule.
- **commentary (sArabodhinI)** — main body, normal weight, justified
  manipravala (Devanagari + Tamil mix, e.g. `श्रीपादத்தை`, `मतनिरसनरूपप्रयोजनத்தையும்`).

Header (top ~10%): archive.org URL + running head (e.g. `गुरुपरंपरसारः`) + double rules — ignore for GT.
Footer (bottom ~6%): printed folio + `CC-0 … Muthulakshmi Research Academy` — ignore for GT.

## Sources

- Scans (CC-0, Muthulakshmi Research Academy):
  - `/media/vvasuki/vData/text/granthasangrahaH/AgamaH/vaiShNavaH/shrIvaiShNavaH/venkaTanAtha/rts/sArabodhinI/01.pdf` (630 pp) = **Vol.1: 00–22**, printed 1–591
  - `.../sArabodhinI/02.pdf` (616 pp) = **Vol.2: 23–32**, printed 1–579 (independent pagination!)
  - The two PDFs are **not** two halves; they are Vol.1 and Vol.2 with separate printed pagination.
- Ground truth MDs (2-script, Devanagari+Tamil):
  - core only: `vishvAsa/.../rahasya-traya-sAraH/mUlam/2-script/*.md` (33 files, 1073 units `00-0.1 … 32-…`)
  - commentary only: `vishvAsa/.../meta/private/maNipravALa-training-data/sAra-bodhinI/*.md` (33 files, 1073 units, same IDs)
  - Both sets share unit IDs (`मूलम् - X.Y`), so core unit `00-0.4` and comm unit `00-0.4` comment on the same mula.

Script note: MDs write Sanskrit stems in Devanagari (`ईश्वरमुनिகள்`) where scans sometimes use Tamil (`ஈச்வரமுனிகள்`).
Do **not** use MD text verbatim as image GT without script check — see `scripts/align.py` normalization
(Devanagari runs → Tamil) and manual verification. Target 5-commentary PDFs use Devanagari convention
closer to the MDs (e.g. `(सा.दी.)`), so synthetic rendering from MDs is also viable.

## TOC → printed → adhikara (see `metadata/toc_printed_mapping.csv`)

Vol.1 (01.pdf): 00:1–32, 01:33–53, 02:55–63, 03:65–83, 04:85–114, 05:115–184, 06:185–229,
07:231–244, 08:245–274, 09:275–302, 10:303–312, 11:313–350, 12:351–380, 13:381–394,
14:395–408, 15:409–438, 16:439–457, 17:459–476, 18:477–511, 19:513–526, 20:527–548,
21:549–564, 22:565–591.
Vol.2 (02.pdf): 23:1–48, 24:49–111, 25:113–167, 26:169–191, 27:193–307, 28:309–378,
29:379–525, 30:527–542, 31:543–556, 32:557–579.
Gaps are blank/title/plate pages. Front matter (~30 pp, incl. colour plates of acharyas)
precedes printed 1 in both PDFs — use footer digits for exact `printed ↔ pdf_index` (offset drifts
after each unnumbered plate; approx `printed = pdf_index − 30…33`).

## Pipeline scripts (run in this order)

1. `scripts/parse_mds.py` → `texts/core/*.txt`, `texts/commentary/*.txt`,
   `metadata/units_{core,commentary}.jsonl` (1073+1073 units), `metadata/parse_stats.json`.
2. `scripts/render_pages.py --dpi 150 --subset01 31:45 --subset02 0:0`
   → `images/sarabodhini-01/pXXXX.jpg` (JPEG, shareable; 300 DPI for final training),
   `metadata/pages_*.csv` (`pdf_index,printed_page,image,w,h`).
3. `scripts/baseline_ocr.py --images images/sarabodhini-01 --glob "p00*.jpg" --out metadata/baseline_01 -l tam+san --psm 6`
   (uses local Tesseract from `/tmp/tess-root`, no sudo) → `*.txt,*.hocr,*.tsv` anchors.
4. `python3 -m doc_curation.image.checker.detect_blocks` (moved out of this repo's
   `scripts/`; legacy `--images/--out` mode retained)
   → `crops/*_{core,comm}.jpg` + `*_blocks.json` (header/separator heuristic; see
   `data/README.md` for the current split/snap logic and crop-edge audit).
5. `scripts/align.py --baseline metadata/baseline_01 --out metadata/align_pilot.csv`
   → fuzzy (rapidfuzz `token_set_ratio` on Devanagari→Tamil normalized text) mapping
   page → best core/comm unit + `align_pilot_details/*.json`. Constrain candidates by TOC
   adhikara for production (pilot shows sequential hits e.g. p0043→00-0.4 core+comm 62–63).
6. `scripts/make_gt.py --vol 01|02 [--write]` → per-crop GT text for all boxed crops
   (`data/RTS_sb_0N/gt/{core,comm}/pXXXX.txt`, byte-exact unit-body substrings) + report
   `metadata/gt_report_0N.csv`; dry-run by default. OCR cache `metadata/ocr_0N/`.
   Statuses `auto-approx`/`needs-review`; doubtful pairs are renamed `pXXXX_doubtful.{jpg,txt}`
   (best-guess text) and excluded from training by name; gold/verified entries preserved; idempotent.
   Done: Vol.1 1086 auto-approx + 28 needs-review (25 doubtful images, 21 texts); Vol.2 1117 + 15
   (11 doubtful images, 8 texts); see `data/README.md` "GT generation" for the flag vocabulary.

## Pilot status (14 pp, 01.pdf idx 31–44, printed 1–14 → adhikara 00)

- `images/sarabodhini-01/p0031–p0044.jpg` (150 DPI JPEG, 7 MB) + `metadata/pages_01.csv`.
- `metadata/baseline_01/*.txt` (Tesseract `tam+san`, PSM 6) — usable anchors despite mixed-word errors
  (e.g. `श्रीपादத்தை` → `श्रीपाद्$@8@`).
- `metadata/blocks_01/crops/` + `blocks_index.json` (header-line detector needs lower width threshold —
  currently 0/1 lines on most pp; crops still usable with fixed margins).
- `metadata/align_pilot.csv`: e.g. p0043→00-0.4 (core 62.4/comm 63.1, verified: top verse
  `पापिष्ठः क्षत्रबन्धुश्च…` + commentary `இப்படி सर्वमुमुक्षु…`), p0041–42→00-0.3, p0044→00-0.5.
  Early front-matter pp score low (<45, random adhikaras) — expected; restrict to 00 for Vol.1 pp 1–32.

## Common shareable format (for Tesseract/Kraken/TrOCR)

Per page (final): `images/.../pXXXX.jpg` + `metadata/training_index.jsonl` rows
`{image, pdf, pdf_index, printed_page, adhikara, core_units:[…], comm_units:[…],
core_box, comm_box, core_text, comm_text, status:auto|verified}`.
Convert:
- Tesseract LSTM: line crops + `.box`/`.lstmf` via `scripts/export_tesseract.py` (TODO: line segmentation
  from hOCR + GT line alignment by DP).
- Kraken: PAGE-XML via `scripts/export_kraken.py` (TODO).
- TrOCR/HF: `datasets` image-text pairs (line crops) via `scripts/export_hf.py` (TODO).
Line segmentation: use Tesseract TSV/hOCR word boxes grouped to lines, then DP-align to GT lines.

## Next: train/compare + apply to targets

Targets: `.../rts/5-comm/ra-tra-sA_5-commentaries_{1,2,3,4}.pdf` (656/754/754/806 pp) +
existing `*.pdf.txt` baselines (likely Drive/Vision OCR, 3–4.5 MB each — evaluate first).
Targets have 5 commentaries (`(सा.दी.) (सा.स्वा.) (सा.प्र.) (सा.वि.) (सा.सं.)`) + mula, denser layout
than Sarabodhini — expect domain shift; fine-tune on Sarabodhini then adapt with few-shot target pages.

Full render (1246 pp @200–300 DPI JPEG) ~1–2 GB + ~1 hr; run in chunks to avoid timeouts
(`--subset01 31:200` etc.), then full baseline + constrained align, manual-verify ~20–50 pp,
export all three formats, train and compare (CER/WER on held-out pp), apply best to 5-comm.
