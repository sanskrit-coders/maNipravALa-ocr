# data — canonical RTS volumes (images first, GT later)

- `data/RTS_sb_01/` = Vol.1 (01.pdf, 630 pp, adhikaras 00-22). `pages_raw/pXXXX.jpg` (150 DPI JPEG, pdf_index),
  `images/{core,comm}/pXXXX.jpg` (block crops via detect_blocks, 0.3-width merged lines),
  `metadata/{pages.csv,blocks.json}`, `gt/{core,comm}/` (page-part GT; gold for eval blocks, auto/needs-review otherwise).
- `data/RTS_sb_02/` = Vol.2 (02.pdf, 616 pp, adhikaras 23-32). Same layout; GT deferred (images only for now).
- Splits are path lists partitioning the above (no copies):
  `training/split_train.jsonl`, `evaluation/split_eval.jsonl` (+ `evaluation/eval_index.jsonl`),
  each row `{image, gt, page, block, unit, status}` with repo-relative `data/...` paths.
- Never train on eval paths. Full Vol.1/Vol.2 render+detect running in background
  (`data/RTS_sb_0*/metadata/full_run.log`); rerun idempotently:
  `python3 training/scripts/render_pages.py --vol 01 --dpi 150 --offset01 30`
  `python3 training/scripts/detect_blocks.py --vol 01` (same for 02 with --offset02 31).

## Header-fix note (first-line cut)
`detect_blocks.py` fallback `header_bottom=10%` cut the first text line on pages without
running headers (front matter; text starts ~7-8%). Fixed to URL-aware `4.5%+8px`
(`training/scripts/detect_blocks.py::split_page`); separator logic unchanged (0.3-width merged).
Verified: `RTS_sb_02/images/comm/p0029.jpg` now starts `ஸ்ரீமத் ரஹஸ்யத்ரயஸாரம்…`,
`p0030.jpg` starts `இப்படிப்பட்ட அத்புதமான…` (was cut). Full re-detect done:
Vol.1 630 pages_raw / 630 comm / 242 core (873 blocks), Vol.2 616 / 616 / 423 (1039 blocks).
Pilot gold seps stable (p0043 y=245, p0044 y=453); gold GT in splits still valid (crops superset).
