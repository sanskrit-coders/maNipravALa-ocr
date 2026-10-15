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
