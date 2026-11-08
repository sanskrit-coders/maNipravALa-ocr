# Google Drive Document OCR (`drive_ocr_images.py`)

**Status:** **Active production recognizer** (used in Stage 7 of [`current-pipeline.md`](../current-pipeline.md)).

## Overview & Specification

| Property | Specification |
|---|---|
| **Wrapper script** | [`training/scripts/script_cls/drive_ocr_images.py`](../../training/scripts/script_cls/drive_ocr_images.py) |
| **Python environment** | `/home/vvasuki/gitland/python_venv/bin/python` (`doc_curation` + `google-api-python-client`, with self-contained `DriveClient` fallback for Kaggle kernels) |
| **API mechanism** | Uploads `.jpg`/`.pdf` faux pages in parallel (`ThreadPoolExecutor`, `--workers`, default `5`) to Google Drive as `application/vnd.google-apps.document` and exports `mime_type="text/plain"` (`.jpg.txt`, avoiding embedded base64 images and reducing `.txt` file size by 250×), then deletes the temporary remote Drive file |
| **Input** | Script-pure (`_dev_` or `_tam_`) faux-page images containing up to 40 tagged phrase/line strips (`#T01`..`#T40`), batch-packed across `--page-batch` PDF pages |
| **Output** | UTF-8 text file (`{faux_path}.txt`) with `#T01`..`#T40` markers and recognized Devanāgarī or Tamil text |

## Why Script-Pure Phrase-Strip Faux Pages Are Required

Controlled benchmarks on `RTS_sb_01` ([`evaluation/summary.md`](../../evaluation/summary.md)) and full-volume runs established three critical properties of Google Drive's document OCR engine:

1. **Unsplit mixed-script pages fail (Block CER 0.435):**
   - When fed raw Maṇipravāḷa pages containing inline switches between Devanāgarī and Tamil, Drive's line-level language model locks onto the dominant script of each line and transliterates or garbles the minority script into phonetic/Latin approximations, while reordering dense multi-column commentary lines.
2. **Isolated single-word crops fail (`4.drive.wordwise.md` post-mortem):**
   - Drive OCR is a line/document recognizer, not an isolated-word classifier. When fed faux pages of isolated 1-word or sub-word crops, it loses horizontal baseline and language-model context, dropping 76%–100% of short Sanskrit words or hallucinating Latin strings.
3. **Script-pure linewise phrase strips succeed (Block CER 0.112; 100% tag recovery on Vol. 1 pilot):**
   - When consecutive same-script runs on each line are grouped into multi-word horizontal strips ([`make_segments`](../../training/scripts/script_cls/pipeline.py#L611-L637)) and stacked onto separate Devanāgarī (`_dev_`) and Tamil (`_tam_`) faux pages, Drive OCR achieves **0.112 block CER** (3.9× lower error than unsplit Drive and 2.0×–2.1× lower than local Kraken/Tesseract models).

## Faux-Page Geometry Rules for Drive OCR ([`compose_faux`](../../training/scripts/script_cls/pipeline.py#L664-L709))

To prevent layout and script-detection failure modes inside Google Drive OCR, `compose_faux` enforces four geometric invariants:

1. **Internal gap compaction (`_compact_crop_whitespace`):**
   - Wide internal whitespace gaps (`> 36` px, such as running headers with left/right page numbers) are compacted to `22` px so Drive never splits a row into two vertical columns.
2. **Horizontal vs. vertical proximity (`ROW_PITCH = 136`, strip height `82` px, paste `x = 146`):**
   - The vertical white gap between rows (`54` px) is nearly double the horizontal gap between `#Txx` (`x ≈ 118`) and the strip (`x = 146`, gap `28` px = 1 space). This prevents Drive's layout detector from grouping `#T01..#T40` vertically into a detached left column while keeping a 1-space token boundary after `#Txx`.
3. **6-Consonant Script Anchors on narrow rows (`கஙசஞடண` / `कखगघङच`):**
   - On short 1-word or suffix rows (`im.width < 460` px, e.g. `மான,`, `என்று,`, `முமாம்.`), the 4 ASCII characters `#Txx` would otherwise equal or outnumber the Indic glyphs on the row, causing Drive's line-level script detector to pick Latin/ASCII (`என்று,` → `6T601M,`, `மான,` → `LD IT 60T,`, `முமாம்.` → `(LOLOLD.`).
   - Appending `கஙசஞடண` (on `_tam_` pages) or `कखगघङच` (on `_dev_` pages) `36` px to the right of narrow crops guarantees that Indic glyphs outnumber ASCII `#Txx` by >= 2.5×, locking Drive's line recognizer into the target script. The anchor is stripped by `ANCHOR_STRIP_RE` in [`parse_tagged`](../../training/scripts/script_cls/pipeline.py#L734-L804).
