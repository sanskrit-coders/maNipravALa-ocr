# maNipravALa-ocr Development History

This document records the chronological history of approaches, datasets, models, pipeline experiments, errors made, and results obtained in `maNipravALa-ocr`, ordered in **reverse chronological order** (most recent first).

Per-component technical details live in [`summary/`](summary/overall.md):
- [summary/overall.md](summary/overall.md) — Current system overview, I/O summary, and active roadmap.
- [summary/current-pipeline.md](summary/current-pipeline.md) — Stage-by-stage specification of the current production Manipravalam OCR pipeline.
- **Active models ([`summary/models/`](summary/models/)):**
  - [summary/models/patch-script-classifier.md](summary/models/patch-script-classifier.md) — Base 48×48 patch script classifier (`script_cls.pt`, **active**).
  - [summary/models/google-drive-ocr.md](summary/models/google-drive-ocr.md) — Google Drive Document OCR recognizer (`drive_ocr_images.py`, **active**).
- **Obsolete models ([`summary/models/obsolete/`](summary/models/obsolete/)):**
  - [summary/models/obsolete/script-boundary-classifier.md](summary/models/obsolete/script-boundary-classifier.md) — Mixed-word boundary segmenter (Shiro-rekhā rule, projection-glyph DP, and boundary CNNs).
  - [summary/models/obsolete/word-classifier.md](summary/models/obsolete/word-classifier.md) — 4-class word-level script classifier (`script_sep.pt`, `word_cls.pt`).
  - [summary/models/obsolete/lines_to_words.md](summary/models/obsolete/lines_to_words.md) — Projection-valley and TSV-constrained word segmentation.
  - [summary/models/obsolete/column-separator.md](summary/models/obsolete/column-separator.md) — 256-column script classifier (`column_sep.pt`).
  - [summary/models/obsolete/line-classifier.md](summary/models/obsolete/line-classifier.md) — Whole-line dominant-script classifier (`line_cls`).
  - [summary/models/obsolete/kraken-word-ocr.md](summary/models/obsolete/kraken-word-ocr.md) — Kraken 7.1.1 VGSL word-level dual-script OCR (`best_0.8900.safetensors`).
  - [summary/models/obsolete/dev-tam-chars-ocr.md](summary/models/obsolete/dev-tam-chars-ocr.md) — Kraken VGSL whole-line mixed-script OCR (`devTamCharsOCR` / `kaggle_v10_best.pt`).
  - [summary/models/obsolete/tesseract-paddle-baselines.md](summary/models/obsolete/tesseract-paddle-baselines.md) — Tesseract 5 and PaddleOCR PP-OCRv4 baselines.
- [evaluation/summary.md](evaluation/summary.md) & [evaluation/COMPARE.md](evaluation/COMPARE.md) — Controlled line- and block-level benchmarks on `RTS_sb_01`/`02`.
- [data/README.md](data/README.md) — Canonical volumes (`RTS_sb_01`, `RTS_sb_02`), block detection, GT alignment, and `word-image-maps`.

---

## 1. 2026-11-07 — Phase 1 (Vol. 4 Deterministic Cleanup & Promotion) and Phase 2 (Vol. 1 Intra-Word Manipravalam Pipeline Fixes & Pilot)

- **Docs:** [summary/overall.md](summary/overall.md), [summary/patch-script-classifier.md](summary/patch-script-classifier.md), [evaluation/pilot_v1/](evaluation/pilot_v1/).
- **Phase 1 — Volume 4 Cleanup & Promotion:**
  - Removed contaminated/failed Volume 4 outputs (`4.drive.corrected.md`, `4.drive.corrected.raw.md`, `4.drive.corrected2.md`, `4.drive.wordwise.md`) and moved `4.drive.md`, `4.drive.raw.md`, and `4.drive.report.md` out of gitignored `raw/drive/local/` into tracked `raw/drive/`.
  - Applied a deterministic cleanup pass (`tag_fixes=7`, `rep_fixes=404`, `verse_fixes=741` via `plain-drive/4.md` cross-reference for dropped `०` in `।। १०२ ।।`, `lat_fixes=161`), verified all 10 audit pages (`5, 100, 109, 141, 158, 380, 408, 437, 534, 700`), archived the old Kraken output to `/tmp/opencode/4.kraken-wordwise.md`, and promoted the cleaned `4.drive.md` (24,209 lines, 1,163,914 chars) to `raw/tesseract/4.md`.
- **Phase 2 — Root-Cause Fixes & Volume 1 Pilot (`p0029, p0032, p0035, p0055` = Printed Pages `14, 17, 20, 40`):**
  - **Root causes identified and fixed in `training/scripts/apply_ocr_pdf.py` and `training/scripts/script_cls/pipeline.py`:**
    1. **Line segmentation (`segment_lines_projection` in `apply_ocr_pdf.py`):** Replaced connected-component `find_objects` (which merged 2–3 lines via touching Tamil descenders/Devanagari matras) with horizontal projection-profile segmentation + pitch-based recursive multi-line valley splitting (`unit_h` + `pitch`) and non-overlapping valley clamping. Separates all `35, 34, 35, 32` printed lines on `p0029, p0032, p0035, p0055` with zero merged lines.
    2. **Why `script_sep.pt` was retired (`train_script_sep.py` diagnosis):** `load_real()` in `train_script_sep.py` labeled *all* real scan crops from `p0036.jsonl` as `CLS['mixed']` while `dev/tam/other` were purely synthetic font renders, training the CNN to classify scan paper texture as `mixed`. Replaced with the retrained 48×48 patch classifier `training/local/script_cls.pt` (0.9921 real-page validation accuracy).
    3. **Intra-word splitting & short-run merging (`classify_word`, `_merge_short_runs`, `_safe_split_x`):** Added center-patch sampling for narrow words (`w <= 40`), mean-probability argmax fallback when all patches are `<= 0.45`, minimum-ink (`<= 6` dark px) column splitting in `_safe_split_x` to cut across 1–3 px Shiro-rekhā overhangs (e.g. `निवर्तनத்தில்`), and merging of `"other"` and narrow runs into adjacent `dev`/`tam` runs.
    4. **Line-level `pre_space` and punctuation absorption (`_is_punct_or_other`):** Set `pre_space = not (prev_script == "dev" and script == "tam" and (rx0 - prev_rx1) <= 5)` for multi-run boxes so true `dev→tam` Manipravalam words join seamlessly without spaces (`अनुवदिத்தாரிறே?`, `समाश्रयिக்கையில்`, `पुरुषार्थलाभத்தில்`, `स्वरूपप्राप्तமான`, `निरुपाधिकமாக`) while inter-word transitions preserve spaces. Absorbed `"other"` and low-ink punctuation runs (`ink_rows < 18 or ink_area < 110`, such as `...`, `-`, `।`, quotes) into adjacent `dev`/`tam` runs.
    5. **Faux-page geometry & script anchors (`_compact_crop_whitespace`, `compose_faux`, `parse_tagged`):** Compacted wide internal crop gaps (`> 36` px → `22` px) to prevent Drive 2-column reordering; used `ROW_PITCH = 136` (strip height `82` px, `54` px vertical gap) with strips pasted at `x = 146` (`28` px = 1-space gap after `#Txx`); and rendered a 6-consonant script anchor (`கஙசஞடண` on `_tam_` pages, `कखगघङच` on `_dev_` pages) to the right of narrow rows (`im.width < 460`) so Drive's line-level script detector never misclassifies short 3–4 letter Tamil suffixes (`மான,`, `என்று,`, `முமாம்.`) as Latin/ASCII (`LD IT 60T,`, `6T601M,`, `(LOLOLD.`). Stripped the anchor in `parse_tagged` and removed `\b` from `TAG_TOKEN_RE` (since Python `re` treats Indic base letters as `\w`).
  - **Volume 1 Pilot Results (`evaluation/pilot_v1/p00029.md .. p00055.md`):**
    - **First-pass Drive OCR tag recovery:** **568 / 568 segments (100.00%)** across 16 faux pages (`0` retry pages needed).
    - **Comparison against previous Volume 1 outputs:**
      | PDF Page (Printed) | Metric | Old `linewise-script-split-drive-ocr/v1.txt` | Old `script-split-tesseract/v1.md` | **New Phase 2 Pipeline (`evaluation/pilot_v1/`)** |
      |---|---|---|---|---|
      | `p0029` (`[[P14]]`) | Lines / Chars / Mixed Words / Latin | 12 / 404 / 0 / 2 | 11 / 399 / 0 / 0 | **35 / 2,632 / 23 / 1** |
      | `p0032` (`[[P17]]`) | Lines / Chars / Mixed Words / Latin | 38 / 2,931 / 0 / 26 | 32 / 2,449 / 1 / 0 | **34 / 2,604 / 22 / 1** |
      | `p0035` (`[[P20]]`) | Lines / Chars / Mixed Words / Latin | 36 / 2,723 / 0 / 34 | 34 / 2,429 / 0 / 0 | **35 / 2,586 / 28 / 0** |
      | `p0055` (`[[P40]]`) | Lines / Chars / Mixed Words / Latin | 31 / 2,814 / 0 / 12 | 30 / 2,496 / 0 / 0 | **32 / 2,715 / 37 / 2** |
      | **Total (4 pages)** | **Lines / Chars / Mixed Words / Latin** | **117 / 8,872 / 0 / 74** | **107 / 7,773 / 1 / 0** | **136 / 10,537 / 110 / 4** |

---

## 2. 2026-11-07 (Earlier) — Comprehensive Audit & Post-Mortem of Volume 4 and Volume 1 Runs

- **What was done:**
  - Audited all four full-volume OCR outputs for `ra-tra-sA_5-commentaries_4.pdf` (806 PDF pp., 791 printed pp.) and pilot outputs for `ra-tra-sA_5-commentaries_1.pdf` directly against high-resolution PDF page scans (`5, 100, 109, 141, 158, 380, 408, 437, 534, 700` on Vol. 4; `14, 17, 20, 40` on Vol. 1) and faux-page renders.
- **Key findings & errors uncovered:**
  1. **Misunderstanding Volume 4's content (Anubandham vs. Manipravalam):**
     - Volume 4 (`ra-tra-sA_5-commentaries_4.pdf`) is the *Anubandham* (Appendix) and contains **zero classical intra-word Manipravalam commentary**. It comprises: (a) pure modern Tamil chapter summaries (printed pp. 1–34), (b) Devanagari Sanskrit verses followed by hyphen-separated Tamil *padavurai* (`अयं देहभृत्- இந்த ஜீவன்`) and Tamil translations (pp. 35–407 & 534–780), (c) **100% pure Devanagari Sanskrit commentary** (`अधिकारसङ्ग्रहः - सव्याख्यानः` and `कारिकादर्पणम्`, pp. 408–533), and (d) pure Tamil Divya Prabandham pasurams (pp. 781–794).
     - True intra-word Manipravalam (`श्रियःபதியாய்`, `ब्रह्माவுக்கு`, `आचार्यसम्बन्धத்தினுடைய`) lives in **Volumes 1–3** (`ra-tra-sA_5-commentaries_{1,2,3}.pdf`) and `RTS_sb_01`/`02`.
  2. **The Page 408 `(द्वया-)` → `(द्रयाணன்` false regression target:**
     - On printed page 408 (PDF index 417), lines 4, 10, and 21 contain the pure Sanskrit heading `(द्वया-)`. There is no Tamil anywhere in the page body.
     - Earlier `script-split-tesseract/v4.md` split `(द्वया-)` before `-)`, routing `(द्वया` to Sanskrit Tesseract (misread as `(द्रया`) and `-)` to Tamil Tesseract, which hallucinated Tamil `ணன்`, `னை`, `ண்`—creating the chimera tokens `(द्रयाணன்`, `(द्रया னை`, `(द्रया ண்`.
     - Subsequent sessions treated those Tesseract hallucinations as "ground-truth mixed words lost by Drive OCR," when `4.drive.md` had actually read them **correctly** as `द्वया-` / `(द्वया-`.
  3. **Why `4.drive.md` succeeded on Volume 4 while `4.drive.wordwise.md`, `4.drive.corrected2.md`, and `tesseract/4.md` failed:**
     - `4.drive.md` (linewise phrase strips from retrained `script_cls.pt`, 1,569 faux pp.) preserved >99.8% of Volume 4 cleanly.
     - `4.drive.wordwise.md` (isolated word/sub-word crops, 2,390 faux pp.) failed because Google Drive OCR loses language/baseline context on isolated word fragments, dropping 76%–100% of Devanagari tokens on mixed pages.
     - `4.drive.corrected2.md` contaminated `4.drive.md` by overwriting its mixed lines with the broken `wordwise` output.
     - `tesseract/4.md` (Kraken wordwise VGSL) had ~10% CER (~100,000 char errors) and 1,456 intra-word script-confusion tokens.
  4. **The `"other"`-class drop bug in `pipeline.py`:**
     - `pipeline.py` only composed `script in ("dev", "tam")` and dropped `"other"` runs, deleting middle `०` digits in 3-digit Devanagari verse numbers (`।। १ २ ।।` for `।। १०२ ।।`) and closing parentheses.

---

## 2. 2026-10-26 to 2026-10-27 — Volume 4 Wordwise Shiro-Rekhā Faux Pages & Post-Processing (`4.drive.wordwise.md`, `4.drive.corrected2.md`)

- **Docs:** [summary/obsolete/script-boundary-classifier.md](summary/obsolete/script-boundary-classifier.md), [summary/obsolete/word-classifier.md](summary/obsolete/word-classifier.md), [summary/obsolete/lines_to_words.md](summary/obsolete/lines_to_words.md).
- **What was tried:**
  - Built `/media/vvasuki/vData/.../rts/5-comm/faux-pages-wordwise-shiro-rekhA/ra-tra-sA_5-commentaries_4_faux.pdf` (2,390 faux pages at 150 dpi, Oct 26 14:44) using `line separator → word_segments → script_sep.pt → Shiro-rekhā split → wordwise faux pages → Drive OCR`, producing `raw/drive/local/4.drive.wordwise.md` (Oct 27 10:28).
  - Ran post-processing scripts to produce `4.drive.corrected.md` (Oct 27 09:32) and `4.drive.corrected2.md` (Oct 27 09:59).
- **Errors made:**
  1. **Emitting isolated word and sub-word crops to Google Drive OCR:** Drive OCR is a document/line recognizer that requires multi-word horizontal context. Fed isolated 1–3 akshara crops and sliced visargas/matras (`ीः-`, `ा-`, `ति-`), Drive either returned empty `\` or hallucinated Latin strings (`_qq4Hq sed: 47:`, `3அÀ 3544-`, `tafuqGiiH`, `dysfரிாரிறீர்`).
  2. **Using stale `script_sep.pt` instead of retrained `script_cls.pt`:** `script_sep.pt` had not been retrained on the verified `word-image-maps`; it misclassified pure Tamil words (`உடையவர்களுடை`, `தீர்மானங்கள`) onto Devanagari faux pages and triggered false Shiro-rekhā cuts on pure words.
  3. **Contaminating `4.drive.md` in `4.drive.corrected2.md`:** While fixing verse numbers (`।। १०२ ।।`) and deduplicating line endings, the script substituted mixed *padavurai* lines with the corrupted `wordwise` cache.
- **Results:**
  - On mixed pages (pp. 100, 109, 141, 534, 700), `4.drive.wordwise.md` and `4.drive.corrected2.md` lost **76%–100% of Devanagari text** and had **0 mixed-script words**. Retired and removed.

---

## 3. 2026-10-26 to 2026-10-27 — Retrained Patch Classifier (`script_cls.pt`) & Volume 4 Linewise Drive OCR (`4.drive.md`)

- **Docs:** [summary/patch-script-classifier.md](summary/patch-script-classifier.md), [summary/overall.md](summary/overall.md), [evaluation/summary.md](evaluation/summary.md).
- **What was tried:**
  - Retrained the 48×48 patch residual CNN (`training/local/script_cls.pt`) on **69,366** real word crops from the training-split `word-image-maps` and evaluated on **1,763** held-out eval words (`training/local/script_cls_metrics.json`).
  - Built `/media/vvasuki/vData/.../rts/5-comm/faux-pages-linewise-patch/ra-tra-sA_5-commentaries_4_faux.pdf` (1,569 faux pages, Oct 26 07:57) using `pipeline.py --linewise` (packing up to 40 coherent same-script phrase/line strips per faux page).
  - Sent all 1,569 faux pages to Google Drive OCR in 79 batches of 20 pages, producing `4.drive.raw.md` and collating `4.drive.md` (Oct 27 08:13).
- **Results:**
  - Patch validation accuracy on real word-image maps reached **0.9921** (`dev` 0.989, `tam` 0.995, `other` 0.966).
  - `4.drive.md` reconstructed **24,209 lines** and **1,165,663 chars** across 791 printed pages with **>99.8% completeness** and virtually zero Latin garbage (only 30 ≥3-char Latin words across 3.18 MB, all genuine English front-matter/printer stamps).
  - All 23 empty faux pages (`1,553 / 1,569` parsed) were verified to be 1-unit margin specks (`n_units = 1`).
- **Residual errors in `4.drive.md`:**
  - Dropping `"other"` segments in `pipeline.py` deleted middle `०` digits in some 3-digit Devanagari verse numbers (`।। १ २ ।।` for `।। १०२ ।।`).
  - 6 stray `#T` tags leaked into text, and bounding-box overlaps caused occasional line-end phrase repetitions.
  - Output was written to `raw/drive/local/`, which matched `*local*` in `~/.gitignore_global` and remained untracked.

---

## 4. 2026-10-25 — Word-Image Maps Dataset & Kraken 7.1.1 Word-Level OCR (`tesseract/4.md`)

- **Docs:** [data/README.md](data/README.md#L18-L32), [summary/overall.md](summary/overall.md), `training/local/kraken_words_metrics.json`.
- **What was tried:**
  - Built **95,411 high-confidence word-image maps** across **2,203 pages** of `RTS_sb_01` and `RTS_sb_02` (`data/RTS_sb_*/word-image-maps/{core,comm}/pXXXX.json`) by aligning page GT to Tesseract TSV candidate boxes via rendered-word/image similarity DP (`confidence ≥ 0.45`).
  - Trained a Kraken 7.1.1 VGSL word-level recognizer (`training/local/kraken_words_model/best_0.8900.safetensors`, 4.1M params, Kaggle `vishvasvasuki/rts-word-ocr-train`) on **92,976** word crops; evaluated on **2,435** held-out word crops.
  - Applied the Kraken word model across all 806 pages of Volume 4 via `training/scripts/apply_ocr_pdf.py` to produce `raw/tesseract/4.md` (1,141,993 chars).
- **Results:**
  - Held-out word-crop evaluation: character accuracy **90.34%**, word accuracy **80.53%** (Tamil **94.31%**, Devanagari **88.81%**; 2,167 char errors: 306 insertions, 1,096 deletions, 765 substitutions).
- **Errors / why it failed in production (`tesseract/4.md`):**
  - A 90.34% character accuracy (80.53% word accuracy) translates to **~100,000 character errors** across Volume 4 (1 in 5 words wrong).
  - Because a single 1D-CTC head decodes both Devanagari and Tamil alphabets without a language model on isolated word crops, visually similar strokes across scripts caused **1,456 spurious intra-word mixed tokens** (`एா`, `எः`, `५ெ`, `तர்`) and mangled Sanskrit conjuncts (`चरणाविति निर्देशः पत्नीवैशिष्ट्यबाधकः` → `च्रणाविति निदेपरः मत्ीै्रिष्ट्यवाधकः`).

---

## 5. 2026-10-23 to 2026-10-24 — Fixed Word Boxes, Shiro-Rekhā Rule, Boundary CNNs, & Volume 1 Linewise Faux Build

- **Docs:** [summary/obsolete/lines_to_words.md](summary/obsolete/lines_to_words.md), [summary/obsolete/script-boundary-classifier.md](summary/obsolete/script-boundary-classifier.md), [summary/obsolete/word-classifier.md](summary/obsolete/word-classifier.md), [summary/obsolete/column-separator.md](summary/obsolete/column-separator.md).
- **What was tried:**
  - Rebuilt word boxes using Tesseract TSV candidate gaps + word-width DP: **112,948 boxes** (`train_words_fixed.jsonl` 58,403 `sb_02` + `train_words_fixed_sb01.jsonl` 54,545 `sb_01`) and **14,208** high-confidence `dev→tam` boundary labels (`train_boundary_fixed*.jsonl`).
  - Swept projection-valley minimum gaps against TSV boundaries: found **7 px at 300 dpi** optimal (**F1 0.745**, P 0.809, R 0.690).
  - Evaluated 6 mixed-word boundary methods on the 14,208 `dev→tam` words:
    1. **Projection-glyph DP rule:** MAE **0.0320**, p@5 **0.718**, segment P **0.964** / R **0.959** (**best**).
    2. **Shiro-rekhā headline rule:** MAE **0.0320**, p@5 **0.718**, segment P **0.960** / R **0.956**.
    3. **Combined 2-channel (image + base-projection) CNN (`boundary_combined.pt`):** MAE **0.1551**, p@5 **0.456**, segment P **0.891** / R **0.862**.
    4. **Image-only CNN (`boundary_image.pt`):** MAE **0.1777**, p@5 **0.412**.
    5. **Column CNN (`column_sep.pt`):** 256-column `dev/tam/other` classifier; real column accuracy **0.715**, boundary MAE **0.087**.
    6. **Rendered Noto CNN (`boundary_sep.pt`):** MAE **0.0002** on synthetic Noto renders, but did not transfer to book scans.
  - Built `/media/vvasuki/vData/.../rts/5-comm/faux-pages-linewise-fixed/ra-tra-sA_5-commentaries_1_faux.pdf` (2,407 faux pp., 300 dpi, Oct 24) using `pipeline.py --linewise --word-classifier training/local/word_cls.pt`.
- **Errors & limitations discovered:**
  - **Line-level GT assignment bugs in `train_words_fixed.jsonl`:** ~5,301 boxes still contained merged tokens or wrong-line assignments (e.g. `p0538` core line 4 mapped GT `कुरु’’` to a Tamil image crop). Retraining `script_sep.pt` on `train_words_fixed.jsonl` collapsed (`dev` 0.676, `tam` 0.777, `mixed` 0.151).
  - **Tamil-first mixed words (~4%):** Found 1,645 Tamil-first mixed words (`tam-dev-tam` like `இவ்வधिकारार्थம்`, `tam-dev` like `இவ்வर्थ`) out of 41,504 mixed tokens in `RTS_sb_02`, which a single `dev→tam` boundary head cannot split without recursion.
  - **Synthetic `word_cls.pt` failure on Volume 1 scans:** Because `faux-pages-linewise-fixed` (Oct 24) used the synthetic `word_cls.pt` before `script_cls.pt` was retrained on real `word-image-maps` (Oct 26), real Tamil words on Vol. 1 p. 20 (`ஸர்வருக்கும் மோக்ஷ`, `காரணமென்று`) were misrouted onto the Devanagari faux page and vice versa.

---

## 6. 2026-10-18 to 2026-10-23 — Block Detection, GT Alignment, From-Scratch Line OCR (`devTamCharsOCR`), & Controlled Benchmarks

- **Docs:** [evaluation/summary.md](evaluation/summary.md), [evaluation/COMPARE.md](evaluation/COMPARE.md), [data/README.md](data/README.md), [summary/obsolete/line-classifier.md](summary/obsolete/line-classifier.md).
- **What was tried:**
  - Deskewed and segmented `RTS_sb_01` (630 pp → 525 core + 599 comm blocks) and `RTS_sb_02` (616 pp → 554 core + 586 comm blocks) via `doc_curation.image.checker.detect_blocks`, with Hough wavy-rule fallback and crop-edge audits (`crop_errors.json`).
  - Aligned block crops to digital edition units via `training/scripts/make_gt.py` (Vol. 1: 1,086 auto-approx GT; Vol. 2: 1,117 auto-approx GT).
  - Benchmarked 6 OCR engines/pipelines across **445 feasible held-out lines** and **5 hand-verified `RTS_sb_01` blocks** (89 lines, 6,193 non-space GT chars):
    1. **PaddleOCR PP-OCRv4 (`ta`/`hi`, Indic-OCR repo):** Line CER **0.710–0.732** (even with oracle script routing). Pretrained Hindi/Tamil models rendered the minority script as Latin garbage; Indic-OCR repo had no trained weights and stub fine-tuning.
    2. **Tesseract 5 (`tam+san`, PSM 7):** Line CER **0.476**, whole-line block CER **0.352**. Local script-split (`san`/`tam` on script segments) improved block CER to **0.227** (confirming script-splitting helps local engines too, though 2× worse than Drive).
    3. **devTamCharsOCR (`kaggle_v10_best.pt`, `kaggle_v13_best.pt`):** Kraken VGSL line architecture trained from scratch on 16.5k lines (40 epochs, fp32, no augmentation). Best whole-line recognizer: line CER **0.292**, block CER **0.238** (and **0.285** on mixed lines). However, when fed isolated short script runs/segments, it collapsed (CER **0.812** on runs, **0.599** on segments, emitting empty strings on 27% of runs) because 1–4 akshara fragments are outside its full-line training distribution.
    4. **Whole-block Google Drive OCR (unsplit):** Block CER **0.435** (reordered/dropped lines on dense commentary and garbled minority-script headings into Latin/phonetics).
    5. **`scriptSepDriveOCR` & `linewiseScriptSepDriveOCR` (`pipeline.py --linewise`):** Block CER **0.114** and **0.112** (**best overall** — 3.9× better than unsplit Drive, 2.1× better than `devTamCharsOCR`, 2.0× better than script-split Tesseract).
    6. **Hybrid combination (`combine_group.py`):** Replacing empty or Latin-containing Drive segments with `devTamCharsOCR` hurt or gave no gain (CER **0.116–0.131** vs **0.114** baseline).
- **Hard-won training lessons (`devTamCharsOCR`):**
  - Kraken's weight initialization (`uniform ±0.1` conv, orthogonal LSTM + forget-bias 1.0, Xavier linear) was mandatory (default PyTorch init stalled at 0.85–0.88 CER); AMP fp16 NaNed on T4; image augmentation degraded CER from 0.29 to 0.88; probability-averaging checkpoints hurt greedy CTC decoding (0.315 vs 0.292).
