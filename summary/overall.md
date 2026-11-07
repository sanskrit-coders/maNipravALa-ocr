# Manipravalam OCR summary

This directory splits classifier-specific notes out of the legacy
`evaluation/summary.md`.

Classifier documents:

- [patch-script-classifier.md](patch-script-classifier.md) — base 48×48 patch classifier (active production classifier, 0.9921 real-page accuracy).
- [obsolete/lines_to_words.md](obsolete/lines_to_words.md) — GT-constrained word boxes; old-vs-fixed evaluation.
- [obsolete/word-classifier.md](obsolete/word-classifier.md) — word-level `dev/tam/other/mixed` classifier.
- [obsolete/script-boundary-classifier.md](obsolete/script-boundary-classifier.md) — mixed-word boundary segmenter, Shiro-rekhā rule, CNN metrics, and wordwise faux-page post-mortem.
- [obsolete/column-separator.md](obsolete/column-separator.md) — per-column separator.
- [obsolete/line-classifier.md](obsolete/line-classifier.md) — experimental line classifier.

## Classifier I/O summary

| classifier | input | output labels | status |
|---|---|---|---|
| patch script (`script_cls.pt`) | 48×48 grayscale patch | `dev/tam/other/null` per patch | **production** (0.9921 real-page val acc on word-image maps) |
| word script (`script_sep.pt`, `word_cls.pt`) | 64×256 word image | `dev/tam/other/mixed` per word | obsolete / not retrained on clean word-image maps |
| script boundary (`shiro_label` / `boundary_combined.pt`) | 64×256 word image (+ optional base-projection channel) | boundary column → `dev` prefix + `tam` suffix | Shiro/projection rule best (MAE 0.0320); fails if emitted as isolated wordwise crops to Drive OCR |
| column separator (`column_sep.pt`) | 64×256 word image | `dev/tam/other` per column (256) | obsolete (real col acc 0.715) |
| line script (`line_cls`) | 64×1024 line image | `dev/tam/other` per line | experimental (~0.71 real line acc) |

## Pipelines

**Active production pipeline (`linewiseScriptSepDriveOCR`):**
`line separator` → `patch classifier (script_cls.pt)` + `safe ink-free word/run split` → `linewise phrase-strip faux page construction` → `Drive OCR` → `OCR collation`.

**Tested alternate (`wordwise-shiro-rekhA`, retired after Vol. 4 failure):**
`line separator` → `word separator (projection-valley)` → `word classifier (script_sep.pt)` → `script boundary classifier (Shiro-rekhā rule)` → `wordwise faux page construction` → `Drive OCR` → `OCR collation`.

## Current state & lessons learned (2026-11)

### 1. Assets & classifiers
- **Word-image maps:** **95,411 high-confidence box→text entries** across **2,203 pages**, stored as `data/RTS_sb_*/word-image-maps/{comm,core}/pNNNN.json` (render-and-DP alignment, confidence ≥ 0.45).
- **Patch script classifier (`training/local/script_cls.pt`):** Retrained on **69,366** training-split word-image-map words and evaluated on **1,763** evaluation-split words: patch accuracy **0.9921** (`dev` 0.989 / `tam` 0.995 / `other` 0.966). Backup of earlier synthetic model: `script_cls_synth_backup.pt`.
- **Word boxes & word classifier (`script_sep.pt`):** `train_words_fixed*.jsonl` (112,948 boxes) still had line-level GT assignment errors (e.g. `p0538` core line 4 assigns GT `कुरु’’` to a Tamil crop) because it bypassed `word-image-maps`. Consequently, `script_sep.pt` was never retrained on the verified `word-image-maps` and misclassifies real Tamil words on scans.
- **Script boundary:** On 14,208 `sb_01+sb_02` `dev→tam` words, the **Shiro-rekhā / projection-glyph rule** is best (boundary MAE **0.0320**, p@5 **0.718**, segment P **0.964** / R **0.959**), outperforming the combined image+projection CNN (`boundary_combined.pt`, MAE 0.1551, p@5 0.456).

### 2. Full-volume OCR experiments on `ra-tra-sA_5-commentaries_4.pdf` (806 PDF pages, 791 printed pages)
Four full-volume outputs were generated for Volume 4 under `.../rahasya-traya-sAraH/5-tIkAH/raw/` and audited against the original PDF page scans:

| output file | pipeline / faux source | lines | chars | audit verdict against PDF scans |
|---|---|---:|---:|---|
| `drive/local/4.drive.md` | **Linewise patch + Drive OCR** (`faux-pages-linewise-patch`, 1,569 faux pp, 79 batches of 20) | 24,209 | 1,165,663 | **Best by far (>99.8% complete, near-zero Latin garbage).** Cleanly preserves Sanskrit verses, Sanskrit commentary, and Tamil padavurai glosses (`अयं देहभृत्- இந்த ஜீவன்`). Only minor defects: dropped `०` in 3-digit verse numbers (`।। १ २ ।।` for `।। १०२ ।।`), 6 leaked `#T` tags, and occasional line-end phrase repeats. |
| `tesseract/4.md` | **Kraken 7.1.1 wordwise** (`best_0.8900.safetensors` on `word_segments` crops) | 24,470 | 1,141,993 | **Unusable (~100k char errors).** 90.34% char accuracy / 80.53% word accuracy means 1 in 5 words has an error, and the single dual-script CTC head hallucinates 1,456 intra-word script-confusion tokens (`एா`, `எः`, `५ெ`, `तர்`). |
| `drive/local/4.drive.wordwise.md` | **Wordwise Shiro-rekhā + Drive OCR** (`faux-pages-wordwise-shiro-rekhA`, 2,390 faux pp) | 24,615 | 1,156,622 | **Catastrophic failure on mixed pages.** Lost **76%–100% of Devanagari text** on mixed pages (pp. 100, 109, 141, 534, 700), replacing Sanskrit words with Latin/symbol hallucinations (`_qq4Hq sed: 47:`, `3அÀ 3544-`, `tafuqGiiH`). |
| `drive/local/4.drive.corrected2.md` | Post-processed merge of `4.drive.md` + `wordwise` | 24,264 | 1,137,771 | **Corrupted.** Repaired verse numbers (`।। १०२ ।।`) and added trailing `\`, but overwrote clean `4.drive.md` mixed lines with the garbled `wordwise` output. |

### 3. Key diagnostic findings ("Why results seemed bad")
1. **Volume 4 is an Anubandham (Appendix), NOT intra-word Manipravalam:**
   - Visual inspection of Volume 4's TOC and scans shows **zero classical intra-word Manipravalam commentary**:
     - Printed pp. 1–34: pure modern Tamil chapter summaries.
     - Printed pp. 35–407 & 534–780: Devanagari Sanskrit verses + Tamil *padavurai* (`ப-ரை`, where Sanskrit lemmas and Tamil glosses are separated by hyphens/spaces, e.g. `स्वाप--தூக்கம்`) + Tamil translations.
     - Printed pp. 408–533: **100% pure Devanagari Sanskrit commentary** (`अधिकारसङ्ग्रहः - सव्याख्यानः` and `कारिकादर्पणम्`; only the running header is Tamil).
     - Printed pp. 781–794: pure Tamil Divya Prabandham pasurams and commentary.
   - True intra-word Manipravalam (`श्रियःபதியாய்`, `ब्रह्माவுக்கு`, `आचार्यसम्बन्धத்தினுடைய`) lives in **Volumes 1–3** (`ra-tra-sA_5-commentaries_{1,2,3}.pdf`) and `RTS_sb_01`/`02`.
2. **The Printed Page 408 `(द्वया-)` → `(द्रयाணன்` illusion:**
   - On printed page 408 (PDF index 417), lines 4, 10, and 21 contain the pure Sanskrit heading `(द्वया-)`. There is no Tamil anywhere in the body of page 408.
   - In the legacy `script-split-tesseract/v4.md` output, the splitter cut `(द्वया-)` before `-)` and sent `(द्वया` to Sanskrit Tesseract (misread as `(द्रया`) and `-)` to Tamil Tesseract, which hallucinated Tamil `ணன்`, `னை`, `ண்` from the `-)` strokes—creating the chimera tokens `(द्रयाணன்`, `(द्रया னை`, `(द्रया ண்`.
   - Earlier sessions mistook those Tesseract hallucinations for "ground-truth mixed words lost by Drive OCR," when `4.drive.md` had actually read them **correctly** as `द्वया-` / `(द्वया-`!
3. **Why `faux-pages-wordwise-shiro-rekhA` (`4.drive.wordwise.md`) collapsed:**
   - **Context starvation in Google Drive OCR:** Drive OCR is a line/phrase-context engine. When fed faux pages of isolated 1-word or sub-word crops (`wordwise-shiro`), it lacks horizontal line context and language-model grounding, causing it to drop short Sanskrit words or hallucinate Latin strings (`tafuqGiiH`, `dysfரிார்`).
   - **Cross-script bleeding:** Using the old `script_sep.pt` placed Tamil words onto Devanagari faux pages and sliced Sanskrit visargas/matras (`ीः-`, `ा-`) onto Tamil faux pages.
4. **Why `"other"`-class filtering drops digits and punctuation:**
   - `pipeline.py` only composes `script in ("dev", "tam")` and drops `"other"` segments. When the middle zero `०` in `१०२` or a closing parenthesis `)` is labeled `"other"`, it is omitted from the faux page (`।। १ २ ।।`, `(द्वया-`).
5. **Why Volumes 1–3 remain hard (the Intra-Word Granularity Paradox):**
   - Unlike Volume 4, Volumes 1–3 switch scripts *inside words* (`आचार्यसम्बन्धத்தினுடைய`).
   - Splitting inside every mixed word creates isolated 1–3 akshara Tamil suffix strips (`வுக்கு`, `த்தினுடைய`) that lack word/line context on faux pages. Conversely, earlier Volume 1 faux pages (`faux-pages-linewise-fixed`, Oct 24) were built *before* `script_cls.pt` was retrained on the 95,411 real word-image maps (Oct 26), causing severe cross-script routing errors on Vol. 1 p20.

## Ways forward

1. **Immediate (Volume 4 completion — no re-OCR needed):**
   - Use `raw/drive/local/4.drive.md` as the authoritative Volume 4 text (retire `4.drive.wordwise.md` and `4.drive.corrected2.md`, and keep `tesseract/4.md` only as a diagnostic archive `4.kraken-wordwise.md`).
   - Run a lightweight, deterministic cleanup script on `4.drive.md` to: (a) remove the 6 leaked `#T` tags, (b) restore dropped `०` in 3-digit verse numbers (`।। १ २ ।।` → `।। १०२ ।।`, cross-checking `plain-drive/4.md` where needed), (c) deduplicate the few line-end repeat phrases, and (d) standardize page markers `[[२]]`–`[[७९४]]`.
2. **Fix `pipeline.py` before running Volumes 1–3 (`ra-tra-sA_5-commentaries_{1,2,3}.pdf`):**
   - **Never drop `"other"` runs:** Absorb `"other"` (digits, dandas, parentheses, hyphens) into the adjacent `dev` or `tam` segment so verse numbers and punctuation are never deleted.
   - **Never emit isolated wordwise faux pages to Drive OCR:** Always group consecutive same-script runs on a line into **phrase/line strips** (`make_segments`), and pad/anchor short sub-word suffixes so Drive OCR sees coherent horizontal strips.
   - **Use the retrained `script_cls.pt` (0.9921 real-page patch accuracy) + Shiro-rekhā safe split:** Volume 1 has not yet been run with the Oct 26 retrained `script_cls.pt`. For mixed words where `script_cls.pt` detects both `dev` and `tam`, snap the internal split to the Shiro-rekhā / ink-free valley (`_safe_split_x`), keeping outer word bounds intact.
   - **Filter 1-speck faux segments:** Skip segments narrower than a minimum ink threshold so smudge crops do not consume tags or Drive batches.
3. **Alternative / complementary architecture for dense Manipravalam (Volumes 1–3):**
   - **In-place Context-Preserving Masked Line Faux Pages:** Instead of cutting mixed lines into disconnected micro-strips, emit each mixed line twice as a **full-width line strip**: once on the Devanagari faux page with pure-Tamil runs masked/whited-out (or replaced by baseline guides), and once on the Tamil faux page with pure-Devanagari runs masked. Drive OCR then preserves full line scale and baseline geometry without tag-fragmentation.
   - **Line-level / Phrase-level Local Recognizer or LLM Post-Correction:** If training a local recognizer, train on **multi-word line/phrase strips** (or fine-tune a vision-language / seq2seq line model on the 2,203 aligned `RTS_sb_01`/`02` pages), never on isolated words without context.

## OCR models inventory

- **Mixed-script word OCR (`training/local/kraken_words_model/best_0.8900.safetensors`):**
  - Kraken 7.1.1 VGSL, **4.1M params**, trained on **92,976** word crops from training word-image maps; evaluated on **2,435** eval word crops (`vishvasvasuki/rts-word-ocr-train`).
  - Character accuracy **90.34%**, word accuracy **80.53%** (Tamil **94.31%**, Devanagari **88.81%**; 2,167 char errors: 306 ins, 1,096 del, 765 sub).
  - **Verdict:** Useful as a local alignment/confidence signal, but 80.5% word accuracy is too noisy for direct page transcription (`tesseract/4.md`).
- **Mixed-script line OCR (`training/local/kaggle_v10_best.pt`, `devTamCharsOCR`):**
  - Kraken VGSL line model trained on 16.5k lines: **0.292 line CER / 0.238 block CER** on whole lines, but collapses on isolated short runs (0.812 CER).
- **Linewise Script-Separated Drive OCR (`linewiseScriptSepDriveOCR`):**
  - **0.112 block CER** on the 5 verified `RTS_sb_01` Manipravalam blocks (2.0× better than local script-split Tesseract at 0.227, 2.1× better than `devTamCharsOCR` at 0.238, and 3.9× better than unsplit Drive at 0.435). Production winner on Volume 4 (`4.drive.md`).

## Artifacts

- `training/scripts/script_cls/pipeline.py` — production linewise faux-page + Drive OCR pipeline.
- `training/local/script_cls.pt` — 48×48 patch script classifier retrained on word-image maps (**0.9921** val acc).
- `training/local/kraken_words_model/best_0.8900.safetensors` — Kraken word OCR model (`training/local/kraken_words_metrics.json`).
- `training/local/kaggle_v10_best.pt` — `devTamCharsOCR` whole-line model.
- `data/RTS_sb_*/word-image-maps/` — 95,411 verified per-page word image→text maps across 2,203 pages.
- `data/script-separator/*.jsonl` — word box and boundary datasets (`sb_01` and `sb_02`).
- `evaluation/summary.md` — full benchmark evaluation (block/line CER, baselines, and full-volume post-mortem).
