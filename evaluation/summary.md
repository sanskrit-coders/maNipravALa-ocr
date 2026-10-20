# OCR evaluation for mixed Tamil–Devanagari (manipravala) print

This report evaluates OCR engines and page pipelines for **sArabodhinI vol. 01–02** — a
manipravala print in which Tamil and Devanagari alternate *inside words and lines* (Tamil suffixes
on Sanskrit stems, Devanagari commentary labels, quoted mUlam). Two test beds are used: a held-out
**line** split of 534 crops from the same volumes, and **three real page excerpts** (25 GT lines)
hand-transcribed from the PDFs.

Candidates: **Tesseract** (`tam+san`), **PaddleOCR PP-OCRv4** pretrained models (the engine behind
the Indic-OCR/BytesByJay repo), **Google Drive** document OCR, **devTamCharsOCR** (our from-scratch line recognizer), and
**scriptSepDriveOCR** (a script-run splitting page pipeline that reflows script-pure runs onto
faux pages before OCR).

**Bottom line**

| approach | line CER (445-line set) | page CER (3 excerpts) | note |
|---|---|---|---|
| Tesseract 5 `tam+san`, psm 7 | **0.476** | — | best zero-shot engine on line crops |
| PaddleOCR PP-OCRv4 `ta`/`deva` (rec, oracle route) | 0.71–0.73 | — | shipped "Indic-OCR" engine; not usable |
| **devTamCharsOCR (v10)** (from scratch, train split only) | **0.292** | 0.351 | strongest line-crop model; degrades on pages fed whole |
| Drive OCR on the whole text region | — | 0.26 (0.46 / 0.07) | erratic; output frequently unusable |
| **scriptSepDriveOCR (v2)** | — | **0.082** (0.097 / 0.032 / 0.119) | best real-page output; 4× better than feeding lines whole |

Five findings dominate everything below: (1) **script purity inside a recognized unit is the
single biggest factor** — every engine fails on mixed units; (2) pretrained PaddleOCR transfer is
poor for this material; (3) devTamCharsOCR is the best single-line recognizer, but it is a
line-crop model and loses accuracy on full page lines; (4) **scriptSepDriveOCR** (splitting page
lines by script and assembling the page from script-pure runs) beats all alternatives on pages by
~4×; (5) **combining the two systems does not reduce CER** (§4.6). The remaining blocker is
throughput, not accuracy.

---

## 1. Models and configuration

Five systems are compared; they differ in their unit of recognition and in what they were trained on:

* **Tesseract 5** — pretrained line OCR (`tam+san`); the zero-shot baseline.
* **PaddleOCR PP-OCRv4** — pretrained recognizers for Tamil and Devanagari (`ta`, `devanagari`);
  the recognition engine shipped by the Indic-OCR repo.
* **Google Drive OCR** — cloud document OCR, used on pages (whole text region or composed faux
  pages), not as a line model.
* **devTamCharsOCR** — our from-scratch line recognizer (kraken-VGSL spec) trained on 16.5k
  Devanagari/Tamil line crops from these volumes; outputs one line of text per line crop.
* **scriptSepDriveOCR** — a page-level pipeline: classifies script at word/run level, reflows
  script-pure runs onto faux pages, OCRs those with Google Drive, and reassembles the page.

| engine | version / model | how it was run |
|---|---|---|
| Tesseract | 5.x, `tam+san` traineddata | psm 7 on line crops; psm 6 where used on page bands |
| PaddleOCR | 2.7.3 / 2.6.2 (CPU, 6 threads) | PP-OCRv4 `ta` and `devanagari` recognizers, PP-OCRv3 multilingual detector; `rec` (line crops) and `det+rec` (`.ocr(img, cls=True)`) |
| Google Drive OCR | cloud document OCR (OAuth client) | whole text region, and per-script faux pages (`drive_ocr_images.py`) |
| devTamCharsOCR | kraken-VGSL spec, torch replica, v10/v13 | CTC greedy decode on line crops (§6) |
| scriptSepDriveOCR | script classifier + faux pages + Drive OCR | full page, 300 dpi (§4.3) |

### 1.1 The PaddleOCR candidate (the Indic-OCR repo)

The repo (BytesByJay/Indic-OCR, commit `main`, cloned 2026-10-18; 4 commits, no releases, no
weights) advertises a multi-script OCR stack. Audited:

| component | claim | reality |
|---|---|---|
| OCR engine | "PaddleOCR/TrOCR" | thin wrapper over upstream PaddleOCR; TrOCR backend is `microsoft/trocr-base-handwritten` (English handwriting only) |
| script classifier | automatic script detection | architecture + training code only; **no weights** — random init as shipped |
| fine-tuning (`train.py`) | OCR fine-tuning | **stub** ("OCR fine-tuning is not yet implemented") |
| preprocessing | "advanced preprocessing" | OpenCV NLM denoise + CLAHE + unsharp mask |

So its recognition accuracy is exactly PaddleOCR's pretrained accuracy. We evaluate that engine
honestly (both modes, both scripts, with an oracle router as an upper bound) rather than through
the repo's wrapper.

**Preprocessing ablation:** Indic-OCR's denoise+CLAHE+unsharp on the same models changes nothing
materially — oracle best-of CER 0.732 with and without it on 445 lines (per-language: `ta` 0.760
vs 0.757, `hi` 0.897 vs 0.912; all-534 best-of 0.843 vs 0.839).

**Runtime (CPU):** rec ≈ 1.6 lines/s; det+rec ≈ 0.4 lines/s.

Reproduce:

```bash
training/scripts/indic-ocr/setup_env.sh
/tmp/opencode/indicocr-venv/bin/python training/scripts/indic-ocr/eval_paddleocr.py \
    --mode both --langs ta,hi --dump evaluation/line_eval_paddleocr.jsonl
python3 training/scripts/indic-ocr/compare_with_baseline.py
```

## 2. Data and metrics

### 2.1 Line set

* 534 single-line crops from vol. 01–02, auto-aligned GT.
* 89 lines carry broken alignment targets (tiny slivers with 1000+ char references, discovered
  while building the line data). They are excluded from the fair comparison; the resulting
  **445 CTC-feasible lines** are used for all engine numbers. Both figures are reported (§3.2).
* Composition of the feasible set: 55 Devanagari-dominant, 343 mixed (Tamil+Devanagari),
  47 Tamil-dominant lines.

### 2.2 Page set

Hand-transcribed excerpts (2× zoomed line crops; corpus cross-checks where the printed text
matches an e-text). GT and all stored predictions live in `evaluation/page_gt/`.

| page | lines | chars | content |
|---|---|---|---|
| PDF1 p100 | 11 | 746 | mUlam opening + `(सा.दी.)` / `(सा.स्वा.)` commentary |
| PDF2 p200 | 9 | 812 | mUlam + `(सा.प्र.)` commentary |
| PDF3 p400 | 5 | 409 | `चरमश्लोकाधिकारः` commentary (Tamil clusters partly uncertain) |

### 2.3 Metrics and conventions

* CER/WER via `jiwer`, on raw and whitespace-stripped text, identical across all engines
  (`training/scripts/script_cls/eval_lines.py`; per-line dumps under `evaluation/`).
* **Oracle script routing:** where an engine must choose a script-specific model, we report
  `best-of` = per-line minimum-CER pick. This is an upper bound for any automatic router (mixed
  lines are unrouteable in principle anyway).
* Page-level and line-level CERs are **not directly comparable**: page excerpts include headers,
  item numbers, and line-break artifacts, and their GT is smaller and less certain (see §8).
* Tesseract is run in line mode (psm 7); PaddleOCR in `rec` and `det+rec` modes; Drive OCR on
  page-sized regions and on composed faux pages.

## 3. Line-level results

### 3.1 Feasible subset — 445 lines (fair comparison)

| engine | CER | CER no-space | WER | WER no-space |
|---|---|---|---|---|
| Tesseract `tam+san` psm7 | **0.476** | 0.484 | 0.976 | 0.998 |
| PaddleOCR rec `ta` | 0.757 | 0.762 | 1.030 | 1.000 |
| PaddleOCR rec `hi` | 0.912 | 0.944 | 1.036 | 1.000 |
| PaddleOCR rec best-of (oracle) | 0.732 | 0.735 | 1.022 | 1.000 |
| PaddleOCR rec best-of + Indic-OCR preprocessing | 0.732 | 0.731 | 1.017 | 1.000 |
| PaddleOCR det+rec `ta` | 0.755 | 0.774 | 1.116 | 1.000 |
| PaddleOCR det+rec best-of (oracle) | 0.710 | 0.728 | 1.117 | 1.000 |
| **devTamCharsOCR (v10)** (train split only) | **0.292** | **0.293** | **0.608** | 0.980 |
| devTamCharsOCR (v13 repeat) | 0.294 | 0.295 | 0.605 | 0.964 |

### 3.2 All 534 lines (includes the 89 broken-target lines)

| engine | CER | CER no-space | WER |
|---|---|---|---|
| Tesseract `tam+san` psm7 | 0.705 | 0.710 | 0.986 |
| PaddleOCR rec best-of | 0.839 | 0.839 | 1.012 |
| PaddleOCR det+rec best-of | 0.707 | 0.727 | 1.096 |

The 89 broken lines have huge references and dominate character-weighted CER; the det+rec
pipeline "wins" there only by emitting nothing on slivers, capping their per-line CER at 1.0
instead of >1.0.

### 3.3 By GT script composition (feasible subset)

| script | n | Tesseract | Paddle rec `ta` | Paddle rec best (oracle) |
|---|---|---|---|---|
| devanagari-dominant | 55 | **0.485** | 0.942 | 0.731 |
| mixed (Tamil+Devanagari) | 343 | **0.492** | 0.752 | 0.750 |
| tamil-dominant | 47 | **0.312** | 0.573 | 0.570 |

Oracle picks (all 534): rec `ta` 455 / `hi` 79; det+rec `ta` 390 / `hi` 144. **PaddleOCR never
beats Tesseract in any script bucket.**

### 3.4 Qualitative failure modes

| line (GT) | PaddleOCR best | Tesseract |
|---|---|---|
| தன்பற்றில் - தானே தன் **संबन्ध**த்தை உண்டாக்குகையில். … | `Id Miog, ' ounைoூs1g- Go$h- ms ng - gudr?` | தன்பற்றில் - தானே தன் **84**த்தை உண்டாக்குகையில். … |
| **सङ्ग्रहोक्ति.** தகவோக்கத்தில் - **कृपाधिक्य**த்தில். இது 'அருள் | `னானஎன்பதினிப்uொிர.தகேவாக்கத்தில்-துரித்தில.இதரஅஞளர்` | னான்” என்பதின் **संग्रहोक्ति**. தகவோக்கத்தில் - **ஊரிண**த்தில். இது “அருள் |
| பாட்டில் சொன்ன இன்பம் முதலான … (Tamil only) | `பாட்டல்ெசானனஇன்பம்ழதலான…` (CER 0.27, spaces lost) | — |

* **Mixed-script lines defeat both models.** The `ta` model renders Devanagari as Latin garbage
  (`संबन्ध` → `Go$h-`) and vice versa (`hi` outputs digits/Latin for Tamil). 343/445 feasible
  lines are mixed — this is the dominant error source everywhere.
* **Even same-script text loses spacing** and confuses similar ligatures (`சொன்ன` → `ெசானன`).
* The Devanagari model (`hi`, trained on Hindi) is weakest: 0.91 CER alone.

### 3.5 Takeaways

* On clean line crops, devTamCharsOCR is **1.6× better than Tesseract** and **2.5× better than
  PaddleOCR**; Tesseract remains the best zero-shot engine.
* Character error is dominated by *script mixing*, not by individual glyph shapes: models trained
  on pure-script data are reliable, mixed lines are where they break.
* Line-level accuracy does not transfer to page lines: devTamCharsOCR scores 0.29 on crops but
  0.35 on page lines fed whole (§4.4).

## 4. Page-level evaluation

### 4.1 Why real pages are harder

Printed pages mix scripts not only between lines but *inside* a line: a Devanagari mUlam quote
with Tamil commentary labels, Tamil verbal suffixes on Sanskrit stems, parentheses and item
numbers. Whole-line engines (ours included) then apply a model trained on predominantly pure
script to a unit containing both scripts; whole-page OCR engines reorder, drop or garble lines.

### 4.2 Line segmentation audit (`training/scripts/script_cls/audit_lines.py`)

Before trusting the pipeline we audited `apply_ocr_pdf.segment_lines` on four pages (PDF1 p50,
p100; PDF2 p200; PDF3 p400):

* All Tesseract word-box centers fall inside their segment (35/35, 28/28, 25/25, 31/31); no
  lines are merged or split.
* Ink coverage 84–92%. The only missed ink is full-width section rules (p100: ~56k px of rules
  plus ~1.2k px of stray marks; on p50 *all* missed ink is rules).
* Fully-contained box counts are lower only where Tesseract boxes straddle the ±2 px crop edge
  (p100: 21/35), which occasionally clips descenders but loses no line content.

Line segmentation is not a bottleneck; the rest of this section uses it as-is.

### 4.3 Script-run splitting pipeline (`training/scripts/script_cls/pipeline.py`)

Inspired by `doc_curation.pdf.drive_ocr.split_and_ocr_all` (which boxes lines by script before
OCR), the pipeline boxes *runs* and assembles the page from script-pure pieces:

1. Render PDF page at 300 dpi; segment lines (above).
2. Word segmentation by horizontal ink projection (gap ≥ 10 px).
3. Sliding 48×48 patch classifier, stride 8; per-patch labels `dev/tam/other/null`; a run keeps
   a patch's label only above 0.45 confidence (else `null`).
4. Merge patches into runs; drop runs < 20 px wide; split words at script transitions.
5. Compose one faux page per script: each run is a line, scaled to 90 px height (≥0.6), 130 px
   vertical gap, width cap 1780 px, JPEG q92; 20 runs per faux page.
6. OCR each faux page with Google Drive. Every run carries a printed `#TNN` tag to its left; the
   parser maps tags → runs, so dropped or merged OCR lines cannot shift the page. Continuation
   lines bind to the previous tag; markdown escapes and trailing base64 image blobs are stripped.
7. Missing tags trigger one retry of the affected chunk (no per-run fallback); runs that remain
   unreadable are left empty. Reassemble the page with per-run `pre_space` from the patch map.

The tag scheme (v2) replaced the pilot's positional mapping and cut the 299-run p100 page from
~40 min to ~4 minutes of Drive calls.

**Script classifier:** synthetic 48×48 patches rendered from `data/texts/{core,commentary}` with
the Devanagari/Tamil font sets (plus noise/blank for `other`/`null`), small CNN, CPU ~25 min.
Validation accuracy **0.937** (dev 0.951 / tam 0.938 / other 0.417 / null 0.975). The weak
`other` class is harmless in practice: only high-confidence dev/tam runs are batched, and the
page overlay (`/tmp/opencode/scriptcls_run/overlay/p0100.jpg` during the pilot) shows boxes
tracking script boundaries even inside words (`राजा|வுக்கும்`, `प्रियत|மங்களையும்`, `नाय|ं|देवो`).

### 4.4 Results

| system | CER p100 | CER p200 | CER p400 | mean CER | mean WER |
|---|---|---|---|---|---|
| **script runs + faux pages + Drive OCR (v2)** | **0.0969** | **0.0315** | **0.1188** | **0.082** | 0.396 |
| devTamCharsOCR (page lines fed whole) | 0.3475 | 0.3517 | 0.3543 | 0.351 | 0.805 |
| Drive OCR on the whole text region (no split) | 0.4564 | 0.0674 | — | 0.262* | 0.599* |

\* two pages only. (The positional pilot scored 0.0944 on p100; the tag version's 0.0969 is
within GT/parse noise.) scriptSepDriveOCR beats devTamCharsOCR by ~4× and the unsplit Drive
baseline on every page. Whole-page Drive OCR is qualitatively unusable (garbled,
reordered, base64 blobs), and Tesseract on such lines is worse still (§3).

### 4.5 Residual errors

Errors are mostly *assembly/recognition* artifacts: stray markdown (`|`, `\*`, `\.`), a dropped
closing paren, missing word spaces where the printed gap is < 10 px, and Tamil vowel-sign/matra
confusions (`नैर्भर्य` for `नैश्शर्य`, `प्रदर्शिபா` for `प्रदर्शिया`, `कृषनै`). The p400 loss is
dominated by dense Tamil commentary and 300 dpi glyph ambiguity, not by the script split; its GT
also carries a few uncertain Tamil clusters.

### 4.6 Combining devTamCharsOCR with scriptSepDriveOCR (negative result)

The two page systems fail differently, so we tested whether mixing them lowers CER. All variants
reuse the same run segmentation and page assembly and differ only in the per-run text source
(`training/scripts/script_cls/combine_eval.py`; per-run texts in
`evaluation/page_gt/combine_runs_*.jsonl`).

| variant | CER p100 | CER p200 | CER p400 | mean |
|---|---|---|---|---|
| **scriptSepDriveOCR (baseline)** | **0.0969** | **0.0315** | **0.1188** | **0.082** |
| devTamCharsOCR on script-pure runs | 0.7700 | 0.7135 | 0.6861 | 0.723 |
| run-level hybrid (devTam if Drive empty/suspicious) | 0.1077 | 0.0315 | 0.1188 | 0.086 |
| devTam fills only empty Drive runs | 0.0969 | 0.0315 | 0.1188 | 0.082 |
| per-line oracle selection (upper bound) | 0.0969 | 0.0315 | 0.1188 | 0.082 |

* **devTamCharsOCR does not transfer to isolated runs.** It is a full-line model; on tight 1–4
  akshara crops it loses context and decodes at CER ≈ 0.69–0.77, so it cannot be the backend for
  faux pages as-is.
* **Hybrid/repair rules do not help.** Filling empty Drive runs with devTam changes nothing (those
  runs are sub-glyph stubs unreadable by both). The suspicious-text rule even hurt p100
  (0.108 vs 0.097): two lines that looked broken in Drive were still better than devTam's.
* **Script-conditional mixes are worse too.** Using devTam for only one script's runs (Drive for
  the rest) gives CER 0.39/0.70/0.42 when Devanagari runs come from devTam and 0.49/0.06/0.39 when
  Tamil runs do (p100/p200/p400).
* **There is no complementarity at line granularity.** A per-line oracle that may pick either
  system selects scriptSepDriveOCR on all 25 GT lines (`dev_line_wins = 0` everywhere).

Combining the current two systems therefore does **not** reduce CER. The productive combination is
to put the recognizer into the same regime as Drive — train/fine-tune a script-pure run recognizer
and use it as the local backend (removing the Drive dependency) — rather than mixing a full-line
model with run-level inputs.

## 5. Verdict and recommendations

* **For line crops** (clean scans): devTamCharsOCR (CER 0.29) is the best available; Tesseract
  (0.48) is the best zero-shot engine and the only sensible pretrained fallback. **PaddleOCR
  pretrained models are not usable** (0.71–0.73), and the Indic-OCR repo adds no accuracy of its
  own (no weights, stub fine-tuning, English-only TrOCR, preprocessing that changes nothing).
* **For real pages:** feed neither whole lines nor whole pages to a single-script engine.
  scriptSepDriveOCR is the winning pattern by a wide margin (mean page CER **0.082** vs 0.351 for
  devTamCharsOCR and 0.262 for unsplit Drive OCR).
* **Do not combine the two systems as-is:** run-level and line-level mixing of devTamCharsOCR with
  scriptSepDriveOCR gives no gain (§4.6) — they are not complementary at these units.
* **Fine-tuning PaddleOCR's PP-OCRv4** on our 16.5k feasible train lines remains a plausible
  follow-up (it imports Devanagari/Tamil glyph knowledge our from-scratch model lacks), but it is
  now a comparison, not a necessity: the from-scratch model already reaches 0.292.
* **Production path:** the accuracy blocker is solved; the throughput blocker is not. Drive OCR
  latency dominates (2–4 min/page healthy, arbitrary stalls otherwise). Two mitigations, in
  order of value: (a) recognize script-pure faux pages with a **local** per-script recognizer
  (fine-tuned PP-OCRv4 or a script-pure variant of devTamCharsOCR) instead of Drive; (b) reduce calls
  by merging tiny fragments before batching (raise/word-adaptive gap).
* **Use Tesseract for line segmentation and page inventory** (headers, rules, item numbers) —
  it is reliable there; do not use it as the final recognizer on mixed lines.

## 6. Training devTamCharsOCR (for context and reuse)

Trained from scratch on the **train split** of the line data (16.5k CTC-feasible lines, vol
01–02; eval split never seen).

| item | value |
|---|---|
| architecture | exact kraken VGSL spec `[1,64,0,1 Cr3,13,32 … S1(1x0)1,3 Lbx200×3 …] O1c165` — torch-only replica, verified bit-identical to kraken (keys, outputs, init) via `training/scripts/verify_kernel_arch.py` |
| training | Kaggle T4, batch 32, lr 3e-3 (warmup 400, cosine → 5e-5), 40 epochs, fp32, **no augmentation**, seed 1234 |
| result (445 eval lines) | **CER 0.2922 / cer_nows 0.2931 / WER 0.6083** (v10; repeat v13: 0.2940 / 0.2953 / 0.6047) |
| kernel | `vishvasvasuki/manipravala-lines-train` (private, GPU) |

Hard-won details (each cost a Kaggle run): kraken's **init matters** (wrong init stalls at
0.85–0.88; replicate conv `uniform ±0.1`, LSTM orthogonal + forget-bias 1.0, Linear Xavier);
**AMP fp16 occasionally NaNs** on T4 → use fp32; **augmentation severely hurts** this dataset
(0.88 vs 0.29); **80-epoch cosine is worse than 40** (0.354 — LR stays in the noise zone);
probability-averaging the two checkpoints *hurts* CTC decoding (0.315) — no ensemble. Scripts:
`training/scripts/eval_kraken_manual.py`, `eval_kraken_ensemble.py`,
`training/kaggle/kernel/{train.py,arch.py}`.

## 7. Artifacts

| file | contents |
|---|---|
| `evaluation/line_eval_paddleocr.jsonl` | per-line rec-mode predictions (`pred_ta`, `pred_hi`, oracle `pred`) |
| `evaluation/line_eval_paddleocr.jsonl.summary.json` | rec-mode aggregate metrics |
| `evaluation/line_eval_paddleocr_det.jsonl(+.summary.json)` | det+rec-mode predictions |
| `evaluation/line_eval_paddleocr_pp.jsonl(+.summary.json)` | rec-mode with Indic-OCR preprocessing |
| `evaluation/line_eval_baseline.jsonl` | Tesseract baseline (534 lines) |
| `evaluation/line_eval_kaggle_v10_best.jsonl` | devTamCharsOCR (v10) per-line predictions (0.292 CER) |
| `evaluation/line_eval_kaggle_v13_best.jsonl` | devTamCharsOCR (v13 repeat) per-line predictions |
| `evaluation/page_gt/ra-tra-sA_{1_p0100,2_p0200,3_p0400}.{gt,pred_scriptcls,pred_linenet_v10,pred_drive_whole_region}.txt` | page mini-GTs (11/9/5 lines) and the stored predictions |
| `evaluation/page_gt/ra-tra-sA_*.pred_{devruns,hybrid}.txt` | page assemblies for the combination variants (§4.6) |
| `evaluation/page_gt/combine_runs_{100,200,400}.jsonl` | per-run Drive vs devTam texts (reusable without re-OCR) |
| `evaluation/combine_eval.json` | CER for the combination variants (§4.6) |
| `training/local/kaggle_v10_best.pt` | devTamCharsOCR best checkpoint (kraken-spec, 40 ep, fp32, no aug) |
| `training/local/kaggle_v13_best.pt` | repeat-run checkpoint (same config) |
| `training/local/script_cls.pt` | script-patch classifier (48×48, dev/tam/other/null; val_acc 0.937) |
| `training/scripts/indic-ocr/` | `setup_env.sh`, `eval_paddleocr.py`, `compare_with_baseline.py` |
| `training/scripts/script_cls/` | `train_classifier.py`, `pipeline.py`, `audit_lines.py`, `combine_eval.py`, `drive_ocr_images.py`, `eval_lines.py` |
| `data/RTS_5c_01/ocr_scriptcls/` | pilot output for PDF1 p100 (`.txt`, `runs.jsonl`) |
| `/tmp/opencode/scriptcls_v2_out/` | tag-scheme (v2) outputs for the three evaluated pages |

## 8. Caveats and open questions

* **Scope:** 3 pages from 3 volumes but only 25 GT lines total; line-level set is 445 lines from
  vol 01–02. The ~4× page-level gap is consistent across all three pages, but absolute page CERs
  will move with GT precision (the p400 GT was revised by ~25 chars after zoom review and still
  carries uncertain Tamil clusters).
* **Page numbers are not comparable to line numbers:** page GT includes headers/item numbers and
  line-break artifacts that line-crop GT avoids.
* **Drive API behavior** is erratic (0.067 on p200's clean 9-line band vs 0.456 on p100;
  multi-minute stalls, one run delayed overnight) — single-page whole-region numbers mean little.
  Script splitting wins on every page regardless.
* **Open questions:** (1) does the script-split gain hold for a *local* per-script recognizer, or
  is it specific to Drive OCR? (2) can a script-pure variant of devTamCharsOCR close the gap on
  faux pages? (3) how much does word-gap tuning recover the remaining spacing errors?
  (4) evaluate on more pages/volumes, especially pages with two-column layouts (none seen so far).
* Output text keeps Drive's markdown escapes; the parser strips common ones but a stricter
  normalizer is needed before this is usable as an e-text.
