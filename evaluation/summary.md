> Classifier-specific notes are now split into `summary/` (see `summary/overall.md`). This file remains the legacy full evaluation.

# OCR evaluation for mixed Tamil–Devanagari (manipravala) print

This report evaluates OCR engines and page pipelines for **sArabodhinI vol. 01–02** — a
manipravala print in which Tamil and Devanagari alternate *inside words and lines* (Tamil suffixes
on Sanskrit stems, Devanagari commentary labels, quoted mUlam). Two test beds are used: a held-out
**line** split of 534 crops from the same volumes, and **five hand-verified block excerpts**
(89 printed lines, 6,193 non-space GT characters) from `RTS_sb_01`. Every number below comes from
those `RTS_sb_*` sets; the earlier `RTS_5c` pilot is archived and is not used for evaluation.

Candidates: **Tesseract** (`tam+san`), **PaddleOCR PP-OCRv4** pretrained models (the engine behind
the Indic-OCR/BytesByJay repo), **Google Drive** document OCR, **devTamCharsOCR** (our from-scratch line recognizer),
**scriptSepDriveOCR** (a script-run splitting page pipeline that reflows script-pure segments onto faux pages before OCR), and **linewiseScriptSepDriveOCR** (the same dense, pure-script pages with line-strip ordering).

**Bottom line**

| approach | line CER (445-line set) | block CER (5 `RTS_sb_01` blocks) | note |
|---|---|---|---|
| Tesseract 5 `tam+san`, psm 7 | **0.476** | 0.352 | best zero-shot engine on mixed lines |
| Tesseract, local script-split (`san`/`tam` on script segments) | — | **0.227** | best local configuration; 0.308 with the same mixed model on the same segments |
| PaddleOCR PP-OCRv4 `ta`/`deva` (rec, oracle route) | 0.71–0.73 | — | shipped "Indic-OCR" engine; not usable |
| **devTamCharsOCR (v10)** (from scratch, train split only) | **0.292** | 0.238 | strongest line-crop model; transfers to block lines |
| devTamCharsOCR on script runs | — | 0.812 | isolated 1–4-akshara crops are outside its training regime |
| devTamCharsOCR on script segments | — | 0.599 | script-pure input does **not** close the gap |
| Drive OCR on the whole text block | — | 0.435 | erratic; output often reorders/drops lines |
| **scriptSepDriveOCR (v2)** | — | **0.114** | pure-script line strips; 3.9× better than whole-block Drive |
| linewiseScriptSepDriveOCR | — | **0.112** | same dense pages; best micro average |

Line and block CERs use different GT units and are **not comparable** (see §2.3). Six findings
dominate everything below: (1) **script mixing is a major failure mode for pretrained engines** —
Tesseract and PaddleOCR substitute or drop Devanagari in mixed units, and Paddle is clearly worse
on mixed than on either pure bucket; (2) pretrained PaddleOCR transfer is poor for this material; (3) devTamCharsOCR is the best single-line
recognizer and even improves slightly on block lines, but it collapses on the short fragments
that faux pages require; (4) **scriptSepDriveOCR** (dense, pure-script line strips) beats the best alternative by **2.1×** in the block micro average (3.9× vs unsplit Drive), and the packed linewise variant is marginally better still (**0.112**); (5) the script-split gain **does hold for a local
per-script recognizer** (Tesseract: 0.352 → 0.227), so it is not Drive-specific, although Drive
keeps a 2.0× lead; (6) a **script-pure devTamCharsOCR input variant does not close the gap**
(0.599 vs 0.238 whole-line), because the deficit is the short-fragment regime, not script mixing;
combining the two systems does not reduce CER (§4.6). The remaining blocker is throughput, not
accuracy.

---

## 1. Models and configuration

Six systems are compared; they differ in their unit of recognition and in what they were trained on:

* **Tesseract 5** — pretrained line OCR (`tam+san`); the zero-shot baseline. Also used in a
  script-split configuration: the same page segmentation as the pipeline, but each script-pure
  run/segment is recognized with the matching `san` (Devanagari) or `tam` model.
* **PaddleOCR PP-OCRv4** — pretrained recognizers for Tamil and Devanagari (`ta`, `devanagari`);
  the recognition engine shipped by the Indic-OCR repo.
* **Google Drive OCR** — cloud document OCR, used on blocks/pages (whole text region or composed
  faux pages), not as a line model.
* **devTamCharsOCR** — our from-scratch line recognizer (kraken-VGSL spec) trained on 16.5k
  Devanagari/Tamil line crops from these volumes; outputs one line of text per line crop.
* **scriptSepDriveOCR** — a block/page-level pipeline: classifies script at word/run level,
  reflows script-pure line segments onto faux pages, OCRs those with Google Drive, and reassembles the page.
* **linewiseScriptSepDriveOCR** — the same dense, pure-script line-strip pages, ordered
  line-wise and packed across multiple printed lines per page.

| engine | version / model | how it was run |
|---|---|---|
| Tesseract | 5.x, `tam+san`, `san`, `tam` traineddata | psm 7 on line/run/segment crops |
| PaddleOCR | 2.7.3 / 2.6.2 (CPU, 6 threads) | PP-OCRv4 `ta` and `devanagari` recognizers, PP-OCRv3 multilingual detector; `rec` (line crops) and `det+rec` (`.ocr(img, cls=True)`) |
| Google Drive OCR | cloud document OCR (OAuth client) | whole text block, and per-script faux pages (`drive_ocr_images.py`) |
| devTamCharsOCR | kraken-VGSL spec, torch replica, v10/v13 | CTC greedy decode on line/run/segment crops (§6) |
| scriptSepDriveOCR | script classifier + faux pages + Drive OCR | block/page, 150 dpi crop upscaled 2×; 20 runs per faux page |
| linewiseScriptSepDriveOCR | script classifier + dense faux pages + Drive OCR | 40 pure-script line strips/page; no mixed pages; 300 dpi + safe boundaries |
| script classifier | 48×48 patch residual CNN, 4 classes | synthetic val acc 0.996 (dev 1.000 / tam 0.998 / other 0.985 / null 0.986) |
| word-level script separator | 64×256 word CNN, 3 classes | synthetic val acc 0.9996 (dev 0.9987 / tam 1.0000 / other 1.0000) |

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

* 534 single-line crops from `RTS_sb_01`/`RTS_sb_02`, auto-aligned GT.
* 89 lines carry broken alignment targets (tiny slivers with 1000+ char references, discovered
  while building the line data). They are excluded from the fair comparison; the resulting
  **445 CTC-feasible lines** are used for all engine numbers. Both figures are reported (§3.2).
* Composition of the feasible set: 34 Devanagari-only, 394 mixed (Tamil+Devanagari),
  17 Tamil-only lines (raw-script classification; §3.3).

### 2.2 Block set (`RTS_sb_01`)

Five verified text blocks from the first volume. The images are 150 dpi crops of the relevant
text region (headers and rules excluded), upscaled 2× with LANCZOS before the pipeline, i.e.
approximately the 300 dpi scale used in production.

| block | printed lines | non-space GT chars | content |
|---|---|---|---|
| `comm/p0042` | 30 | 1,853 | Tamil commentary with Sanskrit quotations |
| `core/p0043` | 2 | 90 | two-line Devanagari mUlam verse with Tamil frame |
| `comm/p0043` | 28 | 2,132 | dense Tamil commentary with Sanskrit sUtras |
| `core/p0044` | 6 | 385 | Devanagari/Tamil mUlam passage |
| `comm/p0044` | 23 | 1,733 | Tamil commentary with Sanskrit quotations |

Total: **89 printed lines, 6,193 non-space GT characters**. The five blocks are the first five
verified rows of `evaluation/eval_index.jsonl` / `evaluation/split_eval.jsonl`. Their GT is contiguous block
text, **not aligned to the printed lines**, so all block scoring is whitespace-stripped
character CER over the whole block.

The earlier `RTS_5c_*` volumes and their pilot predictions were moved to `data/RTS_5c/` and
`evaluation/RTS_5c/`; they are kept for reference only.

### 2.3 Metrics and conventions

* **Line-level:** CER/WER via `jiwer`, on raw and whitespace-stripped text, identical across all
  engines (`training/scripts/script_cls/eval_lines.py`; per-line dumps under `evaluation/`).
* **Block-level:** whitespace-stripped character CER via
  `rapidfuzz.distance.Levenshtein.distance / len(GT without spaces)`, micro-averaged over blocks
  (i.e. summed edit distance / summed GT characters). Line breaks in the GT are ignored.
* **Oracle script routing:** where an engine must choose a script-specific model, we report
  `best-of` = per-line minimum-CER pick. This is an upper bound for any automatic router (mixed
  lines are unrouteable in principle anyway).
* Line-level and block-level CERs are **not directly comparable**: the units, GT lengths and
  error weighting differ (line GT is per printed line; block GT is a contiguous passage). Compare
  systems only within the same table.
* Tesseract is run in psm 7 on line crops and on script runs/segments; PaddleOCR in `rec` and
  `det+rec` modes; Drive OCR on whole blocks and on composed faux pages.

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

### 3.3 By GT script composition (feasible subset, raw CER)

| script | n | Tesseract | Paddle rec best (oracle) | devTamCharsOCR |
|---|---|---|---|---|
| Devanagari-only | 34 | 0.493 | 0.688 | 0.408 |
| mixed (Tamil+Devanagari) | 394 | 0.478 | 0.740 | **0.285** |
| Tamil-only | 17 | 0.336 | 0.538 | 0.300 |

Oracle picks (all 534): rec `ta` 455 / `hi` 79; det+rec `ta` 390 / `hi` 144. **PaddleOCR never
beats Tesseract in any script bucket.** The crucial row is devTamCharsOCR: its mixed lines are
*slightly easier* than its pure-script lines, so it does not suffer from script mixing — unlike
the pretrained engines. Tesseract's pure-Devanagari bucket is no easier than its mixed bucket
(0.493 vs 0.478), so line-level composition alone does not predict the block-level split gain;
the block experiment (§4) is the controlled test.

### 3.4 Qualitative failure modes

Line-level examples from the eval split (outputs verbatim; CER in parentheses; links are the
actual crops). `linewiseScriptSepDriveOCR` is included for each line; `scriptSepDriveOCR` is not
defined for a single isolated line (it needs a block of runs). The linewise texts below are the
raw variant; block-level primary numbers in §4.4 additionally drop `other` runs.

* **Mixed line `01_p0042_comm_L00`** ([crop](examples/01_p0042_comm_L00.png); Tesseract 0.236,
  devTamCharsOCR 0.036, linewise 0.055)
  * GT: `தன்பற்றில் - தானே தன் संबन्धத்தை உண்டாக்குகையில். இது, 'நின்று`
  * Tesseract: `தன்பற்றில்‌ - தானே தன்‌ த்தை உண்டாக்குகையில்‌. இது, “நின்று தன்‌` — the Sanskrit word is replaced by a Tamil fragment and a tail is hallucinated.
  * PaddleOCR: `Is Mnu, To qunoூsumos Goghk mo mau -ougnms` — Devanagari rendered as Latin garbage, word boundaries lost.
  * devTamCharsOCR: `தன்பற்றில் - தானே தன் संबन्धத்தை உண்டாக்குகையில். இது, நின்று த` — right script and words; only the apostrophe is missing and a `த` is added at the end.
  * linewiseScriptSepDriveOCR: `தன்பற்றில் - தானே தன் संबन्धத்தை உண்டாக்குகையில். இது, 'நின்று தன்` — essentially exact; only the apostrophe glyph differs.
* **Mixed line `01_p0042_comm_L03`** ([crop](examples/01_p0042_comm_L03.png); Tesseract 0.538,
  devTamCharsOCR 0.346, linewise 0.308)
  * GT: `सङ्ग्रहोक्ति. தகவோக்கத்தில் - कृपाधिक्यத்தில். இது 'அருள்`
  * Tesseract: `னான்‌” என்பதின்‌ 898, தகவோக்கத்தில்‌ - कृपाधिक्यं). இது “அருள்‌` — re-routes the line start and substitutes `सङ्ग्रहोक्ति` with digits.
  * devTamCharsOCR: `னான் என்பதின் संग्रहोक्ति, தகவோக்கத்தில் - कपाधिक्यத்தில். இது "ருள்` — prepends `னான் என்பதின்` from the previous line, misreads `कृपा` as `कपा`, and clips the final `அ`.
  * linewiseScriptSepDriveOCR: `னான் என்பதின் संग्रहोक्ति. தகவோக்கத்தில் कृपाधिक्यத்தில். இது அருள்` — nearly exact; the tag digits are gone, but the leading `ஸ`/`ங` shapes remain.
* **Tamil-only line `01_p0042_comm_L12`** ([crop](examples/01_p0042_comm_L12.png); Tesseract
  0.052, devTamCharsOCR 0.034, linewise 0.000)
  * GT: `பாட்டில் சொன்ன இன்பம் முதலான பத்து உபகாரங்களையும் செய்வதற்காகவே.`
  * Tesseract: `பாட்டில்‌ சொன்ன இன்பம்‌ முதலான பத்து உபகாரங்களையும்‌ செய்வதற்காகவே.` — essentially exact.
  * PaddleOCR: `பாட்டலெசான்னஇன்பம்ழதலானபத்தஉபகாரங்கைளயமெசய்வதற்காகேவ.` — all word spaces lost, ligatures corrupted (`சொன்ன`→`ெசானன`).
  * devTamCharsOCR: `ாட்டில் சொன்ன இன்பம் முதலான பத்து உகாரங்களையும் செய்வதற்காகவே.` — near-exact; drops the leading `ப` and one `ப` inside `உபகார`.
  * devTamCharsOCR on the script segments: exact (CER 0.000) — splitting a pure line does not hurt this example.
  * linewiseScriptSepDriveOCR: `பாட்டில் சொன்ன இன்பம் முதலான பத்து உபகாரங்களையும் செய்வதற்காகவே.` — exact.
* **Mixed line `01_p0042_comm_L24`** ([crop](examples/01_p0042_comm_L24.png); Tesseract 0.463,
  devTamCharsOCR 0.019, linewise 0.167)
  * GT: `उक्तिயாலும், स्पष्टமாகக் காண்பிக்கும். தொல்வழியே - अनादिயான`
  * Tesseract: `தாலும்‌, எண்யாலும்‌, மாகக்‌ காண்பிக்கும்‌. தொல்வழியே - யான` — drops the Devanagari words.
  * devTamCharsOCR: `उक्तिயாலும், स्पष्टமாகக் காண்பிக்கும். தொல்வழியே - अनादिயன` — near-perfect (one vowel sign).
  * linewiseScriptSepDriveOCR: `தாலும், उक्तिயாலும், स्पष्टமாகக் காண்பிக்கும். தொல்வழியே  अनादि ।யான` — duplicates the line start, adds a stray danda (`।`) and a double space; the Tamil `உ` survives.
* **Devanagari-only line `01_p0043_core_L00`** ([crop](examples/01_p0043_core_L00.png);
  Tesseract 0.232, devTamCharsOCR 0.130, linewise 0.130)
  * GT: `क्षत्रबन्धुश्च पुण्डरीकश्च पुण्यकृत् । आचार्यवत्तया मुक्तौ तस्मादाचार्यवान्`
  * Tesseract: `“पापिष्ठः क्षनबन्धुर्च पुण्डरीकरच पुण्यकृत्‌ । आचार्यवत्तया मुक्तौ तस्मादाचार्यवान्‌` — prepends a word from the previous printed line and confuses `श`/`र`.
  * devTamCharsOCR: `िष्टः्षबन्धशच पुण्डरीकशच पुण्यकृत् । आचार्यवत्तया मुक्तो तस्मादाचार्यवान्` — loses the line start and `मुक्तौ`→`मुक्तो`.
  * linewiseScriptSepDriveOCR: ` 'पापिष्ठः क्षत्रबन्धुश्च पुण्डरीकश्च पुण्यकृत् । आचार्यवत्तया मुक्तौ तस्मादाचार्यवान्` — almost exact; adds a leading space and an apostrophe, and inherits `"पापिष्ठः` from the previous printed line.
* **Dense mixed line `01_p0043_comm_L05`** ([crop](examples/01_p0043_comm_L05.png); Tesseract
  0.353, devTamCharsOCR 0.147, linewise 0.191)
  * GT: `अतिशयित-पापமுள்ளவனான. क्षत्रबन्धुश्च - இந்த चकारம் आचार्यवत्तै இருந்தாலும்`
  * Tesseract: `எ19-ராமுள்ளவனான. कषत्रबन्धुङ्च- இந்த चकारा आचार्यवत्ते இருந்தாலும்‌ पापिष्ठ-` — substitutes the first word and appends the next line's text.
  * devTamCharsOCR: `ितपापமுள்ளவனான. क्ष्बन्धुश्च இந்த चकारம் आचार्यवत्तै இருந்தாலும் ்` — drops line-start chars, but keeps the script mixture.
  * linewiseScriptSepDriveOCR: `शयित-पापமுள்ளவனான. क्षत्रबन्धुश्च- இந்த चकारம் ம் आचार्यवत्तै இருந்தாலும் पापिष्ठ-` — close; one Tamil `ம்` is inserted and the inherited tail remains.
* **Mixed prose line `01_p0044_core_L01`** ([crop](examples/01_p0044_core_L01.png); Tesseract
  0.526, devTamCharsOCR 0.053, linewise 0.053)
  * GT: `आचार्यवंशம் भगवाனளவுஞ்செல்ல अनुसन्धिக்கவேணுமென்றோதப்பட்டது.`
  * Tesseract: `आचार्य-वंशा *ளனளவுஞ்செல்ல अनुसन्यि 4०/69/0102 079 01८ 10 ८ ॐ.` — the Tamil half collapses into digits/symbols.
  * devTamCharsOCR: `आचार्यंश் भगवाனளவுஞ்செல்ல अनुसन्धिக்கவேணுமென்றுதப்பட்டது.` — near-exact; `वंशம்` and `ஓ` differ.
  * linewiseScriptSepDriveOCR: `आचार्य-वंशம் भगवा'னளவுஞ்செல்ல अनुसन्धि,க்கவேணுமென்றோதப்பட்டது.` — near-exact; one apostrophe is inserted.

Failure modes:

* **Mixed-script lines defeat the pretrained engines.** PaddleOCR's `ta` model renders Devanagari
  as Latin garbage (`Goghk` for `संबन्ध`) and its `hi` model does the reverse; 394/445 feasible lines
  are mixed. Tesseract degrades more gracefully but silently substitutes, drops or duplicates
  Devanagari words.
* **devTamCharsOCR is strongest on natural lines, including mixed ones.** Its remaining errors are
  edge clipping, quote/punctuation, and occasional vowel-sign/word errors on Devanagari-only
  lines — not script confusion.
* **Isolated fragments are a different domain.** On script runs devTamCharsOCR emits nothing for
  27% of runs and scores 0.812; even on larger script segments it scores 0.599 (§4).
* **Dense linewise pages are now the best variant.** Packing 40 pure-script line strips per page
  reduces calls and context loss; the remaining errors are Drive tag/space slips (0.112 vs
  0.114 page-batched).
* **Same-script lines are easy for Tesseract and devTamCharsOCR**, but PaddleOCR still loses word
  spaces and confuses similar Tamil ligatures.
* The Devanagari PaddleOCR model (`hi`, trained on Hindi) is the weakest: 0.91 CER alone.

### 3.5 Takeaways

* On clean line crops, devTamCharsOCR is **1.6× better than Tesseract** and **2.5× better than
  PaddleOCR**; Tesseract remains the best zero-shot engine.
* Script mixing is a major error source for the pretrained engines (especially PaddleOCR), but
  **not for devTamCharsOCR** (pure-script lines are not easier for it). This distinction decides
  §4.5.
* devTamCharsOCR transfers to block lines on its home regime (0.238 block CER vs 0.292 line CER),
  but it collapses when the same model is applied to the short runs/segments that faux pages
  need (0.812/0.599). That is a training-domain gap, not a script gap.

## 4. Block-level evaluation

### 4.1 Why block lines are harder

Printed blocks mix scripts not only between lines but *inside* a line: a Devanagari mUlam quote
with Tamil commentary labels, Tamil verbal suffixes on Sanskrit stems, parentheses and item
numbers. Whole-line engines (ours included) then apply a model trained on predominantly pure
script to a unit containing both scripts; whole-block OCR engines reorder, drop or garble lines.
The five `RTS_sb_01` blocks contain 89 printed lines, 898 classified runs and 593 script
segments.

### 4.2 Line segmentation audit

Before trusting the pipeline we audited `segment_lines` on the five upscaled block images
(`training/scripts/script_cls/audit_lines.py`, `audit_page` applied to the block crops):

* All Tesseract word-boxes fall inside their segment (30/30, 2/2, 28/28, 6/6, 23/23); no lines
  are merged or split.
* Ink coverage is **99.996%** on every block (missed ink 2–33 px per block, i.e. isolated specks).
* The two-column heuristic flagged two segments in `comm/p0043` and one in `comm/p0044`; these
  are wide inter-word gaps in dense commentary, not column layouts (the volume is single-column).
* The audit is not a bottleneck; the rest of this section uses line segmentation as-is.

### 4.3 Script-run splitting pipeline (`training/scripts/script_cls/pipeline.py`)

Inspired by `doc_curation.pdf.drive_ocr.split_and_ocr_all` (which boxes lines by script before
OCR), the pipeline boxes *runs* and assembles the page from script-pure pieces:

1. Load the block image; upscale 2× so a 150 dpi crop is at approximately 300 dpi.
2. Segment lines (above).
3. Word segmentation by horizontal ink projection (gap ≥ 10 px).
4. Sliding 48×48 patch classifier, stride 8; per-patch labels `dev/tam/other/null`; a run keeps
   a patch's label only above 0.45 confidence (else `null`).
5. Merge patches into runs; drop runs < 20 px wide; split words at script transitions **only at ink-free columns**; if none exists, assign the whole word to its dominant script. Outer runs always use full word bounds.
6. Compose one faux page per script: each complete line/script segment is a strip, scaled to
   90 px height (≥0.6), 120 px row pitch, width cap 1780 px, JPEG q92; up to 40 strips per page.
7. OCR each faux page with Google Drive. Every strip carries a printed `#TNN` tag to its left;
   the parser maps tags → segments, so dropped or merged OCR lines cannot shift the page. Continuation
   lines bind to the previous tag; markdown escapes and trailing base64 image blobs are stripped.
8. Missing tags trigger one retry of the affected chunk (no per-segment fallback); segments that
   remain unreadable are left empty. Reassemble the block with per-segment `pre_space`.

In the five-block + seven-example run, 148 faux pages were composed and 74 of the 222 Drive
pages (33%) needed a retry for missing tags. After retries, 26/898 runs (2.9%) remain empty. 43/898 runs (4.8%)
contain a Latin character, mostly leaked tag/escape characters.

**Script classifier:** synthetic 48×48 patches rendered from `data/texts/{core,commentary}` with
the Devanagari/Tamil font sets (plus noise/blank for `other`/`null`), residual CNN (0.84M
parameters, 20k steps on a Kaggle T4). Word-disjoint validation accuracy **0.996** (dev 1.000 /
tam 0.998 / other 0.985 / null 0.986). A short distillation pass on unlabelled RTS_sb_01
p0030–p0041 patches with synthetic replay keeps synthetic accuracy at 0.996 and reaches 95.6%
agreement with the earlier model on held-out real pages. The block OCR numbers in §4 use this
classifier. Faux pages are now pure-script: each page packs up to 40 complete line/script strips,
with multiple printed lines per page; `other` segments are not composed. On the five eval
blocks, line-level script presence is 64/69 versus 65/69 for the earlier model.

`linewiseScriptSepDriveOCR` uses the same steps and the same pure-script line-strip pages,
with line order preserved across a page.

### 4.4 Results

Whitespace-stripped CER per block; micro is over all five blocks (6,193 GT chars).

| system | comm p0042 | core p0043 | comm p0043 | core p0044 | comm p0044 | **micro** |
|---|---|---|---|---|---|---|
| **scriptSepDriveOCR (v2)** | **0.079** | **0.033** | 0.226 | **0.052** | **0.034** | 0.114 |
| linewiseScriptSepDriveOCR | **0.079** | **0.033** | **0.218** | **0.052** | **0.034** | **0.112** |
| Tesseract on whole lines (`tam+san`) | 0.241 | 0.133 | 0.445 | 0.377 | 0.362 | 0.352 |
| Tesseract on script runs (per-script models) | 0.155 | 0.133 | 0.422 | 0.120 | 0.155 | 0.244 |
| **Tesseract on script segments (per-script models)** | 0.144 | 0.111 | 0.399 | 0.125 | 0.132 | **0.227** |
| Tesseract on script segments (mixed `tam+san` model, control) | 0.180 | 0.144 | 0.487 | 0.278 | 0.239 | 0.308 |
| devTamCharsOCR on whole lines | 0.103 | 0.233 | 0.443 | 0.200 | 0.140 | 0.238 |
| devTamCharsOCR on script runs | 0.812 | 0.800 | 0.871 | 0.691 | 0.767 | 0.812 |
| devTamCharsOCR on script segments | 0.470 | 0.256 | 0.782 | 0.449 | 0.563 | 0.599 |
| Drive OCR on the whole block (no split) | 0.277 | 0.222 | 0.551 | 0.540 | 0.450 | 0.435 |

Each faux page contains only Tamil or only Devanagari strips. A strip is a complete
line/script segment (not a single word); up to 40 strips are packed per page, with multiple
printed lines per page. The `other` segments are not composed.

linewiseScriptSepDriveOCR is the best micro result. It wins `comm/p0043` (0.218 vs 0.226) and
ties page-batched separation on the other four blocks. The split is worth **3.9×** against
unsplit Drive (0.435), **2.1×** against the best devTamCharsOCR configuration (0.238), and
**2.0×** against the best local script-split Tesseract configuration (0.227). The unsplit Drive
output is erratic: 0.031 on `core/p0044` but 0.551 on `comm/p0043`, where dense Tamil commentary
makes it reorder and drop lines.

The most important pattern in the table is the **opposite response to splitting**:

* Tesseract improves monotonically as units get more script-pure and better sized:
  whole lines 0.352 → runs 0.244 → segments 0.227.
* devTamCharsOCR collapses on fragments: whole lines 0.238, runs 0.812, segments 0.599.

Both are "local" models; the difference is their training regime. Tesseract's per-script models
were trained on exactly the kind of short, script-pure crop that a faux page contains, while
devTamCharsOCR was trained on complete printed lines. §4.5 isolates how much of the Tesseract
gain comes from segmentation versus per-script routing.

### 4.5 Answering the two open questions

**Q1 — does the script-split gain hold for a local per-script recognizer, or is it Drive-specific?**

It holds, but Drive still wins. Using the pipeline's own segmentation and a local Tesseract
(`training/scripts/script_cls/control_seg.py` reproduces the mixed-model control):

| configuration | micro CER |
|---|---|
| whole lines, mixed `tam+san` model | 0.352 |
| script segments, mixed `tam+san` model (segmentation control) | 0.308 |
| script segments, per-script `san`/`tam` models | **0.227** |

So segmentation alone (shorter units, no cross-script model confusion) recovers 0.044 CER, and
per-script model routing recovers another 0.081. The per-script split improves **all five
blocks** (0.241→0.144, 0.133→0.111, 0.445→0.399, 0.377→0.125, 0.362→0.132). This is a genuine
local per-script gain, not a Drive artefact. But the best local configuration (0.227) is still
2.0× worse than Drive on the same blocks (0.114), and one block (`comm/p0043`, 0.399) remains
far from usable. Drive's advantage is recognition quality on the fragments, not the script split.

**Q2 — can a script-pure variant of devTamCharsOCR close the gap on faux pages?**

Not as an input variant, and the evidence argues against a per-script retrain as the next step.
Applied to script-pure segments, devTamCharsOCR scores **0.599**, much worse than the same model
on whole lines (**0.238**) and 5.2× worse than Drive on the same blocks (**0.114**). On natural
pure-script lines the model is not better either: 0.408 on Devanagari-only and 0.300 on
Tamil-only lines versus **0.285** on mixed lines (§3.3). Script purity is therefore not what
devTamCharsOCR lacks.

What it lacks is the **short-fragment regime**:

* 27% of script runs get an empty devTamCharsOCR output; short 1–4-akshara crops are outside the
  line-crop training distribution.
* Segment crops are larger and improve to 0.599, but still lose the full-line context the model
  was trained on (and Drive gets 0.114 on the same blocks).
* A script-pure *model* trained on the same line-crop distribution would inherit the same
  short-fragment weakness. Closing the gap requires training on faux-page-style fragments (or
  fine-tuning a pretrained recognizer such as PP-OCRv4), not merely restricting training data to
  one script.

### 4.6 Combining devTamCharsOCR with scriptSepDriveOCR (negative result)

The two systems fail differently, so we tested whether mixing them lowers CER. All variants
reuse the same segment segmentation and block assembly and differ only in the per-segment text source
(`training/scripts/script_cls/combine_group.py`; segment texts in
`evaluation/sb_eval/runs_*.jsonl`; scores in `evaluation/sb_eval/combine_group.json`).

| variant | micro CER |
|---|---|
| **scriptSepDriveOCR (baseline)** | **0.114** |
| devTamCharsOCR on script segments (all segments) | 0.812 |
| devTam replaces Drive only for Devanagari segments | 0.423 |
| devTam replaces Drive only for Tamil segments | 0.453 |
| devTam fills only empty Drive segments | 0.116 |
| segment-level hybrid (devTam if Drive text is empty or suspicious) | 0.131 |

* **devTamCharsOCR does not transfer to isolated segments.** It is a full-line model; on tight
  1–4-akshara crops it loses context, emits nothing for 27% of runs, and decodes at CER 0.68–0.86
  depending on the block.
* **Repair rules do not help.** Filling empty Drive runs with devTam changes the micro CER from
  0.1143 to 0.1156 (the empty runs are sub-glyph stubs that neither engine can read).
* **The suspicious-text hybrid hurts slightly** (0.131): runs whose Drive text contains a Latin
  character are not reliably better when replaced by devTam.
* **Script-conditional mixes are much worse** (0.423 when devTam handles only Devanagari segments,
  0.453 when only Tamil segments), because the segment-level devTam texts dominate the error.
* **There is no complementarity at these units.** Because block GT is not line-aligned, we did
  not compute a per-line oracle; the segment-level results are already decisive.

Combining the current two systems therefore does **not** reduce CER. The productive combination
is to train a recognizer in the same regime as Drive — faux-page fragments — rather than mixing
a full-line model with segment-level inputs.

### 4.7 Qualitative block example

`core/p0043` ([block image](../data/RTS_sb_01/images/core/p0043.jpg), 2 lines, 90 GT chars;
CERs: devTamCharsOCR 0.233, Tesseract segments 0.111, scriptSepDriveOCR 0.033,
linewise 0.033):

* GT: `"पापिष्ठः क्षत्रबन्धुश्च पुण्डरीकश्च पुण्यकृत् । आचार्यवत्तया मुक्तौ तस्मादाचार्यवान् भवेत् ॥" என்று`
* devTamCharsOCR (whole lines): `पापटः्षवन्धुच पुण्डरीकश्च पुण्यकृत् । आचार्यवत्तया मुक्तो तस्मादाचार्यवान्` / `ன்று` — loses the opening quote, the first two words and the final `भवेत् ॥"`.
* Tesseract (script segments): `"पापिष्ठः क्षत्रबन्धुरुच पुण्डरीकङच पुण्यकृत्‌ । आचार्यवत्तया मुक्तौ तस्मादाचार्यवान्‌` / `भवेत्‌்‌” என்று` — all words present; `श`→`र`/`ङ`, `॥"`→`”`.
* scriptSepDriveOCR: `“पापिष्ठः क्षत्रबन्धुश्च पुण्डरीकश्च पुण्यकृत् । आचार्यवत्तया मुक्तौ तस्मादाचार्यवान्` / `भवेत् ॥'' என்று` — almost exact; only the quote glyph/spacing and closing `"` differ.
* linewiseScriptSepDriveOCR: `“पापिष्ठः क्षत्रबन्धुश्च पुण्डरीकश्च पुण्यकृत् । आचार्यवत्तया मुक्तौ तस्मादाचार्यवान्` / `भवेत् ॥'' என்று` — same output on this block.

The dense `comm/p0043` block ([image](../data/RTS_sb_01/images/comm/p0043.jpg), CER 0.226) shows
the other failure mode: the words are mostly right, but Drive leaks tag/escape characters into
the text (`நிர்பந்த'முண்டோ?`, `Sh uபில்`, `uயாதோ?`), and the residual errors concentrate in
Tamil vowel signs and word spacing. This block alone accounts for most of the gap between 0.114
and a production-ready figure.

### 4.8 Word-level script separation and cut-strip fix

The original linewise faux pages could start/end inside a word (`घन`, `रि ध्य`, ...; v1 p17),
and on the v1 index page many Devanagari headings were misrouted to the Tamil page. The causes
were (i) patch-level splitting inside a word at 150 dpi and (ii) run boundaries placed at the
midpoint between patch centres rather than at an ink-free column.

Changes in `pipeline.py`:

1. `word_segments` remains the word separator: it yields complete word-image boxes by horizontal
   ink projection.
2. `classify_word` now keeps outer runs at the full word bounds and snaps every internal
   script boundary to the nearest column with no ink; if no safe column exists, the whole word
   gets its dominant script instead of being cut.
3. `classify_word_hybrid` optionally uses a word-level CNN script separator for wide, pure-script
   words, and falls back to the safe patch runs for short tokens and genuinely mixed-script
   words.
4. Default page processing is 300 dpi; the Kaggle driver had used 150 dpi.

The word-level script separator is trained on Kaggle (`vishvasvasuki/word-script-cls-train`,
3,000 steps, synthetic words at 24–48 px text height): **val acc 0.9996** (dev 0.9987, tam
1.0000, other 1.0000; 20k train / 5k word-disjoint val). It is stored at
`training/local/word_cls.pt`; pass `--word-classifier training/local/word_cls.pt`.

Real-page check on volume 1:

* p14 (विषय सूचिका): old 150 dpi linewise had 18 Tamil faux segments, including misrouted
  Devanagari headings, and cut headings. New 300 dpi + safe boundaries + word model gives 17
  complete Devanagari headings and 3 Tamil segments (the Tamil heading only).
* p17 (mixed shlokas/commentary): old 150 dpi dev strips began mid-word (`घन`, `रि ध्य`, ...);
  the new 300 dpi dev page contains 38 complete line/phrase strips.
* p40: mixed-script words such as `आविष्करि` + Tamil `த்தும்` are deliberately split at the
  script boundary; each side is now a complete glyph run with no partial akshara.

An improved v1 linewise candidate is generated to
`/media/vvasuki/vData/text/granthasangrahaH/AgamaH/vaiShNavaH/shrIvaiShNavaH/venkaTanAtha/rts/5-comm/faux-pages-linewise-fixed/ra-tra-sA_5-commentaries_1_faux.pdf`
(2,407 faux pages, 1.00 GB; the old 150 dpi v1 PDF had 3,463 pages). This candidate still has residual script-routing errors on real pages (see below).

**Residual real-page failures (2026-10-22).** The word-level model is trained on synthetic words; on this book’s Tamil/Devanagari font it still confuses some visually similar glyphs. For example, on vol. 1 p20 the Tamil word `சர்வ` and parts of Tamil commentary are occasionally routed to the Devanagari page, and inline Devanagari words can be moved the other way. A line-level model trained on real patch labels from the 12 `manipravala-script-cls-real` pages reached only ~0.71 line accuracy with 103 usable lines, so it is not used by default. The safe ink-free boundaries fix cut glyphs, but near-perfect script routing needs **real font-specific word/line labels** (roughly a few hundred crops from p14, p17, p20 and similar pages) and a short fine-tune. Until then the default pipeline should use 300 dpi + safe boundaries without the experimental word/line classifier; the included `word_cls.pt` is a synthetic-domain baseline.

**Mixed-word script-separator data (2026-10-22).** `data/script-separator/train.jsonl`
contains 28,198 mixed Tamil/Devanagari words from 1,070 `RTS_sb_02` pages (comm+core). Each record stores the source image path, a box, the GT token, and the script runs with
approximate x-fractions; no crop images are stored. Per-page GT token sequences were aligned
to image word boxes by a monotone DP over character width (`GAP_WORD=4`, with skip transitions
for headers/rules). A verification overlay was written only under `/tmp/opencode/`.

A 4-class word script separator (`dev/tam/other/mixed`) was retrained on these real
mixed words plus synthetic pure/mixed words: synthetic val **0.997** and real mixed
val **1.000** (57 real samples, 12 held out; as always, this small real set is
optimistic). It is stored at `training/local/script_sep.pt`; training code is
`training/scripts/script_cls/train_script_sep.py`. The model currently **detects** a
mixed word and routes it to the safe patch split; it does **not** yet predict the exact
column boundary inside the word. A column-level separator is the next step for that.

A first column-level separator was trained from this JSONL (`training/local/column_sep.pt`,
`training/scripts/script_cls/train_column_sep.py`, Kaggle `vishvasvasuki/rts-column-sep-train`).
Held-out real column accuracy is **0.715** with boundary MAE **0.087** of word width (24,890 component-gap-refined entries). It is wired
as the optional `--column-classifier` path (`classify_word_columns` in `pipeline.py`), but it is
**not enabled by default**: the run labels are derived from GT character counts, so boundaries are
approximate and the first model fragments some words. The Data+code are ready for a better model
with exact glyph boundaries; more examples alone do not remove this error floor because the run targets are still approximate.

**Word-level 4-class retraining (2026-10-23).** The 4-class word separator (`dev/tam/other/mixed`)
was retrained on the 24,890 mixed-word boxes plus synthetic pure/mixed words: synthetic val **0.998**,
real mixed val **1.000** (mixed class 0.993). It is stored at `training/local/script_sep.pt`.
However, the real data contains **only mixed words**; there are no real pure-word negatives. When wired
with connected-component splitting, it misclassifies pure Tamil components as Devanagari (e.g. page 20
Tamil words appear on the Devanagari page). The model is therefore **not enabled by default**. The next
required dataset is `train_words.jsonl` with all GT words labeled `dev/tam/other/mixed`, not only mixed
words.

**All-word dataset and 4-class retrain (2026-10-23).** Generated
`data/script-separator/train_words.jsonl` with **51,210** GT words from 1,070 pages, labeled
`dev/tam/mixed/other` (all mixed words plus bounded pure/other samples per page), still without
stored crops. Retrained the 4-class word model: real per-class dev **0.999**, tam **1.000**,
other **0.993**, mixed **0.986**; synthetic val **0.996**. The model is stored at
`training/local/script_sep.pt`.

This exposes the remaining blocker: **word/component segmentation**, not classifier capacity.
The classifier is word-image based; feeding connected components (aksharas) fragments even accurate
Tamil words into Devanagari strips. The column model (real column acc **0.715**) is limited by
approximate run boundaries. Near-perfect end-to-end routing needs either gold word boxes or a
component/akshara-level model trained with exact glyph boundaries; more whole-word examples do not
solve that.

**Boundary segmenter experiment (2026-10-23).** A heatmap boundary model was trained on **23,025**
two-run `dev→tam` mixed words from the refined JSONL (Kaggle `vishvasvasuki/rts-boundary-sep-train`).
On boundaries derived from the original RTS image boxes using shiro-rekhā/glyph-gap detection (18,175 high-confidence mixed words), the held-out model reaches boundary MAE **0.116**, boundary precision@5% **0.643**, segment precision **0.916**, recall **0.906**. The column model is better on boundary MAE
(**0.087**) but still not near-perfect. Both are optional and not enabled by default. The limiting
factor remains word-box/run-label precision, not the number of examples.

**Rendered Noto boundary segmenter (2026-10-23).** Using the GT mixed words, I rendered each
Devanagari prefix with Noto Sans Devanagari and each Tamil suffix with Noto Sans Tamil, with
HarfBuzz-derived advances/glyph-cluster counts defining the exact boundary. A boundary heatmap
trained on 23k rendered examples reaches held-out **MAE 0.0002**, boundary precision@5% **1.000**,
segment precision **0.999**, recall **0.999** (`training/local/boundary_sep.pt`,
`train_render_boundary.py`). It is wired as optional `--boundary-classifier`.

On the scanned book pages this rendered-domain model does **not** transfer reliably: the source
font differs, and several `RTS_sb_02` word boxes are merged/misaligned (e.g. GT
`तत्त्वहितविषयமான` maps to a Tamil crop). A real-domain fine-tune needs correct word boxes and
source-font boundary labels.

## 5. Verdict and recommendations

* **For line crops** (clean scans): devTamCharsOCR (CER 0.29) is the best available; Tesseract
  (0.48) is the best zero-shot engine and the only sensible pretrained fallback. **PaddleOCR
  pretrained models are not usable** (0.71–0.73), and the Indic-OCR repo adds no accuracy of its
  own (no weights, stub fine-tuning, English-only TrOCR, preprocessing that changes nothing).
* **For blocks/pages:** feed neither whole lines nor whole blocks to a single-script engine.
  Pure-script line strips are the winning pattern: 0.114 (scriptSep) and **0.112** (linewise)
  vs 0.238 for devTamCharsOCR on whole lines and 0.435 for unsplit Drive OCR. The dense,
  multi-line page layout is what makes the linewise variant viable.
* **Q1 (local per-script backend): yes, the gain is real.** Script-segmenting and using
  per-script Tesseract models cuts the local CER from 0.352 to 0.227 (36% relative), and improves
  every block. It is not Drive-specific. But it is not parity: Drive remains 2.0× better on the
  same blocks.
* **Q2 (script-pure devTamCharsOCR): no.** Script-pure segments score 0.599 vs 0.238 for whole
  lines, and pure natural lines are not easier than mixed ones. The model needs fragment-regime
  training, not script-pure training.
* **Do not combine the two systems as-is:** segment-level mixing of devTamCharsOCR with
  scriptSepDriveOCR gives no gain and often hurts (§4.6) — they are not complementary at these
  units.
* **Fine-tuning PaddleOCR's PP-OCRv4** on our 16.5k feasible train lines remains a plausible
  follow-up (it imports Devanagari/Tamil glyph knowledge our from-scratch model lacks), but it is
  now a comparison, not a necessity: the from-scratch model already reaches 0.292 on lines.
* **Production path:** the accuracy blocker is narrowed but not solved locally. Drive OCR latency
  dominates (minutes per block/page, arbitrary stalls). Two mitigations, in order of value:
  (a) train a **local fragment/run recognizer** (fine-tuned PP-OCRv4 or a devTamCharsOCR variant
  trained on faux-page-style segments) to replace Drive; (b) reduce calls by merging tiny
  fragments before batching (raise/word-adaptive gap). The current best local split (0.227) is
  not yet good enough to replace Drive (0.114).
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
| `evaluation/sb_eval/metrics.json` | per-block and per-example CER for all variants, micro averages |
| `evaluation/sb_eval/whole_scores.json` | CER of the unsplit whole-block Drive baseline |
| `evaluation/sb_eval/{block}.pred_{variant}.txt` | block predictions (`tess_line`, `tess_run`, `tess_seg`, `dev_line`, `dev_run`, `dev_seg`, `drive_sep`, `drive_line`, `drive_whole`) |
| `evaluation/sb_eval/runs_{block}.jsonl` | per-run boxes, scripts, Drive/devTam/Tesseract texts (reusable without re-OCR) |
| `evaluation/sb_eval/example_{line}.json` | qualitative line crops: GT, per-variant predictions, CER |
| `evaluation/examples/{line}.{png,gt.txt}` | the seven qualitative crops and their GT (linked from §3.4) |
| `evaluation/line_eval_paddleocr.jsonl` | per-line rec-mode predictions (`pred_ta`, `pred_hi`, oracle `pred`) |
| `evaluation/line_eval_paddleocr.jsonl.summary.json` | rec-mode aggregate metrics |
| `evaluation/line_eval_paddleocr_det.jsonl(+.summary.json)` | det+rec-mode predictions |
| `evaluation/line_eval_paddleocr_pp.jsonl(+.summary.json)` | rec-mode with Indic-OCR preprocessing |
| `evaluation/line_eval_baseline.jsonl` | Tesseract baseline (534 lines) |
| `evaluation/line_eval_kaggle_v10_best.jsonl` | devTamCharsOCR (v10) per-line predictions (0.292 CER) |
| `evaluation/line_eval_kaggle_v13_best.jsonl` | devTamCharsOCR (v13 repeat) per-line predictions |
| `training/scripts/script_cls/eval_sb.py` | block/example harness (all variants, segmentation, faux pages, scoring) |
| `training/scripts/script_cls/audit_lines.py` | line-segmentation audit |
| `training/scripts/script_cls/pipeline.py` | production block/page pipeline (word separator + safe script partitions + optional word classifier) |
| `training/scripts/script_cls/train_word_cls.py` | Kaggle-ready word-level script separator training (synthetic words, 24–48 px) |
| `training/scripts/script_cls/drive_ocr_images.py` | Drive OCR helper (skips existing outputs) |
| `training/scripts/script_cls/combine_group.py` | segment-level combination variants for the `RTS_sb` blocks (§4.6) |
| `training/scripts/script_cls/control_seg.py` | mixed-model segmentation control for the Q1 decomposition (§4.5) |
| `evaluation/sb_eval/combine_group.json` | segment-level combination-variant CERs per block and micro |
| `evaluation/sb_eval/tess_seg_mixed.json` | segmentation control: mixed Tesseract model on script segments |
| `training/scripts/script_cls/combine_eval.py` | earlier PDF-page combination script (pilot artifacts in `evaluation/RTS_5c/`) |
| `data/RTS_sb_01/{images,gt}/` | block crops (`comm/`, `core/`) and block GT |
| `evaluation/eval_index.jsonl`, `evaluation/split_eval.jsonl` | canonical verified blocks and train/eval split |
| `data/RTS_5c/`, `evaluation/RTS_5c/` | archived `RTS_5c` volumes, pilot predictions and combination runs (reference only) |
| `training/local/kaggle_v10_best.pt` | devTamCharsOCR best checkpoint (kraken-spec, 40 ep, fp32, no aug) |
| `training/local/kaggle_v13_best.pt` | repeat-run checkpoint (same config) |
| `training/local/script_cls.pt` | script-patch classifier (48×48, dev/tam/other/null; synthetic val_acc 0.996) |
| `training/local/word_cls.pt` | word-level script separator (3-class; synthetic val_acc 0.9996) |
| `training/scripts/script_cls/pipeline.py` | word separator + safe script partitions + optional word classifier |
| `training/local/script_cls_metrics.json` | script-classifier validation metrics and confusion matrices |
| `training/scripts/indic-ocr/` | `setup_env.sh`, `eval_paddleocr.py`, `compare_with_baseline.py` |
| `/tmp/opencode/sb_eval_v2/faux/` | composed faux pages for the re-run (ephemeral) |

### 7.1 Model locations

| model | stored at | notes |
|---|---|---|
| script classifier (current) | `training/local/script_cls.pt` | 48×48, 4-class residual CNN; 3.3 MB; synthetic val 0.996 |
| script-classifier training metrics | `training/local/script_cls_metrics.json` | synthetic and real-page validation, confusion matrices |
| script-classifier training code | `training/scripts/script_cls/train_classifier.py` | Kaggle-ready; corpus/fonts under `/kaggle/input` |
| script classifier (Kaggle) | `vishvasvasuki/manipravala-script-cls` | training kernel; datasets `manipravala-script-cls-data` and `manipravala-script-cls-real` |
| word-level script separator (Kaggle) | `vishvasvasuki/word-script-cls-train` | 3,000-step CNN; datasets `manipravala-script-cls-data`, `rts-ocr-code`; val 0.9996 |
| script classifier (real adaptation) | `vishvasvasuki/manipravala-script-cls-ft-replay` | distillation + synthetic replay; checkpoint is the shipped `script_cls.pt` |
| devTamCharsOCR v10 | `training/local/kaggle_v10_best.pt` | kraken-VGSL spec, 40 epochs, fp32, no augmentation |
| devTamCharsOCR v13 | `training/local/kaggle_v13_best.pt` | repeat run, same configuration |
| Tesseract models | `/tmp/tess-root/usr/share/tesseract-ocr/5/tessdata/` | `tam.traineddata`, `san.traineddata`; ephemeral unpack, not in the repo |
| PaddleOCR models | `~/.paddleocr/whl/{det,rec,cls}/...` | `Multilingual_PP-OCRv3_det`, `ta_PP-OCRv4_rec`, `devanagari_PP-OCRv4_rec`; downloaded, not vendored |
| Google Drive OCR | none local | OAuth client at `vvasuki-git/sysconf/.../ocr_app_oauth.json` |

## 8. Caveats and open questions

* **Scope:** five blocks from vol. 01 (89 printed lines, 6,193 non-space GT chars) and 445 line
  crops from vol. 01–02. The micro-average ranking (scriptSepDriveOCR, linewise, local
  splits) holds, but individual blocks differ: linewise wins only the dense `comm/p0043`, and
  absolute CERs will move with block boundaries and GT precision. `comm/p0043` is the hardest and drives much of the residual.
* **Block GT is not line-aligned:** it is contiguous text for the crop, so block CER cannot be
  compared with the 445-line CER, and per-line diagnostics on blocks are approximate.
* **Runtime:** the dense-layout re-run composed 56 faux pages and needed 21 retry pages
  (77 Drive calls), taking ~11 min wall clock. The 40-strip pages cut Drive calls by 3× versus
  the old layout; retries (27% of calls) remain the main cost.
* **Page layout:** each faux page is one script only and packs up to 40 complete line/script
  strips; multiple printed lines share a page. `other` segments are omitted. This replaced the
  old one-run-per-row column pages.
* **Drive API behavior is erratic:** 0.031 on `core/p0044` vs 0.551 on `comm/p0043` for the same
  whole-block call, with occasional multi-minute stalls. Script splitting wins on every block
  regardless, but single-block whole-region numbers mean little.
* **Tag/markdown pollution:** ~5% of runs contain Latin characters, mostly leaked `#TNN` tags or
  markdown escapes (`hd`, `Sh`, `u`, digits). The parser strips common escapes, but a stricter
  normalizer is needed before the output is usable as an e-text.
* **Q1 and Q2 are answered above** (§4.5). Remaining questions: (a) how much does word-gap tuning
  recover the residual spacing errors? (b) more blocks/volumes, especially two-column layouts
  (none seen so far); (c) can a recognizer trained on faux-page fragments (rather than line
  crops) approach Drive's 0.114 while running locally? (d) how far does a lightly fine-tuned
  PP-OCRv4 get on the same fragment regime?
