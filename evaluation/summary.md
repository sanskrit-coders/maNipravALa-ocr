# OCR evaluation for mixed Tamil–Devanagari (manipravala) print

This report evaluates OCR engines and page pipelines for **sArabodhinI vol. 01–02** — a
manipravala print in which Tamil and Devanagari alternate *inside words and lines* (Tamil suffixes
on Sanskrit stems, Devanagari commentary labels, quoted mUlam). Two test beds are used: a held-out
**line** split of 534 crops from the same volumes, and **five hand-verified block excerpts**
(89 printed lines, 6,193 non-space GT characters) from `RTS_sb_01`. Every number below comes from
those `RTS_sb_*` sets; the earlier `RTS_5c` pilot is archived and is not used for evaluation.

Candidates: **Tesseract** (`tam+san`), **PaddleOCR PP-OCRv4** pretrained models (the engine behind
the Indic-OCR/BytesByJay repo), **Google Drive** document OCR, **devTamCharsOCR** (our from-scratch line recognizer),
**scriptSepDriveOCR** (a script-run splitting page pipeline that reflows script-pure runs onto
faux pages before OCR), and **linewiseScriptSepDriveOCR** (the same pipeline with one faux page
per printed line).

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
| **scriptSepDriveOCR (v2)** | — | **0.127** | best block output; 3.4× better than whole-block Drive |
| linewiseScriptSepDriveOCR | — | 0.143 | per-line faux pages; worse than page-batched script runs on 4/5 blocks |

Line and block CERs use different GT units and are **not comparable** (see §2.3). Six findings
dominate everything below: (1) **script mixing is a major failure mode for pretrained engines** —
Tesseract and PaddleOCR substitute or drop Devanagari in mixed units, and Paddle is clearly worse
on mixed than on either pure bucket; (2) pretrained PaddleOCR transfer is poor for this material; (3) devTamCharsOCR is the best single-line
recognizer and even improves slightly on block lines, but it collapses on the short fragments
that faux pages require; (4) **scriptSepDriveOCR** (splitting block lines by script and
assembling the block from script-pure runs) beats the best alternative by 1.9× in the block
micro average (3.4× vs unsplit Drive), while per-line batching
 (linewiseScriptSepDriveOCR, 0.143) is worse on 4/5 blocks, so script runs should be
batched per block/page, not per line; (5) the script-split gain **does hold for a local
per-script recognizer** (Tesseract: 0.352 → 0.227), so it is not Drive-specific, although Drive
keeps a 1.8× lead; (6) a **script-pure devTamCharsOCR input variant does not close the gap**
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
  reflows script-pure runs onto faux pages, OCRs those with Google Drive, and reassembles the page.
* **linewiseScriptSepDriveOCR** — the same pipeline with line-wise batching: line segmentation →
  script separation within each line → one faux page per printed line → Drive OCR. It isolates
  the effect of the OCR batching unit from the script separation itself (§4.4).

| engine | version / model | how it was run |
|---|---|---|
| Tesseract | 5.x, `tam+san`, `san`, `tam` traineddata | psm 7 on line/run/segment crops |
| PaddleOCR | 2.7.3 / 2.6.2 (CPU, 6 threads) | PP-OCRv4 `ta` and `devanagari` recognizers, PP-OCRv3 multilingual detector; `rec` (line crops) and `det+rec` (`.ocr(img, cls=True)`) |
| Google Drive OCR | cloud document OCR (OAuth client) | whole text block, and per-script faux pages (`drive_ocr_images.py`) |
| devTamCharsOCR | kraken-VGSL spec, torch replica, v10/v13 | CTC greedy decode on line/run/segment crops (§6) |
| scriptSepDriveOCR | script classifier + faux pages + Drive OCR | block/page, 150 dpi crop upscaled 2×; 20 runs per faux page |
| linewiseScriptSepDriveOCR | script classifier + per-line faux pages + Drive OCR | one faux page per printed line (`--linewise`) |
| script classifier | 48×48 patch residual CNN, 4 classes | synthetic val acc 0.996 (dev 1.000 / tam 0.998 / other 0.985 / null 0.986) |

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
  devTamCharsOCR 0.036, linewise 0.109)
  * GT: `தன்பற்றில் - தானே தன் संबन्धத்தை உண்டாக்குகையில். இது, 'நின்று`
  * Tesseract: `தன்பற்றில்‌ - தானே தன்‌ த்தை உண்டாக்குகையில்‌. இது, “நின்று தன்‌` — the Sanskrit word is replaced by a Tamil fragment and a tail is hallucinated.
  * PaddleOCR: `Is Mnu, To qunoூsumos Goghk mo mau -ougnms` — Devanagari rendered as Latin garbage, word boundaries lost.
  * devTamCharsOCR: `தன்பற்றில் - தானே தன் संबन्धத்தை உண்டாக்குகையில். இது, நின்று த` — right script and words; only the apostrophe is missing and a `த` is added at the end.
  * linewiseScriptSepDriveOCR: `தன்பற்றில் தானே தன் संबन्धத்தை உண்டாக்குகையில் இது, ‘நின்று தன்` — close, but drops the hyphen and the comma spacing.
* **Mixed line `01_p0042_comm_L03`** ([crop](examples/01_p0042_comm_L03.png); Tesseract 0.538,
  devTamCharsOCR 0.346, linewise 0.327)
  * GT: `सङ्ग्रहोक्ति. தகவோக்கத்தில் - कृपाधिक्यத்தில். இது 'அருள்`
  * Tesseract: `னான்‌” என்பதின்‌ 898, தகவோக்கத்தில்‌ - कृपाधिक्यं). இது “அருள்‌` — re-routes the line start and substitutes `सङ्ग्रहोक्ति` with digits.
  * devTamCharsOCR: `னான் என்பதின் संग्रहोक्ति, தகவோக்கத்தில் - कपाधिक्यத்தில். இது "ருள்` — prepends `னான் என்பதின்` from the previous line, misreads `कृपा` as `कपा`, and clips the final `அ`.
  * linewiseScriptSepDriveOCR: `னான் என்பதின் संग्रहोक्ति. தகவோக்கத்தில் कृपाधिक्यத்தில். இது 66அருள்` — nearly right, but leaks the tag digits `66` into the text.
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
  Tesseract 0.232, devTamCharsOCR 0.130, linewise 0.145)
  * GT: `क्षत्रबन्धुश्च पुण्डरीकश्च पुण्यकृत् । आचार्यवत्तया मुक्तौ तस्मादाचार्यवान्`
  * Tesseract: `“पापिष्ठः क्षनबन्धुर्च पुण्डरीकरच पुण्यकृत्‌ । आचार्यवत्तया मुक्तौ तस्मादाचार्यवान्‌` — prepends a word from the previous printed line and confuses `श`/`र`.
  * devTamCharsOCR: `िष्टः्षबन्धशच पुण्डरीकशच पुण्यकृत् । आचार्यवत्तया मुक्तो तस्मादाचार्यवान्` — loses the line start and `मुक्तौ`→`मुक्तो`.
  * linewiseScriptSepDriveOCR: ` 'पापिष्ठः क्षत्रबन्धुश्च पुण्डरीकश्च पुण्यकृत् आचार्यवत्तया मुक्तौ तस्मादाचार्यवान्` — almost exact; adds a leading space and an apostrophe, drops the `।` after `पुण्यकृत्`, and inherits `"पापिष्ठः` from the previous printed line.
* **Dense mixed line `01_p0043_comm_L05`** ([crop](examples/01_p0043_comm_L05.png); Tesseract
  0.353, devTamCharsOCR 0.147, linewise 0.191)
  * GT: `अतिशयित-पापமுள்ளவனான. क्षत्रबन्धुश्च - இந்த चकारம் आचार्यवत्तै இருந்தாலும்`
  * Tesseract: `எ19-ராமுள்ளவனான. कषत्रबन्धुङ्च- இந்த चकारा आचार्यवत्ते இருந்தாலும்‌ पापिष्ठ-` — substitutes the first word and appends the next line's text.
  * devTamCharsOCR: `ितपापமுள்ளவனான. क्ष्बन्धुश्च இந்த चकारம் आचार्यवत्तै இருந்தாலும் ்` — drops line-start chars, but keeps the script mixture.
  * linewiseScriptSepDriveOCR: `शयित - पापமுள்ளவனான. क्षत्रबन्धुश्च- இந்த चकार:9 आचार्यवत्तै இருந்தாலும் पापिष्ठ-` — good core, but leaks `:9` and inherits the next line.
* **Mixed prose line `01_p0044_core_L01`** ([crop](examples/01_p0044_core_L01.png); Tesseract
  0.526, devTamCharsOCR 0.053, linewise 0.070)
  * GT: `आचार्यवंशம் भगवाனளவுஞ்செல்ல अनुसन्धिக்கவேணுமென்றோதப்பட்டது.`
  * Tesseract: `आचार्य-वंशा *ளனளவுஞ்செல்ல अनुसन्यि 4०/69/0102 079 01८ 10 ८ ॐ.` — the Tamil half collapses into digits/symbols.
  * devTamCharsOCR: `आचार्यंश் भगवाனளவுஞ்செல்ல अनुसन्धिக்கவேணுமென்றுதப்பட்டது.` — near-exact; `वंशம்` and `ஓ` differ.
  * linewiseScriptSepDriveOCR: `आचार्य-- वंश,ம் भगवाனளவுஞ்செல்ல अनुसन्धि,க்கவேணுமென்றோதப்பட்டது.` — near-exact; adds a second hyphen and an extra comma after `अनुसन्धि`.

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
* **linewiseScriptSepDriveOCR leaks tags/escapes.** Per-line faux pages have few runs and smaller
  text, so Drive misreads some `#TNN` tags as text (`66`, `:9`, `Sh`, `u`, `hd`), and the
  per-line pages sometimes lose the trailing/leading space. This is a visible part of its
  residual error (0.143 vs 0.127 for page-batched script separation).
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
5. Merge patches into runs; drop runs < 20 px wide; split words at script transitions.
6. Compose one faux page per script: each run is a line, scaled to 90 px height (≥0.6), 130 px
   vertical gap, width cap 1780 px, JPEG q92; 20 runs per faux page.
7. OCR each faux page with Google Drive. Every run carries a printed `#TNN` tag to its left; the
   parser maps tags → runs, so dropped or merged OCR lines cannot shift the page. Continuation
   lines bind to the previous tag; markdown escapes and trailing base64 image blobs are stripped.
8. Missing tags trigger one retry of the affected chunk (no per-run fallback); runs that remain
   unreadable are left empty. Reassemble the block with per-run `pre_space` from the patch map.

In the five-block + seven-example run, 148 faux pages were composed and 74 of the 222 Drive
pages (33%) needed a retry for missing tags. After retries, 26/898 runs (2.9%) remain empty. 43/898 runs (4.8%)
contain a Latin character, mostly leaked tag/escape characters.

**Script classifier:** synthetic 48×48 patches rendered from `data/texts/{core,commentary}` with
the Devanagari/Tamil font sets (plus noise/blank for `other`/`null`), residual CNN (0.84M
parameters, 20k steps on a Kaggle T4). Word-disjoint validation accuracy **0.996** (dev 1.000 /
tam 0.998 / other 0.985 / null 0.986). A short distillation pass on unlabelled RTS_sb_01
p0030–p0041 patches with synthetic replay keeps synthetic accuracy at 0.996 and reaches 95.6%
agreement with the earlier model on held-out real pages. The block OCR numbers in §4 use this
classifier. Its `other` predictions are dropped from the final assembly: on these pages `other`
is punctuation/tag debris, and the raw unfiltered `drive_sep` micro CER is 0.1400 versus 0.1266
after the filter. On the five eval blocks, line-level script presence is 64/69 versus 65/69 for
the earlier model.

`linewiseScriptSepDriveOCR` uses the same steps but composes one faux page per printed line
(`--linewise`), i.e. no cross-line batching.

### 4.4 Results

Whitespace-stripped CER per block; micro is over all five blocks (6,193 GT chars).

| system | comm p0042 | core p0043 | comm p0043 | core p0044 | comm p0044 | **micro** |
|---|---|---|---|---|---|---|
| **scriptSepDriveOCR (v2)** | **0.068** | **0.022** | 0.263 | **0.042** | **0.046** | **0.127** |
| linewiseScriptSepDriveOCR | 0.109 | 0.033 | **0.240** | 0.065 | 0.085 | 0.143 |
| Tesseract on whole lines (`tam+san`) | 0.241 | 0.133 | 0.445 | 0.377 | 0.362 | 0.352 |
| Tesseract on script runs (per-script models) | 0.155 | 0.133 | 0.422 | 0.120 | 0.155 | 0.244 |
| **Tesseract on script segments (per-script models)** | 0.144 | 0.111 | 0.399 | 0.125 | 0.132 | **0.227** |
| Tesseract on script segments (mixed `tam+san` model, control) | 0.180 | 0.144 | 0.487 | 0.278 | 0.239 | 0.308 |
| devTamCharsOCR on whole lines | 0.103 | 0.233 | 0.443 | 0.200 | 0.140 | 0.238 |
| devTamCharsOCR on script runs | 0.812 | 0.800 | 0.871 | 0.691 | 0.767 | 0.812 |
| devTamCharsOCR on script segments | 0.470 | 0.256 | 0.782 | 0.449 | 0.563 | 0.599 |
| Drive OCR on the whole block (no split) | 0.277 | 0.222 | 0.551 | 0.540 | 0.450 | 0.435 |

The two Drive-split rows drop `other` runs from the final assembly; before that filter the raw
micro CERs are 0.1400 (page-batched) and 0.1592 (linewise). The filter removes punctuation/tag
debris (`evaluation/sb_eval/metrics_filtered.json`).

scriptSepDriveOCR beats every alternative in the micro average and 4/5 individual blocks (on
`comm/p0043`, linewise wins 0.240 vs 0.263 because that page's dense Tamil commentary benefits
from per-line context). The split is worth **3.4×** against unsplit Drive (0.435), **1.9×**
against the best devTamCharsOCR configuration (0.238), and **1.8×** against the best local
script-split Tesseract configuration (0.227). The unsplit Drive output is erratic: 0.031 on
`core/p0044` but 0.551 on `comm/p0043`, where dense Tamil commentary makes it reorder and drop
lines. linewiseScriptSepDriveOCR is usually a little worse than page-batched separation
(0.143 vs 0.127 micro), and worse on 4/5 blocks.

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
1.8× worse than Drive on the same blocks (0.127), and one block (`comm/p0043`, 0.399) remains
far from usable. Drive's advantage is recognition quality on the fragments, not the script split.

**Q2 — can a script-pure variant of devTamCharsOCR close the gap on faux pages?**

Not as an input variant, and the evidence argues against a per-script retrain as the next step.
Applied to script-pure segments, devTamCharsOCR scores **0.599**, much worse than the same model
on whole lines (**0.238**) and 4.7× worse than Drive on the same blocks (**0.127**). On natural
pure-script lines the model is not better either: 0.408 on Devanagari-only and 0.300 on
Tamil-only lines versus **0.285** on mixed lines (§3.3). Script purity is therefore not what
devTamCharsOCR lacks.

What it lacks is the **short-fragment regime**:

* 27% of script runs get an empty devTamCharsOCR output; short 1–4-akshara crops are outside the
  line-crop training distribution.
* Segment crops are larger and improve to 0.599, but still lose the full-line context the model
  was trained on (and Drive gets 0.127 on the same blocks).
* A script-pure *model* trained on the same line-crop distribution would inherit the same
  short-fragment weakness. Closing the gap requires training on faux-page-style fragments (or
  fine-tuning a pretrained recognizer such as PP-OCRv4), not merely restricting training data to
  one script.

### 4.6 Combining devTamCharsOCR with scriptSepDriveOCR (negative result)

The two systems fail differently, so we tested whether mixing them lowers CER. All variants
reuse the same run segmentation and block assembly and differ only in the per-run text source
(`training/scripts/script_cls/combine_sb.py`; per-run texts in
`evaluation/sb_eval/runs_*.jsonl`; scores in `evaluation/sb_eval/combine_filtered.json`).

| variant | micro CER |
|---|---|
| **scriptSepDriveOCR (baseline)** | **0.127** |
| devTamCharsOCR on script runs (all runs) | 0.812 |
| devTam replaces Drive only for Devanagari runs | 0.427 |
| devTam replaces Drive only for Tamil runs | 0.463 |
| devTam fills only empty Drive runs | 0.127 |
| run-level hybrid (devTam if Drive text is empty or suspicious) | 0.130 |

* **devTamCharsOCR does not transfer to isolated runs.** It is a full-line model; on tight
  1–4-akshara crops it loses context, emits nothing for 27% of runs, and decodes at CER 0.68–0.86
  depending on the block.
* **Repair rules do not help.** Filling empty Drive runs with devTam changes the micro CER from
  0.1266 to 0.1269 (the empty runs are sub-glyph stubs that neither engine can read).
* **The suspicious-text hybrid hurts slightly** (0.130): runs whose Drive text contains a Latin
  character are not reliably better when replaced by devTam.
* **Script-conditional mixes are much worse** (0.427 when devTam handles only Devanagari runs,
  0.463 when only Tamil runs), because the run-level devTam texts dominate the error.
* **There is no complementarity at these units.** Because block GT is not line-aligned, we did
  not compute a per-line oracle; the run-level results are already decisive.

Combining the current two systems therefore does **not** reduce CER. The productive combination
is to train a recognizer in the same regime as Drive — faux-page fragments — rather than mixing
a full-line model with run-level inputs.

### 4.7 Qualitative block example

`core/p0043` ([block image](../data/RTS_sb_01/images/core/p0043.jpg), 2 lines, 90 GT chars;
CERs: devTamCharsOCR 0.233, Tesseract segments 0.111, scriptSepDriveOCR 0.022,
linewise 0.033):

* GT: `"पापिष्ठः क्षत्रबन्धुश्च पुण्डरीकश्च पुण्यकृत् । आचार्यवत्तया मुक्तौ तस्मादाचार्यवान् भवेत् ॥" என்று`
* devTamCharsOCR (whole lines): `पापटः्षवन्धुच पुण्डरीकश्च पुण्यकृत् । आचार्यवत्तया मुक्तो तस्मादाचार्यवान्` / `ன்று` — loses the opening quote, the first two words and the final `भवेत् ॥"`.
* Tesseract (script segments): `"पापिष्ठः क्षत्रबन्धुरुच पुण्डरीकङच पुण्यकृत्‌ । आचार्यवत्तया मुक्तौ तस्मादाचार्यवान्‌` / `भवेत्‌்‌” என்று` — all words present; `श`→`र`/`ङ`, `॥"`→`”`.
* scriptSepDriveOCR: `" पापिष्ठः क्षत्रबन्धुश्च पुण्डरीकश्च पुण्यकृत् आचार्यवत्तया मुक्तौ तस्मादाचार्यवान्` / `भवेत् ॥ என்று` — almost exact; one extra space after the quote, the `।` after `पुण्यकृत्` is dropped, and the closing `॥"` is lost.
* linewiseScriptSepDriveOCR: `" पापिष्ठः क्षत्रबन्धुश्च पुण्डरीकश्च पुण्यकृत् आचार्यवत्तया मुक्तौ तस्मादाचार्यवान्` / `भवेत् ॥ 1' என்று` — same, but leaks `1'` into the second line.

The dense `comm/p0043` block ([image](../data/RTS_sb_01/images/comm/p0043.jpg), CER 0.263) shows
the other failure mode: the words are mostly right, but Drive leaks tag/escape characters into
the text (`நிர்பந்த'முண்டோ?`, `Sh uபில்`, `uயாதோ?`), and the residual errors concentrate in
Tamil vowel signs and word spacing. This block alone accounts for most of the gap between 0.127
and a production-ready figure.

## 5. Verdict and recommendations

* **For line crops** (clean scans): devTamCharsOCR (CER 0.29) is the best available; Tesseract
  (0.48) is the best zero-shot engine and the only sensible pretrained fallback. **PaddleOCR
  pretrained models are not usable** (0.71–0.73), and the Indic-OCR repo adds no accuracy of its
  own (no weights, stub fine-tuning, English-only TrOCR, preprocessing that changes nothing).
* **For blocks/pages:** feed neither whole lines nor whole blocks to a single-script engine.
  scriptSepDriveOCR is the winning pattern by a wide margin (block CER **0.127** vs 0.238 for
  devTamCharsOCR on whole lines and 0.435 for unsplit Drive OCR); batching per printed line
  (linewiseScriptSepDriveOCR) is worse (0.143).
* **Q1 (local per-script backend): yes, the gain is real.** Script-segmenting and using
  per-script Tesseract models cuts the local CER from 0.352 to 0.227 (36% relative), and improves
  every block. It is not Drive-specific. But it is not parity: Drive remains 1.8× better on the
  same blocks.
* **Q2 (script-pure devTamCharsOCR): no.** Script-pure segments score 0.599 vs 0.238 for whole
  lines, and pure natural lines are not easier than mixed ones. The model needs fragment-regime
  training, not script-pure training.
* **Do not combine the two systems as-is:** run-level mixing of devTamCharsOCR with
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
  not yet good enough to replace Drive (0.127).
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
| `evaluation/sb_eval/metrics_filtered.json` | primary filtered `drive_sep`/`drive_line` CERs (`other` runs dropped) |
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
| `training/scripts/script_cls/pipeline.py` | production block/page pipeline |
| `training/scripts/script_cls/drive_ocr_images.py` | Drive OCR helper (skips existing outputs) |
| `training/scripts/script_cls/combine_sb.py` | run-level combination variants for the `RTS_sb` blocks (§4.6) |
| `training/scripts/script_cls/control_seg.py` | mixed-model segmentation control for the Q1 decomposition (§4.5) |
| `evaluation/sb_eval/combine.json` | combination-variant CERs per block and micro |
| `evaluation/sb_eval/combine_filtered.json` | combination variants after dropping `other` runs |
| `evaluation/sb_eval/tess_seg_mixed.json` | segmentation control: mixed Tesseract model on script segments |
| `training/scripts/script_cls/combine_eval.py` | earlier PDF-page combination script (pilot artifacts in `evaluation/RTS_5c/`) |
| `data/RTS_sb_01/{images,gt}/` | block crops (`comm/`, `core/`) and block GT |
| `evaluation/eval_index.jsonl`, `evaluation/split_eval.jsonl` | canonical verified blocks and train/eval split |
| `data/RTS_5c/`, `evaluation/RTS_5c/` | archived `RTS_5c` volumes, pilot predictions and combination runs (reference only) |
| `training/local/kaggle_v10_best.pt` | devTamCharsOCR best checkpoint (kraken-spec, 40 ep, fp32, no aug) |
| `training/local/kaggle_v13_best.pt` | repeat-run checkpoint (same config) |
| `training/local/script_cls.pt` | script-patch classifier (48×48, dev/tam/other/null; synthetic val_acc 0.996) |
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
* **Runtime:** the re-run composed 151 faux pages and needed 86 retry pages (237 Drive calls),
  taking ~60 min wall clock. The retry rate (36% of calls) is the main cost; missing tags on
  composed pages remain the weakest point. For 3000 pages this is still weeks, so throughput
  remains the production blocker.
* **`other` filter:** the primary Drive-split rows drop `other` runs. On these pages `other` is
  punctuation/tag debris, and dropping it lowers the micro CER from 0.1400 to 0.1266
  (page-batched) and from 0.1592 to 0.1434 (linewise). If Latin/digit text matters on a page,
  keeping `other` runs is the right choice.
* **Drive API behavior is erratic:** 0.031 on `core/p0044` vs 0.551 on `comm/p0043` for the same
  whole-block call, with occasional multi-minute stalls. Script splitting wins on every block
  regardless, but single-block whole-region numbers mean little.
* **Tag/markdown pollution:** ~5% of runs contain Latin characters, mostly leaked `#TNN` tags or
  markdown escapes (`hd`, `Sh`, `u`, digits). The parser strips common escapes, but a stricter
  normalizer is needed before the output is usable as an e-text.
* **Q1 and Q2 are answered above** (§4.5). Remaining questions: (a) how much does word-gap tuning
  recover the residual spacing errors? (b) more blocks/volumes, especially two-column layouts
  (none seen so far); (c) can a recognizer trained on faux-page fragments (rather than line
  crops) approach Drive's 0.127 while running locally? (d) how far does a lightly fine-tuned
  PP-OCRv4 get on the same fragment regime?
