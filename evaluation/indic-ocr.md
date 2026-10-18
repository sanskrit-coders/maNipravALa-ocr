# Indic-OCR (BytesByJay) — accuracy evaluation

Evaluated as a candidate OCR engine for **sArabodhinI vol. 01–02** (mixed Tamil–Devanagari
manipravala print), against our held-out line eval split and the Tesseract `tam+san` baseline.

**Bottom line:** the repository ships **no trained models** — its recognition accuracy *is*
PaddleOCR's pretrained accuracy, and on our eval lines that is **CER 0.73 (rec-only, oracle
script pick) / 0.71 (det+rec)** on the 445 CTC-feasible lines, versus **0.48 for Tesseract
`tam+san`**. PaddleOCR is ~1.5× worse than Tesseract in every script bucket. As shipped, the
project's script classifier is randomly initialized, so even that number is an upper bound.

---

## 1. What the repository actually ships

Repo audit (commit `main`, cloned 2026-10-18; 4 commits, no releases, no weights):

| Component | Claim | Reality |
|---|---|---|
| OCR engine (`src/ocr_engine.py`) | "PaddleOCR/TrOCR" | Thin wrapper: `PaddleOCR(use_angle_cls=True, lang=..., use_gpu=..., show_log=False)` + `ocr.ocr(img, cls=True)`; also EasyOCR and TrOCR backends. All recognition is the upstream engines' |
| TrOCR backend | "state-of-the-art" | `microsoft/trocr-base-handwritten` — an **English** handwriting model; cannot produce any Indic script |
| Script classifier (`src/script_classifier.py`) | automatic script detection | Architecture + training code only; **no weights in the repo** (`models/` contains a README). As shipped it predicts from random init |
| Preprocessing (`src/preprocessing.py`) | "advanced preprocessing" | OpenCV pipeline: NLM denoise, CLAHE, unsharp mask (`preprocess_for_ocr`); optional deskew/binarize/resize for training-path |
| Evaluation tools (`src/evaluation.py`) | CER/WER | self-contained pure-Python Levenshtein/CER (+ sklearn metrics for the classifier) |
| `train.py` | fine-tuning | **stub**: `train_ocr()` logs "OCR fine-tuning is not yet implemented." Only the script-classifier task has real training code; no artifacts published |

So "Indic-OCR accuracy" reduces to: **PaddleOCR pretrained models** (PP-OCRv4 `ta` / `devanagari`
recognition + PP-OCRv3 multilingual detection) **plus** optional OpenCV preprocessing and
**oracle-or-random script selection**.

## 2. What we evaluated and how

* **Data:** our OCR eval split — 534 single-line crops from vol. 01–02. 89 lines carry broken
  auto-alignment targets (tiny slivers with 1000+ char references, found while building the
  line data); the **445 CTC-feasible lines** are the fair comparison set used for all local
  engine evals. Both numbers are reported.
* **Engine:** `paddleocr==2.7.3` + `paddlepaddle==2.6.2` (CPU) — the API generation the repo's
  code is written against. Models auto-downloaded: `Multilingual_PP-OCRv3_det_infer`,
  `ta_PP-OCRv4_rec_infer`, `devanagari_PP-OCRv4_rec_infer`, `ch_ppocr_mobile_v2.0_cls_infer`.
* **Modes:**
  * `rec` — recognition only on the line crop (`det=False`): the fair line-level comparison
    with our Tesseract (psm 7) and kraken evals.
  * `det+rec` — the exact call path Indic-OCR uses (`.ocr(img, cls=True)`).
* **Languages:** `ta` and `hi` runs for every line; `best` = per-line minimum-CER pick
  (an **oracle** script selector — upper bound for a perfectly trained version of Indic-OCR's
  classifier; mixed-script lines are unrouteable in principle anyway).
* **Metrics:** jiwer CER/WER on raw and whitespace-stripped text, identical to
  `eval_lines.py` / the Tesseract baseline.
* **Runtime (CPU, 6 threads):** rec ≈ 1.6 lines/s; det+rec ≈ 0.4 lines/s.

Scripts: `training/scripts/indic-ocr/{setup_env.sh,eval_paddleocr.py,compare_with_baseline.py}`.
Commands:

```bash
training/scripts/indic-ocr/setup_env.sh                       # venv + pinned deps + repo clone
/tmp/opencode/indicocr-venv/bin/python training/scripts/indic-ocr/eval_paddleocr.py \
    --mode both --langs ta,hi --dump evaluation/line_eval_paddleocr.jsonl
python3 training/scripts/indic-ocr/compare_with_baseline.py
```

## 3. Results

### 3.1 Feasible subset — 445 lines (fair comparison)

| engine | CER | CER no-space | WER | WER no-space |
|---|---|---|---|---|
| Tesseract `tam+san` psm7 (baseline) | **0.476** | 0.484 | 0.976 | 0.998 |
| PaddleOCR rec `ta` | 0.757 | 0.762 | 1.030 | 1.000 |
| PaddleOCR rec `hi` | 0.912 | 0.944 | 1.036 | 1.000 |
| PaddleOCR rec best-of (oracle) | 0.732 | 0.735 | 1.022 | 1.000 |
| PaddleOCR rec best-of + Indic-OCR preprocessing | 0.732 | 0.731 | 1.017 | 1.000 |
| PaddleOCR det+rec `ta` | 0.755 | 0.774 | 1.116 | 1.000 |
| PaddleOCR det+rec best-of (oracle) | 0.710 | 0.728 | 1.117 | 1.000 |

### 3.2 All 534 lines (includes the 89 broken-target lines)

| engine | CER | CER no-space | WER |
|---|---|---|---|
| Tesseract `tam+san` psm7 | 0.705 | 0.710 | 0.986 |
| PaddleOCR rec best-of | 0.839 | 0.839 | 1.012 |
| PaddleOCR det+rec best-of | 0.707 | 0.727 | 1.096 |

(The 89 broken lines have huge references and dominate character-weighted CER; the
det+rec pipeline "wins" there only by emitting nothing on slivers, capping their per-line
CER at 1.0 instead of >1.0.)

### 3.3 By GT script composition (feasible subset)

| script | n | Tesseract | Paddle rec `ta` | Paddle rec best (oracle) |
|---|---|---|---|---|
| devanagari-dominant | 55 | **0.485** | 0.942 | 0.731 |
| mixed (Tamil+Devanagari) | 343 | **0.492** | 0.752 | 0.750 |
| tamil-dominant | 47 | **0.312** | 0.573 | 0.570 |

Oracle script picks (all 534): rec `ta` on 455 lines, `hi` on 79; det+rec: `ta` 390 / `hi` 144.
PaddleOCR never beats Tesseract in any bucket.

## 4. Qualitative examples

| line (GT) | PaddleOCR best | Tesseract |
|---|---|---|
| தன்பற்றில் - தானே தன் **संबन्ध**த்தை உண்டாக்குகையில். … | `Id Miog, ' ounைoூs1g- Go$h- ms ng - gudr?` | தன்பற்றில் - தானே தன் **84**த்தை உண்டாக்குகையில். … |
| **सङ्ग्रहोक्ति.** தகவோக்கத்தில் - **कृपाधिक्य**த்தில். இது 'அருள் | `னானஎன்பதினிப்uொிர.தகேவாக்கத்தில்-துரித்தில.இதரஅஞளர்` | னான்” என்பதின் **संग्रहोक्ति**. தகவோக்கத்தில் - **ஊரிண**த்தில். இது “அருள் |
| பாட்டில் சொன்ன இன்பம் முதலான … (Tamil only) | `பாட்டல்ெசானனஇன்பம்ழதலான…` (CER 0.27, spaces lost) | — |

Failure modes confirmed by the dumps:

* **Mixed-script lines defeat both models.** The `ta` model renders Devanagari as Latin garbage
  (`संबन्ध` → `Go$h-`) and vice versa (`hi` outputs digits/Latin for Tamil). 343/445 feasible
  eval lines are mixed — this is the dominant error source.
* **Even same-script text loses spacing** and confuses similar ligatures (`சொன்ன` → `ெசானன`).
* Devanagari model (`hi`, trained on Hindi) is the weakest: 0.91 CER alone.

## 5. Verdict

* **As-shipped Indic-OCR recognition = PaddleOCR = CER ≈ 0.71–0.73** on our feasible eval set —
  **not usable** for manipravala (Tesseract, despite its own issues, is at 0.48 and remains the
  better zero-shot engine for Tamil+Sanskrit).
* None of the project's own components can currently be exercised as advertised: the script
  classifier has no weights, TrOCR is English-only, and the advertised OCR fine-tuning
  (`train.py --task ocr`) is a stub. The claimed multi-script accuracy is unreproducible from
  the repo.
* The one interesting path forward the repo points at (and that matches our own findings):
  **fine-tune a recognizer on our line data**. Fine-tuning PaddleOCR's PP-OCRv4 recognizers
  (upstream PaddleOCR, not this repo's stub) would import Devanagari/Tamil glyph knowledge,
  which our from-scratch kraken model lacks. Worth a follow-up experiment with the ~16.5k
  feasible training lines.
* The OpenCV preprocessing ablation (`--preprocess repo`: Indic-OCR's denoise + CLAHE +
  unsharp on the same models) changes **nothing materially**: oracle best-of CER 0.732 with
  and without it on the 445-line subset (per-language: `ta` 0.760 vs 0.757, `hi` 0.897 vs
  0.912; all-534 best-of 0.843 vs 0.839). The project's own contribution to recognition
  accuracy is ~zero.

## 6. Artifacts

| file | contents |
|---|---|
| `evaluation/line_eval_paddleocr.jsonl` | per-line rec-mode predictions (`pred_ta`, `pred_hi`, oracle `pred`) |
| `evaluation/line_eval_paddleocr.jsonl.summary.json` | rec-mode aggregate metrics |
| `evaluation/line_eval_paddleocr_det.jsonl(+.summary.json)` | det+rec-mode predictions |
| `evaluation/line_eval_paddleocr_pp.jsonl(+.summary.json)` | rec-mode with Indic-OCR preprocessing |
| `evaluation/line_eval_baseline.jsonl` | Tesseract baseline (534 lines) |
| `training/scripts/indic-ocr/` | `setup_env.sh`, `eval_paddleocr.py`, `compare_with_baseline.py` |
