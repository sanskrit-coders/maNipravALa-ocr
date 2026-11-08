# Kraken Word-Level Dual-Script OCR (`kraken_words_model`)

**Status:** **Obsolete / diagnostic only** (not used in [`current-pipeline.md`](../../current-pipeline.md)).

## Model Specification

| Property | Specification |
|---|---|
| **Weights** | `training/local/kraken_words_model/best_0.8900.safetensors` |
| **Metrics file** | `training/local/kraken_words_metrics.json` |
| **Training script** | [`training/scripts/train_kraken_words.py`](../../../training/scripts/train_kraken_words.py) |
| **Inference script** | [`training/scripts/apply_ocr_pdf.py`](../../../training/scripts/apply_ocr_pdf.py) |
| **Kaggle dataset / kernel** | `vishvasvasuki/rts-word-image-ocr` / `vishvasvasuki/rts-word-ocr-train` |
| **Architecture** | Kraken 7.1.1 VGSL (`[1,128,0,1 Cr3,3,32 Do0.1,2 Mp2,2 Cr3,3,64 Do0.1,2 Mp2,2 Cr3,3,128 Do0.1,2 Mp2,1 Cr3,3,256 Do0.1,2 Mp2,1 S1(1x0)1,3 Lbx256 Do0.2,2 Lbx256 Do0.2]`), **4.1M parameters** |
| **Input** | Single word crop image (`box` from `word_segments`), vertically normalized to height 128 px |
| **Output** | UTF-8 word string decoded by a single 1D-CTC head covering both Devanāgarī and Tamil codepoints |

## Training & Held-Out Evaluation Metrics

- **Training corpus:** **92,976** high-confidence word crops (`confidence >= 0.45`) from `data/RTS_sb_01/word-image-maps` and `data/RTS_sb_02/word-image-maps`.
- **Held-out evaluation corpus:** **2,435** word crops (`21,013` total characters).
- **Character accuracy:** **90.34%** (`9.66%` CER; `2,167` character edits: `306` insertions, `1,096` deletions, `765` substitutions).
  - Pure Tamil word crops: **94.31%** char accuracy.
  - Pure Devanāgarī word crops: **88.81%** char accuracy.
- **Exact word accuracy:** **80.53%**.

## Why Moved to `models/obsolete/`

When applied across all 806 pages of Volume 4 (producing the legacy `tesseract/4.md`, archived at `/tmp/opencode/4.kraken-wordwise.md`):
1. **High word error rate in production (~1 in 5 words wrong):** An 80.53% word accuracy (9.66% CER) translates to **~100,000 character errors** across a 1.15M-character volume, severely mangling Sanskrit conjuncts (e.g., `चरणाविति निर्देशः पत्नीवैशिष्ट्यबाधकः` → `च्रणाविति निदेपरः मत्ीै्रिष्ट्यवाधकः`).
2. **Cross-script CTC confusion (1,456 chimeric words):** Because a single CTC softmax layer decodes both Devanāgarī and Tamil alphabets on isolated word crops without word-level or line-level language priors, visually similar strokes across scripts (such as Devanāgarī `प`/`ए`/`त`/`।` vs. Tamil `ப`/`எ`/`ா`/`ர`) frequently switch scripts mid-character (`एா`, `எः`, `५ெ`, `तர்`).
