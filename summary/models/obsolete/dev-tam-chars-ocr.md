# Whole-Line Mixed-Script Kraken OCR (`devTamCharsOCR`)

**Status:** **Obsolete / superseded by `tadevaOCR`** (not used in [`current-pipeline.md`](../../current-pipeline.md)).

## Model Specification

| Property | Specification |
|---|---|
| **Checkpoints** | `training/local/kaggle_v10_best.pt`, `training/local/kaggle_v13_best.pt`, `training/local/manual/best.pt`, `training/local/manual2/best.pt` |
| **Training scripts** | [`training/scripts/train_FromScratch.py`](../../../training/scripts/train_FromScratch.py), [`training/scripts/pack_uint8.py`](../../../training/scripts/pack_uint8.py) |
| **Evaluation script** | [`training/scripts/eval_5blocks.py`](../../../training/scripts/eval_5blocks.py) |
| **Architecture** | Kraken VGSL whole-line CNN + BiLSTM CTC recognizer trained from scratch in `fp32` |
| **Input** | Full grayscale text-line crop |
| **Output** | Mixed Devanāgarī + Tamil UTF-8 line transcription |

## Training & Benchmark Metrics ([`evaluation/summary.md`](../../../evaluation/summary.md))

- **Training corpus:** ~16,500 aligned lines from `RTS_sb_01` and `RTS_sb_02` (40 epochs, `fp32`, no geometric augmentation).
- **445 feasible held-out lines:** Line CER **0.292** (best single whole-line local recognizer, ahead of Tesseract `tam+san` at **0.476** and PaddleOCR at **0.710**).
- **5 hand-verified `RTS_sb_01` Maṇipravāḷa blocks (89 lines, 6,193 non-space GT chars):**
  - Whole-line block CER: **0.238** (**0.285** on mixed lines).
  - Isolated script-run CER: **0.812** (emitted empty strings on 27% of short runs).
  - Grouped script-segment CER: **0.599**.

## Why Moved to `models/obsolete/`

1. **2.1× higher error than `tadevaOCR`:** On the same 5 verified `RTS_sb_01` blocks, script-separated phrase strips with Google Drive OCR achieved **0.112 block CER** vs. **0.238** for `devTamCharsOCR`.
2. **Collapse on short script runs:** Because `devTamCharsOCR` was trained exclusively on full-width page lines, feeding it short 1–4 akshara script segments falls outside its BiLSTM length distribution (0.599–0.812 CER).
