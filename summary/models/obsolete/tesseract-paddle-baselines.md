# Tesseract 5 & PaddleOCR PP-OCRv4 Baselines

**Status:** **Obsolete baselines** (not used in [`current-pipeline.md`](../../current-pipeline.md)).

## Models Evaluated ([`evaluation/summary.md`](../../../evaluation/summary.md))

| Engine / Configuration | 445-Line CER | 5-Block Whole-Line CER | 5-Block Script-Split CER | Verdict |
|---|---:|---:|---:|---|
| **PaddleOCR PP-OCRv4 (`ta` / `hi`)** | 0.710–0.732 | — | — | **Unusable.** Single-script models render the other script as Latin garbage; `Indic-OCR` repo had no trained Manipravalam weights. |
| **Tesseract 5 (`tam+san`, PSM 7)** | 0.476 | 0.352 | — | **Poor.** Dual-language mode confuses Devanāgarī and Tamil character sets on mixed lines and produces chimera tokens like `(द्रयाணன்` for `(द्वया-)`. |
| **Tesseract 5 (`san` / `tam` routed by `script_cls.pt`)** | — | — | **0.227** | **2.0× worse than Drive OCR (0.112).** Routing script-pure segments to single-script `san` and `tam` Tesseract improves over `tam+san` (0.352 → 0.227), but remains twice as error-prone as Google Drive OCR on the same strips. |
