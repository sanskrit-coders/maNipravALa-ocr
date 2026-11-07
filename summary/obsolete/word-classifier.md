# Word classifier (dev / tam / other / mixed)

## Input and output labels

| | |
|---|---|
| **Input** | One 64×256 grayscale word image (word box resized/padded; text height 24–48 px). |
| **Output labels** | 4 classes: `dev`, `tam`, `other`, `mixed`. |
| **Granularity** | One label per whole word image. |


- Model: 64×256 word CNN, 4 classes: `dev`, `tam`, `other`, `mixed`.
- Boxes: TSV↔GT alignment rebuilt with TSV text/width DP + column filtering; **112,948**
  boxes. Outlier gaps and wrong-column boxes are filtered.
- Retraining still fails because some **GT line/word assignments are wrong**, not just the
  alignment. Example: `RTS_sb_02/core/p0538.jpg` line 4, GT word `कुरु’’` is assigned a box
  whose image is Tamil (`ிளிஈத்தை`).
- Latest local retrain: real dev **0.676**, tam **0.777**, mixed **0.151**; not usable.
  Old `script_sep.pt` retained; faux pages not regenerated.


Usage: `--word-classifier training/local/script_sep.pt`.

The classifier identifies mixed words on synthetic/in-domain data, but was moved to `obsolete/` for two reasons:

1. **Wrong training source used for retrain:** The failed retrain (`dev` 0.676 / `tam` 0.777 / `mixed` 0.151) used `train_words_fixed.jsonl` (line-level TSV↔GT width DP, which still had line-assignment errors like `p0538`) instead of the **95,411 high-confidence `word-image-maps`** (`confidence ≥ 0.45`) that succeeded for `script_cls.pt` (0.9921 accuracy).
2. **End-to-end failure in `faux-pages-wordwise-shiro-rekhA` (`4.drive.wordwise.md`):** Using the un-retrained `script_sep.pt` in the Volume 4 wordwise pipeline misclassified numerous Tamil words (`உடையவர்களுடை`, `தீர்மானங்கள`, `அநுஸரிக்கவேண்`) onto Devanagari faux pages and triggered spurious Shiro-rekhā splits that sliced Sanskrit visargas/matras (`ीः-`, `ा-`) onto Tamil faux pages. When those isolated word/sub-word fragments were sent to Google Drive OCR, Drive collapsed from context starvation.