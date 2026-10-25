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

The classifier identifies mixed words reliably. The remaining problem is splitting a mixed word:
connected-component inference feeds aksharas to a word-level model, so the boundary model /
column separator is required for the split.