# Word classifier (dev / tam / other / mixed)

## Input and output labels

| | |
|---|---|
| **Input** | One 64×256 grayscale word image (word box resized/padded; text height 24–48 px). |
| **Output labels** | 4 classes: `dev`, `tam`, `other`, `mixed`. |
| **Granularity** | One label per whole word image. |


- Model: 64×256 word CNN, 4 classes: `dev`, `tam`, `other`, `mixed`.
- Training data: `data/script-separator/train_words.jsonl` — **51,210** words from 1,070 pages,
  all labeled from GT; no stored crops.
- Validation (real, page-disjoint): dev **0.999**, tam **1.000**, other **0.993**,
  mixed **0.986**; synthetic val **0.996**.
- Model: `training/local/script_sep.pt`.
- Training code: `training/scripts/script_cls/train_script_sep.py`.
- Kaggle: `vishvasvasuki/rts-script-sep-train`.

Usage: `--word-classifier training/local/script_sep.pt`.

The classifier identifies mixed words reliably. The remaining problem is splitting a mixed word:
connected-component inference feeds aksharas to a word-level model, so the boundary model /
column separator is required for the split.