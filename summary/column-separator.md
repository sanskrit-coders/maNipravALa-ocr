# Column separator

- Model: per-column CNN over a 64×256 word image, 3 classes (`dev/tam/other`).
- Training data: 24,890 mixed words with component-gap-refined boundaries.
- Validation: real column accuracy **0.715**, boundary MAE **0.087** of word width.
- Model: `training/local/column_sep.pt`.
- Training code: `training/scripts/script_cls/train_column_sep.py`.
- Kaggle: `vishvasvasuki/rts-column-sep-train`.

Wired as optional `--column-classifier`; not enabled by default. Boundary labels remain
approximate (character counts/component gaps), so the separator still fragments some words.
