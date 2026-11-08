# Line classifier (experimental)

## Input and output labels

| | |
|---|---|
| **Input** | One 64×1024 grayscale line image. |
| **Output labels** | 3 classes: `dev`, `tam`, `other`. |
| **Granularity** | One label per line (dominant script). |


- Model: 64×1024 line CNN, 3 classes (`dev/tam/other`).
- Training: synthetic lines (pure script + 20% mixed).
- Synthetic val accuracy **1.000**.
- Real-line validation on 103 reconstructed real lines: **~0.71**; not used by default.
- Code: `training/scripts/script_cls/train_line_cls.py` (source in `/tmp/opencode`).
- Kaggle: `vishvasvasuki/line-script-cls-train`.

Kept as an experiment; the word-level classifier is the production route.