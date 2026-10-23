# Patch script classifier

## Input and output labels

| | |
|---|---|
| **Input** | One 48×48 grayscale patch (sliding stride 8; patch is centered on a word column, normalized by `1 - pixel/255`). |
| **Output labels** | 4 classes: `dev`, `tam`, `other`, `null`. `null` means low confidence/blank and is discarded. |
| **Granularity** | One label per patch; adjacent patches are merged into script runs. |


- Model: 48×48 patch residual CNN, 4 classes (`dev/tam/other/null`).
- Synthetic validation accuracy **0.996** (dev 1.000 / tam 0.998 / other 0.985 / null 0.986).
- Model: `training/local/script_cls.pt`.
- Training code: `training/scripts/script_cls/train_classifier.py`.
- Kaggle: `vishvasvasuki/manipravala-script-cls` and adaptation
  `vishvasvasuki/manipravala-script-cls-ft-replay`.

The patch classifier is the base for safe word/run splits (`pipeline.py::classify_word`).
It is weaker on this book’s font than the word-level classifier.