# Patch script classifier

- Model: 48×48 patch residual CNN, 4 classes (`dev/tam/other/null`).
- Synthetic validation accuracy **0.996** (dev 1.000 / tam 0.998 / other 0.985 / null 0.986).
- Model: `training/local/script_cls.pt`.
- Training code: `training/scripts/script_cls/train_classifier.py`.
- Kaggle: `vishvasvasuki/manipravala-script-cls` and adaptation
  `vishvasvasuki/manipravala-script-cls-ft-replay`.

The patch classifier is the base for safe word/run splits (`pipeline.py::classify_word`).
It is weaker on this book’s font than the word-level classifier.
