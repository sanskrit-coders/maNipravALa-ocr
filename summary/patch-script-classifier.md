# Patch script classifier

## Input and output labels

| | |
|---|---|
| **Input** | One 48×48 grayscale patch (sliding stride 8; patch is centered on a word column, normalized by `1 - pixel/255`). |
| **Output labels** | 4 classes: `dev`, `tam`, `other`, `null`. `null` means low confidence/blank and is discarded. |
| **Granularity** | One label per patch; adjacent patches are merged into script runs. |


- Model: 48×48 patch residual CNN, 4 classes (`dev/tam/other/null`).
- Synthetic validation accuracy **0.996** (dev 1.000 / tam 0.998 / other 0.985 / null 0.986).
- Real-page retrain from word-image maps: **69,366** training words, **1,763** evaluation
  words; patch accuracy **0.9921** (dev **0.989** / tam **0.995** / other **0.966**).
- Model: `training/local/script_cls.pt`.
- Training code: `training/scripts/script_cls/train_classifier.py`.
- Kaggle: `vishvasvasuki/manipravala-script-cls` and adaptation
  `vishvasvasuki/manipravala-script-cls-ft-replay`.

The patch classifier is the base for safe word/run splits (`pipeline.py::classify_word`).

## Production lessons (2026-10 / 2026-11)

- **Why it is the active classifier (`pipeline.py --linewise`):** After retraining on the 69,366 verified `word-image-maps` crops (0.9921 real-page accuracy), `script_cls.pt` + linewise strip packing produced `raw/drive/local/4.drive.md` on Volume 4 (1,569 faux pages, >99.8% text completeness, near-zero Latin hallucinations), outperforming both the wordwise Shiro-rekhā pipeline (`4.drive.wordwise.md`) and Kraken word OCR (`tesseract/4.md`).
- **Known pipeline bug with `"other"` labels:** `pipeline.py` currently composes only `script in ("dev", "tam")` and omits `"other"` segments. When a narrow digit (such as the middle `०` in `१०२`) or a parenthesis `)` is classified as `"other"`, it is dropped from the faux page (`।। १ २ ।।`, `(द्वया-`). Fix: absorb `"other"` runs into the adjacent `dev` or `tam` run before composing faux pages.
- **Limitation on dense intra-word Manipravalam (Volumes 1–3):** Earlier Volume 1 faux pages (`faux-pages-linewise-fixed`, Oct 24) were built *before* the Oct 26 real-page retraining of `script_cls.pt` and used the synthetic `word_cls.pt` override, which misrouted real Tamil words onto Devanagari pages. Even with the retrained `script_cls.pt`, splitting inside mixed words (`आचार्यसम्बन्धத்தினுடைய`) produces short bound-morpheme suffixes (`த்தினுடைய`, `வுக்கு`) that lose line context if not grouped or padded.