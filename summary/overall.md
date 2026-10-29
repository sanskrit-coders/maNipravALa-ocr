# Manipravalam OCR summary

This directory splits classifier-specific notes out of the legacy
`evaluation/summary.md`.

Classifier documents:

- [word-segmenter.md](lines_to_words.md) — GT-constrained word boxes; old-vs-fixed evaluation.
- [word-classifier.md](word-classifier.md) — word-level `dev/tam/other/mixed` classifier.
- [script-boundary-classifier.md](script-boundary-classifier.md) — mixed-word boundary
  segmenter, data, metrics, merged-box examples.
- [column-separator.md](column-separator.md) — per-column separator.
- [patch-script-classifier.md](patch-script-classifier.md) — base 48×48 patch classifier.
- [line-classifier.md](line-classifier.md) — experimental line classifier.

## Classifier I/O summary

| classifier | input | output labels |
|---|---|---|
| patch script | 48×48 grayscale patch | `dev/tam/other/null` per patch |
| word script | 64×256 word image | `dev/tam/other/mixed` per word |
| script boundary | 64×256 word image (+ optional base-projection channel) | boundary column → `dev` prefix + `tam` suffix |
| column separator | 64×256 word image | `dev/tam/other` per column (256) |
| line script | 64×1024 line image | `dev/tam/other` per line |

## Pipelines
`line separator` → `patch classifier` → `faux page construction` →
`Drive OCR` → `OCR collation`.


Alternate

`line separator` → `word separator (projection-valley)` → `word classifier` →
`script boundary classifier (Shiro-rekhā rule)` → `faux page construction` →
`Drive OCR` → `OCR collation`.

## Current state

- Word-image maps: **95,411 high-confidence box→text entries** across **2,203 pages**,
  stored as `data/RTS_sb_*/word-image-maps/{comm,core}/pNNNN.json` (render-and-DP alignment,
  confidence ≥ 0.45).
- Word boxes: rebuilt from Tesseract-TSV candidate spaces + DP; 112,948 boxes
  (58,403 sb_02 + 54,545 sb_01), 84,667 boundaries. 1,716 lines used the projection fallback.
- Word classifier retraining is blocked at the **GT line/word ↔ image assignment** level:
  e.g. `p0538` core line 4 has GT `कुरु’’` on a box whose image is Tamil. TSV alignment is
  improved, but labels are still wrong for some lines. Old `script_sep.pt` retained; faux
  pages not regenerated.
- Patch script classifier retrained on **training split** word-image maps and evaluated on
  **evaluation split** pages: patch accuracy **0.9921** (dev 0.989 / tam 0.995 / other 0.966).
  Model: `training/local/script_cls.pt`; backup of the synthetic model:
  `/tmp/opencode/script_cls_synth_backup.pt`.
- Script boundary: on 14,208 sb_01+sb_02 dev→tam words, the **projection-glyph rule** is best
  (boundary MAE 0.0320, p@5 0.718, segment P 0.964 / R 0.959). The combined image+projection
  CNN retrain reaches MAE 0.1551 / p@5 0.456 / segment P 0.891 / R 0.862; earlier image CNN
  variants were MAE 0.173–0.178. The rendered-Noto CNN is exact only on rendered images.
- GT-free projection-valley word separation with **TSV word boundaries** reaches boundary F1
  **0.745** at a **7 px** cutoff (P 0.809 / R 0.690). Earlier lower numbers came from bad
  projected word boxes, not from the gap distribution. See `lines_to_words.md`.
- Remaining gaps: **Tamil-first words** (1,645; ~4%) need multi-boundary/recursive treatment;
  the production word separator still needs GT-free word counts; the combined boundary CNN
  retrain is blocked by a Kaggle dataset upload issue.
- `pipeline.py` keeps safe ink-column splitting as the default; trained word/boundary/column
  models are opt-in.

## OCR

- **Mixed-script word OCR:** Kraken 7.1.1 VGSL, **4.1M params**, trained on **92,976**
  word crops from the training word-image maps; evaluated on **2,435** word crops from
  evaluation-split pages.
- **Training:** Kaggle GPU (`vishvasvasuki/rts-word-ocr-train`), dataset
  `vishvasvasuki/rts-word-image-ocr`, batch 16, early stopping; best checkpoint at epoch 3
  (13 epochs run, best val score 0.8900).
- **Evaluation:** character accuracy **90.34%**, word accuracy **80.53%**; per-script right:
  Tamil **94.31%**, Devanagari **88.81%**.
- Errors: 2,167 character errors — 306 insertions, 1,096 deletions, 765 substitutions.
- Model: `training/local/kraken_words_model/best_0.8900.safetensors`;
  metrics: `training/local/kraken_words_metrics.json`.

## Artifacts

- `training/scripts/script_cls/pipeline.py` — production pipeline.
- `data/script-separator/*.jsonl` — old/fixed word boxes, sb_01/sb_02 boundary data.
- `lines_to_words.md` — box source and old-vs-fixed evaluation.
- `training/local/script_sep.pt` — 4-class word model.
- `training/local/boundary_sep.pt` — optional shiro-trained boundary model.
- `training/local/boundary_image.pt`, `boundary_proj.pt` — comparison CNNs.
- `training/local/column_sep.pt` — optional column model.
- `training/local/script_cls.pt` — patch script classifier retrained on word-image maps.
- `data/RTS_sb_*/word-image-maps/` — per-page high-confidence word image→text maps.
- `evaluation/summary.md` — legacy full evaluation (block/line CER, baselines).
