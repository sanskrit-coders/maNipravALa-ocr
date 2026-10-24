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

## Current pipeline

`line separator` → `word separator (projection-valley)` → `word classifier` →
`script boundary classifier (projection-glyph rule)` → `faux page construction` →
`Drive OCR` → `OCR collation`.

## Current state

- Word boxes: the earlier **projection+DP** mapping is replaced by the GT-constrained
  **base-projection valley** segmenter. It produces 58,403 sb_02 boxes + 54,545 sb_01 boxes;
  sb_02 shared/merged boxes drop from 5,301 to 0, and width/char outliers from 21.7% to 6.0%.
- Word classifier (4-class `dev/tam/other/mixed`) retrained on sb_01+sb_02 fixed boxes:
  real per-class dev 0.999 / tam 1.000 / other 0.997 / mixed 0.983; synthetic 0.996.
- Script boundary: on 14,208 sb_01+sb_02 dev→tam words, the **projection-glyph rule** is best
  (boundary MAE 0.0320, p@5 0.718, segment P 0.964 / R 0.959). Image CNN variants are weaker
  (MAE 0.173–0.178, P ~0.88). The rendered-Noto CNN is exact only on rendered images.
- Remaining gaps: **Tamil-first words** (1,645; ~4%) need multi-boundary/recursive treatment;
  the production word separator still needs GT-free word counts; the combined boundary CNN
  retrain is blocked by a Kaggle dataset upload issue.
- `pipeline.py` keeps safe ink-column splitting as the default; trained word/boundary/column
  models are opt-in.

## Artifacts

- `training/scripts/script_cls/pipeline.py` — production pipeline.
- `data/script-separator/*.jsonl` — old/fixed word boxes, sb_01/sb_02 boundary data.
- `lines_to_words.md` — box source and old-vs-fixed evaluation.
- `training/local/script_sep.pt` — 4-class word model.
- `training/local/boundary_sep.pt` — optional shiro-trained boundary model.
- `training/local/boundary_image.pt`, `boundary_proj.pt` — comparison CNNs.
- `training/local/column_sep.pt` — optional column model.
- `training/local/script_cls.pt` — patch script classifier.
- `evaluation/summary.md` — legacy full evaluation (block/line CER, baselines).
