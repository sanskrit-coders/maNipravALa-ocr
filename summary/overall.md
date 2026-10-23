# Manipravalam OCR summary

This directory splits classifier-specific notes out of the legacy
`evaluation/summary.md`.

Classifier documents:

- [word-segmenter.md](word-segmenter.md) — GT-constrained word boxes; old-vs-fixed evaluation.
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

## Current state

- The word-level classifier is near-perfect on real page-disjoint validation
  (dev 0.999 / tam 1.000 / other 0.993 / mixed 0.986).
- The remaining production blocker is **word-box quality and the mixed-word split**, not
  classifier capacity. 5,301 GT words share merged boxes in `RTS_sb_02`.
- `pipeline.py` keeps safe ink-column splitting as the default; word/boundary/column models
  are opt-in.

## Artifacts

- `training/scripts/script_cls/pipeline.py` — production pipeline.
- `data/script-separator/*.jsonl` — word/boundary/boundary-feature training data.
- `training/local/script_sep.pt` — 4-class word model.
- `training/local/boundary_sep.pt` — optional boundary model.
- `training/local/column_sep.pt` — optional column model.
- `training/local/script_cls.pt` — patch script classifier.
- `evaluation/summary.md` — legacy full evaluation (block/line CER, baselines).
