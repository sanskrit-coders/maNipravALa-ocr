# Patch Script Classifier (`script_cls.pt`)

**Status:** **Active production model** (used in Stage 3 of [`current-pipeline.md`](../current-pipeline.md)).

## Input and Output Specification

| Property | Specification |
|---|---|
| **Weights** | `training/local/script_cls.pt` (synthetic pretrain backup: `training/local/script_cls_synth_backup.pt`) |
| **Metrics file** | `training/local/script_cls_metrics.json` |
| **Training script** | [`training/scripts/script_cls/train_classifier.py`](../../training/scripts/script_cls/train_classifier.py) |
| **Architecture** | 48×48 grayscale residual CNN (`Net`), 4 output classes |
| **Input** | One 48×48 grayscale patch centered at column `x` along a word box (`STRIDE = 8` px; vertically scaled to 48 px when `h > 48` or centered on a 48×48 white canvas when `h <= 48`), normalized as `1.0 - pixel / 255.0` |
| **Output labels** | `0: dev`, `1: tam`, `2: other`, `3: null` (`null` = blank/low-confidence background, discarded) |
| **Granularity** | One class + softmax probability per 8-px sliding patch; merged into contiguous word/sub-word script runs |

## Training & Validation Metrics

- **Synthetic pretraining (`vishvasvasuki/manipravala-script-cls`):**
  - Validation accuracy: **0.996** (`dev` 1.000 / `tam` 0.998 / `other` 0.985 / `null` 0.986).
- **Real-scan retraining on `word-image-maps` (`vishvasvasuki/manipravala-script-cls-ft-replay`, 2026-10-26):**
  - Trained on **69,366** high-confidence real word crops (`confidence >= 0.45`) from `data/RTS_sb_01/word-image-maps` and `data/RTS_sb_02/word-image-maps`; evaluated on **1,763** held-out real words.
  - Overall real-patch validation accuracy: **0.9921** (`dev` **0.989** / `tam` **0.995** / `other` **0.966**).

## Integration in `pipeline.py`

In [`pipeline.py::classify_word`](../../training/scripts/script_cls/pipeline.py#L524-L577):
1. **Center-patch sampling for narrow words:** When a word box has width `w <= 40` px, samples the true horizontal midpoint `[(x0 + x1) // 2]` instead of clamping to `x0 + 24`.
2. **Mean-probability fallback:** Keeps patches with `cls != 3` (`null`) and `prob > 0.45`; if all patches in a narrow word fall below `0.45`, uses the mean softmax probability argmax across `dev`/`tam`/`other` so short words are never dropped.
3. **Run merging ([`_merge_short_runs`](../../training/scripts/script_cls/pipeline.py#L482-L521)):** Collapses adjacent identical labels and folds `"other"` runs or runs narrower than `max(20, int(0.45 * h))` into the larger neighbor.
4. **Minimum-ink transition split ([`_safe_split_x`](../../training/scripts/script_cls/pipeline.py#L461-L479)):** At `dev ↔ tam` transitions inside a mixed Maṇipravāḷa word, snaps the split point to the nearest `0`-ink column within `±14` px, or falls back to the minimum-ink column (`<= 6` dark pixels) when a 1–3 px Shiro-rekhā overhang touches the first Tamil glyph.
5. **Line-level `"other"` absorption ([`_is_punct_or_other`](../../training/scripts/script_cls/pipeline.py#L1010-L1033)):** Whole-word `"other"` tokens (dandas `।`, numerals, hyphens `-`, ellipses `...`, quotes) are absorbed into the adjacent `dev` or `tam` run on the same line so no punctuation or verse digits are ever lost.
