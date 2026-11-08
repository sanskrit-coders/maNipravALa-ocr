# Word Classifier (`script_sep.pt` / `word_cls.pt`)

**Status:** **Obsolete / retired** (do **not** pass `--word-classifier` to [`pipeline.py`](../../../training/scripts/script_cls/pipeline.py)).

## Input and Output Specification

| | |
|---|---|
| **Weights** | `training/local/script_sep.pt`, `training/local/word_cls.pt` |
| **Metrics files** | `training/local/script_sep_metrics.json`, `training/local/word_cls_metrics.json` |
| **Training scripts** | [`training/scripts/script_cls/train_script_sep.py`](../../../training/scripts/script_cls/train_script_sep.py), [`train_word_cls.py`](../../../training/scripts/script_cls/train_word_cls.py) |
| **Input** | One 64×256 grayscale word image (word box resized/padded; text height 24–48 px) |
| **Output labels** | 4 classes: `dev`, `tam`, `other`, `mixed` |
| **Granularity** | One label per whole word image |

## Why Moved to `models/obsolete/`

1. **Synthetic-vs-real domain-shortcut bug in `train_script_sep.py` (`load_real`):**
   - Inspection of [`train_script_sep.py::load_real`](../../../training/scripts/script_cls/train_script_sep.py#L95-L102) revealed that **every single real scanned word crop** loaded from `p0036.jsonl` was hardcoded to `CLS["mixed"]`, while `dev`, `tam`, and `other` were generated exclusively from synthetic computer fonts (`synth_word`).
   - Consequently, `script_sep.pt` learned a domain shortcut: **"scanned paper texture = `mixed`, crisp synthetic font = pure"**, causing it to mislabel almost all real scanned words as `mixed` or misroute Tamil words onto Devanāgarī faux pages.
2. **Corrupted GT line-assignment in `train_words_fixed.jsonl`:**
   - Retraining on `train_words_fixed.jsonl` (112,948 boxes) collapsed to real `dev` **0.676**, `tam` **0.777**, `mixed` **0.151** because `train_words_fixed.jsonl` bypassed the verified `word-image-maps` (`confidence >= 0.45`) and contained ~5,301 merged/misaligned boxes (e.g., `RTS_sb_02/core/p0538.jpg` line 4 assigned GT `कुरु’’` to a Tamil crop `ிளிஈத்தை`).
3. **End-to-end failure in `faux-pages-wordwise-shiro-rekhA` (`4.drive.wordwise.md`) and `faux-pages-linewise-fixed`:**
   - Passing `--word-classifier` overrode the accurate 48×48 patch classifier (`script_cls.pt`, **0.9921** real-page accuracy), misrouting whole Tamil words onto Devanāgarī faux pages and triggering false Shiro-rekhā splits on pure words.
