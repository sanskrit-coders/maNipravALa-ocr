# Script boundary classifier (mixed word → devanAgarI/tamiL split)

Goal: given a mixed-script word image, find the boundary after the Devanagari prefix so
Devanagari and Tamil strips can be composed separately.

## Data

- `data/script-separator/train_words.jsonl`: 51,210 GT words from 1,070 `RTS_sb_02` pages,
  labeled `dev/tam/mixed/other`. No crop images are stored; training opens the source image
  boxes on the fly.
- `data/script-separator/train_refined.jsonl`: 24,890 mixed words with component-gap-refined
  boundaries.
- `data/script-separator/train_noto.jsonl`: boundaries from Noto Sans Devanagari/Tamil
  HarfBuzz advances and glyph-cluster counts.
- `data/script-separator/train_shiro.jsonl`: 18,175 high-confidence mixed words whose boundary
  is the right end of the shiro-rekhā stroke in the **original image box**.
- `data/script-separator/train_proj.jsonl`: 8,159 high-confidence words using base-projection
  glyph segmentation (DP) constrained to agree with the shiro-rekhā endpoint.

## Models and results

| model | labels | result |
|---|---|---|
| rendered Noto boundary CNN | 23,025 rendered words | MAE 0.0002, p@5 1.000, seg P 0.999 / R 0.999 — rendered domain only |
| original-box shiro boundary CNN | 18,175 shiro labels | MAE 0.116, p@5 0.643, seg P 0.916 / R 0.906 |
| column separator | 24,890 refined mixed words | column acc 0.715, boundary MAE 0.087 |
| base-projection feature handcrafted | 8,159 projection/shiro labels | label set in `train_proj.jsonl`; 2-channel training run failed with 0 loaded samples |

The rendered model is not used on scanned pages. The shiro model is stored at
`training/local/boundary_sep.pt` and wired as optional `--boundary-classifier`.

Training code:
- `training/scripts/script_cls/train_boundary_sep.py`
- `training/scripts/script_cls/train_render_boundary.py`
- label builders: `make_proj_labels.py`, `recompute_noto.py`, `shiro_label.py` (in `/tmp/opencode`)

## Why boxes are still imperfect (examples)

Many source “word” boxes contain multiple GT words. `train_words.jsonl` has **5,301 shared
boxes** (same image+box used by two or more GT tokens). Examples:

- `comm/p0032.jpg` box `(810, 217, 1008, 254)`: GT `व्याख्यातமாயிற்று.` **and** `இனி`.
- `comm/p0032.jpg` box `(566, 303, 907, 341)`: GT `प्रभावरक्षाधिकारமென்றும்` **and** `நான்கு`.
- `comm/p0032.jpg` box `(117, 608, 234, 646)`: GT `क्लेशங்` **and** `களை`.
- `comm/p0032.jpg` box `(280, 649, 649, 687)`: GT `साक्षाத்கரித்துக்கொண்டிருக்கும்` **and** `भगवाன்`.
- `core/p0033.jpg` box `(126, 121, 428, 166)`: GT `अशक्यमानது` **and** `அவனுக்கு`.
- `core/p0049.jpg` box `(838, 2, 1113, 59)`: GT `तत्त्वहितविषयमान`, but the crop actually
  contains Tamil `பிறப்பிக்கைக்கும்,` — a misaligned box.

These are why a correct shiro-rekhā endpoint can still be the wrong word boundary.

## Projection / glyph-count approach

`make_proj_labels.py` counts glyph clusters from Noto shaping (`k_dev`, `k_tam`), then uses a DP over the **base-region ink projection** to choose `k_dev+k_tam` glyph slots and sets the boundary after `k_dev` glyphs. Labels that disagree with the shiro-rekhā endpoint by more than 0.18 are discarded; 8,159 high-confidence words remain in `train_proj.jsonl`.

A 2-channel boundary CNN (image + base-projection channel) was attempted, but the last Kaggle run loaded **0 samples** due to a dataset/JSONL path issue and produced no metric. The label set itself is ready; the training loader needs one more fix.

## Next improvements

- Use base-projection valleys to number glyphs and set the boundary after the
  `k_dev`-th glyph (e.g. `त, त्त्व, हि, त, वि, ष, य | மா, ய்`).
- Add the base-projection profile as an extra input channel to the boundary CNN.
- Filter/repair merged boxes before training.
