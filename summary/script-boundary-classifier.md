# Script boundary classifier (mixed word → devanAgarI/tamiL split)

## Input and output labels

| | |
|---|---|
| **Input** | One 64×256 grayscale word image. The 2-channel variant adds a base-region ink-projection channel (same 64×256 shape). |
| **Output labels** | A boundary heatmap over 256 columns; `argmax` gives one boundary fraction. The downstream split is `dev` prefix + `tam` suffix. |
| **Granularity** | One boundary per mixed word (not one label per column). |


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

## Tamil-first mixed words in GT

Yes. Scanning all `RTS_sb_02/gt/**/*.txt` files, **1,645** whitespace-delimited mixed
tokens have Tamil before Devanagari, spread over **747 files**. Out of 41,504 mixed tokens,
about 4.0% are Tamil-first. Patterns:

| pattern | count |
|---|---:|
| `dev-tam` | 38,416 |
| `tam-dev-tam` | 1,078 |
| `dev-tam-dev-tam` | 712 |
| `dev-tam-dev` | 707 |
| `tam-dev` | 465 |
| `tam-dev-tam-dev` | 57 |
| `tam-dev-tam-dev-tam` | 45 |
| other dev-start | 24 |

Examples:

- `comm/p0032.txt:5` — `இருபத்திரண்டधिकाररूपமான` (`tam-dev-tam`)
- `comm/p0033.txt:9` — `இவ்வधिकारार्थம்` (`tam-dev-tam`)
- `comm/p0034.txt:3` — `இவ்வर्थ` (`tam-dev`)
- `comm/p0036.txt:1` — `யிत्यादिना` (`tam-dev`)
- `comm/p0037.txt:1` — `இவ்வपहारத்தை` (`tam-dev-tam`)
- `comm/p0038.txt:3` — `என்றभिप्रायம்.` (`tam-dev-tam`)

Therefore a boundary segmenter must not assume Devanagari is always the first script; it
should either detect the first script run or support `tam-dev(-tam)` patterns.

## Where the word boxes came from, and the fix

The original training boxes came from `pipeline.word_segments`: fixed 4 px horizontal
ink-gap projection, followed by a width-based DP mapping of GT words onto those boxes.
This merged/misplaced words (see examples below).

Replacement (`/tmp/opencode/fix_boxes.py`, generating
`data/script-separator/train_words_fixed.jsonl`):

1. Segment image lines.
2. Assign contiguous GT words to lines with a width/glyph-count DP.
3. Split each line into exactly the assigned number of words at **base-projection valleys**,
   using GT glyph counts to set expected cut positions.
4. Keep labels from GT; store only image path + box + text + runs.

This produces **58,403** fixed boxes (all mixed words plus bounded pure/other words per page).
The `p0049` example is corrected: `तत्त्वहितविषयமान` now has box `(465,2,756,59)`, not the
Tamil `பிறப்பிக்கைக்கும்,` crop.

## Glyph-numbered projection labels

- `make_proj_labels.py` counts glyph clusters per script run from Noto shaping
  (`k_dev`, `k_tam`), then uses DP over the **base-region ink projection** to choose
  `k_dev+k_tam` glyph slots. The boundary is after the `k_dev`-th glyph.
- Example target: `त, त्त्व, हि, त, वि, ष, य | மா, ய்` — boundary after `य`.
- 8,159 high-confidence words agree with the shiro-rekhā endpoint and are stored in
  `data/script-separator/train_proj.jsonl`.
- A 2-channel boundary CNN (image + base-projection channel) was specified, but the last
  Kaggle run loaded **0 samples** due to a dataset/JSONL path issue; it produced no metric yet.

## Next improvements

- Train the 2-channel (image + base-projection) boundary CNN on `train_proj.jsonl` once the
  loader path issue is fixed.
- Handle Tamil-first words explicitly (`tam-dev(-tam)` is ~4% of mixed tokens).
- Filter/repair remaining merged boxes before retraining.