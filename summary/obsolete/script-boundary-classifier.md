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
- `data/script-separator/train_words_fixed.jsonl`: 58,403 GT-constrained boxes from `RTS_sb_02`.
- `data/script-separator/train_words_fixed_sb01.jsonl`: same fixed-box generator applied to
  `RTS_sb_01` — **54,545** boxes.
- `data/script-separator/train_boundary_fixed_sb01.jsonl`: **6,513** high-confidence
  dev→tam boundary labels from sb_01. Combined with the 7,695 sb_02 labels, there are
  **14,208** boundary words ready for the combined retrain.

## Models compared

Rule baselines below are on the combined **14,208** sb_01+sb_02 dev→tam words; the image CNN rows are from the earlier **7,695 sb_02** run (combined CNN retrain blocked by the Kaggle dataset upload).
(`train_boundary_fixed.jsonl`); labels are the consensus of the shiro-rekhā endpoint and the
base-projection `k_dev`-th-glyph rule; page-disjoint train/val: 6,907 / 788.

1. **Shiro-rekhā rule** — no learning; boundary is the right end of the longest horizontal
   headline stroke.
2. **Projection-glyph rule** — no learning; split the word into `k_dev+k_tam` glyph slots at
   base-projection valleys; boundary after the `k_dev`-th glyph.
3. **Image-only boundary CNN** — 64×256 word image; 256-column boundary heatmap.
4. **Image + base-projection CNN** — same CNN with a second 64×256 channel containing the
   base-region ink projection.
5. **Rendered Noto CNN** — trained on Noto-rendered words; exact in rendered domain, not used
   on scans.
6. **Column CNN** — per-column `dev/tam/other` labels, boundary from the first transition.

| method | MAE | p@5 | segment P | segment R |
|---|---:|---:|---:|---:|
| shiro-rekhā rule (sb_01+sb_02) | **0.0320** | **0.718** | 0.960 | 0.956 |
| projection-glyph rule (sb_01+sb_02) | **0.0320** | **0.718** | **0.964** | **0.959** |
| image-only CNN (sb_02 train) | 0.1777 | 0.412 | 0.878 | 0.842 |
| image+projection CNN (sb_02 train) | 0.1726 | 0.423 | 0.882 | 0.850 |
| **image+projection CNN (sb_01+sb_02 retrain)** | **0.1551** | **0.456** | **0.891** | **0.862** |
| rendered Noto CNN | 0.0002 | 1.000 | 0.999 | 0.999 (rendered domain only) |
| column CNN (older data) | 0.087 | — | — | — |

The **projection-glyph rule is still the best source-domain boundary classifier**. The
combined CNN improves substantially over the earlier image-only CNN but remains well behind
the rule (MAE 0.1551 vs 0.0320). The rendered model is exact only on rendered images.

Best boundary models saved:
- `training/local/boundary_image.pt`
- `training/local/boundary_proj.pt`
- `training/local/boundary_combined.pt` — combined retrain (14,208 words; 1,431 page-disjoint
  validation words; MAE 0.1551, p@5 0.456, segment P 0.891 / R 0.862).

## Why the earlier evaluation had only 7,695 words

That number was **not** a limit of the method: it was the intersection of
`RTS_sb_02` fixed boxes, dev→tam order, and agreement between the shiro-rekhā and
base-projection endpoints. `RTS_sb_01` **was not used at all** in that run. Its 54,545 fixed
boxes and 6,513 boundary labels have now been generated and added to the combined training set.

For the next retrain, use both:
- `RTS_sb_02` fixed boxes (`train_words_fixed.jsonl`);
- `RTS_sb_01` fixed boxes (`train_words_fixed_sb01.jsonl`).

The strict agreement filter is still useful for evaluation, but training can use the union of
the two corpora with a confidence weight instead of discarding all disagreeing boxes.

## Tamil→Devanagari and recursive segmentation

The single-boundary model assumes `dev` prefix → `tam` suffix. GT has 1,645 Tamil-first
mixed tokens, so the general operator is a recursive script-run segmenter:

1. Run the word classifier on a word image.
2. If pure → return it.
3. If mixed → apply a boundary scorer to propose the first script transition.
4. Split; run the word classifier on both left and right segments.
5. Recurse on any segment still labeled `mixed` (this handles `tam-dev-tam`,
   `dev-tam-dev`, etc.).
6. Stop when every segment is pure or no valid boundary remains.

For Tamil-first words, either:
- train a **second boundary head** on horizontally mirrored / Tamil-first examples, or
- use a **direction-agnostic multi-transition model** (column/glyph-level labels).

The recursion uses the word classifier as the stopping criterion, so a bad word label is
recoverable by the boundary check; conversely a boundary that yields one pure and one mixed
segment triggers another split.

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