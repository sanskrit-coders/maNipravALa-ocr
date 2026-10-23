# Word segmenter (GT-constrained line + projection-valley boxes)

## Input and output labels

| | |
|---|---|
| **Input** | One line image (grayscale/binary) plus the ordered GT word sequence for the page. |
| **Output** | One bounding box per GT word (`[x0,y0,x1,y1]`) with the GT `dev/tam/mixed/other` label and word text. |
| **Granularity** | One box per whitespace-delimited GT word. |

## Where the old boxes came from

The old dataset used `pipeline.word_segments`: a fixed **4 px** horizontal ink-gap
projection, followed by a width-based DP mapping GT words onto those boxes. On this book this
merged words, split words at glyph gaps, and sometimes mapped a Devanagari GT token to a Tamil
crop.

## Fixed segmenter

`/tmp/opencode/fix_boxes.py`:

1. Segment image lines with `pipeline.segment_lines`.
2. Assign contiguous GT words to lines with a dynamic program over line width and GT glyph
   counts.
3. Split each line into exactly the assigned number of words at **base-projection valleys**
   (lower part of the line plus full ink projection), using GT glyph counts for expected cut
   positions.
4. Keep the GT label (`dev/tam/mixed/other`) and store the image path + box + text + runs.

Output: `data/script-separator/train_words_fixed.jsonl` — **58,403** boxes from 1,119 images
(all mixed words plus bounded pure/other words per page). No crop images are stored.

## Evaluation: fixed vs old projection+DP

Compared on the **35,274 shared `(image, gt_index)` entries**:

| metric | result |
|---|---:|
| shared/duplicate boxes, old | **5,301** |
| shared/duplicate boxes, fixed | **0** |
| old↔fixed box IoU, median | **0.0** |
| old↔fixed box IoU ≤ 0.5 | **31,495 / 35,274 (89.3%)** |
| old↔fixed box IoU ≤ 0.2 | **29,175 / 35,274 (82.7%)** |
| width/char ratio outside 0.55–1.8, old | **11,074 / 51,089 (21.7%)** |
| width/char ratio outside 0.55–1.8, fixed | **3,489 / 58,403 (6.0%)** |

The median old↔fixed IoU is **0.0**: the old mapping frequently pointed at an entirely
different word or a merged region.

Concrete examples:

- `comm/p0032.jpg` old box `(810,217,1008,254)` was shared by GT
  `व्याख्यातமாயிற்று.` **and** `இனி`.
- `comm/p0032.jpg` old box `(566,303,907,341)` was shared by
  `प्रभावरक्षाधिकारமென்றும்` **and** `நான்கு`.
- `core/p0049.jpg` old box `(838,2,1113,59)` for `तत्त्वहितविषयमान` actually contained
  Tamil `பிறப்பிக்கைக்கும்,`; the fixed box is `(465,2,756,59)`.

## Limitations

- Line assignment depends on estimated px-per-glyph; pages with unusual spacing can still be
  off.
- The fixed set is a better base for script/word/boundary training, but it is not perfect.
- Tamil-first mixed words still require a multi-boundary or column/glyph-level model.
