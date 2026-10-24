# Lines to words 
(GT-constrained line + projection-valley boxes)

## Input and output labels

| | |
|---|---|
| **Input** | One line image (grayscale/binary) plus the ordered GT word sequence for the page. |
| **Output** | One bounding box per GT word (`[x0,y0,x1,y1]`) with the GT `dev/tam/mixed/other` label and word text. |
| **Granularity** | One box per whitespace-delimited GT word. |

## Projection+DP model

The earlier box generator split each line at a fixed **4 px** horizontal ink gap
(`pipeline.word_segments`), then used a width-based dynamic program to assign contiguous GT
words to those boxes. It did not use GT glyph counts.

Consequences:
- adjacent GT words merged into one box;
- some words split at internal glyph gaps;
- occasional Devanagari GT words mapped to Tamil regions.

## Base-projection valley model

`/tmp/opencode/fix_boxes.py`:

1. Segment image lines with `pipeline.segment_lines`.
2. Assign contiguous GT words to lines with a dynamic program over line width and GT glyph
   counts.
3. Split each line into exactly the assigned number of words at **base-projection valleys**
   (lower part of the line plus full ink projection), using GT glyph counts for expected cut
   positions.
4. Keep the GT label (`dev/tam/mixed/other`) and store image path + box + text + runs.

Output: `data/script-separator/train_words_fixed.jsonl` — **58,403** boxes from 1,119 images
(all mixed words plus bounded pure/other words per page). No crop images are stored.

## Evaluation: projection+DP vs base-projection valley

Compared on the **35,274 shared `(image, gt_index)` entries**:

| metric | result |
|---|---:|
| shared/duplicate boxes, projection+DP | **5,301** |
| shared/duplicate boxes, base-projection valley | **0** |
| projection+DP ↔ base-projection valley box IoU, median | **0.0** |
| projection+DP ↔ base-projection valley box IoU ≤ 0.5 | **31,495 / 35,274 (89.3%)** |
| projection+DP ↔ base-projection valley box IoU ≤ 0.2 | **29,175 / 35,274 (82.7%)** |
| width/char ratio outside 0.55–1.8, projection+DP | **11,074 / 51,089 (21.7%)** |
| width/char ratio outside 0.55–1.8, base-projection valley | **3,489 / 58,403 (6.0%)** |

The median projection+DP ↔ base-projection valley IoU is **0.0**: the projection+DP mapping
frequently pointed at an entirely different word or a merged region.

Concrete examples:

- `comm/p0032.jpg` projection+DP box `(810,217,1008,254)` was shared by GT
  `व्याख्यातமாயிற்று.` **and** `இனி`.
- `comm/p0032.jpg` projection+DP box `(566,303,907,341)` was shared by
  `प्रभावरक्षाधिकारமென்றும்` **and** `நான்கு`.
- `comm/p0032.jpg` projection+DP box `(117,608,234,646)` was shared by `क्लेशங்` **and**
  `களை`.
- `comm/p0032.jpg` projection+DP box `(280,649,649,687)` was shared by
  `साक्षाத்கரித்துக்கொண்டிருக்கும்` **and** `भगवाன்`.
- `core/p0033.jpg` projection+DP box `(126,121,428,166)` was shared by `अशक्यमानது`
  **and** `அவனுக்கு`.
- `core/p0049.jpg` projection+DP box `(838,2,1113,59)` for `तत्त्वहितविषयमान` actually contained Tamil `பிறப்பிக்கைக்கும்,`; the base-projection valley box is `(465,2,756,59)`.

## Limitations

- Line assignment depends on the estimated px-per-glyph; unusual spacing can still be off.
- The base-projection valley set is a better base for script/word/boundary training, but not
  perfect.
- Tamil-first mixed words still require a multi-boundary or column/glyph-level model.
