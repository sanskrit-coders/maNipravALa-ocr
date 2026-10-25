# Line → words (projection-valley separator)

## Input and output labels

| | |
|---|---|
| **Input** | One grayscale line image. |
| **Output** | Word boxes `[x0,y0,x1,y1]` (no GT word sequence). |
| **Granularity** | One box per detected word boundary. |

## Current model

1. Use the independent Tesseract TSV word boxes to get candidate word spaces.
2. Candidate boundaries are the gaps between consecutive TSV boxes. Image-projection gaps
   (full-height plus sliding-window slant) are the fallback when TSV has too few boxes.
3. A word-width DP assigns the GT words to candidate gaps, so boxes are cut at real spaces
   rather than at proportional glyph-count positions.
4. All gaps inside a selected word box are negatives.


## Rebuilt boxes

The rebuilt JSONLs now contain **112,948 word boxes** (58,403 sb_02 + 54,545 sb_01) cut at
TSV/actual spaces. **84,667 word boundaries** were rebuilt; **1,716 lines** fell back to the
projection-gap path.

## Independent check with Tesseract word boxes

Using the TSV word boxes as an independent check (100 pages, 2,043 lines):

- word gaps: **11,130**
- internal gaps: **61,574**
- no gap found near a TSV boundary: 2,101

Word gaps are clearly larger than internal gaps; the old projected boxes had put some
boundaries inside words.

## Minimum-gap sweep

With TSV word boundaries and projection gaps:

| gap population | p25 | p50 | p75 | p95 |
|---|---:|---:|---:|---:|
| word gaps | 5 px | 10 px | 15 px | 23 px |
| internal gaps | 2 px | 3 px | 4 px | 5 px |

The populations are now clearly separated. Sweeping a minimum gap:

| min gap | boundary P | boundary R | boundary F1 | FP/line | FN/line |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.147 | 0.994 | 0.256 | 32.1 | 0.0 |
| 3 | 0.213 | 0.905 | 0.345 | 18.5 | 0.5 |
| 4 | 0.338 | 0.826 | 0.479 | 9.0 | 1.0 |
| 5 | 0.530 | 0.771 | 0.628 | 3.8 | 1.3 |
| 6 | 0.742 | 0.725 | 0.733 | 1.4 | 1.5 |
| **7** | **0.809** | **0.690** | **0.745** | **0.9** | **1.7** |
| 8 | 0.835 | 0.655 | 0.734 | 0.7 | 1.9 |
| 9 | 0.878 | 0.609 | 0.719 | 0.5 | 2.2 |
| 10 | 0.878 | 0.558 | 0.682 | 0.4 | 2.5 |
| 12 | 0.864 | 0.436 | 0.579 | 0.4 | 3.1 |
| 15 | 0.828 | 0.275 | 0.413 | 0.3 | 4.0 |

**7 px at 300 dpi is the best single cutoff** (P 0.809 / R 0.690 / F1 0.745).

## Boundary precision / recall / F1

- **Boundary precision** = TP / (TP + FP)
- **Boundary recall** = TP / (TP + FN)
- **Boundary F1** = harmonic mean

On [p0330 line 0](../data/RTS_sb_02/images/comm/p0330.jpg), the rebuilt word gaps are:

`விஷயமென்கிறார்(9px)प्रज्ञाप्रासादमिति(4px)।(16px)प्रज्ञाप्रासादं(7px)இங்கு(9px)प्रज्ञैயாவது(9px)महाप्रभावனான(9px)आचार्योपदेशजन्यமாயும்`

The large gaps are between words. The only internal gap above 5 px is an 8 px gap inside
`விஷயமென்கிறார்`; all other internal gaps on this line are 2–5 px. This confirms that the
earlier `word(8px)`/`word(9px)` intra-word examples were artefacts of the old projected boxes.

## Next steps

- Retrain the word classifier on the rebuilt boxes.
- Train the line-image detector on the rebuilt boundaries.
- Regenerate the faux pages only after those checks pass.
