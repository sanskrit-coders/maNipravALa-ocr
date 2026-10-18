# data — canonical RTS volumes (images first, GT later)

- `data/RTS_sb_01/` = Vol.1 (01.pdf, 630 pp, adhikaras 00-22). `pages_raw/pXXXX.jpg` (150 DPI JPEG, pdf_index),
  `images/{core,comm}/pXXXX.jpg` (deskewed block crops via detect_blocks),
  `metadata/{pages.csv,blocks.json,skew.json}`, `gt/{core,comm}/` (page-part GT; gold for eval blocks, auto/needs-review otherwise).
- `data/RTS_sb_02/` = Vol.2 (02.pdf, 616 pp, adhikaras 23-32). Same layout; GT deferred (images only for now).
- Splits are path lists partitioning the above (no copies):
  `training/split_train.jsonl`, `evaluation/split_eval.jsonl` (+ `evaluation/eval_index.jsonl`),
  each row `{image, gt, page, block, unit, status}` with repo-relative `data/...` paths.
- Never train on eval paths. Rerun idempotently:
  `python3 training/scripts/render_pages.py --vol 01 --dpi 150 --offset01 30`
  `python3 -m doc_curation.image.checker.detect_blocks --vol 01` (run from the
  `doc_curation` repo root; `--vol 02` same; `--root` or `$MANIPRAVALA_OCR_ROOT`
  point at this repo, else a sibling `maNipravALa-ocr` is auto-detected).

## Header-fix note (first-line cut)
`detect_blocks.py` fallback `header_bottom=10%` cut the first text line on pages without
running headers (front matter; text starts ~7-8%). Fixed to URL-aware `4.5%+8px`
(`doc_curation/doc_curation/image/checker/detect_blocks.py::split_page`); separator logic unchanged (0.3-width merged).
The fragmentary-header fallback now honours rule-like fragments only: verse shirorekhas
(long horizontal strokes with text below, ink_below >= 0.5) had pushed `header_bottom` into
the text (vol02 p0598: verse 74's second-line top bar moved it from 87 to 245, cutting the
verse in half). Fix restored cut first lines / full verses on 15 vol02 cores (p0229 245->87,
p0512 230->86, p0598 245->87, ...); vol01 unaffected. Section-opening pages keep the
title + wavy rule + verse in the core top (p0588 convention).
Verified: `RTS_sb_02/images/comm/p0029.jpg` now starts `ஸ்ரீமத் ரஹஸ்யத்ரயஸாரம்…`,
`p0030.jpg` starts `இப்படிப்பட்ட அத்புதமான…` (was cut).

## Deskew fix (main split fix; core was silently merged into comm)
Scans are tilted up to ~2deg. A tilted separator rule is eroded by the wide horizontal
opening, so no row kept a 0.35w run and the core was never split from the commentary
(reported: vol01 p0041/p0051, vol02 p0048, plus many more). `split_page` now deskews each
page first: angle = argmax of dark px surviving a 0.6w horizontal opening over ±2deg
(`estimate_skew`), cached in `metadata/skew.json` so re-runs are fast (`--recompute-skew`
to redo). Crops are written from the deskewed page.

Three follow-up lessons (all found after user review of vol01):
- NN rotation of a 1-2px line leaves aliasing gaps; a 3px horizontal close bridges them
  before the opening (5px already bridges word gaps on dense pages and invents full-width
  "rules" at wrong angles, vol02 p0264/p0595). Without the 3px close, vol01 p0086 (1.1deg
  tilt, thin rule) scored 0 -> no deskew -> separator missed -> its mUlam (adhikara title +
  verse) leaked into comm. Now core 125-629, sep 633, comm 639+.
- The book marks the END of every mUlam passage with a SHORT CENTRED rule (0.32-0.45w,
  e.g. vol01 p0070: 0.38w at 1.4deg) before the passage's commentary starts. Such rules can
  never win the 0.6w metric, so when the first pass finds no separator the page is
  re-searched at fine angles for a thin (<=8px @0.5 scale) short rule with clear space
  below (`refine_rule_angle`), and the refined angle is cached. This recovered ~117 cores
  in vol01 (p0070-class) and ~30 in vol02. Verified via GT text: vol01 p0587 bold top
  matches core unit 21-21.6 word-for-word, below-rule matches its commentary; like p0109
  (mUlam 24-24.30) and p0050 (mUlam 00-0.11).
- Ghost pages (backs of plates/adhyAya pages with only faint mirrored bleed-through;
  0.5th-pct gray>100 over body rows) yield bogus full-page comm crops and are skipped like
  plates (23 pp in vol01 incl. p0009/p0085/p0095/p0115, 22 in vol02; their verify metric:
  real text pages, even low-ink p0307/p0262, have 0.5th-pct gray <=3).

Layout insight this fixed: mUlam and commentary alternate. Each mUlam unit is displayed
in bold at a page top (usually right below the header), a short centred rule ends it, then
its commentary runs below, often spilling the next page (that spill page has no top core).
So cores ~ units: 525+554 = 1079 core crops (plus vol01 p0042 kept as an
`absent-by-design` marker) vs 1073 GT units (some units share a page or are
front-matter-only; near-parity, so no gross over-detection).

- Result: Vol.1 630 pages_raw / 599 comm / 525 core (1125 blocks), Vol.2 616 / 586 / 554
  (1140 blocks); colour plates (p0012/14/16/18/20/22/24/26 each vol) and ghost pages excluded.
- Fixed examples: vol01 p0041 (core 118-532, sep 537), p0051 (139-464, 468), p0086/p0088,
  p0070-class; vol02 p0041/0048/0051 all split. False cores removed: the photo plates,
  vol01 p0172, both title pages p0000 and vol02 p0005. vol02 p0109 was REMOVED by a bad
  deskew (-1.1deg from the old estimator) and has been RESTORED (it is genuine: mUlam
  24-24.30 above the rule at 954, verified against `data/texts/core/24_...`).
- Rule-vs-text-headline: only lines with clear space below (ink_below<0.15 in a 10px band)
  are treated as rules. Devanagari shirorekhas / Tamil ligature strokes have text under them
  and are rejected. Fixes vol01 p0059 (a 451px headline above the true 1005px rule had been
  picked as separator -> core truncated to 3 of 4 verse lines, and comm included the 4th
  line; now core 118-369, sep 373.5, comm 380+). Also corrected seps vol01 p0218 (556->674),
  vol02 p0410 (573->630), p0564 (258->455).
- No previously-correct split changed (no separator lost or shifted >15px).
- Independent audit for missed separators (the p0070/p0086 error class): on every page
  without a core, search raw binarized rows for thin near-horizontal segments (HoughLinesP,
  >=0.33w, clear space below) at any small angle. After the fixes: vol01 0 flags; vol02 only
  p0381, visually confirmed a text-line artifact (no rule, no bold block).

## Wavy/curved separator rules (p0228-class, user-reported)
Some mUlam-end rules are not straight: vol01 p0228's rule is fragmented across y268-288
(a wavy line whose pieces never form one >=0.32w row), and vol02 p0385/p0442/p0365 end in
long ARCS drooping 20+px at both ends. Such pages kept their core merged into comm.
Fixes in `detect_blocks.py`:
- Hough fallback in `refine_rule_angle` (`_hough_rule`): HoughLinesP segments (>=0.08w,
  |angle|<=2.5deg), clustered per angle/sub-degree with intercept tolerance, merged across
  gaps, accepted if total span >=0.32w, thin, and with clear space below. A text-chained
  false positive class (vol01 p0058/p0490: Hough linking letter strokes over several text
  rows) is rejected by an "ink above" check (row band above the line must be <=0.10: true
  rules scored <=0.018, text chains ~0.32).
- If the refined angle still yields no separator, `split_page` builds the separator
  manually from the refined pass extent (rotate endpoints by the refined angle, keep if
  width >0.28w and 0.10h < y < 0.92h).
- Rule extent measurement raised its downward cap 20->28px so the arc's dipping ends are
  cleared (vol02 p0385 sep y338, comm now 367+; p0442 comm 291+); a text-safe clamp keeps
  the dipped tail from shaving the next text line's tops (p0323 rule tail ends 406, text
  starts 408, comm kept at 407).
- Core-side crest lift: a tilted/wavy rule's top can rise 5-13px above the detected row
  (vol01 p0330 rows 268-279). The core now ends above the rule's true top when the rows
  between are rule-like (run >=0.10w) and the two rows above the tip are blank (so merged
  text descenders cannot pull the crop into text). Fixed rule-ink leaks at the bottom of
  core crops on 26+1 pages (vol01 p0268/p0418-class), verified by a crop-edge audit (max
  contiguous dark run <=0.25w in the top/bottom 8/4 rows) - now clean for every core.
- Recoveries: +35 genuine cores in vol01, +3 in vol02; bold-ratio validation (2x2-erode
  core/comm >=1.11) passes for all; eval gold p0042/43/44 and training pages reproduce
  (1-2px whitespace trims only); re-runs are idempotent.
- Only remaining audit flag: vol01 p0441 comm top - visually verified a Devanagari
  shirorekha (connected letter run 0.48w), not a rule; it is a section-title page and the
  comm crop starts correctly below the double rule.

## Split-boundary fixes (pixel-clean crops)
- Header fallback 10% cut first lines on rule-less front matter -> URL-aware 4.5%+8px,
  plus fragment-aware fallback (faded header rules, e.g. vol02 p0057) via 0.12-width opening.
- Separator padding + rule-extent: core ends 2px above rule top; comm starts 4px below the
  rule's measured full-width extent (connected-component downward-only + halo row scan),
  fixing rule remnants (vol02 p0057 comm showed rule tail; slope/faded ends).
- High separators no longer counted as header lines (vol01 p0043 sep at 13.8% had swallowed
  its core); re-runs preserve curated gt/status/unit in blocks.json.
- Verified: vol02 p0029/p0030 first lines intact; pilot seps stable (p0043 y=244, p0044 y=455).
- Open for GT phase (deferred): tall cores (>0.55 page height) on text pages may be long
  real mUlam sections or a mid-page rule. vol02 p0590 RESOLVED: genuine mUlam - its bold
  Tamil prose (core 119-1087 above sep y1089) matches unit 32-32.3 in
  `data/texts/core/32_nigamanAdhikAraH.txt` word-for-word; the nigamana adhikara has long
  prose mUlam units (cf. p0588/89/91/92 cores 469-811px, same pattern). Page-wide 2x2-erode
  profile: 0.72-0.73 above the rule (bold, uniform) vs 0.63-0.66 below (commentary).
  vol01 p0050 core is VERIFIED genuine (bold mUlam 00-0.11: "குருகைக் காவலப்பன்..." matches
  `data/texts/core/00_shrIguruparaMparAsAraH.txt`; reads like prose so it can look like
  commentary). vol01 p0172 (0.57H) was a false core and is now gone. A few mid-page rules
  sit at genuine discourse boundaries (vol02 p0082: above weakly matches mula 24-24.3, below
  matches comm 24-24.3 start) - geometry kept, content label to confirm; never auto-assign
  GT there. Library stickers intrude at page edges on some pages (e.g. p0057 left) - recorded
  noise, exclude overlapped lines in GT phase.

## Crop-edge invariant + detector move (cropped_over_line era)
The detector moved to `doc_curation/doc_curation/image/checker/detect_blocks.py`
(run as `python3 -m doc_curation.image.checker.detect_blocks` from the doc_curation
repo; the old `training/scripts/detect_blocks.py` is gone).

Every regenerated crop is now checked by `cropped_over_line`: the outermost row/column
must not sit on ink (a line or rule). Violations go to `metadata/crop_errors.json` and
print as `CROP-CUT vol:page:block {edges}`. Each edge entry carries `ink/runs/maxrun/kind`
(rule vs text-strokes) and `cut`: True when ink continues just outside the boundary (a
line is really severed), False when a line simply starts/ends exactly at the boundary
(nothing lost). `--check-crops` re-audits existing crops without regenerating.

Boundary-snapping fixes this era (all in the moved file):
- `_sbw` normalization: the Otsu mask for snapping was 0/255, so helper run/ink
  thresholds behaved as "any ink"; now `> 0` bool (pixel-count semantics).
- `_fix_boundary_top` / `_snap_bottom`: cycle guard, `max_steps`, and crest/taper clauses
  (`len(rr)==1`, `<=3 runs with maxrun>=0.02w`, `<=8 runs with maxrun>=0.10w`) so a
  wavy rule's ragged bottom / tapered crest is stepped past or trimmed, but only when
  the rows are below the checker flag level (early-out keeps genuine first lines).
- `_footer_cut` rewrite (ghost tails, small print, ornaments): real text rows carry
  >=6 runs and span >=0.15w (filters centered ornaments, diamonds, floral marks and the
  page number); the ink threshold self-calibrates from the page's own candidate rows
  (median, 50%) because fixed fractions of w fail on small-font index pages (vol01 p0243:
  list lines carry 40-80px ink; the old >=0.10w rule skipped them and a cut sliced through
  line "53" mid-glyph); a candidate band separated from the body by a far gap (>0.08h)
  is discarded (vol02 p0079: the wide "CC-0" footer line). Scan starts at 0.94h and the
  walk down follows descenders past it (vol02 p0172/p0500: last line bottom sits at
  0.943-0.945h, right above the CC footer at ~0.955h+). Result: blank/ghost/footer tails
  are excluded (79 vol01 + 90 vol02 comm tails shortened >100px, e.g. p0094 comm
  1697->529 at the adhikara-end colophon, vol02 p0079 comm 1649->911 before the ghost).
- Empty-comm crash guard: `comm_box` is dropped when its height <=30px (p0032 had an
  empty (1178,1178) box because the old footer cut mistook a separator rule for a text
  line; the rule-like rows now also require >3 runs).

Current `crop_errors.json` state (both volumes regenerated): vol01 57 flags, vol02 389.
Classification: vol01 - 24 of 57 at-edge (benign, line starts/ends exactly at the
boundary), the rest rule-fringe cuts (verified visually: the rule bottom is trimmed,
every text line complete; only 4 flags carry >60 ink and all are rule/slant-rule fringes:
p0441 comm top, p0328/p0182/p0006 core bottoms) plus L/R page-edge artifacts (stickers/
page border, crop already reaches the image edge - p0060/p0004 comm right). vol02 - the
dominant class (280 core-top) is the merged header-rule/first-line dip: the wavy rule's
bottom edge sits 1-3px above/on the first line's matra tips, so no ink-free boundary
exists (115 audit as rule-fringe via wide runs above the edge, the rest are matra-tip
dips <=114 ink, verified visually on p0552/p0228/p0603/p0317); bottoms are descender
tips/rule tops (72 inspect, top ink 126). These are report-only residuals; no GT text
is lost (dip rows are re-includable at GT alignment time if ever needed).
