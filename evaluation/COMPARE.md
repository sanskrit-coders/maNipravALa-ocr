# Train/compare — separated core/comm (eval held out, no ambiguity)

Eval layout (fixed; was ambiguous before):
- `data/RTS_sb_01/images/core/{p0043,p0044}.jpg` + `data/RTS_sb_01/gt/core/*.txt` (p0042 has NO core — no file, not empty file)
- `data/RTS_sb_01/images/comm/{p0042,p0043,p0044}.jpg` + `data/RTS_sb_01/gt/comm/*.txt` (page-part GT, not full unit)
- `evaluation/eval_index.jsonl` explicit nullable core paths; `evaluation/pred/{core,comm}/` same split.
- Score: `python3 training/scripts/evaluate.py --pred evaluation/pred --out evaluation/results.json`
  (core and comm scored separately).

## Baseline (Tesseract tam+san PSM6 on crops, same GT)

| page/block | GT chars | CER | WER | note |
|---|---|---|---|---|
| p0042/comm (00-0.3-tail) | 2050 | 0.2003 | 0.7071 | commentary-only page; best (pure Tamil flow) |
| p0043/core (00-0.4 verse) | 100 | 0.3010 | 0.7273 | short Devanagari verse `पापिष्ठः…` |
| p0043/comm (00-0.4-full) | 2330 | 0.3076 | 0.8191 | starts `இப்படி सर्वमुमुक्षु…` |
| p0044/core (00-0.5 prose) | 418 | 0.3531 | 1.0938 | `आचार्यवत्तै…` mixed prose, hardest |
| p0044/comm (00-0.5-full) | 1902 | 0.3232 | 0.8393 | starts `மில்லையாகையால்…` |
| MEAN core (2) | — | 0.3271 | — | |
| MEAN comm (3) | — | 0.2770 | — | |

Old page-level numbers (CER 0.43, full-unit GT + header/footer noise) are deprecated
(`evaluation/pred_tesseract/`, `results_tesseract.json` removed).
Kraken/TrOCR: `training/scripts/train_kraken.py`, `train_trocr.py` write same `pred/{core,comm}/`
paths, then re-run `evaluate.py`. Kraken install (torch) timed out >300s here; TrOCR needs GPU —
run on Colab, keep `evaluation/` strictly held out.

## Root cause of p0042_core bug (fixed)

`evaluation/gt/p0042_core.txt` was auto-created (since removed; canonical `data/RTS_sb_01/` has no p0042 core file by design) from unit mapping (`p0042→00-0.3` core+comm)
without checking block presence. `align.py` fuzzy core match spuriously fired because the
commentary quotes/discusses core words (e.g. verse words appear in comm), not because a core
block exists. `detect_blocks.py` (0.6-width threshold) correctly said `has_core=False` for
p0042–44, but eval ignored it. Fix: GT only from visually confirmed blocks (p0042 comm-only;
p0043 sep y=245/249; p0044 sep y=453/456, kw=0.3 detector); core GT deleted for p0042;
comm GT cut to page-part (p0042 = unit tail from `தன்பற்றில்`, offset 1297/3347;
p0043/44 = full units, start/end verified against crop OCR).

## Training separation TODO (same rule applies)

`training/images/sarabodhini-01/*.jpg` are RAW full pages (unscored, no GT assigned —
`training_index.jsonl` units empty). Do NOT assign unit GT to full pages. Procedure before any
training use: split to `core`/`comm` crops (see `training/metadata/blocks_01/crops/`,
threshold 0.3-width + manual verify as done for eval), assign page-part GT only to present
blocks, keep `training/` and `evaluation/` disjoint (note: 00-0.3 head is on train p0041,
tail on eval p0042 — same unit across split; for production split by adhikara, e.g. train Vol.1,
eval Vol.2/targets).
