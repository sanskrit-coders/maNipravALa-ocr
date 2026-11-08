# Resolving the Granularity–Context Paradox in Biscriptal Maṇipravāḷa OCR via Sliding-Patch Script Separation and Context-Engineered Faux Pages

**Target Venue:** *International Journal on Document Analysis and Recognition (IJDAR)* / *ACM Transactions on Asian and Low-Resource Language Information Processing (TALLIP)*

---

## Abstract

Classical South Indian *Maṇipravāḷa* ("ruby-and-coral") philosophical literature presents a uniquely severe challenge for Optical Character Recognition (OCR): Sanskrit and Tamil are interleaved not only across sentences and phrases, but **intra-word**, where Sanskrit nominal or verbal stems printed in the Devanāgarī abugida are directly inflected with Tamil case markers, clitics, or verbal suffixes printed in the Tamil script without intervening whitespace (e.g., `पुरुषार्थलाभத்தில்`, `अनुवदिத்தாரிறே?`, `स्वरूपप्राप्तமான`). Furthermore, traditional multi-commentary (*vyākhyāna-pañcaka*) editions typeset up to six distinct textual layers—root text (*mūlam*) and five classical commentaries—on every printed page separated by horizontal rules.

Through a systematic empirical investigation across **2,970 printed pages** (`6.67M` Unicode characters) of Vedānta Deśika's *Śrīmad-Rahasya-Traya-Sāra* and its commentaries, we identify what we term the **Granularity–Context Paradox** of biscriptal OCR:
1. **Whole-line and whole-page commercial/open-source OCR engines fail** when presented with unsplit biscriptal lines (PaddleOCR PP-OCRv4: **0.710 CER**; Google Drive Document OCR: **0.435 CER**; bilingual Tesseract `tam+san`: **0.352 CER**), because their internal line-level language and script models lock onto the dominant script of each line and transliterate or hallucinate the minority script into phonetic/Latin approximations while scrambling dense multi-layer layouts.
2. **Custom dual-script CTC recognizers trained from scratch suffer from capacity and language-model starvation**: a 40-epoch biLSTM-CTC line model (`devTamCharsOCR`) reaches **0.238 CER**, while a 4.1M-parameter Kraken VGSL word recognizer trained on **92,976 real word crops** achieves **90.34% character accuracy** (**80.53% word accuracy**), which still leaves nearly one in five words erroneous (~100,000 character errors per volume) and emits **1,456 spurious intra-word script chimeras** (`एா`, `எः`) due to unconstrained decoding over the union alphabet $\Sigma_{\text{dev}} \cup \Sigma_{\text{tam}}$.
3. **Wordwise script separation starves document OCR of visual and linguistic context**: splitting intra-word boundaries accurately (MAE **0.0320** via Shiro-rekhā headline and projection-glyph dynamic programming) and packing isolated word/sub-word crops onto single-script faux pages causes Google Drive OCR to drop **76%–100% of short Sanskrit words** and misclassify short Tamil suffixes (`மான,`, `என்று,`) as ASCII/Latin (`LD IT 60T,`, `6T601M,`) due to competition from row-index tags (`#T01`).

We resolve this paradox with a **Phrase-Strip Script-Separated Faux-Page Pipeline** (`tadevaOCR`) that combines: (i) recursive pitch-based line segmentation and mid-band ink-filtered word segmentation; (ii) a **48×48 sliding-patch residual CNN** (`script_cls.pt`, **0.9921 real-scan accuracy**) paired with minimum-ink column snapping (`<= 6` dark pixels) to sever 1–3 px Devanāgarī *Shiro-rekhā* overhangs at intra-word `dev ↔ tam` junctions while tracking `pre_space` flags; (iii) **same-script phrase-strip coalescing**, which preserves multi-word horizontal baseline and language-model context across mono-script spans; and (iv) **context-engineered faux pages** featuring a $1:2$ horizontal-to-vertical proximity ratio (`28 px` vs. `54 px`) and **6-consonant phonotactic script anchors** (`கஙசஞடண` / `कखगघङच`) appended to narrow rows to override ASCII tag bias.

On a controlled benchmark of hand-verified *Sārabodhinī* commentary blocks, our pipeline reduces Character Error Rate (CER) to **0.112**—a **3.9× reduction** over unsplit Google Drive OCR (`0.435`), **3.1×** over bilingual Tesseract (`0.352`), and **2.1×** over custom dual-script Kraken CTC (`0.238`). On a Volume 1 intra-word Maṇipravāḷa benchmark, it achieves **100.0% (`568 / 568`) `#Txx` tag recovery**, recovers **110 intra-word mixed tokens** (vs. `0` in unsplit/wordwise baselines), and reduces stray Latin hallucinations by **94.6%**. Deployed across all four volumes (**2,970 pages**, **97,198 lines**, **57,150 intra-word biscriptal words**), the pipeline produces a complete, publication-grade digital corpus and deterministically dissects **11,683 commentary blocks** across **2,103 chapter pages** into **165 structured per-commentary, per-chapter files** with **zero unresolved blocks**.

**Keywords:** Biscriptal OCR · Maṇipravāḷa · Intra-Word Code-Switching · Devanāgarī · Tamil Script · Script Identification · Faux-Page Context Engineering · Historical Document Analysis

---

## 1. Introduction

### 1.1 The Typographic Challenge of Biscriptal *Maṇipravāḷa*

In the Śrīvaiṣṇava scholastic tradition of South India (12th–19th centuries CE), philosophical treatises and commentaries were composed in *Maṇipravāḷa* (literally "ruby [*maṇi*, Sanskrit] and coral [*pravāḷa*, Tamil]"), a literary register that weaves Sanskrit technical terminology and quotations seamlessly into Tamil syntactic and morphological frames. When printed in modern scholarly editions of the 20th century, *Maṇipravāḷa* is rendered **biscriptally**: Sanskrit lexical stems, compounds, and quotations are typeset in the **Devanāgarī** script, while Tamil stems, case endings, postpositions, and finite verbal morphology are typeset in the **Tamil** script.

Crucially, unlike bilingual documents where script switching occurs at paragraph, sentence, or clause boundaries, printed *Maṇipravāḷa* exhibits pervasive **intra-word script switching** (macaronic morphology without whitespace):
- **Sanskrit nominal stem + Tamil case suffix / clitic:**
  `पुरुषार्थलाभத்தில்` (*puruṣārtha-lābha-ttil*, locative), `स्वरूपप्राप्तமான` (*svarūpa-prāpta-māṉa*, adjectival), `भगवल्लक्ष्म्यादिகளில்` (*bhagaval-lakṣmy-ādi-kaḷil*, plural locative), `रहस्यत्रयத்துக்கு` (*rahasya-traya-ttukku*, dative).
- **Sanskrit verbal root/stem + Tamil verbal inflection:**
  `अनुवदिத்தாரிறே?` (*anuvadi-ttār-iṟē*, finite past honorific + clitic), `समाश्रयिக்கையில்` (*samāśrayi-kkaiyil*, verbal noun locative).
- **Tamil demonstrative prefix + Sanskrit stem + Tamil suffix (tri-segment words):**
  `இவ்வधिकारार्थம்` (*ivv-adhikārārtha-m*, `tam → dev → tam`).

This intra-word biscriptal coupling violates the foundational assumption of nearly all modern OCR architectures: that a whitespace-delimited word—and typically an entire text line—belongs to a single writing system with uniform baseline geometry and a single character transition distribution. Structurally, Devanāgarī and Tamil possess incompatible vertical typographic profiles:
1. **Devanāgarī** hangs glyphs from a continuous upper horizontal headline (*Shiro-rekhā*) at roughly $0.18H$–$0.30H$ of line height, with super-headline vowel diacritics (*mātrās*: `-े`, `-ै`, `-ो`, `-ौ`, *repha*) and sub-baseline vocalic suffixes (`-ु`, `-ू`, `-ृ`) and conjuncts.
2. **Tamil** lacks a continuous headline; its characters sit on a lower baseline with prominent rounded loops, tall ascending vowel loops (`இ`, `ஈ`, `ை`, `ௌ`), and deep sub-baseline descenders (`ु`/`ू` ligatures in `கு`, `சூ`, `ழு`, `ஞ`, `ளு`).
3. At an **intra-word `dev → tam` junction**, the printing type for the final Devanāgarī *akṣara* frequently extends its upper *Shiro-rekhā* headline $1\text{–}3\text{ px}$ into the bounding column of the first Tamil glyph (e.g., `निवर्तनத்தில்`), creating a physical ink bridge between the two scripts.

```mermaid
flowchart TD
    subgraph Page["Multi-Commentary Printed Page (2,970 Pages across Vols. 1–4)"]
        M["मूलम् (Root Text by Vedānta Deśika)"]
        R1["─── Horizontal Rule ───"]
        C1["1. सारदीपिका (Sāradīpikā)"]
        R2["─── Horizontal Rule ───"]
        C2["2. सारास्वादिनि (Sārāsvādinī)"]
        R3["─── Horizontal Rule ───"]
        C3["3. सारप्रकाशिका (Sāraprakāśikā)"]
        R4["─── Horizontal Rule ───"]
        C4["4. सारविवरणी (Sāravivaraṇī)"]
        R5["─── Horizontal Rule ───"]
        C5["5. सारप्रकाशिकासङ्ग्रहः (Sāraprakāśikā-saṅgrahaḥ)"]
    end

    subgraph IntraWord["Intra-Word Biscriptal Coupling (No Whitespace)"]
        W1["Devanāgarī Stem: पुरुषार्थलाभ"] ---|"1–3 px Shiro-rekhā overhang"| W2["Tamil Suffix: த்தில்"]
        W3["Tamil Prefix: இவ்வ"] --- W4["Devanāgarī Stem: धिकारार्थ"] --- W5["Tamil Suffix: ம்"]
    end

    Page --> IntraWord
```

### 1.2 The Granularity–Context Paradox

When digitizing a monumental biscriptal corpus such as the four-volume *Vyākhyāna-pañcaka-sahita Śrīmad-Rahasya-Traya-Sāra* (2,970 PDF pages) and the two-volume *Sārabodhinī* edition (1,246 PDF pages), a fundamental tension emerges between **script homogeneity** (which requires fine-grained spatial segmentation down to sub-word fragments) and **visual/linguistic context** (which modern sequence-to-sequence and cloud document OCR engines require for accurate recognition). We call this the **Granularity–Context Paradox**:

- **At Coarse Granularity (Full Page or Full Line):** Visual and linguistic context is maximal, but script homogeneity is violated. Commercial VLM/cloud OCR engines (such as Google Drive Document OCR) and scene-text engines (PaddleOCR) employ strong line-level script classification and neural language models. When fed a line containing interleaved Devanāgarī and Tamil words, the engine locks onto the majority script of the line and forces the minority script through the wrong decoder—transliterating Devanāgarī into Tamil phonetics or garbled Latin (`tafuqGiiH`), or dropping dense multi-commentary blocks entirely (**0.435–0.710 CER**).
- **At Fine Granularity (Isolated Word or Sub-Word Fragment):** Script homogeneity is achieved by splitting mixed words at script boundaries, but visual and linguistic context collapses. When isolated 1-word or sub-word crops (such as a 2-letter Tamil suffix `த்தில்` or a 2-*akṣara* Sanskrit stem `द्रय`) are arranged on synthetic "faux pages" for cloud OCR, the recognizer loses its multi-word horizontal baseline estimate and n-gram/subword language model priors. Consequently, Google Drive OCR drops **76%–100% of short Devanāgarī stems** and misreads short Tamil suffixes (`மான,`, `என்று,`) as ASCII/Latin lookalikes (`LD IT 60T,`, `6T601M,`) because the 4-character ASCII row tag (`#T01`) dominates the line-level script detector.
- **With Custom Dual-Script Models Trained from Scratch:** Training a unified CTC recognizer over $\Sigma_{\text{dev}} \cup \Sigma_{\text{tam}}$ avoids cloud OCR heuristics, but encounters severe **capacity and lexicon starvation**: even with **92,976 verified real word crops**, a 4.1M-parameter Kraken VGSL word model tops out at **90.34% character accuracy** (**80.53% word accuracy**), translating to ~100,000 character errors per volume and **1,456 impossible intra-word script chimeras** (`एா`, `எः`).

### 1.3 Contributions

This paper presents a complete methodological and empirical resolution of the Granularity–Context Paradox:
1. **Weakly Supervised Render-and-DP Dataset Construction (Section 3):** We introduce a rendering-based Dynamic Programming (DP) alignment algorithm that pairs existing unaligned digital editions with Tesseract candidate word bounding boxes, generating **95,411 high-confidence word-image maps** across **2,203 scanned pages** and **14,208 intra-word script boundary annotations** without manual bounding-box annotation.
2. **Systematic Taxonomy & Empirical Benchmark of 9 OCR Architectures (Section 2 & Section 5):** We evaluate nine distinct paradigms—spanning PaddleOCR, bilingual Tesseract, unsplit Google Drive OCR, custom line-level and word-level Kraken VGSL CTC models, 1D/2D boundary CNNs, and synthetic-pretrained vs. real-scan-retrained classifiers—documenting exact failure mechanisms including the **synthetic-to-scan domain trap**, **CTC union-alphabet chimeras**, and **Tesseract glyph illusions** (e.g., `(द्वया-)` misread as `(द्रयाணன்`).
3. **Phrase-Strip Script Separation & Context-Engineered Faux Pages (Section 4):** We design an 8-stage pipeline (`tadevaOCR`) that:
   - Splits intra-word `dev ↔ tam` junctions using a **0.9921-accurate 48×48 sliding-patch residual CNN** and minimum-ink column snapping (`<= 6` px) across *Shiro-rekhā* overhangs while preserving `pre_space` concatenation flags and absorbing punctuation (`_is_punct_or_other`);
   - Coalesces adjacent same-script tokens on each line into **coherent multi-word phrase strips**, restoring horizontal baseline and linguistic context;
   - Enforces geometric and phonotactic invariants on faux pages, including a **$1:2$ horizontal-to-vertical gap ratio** (`28 px` vs. `54 px`), internal whitespace compaction, and **6-consonant phonotactic script anchors** (`கஙசஞடண` / `कखगघङच`) that eliminate Latin lookalike hallucinations on short suffixes.
4. **Full 4-Volume Corpus Production & Deterministic Commentary Dissection (Section 5.5):** We deploy the pipeline across all **2,970 pages** (`6.67M` Unicode characters, `57,150` intra-word mixed tokens) of *Śrīmad-Rahasya-Traya-Sāra* and automatically dissect **2,103 chapter pages** (`11,683` visual blocks separated by horizontal rules) into **165 structured per-commentary, per-chapter Markdown files** with **zero unresolved blocks**.

---

## 2. Related Work and Failure-Mode Taxonomy

### 2.1 Multilingual and Multiscript OCR in South Asian Documents

Standard OCR systems for Indic scripts—including Tesseract 4/5 LSTM [Smith, 2007], Kraken VGSL [Kiessling, 2019], Calamari [Wick et al., 2020], and commercial cloud vision APIs—treat multiscript recognition through one of two mechanisms:
1. **Multi-model script routing:** A page- or line-level script identification module routes each region to a script-specific recognizer. While effective when script switches occur at paragraph or line boundaries, line-level routing fails whenever two scripts coexist within the same line or word.
2. **Unified alphabet CTC decoding:** A single recurrent or convolutional-transformer network is trained over the union of character sets $\Sigma = \Sigma_1 \cup \Sigma_2 \cup \Sigma_{\text{punct}}$. In low-resource scholarly domains without a massive biscriptal language model, unified CTC decoders suffer from cross-script confusion whenever visual strokes in script $A$ resemble strokes in script $B$ (e.g., Devanāgarī *daṇḍa* `।` / *aa-mātrā* `ा` vs. Tamil *kāl* `ா`; Devanāgarī `ए` vs. Tamil `எ`).

### 2.2 Empirical Failure-Mode Taxonomy Across Existing Paradigms

Over the course of our investigation (documented in [`history.md`](../history.md)), we implemented and benchmarked eight preliminary architectures before arriving at the production pipeline. Table 1 summarizes the failure modes observed across these paradigms.

| Paradigm | Representative Engine / Model | Granularity | Observed CER / Acc. | Primary Failure Mechanism |
|---|---|---|---|---|
| **1. Scene-Text Dual-Pipeline** | PaddleOCR PP-OCRv4 (`ta` + `hi`) | Line | `0.710–0.732` CER | No native Tamil detector; recognizer forces non-target script into Latin garbage even under oracle routing. |
| **2. Bilingual LSTM Line OCR** | Tesseract 5 (`tam+san`, PSM 7) | Line | `0.352–0.476` CER | Dual-language LSTM emits script chimeras at intra-word junctions; misreads Sanskrit ligatures (`(द्वया-)`) as mixed Devanāgarī–Tamil (`(द्रयाணன்`). |
| **3. Unsplit Cloud Document OCR** | Google Drive OCR (Raw Page/Block) | Block / Page | `0.435` CER | Line-level language model locks onto dominant script per line, transliterating minority script into Latin/phonetics; scrambles multi-commentary layouts. |
| **4. Dual-Script Line CTC (Scratch)** | Kraken VGSL `devTamCharsOCR` (16.5k lines, 40 ep.) | Line | `0.238–0.292` CER | Insufficient training diversity relative to commercial VLM; collapses on short isolated phrases (`0.812` CER); sensitive to LSTM forget-gate init. |
| **5. Synthetic-Trained Word/Column CNNs** | `word_cls.pt`, `script_sep.pt`, `col_cls.pt`, `boundary_comb.pt` | Word / Column | `0.779` real word acc.; `0.155` boundary MAE | **Synthetic-to-Scan Domain Trap:** Models achieving `99.1%–99.8%` on font-rendered synthetic data degrade severely on 1960s letterpress bleed and touching glyphs. |
| **6. Dual-Script Word CTC (Real Maps)** | Kraken 7.1.1 VGSL (`best_0.8900.safetensors`, 93k real words) | Word | `0.0966` CER (`80.53%` word acc.) | Without line context or lexicon, `9.66%` CER yields ~1 in 5 words wrong (`~100k` char errors/vol) and **1,456 impossible intra-word script chimeras** (`एா`, `எः`). |
| **7. Wordwise Shiro-Rekhā + Cloud OCR** | `wordwise-shiro-rekhA` + Google Drive OCR | Word / Sub-Word | **76%–100% Dev. word loss** | **Context Starvation:** Isolated 1-word/sub-word crops lack horizontal baseline and LM context; Drive drops short stems or outputs Latin (`tafuqGiiH`). |
| **8. Naive Tagged Faux Pages** | Unanchored `#Txx` Faux Pages + Drive OCR | Phrase Strip | `74` stray Latin chars / 4 pp. | **ASCII Tag Bias:** On narrow 1-word suffix rows (`மான,`, `என்று,`), 4 ASCII chars (`#T01`) outnumber Tamil glyphs, flipping Drive's line script detector to Latin (`6T601M,`). |
| **9. Proposed Pipeline** | **`tadevaOCR` (Patch CNN + Phrase Strips + Anchored Faux Pages)** | **Coalesced Same-Script Phrase Strip** | **`0.112` CER** (`100%` tag recovery) | **Resolves Granularity–Context Paradox:** Splits intra-word boundaries via 0.9921 real-scan patch CNN, re-groups same-script phrase strips, and locks script detection with 6-consonant anchors. |

---

## 3. Weakly Supervised Dataset Construction via Render-and-DP Alignment

Training and evaluating real-scan script classifiers, boundary detectors, and word recognizers requires tens of thousands of word bounding boxes paired with ground-truth Unicode transcriptions and script labels (`dev`, `tam`, `mixed`). Because manual bounding-box annotation at this scale is prohibitive, we developed a **Render-and-DP Weakly Supervised Alignment Pipeline** ([`data/README.md`](../../data/README.md)) using two digitized volumes of the *Sārabodhinī* edition (`RTS_sb_01`, 630 pages; `RTS_sb_02`, 616 pages), for which proofread page-level digital texts existed on Archive.org / Wikisource.

```mermaid
flowchart LR
    A["Scanned Page Image (300 DPI)\nRTS_sb_01 (630 pp.) & RTS_sb_02 (616 pp.)"] --> B["Horizontal Rule Detection\nCore (Mūlam) vs. Comm (Sārabodhinī)"]
    B --> C["Tesseract (tam+san) TSV\nCandidate Word Bounding Boxes"]
    D["Proofread Digital Text\n(Page-Aligned Ground Truth)"] --> E["Synthetic Word Renderer\n(Lohit-Devanagari / NotoSansTamil)"]
    C --> F["Visual + OCR Similarity Matrix\nS(i, j) = α·Sim_OCR + (1-α)·Sim_Visual"]
    E --> F
    F --> G["Monotonic Dynamic Programming\nOptimal Box-to-Token Alignment"]
    G --> H["Confidence Filtering (conf ≥ 0.45)\n95,411 Word-Image Maps across 2,203 Block Pages"]
```

### 3.1 Algorithmic Formulation

For each scanned page $p$, we first partition the page at the horizontal separator rule into the upper `core` (*mūlam*) block and lower `comm` (*Sārabodhinī*) block. Let $B = (b_1, b_2, \dots, b_M)$ denote the sequence of candidate word bounding boxes extracted in reading order from `tesseract -l tam+san --psm 6 tsv`, and let $W = (w_1, w_2, \dots, w_N)$ denote the ground-truth whitespace-delimited tokens from the digital edition for that block.

Because Tesseract's bilingual OCR text $o_i$ on mixed words is noisy (CER $\approx 0.35$), pure string-edit alignment fails whenever several consecutive mixed words are garbled. We augment string similarity with **direct visual template similarity**: each ground-truth token $w_j$ is rendered synthetically into a grayscale bitmap $\tilde{I}(w_j)$ using script-appropriate fonts (*Lohit-Devanagari* and *Noto Sans Tamil*), normalized to match the height of crop $I(b_i)$, and compared via normalized cross-correlation and width-ratio consistency. Monotonic dynamic programming then finds the alignment path $\pi$ maximizing cumulative similarity, retaining only word pairs $(b_i, w_j)$ with combined alignment confidence $\ge 0.45$.

### 3.2 Resulting Corpus Statistics

Table 2 summarizes the resulting weakly supervised dataset (`data/RTS_sb_01/word-image-maps` and `data/RTS_sb_02/word-image-maps`), which comprises **95,411 high-confidence real word-image mappings** across **2,203 block JSON files**.

| Sub-Corpus | Scanned Pages | Block JSONs (`core` + `comm`) | Aligned Word Boxes ($\text{conf} \ge 0.45$) | Pure Devanāgarī (`dev`) | Pure Tamil (`tam`) | Intra-Word Mixed (`mixed`) |
|---|---:|---:|---:|---:|---:|---:|
| **`RTS_sb_01`** (Vol. 1) | 630 | 1,115 | 46,842 | 17,910 | 21,884 | 7,048 |
| **`RTS_sb_02`** (Vol. 2) | 616 | 1,088 | 48,569 | 18,412 | 22,997 | 7,160 |
| **Total Real-Scan Corpus** | **1,246** | **2,203** | **95,411** | **36,322 (38.1%)** | **44,881 (47.0%)** | **14,208 (14.9%)** |

Nearly **15% of all high-confidence words (`14,208` tokens)** in the *Sārabodhinī* corpus are intra-word mixed-script tokens (predominantly `dev → tam`), confirming that intra-word code-switching is a core structural feature of the language rather than a rare edge case.

---

## 4. Methodology: Phrase-Strip Script Separation and Context-Engineered Faux Pages

To resolve the Granularity–Context Paradox, our production pipeline ([`training/scripts/script_cls/pipeline.py`](../../training/scripts/script_cls/pipeline.py)) separates **where script boundaries are detected** (at the 8-pixel sub-word patch and single-column ink level) from **how image regions are presented to the OCR recognizer** (as coalesced, multi-word same-script phrase strips embedded in geometrically and phonotactically engineered faux pages). Figure 2 illustrates the end-to-end 8-stage architecture.

```mermaid
flowchart TD
    S0["Input PDF Page (Rendered at 300 DPI)\nMulti-Commentary Maṇipravāḷa Layout"] --> S1
    S1["Stage 1: Pitch-Based Recursive Line Segmentation\n• Suppress horizontal rules (row_ink > 0.55·W)\n• Estimate unit_h & pitch; recursively split merged multi-line bands"] --> S2
    S2["Stage 2: Mid-Band Ink-Filtered Word Segmentation\n• Split at vertical valleys ≥ 4 px\n• Require total ink ≥ 18 px & mid-band (0.25H..0.75H) ink ≥ 8 px"] --> S3
    S3["Stage 3: 48×48 Sliding-Patch Residual CNN (script_cls.pt)\n• 8 px stride + center-patch sampling for narrow words (w ≤ 40 px)\n• _merge_short_runs + _safe_split_x (min-ink ≤ 6 px across Shiro-rekhā)"] --> S4
    S4["Stage 4: Intra-Word pre_space Tracking & Punctuation Absorption\n• pre_space = False if prev='dev', curr='tam', gap ≤ 5 px\n• Absorb 'other' tokens (dandas, digits, hyphens) into neighbor"] --> S5
    S5["Stage 5: Same-Script Phrase-Strip Coalescing\n• Merge consecutive same-script runs per line (~2.5× fewer strips)\n• Compact internal whitespace > 36 px to 22 px"] --> S6
    S6["Stage 6: Context-Engineered Faux-Page Composition\n• Separate _dev_ & _tam_ pages (40 rows/page, ROW_PITCH=136 px)\n• 28 px tag-to-strip gap vs. 54 px vertical gap\n• Append 6-consonant script anchor (கஙசஞடண / कखगघङच) if w < 460 px"] --> S7
    S7["Stage 7: Cross-Page Batch Packing & Async Google Drive OCR\n• Pack 25 PDF pages into full 40-row faux pages (~2× fewer API calls)\n• Export text/plain via Drive API while overlapping CPU section detection"] --> S8
    S8["Stage 8: Fault-Tolerant Tag Collation & 5-Commentary Dissection\n• Parse #Txx variants, strip anchors, deduplicate wrapped words\n• Join split stems (pre_space=False), align horizontal rules (---)\n• Dissect into 5 commentary folders × 33 adhikāras"]
```

### 4.1 Stage 1: Pitch-Based Recursive Projection Line Segmentation

Traditional horizontal projection profile segmentation fails on *Maṇipravāḷa* pages for two reasons: (i) full-width horizontal separator rules between commentaries create high-ink rows that skew threshold estimates, and (ii) deep Tamil sub-baseline descenders (`ழ`, `ஞ`, `ளு`, `கு`) in line $k$ frequently touch or overlap the upper Devanāgarī *mātrās* (`-े`, `-ै`, `-ो`) of line $k+1$, merging 2–6 printed lines into a single connected vertical band.

In [`segment_lines`](../../training/scripts/script_cls/pipeline.py#L223-L321), we binarize the grayscale page $G$ via an adaptive percentile threshold $T = \min(200, \max(150, P_{12}(G) + 25))$ and suppress horizontal separator rules by zeroing any row whose dark pixel count exceeds $0.55 \times W_{\text{span}}$ (extending $\pm 2\text{ px}$ vertically). After extracting initial vertical bands (`row_ink >= 6`, bridging gaps $\le 2\text{ px}$), we estimate the robust single-line height $H_{\text{unit}} = P_{35}(\{h_i\})$ and inter-line pitch $P_{\text{line}} = \text{median}(\{\Delta y_i\})$. Any band exceeding:
$$H_{\max} = \max\left(H_{\text{unit}} + 0.55 \cdot P_{\text{line}},\; 1.55 \cdot H_{\text{unit}}\right)$$
is **recursively subdivided** by estimating the integer line count $K = \text{round}((h_i + \text{gap}) / P_{\text{line}})$ and locating the $K-1$ local minima of a 5-pixel box-filtered projection profile within search windows of $\pm 0.35 P_{\text{line}}$ around the expected pitch multiples. This eliminates multi-line mergers without clipping super-headline Devanāgarī *mātrās* or sub-baseline Tamil descenders (`±3 px` vertical padding).

### 4.2 Stage 2: Mid-Band Ink-Filtered Word Segmentation

Given a segmented line crop of height $H$, [`_word_boxes`](../../training/scripts/script_cls/pipeline.py#L430-L458) computes the vertical column ink profile $\text{col}(x) = \sum_{y=0}^{H-1} \mathbb{I}[\text{dark}(y, x)]$. A column $x$ is marked as active word ink if $\text{col}(x) > 1$, and interior gaps narrower than `MIN_WORD_GAP = 4 px` are bridged so slightly kerning Tamil glyphs remain intact while true inter-word spaces ($\ge 4\text{ px}$) split cleanly.

To prevent tips of ascenders or descenders from adjacent lines (`±3 px` padding) from spawning phantom miniature word boxes, each candidate span $[x_0, x_1]$ must satisfy a **dual ink-density invariant**:
$$\sum_{y=0}^{H-1} \sum_{x=x_0}^{x_1-1} \mathbb{I}[\text{dark}(y, x)] \ge 18 \quad \text{and} \quad \sum_{y=\lfloor 0.25H \rfloor}^{\lceil 0.75H \rceil} \sum_{x=x_0}^{x_1-1} \mathbb{I}[\text{dark}(y, x)] \ge 8$$
Candidate spans failing the middle-band ($0.25H\text{..}0.75H$) check contain only peripheral bleed and are discarded.

### 4.3 Stage 3: Sliding 48×48 Patch Script Classification and Minimum-Ink Intra-Word Splitting

A central finding of our study (Section 5.3) is that **whole-word classifiers fail to generalize from synthetic training data to historical letterpress scans**, whereas **local $48 \times 48$ sliding patches retrained on real word maps achieve 99.21% accuracy**.

#### 4.3.1 Patch Architecture (`script_cls.pt`)
We employ a 4-class (`0: dev`, `1: tam`, `2: other`, `3: null`) residual CNN (`Net`, **1.2M parameters**, [`train_classifier.py`](../../training/scripts/script_cls/train_classifier.py)) operating on $48 \times 48$ grayscale patches normalized as $1.0 - \text{pixel}/255.0$. For each word box $[x_0, x_1]$ of height $h$, [`classify_word`](../../training/scripts/script_cls/pipeline.py#L524-L577) slides a window of width $48\text{ px}$ at `STRIDE = 8 px`:
- **Center-patch sampling for narrow words:** Early iterations clamped the sample coordinate to `x0 + 24`, which sampled whitespace to the right of narrow 1-character words ($w \le 40\text{ px}$). Sampling the true midpoint $x_c = \lfloor(x_0 + x_1)/2\rfloor$ when $w \le 40\text{ px}$ ensures narrow stems and suffixes are centered on the $48 \times 48$ canvas.
- **Mean-probability fallback:** Patches classified as `null` (`cls == 3`) or with maximum softmax probability $< 0.45$ are filtered out; if all patches in a short word fall below $0.45$, we assign the argmax of the mean softmax vector over `{dev, tam, other}` so no word is ever dropped.

#### 4.3.2 Run Merging and Minimum-Ink Splitting Across *Shiro-rekhā* Overhangs
Adjacent 8-px patch predictions are coalesced by [`_merge_short_runs`](../../training/scripts/script_cls/pipeline.py#L482-L521), which folds interior `"other"` patches and transient single-patch spikes narrower than $\max(20\text{ px}, \lfloor 0.45h \rfloor)$ into the dominant neighboring script.

When a word contains a genuine `dev ↔ tam` transition at patch boundary $\hat{x}$ (e.g., `पुरुषार्थलाभ` + `த்தில்`), cutting directly at $\hat{x}$ frequently slices through a character stroke because of the $8\text{ px}$ patch stride. Conversely, searching strictly for a $0$-ink column fails whenever the Devanāgarī *Shiro-rekhā* headline overhangs $1\text{–}3\text{ px}$ above the first Tamil letter (`निवर्तनத்தில்`). Our boundary snapper [`_safe_split_x`](../../training/scripts/script_cls/pipeline.py#L461-L479) searches the window $[\hat{x} - 14, \hat{x} + 14]$ and selects:
$$x^* = \begin{cases} \arg\min_{x \in [\hat{x}-14, \hat{x}+14],\, \text{col}(x)=0} |x - \hat{x}| & \text{if a } 0\text{-ink column exists,} \\[6pt] \arg\min_{x \in [\hat{x}-14, \hat{x}+14]} \left(\text{col}(x) + 0.15|x - \hat{x}|\right) & \text{if } \min_x \text{col}(x) \le 6, \\[6pt] \hat{x} & \text{otherwise.} \end{cases}$$
Allowing a minimum-ink cut of $\le 6$ dark pixels cleanly severs a 1–3 px *Shiro-rekhā* overhang without truncating either glyph.

### 4.4 Stage 4: Intra-Word `pre_space` Tracking and Punctuation Absorption

To reconstruct mixed words without inserting spurious whitespace during collation, [`_word_runs`](../../training/scripts/script_cls/pipeline.py#L580-L608) attaches a boolean `pre_space` attribute to every run:
1. Sub-word runs produced by splitting a single mixed word (`j > 0`) receive `pre_space = False`.
2. Even when Stage 2's valley splitter separates a Sanskrit stem and its Tamil suffix across a micro-gap of $4\text{–}5\text{ px}$, we set:
   $$\text{pre\_space} = \text{False} \quad \text{whenever} \quad (\text{prev\_script} = \text{dev}) \land (\text{curr\_script} = \text{tam}) \land (x_0^{\text{curr}} - x_1^{\text{prev}} \le 5\text{ px}).$$
3. **Line-Level `"other"` Absorption ([`_is_punct_or_other`](../../training/scripts/script_cls/pipeline.py#L1010-L1033)):** Whole-word tokens classified as `"other"`—such as Sanskrit double *daṇḍas* (`।।`), verse numbers (`१०२`), parentheses, hyphens, or low-ink punctuation (`w <= 16 px` or `ink_density < 0.08`)—are never discarded or routed to a separate recognizer; they are absorbed into the adjacent `dev` or `tam` run on the same line, preserving all punctuation and verse numbering in context.

### 4.5 Stage 5: Same-Script Phrase-Strip Coalescing and Whitespace Compaction

Rather than emitting one crop per word (which starves the OCR engine of baseline and language context), [`make_segments`](../../training/scripts/script_cls/pipeline.py#L611-L637) coalesces all consecutive runs on a line that share the same script (`dev` or `tam`) into a **single horizontal phrase strip** $[x_{\text{start}}, x_{\text{end}}]$.
- A pure Tamil or pure Devanāgarī line becomes **one full-line strip**.
- A line with one intra-word switch (`... समाश्रयिக்கையில் ...`) becomes just **two or three multi-word phrase strips**, reducing total segment count by **~2.5×** while providing 3–10 words of contiguous mono-script context per strip.
- **Internal Whitespace Compaction ([`_compact_crop_whitespace`](../../training/scripts/script_cls/pipeline.py#L644-L661)):** Running headers and centered verse lines often contain wide internal blank spans ($> 36\text{ px}$). If left intact, Google Drive's layout analyzer interprets the wide gap as a two-column page break and reads the right half out of order. We compress any internal blank span $> 36\text{ px}$ down to $22\text{ px}$ before pasting onto the faux page.

### 4.6 Stage 6: Context-Engineered Faux Pages and Phonotactic Script Anchors

All `dev` phrase strips across a batch of pages are packed onto Devanāgarī faux pages (`_dev_`), and all `tam` phrase strips onto Tamil faux pages (`_tam_`), with up to `SEGMENTS_PER_FAUX = 40` rows per page (`PAGE_W = 1800 px`, `ROW_PITCH = 136 px`, maximum strip height `82 px`).

To eliminate the layout and script-detection failure modes of black-box cloud document OCR (Section 2.2), [`compose_faux`](../../training/scripts/script_cls/pipeline.py#L664-L709) enforces two crucial invariants:

```mermaid
flowchart LR
    subgraph FauxRow["Single Row on Context-Engineered Faux Page (ROW_PITCH = 136 px, Vertical Gap = 54 px)"]
        direction LR
        Tag["#T01\n(x = 16..118 px\n34pt Bold Sans)"] ---|"28 px Horizontal Gap\n(< 54 px Vertical Gap)"| Crop["Coalesced Same-Script Phrase Strip\n(x = 146 px, height ≤ 82 px)\nE.g., 'மான,' or 'पुरुषार्थलाभ'"]
        Crop ---|"36 px Gap\n(Only if width < 460 px)"| Anchor["6-Consonant Script Anchor\nTamil: 'கஙசஞடண'\nDevanāgarī: 'कखगघङच'"]
    end
```

1. **The $1:2$ Horizontal-to-Vertical Proximity Ratio:**
   When `#T01..#T40` tags are placed too far to the left of the image strips or when rows are packed too tightly vertically, Drive's document layout engine groups `#T01..#T40` vertically as an independent left margin column and outputs all 40 tags before any text. By rendering `#Txx` at $x = 16\text{ px}$ (ending at $x \approx 118\text{ px}$) and pasting the phrase strip at $x = 146\text{ px}$, the **horizontal gap is $28\text{ px}$** (roughly one word space), whereas the **vertical white gap between rows is $136 - 82 = 54\text{ px}$**. Because horizontal proximity is nearly $2\times$ tighter than vertical proximity, the layout analyzer strictly groups each `#Txx` tag horizontally with its phrase strip while preserving a space delimiter.
2. **6-Consonant Phonotactic Script Anchors on Narrow Strips (`im.width < 460 px`):**
   When a phrase strip consists of a single short word or suffix (e.g., a 3-letter Tamil suffix `மான,`, `என்று,`, `முமாம்.` split from a mixed word), the row contains 4 ASCII characters (`#`, `T`, `0`, `1`) and only 3–4 Indic glyphs. Google Drive's line-level script classifier consequently assigns the line to the **Latin/ASCII** script and decodes the Tamil curves as visually similar ASCII characters:
   - `என்று,` $\rightarrow$ `6T601M,`
   - `மான,` $\rightarrow$ `LD IT 60T,`
   - `முமாம்.` $\rightarrow$ `(LOLOLD.`
   To defeat this ASCII tag bias, whenever a strip has width $< 460\text{ px}$, `compose_faux` renders a crisp 6-consonant script anchor $36\text{ px}$ to the right of the crop:
   - **Tamil faux pages (`_tam_`):** `கஙசஞடண` (the first six *vallinam/mellinam* stops in standard alphabetical order)
   - **Devanāgarī faux pages (`_dev_`):** `कखगघङच` (the first six *ka-varga/ca-varga* stops)
   This guarantees that every narrow row contains at least $9\text{–}10$ unmistakable Indic glyphs versus $4$ ASCII tag characters ($\ge 2.25\times$ majority), locking Drive's line-level script detector into Tamil or Devanāgarī. During collation, [`parse_tagged`](../../training/scripts/script_cls/pipeline.py#L734-L804) strips the trailing anchor via a tolerant regular expression (`ANCHOR_STRIP_RE`) that handles occasional OCR drops of 1–2 anchor letters.

### 4.7 Stages 7–8: Cross-Page Batch Packing, Fault-Tolerant Collation, and Multi-Commentary Dissection

- **Cross-Page Batch Packing & Pipelined Execution ([`run_pipeline.py`](../../training/scripts/script_cls/run_pipeline.py)):** Rather than flushing partially empty faux pages after every source PDF page, segments from `--page-batch 25` PDF pages are accumulated and packed into completely full 40-row `_dev_` and `_tam_` faux pages, cutting the number of faux images and Google Drive API calls by **~50%**. While `ThreadPoolExecutor` workers upload faux pages to Google Drive as `application/vnd.google-apps.document` and export `text/plain` (`.jpg.txt`, avoiding base64 image bloat by $250\times$), the main thread concurrently runs [`detect_blocks.split_page_sections`](../../training/scripts/script_cls/detect_blocks.py) on the 300-DPI page images to locate horizontal separator rules (`---`).
- **Fault-Tolerant Tag Parsing & Boundary-Word Deduplication ([`parse_tagged`](../../training/scripts/script_cls/pipeline.py#L734-L804) & [`reconstruct`](../../training/scripts/script_cls/pipeline.py#L807-L876)):** Drive OCR occasionally reads `#T01` as `#TOI`, `#T0l`, `#101`, or `#T 01`. Our parser normalizes OCR confusions (`O/o/Q -> 0`, `I/l/| -> 1`, `S -> 5`, `B -> 8`) and enforces sequential tag plausibility (`1 <= idx <= 40`). Reconstructed runs on each line are joined using `pre_space` (concatenating `dev + tam` intra-word splits directly without whitespace when `pre_space == False`), and [`_dedup_join`](../../training/scripts/script_cls/pipeline.py#L761-L804) removes duplicate boundary words caused when a slight kerning split repeats a 1–2 character fragment across adjacent strips.
- **Deterministic 5-Commentary Structural Dissection ([`split_5_tikas.py`](file:///home/vvasuki/gitland/sanskrit-coders/doc_curation/doc_curation_projects/vedaanta/rts/split_5_tikas.py)):** Using the horizontal separator rules (`---`) detected on each page and the numbered commentary markers (`(१)`, `(२)`, `(३)`, `(४)`, `(५)` / `1.`..`5.`), each page of Volumes 1–3 is deterministically partitioned into its root text (`<details open><summary>मूलम्</summary>...</details>`) and its five individual commentaries (*Sāradīpikā*, *Sārāsvādinī*, *Sāraprakāśikā*, *Sāravivaraṇī*, and *Sāraprakāśikā-saṅgrahaḥ*).

---

## 5. Experimental Evaluation and Ablation Studies

### 5.1 Controlled End-to-End Benchmark Across 6 OCR Paradigms (`RTS_sb_01`)

To evaluate end-to-end recognition accuracy under controlled conditions, we constructed a ground-truth benchmark ([`evaluation/summary.md`](../../evaluation/summary.md)) consisting of **5 hand-verified blocks** (**89 printed lines**, **6,193 non-space Unicode characters**) from *Sārabodhinī* Volume 1 (`RTS_sb_01`), spanning both Sanskrit-heavy root text (`core`) and dense intra-word mixed Maṇipravāḷa commentary (`comm`).

Following standard OCR evaluation protocol (`evaluation/eval_cer.py`), both hypothesis and reference texts are normalized by stripping editorial brackets, normalizing punctuation, removing spaces, and mapping Devanāgarī digits (`०`–`९`) to ASCII (`0`–`9`). Character Error Rate is computed via exact Levenshtein edit distance:
$$\text{CER} = \frac{S + D + I}{N_{\text{ref}}}$$

| Rank | Pipeline / Architecture | Engine Type | `p00029_core` (Sanskrit-heavy) | `p00029_comm` (Mixed Comm.) | `p00030_core` (Mixed Root) | `p00030_comm` (Mixed Comm.) | `p00031_comm` (Mixed Comm.) | **Overall Micro-Avg CER** | **Relative Error Reduction** |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| **1** | **`tadevaOCR` (Proposed)** | **Patch CNN (`script_cls.pt`) + Phrase Strips + Anchored Drive OCR** | **0.097** | **0.114** | **0.126** | **0.105** | **0.118** | **0.112** | **Baseline Best (1.00×)** |
| 2 | `scriptSepTessOCR` | Patch CNN (`script_cls.pt`) + Routed Tesseract (`san` / `tam`) | 0.174 | 0.231 | 0.245 | 0.228 | 0.241 | **0.227** | $2.03\times$ higher error |
| 3 | `devTamCharsOCR` (Line CTC) | Custom Kraken VGSL biLSTM-CTC (16.5k lines, 40 epochs) | 0.140 | 0.285 | 0.262 | 0.244 | 0.251 | **0.238** | $2.13\times$ higher error |
| 4 | `bilingualTessOCR` | Unsplit Tesseract 5 (`-l tam+san --psm 6`) | 0.289 | 0.368 | 0.371 | 0.359 | 0.364 | **0.352** | $3.14\times$ higher error |
| 5 | `unsplitDriveOCR` | Unsplit Google Drive Document OCR (Raw Block Image) | 0.312 | 0.461 | 0.448 | 0.452 | 0.449 | **0.435** | $3.88\times$ higher error |
| 6 | `paddleDualOCR` | PaddleOCR PP-OCRv4 (`ta` + `hi`, confidence-routed) | 0.684 | 0.719 | 0.715 | 0.708 | 0.714 | **0.710** | $6.34\times$ higher error |

**Key Findings from Table 3:**
1. **3.88× Error Reduction Over Unsplit Google Drive OCR (`0.435` $\rightarrow$ `0.112` CER):** The exact same underlying cloud recognizer (Google Drive Document OCR) goes from the second-worst engine (`0.435` CER) when fed raw mixed-script blocks to the **undisputed best engine (`0.112` CER)** when fed script-separated phrase strips on context-engineered faux pages.
2. **2.03× Error Reduction Over Script-Separated Tesseract (`0.227` $\rightarrow$ `0.112` CER):** Using the identical `script_cls.pt` segmentation but routing the resulting mono-script strips to Google Drive OCR rather than single-script Tesseract (`san` / `tam`) cuts the character error rate in half, demonstrating the superior visual-linguistic decoding power of large pretrained document models once script homogeneity is satisfied.
3. **Inspection of Residual Errors (`0.112` CER):** A detailed line-by-line audit of `p00029` ([`evaluation/pilot_v1/p00029_errors_marked.md`](../../evaluation/pilot_v1/p00029_errors_marked.md)) revealed that over half of the residual `0.112` CER against the Wikisource reference is attributable to (a) editorial square-bracket insertions `[...]` and footnote markers in the Wikisource text that are absent from the physical print, and (b) legitimate orthographic variants between print editions (e.g., printed `इष्या` vs. normalized `ईर्ष्या`). True visual OCR errors on `p00029` amounted to just **5 characters out of ~1,680 (`~0.3%` true visual CER)**, such as a blurred *anusvāra* (`अहं` $\rightarrow$ `अह`) or a faint *pulli* on a word-final Tamil consonant (`न्` vs. `न`).

---

### 5.2 Intra-Word Script Boundary Detection: Structural Geometry vs. Learned CNNs

We evaluated six distinct methods for locating the exact horizontal split point $x^* \in [0, W]$ inside **14,208 real `dev → tam` mixed-script word crops** from `RTS_sb_01` and `RTS_sb_02` ([`summary/models/obsolete/script-boundary-classifier.md`](../models/obsolete/script-boundary-classifier.md)). Ground-truth boundaries were established by aligning rendered Devanāgarī stems against the mixed word crop.

Let $\hat{r}, r \in [0, 1]$ denote the predicted and ground-truth normalized split positions ($x / W$). We report Mean Absolute Error ($\text{MAE} = \mathbb{E}[|\hat{r} - r|]$), pixel accuracy within $\pm 5\text{ px}$ (`p@5`) and $\pm 10\text{ px}$ (`p@10`), and sub-word segment IoU Precision/Recall/F1.

| Boundary Detection Method | Architecture / Signal | Boundary MAE $\downarrow$ | `p@5 px` $\uparrow$ | `p@10 px` $\uparrow$ | Segment Precision $\uparrow$ | Segment Recall $\uparrow$ | Segment F1 $\uparrow$ |
|---|---|---:|---:|---:|---:|---:|---:|
| **1. Projection-Glyph DP (`projection-glyph`)** | **Vertical ink valleys + per-glyph Shiro-rekhā / baseline density DP** | **0.0320** | **0.718** | **0.884** | **0.964** | **0.959** | **0.961** |
| **2. Shiro-Rekhā Headline Rule (`shiro-rekhA`)** | **Longest upper-62% horizontal dark run ($\text{len} \ge 0.22W$, $\text{start} \le 0.45W$)** | **0.0341** | **0.702** | **0.871** | **0.958** | **0.952** | **0.955** |
| 3. Retrained Patch CNN + `_safe_split_x` (`script_cls.pt`) | 48×48 Real-Scan Residual CNN (stride 8 px) + Min-Ink Valley Snap ($\le 6\text{ px}$) | 0.0412 | 0.684 | 0.865 | 0.954 | 0.951 | 0.952 |
| 4. Combined 2-Channel Boundary CNN (`boundary_comb.pt`) | Grayscale + Horizontal Run-Length Channel (`64×256` Conv + `1×64` Head) | 0.1551 | 0.456 | 0.612 | 0.891 | 0.862 | 0.876 |
| 5. Column Transition Classifier (`col_cls.pt`) | `64×256` 2D CNN + 1D Conv Column Head (Synthetic-Trained) | 0.2140 | 0.312 | 0.489 | 0.798 | 0.764 | 0.781 |
| 6. 1D Vertical Profile CNN (`boundary_1d.pt`) | 4-Channel 1D Conv over Vertical Projection Profiles | 0.2480 | 0.245 | 0.410 | 0.742 | 0.701 | 0.721 |

**Why Structural Rules and Patch+Min-Ink Outperform End-to-End Coordinate Regression CNNs:**
Direct coordinate regression CNNs (`boundary_comb.pt`, MAE `0.1551`) resize variable-width words (ranging from $45\text{ px}$ to $480\text{ px}$) to a fixed `64×256` tensor, distorting aspect ratios and blurring the 1–3 px *Shiro-rekhā* termination point. In contrast, the explicit *Shiro-rekhā* headline rule and the fixed-aspect-ratio $48 \times 48$ sliding patch CNN (`script_cls.pt`) combined with full-resolution minimum-ink snapping (`_safe_split_x`) preserve exact pixel geometry. Moreover, unlike the pure `dev → tam` *Shiro-rekhā* rule (which assumes a single transition from left Devanāgarī to right Tamil), the sliding patch CNN + `_safe_split_x` handles **arbitrary multi-transition words** such as `tam → dev → tam` (`இவ்வधिकारार्थம்`) without structural assumptions.

---

### 5.3 The Synthetic-to-Scan Domain Trap Across Granularities

Table 5 compares all script identification models trained during the project, highlighting the dramatic performance divergence between synthetic font-rendered validation sets and real 1960s letterpress scans.

| Model | Granularity | Parameters | Training Data | Synthetic Val Acc. | Real-Scan Acc. (`RTS_sb_01`/`02`) | Status & Diagnostic Finding |
|---|---|---:|---|---:|---:|---|
| **`script_cls.pt` (Retrained)** | **48×48 Patch (stride 8 px)** | **1.2M** | **69,366 real word crops (`word-image-maps`)** | **0.9960** | **0.9921** (`dev` `0.989`, `tam` `0.995`, `other` `0.966`) | **Active Production Model.** Local 48×48 receptive field + real-scan retraining achieves near-oracle script separation. |
| `script_cls_synth_backup.pt` | 48×48 Patch | 1.2M | 240,000 synthetic patches (12 fonts) | 0.9960 | 0.9140 | Degrades on ink-bleed and broken *Shiro-rekhā* strokes without real-scan fine-tuning. |
| `script_sep.pt` | 64×256 Whole Word (`dev`/`tam`/`mixed`/`other`) | 422K | 112,948 real word boxes (`train_words_fixed`) | — | 0.9410 (`mixed` F1 = `0.862`) | Whole-word aspect-ratio warping misses short 1–2 char Tamil suffixes on long Sanskrit compounds. |
| `word_cls.pt` | 48×192 Whole Word | 380K | 180,000 synthetic word renders | 0.9942 | **0.7790** | **Severe Domain Trap:** Overfits clean synthetic kerning; misclassifies 22.1% of real scanned words. |
| `line_cls.pt` | 48×512 Whole Line | 510K | Synthetic line renders | 0.9980 | **0.6000** (12/20 blocks) | Cannot handle inline Maṇipravāḷa switching; misclassifies mixed commentary lines. |

---

### 5.4 Ablation Study: Faux-Page Granularity and Phonotactic Script Anchors

To isolate the impact of (i) phrase-strip coalescing vs. isolated word crops and (ii) the 6-consonant phonotactic script anchor (`கஙசஞடண` / `कखगघङच`), we conducted an ablation study on the 4-page Volume 1 Maṇipravāḷa benchmark (`p0034`, `p0035`, `p0040`, `p0042` — containing 136 printed lines across root text and 5 commentaries) and on Volume 4 (`806` PDF pages, including the key regression page `printed p. 408 / PDF p. 417`).

| Faux-Page Configuration | Granularity Sent to Drive OCR | Script Anchor on Narrow Rows (`w < 460 px`) | Vol. 1 Pilot `#Txx` Tag Recovery (`568` tags) | Vol. 1 Pilot Intra-Word Mixed Words Recovered | Vol. 1 Pilot Stray ASCII/Latin Chars $\downarrow$ | Vol. 4 Full-Book Output (`806` PDF pp.) |
|---|---|---|---:|---:|---:|---|
| **A. Unsplit Linewise Faux Pages** (`4.drive.linewise_unsplit.md`) | Full unsplit mixed line | None | 551 / 568 (97.0%) | **0** (all minority scripts transliterated) | 312 | `1,139,962` chars; `0` mixed words on p. 408 |
| **B. Wordwise Shiro-Rekhā Faux Pages** (`4.drive.wordwise.md`) | Isolated 1-word / sub-word crops | None | 412 / 568 (72.5%) | 14 (76%–100% Dev. words dropped) | 489 | `819,205` chars (**-28.3% char loss**; catastrophic context starvation) |
| **C. Phrase-Strip Faux Pages (Unanchored)** | Coalesced same-script phrase strips | None | 562 / 568 (98.9%) | 104 | 74 (`6T601M,`, `LD IT 60T,`, `(LOLOLD.`) | — |
| **D. Phrase-Strip + 6-Consonant Script Anchor (Proposed)** | **Coalesced same-script phrase strips** | **`கஙசஞடண` / `कखगघङच` + `28 px` tag gap** | **568 / 568 (100.0%)** | **110** | **4 (-94.6% reduction)** | **`1,152,477` chars (`+12,515` chars over A; `798` `[[P]]` markers)** |

**Qualitative Verification of Intra-Word Mixed Reconstruction on Volume 1 (`p0034`–`p0042`):**
With Configuration D, intra-word *Maṇipravāḷa* tokens are split cleanly at the *Shiro-rekhā* junction, recognized in their native script on `_dev_` and `_tam_` faux pages with full phonotactic anchoring, and seamlessly re-joined (`pre_space = False`):
- `पुरुषार्थलाभ` (`dev`) + `த்தில்` (`tam`) $\rightarrow$ `पुरुषार्थलाभத்தில்`
- `अनुवदि` (`dev`) + `த்தாரிறே?` (`tam`) $\rightarrow$ `अनुवदिத்தாரிறே?`
- `समाश्रयि` (`dev`) + `க்கையில்` (`tam`) $\rightarrow$ `समाश्रयिக்கையில்`
- `स्वरूपप्राप्त` (`dev`) + `மான` (`tam`) $\rightarrow$ `स्वरूपप्राप्तமான`
- `प्रपदन` (`dev`) + `த்தாலே` (`tam`) $\rightarrow$ `प्रपदनத்தாலே`
- `भगवल्लक्ष्म्यादि` (`dev`) + `களில்` (`tam`) $\rightarrow$ `भगवल्लक्ष्म्यादिகளில்`

---

### 5.5 Full 4-Volume Production Deployment and 5-Commentary Dissection

We executed the complete pipeline across all four volumes of *Vyākhyāna-pañcaka-sahita Śrīmad-Rahasya-Traya-Sāra* (**2,970 PDF pages**). Table 7 reports the corpus-level metrics for the raw volume outputs ([`raw/word-split-drive/{1,2,3,4}.md`](file:///home/vvasuki/gitland/vishvAsa/rAmAnujIyam/content/tattvam/venkaTa-nAtha-shAkhA/venkaTanAthaH/rahasya-traya-sAraH/5-tIkAH/raw/word-split-drive/1.md)), and Table 8 reports the automated structural dissection of Volumes 1–3 into the five individual commentary sub-corpora across all 33 *adhikāras* (`00`–`32`).

#### Table 7: Full 4-Volume Production OCR Statistics (`raw/word-split-drive/`)

| Volume | Source PDF Pages | Non-Blank `[[P]]` Pages | File Size (Bytes) | Unicode Characters | Reconstructed Lines | Mixed-Script Lines (`dev` + `tam`) | Intra-Word Mixed Tokens (`dev`+`tam` joined) | Detected `---` Section Rules |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Volume 1 (`1.md`)** | 656 | 644 | 4,610,282 | 1,684,759 | 22,392 | 12,719 (56.8%) | 18,113 | 2,403 |
| **Volume 2 (`2.md`)** | 754 | 743 | 5,153,442 | 1,883,180 | 25,732 | 13,608 (52.9%) | 16,193 | 3,051 |
| **Volume 3 (`3.md`)** | 754 | 742 | 5,363,389 | 1,954,217 | 26,998 | 14,538 (53.8%) | 22,766 | 3,293 |
| **Volume 4 (`4.md`)** | 806 | 798 | 3,119,758 | 1,152,477 | 22,076 | 2,110 (9.6%) | 78 | — |
| **Total (Vols. 1–4)** | **2,970** | **2,927** | **18,246,871** | **6,674,633** | **97,198** | **42,975 (44.2%)** | **57,150** | **8,747** |

*(Note: Volume 4 contains Primarily Tamil-script glosses, verse indices, and appendices, explaining its lower proportion of intra-word mixed tokens compared to the five core Maṇipravāḷa commentaries in Volumes 1–3).*

#### Table 8: Automated Structural Dissection of Volumes 1–3 into Per-Commentary, Per-Adhikāra Files

Across the **2,103 chapter pages** of Volumes 1–3 (`Adhikāras 00–32`), [`split_5_tikas.py`](file:///home/vvasuki/gitland/sanskrit-coders/doc_curation/doc_curation_projects/vedaanta/rts/split_5_tikas.py) partitioned all **11,683 visual blocks** (`1,691` root-text *mūlam* blocks + `9,992` commentary blocks) into 5 commentary directories × 33 *adhikāra* Markdown files (**165 files total**) with **0 unresolved blocks (100.00% structural resolution)**. Each page in every commentary file preserves the exact printed page number (`[[123]]`), embeds the corresponding root text inside `<details open><summary>मूलम्</summary>...</details>`, and follows it with that specific commentary's text.

| Commentary Sub-Corpus | Author | Adhikāra Files | Covered Chapter Pages | Embedded `मूलम्` Blocks | Total Bytes (incl. shared `मूलम्`) | Unique Commentary Characters |
|---|---|---:|---:|---:|---:|---:|
| **1. `sAra-dIpikA`** (*Sāradīpikā*) | Rāmānuja Yati | 33 | 2,103 | 1,691 | 3,642,156 | 762,140 |
| **2. `sArAsvAdinI`** (*Sārāsvādinī*) | Puruṣottama Naidu | 33 | 2,103 | 1,691 | 6,189,415 | 1,676,514 |
| **3. `sAra-prakAshikA`** (*Sāraprakāśikā*) | Śrīnivāsa Rāghavadāsa | 33 | 2,103 | 1,691 | 4,178,075 | 943,021 |
| **4. `sAra-vivaraNI`** (*Sāravivaraṇī*) | Gopāladāsa | 33 | 2,103 | 1,691 | 4,162,766 | 948,542 |
| **5. `sAra-prakAshikA-sangrahaH`** (*Sāraprakāśikā-saṅgrahaḥ*) | — | 33 | 2,103 | 1,691 | 2,986,456 | 522,790 |
| **Root Text (`मूलम्` shared layer)** | **Vedānta Deśika** | *(in all 165)* | **1,691** | **1,691** | — | **625,248** |
| **Combined Structured Repository** | — | **165** | **2,103** | **1,691 unique** | **21,158,868** | **5,478,255** |

---

## 6. Discussion: Diagnostic Case Studies and Lessons Learned

Beyond quantitative metrics, three diagnostic discoveries from our engineering history ([`history.md`](../history.md)) hold broader lessons for low-resource and multiscript document analysis:

### 6.1 The Tesseract Glyph-Chimera Illusion: Why OCR Output Cannot Serve as Script Ground Truth
In early audits of Volume 4 (`printed p. 408 / PDF p. 417`), bilingual Tesseract (`tam+san`) output contained apparent mixed-script tokens such as `(द्रयाணன்`, `(द्रया னை`, and `(द्रया ண்`. When an initial Google Drive OCR run produced pure Devanāgarī tokens at those positions (`(द्वया`), it was initially hypothesized that Drive OCR had "dropped" the Tamil half of those mixed words.

Visual inspection of the 300-DPI scan crop (`p0417`) revealed the opposite: the physical printed page contained the pure Sanskrit hyphenated compound **`(द्वया-)`** (referring to the *Dvaya-mantra*, followed by a line-break hyphen or parenthetical closing). Because the lower loop of the Devanāgarī conjunct `द्व` (`dva`) and its trailing hyphen/parenthesis visually resemble Tamil `ணன்` / `னை` to an unconstrained bilingual LSTM, Tesseract hallucinated a mixed-script chimera (`(द्रयाணன்`). Similarly, our custom Kraken word recognizer (`best_0.8900.safetensors`) hallucinated **1,456 intra-word script chimeras** across Volume 4 (`एா`, `எः`, `५ெ`, `तர்`) whenever a smudge or ligature resembled a glyph from the other alphabet.
> **Lesson 1:** Never infer ground-truth script distributions from bilingual CTC output; unconstrained decoding over a union alphabet $\Sigma_1 \cup \Sigma_2$ systematically fabricates cross-script chimeras on ambiguous strokes.

### 6.2 Why 90.34% Word-Level Character Accuracy Fails at Book Scale
Training a dedicated Kraken 7.1.1 VGSL word recognizer (`4.1M` parameters) on **92,976 real word crops** achieved a respectable held-out **90.34% character accuracy** (**94.31% on Tamil**, **88.81% on Devanāgarī**) and **80.53% exact word accuracy**. In many academic OCR papers, 90%+ character accuracy on a historical biscriptal dataset is reported as a success. However, when deployed across a 1.15M-character volume (~180,000 words), an **80.53% word accuracy means 1 in every 5 words is wrong (~35,000 corrupted words and ~110,000 character errors per volume)**—rendering the resulting text unusable for philological scholarship without exhaustive manual re-keying.
> **Lesson 2:** When a high-capacity commercial VLM/document recognizer exists for the constituent scripts individually, **transforming the document via visual context engineering** (splitting mixed words via a 99.2%-accurate local patch classifier and re-grouping same-script phrase strips) outperforms training an end-to-end recognizer from scratch by more than $2\times$ in CER (`0.112` vs. `0.238`).

### 6.3 Unlocking Black-Box Document OCR via Layout and Phonotactic Context Engineering
Cloud document OCR engines (such as Google Drive OCR) apply undocumented internal heuristics for column segmentation and line-level language identification. Rather than treating these heuristics as unfixable failure modes, our work demonstrates that they obey predictable geometric and statistical thresholds that can be satisfied deterministically during faux-page synthesis:
1. **Horizontal vs. Vertical Grouping:** Ensuring $\Delta x_{\text{tag}\to\text{crop}} \le 0.5 \cdot \Delta y_{\text{row}\to\text{row}}$ (`28 px` vs. `54 px`) and compacting internal blank spans $> 36\text{ px}$ to $22\text{ px}$ completely prevents multi-column mis-grouping (`100.0%` tag recovery).
2. **Phonotactic Script Anchoring:** Appending a deterministic 6-consonant sequence (`கஙசஞடண` or `कखगघङच`) to narrow crops guarantees that target-script glyphs outnumber ASCII metadata tags (`#Txx`) by $\ge 2.25\times$, eliminating Latin lookalike hallucinations (`94.6%` reduction in stray ASCII characters).

---

## 7. Conclusion and Future Work

We have presented the first end-to-end OCR and structural dissection pipeline capable of handling **intra-word biscriptal *Maṇipravāḷa* documents** at multi-volume scale. By identifying and resolving the **Granularity–Context Paradox**—combining a real-scan-retrained $48 \times 48$ sliding-patch residual CNN (`0.9921` accuracy) and minimum-ink *Shiro-rekhā* splitting with coalesced same-script phrase strips and phonotactically anchored faux pages—our pipeline achieves a **0.112 Character Error Rate** on dense biscriptal commentary blocks ($3.9\times$ lower than unsplit cloud OCR and $2.1\times$ lower than custom dual-script CTC models). Applied to the complete four-volume *Vyākhyāna-pañcaka-sahita Śrīmad-Rahasya-Traya-Sāra* (**2,970 pages**, **6.67M Unicode characters**, **57,150 intra-word mixed words**), the system produced a complete, philologically usable digital edition segmented across **165 per-commentary, per-chapter files**.

Future work will explore: (i) extending the Render-and-DP alignment framework and phrase-strip faux-page architecture to **Grantha–Tamil** and **Telugu–Sanskrit** biscriptal manuscripts and early printed editions; and (ii) using our 6.67M-character aligned *Maṇipravāḷa* corpus to pretrain a biscriptal morphological language model capable of constrained lexically aware post-OCR correction using `[[OLD|NEW]]` philological diff markup.

---

## References

1. **Breuel, T. M.** (2008). The OCRopus open source OCR system. In *Document Recognition and Retrieval XV* (Vol. 6815, p. 68150F). SPIE.
2. **Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J.** (2006). Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks. In *Proceedings of the 23rd International Conference on Machine Learning (ICML)* (pp. 369–376).
3. **Kiessling, B.** (2019). Kraken—an universal text recognizer for the humanities. In *Proceedings of the Digital Humanities Conference (DH 2019)*. Utrecht, Netherlands.
4. **Mathew, M., Singh, A. K., & Jawahar, C. V.** (2016). Multilingual OCR for Indic scripts. In *12th IAPR Workshop on Document Analysis Systems (DAS)* (pp. 186–191). IEEE.
5. **Smith, R.** (2007). An overview of the Tesseract OCR engine. In *Ninth International Conference on Document Analysis and Recognition (ICDAR 2007)* (Vol. 2, pp. 629–633). IEEE.
6. **Venkatachalapathy, A. R.** (2006). *In Those Days There Was No Coffee: Writings in Cultural History*. Yoda Press (for historical context on Tamil and Maṇipravāḷa print typography).
7. **Wick, C., Reul, C., & Puppe, F.** (2020). Calamari—a high-performance Tensorflow-based deep learning package for optical character recognition. *Digital Humanities Quarterly*, 14(2).
