"""Create GT text files by aligning Tesseract OCR of crops to unit bodies.

Pipeline per volume:
  1. OCR every crop (core+comm) with Tesseract tam+san (cached under metadata/ocr_<vol>/).
  2. Walk core crops in page order against unit list in order (fuzzy window match)
     -> sequential unit assignment, catches toc-range fuzz and missed cores.
  3. Assign each comm crop the unit of the last assigned core at or before its page.
  4. For each comm crop: fuzzy-align OCR to the unit body (partial_ratio_alignment),
     snap boundaries to whitespace-delimited tokens -> byte-exact substring.
     Chain constraint per unit: first crop starts at 0; later crops start >= prev end.
  5. Write gt files + update blocks.json (unit/gt/status/note), preserving
     verified / verified-visually / absent-by-design entries.
  6. Write a report CSV for review.

Usage:
  python3 training/scripts/make_gt.py --vol 01              # dry run (report only)
  python3 training/scripts/make_gt.py --vol 01 --write      # write gt + blocks.json
  python3 training/scripts/make_gt.py --vol 01 --ocr-only   # just (re)build OCR cache
"""
import argparse
import csv
import json
import os
import pathlib
import re
import subprocess
import sys
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed

from rapidfuzz import fuzz
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

BASE = pathlib.Path(__file__).resolve().parents[2]  # repo root
TESS = os.environ.get("TESSERACT_BIN", "/tmp/tess-root/usr/bin/tesseract")
TESSDATA = os.environ.get(
    "TESSDATA_PREFIX", "/tmp/tess-root/usr/share/tesseract-ocr/5/tessdata/"
)
VOL_ADHIKARA_RANGE = {"01": ("00", "22"), "02": ("23", "32")}

CORE_FLOOR = 50.0
AUTO_FLOOR = 60.0
PRESERVE_STATUS = {"verified", "verified-visually", "absent-by-design"}

DEV_RE = re.compile(r"[\u0900-\u097F]+")
ZERO_WIDTH = dict.fromkeys(map(ord, "\u200c\u200d"), None)
WORD_RE = re.compile(r"\S+")


def deva_to_tamil(s):
    def repl(m):
        try:
            return transliterate(m.group(0), sanscript.DEVANAGARI, sanscript.TAMIL)
        except Exception:
            return m.group(0)

    return DEV_RE.sub(repl, s)


def normalize(s):
    s = s.translate(ZERO_WIDTH).replace("\u00a0", " ")
    s = deva_to_tamil(s)
    s = re.sub(r"[^\u0B80-\u0BFFa-zA-Z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def token_stream(text):
    """Whitespace-delimited tokens of raw text + their (start, end) raw offsets."""
    toks, spans = [], []
    for m in WORD_RE.finditer(text):
        n = normalize(m.group(0))
        if n:
            toks.append(n)
            spans.append((m.start(), m.end()))
    return toks, spans


def ocr_crop(img_path, cache_path, workers_env):
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    env = dict(os.environ, TESSDATA_PREFIX=TESSDATA, OMP_THREAD_LIMIT="1")
    out = subprocess.run(
        [TESS, str(img_path), "stdout", "-l", "tam+san", "--psm", "6"],
        capture_output=True, text=True, env=env,
    ).stdout
    cache_path.write_text(out, encoding="utf-8")
    return out


def load_units(vol):
    lo, hi = VOL_ADHIKARA_RANGE[vol]
    units = {}
    for name in ("core", "commentary"):
        for line in open(BASE / f"training/metadata/units_{name}.jsonl", encoding="utf-8"):
            u = json.loads(line)
            adh = u["unit_id"].split("-")[0]
            if lo <= adh <= hi:
                units.setdefault(u["unit_id"], {})[name] = u["body"]
    return units


def unit_sort_key(uid):
    adh, num = uid.split("-")
    return (int(adh), tuple(int(x) for x in num.split(".")))


def core_match_score(ocr_norm, body):
    body_norm = normalize(body)
    s = fuzz.token_set_ratio(ocr_norm[:2500], body_norm[:2500])
    if len(body_norm) < 800:
        s = max(s, fuzz.partial_ratio(ocr_norm[:600], body_norm))
    return s


def snap_tokens(body, ds, de, body_toks, body_spans, starts, n_toks):
    """Map normalization-space [ds:de] to token indices; return (i, j)."""
    i = max(0, bisect_right(starts, ds) - 1)
    j = min(n_toks - 1, max(i, bisect_right(starts, max(ds, de - 1)) - 1))
    return i, j


def find_slice(body, ocr_norm):
    """Coarse-align whole OCR against body; snap to token boundaries.
    Returns (start_raw, end_raw, score) or None."""
    body_toks, body_spans = token_stream(body)
    if not body_toks:
        return None
    parts, starts, pos = [], [], 0
    for t in body_toks:
        starts.append(pos)
        parts.append(t)
        pos += len(t) + 1
    body_norm = " ".join(parts)
    if len(ocr_norm) < 20 or len(body_norm) < 20:
        return None
    al = fuzz.partial_ratio_alignment(ocr_norm, body_norm)
    if al is None:
        return None
    i, j = snap_tokens(body, al.dest_start, al.dest_end, None, None, starts, len(body_toks))
    return body_spans[i][0], body_spans[j][1], al.score


def label_for(start_raw, end_raw, body_len):
    at_start = start_raw <= 2
    at_end = end_raw >= body_len - 2
    if at_start and at_end:
        return "full"
    if at_start:
        return "head"
    if at_end:
        return "tail"
    return "part"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol", required=True, choices=["01", "02"])
    ap.add_argument("--root", default=None, help="data root (default data/RTS_sb_<vol>)")
    ap.add_argument("--write", action="store_true", help="write gt files + blocks.json")
    ap.add_argument("--ocr-only", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    data_dir = pathlib.Path(args.root) if args.root else BASE / f"data/RTS_sb_{args.vol}"
    blocks_path = data_dir / "metadata/blocks.json"
    blocks_doc = json.loads(blocks_path.read_text(encoding="utf-8"))
    blocks = blocks_doc["blocks"]
    ocr_dir = BASE / f"training/metadata/ocr_{args.vol}"
    (ocr_dir / "core").mkdir(parents=True, exist_ok=True)
    (ocr_dir / "comm").mkdir(parents=True, exist_ok=True)

    units = load_units(args.vol)
    ordered_units = sorted(units, key=unit_sort_key)

    crops = [b for b in blocks if b.get("box") and b.get("image")]
    crops.sort(key=lambda b: (int(b["page"][1:]), 0 if b["block"] == "core" else 1))
    if args.limit:
        crops = crops[: args.limit]

    # ---- stage 1: OCR ----
    todo = []
    for b in crops:
        cache = ocr_dir / ("core" if b["block"] == "core" else "comm") / f'{b["page"]}.txt'
        todo.append((b, cache))
    missing = [(b, c) for b, c in todo if not c.exists()]
    print(f"OCR: {len(todo)} crops, {len(missing)} to run, {args.workers} workers")
    if missing:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(ocr_crop, BASE / b["image"], c, None): b for b, c in missing}
            for k, fut in enumerate(as_completed(futs), 1):
                fut.result()
                if k % 50 == 0 or k == len(futs):
                    print(f"  ocr {k}/{len(futs)}")
    if args.ocr_only:
        return
    ocr_text = {b["page"] + "/" + b["block"]: c.read_text(encoding="utf-8")
                for b, c in todo}

    # ---- stage 2: core walk ----
    core_crops = [b for b in crops if b["block"] == "core"]
    core_assign = {}
    merged_next_map = {}
    ptr = 0
    report = []
    for b in core_crops:
        ocr_norm = normalize(ocr_text[b["page"] + "/core"])
        lo = max(0, ptr - 2)
        hi = min(len(ordered_units), ptr + 5)
        best, best_i = -1.0, None
        scores = {}
        if len(ocr_norm) >= 30:
            for i in range(lo, hi):
                s = core_match_score(ocr_norm, units[ordered_units[i]]["core"])
                scores[i] = s
                if s > best:
                    best, best_i = s, i
        if best_i is not None and best >= CORE_FLOOR:
            flags = []
            merged_next = None
            if best_i > lo and scores.get(best_i - 1, 0) >= max(60.0, best - 35):
                # box also contains the previous unit's mUlam (section-end page):
                # anchor on the earlier unit, next unit's core is inside this box
                merged_next = ordered_units[best_i]
                best_i -= 1
                best = scores[best_i]
                flags.append(f"merged-core:{merged_next}")
            elif best_i + 1 < hi and scores.get(best_i + 1, 0) >= max(60.0, best - 35):
                # box also contains the next unit's mUlam
                merged_next = ordered_units[best_i + 1]
                flags.append(f"merged-core-next:{merged_next}")
            uid = ordered_units[best_i]
            if best_i < ptr:
                flags.append(f"repeat-unit(prev={ordered_units[ptr-1] if ptr else None})")
            elif best_i > ptr:
                flags.append(f"skipped:{','.join(ordered_units[ptr:best_i])}")
            core_assign[b["page"]] = uid
            if merged_next:
                merged_next_map[b["page"]] = merged_next
            ptr = max(ptr, best_i + 1)
            status = "auto-approx"
            if any(f.startswith(("repeat-unit", "merged-core")) for f in flags):
                status = "needs-review"
            report.append(dict(page=b["page"], block="core", unit=uid,
                               part="full", start=0, end=len(units[uid]["core"]),
                               score=round(best, 1), status=status,
                               flags=";".join(flags)))
        else:
            core_assign[b["page"]] = None
            report.append(dict(page=b["page"], block="core", unit="", part="",
                               start="", end="", score=round(best, 1),
                               status="needs-review", flags="low-core-score"))
    # multi-page cores (unit's mUlam spans >1 page): imperfect pairs -> needs-review
    from collections import Counter as _Counter
    _core_page_count = _Counter(u for u in core_assign.values() if u)
    multipage_units = {u for u, c in _core_page_count.items() if c > 1}
    if multipage_units:
        print(f"NOTE: {len(multipage_units)} units with multi-page cores:",
              sorted(multipage_units))
        for row in report:
            if row["block"] == "core" and row["unit"] in multipage_units:
                row["status"] = "needs-review"
                row["flags"] = (row["flags"] + ";multi-page-core").strip(";")
    # core page per unit (merged-next units without own page: the marker page)
    core_page_of = {u: int(p[1:]) for p, u in core_assign.items() if u}
    for p, v in merged_next_map.items():
        if v not in core_page_of:
            core_page_of[v] = int(p[1:])
        report.append(dict(page=p, block="core", unit=v, part="", start="", end="",
                           score="", status="needs-review",
                           flags=f"core-inside-merged-box:{core_assign[p]}"))
    # missing-core regions: pages strictly between two assigned cores where units were skipped
    assigned = sorted([(int(p[1:]), u) for p, u in core_assign.items() if u])
    assigned_pages = [p for p, _ in assigned]
    uid_idx = {u: i for i, u in enumerate(ordered_units)}
    merged_units = set(merged_next_map.values())
    demote_ranges = []
    for (p1, u1), (p2, u2) in zip(assigned, assigned[1:]):
        gap_units = [u for u in ordered_units[uid_idx[u1] + 1:uid_idx[u2]]
                     if u not in merged_units]
        if gap_units:
            demote_ranges.append((p1 + 1, p2, gap_units))
    skipped = [u for i, u in enumerate(ordered_units) if u not in set(core_assign.values())]
    if skipped:
        print(f"NOTE: {len(skipped)} units have no assigned core: {skipped[:10]}")
        for u in skipped:
            if u not in merged_units:
                report.append(dict(page="", block="core", unit=u, part="", start="",
                                   end="", score="", status="needs-review",
                                   flags="no-core-block-found"))

    # ---- stage 3: comm assignment ----
    # markers: after a merged-core page, until the next assigned core page,
    # comm crops belong to the next unit (whose core was merged into that box)
    markers = []
    for p, v in merged_next_map.items():
        mp = int(p[1:])
        ncp = next((q for q in assigned_pages if q > mp), None)
        markers.append((mp, v, ncp if ncp is not None else 10 ** 6))
    comm_runs = []  # (block, unit or None)
    cur = None
    for b in crops:
        pg = int(b["page"][1:])
        if b["block"] == "core":
            cur = core_assign.get(b["page"])
        else:
            c = cur
            for mp, mv, ncp in markers:
                if mp < pg < ncp:
                    c = mv
            comm_runs.append((b, c))
    demoted = {}  # key -> skipped units note
    for start, end, sus in demote_ranges:
        for b, c in comm_runs:
            pg = int(b["page"][1:])
            if start <= pg < end:
                demoted[b["page"] + "/comm"] = (c or "", ";".join(sus))
    if demoted:
        others = []
        for b, c in comm_runs:
            key = b["page"] + "/comm"
            if key in demoted:
                u, sus = demoted[key]
                if not u:
                    u = sus.split(",")[0]
                row = dict(page=b["page"], block="comm", unit=u, part="",
                           start="", end="", score="", status="needs-review",
                           flags=f"core-missing-region:{sus}")
                if u and u in units:
                    res = find_slice(units[u]["commentary"],
                                     normalize(ocr_text[b["page"] + "/comm"]))
                    if res:
                        row["start"], row["end"] = res[0], res[1]
                report.append(row)
            else:
                others.append((b, c))
        comm_runs = others

    # ---- stage 4: per-unit chain + offsets ----
    by_unit = {}
    for b, u in comm_runs:
        if u:
            by_unit.setdefault(u, []).append(b)

    def next_core_page(uid):
        p = core_page_of.get(uid)
        if p is None:
            return None
        return next((q for q in assigned_pages if q > p), None)

    slices = {}
    for uid, crops_u in by_unit.items():
        body = units[uid]["commentary"]
        n = len(crops_u)
        ncp = next_core_page(uid)
        last_page = int(crops_u[-1]["page"][1:])
        force_len = ncp is None or last_page == ncp - 1
        toks, spans = token_stream(body)
        tok_starts = [s for s, _ in spans]
        from bisect import bisect_left

        def snap_bnd(bnd):
            if bnd is None:
                return None
            i = bisect_left(tok_starts, bnd - 2)
            return tok_starts[i] if i < len(tok_starts) else len(body)

        def refine_start(s_est, ocr_toks):
            """Anchor onto first OCR token if it occurs verbatim near s_est."""
            if not ocr_toks or s_est is None:
                return s_est
            needle = next((t for t in ocr_toks[:4] if len(t) >= 6), None)
            best = None
            for i, t in enumerate(toks):
                if t == needle and abs(tok_starts[i] - s_est) <= 250:
                    if best is None or abs(tok_starts[i] - s_est) < abs(tok_starts[best] - s_est):
                        best = i
            return tok_starts[best] if best is not None else s_est

        # independent OCR estimates per crop
        ests = []
        for b in crops_u:
            key = b["page"] + "/comm"
            ocr_norm = normalize(ocr_text[key])
            toks_o, _ = token_stream(ocr_text[key])
            res = find_slice(body, ocr_norm)
            if res:
                res = (refine_start(res[0], toks_o), res[1], res[2])
            ests.append((key, ocr_norm, res))
        # boundaries: b[k] between crop k and k+1 ~ start estimate of crop k+1
        bounds = []
        for k in range(n - 1):
            key, ocr_norm, res = ests[k]
            key_n, ocr_norm_n, res_n = ests[k + 1]
            s_next = res_n[0] if res_n else None
            e_cur = res[1] if res else None
            flags_k = []
            if s_next is None and e_cur is None:
                bounds.append(None)
                continue
            if s_next is not None and e_cur is not None and abs(s_next - e_cur) > 400:
                flags_k.append(f"boundary-disagree:{e_cur}vs{s_next}")
            if s_next is not None and e_cur is not None and s_next < e_cur - 400:
                flags_k.append(f"boundary-inverted:{e_cur}vs{s_next}")
            bnd = s_next if s_next is not None else e_cur
            bnd = snap_bnd(bnd)
            if bounds and bounds[-1] is not None and bnd < bounds[-1][0]:
                j = bisect_left(tok_starts, bounds[-1][0] + 1)
                bnd = tok_starts[j] if j < len(tok_starts) else len(body)
                flags_k.append("forced-monotonic")
            bounds.append((bnd, flags_k))
        prev_end = 0
        for k, (key, ocr_norm, res) in enumerate(ests):
            b = crops_u[k]
            flags = []
            if k == 0:
                s = 0
                if res and res[0] > 100:
                    flags.append(f"head-start-offset:{res[0]}")
            else:
                if bounds[k - 1] is None:
                    flags.append("chain-broken")
                    s = prev_end
                else:
                    s = bounds[k - 1][0]
                    flags.extend(bounds[k - 1][1])
            if k == n - 1:
                if force_len:
                    e = len(body)
                else:
                    flags.append("chain-truncated")
                    e = res[1] if res else prev_end + 25
            elif bounds[k] is None:
                flags.append("chain-broken-end")
                e = max(prev_end + 25, (res[1] if res else prev_end + 25))
            else:
                e = bounds[k][0]
            e = max(e, s)
            if e - s < 25:
                flags.append("tiny-slice")
            if res:
                slice_score = max(res[2], fuzz.token_set_ratio(
                    ocr_norm, normalize(body[s:e])))
            else:
                slice_score = 0.0
            prev_end = e
            if k == 0 and uid in merged_units:
                flags.append("first-crop-after-merged-core")
            slices[key] = (s, e)
            part = label_for(s, e, len(body))
            status = "needs-review" if "first-crop-after-merged-core" in flags else "auto-approx"
            report.append(dict(page=b["page"], block="comm", unit=uid, part=part,
                               start=s, end=e, score=round(slice_score, 1),
                               status=status, flags=";".join(flags)))

    # ---- stage 5: statuses + write ----
    if args.write:
        (data_dir / "gt/core").mkdir(parents=True, exist_ok=True)
        (data_dir / "gt/comm").mkdir(parents=True, exist_ok=True)
    entry_by_key = {}
    for b in blocks:
        if b.get("box"):
            entry_by_key[b["page"] + "/" + b["block"]] = b

    def mark_doubtful_img(block, page):
        img_dir = data_dir / "images" / block
        plain, doubt = img_dir / f"{page}.jpg", img_dir / f"{page}_doubtful.jpg"
        if args.write:
            if plain.exists() and doubt.exists():
                plain.unlink()
            elif plain.exists():
                plain.rename(doubt)
        return f"data/RTS_sb_{args.vol}/images/{block}/{page}_doubtful.jpg"

    def unmark_doubtful_img(block, page):
        img_dir = data_dir / "images" / block
        plain, doubt = img_dir / f"{page}.jpg", img_dir / f"{page}_doubtful.jpg"
        if args.write and doubt.exists() and not plain.exists():
            doubt.rename(plain)
        return f"data/RTS_sb_{args.vol}/images/{block}/{page}.jpg"

    for row in report:
        if row["status"] == "auto-approx":
            floor = CORE_FLOOR if row["block"] == "core" else AUTO_FLOOR
            if row["score"] < floor:
                row["status"] = "needs-review"
                row["flags"] = (row["flags"] + ";low-score").strip(";")
            elif row["score"] < AUTO_FLOOR:
                row["flags"] = (row["flags"] + ";low-score").strip(";")
        if row["status"] == "needs-review":
            continue
        key = row["page"] + "/" + row["block"]
        b = entry_by_key.get(key)
        if b is None or b.get("status") in PRESERVE_STATUS:
            continue
        b["image"] = unmark_doubtful_img(row["block"], row["page"])
        old_gt = b.get("gt")
        uid = row["unit"]
        body = units[uid]["core" if row["block"] == "core" else "commentary"]
        s, e = row["start"], row["end"]
        rel_gt = f'data/RTS_sb_{args.vol}/gt/{row["block"]}/{row["page"]}.txt'
        sliced = body[s:e]
        if e < len(body):
            sliced = sliced.rstrip()
        if args.write:
            (BASE / rel_gt).write_text(sliced, encoding="utf-8")
            if old_gt and old_gt != rel_gt:
                f = BASE / old_gt
                if f.exists():
                    f.unlink()
        b["gt"] = rel_gt
        b["unit"] = f'{uid}-{row["part"]}'
        b["status"] = "auto-approx"
        b["note"] = (f'auto-GT make_gt.py: {uid}-{row["part"]} '
                     f'[{s}:{e}] score={row["score"]}'
                     + (f' flags={row["flags"]}' if row["flags"] else ""))
    # needs-review rows: image renamed to *_doubtful.jpg + best-guess text as *_doubtful.txt
    nr_img = nr_text = 0
    for row in report:
        if row["status"] != "needs-review" or not row["page"]:
            continue
        if "core-inside-merged-box" in row["flags"] or "no-core-block-found" in row["flags"]:
            continue
        b = entry_by_key.get(row["page"] + "/" + row["block"])
        if b is None or b.get("status") in PRESERVE_STATUS:
            continue
        uid = row["unit"] or None
        gt_rel = None
        if uid and uid in units and row["start"] != "" and row["end"] != "":
            body = units[uid]["core" if row["block"] == "core" else "commentary"]
            s, e = int(row["start"]), int(row["end"])
            if e - s >= 10:
                txt = body[s:e]
                if e < len(body):
                    txt = txt.rstrip()
                gt_rel = (f'data/RTS_sb_{args.vol}/gt/{row["block"]}/'
                          f'{row["page"]}_doubtful.txt')
                if args.write:
                    (BASE / gt_rel).write_text(txt, encoding="utf-8")
                nr_text += 1
        old_gt = b.get("gt")
        if args.write and old_gt and old_gt != gt_rel:
            f = BASE / old_gt
            if f.exists():
                f.unlink()
        b["image"] = mark_doubtful_img(row["block"], row["page"])
        nr_img += 1
        b["gt"] = gt_rel
        b["unit"] = (f'{uid}-{row["part"]}' if row["part"] else f"{uid}-doubtful") if uid else None
        b["status"] = "needs-review"
        b["note"] = f'make_gt.py: needs-review ({row["flags"]})'
    if nr_img:
        print(f"doubtful pairs: {nr_img} images marked, {nr_text} best-guess texts written")
    if args.write:
        blocks_path.write_text(json.dumps(blocks_doc, indent=1, ensure_ascii=False) + "\n",
                               encoding="utf-8")
        print(f"wrote {blocks_path}")

    # ---- report ----
    rep_path = BASE / f"training/metadata/gt_report_{args.vol}.csv"
    if args.write or True:
        with open(rep_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["page", "block", "unit", "part", "start",
                                              "end", "score", "status", "flags"])
            w.writeheader()
            w.writerows(report)
    ok = [r for r in report if r["status"] == "auto-approx"]
    nr = [r for r in report if r["status"] == "needs-review"]
    print(f"report: {rep_path}")
    print(f"auto-approx: {len(ok)}  needs-review: {len(nr)}  (of {len(report)} boxed crops)")
    from collections import Counter
    print("needs-review by block:", Counter(r["block"] for r in nr))
    for r in nr[:15]:
        print(f"  {r['page']} {r['block']:4s} score={r['score']} {r['flags']}")


if __name__ == "__main__":
    main()
