"""Parse 2-script core + commentary MDs into shareable units JSON.

Core source:
  vishvAsa/rAmAnujIyam/content/tattvam/venkaTa-nAtha-shAkhA/venkaTanAthaH/rahasya-traya-sAraH/mUlam/2-script/*.md
  -> only the core (mUlam), top block in PDF when present.

Commentary source:
  vishvAsa/rAmAnujIyam/content/meta/private/maNipravALa-training-data/sAra-bodhinI/*.md
  -> SAra-bodhinI commentary, main body block in PDF.

Output (under training/):
  texts/core/<adhikAra>.txt          # concatenated clean core units, with unit markers
  texts/commentary/<adhikAra>.txt
  metadata/units_core.jsonl          # {unit_id, file, title, body}
  metadata/units_commentary.jsonl
  metadata/parse_stats.json

Unit IDs like 00-0.1, 01-1.1 match <summary>मूलम् - X.Y</summary>.
Body is diplomatic (keeps Devanagari+Tamil mix, strips +++, <details> tags).
"""
import re
import json
import pathlib
import glob

BASE = pathlib.Path(__file__).resolve().parents[1]
CORE_GLOB = "/home/vvasuki/gitland/vishvAsa/rAmAnujIyam/content/tattvam/venkaTa-nAtha-shAkhA/venkaTanAthaH/rahasya-traya-sAraH/mUlam/2-script/*.md"
COMM_GLOB = "/home/vvasuki/gitland/vishvAsa/rAmAnujIyam/content/meta/private/maNipravALa-training-data/sAra-bodhinI/*.md"

def parse_file(fpath):
    t = pathlib.Path(fpath).read_text(encoding="utf-8")
    t2 = re.sub(r"^\+\+\+.*?\+\+\+", "", t, flags=re.S)
    secs = re.findall(r"<details open><summary>(.*?)</summary>(.*?)</details>", t2, flags=re.S)
    out = []
    for title, body in secs:
        title = title.strip()
        body = body.strip()
        # collapse trailing spaces, keep newlines; strip markdown image/links? keep text
        # remove excess blank lines
        body = re.sub(r"\n{3,}", "\n\n", body)
        out.append((title, body))
    return out

def dump_set(file_list, out_txt_dir, jsonl_path):
    out_txt_dir = pathlib.Path(out_txt_dir)
    out_txt_dir.mkdir(parents=True, exist_ok=True)
    all_units = []
    for f in sorted(file_list):
        if f.endswith("_index.md"):
            continue
        adh = pathlib.Path(f).stem  # e.g. 00_shrIguruparaMparAsAraH
        units = parse_file(f)
        txt_lines = []
        for title, body in units:
            # unit_id from title e.g. "मूलम् - 0.1" -> 00-0.1 (prefix adhikAra num)
            m = re.search(r"(\d+\.\d+)", title)
            uid = f"{adh.split('_')[0]}-{m.group(1)}" if m else f"{adh}-{title}"
            all_units.append({"unit_id": uid, "file": adh + ".md", "title": title, "body": body})
            txt_lines.append(f"@@UNIT:{uid} @@TITLE:{title}\n{body}\n")
        (out_txt_dir / f"{adh}.txt").write_text("\n".join(txt_lines), encoding="utf-8")
    with open(jsonl_path, "w", encoding="utf-8") as fw:
        for u in all_units:
            fw.write(json.dumps(u, ensure_ascii=False) + "\n")
    return all_units

if __name__ == "__main__":
    core_files = glob.glob(CORE_GLOB)
    comm_files = glob.glob(COMM_GLOB)
    core_units = dump_set(core_files, BASE / "texts" / "core", BASE / "metadata" / "units_core.jsonl")
    comm_units = dump_set(comm_files, BASE / "texts" / "commentary", BASE / "metadata" / "units_commentary.jsonl")
    stats = {
        "core_files": len(core_files),
        "comm_files": len(comm_files),
        "core_units": len(core_units),
        "comm_units": len(comm_units),
        "core_glob": CORE_GLOB,
        "comm_glob": COMM_GLOB,
    }
    (BASE / "metadata" / "parse_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
