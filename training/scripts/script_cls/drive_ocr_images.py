#!/usr/bin/env python3
"""OCR image files with Google Drive (run with the doc_curation venv python).

Usage: drive_ocr_images.py --key KEY file1.jpg file2.jpg ...
Writes <file>.txt next to each input.
"""
import argparse
import os
import time

from curation_utils.google import drive

ap = argparse.ArgumentParser()
ap.add_argument("--key", required=True)
ap.add_argument("files", nargs="+")
args = ap.parse_args()

client = drive.get_cached_client(google_key=args.key)
for f in args.files:
    if os.path.exists(f + ".txt"):
        continue
    client.ocr_file(local_file_path=f)
    time.sleep(0.5)
