#!/usr/bin/env python3
"""OCR image files with Google Drive (run with the doc_curation venv python).

Usage: drive_ocr_images.py --key KEY [--workers 6] file1.jpg file2.jpg ...
Writes <file>.txt next to each input using text/plain export (no embedded base64 images).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import os
import random
import socket
import threading
import time

try:
    from curation_utils.google import drive
    DriveClient = drive.DriveClient
except ImportError:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    socket.setdefaulttimeout(600)

    class DriveClient:
        def __init__(self, google_key, folder_key="0B1_QBT-hoqqVa0xDRHFmM2EzWUk"):
            scopes = ["https://www.googleapis.com/auth/drive"]
            token_file = os.path.join(os.path.dirname(google_key), "tokens.json")
            creds = Credentials.from_authorized_user_file(token_file, scopes)
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                with open(token_file, "w") as fh:
                    fh.write(creds.to_json())
            self.service = build("drive", "v3", credentials=creds, cache_discovery=False)
            self.folder_key = folder_key

        def upload(self, local_file_path, mime="application/vnd.google-apps.document"):
            media = MediaFileUpload(local_file_path, mimetype=mime, resumable=True, chunksize=1024 * 1024)
            req = self.service.files().create(
                body={"name": os.path.basename(local_file_path), "mimeType": mime, "parents": [self.folder_key]},
                media_body=media,
                supportsAllDrives=True,
            )
            return req.execute(num_retries=3)

        def download_text(self, local_file_path, file_id, mime_type="text/plain"):
            req = self.service.files().export_media(fileId=file_id, mimeType=mime_type)
            with io.FileIO(local_file_path, "wb") as fh:
                downloader = MediaIoBaseDownload(fh, req)
                done = False
                while not done:
                    _, done = downloader.next_chunk(num_retries=5)

        def delete_file(self, file_id):
            self.service.files().delete(fileId=file_id).execute(num_retries=3)

ap = argparse.ArgumentParser()
ap.add_argument("--key", required=True)
ap.add_argument("--workers", type=int, default=int(os.environ.get("DRIVE_WORKERS", "5")))
ap.add_argument("files", nargs="+")
args = ap.parse_args()

# Initialize/refresh credentials once on the main thread before spawning workers
try:
    DriveClient(google_key=args.key)
except Exception:
    pass
_tls = threading.local()
_lock = threading.Lock()


def _get_client(reset=False):
    if reset or not hasattr(_tls, "client"):
        with _lock:
            _tls.client = DriveClient(google_key=args.key)
    return _tls.client


def _ocr_one(f):
    txt_path = f + ".txt"
    if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
        return f
    tmp_path = txt_path + ".tmp"
    last_err = None
    for attempt in range(6):
        client = _get_client(reset=(attempt > 0))
        fid = None
        try:
            fid = client.upload(local_file_path=f)["id"]
            client.download_text(
                local_file_path=tmp_path, file_id=fid, mime_type="text/plain"
            )
            os.replace(tmp_path, txt_path)
            return f
        except Exception as e:
            last_err = e
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            time.sleep(2.5 * (attempt + 1) + random.uniform(0.2, 2.0))
        finally:
            if fid is not None:
                try:
                    client.delete_file(fid)
                except Exception:
                    pass
    raise RuntimeError(f"Drive OCR failed after 6 attempts for {f}: {last_err}")


todo = [
    f for f in args.files
    if not (os.path.exists(f + ".txt") and os.path.getsize(f + ".txt") > 0)
]
if todo:
    workers = max(1, min(args.workers, len(todo)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_ocr_one, f) for f in todo]
        for fut in as_completed(futs):
            fut.result()

