"""
drive_handler.py — Google Drive integration for video bot
"""

import os
import io
import json
import logging
import re
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

logger = logging.getLogger(__name__)
SCOPES = ['https://www.googleapis.com/auth/drive']


def get_service():
    creds_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    if not creds_json:
        raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON not set")
    creds = service_account.Credentials.from_service_account_info(
        json.loads(creds_json), scopes=SCOPES
    )
    return build('drive', 'v3', credentials=creds)


def download_folder_videos(folder_id: str, tmpdir: str, progress_callback=None, cancel_check=None) -> list:
    """Download all MP4 files from a Drive folder."""
    service = get_service()
    query = f"'{folder_id}' in parents and trashed=false"
    files = []
    page_token = None
    while True:
        results = service.files().list(
            q=query,
            fields="nextPageToken, files(id,name,size,mimeType)",
            pageSize=1000,
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        files.extend(results.get("files", []))
        page_token = results.get("nextPageToken")
        if not page_token:
            break

    def natural_key(item: dict):
        return [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", item.get("name", ""))
        ]

    mp4_files = sorted(
        [
            item for item in files
            if item.get("name", "").lower().endswith(".mp4")
            or item.get("mimeType") == "video/mp4"
        ],
        key=natural_key,
    )
    logger.info("Drive folder total files found: %d", len(files))
    logger.info("Drive folder MP4 files found: %d", len(mp4_files))

    downloaded = []
    total = len(mp4_files)
    for index, f in enumerate(mp4_files, start=1):
        if cancel_check and cancel_check():
            raise RuntimeError("Render canceled")
        try:
            out = os.path.join(tmpdir, f["name"])
            req = service.files().get_media(fileId=f["id"])
            fh = io.FileIO(out, "wb")
            dl = MediaIoBaseDownload(fh, req)
            done = False
            while not done:
                _, done = dl.next_chunk()
            fh.close()
            downloaded.append(out)
            logger.info("Downloaded clip %d/%d: %s", index, total, f["name"])
            if progress_callback:
                progress_callback(index, total, f["name"])
        except Exception as e:
            logger.error("Failed to download clip %d/%d %s: %s", index, total, f.get("name"), e)

    return downloaded


def download_audio_file(file_id: str, tmpdir: str) -> str:
    """Download a single audio file from Drive."""
    service = get_service()
    try:
        meta = service.files().get(fileId=file_id, fields="name").execute()
        name = meta.get("name", "audio.mp3")
        out = os.path.join(tmpdir, name)
        req = service.files().get_media(fileId=file_id)
        fh = io.FileIO(out, "wb")
        dl = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            _, done = dl.next_chunk()
        fh.close()
        logger.info(f"Downloaded audio: {name}")
        return out
    except Exception as e:
        logger.error(f"Audio download failed: {e}")
        return None


def upload_file(file_path: str, folder_id: str) -> str:
    """Upload file to Drive folder and return shareable link."""
    service = get_service()
    name = os.path.basename(file_path)
    meta = {"name": name, "parents": [folder_id]}
    media = MediaFileUpload(file_path, mimetype="video/mp4", resumable=True)
    uploaded = service.files().create(body=meta, media_body=media, fields="id").execute()
    fid = uploaded["id"]
    service.permissions().create(fileId=fid, body={"type": "anyone", "role": "reader"}).execute()
    return f"https://drive.google.com/file/d/{fid}/view"
