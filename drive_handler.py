"""
drive_handler.py — Google Drive integration for video bot
"""

import os
import io
import json
import logging
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


def download_folder_videos(folder_id: str, tmpdir: str) -> list:
    """Download all MP4 files from a Drive folder."""
    service = get_service()
    query = f"'{folder_id}' in parents and (mimeType='video/mp4' or name contains '.mp4') and trashed=false"
    results = service.files().list(q=query, fields="files(id,name,size)", pageSize=50).execute()
    files = results.get("files", [])
    logger.info(f"Found {len(files)} video files")

    downloaded = []
    for f in files:
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
            logger.info(f"Downloaded video: {f['name']}")
        except Exception as e:
            logger.error(f"Failed to download {f['name']}: {e}")

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
