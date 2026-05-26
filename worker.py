"""
worker.py — Cloud Run Flask app
Receives render jobs, processes video, uploads to Drive, notifies Telegram
"""

import os
import json
import uuid
import logging
import tempfile
import shutil
import threading
import requests
from flask import Flask, request, jsonify

from drive_handler import download_folder_videos, download_audio_file, upload_file
from video_processor import analyze_clip, build_video

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
jobs = {}  # job_id -> status


def extract_drive_id(url: str, is_folder: bool) -> str:
    import re
    if is_folder:
        patterns = [r'/folders/([a-zA-Z0-9_-]+)', r'id=([a-zA-Z0-9_-]+)']
    else:
        patterns = [r'/file/d/([a-zA-Z0-9_-]+)', r'id=([a-zA-Z0-9_-]+)']
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return ""


def notify_telegram(bot_token: str, user_id: int, text: str):
    """Send message to user via Telegram Bot API."""
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        requests.post(url, json={
            "chat_id": user_id,
            "text": text,
            "parse_mode": "Markdown"
        }, timeout=10)
    except Exception as e:
        logger.error(f"Telegram notify failed: {e}")


def run_render_job(job_id: str, payload: dict):
    """Run in background thread."""
    user_id = payload["user_id"]
    bot_token = payload["bot_token"]
    tmpdir = tempfile.mkdtemp()

    try:
        jobs[job_id] = "downloading"
        notify_telegram(bot_token, user_id,
            f"⏳ *Задание {job_id[:8]}...*\n\n📥 Скачиваю файлы из Drive...")

        # Extract IDs
        video_folder_id = extract_drive_id(payload["video_link"], is_folder=True)
        audio_file_id = extract_drive_id(payload["audio_link"], is_folder=False)

        if not video_folder_id or not audio_file_id:
            raise ValueError("Не могу распознать ссылки Drive")

        # Download videos
        video_clips = download_folder_videos(video_folder_id, tmpdir)
        if not video_clips:
            raise ValueError("В папке нет MP4 файлов")

        notify_telegram(bot_token, user_id,
            f"📥 Скачано {len(video_clips)} клипов. Скачиваю аудио...")

        # Download audio
        audio_path = download_audio_file(audio_file_id, tmpdir)
        if not audio_path:
            raise ValueError("Не удалось скачать аудиофайл")

        # Analyze clips
        jobs[job_id] = "analyzing"
        notify_telegram(bot_token, user_id,
            f"🔍 Анализирую {len(video_clips)} клипов...")

        clips = []
        for vp in video_clips:
            clip_data = analyze_clip(vp)
            if clip_data:
                clips.append(clip_data)

        notify_telegram(bot_token, user_id,
            f"✅ Проанализировано {len(clips)} клипов.\n🎬 Начинаю монтаж...")

        # Build video
        jobs[job_id] = "rendering"
        output_path = os.path.join(tmpdir, "FINAL_VIDEO.mp4")

        def progress(text):
            notify_telegram(bot_token, user_id, text)

        result = build_video(
            clips=clips,
            audio_path=audio_path,
            style=payload["style"],
            user_overrides=payload.get("overrides", {}),
            tmpdir=tmpdir,
            output_path=output_path,
            progress_callback=progress
        )

        # Upload to Drive
        jobs[job_id] = "uploading"
        notify_telegram(bot_token, user_id, "📤 Загружаю готовое видео в Drive...")

        drive_link = upload_file(output_path, video_folder_id)

        # Done!
        jobs[job_id] = "done"
        notify_telegram(bot_token, user_id,
            f"🎉 *ВИДЕО ГОТОВО!*\n\n"
            f"⏱ Длина: {result['duration']}\n"
            f"🥁 BPM: {result['bpm']}\n"
            f"🎬 Клипов: {result['clips_used']}\n"
            f"💾 Размер: {result['file_size_gb']} GB\n\n"
            f"🔗 [Скачать видео]({drive_link})\n\n"
            f"Напиши /start для нового видео."
        )

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        jobs[job_id] = "failed"
        notify_telegram(bot_token, user_id,
            f"❌ Ошибка рендера:\n{str(e)[:300]}\n\nНапиши /start чтобы попробовать снова."
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.route("/render", methods=["POST"])
def render():
    """Accept render job and start processing in background."""
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({"error": "No payload"}), 400

        required = ["user_id", "style", "video_link", "audio_link", "bot_token"]
        for field in required:
            if field not in payload:
                return jsonify({"error": f"Missing field: {field}"}), 400

        job_id = str(uuid.uuid4())
        jobs[job_id] = "queued"

        thread = threading.Thread(
            target=run_render_job,
            args=(job_id, payload),
            daemon=True
        )
        thread.start()

        return jsonify({"job_id": job_id, "status": "queued"}), 200

    except Exception as e:
        logger.error(f"Render endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/status/<job_id>", methods=["GET"])
def status(job_id):
    """Check job status."""
    return jsonify({"job_id": job_id, "status": jobs.get(job_id, "not_found")})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "jobs": len(jobs)}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
