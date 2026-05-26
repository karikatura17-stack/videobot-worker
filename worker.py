"""Cloud Run HTTP entry point for rendering montage jobs."""

import logging
import os
import re
import shutil
import tempfile
import threading
import uuid

import requests
from flask import Flask, jsonify, request

from drive_handler import download_audio_file, download_folder_videos, upload_file
from video_processor import analyze_clip, build_video

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
jobs = {}


def extract_drive_id(url: str, is_folder: bool) -> str:
    patterns = (
        [r"/folders/([a-zA-Z0-9_-]+)", r"id=([a-zA-Z0-9_-]+)"]
        if is_folder
        else [r"/file/d/([a-zA-Z0-9_-]+)", r"id=([a-zA-Z0-9_-]+)"]
    )
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return ""


def notify_telegram(bot_token: str, user_id: int, text: str, parse_mode: str | None = "Markdown"):
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        message = {"chat_id": user_id, "text": text}
        if parse_mode:
            message["parse_mode"] = parse_mode
        requests.post(
            url,
            json=message,
            timeout=10,
        )
    except Exception as exc:
        logger.error("Telegram notify failed: %s", exc)


def run_render_job(job_id: str, payload: dict):
    user_id = payload["user_id"]
    bot_token = payload["bot_token"]
    tmpdir = tempfile.mkdtemp()
    try:
        montage_config = payload.get("montage_config")
        visualizer_config = payload.get("visualizer_config")
        effects_config = payload.get("effects_config")
        # Keep compatibility while a previously deployed bot is still sending old fields.
        if visualizer_config is None and "visualizer" in payload:
            visualizer_config = {"enabled": bool(payload["visualizer"])}
        if effects_config is None:
            effects_config = payload.get("overrides")

        logger.info("Job %s style=%s montage_config=%s", job_id, payload["style"], montage_config)
        logger.info("Job %s visualizer_config=%s effects_config=%s", job_id, visualizer_config, effects_config)

        jobs[job_id] = "downloading"
        notify_telegram(bot_token, user_id, "Downloading videos...")
        video_folder_id = extract_drive_id(payload["video_link"], is_folder=True)
        audio_file_id = extract_drive_id(payload["audio_link"], is_folder=False)
        if not video_folder_id or not audio_file_id:
            raise ValueError("Не могу распознать ссылки Drive")

        video_clips = download_folder_videos(video_folder_id, tmpdir)
        logger.info("Number of clips downloaded: %d", len(video_clips))
        if not video_clips:
            raise ValueError("В папке нет MP4 файлов")
        notify_telegram(bot_token, user_id, f"Downloaded {len(video_clips)} clips.")

        audio_path = download_audio_file(audio_file_id, tmpdir)
        if not audio_path:
            raise ValueError("Не удалось скачать аудиофайл")
        notify_telegram(bot_token, user_id, "Downloaded audio.")

        jobs[job_id] = "analyzing"
        clips = []
        for index, video_path in enumerate(video_clips, start=1):
            clip_data = analyze_clip(video_path)
            if clip_data:
                clips.append(clip_data)
            else:
                logger.warning("Rejected unreadable clip: %s", video_path)
            notify_telegram(bot_token, user_id, f"Analyzed {index}/{len(video_clips)} clips.")
        logger.info("Number of clips analyzed: %d", len(clips))
        if not clips:
            raise ValueError("Не удалось проанализировать видеоклипы")

        jobs[job_id] = "rendering"
        output_path = os.path.join(tmpdir, "FINAL_VIDEO.mp4")

        result = build_video(
            clips=clips,
            audio_path=audio_path,
            style=payload["style"],
            user_overrides=payload.get("overrides"),
            tmpdir=tmpdir,
            output_path=output_path,
            progress_callback=lambda text: notify_telegram(bot_token, user_id, text),
            montage_config=montage_config,
            visualizer_config=visualizer_config,
            effects_config=effects_config,
        )

        jobs[job_id] = "uploading"
        notify_telegram(bot_token, user_id, "Uploading to Drive...")
        drive_link = upload_file(output_path, video_folder_id)
        jobs[job_id] = "done"
        notify_telegram(
            bot_token,
            user_id,
            "*Done!*\n\n"
            f"Duration: {result['duration']}\n"
            f"BPM: {result['bpm']}\n"
            f"Segments: {result['clips_used']}\n"
            f"Size: {result['file_size_gb']} GB\n\n"
            f"[Download video]({drive_link})\n\n"
            "Send /start for another render.",
        )
    except Exception as exc:
        logger.error("Job %s failed: %s", job_id, exc, exc_info=True)
        jobs[job_id] = "failed"
        notify_telegram(
            bot_token,
            user_id,
            f"{str(exc)[:1000]}\n\nОтправьте /start, чтобы попробовать снова.",
            parse_mode=None,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.route("/render", methods=["POST"])
def render():
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
        threading.Thread(target=run_render_job, args=(job_id, payload), daemon=True).start()
        return jsonify({"job_id": job_id, "status": "queued"}), 200
    except Exception as exc:
        logger.error("Render endpoint error: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/status/<job_id>", methods=["GET"])
def status(job_id):
    return jsonify({"job_id": job_id, "status": jobs.get(job_id, "not_found")})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "jobs": len(jobs)}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
