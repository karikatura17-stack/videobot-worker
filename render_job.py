"""Cloud Run Job entry point: performs one full montage render."""

import logging
import os
import re
import shutil
import tempfile

import requests

from drive_handler import download_audio_file, download_folder_videos
from storage_handler import (
    is_job_cancel_requested,
    read_job_config,
    upload_to_gcs,
    write_job_status,
)
from video_processor import RenderCanceled, analyze_clip, build_video

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def status_payload(stage: str, progress: int, message: str, **extra) -> dict:
    payload = {"stage": stage, "progress": progress, "message": message}
    payload.update(extra)
    return payload


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
        requests.post(url, json=message, timeout=10)
    except Exception as exc:
        logger.error("Telegram notify failed: %s", exc)


def run_render_job(job_id: str, payload: dict):
    user_id = payload["user_id"]
    bot_token = payload["bot_token"]
    tmpdir = tempfile.mkdtemp()
    bucket_name = os.environ.get("OUTPUT_BUCKET_NAME", "")

    def update_status(stage: str, progress: int, message: str, **extra):
        try:
            write_job_status(job_id, status_payload(stage, progress, message, **extra), bucket_name)
        except Exception as exc:
            logger.error("Job %s status write failed: %s", job_id, exc)

    def cancel_requested() -> bool:
        try:
            return is_job_cancel_requested(job_id, bucket_name)
        except Exception as exc:
            logger.error("Job %s cancel check failed: %s", job_id, exc)
            return False

    try:
        if not bucket_name:
            raise ValueError("OUTPUT_BUCKET_NAME is not configured. Final video upload requires Google Cloud Storage.")
        object_name = f"renders/{job_id}/FINAL_VIDEO.mp4"
        update_status("queued", 0, "Cloud Run Job started", output_object=object_name)
        logger.info("Selected output bucket: %s", bucket_name)
        logger.info("Selected output object path: %s", object_name)

        montage_config = payload.get("montage_config")
        visualizer_config = payload.get("visualizer_config")
        effects_config = payload.get("effects_config")
        effects_intensity = payload.get("effects_intensity")
        if visualizer_config is None and "visualizer" in payload:
            visualizer_config = {"enabled": bool(payload["visualizer"])}
        if effects_config is None:
            effects_config = payload.get("overrides")
        if effects_intensity is None and isinstance(effects_config, dict):
            effects_intensity = effects_config.get("intensity")

        logger.info("Job %s style=%s montage_config=%s", job_id, payload["style"], montage_config)
        logger.info("Job %s visualizer_config=%s effects_config=%s effects_intensity=%s",
                    job_id, visualizer_config, effects_config, effects_intensity)

        update_status("downloading", 5, "Starting Drive download")
        video_folder_id = extract_drive_id(payload["video_link"], is_folder=True)
        audio_file_id = extract_drive_id(payload["audio_link"], is_folder=False)
        if not video_folder_id or not audio_file_id:
            raise ValueError("Cannot recognize Drive links")

        video_clips = download_folder_videos(
            video_folder_id,
            tmpdir,
            progress_callback=lambda current, total, name: update_status(
                "downloading", 5 + int(15 * current / max(1, total)),
                f"Downloaded clip {current}/{total}",
                clips_found=total,
                clips_downloaded=current,
                current_file=name,
            ),
            cancel_check=cancel_requested,
        )
        logger.info("Number of clips downloaded: %d", len(video_clips))
        if not video_clips:
            raise ValueError("Drive folder contains no MP4 files")

        audio_path = download_audio_file(audio_file_id, tmpdir)
        if not audio_path:
            raise ValueError("Could not download audio file")
        update_status(
            "downloading", 20,
            f"Files downloaded: {len(video_clips)} clips and audio.",
            clips_found=len(video_clips),
            clips_downloaded=len(video_clips),
        )
        notify_telegram(bot_token, user_id, f"10% Files downloaded: {len(video_clips)} clips and audio.")

        update_status("analyzing", 22, "Analyzing clips", clips_found=len(video_clips))
        clips = []
        for index, video_path in enumerate(video_clips, start=1):
            if cancel_requested():
                raise RenderCanceled("Render canceled")
            clip_data = analyze_clip(video_path)
            if clip_data:
                clips.append(clip_data)
            else:
                logger.warning("Rejected unreadable clip: %s", video_path)
            if index == len(video_clips) or index % 5 == 0:
                update_status(
                    "analyzing", 22 + int(8 * index / max(1, len(video_clips))),
                    f"Analyzed clip {index}/{len(video_clips)}",
                    clips_found=len(video_clips),
                    clips_analyzed=index,
                    clips_accepted=len(clips),
                )
        logger.info("Number of clips analyzed: %d", len(clips))
        if not clips:
            raise ValueError("Could not analyze any video clips")
        rejected_clips = len(video_clips) - len(clips)

        update_status("preparing_segments", 30, "Building montage and preparing segments")
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
            effects_intensity=effects_intensity,
            cancel_check=cancel_requested,
            status_callback=update_status,
        )

        update_status("uploading", 95, "Uploading final video", output_object=object_name)
        download_link = upload_to_gcs(output_path, bucket_name, object_name)
        update_status(
            "done", 100, "Completed",
            output_object=object_name,
            download_link=download_link,
            duration=result["duration"],
            bpm=result["bpm"],
            clips_used=result["clips_used"],
            unique_clips_used=result["unique_clips_used"],
            source_clips_available=result["source_clips_available"],
            segments_rendered=result["segments_rendered"],
            repeated_segments=result["repeated_segments"],
            clips_rejected=rejected_clips,
            output_resolution=result["output_resolution"],
            output_fps=result["output_fps"],
            duration_fallbacks=result["duration_fallbacks"],
        )
        notify_telegram(
            bot_token,
            user_id,
            "100% Uploaded to Cloud Storage.\n\n"
            "VIDEO READY!\n\n"
            f"Duration: {result['duration']}\n"
            f"BPM: {result['bpm']}\n"
            f"Unique source clips used: {result['unique_clips_used']}/{result['source_clips_available']}\n"
            f"Montage segments rendered: {result['segments_rendered']}\n"
            f"Repeated segments: {result['repeated_segments']}\n"
            f"Rejected clips: {rejected_clips}\n"
            f"Output: {result['output_resolution']} at {result['output_fps']} fps\n"
            f"File size: {result['file_size_gb']} GB\n\n"
            f"Download link:\n{download_link}\n\n"
            "Send /start for a new video.",
            parse_mode=None,
        )
    except RenderCanceled:
        logger.warning("Job %s canceled", job_id, exc_info=True)
        update_status("canceled", 0, "Render canceled")
        notify_telegram(bot_token, user_id, "Render canceled. Send /start to create a new montage.", parse_mode=None)
    except Exception as exc:
        logger.error("Job %s failed: %s", job_id, exc, exc_info=True)
        update_status("failed", 0, str(exc)[:1000], error=str(exc)[:1000])
        notify_telegram(bot_token, user_id, f"{str(exc)[:1000]}\n\nSend /start to try again.", parse_mode=None)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main() -> int:
    job_id = os.environ.get("JOB_ID", "")
    if not job_id:
        logger.error("JOB_ID is not set")
        return 2
    payload = read_job_config(job_id)
    run_render_job(job_id, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
