"""Cloud Run HTTP launcher for render jobs.

This service is intentionally light: it accepts Telegram render requests,
stores the job config in GCS, starts a Cloud Run Job, and returns immediately.
The heavy FFmpeg work happens in render_job.py.
"""

import json
import logging
import os
import uuid

from flask import Flask, jsonify, request
from google.cloud import run_v2

from storage_handler import (
    read_job_status,
    request_job_cancel,
    signed_job_url,
    write_job_config,
    write_job_status,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

PROJECT_ID = os.environ.get("PROJECT_ID", "")
REGION = os.environ.get("REGION", os.environ.get("CLOUD_RUN_REGION", "europe-west1"))
RENDER_JOB_NAME = os.environ.get("RENDER_JOB_NAME", "videobot-render-job")


def project_id() -> str:
    if PROJECT_ID:
        return PROJECT_ID
    creds_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    if not creds_json:
        raise ValueError("PROJECT_ID or GOOGLE_SERVICE_ACCOUNT_JSON is required")
    return json.loads(creds_json).get("project_id", "")


def status_payload(stage: str, progress: int, message: str, **extra) -> dict:
    payload = {"stage": stage, "progress": progress, "message": message}
    payload.update(extra)
    return payload


def start_render_job_execution(job_id: str):
    client = run_v2.JobsClient()
    name = client.job_path(
        project=project_id(),
        location=REGION,
        job=RENDER_JOB_NAME,
    )
    request_obj = run_v2.RunJobRequest(
        name=name,
        overrides=run_v2.RunJobRequest.Overrides(
            container_overrides=[
                run_v2.RunJobRequest.Overrides.ContainerOverride(
                    env=[run_v2.EnvVar(name="JOB_ID", value=job_id)]
                )
            ]
        ),
    )
    logger.info("Starting Cloud Run Job execution: name=%s job_id=%s", name, job_id)
    return client.run_job(request=request_obj)


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
        write_job_config(job_id, payload)
        write_job_status(job_id, status_payload("queued", 0, "Job accepted; starting Cloud Run Job"))
        operation = start_render_job_execution(job_id)
        status_url = signed_job_url(job_id, "status.json", "GET")
        cancel_url = signed_job_url(job_id, "cancel.flag", "PUT")
        operation_name = getattr(operation, "operation", None)
        logger.info("Cloud Run Job execution submitted: job_id=%s operation=%s", job_id, operation_name)
        return jsonify({
            "job_id": job_id,
            "status": "queued",
            "status_url": status_url,
            "cancel_url": cancel_url,
            "operation": str(operation_name or ""),
        }), 200
    except Exception as exc:
        logger.error("Render launcher error: %s", exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500


@app.route("/cancel/<job_id>", methods=["POST"])
def cancel(job_id):
    try:
        request_job_cancel(job_id)
        write_job_status(job_id, status_payload("cancel_requested", 0, "Cancel requested"))
        return jsonify({"job_id": job_id, "status": "cancel_requested"}), 200
    except Exception as exc:
        logger.error("Cancel endpoint error: %s", exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500


@app.route("/status/<job_id>", methods=["GET"])
def status(job_id):
    stored = read_job_status(job_id)
    return jsonify(stored or {"job_id": job_id, "status": "not_found"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "launcher": "cloud-run-jobs",
        "render_job_name": RENDER_JOB_NAME,
        "region": REGION,
    }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
