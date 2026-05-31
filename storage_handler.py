"""Google Cloud Storage output upload and signed download link generation."""

import datetime
import json
import logging
import os

from google.cloud import storage
from google.oauth2 import service_account

logger = logging.getLogger(__name__)


def storage_context(bucket_name: str | None = None):
    creds_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    if not creds_json:
        raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON not set")
    bucket_name = bucket_name or os.environ.get("OUTPUT_BUCKET_NAME", "")
    if not bucket_name:
        raise ValueError(
            "OUTPUT_BUCKET_NAME is not configured. Final video upload requires Google Cloud Storage."
        )
    credentials = service_account.Credentials.from_service_account_info(
        json.loads(creds_json)
    )
    client = storage.Client(project=credentials.project_id, credentials=credentials)
    return client.bucket(bucket_name)


def upload_to_gcs(file_path: str, bucket_name: str, object_name: str) -> str:
    """Upload a private MP4 output and return a time-limited signed URL."""
    try:
        bucket = storage_context(bucket_name)
        blob = bucket.blob(object_name)
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

        logger.info(
            "Starting Cloud Storage upload: bucket=%s object=%s size_bytes=%d",
            bucket.name, object_name, file_size,
        )
        blob.upload_from_filename(file_path, content_type="video/mp4")
        logger.info("Cloud Storage upload completed: gs://%s/%s", bucket.name, object_name)

        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(days=7),
            method="GET",
        )
        logger.info("Signed URL created: gs://%s/%s", bucket.name, object_name)
        return signed_url
    except Exception:
        logger.exception("Cloud Storage upload or signed URL generation failed: bucket=%s object=%s", bucket_name, object_name)
        raise


def write_job_status(job_id: str, status: dict, bucket_name: str | None = None) -> None:
    bucket = storage_context(bucket_name)
    blob = bucket.blob(f"jobs/{job_id}/status.json")
    payload = {
        "job_id": job_id,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **status,
    }
    blob.upload_from_string(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        content_type="application/json",
    )
    logger.info("Job status updated: gs://%s/jobs/%s/status.json stage=%s progress=%s",
                bucket.name, job_id, payload.get("stage"), payload.get("progress"))


def read_job_status(job_id: str, bucket_name: str | None = None) -> dict | None:
    bucket = storage_context(bucket_name)
    blob = bucket.blob(f"jobs/{job_id}/status.json")
    if not blob.exists():
        return None
    return json.loads(blob.download_as_text())


def write_job_config(job_id: str, payload: dict, bucket_name: str | None = None) -> None:
    bucket = storage_context(bucket_name)
    blob = bucket.blob(f"jobs/{job_id}/config.json")
    blob.upload_from_string(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        content_type="application/json",
    )
    logger.info("Job config written: gs://%s/jobs/%s/config.json", bucket.name, job_id)


def read_job_config(job_id: str, bucket_name: str | None = None) -> dict:
    bucket = storage_context(bucket_name)
    blob = bucket.blob(f"jobs/{job_id}/config.json")
    if not blob.exists():
        raise FileNotFoundError(f"Job config not found: jobs/{job_id}/config.json")
    return json.loads(blob.download_as_text())


def request_job_cancel(job_id: str, bucket_name: str | None = None) -> None:
    bucket = storage_context(bucket_name)
    blob = bucket.blob(f"jobs/{job_id}/cancel.flag")
    blob.upload_from_string("cancel_requested", content_type="text/plain")
    logger.info("Cancel flag written: gs://%s/jobs/%s/cancel.flag", bucket.name, job_id)


def is_job_cancel_requested(job_id: str, bucket_name: str | None = None) -> bool:
    bucket = storage_context(bucket_name)
    return bucket.blob(f"jobs/{job_id}/cancel.flag").exists()


def signed_job_url(job_id: str, name: str, method: str = "GET", bucket_name: str | None = None) -> str:
    bucket = storage_context(bucket_name)
    blob = bucket.blob(f"jobs/{job_id}/{name}")
    kwargs = {
        "version": "v4",
        "expiration": datetime.timedelta(days=7),
        "method": method,
    }
    if method in {"PUT", "POST"}:
        kwargs["content_type"] = "text/plain"
    return blob.generate_signed_url(**kwargs)
