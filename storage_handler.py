"""Google Cloud Storage output upload and signed download link generation."""

import datetime
import json
import logging
import os

from google.cloud import storage
from google.oauth2 import service_account

logger = logging.getLogger(__name__)


def upload_to_gcs(file_path: str, bucket_name: str, object_name: str) -> str:
    """Upload a private MP4 output and return a time-limited signed URL."""
    creds_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    if not creds_json:
        raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON not set")
    if not bucket_name:
        raise ValueError(
            "OUTPUT_BUCKET_NAME is not configured. Final video upload requires Google Cloud Storage."
        )

    try:
        credentials = service_account.Credentials.from_service_account_info(
            json.loads(creds_json)
        )
        client = storage.Client(project=credentials.project_id, credentials=credentials)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_name)

        logger.info("Starting Cloud Storage upload: bucket=%s object=%s", bucket_name, object_name)
        blob.upload_from_filename(file_path, content_type="video/mp4")
        logger.info("Cloud Storage upload succeeded: gs://%s/%s", bucket_name, object_name)

        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(days=7),
            method="GET",
        )
        logger.info("Signed URL generation succeeded: gs://%s/%s", bucket_name, object_name)
        return signed_url
    except Exception:
        logger.exception("Cloud Storage upload or signed URL generation failed: bucket=%s object=%s", bucket_name, object_name)
        raise
