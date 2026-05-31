FROM python:3.11-slim

# Install ffmpeg and OpenCV dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements_worker.txt .
RUN pip install --no-cache-dir -r requirements_worker.txt

COPY worker.py .
COPY render_job.py .
COPY video_processor.py .
COPY drive_handler.py .
COPY storage_handler.py .

ENV PORT=8080

CMD ["python", "worker.py"]
