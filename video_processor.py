"""
video_processor.py — Core video processing engine
Handles: clip analysis, smart assembly, effects, visualizer, final render
"""

import os
import json
import random
import subprocess
import tempfile
import shutil
import logging
import numpy as np
import cv2
import librosa

logger = logging.getLogger(__name__)

# ─── STYLE PRESETS ────────────────────────────────────────────────────────────

STYLE_PRESETS = {
    "phonk": {
        "visualizer_type": "bars",           # bars / waveform / circular
        "visualizer_position": "bottom",      # bottom / top / center_bottom
        "visualizer_color": "#cc44ff",        # purple
        "visualizer_height": 80,              # px height
        "color_filter": "dark_purple",        # color grading preset
        "effects": {
            "film_grain": True,
            "vignette": True,
            "chromatic_aberration": False,
            "rain": False,
            "bloom": False,
            "sparkles": False,
            "vhs": True,
            "flash": True,
        },
        "transition_style": "cut",           # cut / crossfade / glitch
        "contrast_boost": 1.3,
        "saturation": 0.7,
        "brightness": 0.85,
    },
    "japanese": {
        "visualizer_type": "waveform",
        "visualizer_position": "top",
        "visualizer_color": "#ff2244",
        "visualizer_height": 60,
        "color_filter": "neon_blue",
        "effects": {
            "film_grain": False,
            "vignette": True,
            "chromatic_aberration": True,
            "rain": True,
            "bloom": True,
            "sparkles": False,
            "vhs": False,
            "flash": False,
        },
        "transition_style": "crossfade",
        "contrast_boost": 1.4,
        "saturation": 1.2,
        "brightness": 0.9,
    },
    "house": {
        "visualizer_type": "circular",
        "visualizer_position": "center_bottom",
        "visualizer_color": "#ffaa44",
        "visualizer_height": 100,
        "color_filter": "warm_gold",
        "effects": {
            "film_grain": False,
            "vignette": False,
            "chromatic_aberration": False,
            "rain": False,
            "bloom": True,
            "sparkles": True,
            "vhs": False,
            "flash": False,
        },
        "transition_style": "crossfade",
        "contrast_boost": 1.1,
        "saturation": 1.3,
        "brightness": 1.05,
    },
}

COLOR_FILTERS = {
    "dark_purple": "curves=r='0/0 128/80 255/200':g='0/0 128/60 255/150':b='0/0 128/120 255/220'",
    "neon_blue":   "curves=r='0/0 128/60 255/160':g='0/0 128/80 255/180':b='0/0 128/140 255/255'",
    "warm_gold":   "curves=r='0/0 128/140 255/255':g='0/0 128/120 255/220':b='0/0 128/60 255/150'",
}


# ─── CLIP ANALYSIS ────────────────────────────────────────────────────────────

def analyze_clip(clip_path: str) -> dict:
    """Analyze a video clip for quality metrics."""
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Sample frames for analysis
    sample_count = min(10, total_frames)
    frames = []
    for i in range(sample_count):
        frame_idx = int(i * total_frames / sample_count)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()

    if not frames:
        return None

    # Brightness (average luminance)
    brightnesses = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean() for f in frames]
    avg_brightness = np.mean(brightnesses)

    # Blur detection (Laplacian variance — higher = sharper)
    blurs = [cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() for f in frames]
    avg_sharpness = np.mean(blurs)

    # Motion estimation (frame difference)
    motions = []
    for i in range(1, len(frames)):
        diff = cv2.absdiff(frames[i-1], frames[i])
        motions.append(diff.mean())
    avg_motion = np.mean(motions) if motions else 0

    # Dominant color (for seamless matching)
    avg_color = np.mean([f.mean(axis=(0,1)) for f in frames], axis=0)  # BGR

    # First and last frame colors (for transition matching)
    first_color = frames[0].mean(axis=(0,1)) if frames else avg_color
    last_color = frames[-1].mean(axis=(0,1)) if frames else avg_color

    # Quality score
    quality = min(100, int(
        (min(avg_brightness, 200) / 200 * 30) +  # brightness 30pts
        (min(avg_sharpness, 500) / 500 * 50) +   # sharpness 50pts
        (min(avg_motion, 10) / 10 * 20)           # motion 20pts
    ))

    return {
        "path": clip_path,
        "duration": round(duration, 2),
        "fps": fps,
        "width": width,
        "height": height,
        "brightness": round(avg_brightness, 1),
        "sharpness": round(avg_sharpness, 1),
        "motion": round(avg_motion, 2),
        "quality": quality,
        "avg_color": avg_color.tolist(),
        "first_color": first_color.tolist(),
        "last_color": last_color.tolist(),
    }


def color_distance(c1: list, c2: list) -> float:
    """Euclidean distance between two BGR colors."""
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))


def find_best_next_clip(current_clip: dict, candidates: list, used: set) -> dict:
    """Find clip whose first frame best matches current clip's last frame."""
    best = None
    best_score = float('inf')

    for clip in candidates:
        if clip["path"] in used:
            continue
        dist = color_distance(current_clip["last_color"], clip["first_color"])
        if dist < best_score:
            best_score = dist
            best = clip

    return best


# ─── AUDIO BEAT DETECTION ────────────────────────────────────────────────────

def detect_beats(audio_path: str) -> tuple:
    """Detect beats and return beat times and BPM."""
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)
    return float(tempo), beat_times.tolist(), duration


# ─── CLIP VARIANTS ────────────────────────────────────────────────────────────

def prepare_clip_variant(clip: dict, tmpdir: str, variant: str, target_duration: float) -> str:
    """Create a variant of the clip: normal / mirror / reverse / mirror_reverse."""
    src = clip["path"]
    out = os.path.join(tmpdir, f"variant_{variant}_{os.path.basename(src)}")

    filters = []
    if "mirror" in variant:
        filters.append("hflip")
    if "reverse" in variant:
        filters.append("reverse")

    # Trim to target duration if longer
    trim = f"trim=duration={target_duration},setpts=PTS-STARTPTS"
    filters = [trim] + filters

    filter_str = ",".join(filters)
    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-vf", filter_str,
        "-an",  # no audio
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        out
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        logger.warning(f"Variant failed: {result.stderr.decode()[:200]}")
        return src
    return out


# ─── EFFECTS FILTERS ─────────────────────────────────────────────────────────

def build_effects_filter(preset: dict, user_overrides: dict, width: int, height: int) -> str:
    """Build ffmpeg video filter chain for effects."""
    effects = {**preset["effects"], **user_overrides}
    filters = []

    # Color grading
    color_filter_name = preset.get("color_filter", "dark_purple")
    if color_filter_name in COLOR_FILTERS:
        filters.append(COLOR_FILTERS[color_filter_name])

    # Contrast / saturation / brightness
    contrast = preset.get("contrast_boost", 1.0) + random.uniform(-0.05, 0.05)
    saturation = preset.get("saturation", 1.0) + random.uniform(-0.05, 0.05)
    brightness = preset.get("brightness", 1.0) + random.uniform(-0.03, 0.03)
    filters.append(f"eq=contrast={contrast:.2f}:saturation={saturation:.2f}:brightness={brightness:.2f}")

    # Film grain
    if effects.get("film_grain"):
        strength = random.uniform(8, 18)
        filters.append(f"noise=alls={strength:.0f}:allf=t+u")

    # Vignette
    if effects.get("vignette"):
        angle = random.uniform(0.1, 0.3)
        filters.append(f"vignette=angle={angle:.2f}")

    # Chromatic aberration (via rgbashift)
    if effects.get("chromatic_aberration"):
        shift = random.randint(2, 5)
        filters.append(f"rgbashift=rh={shift}:bh=-{shift}")

    # VHS effect (unsharp + slight blur)
    if effects.get("vhs"):
        filters.append("unsharp=5:5:0.8:3:3:0.4")

    # Bloom (glow via blur blend)
    if effects.get("bloom"):
        filters.append("gblur=sigma=2")

    return ",".join(filters) if filters else "null"


# ─── VISUALIZER GENERATION ───────────────────────────────────────────────────

def generate_visualizer(audio_path: str, output_path: str, preset: dict,
                        width: int, height: int, duration: float):
    """Generate beat-reactive visualizer video using ffmpeg."""
    vis_type = preset["visualizer_type"]
    vis_pos = preset["visualizer_position"]
    vis_color = preset["visualizer_color"].lstrip("#")
    vis_h = preset["visualizer_height"]

    # Convert hex color to ffmpeg format
    r = int(vis_color[0:2], 16)
    g = int(vis_color[2:4], 16)
    b = int(vis_color[4:6], 16)

    # Position overlay
    if vis_pos == "bottom":
        y_pos = height - vis_h - 10
    elif vis_pos == "top":
        y_pos = 10
    else:  # center_bottom
        y_pos = height - vis_h - 40

    if vis_type == "bars":
        # Showfreqs as bars
        vis_filter = (
            f"showfreqs=s={width}x{vis_h}:win_size=2048:ascale=log:"
            f"fscale=log:colors=0x{vis_color}|0x{vis_color}:mode=bar"
        )
    elif vis_type == "waveform":
        vis_filter = (
            f"showwaves=s={width}x{vis_h}:mode=cline:"
            f"colors=0x{vis_color}:scale=sqrt"
        )
    else:  # circular
        vis_w = min(width, height) - 100
        vis_filter = (
            f"showcqt=s={vis_w}x{vis_w}:count=1:csp=bt709:"
            f"bar_g=2:sono_g=4:bar_v=9:sono_v=17:tc=0.33"
        )

    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-filter_complex",
        f"[0:a]{vis_filter}[vis]",
        "-map", "[vis]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-t", str(duration),
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    if result.returncode != 0:
        logger.error(f"Visualizer failed: {result.stderr.decode()[-500:]}")
        return False
    return True


# ─── MAIN BUILD FUNCTION ─────────────────────────────────────────────────────

def build_video(
    clips: list,          # list of analyzed clip dicts
    audio_path: str,
    style: str,
    user_overrides: dict,  # effects user turned off
    tmpdir: str,
    output_path: str,
    progress_callback=None
) -> dict:
    """
    Main pipeline:
    1. Sort clips by quality
    2. Build seamless sequence using color matching
    3. Apply mirror/reverse variants for uniqueness
    4. Loop sequence to match audio duration
    5. Apply effects
    6. Overlay visualizer
    7. Merge with audio
    8. Final render
    """

    preset = STYLE_PRESETS.get(style, STYLE_PRESETS["phonk"])

    if progress_callback:
        progress_callback("🔍 Анализирую аудио...")

    # Detect beats and duration
    bpm, beat_times, audio_duration = detect_beats(audio_path)
    logger.info(f"BPM: {bpm:.1f}, Duration: {audio_duration:.1f}s, Beats: {len(beat_times)}")

    if progress_callback:
        progress_callback(f"🎵 BPM: {bpm:.0f}, длина: {audio_duration/60:.1f} мин")

    # Sort clips by quality
    clips = [c for c in clips if c and c.get("quality", 0) > 20]
    clips.sort(key=lambda x: x["quality"], reverse=True)

    if not clips:
        raise ValueError("Нет подходящих видеоклипов")

    # Width/height from first clip
    w = clips[0]["width"]
    h = clips[0]["height"]

    if progress_callback:
        progress_callback(f"🎬 Строю последовательность из {len(clips)} клипов...")

    # Build seamless sequence
    sequence = []
    used = set()
    current = random.choice(clips[:3])  # start with one of top 3
    sequence.append(current)
    used.add(current["path"])

    while True:
        total_duration = sum(c["duration"] for c in sequence)
        if total_duration >= audio_duration:
            break

        next_clip = find_best_next_clip(current, clips, used)
        if next_clip is None:
            # All used — reset and add variants
            used.clear()
            used.add(current["path"])
            next_clip = find_best_next_clip(current, clips, used)
            if next_clip is None:
                next_clip = random.choice(clips)

        # Randomly apply mirror/reverse for uniqueness
        variant = random.choice(["normal", "mirror", "reverse", "mirror_reverse"])
        next_clip = {**next_clip, "_variant": variant}
        sequence.append(next_clip)
        used.add(next_clip["path"])
        current = next_clip

    if progress_callback:
        progress_callback(f"✂️ Подготавливаю {len(sequence)} клипов с вариантами...")

    # Prepare clip variants
    prepared_clips = []
    for i, clip in enumerate(sequence):
        variant = clip.get("_variant", "normal")
        prepared = prepare_clip_variant(clip, tmpdir, variant, clip["duration"])
        prepared_clips.append(prepared)

    if progress_callback:
        progress_callback("🎨 Применяю эффекты и цветокоррекцию...")

    # Build effects filter
    effects_filter = build_effects_filter(preset, user_overrides, w, h)

    # Concat all clips
    concat_list = os.path.join(tmpdir, "concat.txt")
    with open(concat_list, "w") as f:
        for cp in prepared_clips:
            f.write(f"file '{cp}'\n")

    raw_video = os.path.join(tmpdir, "raw_video.mp4")

    # Transition style
    if preset["transition_style"] == "crossfade":
        transition_filter = build_crossfade_concat(prepared_clips, tmpdir, audio_duration)
        concat_cmd = transition_filter
        use_filter_complex = True
    else:
        # Simple concat with effects
        concat_cmd = None
        use_filter_complex = False

    if use_filter_complex and concat_cmd:
        cmd = [
            "ffmpeg", "-y",
            *concat_cmd["inputs"],
            "-filter_complex", concat_cmd["filter"],
            "-map", concat_cmd["output"],
            "-vf", effects_filter,
            "-t", str(audio_duration),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            raw_video
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", concat_list,
            "-vf", effects_filter,
            "-t", str(audio_duration),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            raw_video
        ]

    result = subprocess.run(cmd, capture_output=True, timeout=3600)
    if result.returncode != 0:
        logger.error(f"Concat failed: {result.stderr.decode()[-500:]}")
        raise RuntimeError("Ошибка сборки видео")

    if progress_callback:
        progress_callback("📊 Генерирую визуализатор...")

    # Generate visualizer
    vis_video = os.path.join(tmpdir, "visualizer.mp4")
    vis_ok = generate_visualizer(audio_path, vis_video, preset, w, h, audio_duration)

    if progress_callback:
        progress_callback("🎬 Финальный рендер...")

    # Merge: video + visualizer overlay + audio
    vis_pos = preset["visualizer_position"]
    vis_h = preset["visualizer_height"]
    vis_color = preset["visualizer_color"].lstrip("#")

    if vis_pos == "bottom":
        overlay_x, overlay_y = 0, f"H-h-10"
    elif vis_pos == "top":
        overlay_x, overlay_y = 0, 10
    else:
        overlay_x, overlay_y = f"(W-w)/2", f"H-h-40"

    if vis_ok and os.path.exists(vis_video):
        # Overlay visualizer with transparency blend
        final_cmd = [
            "ffmpeg", "-y",
            "-i", raw_video,
            "-i", vis_video,
            "-i", audio_path,
            "-filter_complex",
            f"[1:v]scale={w}:{vis_h},format=rgba,colorchannelmixer=aa=0.75[vis];"
            f"[0:v][vis]overlay={overlay_x}:{overlay_y}[out]",
            "-map", "[out]",
            "-map", "2:a",
            "-c:v", "libx264", "-preset", "medium", "-crf", "17",
            "-c:a", "aac", "-b:a", "320k",
            "-t", str(audio_duration),
            output_path
        ]
    else:
        # No visualizer — just merge video + audio
        final_cmd = [
            "ffmpeg", "-y",
            "-i", raw_video,
            "-i", audio_path,
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "libx264", "-preset", "medium", "-crf", "17",
            "-c:a", "aac", "-b:a", "320k",
            "-t", str(audio_duration),
            output_path
        ]

    result = subprocess.run(final_cmd, capture_output=True, timeout=3600)
    if result.returncode != 0:
        logger.error(f"Final render failed: {result.stderr.decode()[-500:]}")
        raise RuntimeError("Ошибка финального рендера")

    # Get output file size
    file_size = os.path.getsize(output_path) / (1024 * 1024 * 1024)
    m, s = divmod(int(audio_duration), 60)

    return {
        "duration": f"{m}:{s:02d}",
        "bpm": round(bpm),
        "clips_used": len(sequence),
        "file_size_gb": round(file_size, 2),
        "output": output_path,
    }


def build_crossfade_concat(clips: list, tmpdir: str, max_duration: float) -> dict:
    """Build ffmpeg filter complex for crossfade transitions."""
    inputs = []
    for c in clips:
        inputs += ["-i", c]

    n = len(clips)
    if n == 1:
        return None

    parts = []
    current = "[0:v]"
    offset = 0

    for i in range(1, min(n, 20)):  # limit to 20 clips for filter complexity
        next_in = f"[{i}:v]"
        out = f"[cf{i}]" if i < n - 1 else "[cfout]"

        # Get duration of current source clip
        try:
            cap = cv2.VideoCapture(clips[i-1])
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            fc = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            dur = fc / fps
            cap.release()
        except:
            dur = 5.0

        offset += max(0, dur - 0.5)
        parts.append(f"{current}{next_in}xfade=transition=fade:duration=0.5:offset={offset:.2f}{out}")
        current = f"[cf{i}]"

    return {
        "inputs": inputs,
        "filter": ";".join(parts),
        "output": "[cfout]"
    }
