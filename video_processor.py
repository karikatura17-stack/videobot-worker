"""Video analysis and deterministic-with-variety montage rendering."""

import logging
import os
import random
import subprocess

import cv2
import librosa
import numpy as np

logger = logging.getLogger(__name__)

STYLE_PRESETS = {
    "phonk": {
        "visualizer_type": "bars",
        "visualizer_position": "bottom_overlay",
        "visualizer_color": "purple",
    },
    "japanese": {
        "visualizer_type": "waveform",
        "visualizer_position": "top_overlay",
        "visualizer_color": "red",
    },
    "house": {
        "visualizer_type": "bars",
        "visualizer_position": "center_bottom",
        "visualizer_color": "gold",
    },
}

LOOK_FILTERS = {
    "dark_purple": "curves=r='0/0 .5/.31 1/.78':g='0/0 .5/.24 1/.59':b='0/0 .5/.47 1/.86'",
    "neon_blue": "curves=r='0/0 .5/.24 1/.63':g='0/0 .5/.31 1/.71':b='0/0 .5/.55 1/1'",
    "warm_gold": "curves=r='0/0 .5/.55 1/1':g='0/0 .5/.47 1/.86':b='0/0 .5/.24 1/.59'",
}

VISUALIZER_COLORS = {
    "purple": "cc44ff",
    "red": "ff2244",
    "blue": "3388ff",
    "gold": "ffaa44",
    "white": "ffffff",
    "cyan": "22ddff",
    "pink": "ff55bb",
}

VISUALIZER_HEIGHTS = {"small": 60, "medium": 90, "large": 130}

EFFECTS_REGISTRY = {
    "high_contrast": {
        "title": "High contrast", "emoji": "◐", "category": "Color / Look",
        "stable": True, "filter": "eq=contrast=1.20",
    },
    "saturation_boost": {
        "title": "Saturation boost", "emoji": "🌈", "category": "Color / Look",
        "stable": True, "filter": "eq=saturation=1.25",
    },
    "dark_phonk_grade": {
        "title": "Dark phonk grade", "emoji": "🌑", "category": "Color / Look",
        "stable": True, "filter": LOOK_FILTERS["dark_purple"],
    },
    "cold_neon_grade": {
        "title": "Cold neon grade", "emoji": "🧊", "category": "Color / Look",
        "stable": True, "filter": LOOK_FILTERS["neon_blue"],
    },
    "warm_gold_grade": {
        "title": "Warm gold grade", "emoji": "🌅", "category": "Color / Look",
        "stable": True, "filter": LOOK_FILTERS["warm_gold"],
    },
    "film_grain": {
        "title": "Film grain", "emoji": "🎞", "category": "Texture",
        "stable": True, "filter": "noise=alls=12:allf=t+u",
    },
    "vignette": {
        "title": "Vignette", "emoji": "🌘", "category": "Texture",
        "stable": True, "filter": "vignette=angle=0.18",
    },
    "scanlines": {
        "title": "Scanlines", "emoji": "▤", "category": "Texture",
        "stable": True, "filter": "drawgrid=w=iw:h=4:t=1:c=black@0.16",
    },
    "chromatic_aberration": {
        "title": "Chromatic aberration", "emoji": "⚡", "category": "Energy FX",
        "stable": True, "filter": "rgbashift=rh=3:bh=-3",
    },
}

DEFAULT_MONTAGE_CONFIG = {
    "allow_mirror": True,
    "allow_reverse": False,
    "allow_mirror_reverse": False,
    "allow_random_trim": True,
    "transition_style": "cut",
    "beat_cut_mode": "auto",
    "clip_order_mode": "visual_match",
}



def analyze_clip(clip_path: str) -> dict | None:
    """Measure technical visual attributes used only for assembly choices."""
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        logger.warning("Unreadable clip: %s", clip_path)
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not total_frames or duration <= 0 or not width or not height:
        cap.release()
        logger.warning("Unreadable clip metadata: %s", clip_path)
        return None

    frames = []
    for i in range(min(12, total_frames)):
        frame_idx = int(i * max(total_frames - 1, 0) / max(min(12, total_frames) - 1, 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    if not frames:
        logger.warning("No readable frames in clip: %s", clip_path)
        return None

    grays = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    brightnesses = [gray.mean() for gray in grays]
    sharpnesses = [cv2.Laplacian(gray, cv2.CV_64F).var() for gray in grays]
    saturations = [
        cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1].mean()
        for frame in frames
    ]
    motions = [
        cv2.absdiff(frames[i - 1], frames[i]).mean()
        for i in range(1, len(frames))
    ]
    avg_brightness = float(np.mean(brightnesses))
    avg_sharpness = float(np.mean(sharpnesses))
    avg_motion = float(np.mean(motions)) if motions else 0.0
    brightness_variation = float(np.std(brightnesses))
    contrast = float(np.mean([gray.std() for gray in grays]))
    saturation = float(np.mean(saturations))
    visual_energy = min(100.0, avg_motion * 4.0 + brightness_variation * 2.0)
    avg_color = np.mean([frame.mean(axis=(0, 1)) for frame in frames], axis=0)

    # Kept as a soft technical preference, never a validity gate.
    quality = min(
        100,
        int(
            (min(avg_brightness, 200) / 200 * 30)
            + (min(avg_sharpness, 500) / 500 * 50)
            + (min(avg_motion, 10) / 10 * 20)
        ),
    )
    return {
        "path": clip_path,
        "duration": round(duration, 3),
        "fps": round(float(fps), 3),
        "width": width,
        "height": height,
        "brightness": round(avg_brightness, 2),
        "brightness_variation": round(brightness_variation, 2),
        "sharpness": round(avg_sharpness, 2),
        "motion": round(avg_motion, 2),
        "saturation": round(saturation, 2),
        "contrast": round(contrast, 2),
        "visual_energy": round(visual_energy, 2),
        "quality": quality,
        "avg_color": avg_color.tolist(),
        "first_color": frames[0].mean(axis=(0, 1)).tolist(),
        "last_color": frames[-1].mean(axis=(0, 1)).tolist(),
    }


def color_distance(c1: list, c2: list) -> float:
    return float(np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2))))


def compatibility_score(current_clip: dict, next_clip: dict, target_energy: float | None = None) -> float:
    """Return a soft transition cost; lower values are more visually compatible."""
    score = color_distance(current_clip["last_color"], next_clip["first_color"])
    score += abs(current_clip.get("brightness", 0) - next_clip.get("brightness", 0)) * 0.35
    score += abs(current_clip.get("saturation", 0) - next_clip.get("saturation", 0)) * 0.18
    score += abs(current_clip.get("motion", 0) - next_clip.get("motion", 0)) * 0.8
    if target_energy is not None:
        score += abs(next_clip.get("visual_energy", 0) - target_energy) * 0.15
    score -= next_clip.get("quality", 0) * 0.04
    score += random.uniform(0, 5)
    return score


def choose_next_clip(current: dict, candidates: list, used: set, mode: str, target_energy=None) -> dict:
    available = [clip for clip in candidates if clip["path"] not in used] or candidates
    if mode == "random":
        return random.choice(available)
    if mode == "quality_weighted":
        weights = [max(1, clip.get("quality", 0) + 10) for clip in available]
        return random.choices(available, weights=weights, k=1)[0]
    ranked = sorted(
        available,
        key=lambda clip: compatibility_score(current, clip, target_energy),
    )
    shortlist = ranked[: min(3, len(ranked))]
    return random.choice(shortlist)


def detect_beats(audio_path: str) -> tuple:
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)
    tempo_value = float(np.asarray(tempo).reshape(-1)[0]) if np.asarray(tempo).size else 0.0
    return tempo_value, librosa.frames_to_time(beat_frames, sr=sr).tolist(), duration


def normalize_montage_config(config: dict | None) -> dict:
    merged = {**DEFAULT_MONTAGE_CONFIG, **(config or {})}
    merged["transition_style"] = (
        merged["transition_style"] if merged["transition_style"] in {"cut", "crossfade"} else "cut"
    )
    merged["beat_cut_mode"] = (
        merged["beat_cut_mode"] if merged["beat_cut_mode"] in {"auto", "4_beats", "8_beats", "16_beats"} else "auto"
    )
    merged["clip_order_mode"] = (
        merged["clip_order_mode"] if merged["clip_order_mode"] in {"visual_match", "random", "quality_weighted"} else "visual_match"
    )
    for option in ("allow_mirror", "allow_reverse", "allow_mirror_reverse", "allow_random_trim"):
        merged[option] = bool(merged[option])
    return merged


def normalize_visualizer_config(config: dict | None, preset: dict) -> dict:
    config = config or {}
    result = {
        "enabled": bool(config.get("enabled", True)),
        "type": config.get("type", preset["visualizer_type"]),
        "position": config.get("position", preset["visualizer_position"]),
        "size": config.get("size", "medium"),
        "background_opacity": config.get("background_opacity", "none"),
        "color": config.get("color", preset["visualizer_color"]),
        "glow": config.get("glow", "soft"),
    }
    if result["type"] not in {"bars", "waveform"}:
        result["type"] = preset["visualizer_type"]
    legacy_positions = {"bottom": "bottom_overlay", "top": "top_overlay"}
    result["position"] = legacy_positions.get(result["position"], result["position"])
    if result["position"] not in {"bottom_overlay", "top_overlay", "center_bottom", "embedded_strip"}:
        result["position"] = preset["visualizer_position"]
    if result["size"] not in VISUALIZER_HEIGHTS:
        result["size"] = "medium"
    if result["background_opacity"] not in {"none", "soft", "medium"}:
        result["background_opacity"] = "none"
    if result["color"] not in VISUALIZER_COLORS:
        result["color"] = preset["visualizer_color"]
    if result["glow"] not in {"off", "soft", "strong"}:
        result["glow"] = "soft"
    result["height"] = VISUALIZER_HEIGHTS[result["size"]]
    return result


def normalize_effects_config(config: dict | None, preset: dict) -> dict:
    config = config or {}
    ignored = sorted(key for key, value in config.items() if value and key not in EFFECTS_REGISTRY)
    if ignored:
        logger.warning("Unsupported effects ignored: %s", ", ".join(ignored))
    return {key: bool(config.get(key, False)) for key in EFFECTS_REGISTRY}


def segment_duration_for_mode(bpm: float, mode: str, style: str) -> float:
    beat_duration = 60.0 / bpm if bpm > 0 else 0.5
    if mode == "auto":
        beat_count = random.choice((4, 8)) if style == "phonk" else (4 if style == "japanese" else random.choice((8, 16)))
    else:
        beat_count = int(mode.split("_")[0])
    return max(2.0, beat_duration * beat_count)


def allowed_variants(config: dict) -> list:
    variants = ["normal"]
    if config["allow_mirror"]:
        variants.append("mirror")
    if config["allow_reverse"]:
        variants.append("reverse")
    if config["allow_mirror_reverse"]:
        variants.append("mirror_reverse")
    return variants


def run_ffmpeg(cmd: list, task: str, timeout: int):
    result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")[-1500:]
        logger.error("%s ffmpeg command failed: %s\nstderr: %s", task, " ".join(cmd), stderr)
    return result


def prepare_clip_variant(clip: dict, tmpdir: str, variant: str, target_duration: float,
                         random_trim: bool, width: int, height: int, index: int) -> str:
    src = clip["path"]
    out = os.path.join(tmpdir, f"variant_{index:04d}_{variant}.mp4")
    available_start = max(0.0, clip["duration"] - target_duration)
    start = random.uniform(0, available_start) if random_trim and available_start > 0.25 else 0.0
    filters = [
        f"trim=start={start:.3f}:duration={target_duration:.3f}",
        "setpts=PTS-STARTPTS",
    ]
    if "mirror" in variant:
        filters.append("hflip")
    if "reverse" in variant:
        filters.append("reverse")
    filters.extend([
        f"scale={width}:{height}:force_original_aspect_ratio=decrease",
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
        "setsar=1",
        "format=yuv420p",
    ])
    cmd = [
        "ffmpeg", "-y", "-stream_loop", "-1", "-i", src,
        "-vf", ",".join(filters), "-an", "-r", "30",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-t", f"{target_duration:.3f}", out,
    ]
    result = run_ffmpeg(cmd, "Variant", 180)
    if result.returncode != 0:
        raise RuntimeError("Ошибка подготовки видеоклипа")
    return out


def build_effects_filter(preset: dict, effects: dict) -> str:
    if not any(effects.values()):
        return "null"
    filters = [
        details["filter"]
        for effect_id, details in EFFECTS_REGISTRY.items()
        if effects.get(effect_id) and details["stable"]
    ]
    return ",".join(filters) if filters else "null"


def generate_visualizer(audio_path: str, output_path: str, preset: dict, config: dict,
                        width: int, height: int, duration: float) -> bool:
    vis_type = config["type"]
    vis_h = config["height"]
    color = VISUALIZER_COLORS[config["color"]]
    if vis_type == "bars":
        vis_filter = (
            f"showfreqs=s={width}x{vis_h}:win_size=1024:ascale=sqrt:"
            f"fscale=log:colors=0x{color}|0x{color}:mode=bar:cmode=combined"
        )
    else:
        vis_filter = f"showwaves=s={width}x{vis_h}:mode=cline:colors=0x{color}:scale=sqrt"
    cmd = [
        "ffmpeg", "-y", "-i", audio_path, "-filter_complex", f"[0:a]{vis_filter}[vis]",
        "-map", "[vis]", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-t", str(duration), output_path,
    ]
    return run_ffmpeg(cmd, "Visualizer", 300).returncode == 0


def build_visualizer_overlay_filter(config: dict, width: int, height: int) -> str:
    """Create a transparent full-frame overlay chain for a generated visualizer."""
    vis_h = config["height"]
    if config["position"] == "top_overlay":
        overlay_y = 18
    elif config["position"] == "center_bottom":
        overlay_y = max(0, height - vis_h - 60)
    elif config["position"] == "embedded_strip":
        overlay_y = max(0, height - vis_h)
    else:
        overlay_y = max(0, height - vis_h - 18)

    if config["position"] == "embedded_strip":
        alpha = {"none": 0.45, "soft": 0.65, "medium": 0.85}[config["background_opacity"]]
        box_y, box_h = overlay_y, vis_h
    else:
        alpha = {"none": 0.0, "soft": 0.18, "medium": 0.34}[config["background_opacity"]]
        box_y, box_h = max(0, overlay_y - 6), min(height, vis_h + 12)

    parts = []
    if alpha:
        parts.append(
            f"[0:v]drawbox=x=0:y={box_y}:w=iw:h={box_h}:color=black@{alpha:.2f}:t=fill[base]"
        )
    else:
        parts.append("[0:v]null[base]")
    parts.append("[1:v]format=rgba,colorkey=0x000000:0.18:0.06[vis]")

    if config["glow"] == "off":
        parts.append(f"[base][vis]overlay=0:{overlay_y}[out]")
    else:
        sigma, glow_alpha = (5, 0.45) if config["glow"] == "soft" else (10, 0.65)
        parts.append("[vis]split[vmain][vglow]")
        parts.append(f"[vglow]gblur=sigma={sigma},colorchannelmixer=aa={glow_alpha:.2f}[glow]")
        parts.append(f"[base][glow]overlay=0:{overlay_y}[withglow]")
        parts.append(f"[withglow][vmain]overlay=0:{overlay_y}[out]")
    return ";".join(parts)


def build_crossfade_filter(segment_durations: list, effects_filter: str, transition: str) -> tuple[str, str]:
    fade_duration = 0.35
    if len(segment_durations) == 1:
        return f"[0:v]{effects_filter}[outv]", "[outv]"
    parts = []
    current = "[0:v]"
    offset = 0.0
    for i in range(1, len(segment_durations)):
        offset += max(0.05, segment_durations[i - 1] - fade_duration)
        out = f"[xf{i}]"
        parts.append(f"{current}[{i}:v]xfade=transition={transition}:duration={fade_duration}:offset={offset:.3f}{out}")
        current = out
    parts.append(f"{current}{effects_filter}[outv]")
    return ";".join(parts), "[outv]"


def assemble_raw_video(prepared: list, durations: list, transition_style: str,
                       effects_filter: str, audio_duration: float, tmpdir: str,
                       raw_video: str, fallback_name: str | None = None) -> tuple[bool, str]:
    """Assemble prepared segments and retain stderr for recovery reporting."""
    logger.info("Raw assembly transition_style=%s", transition_style)
    logger.info("Raw assembly effects_filter=%s", effects_filter)
    logger.info("Raw assembly prepared clips=%d", len(prepared))
    logger.info("Raw assembly first prepared paths=%s", prepared[:3])
    logger.info("Raw assembly fallback retry=%s", fallback_name or "none")

    if transition_style == "crossfade" and len(prepared) > 1:
        filter_complex, output_label = build_crossfade_filter(durations, effects_filter, "fade")
        inputs = [item for path in prepared for item in ("-i", path)]
        cmd = [
            "ffmpeg", "-y", *inputs, "-filter_complex", filter_complex, "-map", output_label,
            "-t", str(audio_duration), "-c:v", "libx264", "-preset", "fast", "-crf", "18", raw_video,
        ]
    else:
        concat_list = os.path.join(tmpdir, f"concat_{fallback_name or 'primary'}.txt")
        with open(concat_list, "w", encoding="utf-8") as concat_file:
            for path in prepared:
                concat_file.write(f"file '{path}'\n")
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
            "-vf", effects_filter, "-t", str(audio_duration),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18", raw_video,
        ]

    logger.info("Raw ffmpeg command: %s", " ".join(cmd))
    result = run_ffmpeg(cmd, f"Raw video ({fallback_name or 'primary'})", 3600)
    stderr = result.stderr.decode(errors="replace") if result.stderr else ""
    return result.returncode == 0, stderr


def build_video(clips: list, audio_path: str, style: str, user_overrides: dict | None,
                tmpdir: str, output_path: str, progress_callback=None,
                montage_config: dict | None = None, visualizer_config: dict | None = None,
                effects_config: dict | None = None) -> dict:
    preset = STYLE_PRESETS.get(style, STYLE_PRESETS["phonk"])
    montage = normalize_montage_config(montage_config)
    visualizer = normalize_visualizer_config(visualizer_config, preset)
    effects = normalize_effects_config(effects_config if effects_config is not None else user_overrides, preset)
    logger.info("Selected montage_config: %s", montage)
    logger.info("Selected visualizer_config: %s", visualizer)
    logger.info("Selected effects_config: %s", effects)

    clips = [clip for clip in clips if clip]
    if not clips:
        raise ValueError("Не удалось проанализировать видеоклипы")

    bpm, beat_times, audio_duration = detect_beats(audio_path)
    logger.info("Audio duration %.2fs, BPM %.2f, detected beats %d", audio_duration, bpm, len(beat_times))
    if progress_callback:
        progress_callback(f"30% Audio and BPM analyzed: {bpm:.0f} BPM, {audio_duration:.1f} sec.")

    target_duration = segment_duration_for_mode(bpm, montage["beat_cut_mode"], style)
    width, height = clips[0]["width"], clips[0]["height"]
    target_energy = float(np.mean([clip.get("visual_energy", 0) for clip in clips]))
    sequence = []
    used = set()
    current = random.choice(clips)
    elapsed = 0.0
    overlap = 0.35 if montage["transition_style"] == "crossfade" else 0.0
    while elapsed < audio_duration:
        needed_duration = audio_duration - elapsed + (overlap if sequence else 0.0)
        segment_duration = min(target_duration, max(2.0, needed_duration))
        selected = current if not sequence else choose_next_clip(
            current, clips, used, montage["clip_order_mode"], target_energy
        )
        selected = {**selected, "_segment_duration": segment_duration, "_variant": random.choice(allowed_variants(montage))}
        sequence.append(selected)
        used.add(selected["path"])
        current = selected
        elapsed += segment_duration - (overlap if len(sequence) > 1 else 0.0)
    logger.info("Final sequence length: %d segments, target duration %.3fs", len(sequence), target_duration)
    if progress_callback:
        progress_callback(f"45% Montage plan built: {len(sequence)} segments.")

    prepared = []
    durations = []
    for index, clip in enumerate(sequence, start=1):
        durations.append(clip["_segment_duration"])
        prepared.append(prepare_clip_variant(
            clip, tmpdir, clip["_variant"], clip["_segment_duration"],
            montage["allow_random_trim"], width, height, index,
        ))
    if progress_callback:
        progress_callback("60% Video segments prepared.")

    effects_filter = build_effects_filter(preset, effects)
    raw_video = os.path.join(tmpdir, "raw_video.mp4")
    raw_ok, last_stderr = assemble_raw_video(
        prepared, durations, montage["transition_style"], effects_filter,
        audio_duration, tmpdir, raw_video,
    )
    if not raw_ok:
        raw_ok, last_stderr = assemble_raw_video(
            prepared, durations, "cut", effects_filter,
            audio_duration, tmpdir, raw_video, "retry_1_hard_cut_selected_effects",
        )
    if not raw_ok:
        raw_ok, last_stderr = assemble_raw_video(
            prepared, durations, "cut", "null",
            audio_duration, tmpdir, raw_video, "retry_2_hard_cut_no_effects",
        )
    if not raw_ok:
        try:
            fallback_prepared = [
                prepare_clip_variant(
                    clip, tmpdir, "normal", clip["_segment_duration"],
                    False, width, height, index + len(sequence),
                )
                for index, clip in enumerate(sequence, start=1)
            ]
            raw_ok, last_stderr = assemble_raw_video(
                fallback_prepared, durations, "cut", "null",
                audio_duration, tmpdir, raw_video, "retry_3_normal_no_trim",
            )
        except Exception as exc:
            logger.error("Final raw assembly fallback preparation failed: %s", exc, exc_info=True)
    if not raw_ok:
        stderr_summary = (last_stderr.strip() or "ffmpeg did not return an error message")[-800:]
        raise RuntimeError(f"Ошибка сборки видео: {stderr_summary}")
    if progress_callback:
        progress_callback("70% Raw video assembled.")

    vis_video = os.path.join(tmpdir, "visualizer.mp4")
    vis_ok = False
    if visualizer["enabled"]:
        vis_ok = generate_visualizer(audio_path, vis_video, preset, visualizer, width, height, audio_duration)
        if progress_callback:
            progress_callback("80% Visualizer generated." if vis_ok else "80% Visualizer unavailable; continuing without it.")
    else:
        logger.info("Visualizer disabled; generation skipped")
        if progress_callback:
            progress_callback("80% Visualizer skipped.")

    if progress_callback:
        progress_callback("90% Final render.")
    if vis_ok and os.path.exists(vis_video):
        overlay_filter = build_visualizer_overlay_filter(visualizer, width, height)
        final_cmd = [
            "ffmpeg", "-y", "-i", raw_video, "-i", vis_video, "-i", audio_path,
            "-filter_complex", overlay_filter,
            "-map", "[out]", "-map", "2:a", "-c:v", "libx264", "-preset", "medium",
            "-crf", "17", "-c:a", "aac", "-b:a", "320k", "-t", str(audio_duration), output_path,
        ]
    else:
        final_cmd = [
            "ffmpeg", "-y", "-i", raw_video, "-i", audio_path, "-map", "0:v", "-map", "1:a",
            "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-c:a", "aac",
            "-b:a", "320k", "-t", str(audio_duration), output_path,
        ]
    if run_ffmpeg(final_cmd, "Final render", 3600).returncode != 0:
        raise RuntimeError("Ошибка финального рендера")

    file_size = os.path.getsize(output_path) / (1024 * 1024 * 1024)
    minutes, seconds = divmod(int(audio_duration), 60)
    return {
        "duration": f"{minutes}:{seconds:02d}",
        "bpm": round(bpm),
        "clips_used": len(sequence),
        "file_size_gb": round(file_size, 2),
        "output": output_path,
        "segment_duration": round(target_duration, 3),
    }
