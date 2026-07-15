"""Video analysis and deterministic-with-variety montage rendering."""

import logging
import os
import random
import subprocess
import time
import json

import cv2
import librosa
import numpy as np

logger = logging.getLogger(__name__)


class RenderCanceled(RuntimeError):
    pass

STYLE_PRESETS = {
    "phonk": {
        "visualizer_type": "waveform",
        "visualizer_position": "bottom_overlay",
        "visualizer_color": "white",
    },
    "japanese": {
        "visualizer_type": "waveform",
        "visualizer_position": "bottom_overlay",
        "visualizer_color": "white",
    },
    "house": {
        "visualizer_type": "waveform",
        "visualizer_position": "bottom_overlay",
        "visualizer_color": "white",
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
VISUALIZER_TYPES = {
    "bars": "bars",
    "waveform": "waveform",
    "minimal_corner_bars": "bars",
    "label_bars": "bars",
    "thin_waveform": "waveform",
    "compact_waveform": "waveform",
}
VISUALIZER_AMPLITUDE = {
    "soft": {"bars": "lin", "waveform": "lin"},
    "normal": {"bars": "sqrt", "waveform": "sqrt"},
    "strong": {"bars": "log", "waveform": "cbrt"},
}

EFFECTS_REGISTRY = {
    "high_contrast": {
        "title": "High contrast", "emoji": "◐", "category": "Color / Look",
        "stable": True,
    },
    "saturation_boost": {
        "title": "Saturation boost", "emoji": "🌈", "category": "Color / Look",
        "stable": True,
    },
    "dark_phonk_grade": {
        "title": "Dark phonk grade", "emoji": "🌑", "category": "Color / Look",
        "stable": True,
    },
    "cold_neon_grade": {
        "title": "Cold neon grade", "emoji": "🧊", "category": "Color / Look",
        "stable": True,
    },
    "warm_gold_grade": {
        "title": "Warm gold grade", "emoji": "🌅", "category": "Color / Look",
        "stable": True,
    },
    "film_grain": {
        "title": "Film grain", "emoji": "🎞", "category": "Texture",
        "stable": True,
    },
    "vignette": {
        "title": "Vignette", "emoji": "🌘", "category": "Texture",
        "stable": True,
    },
    "scanlines": {
        "title": "Scanlines", "emoji": "▤", "category": "Texture",
        "stable": True,
    },
    "chromatic_aberration": {
        "title": "Chromatic aberration", "emoji": "⚡", "category": "Energy FX",
        "stable": True,
    },
}

EFFECT_FILTERS = {
    "high_contrast": {
        "soft": "eq=contrast=1.10",
        "normal": "eq=contrast=1.20",
        "strong": "eq=contrast=1.32",
    },
    "saturation_boost": {
        "soft": "eq=saturation=1.12",
        "normal": "eq=saturation=1.25",
        "strong": "eq=saturation=1.42",
    },
    "dark_phonk_grade": {
        "soft": "curves=r='0/0 .5/.43 1/.90':g='0/0 .5/.39 1/.82':b='0/0 .5/.48 1/.94'",
        "normal": LOOK_FILTERS["dark_purple"],
        "strong": "curves=r='0/0 .5/.25 1/.70':g='0/0 .5/.18 1/.50':b='0/0 .5/.44 1/.84'",
    },
    "cold_neon_grade": {
        "soft": "curves=r='0/0 .5/.43 1/.86':g='0/0 .5/.48 1/.92':b='0/0 .5/.57 1/1'",
        "normal": LOOK_FILTERS["neon_blue"],
        "strong": "curves=r='0/0 .5/.18 1/.56':g='0/0 .5/.29 1/.69':b='0/0 .5/.61 1/1'",
    },
    "warm_gold_grade": {
        "soft": "curves=r='0/0 .5/.56 1/1':g='0/0 .5/.51 1/.94':b='0/0 .5/.42 1/.86'",
        "normal": LOOK_FILTERS["warm_gold"],
        "strong": "curves=r='0/0 .5/.61 1/1':g='0/0 .5/.44 1/.82':b='0/0 .5/.17 1/.52'",
    },
    "film_grain": {
        "soft": "noise=alls=6:allf=t+u",
        "normal": "noise=alls=12:allf=t+u",
        "strong": "noise=alls=19:allf=t+u",
    },
    "vignette": {
        "soft": "vignette=angle=0.10",
        "normal": "vignette=angle=0.18",
        "strong": "vignette=angle=0.27",
    },
    "scanlines": {
        "soft": "drawgrid=w=iw:h=4:t=1:c=black@0.08",
        "normal": "drawgrid=w=iw:h=4:t=1:c=black@0.16",
        "strong": "drawgrid=w=iw:h=4:t=1:c=black@0.25",
    },
    "chromatic_aberration": {
        "soft": "rgbashift=rh=1:bh=-1",
        "normal": "rgbashift=rh=3:bh=-3",
        "strong": "rgbashift=rh=5:bh=-5",
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
    "speed_accents_mode": "off",
    "speed_accents_amount": 0.20,
    "speed_accents_speed": 1.25,
    "zoom_pulse": "off",
    "punch_zoom": "off",
    "bass_shake": "off",
    "flash_hit": "off",
}

MOTION_FX_LEVELS = {"off", "soft", "normal", "strong"}
MOTION_FX_KEYS = ("zoom_pulse", "punch_zoom", "bass_shake", "flash_hit")
MOTION_FX_RATES = {"soft": 0.08, "normal": 0.12, "strong": 0.18}
ZOOM_PULSE_AMOUNTS = {"soft": 0.025, "normal": 0.045, "strong": 0.075}
PUNCH_ZOOM_AMOUNTS = {"soft": 0.060, "normal": 0.100, "strong": 0.160}
BASS_SHAKE_PIXELS = {"soft": 8, "normal": 16, "strong": 28}
FLASH_HIT_BRIGHTNESS = {"soft": 0.10, "normal": 0.18, "strong": 0.28}



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


def similarity_penalty(current_clip: dict, next_clip: dict) -> float:
    """Discourage adjacent shots that are technically compatible but nearly identical."""
    color_change = color_distance(current_clip["last_color"], next_clip["first_color"])
    brightness_change = abs(current_clip.get("brightness", 0) - next_clip.get("brightness", 0))
    motion_change = abs(current_clip.get("motion", 0) - next_clip.get("motion", 0))
    if color_change < 12 and brightness_change < 6 and motion_change < 2.5:
        return 28.0
    if color_change < 20 and brightness_change < 10 and motion_change < 4:
        return 12.0
    return 0.0


def compatibility_score(current_clip: dict, next_clip: dict, target_energy: float | None = None) -> float:
    """Return a soft transition cost; lower values are more visually compatible."""
    score = color_distance(current_clip["last_color"], next_clip["first_color"])
    score += abs(current_clip.get("brightness", 0) - next_clip.get("brightness", 0)) * 0.35
    score += abs(current_clip.get("saturation", 0) - next_clip.get("saturation", 0)) * 0.18
    score += abs(current_clip.get("motion", 0) - next_clip.get("motion", 0)) * 0.8
    score += similarity_penalty(current_clip, next_clip)
    if target_energy is not None:
        score += abs(next_clip.get("visual_energy", 0) - target_energy) * 0.15
    score -= next_clip.get("quality", 0) * 0.04
    score += random.uniform(0, 5)
    return score


def choose_next_clip(current: dict, candidates: list, recent_paths: list, usage_counts: dict,
                     mode: str, target_energy=None) -> dict:
    if len(candidates) == 1:
        return candidates[0]

    cooldown_size = min(4, max(2, len(candidates) - 1)) if len(candidates) >= 3 else 1
    cooling_down = set(recent_paths[-cooldown_size:])
    unused = [clip for clip in candidates if usage_counts.get(clip["path"], 0) == 0]
    coverage_pool = unused or candidates
    available = [clip for clip in coverage_pool if clip["path"] not in cooling_down]
    if not available:
        available = [clip for clip in coverage_pool if clip["path"] != current["path"]] or coverage_pool
    visually_varied = [clip for clip in available if similarity_penalty(current, clip) == 0]
    if visually_varied:
        available = visually_varied

    if mode == "random":
        least_used = min(usage_counts.get(clip["path"], 0) for clip in available)
        return random.choice([clip for clip in available if usage_counts.get(clip["path"], 0) <= least_used + 1])
    if mode == "quality_weighted":
        weights = [
            max(1, clip.get("quality", 0) + 10) / (1 + usage_counts.get(clip["path"], 0) * 0.35)
            for clip in available
        ]
        return random.choices(available, weights=weights, k=1)[0]
    ranked = sorted(
        available,
        key=lambda clip: compatibility_score(current, clip, target_energy) + usage_counts.get(clip["path"], 0) * 2.5,
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
        merged["beat_cut_mode"] if merged["beat_cut_mode"] in {"auto", "4_beats", "8_beats", "12_beats", "16_beats"} else "auto"
    )
    merged["clip_order_mode"] = (
        merged["clip_order_mode"] if merged["clip_order_mode"] in {"visual_match", "random", "quality_weighted"} else "visual_match"
    )
    merged["speed_accents_mode"] = (
        merged["speed_accents_mode"] if merged["speed_accents_mode"] in {"off", "auto", "manual"} else "off"
    )
    try:
        merged["speed_accents_amount"] = float(merged["speed_accents_amount"])
    except (TypeError, ValueError):
        merged["speed_accents_amount"] = 0.20
    if merged["speed_accents_amount"] not in {0.10, 0.20, 0.30}:
        merged["speed_accents_amount"] = 0.20
    try:
        merged["speed_accents_speed"] = float(merged["speed_accents_speed"])
    except (TypeError, ValueError):
        merged["speed_accents_speed"] = 1.25
    if merged["speed_accents_speed"] not in {1.15, 1.25, 1.35, 1.50}:
        merged["speed_accents_speed"] = 1.25
    for option in ("zoom_pulse", "punch_zoom", "bass_shake", "flash_hit"):
        if merged.get(option) not in MOTION_FX_LEVELS:
            merged[option] = "off"
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
        "intensity": config.get("intensity", "normal"),
    }
    if result["type"] not in VISUALIZER_TYPES:
        result["type"] = preset["visualizer_type"]
    legacy_positions = {"bottom_overlay": "bottom", "top_overlay": "top", "center_bottom": "bottom"}
    result["position"] = legacy_positions.get(result["position"], result["position"])
    allowed_positions = (
        {"bottom", "top"} if result["type"] in {"bars", "waveform", "thin_waveform"}
        else {"bottom_left", "bottom_right", "top_left", "top_right"}
    )
    if result["position"] not in allowed_positions:
        result["position"] = "bottom" if result["type"] in {"bars", "waveform", "thin_waveform"} else "bottom_left"
    if result["size"] not in VISUALIZER_HEIGHTS:
        result["size"] = "medium"
    if result["background_opacity"] not in {"none", "soft", "medium"}:
        result["background_opacity"] = "none"
    if result["color"] not in VISUALIZER_COLORS:
        result["color"] = preset["visualizer_color"]
    if result["glow"] not in {"off", "soft", "strong"}:
        result["glow"] = "soft"
    if result["intensity"] not in VISUALIZER_AMPLITUDE:
        result["intensity"] = "normal"
    result["height"] = VISUALIZER_HEIGHTS[result["size"]]
    return result


def normalize_effects_config(config: dict | None, preset: dict) -> dict:
    config = config or {}
    if isinstance(config.get("selected"), list):
        selected = set(config["selected"])
        ignored = sorted(selected - set(EFFECTS_REGISTRY))
        config = {key: key in selected for key in EFFECTS_REGISTRY}
    else:
        ignored = sorted(
            key for key, value in config.items()
            if value and key not in EFFECTS_REGISTRY and key != "intensity"
        )
    if ignored:
        logger.warning("Unsupported effects ignored: %s", ", ".join(ignored))
    return {key: bool(config.get(key, False)) for key in EFFECTS_REGISTRY}


def normalize_intensity(value: str | None) -> str:
    return value if value in {"soft", "normal", "strong"} else "normal"


def choose_segment_beat_count(mode: str, style: str, bpm: float,
                              recent_beat_counts: list[int] | None = None) -> int:
    if mode != "auto":
        return int(mode.split("_")[0])
    recent_beat_counts = recent_beat_counts or []
    choices = [12, 8, 4]
    weights = [60, 30, 10]
    if recent_beat_counts[-2:] == [12, 12]:
        weights[0] = 35
    elif recent_beat_counts[-2:] == [8, 8]:
        weights[1] = 15
    if recent_beat_counts and recent_beat_counts[-1] == 4:
        weights[2] = 0
    return random.choices(choices, weights=weights, k=1)[0]


def segment_duration_for_beat_count(bpm: float, beat_count: int) -> float:
    beat_duration = 60.0 / bpm if bpm > 0 else 0.5
    return max(2.0, beat_duration * beat_count)


def duration_aware_segment(clip: dict, mode: str, style: str, bpm: float,
                           recent_beat_counts: list[int]) -> tuple[int, float, bool]:
    preferred = choose_segment_beat_count(mode, style, bpm, recent_beat_counts)
    fallback_order = {
        16: [16, 12, 8, 4],
        12: [12, 8, 4],
        8: [8, 4],
        4: [4],
    }[preferred]
    safe_duration = max(0.25, float(clip["duration"]) - 0.08)
    for beat_count in fallback_order:
        duration = segment_duration_for_beat_count(bpm, beat_count)
        if duration <= safe_duration + 0.02:
            return beat_count, duration, beat_count != preferred
    beat_count = 4
    return beat_count, min(segment_duration_for_beat_count(bpm, beat_count), safe_duration), True


def output_dimensions() -> tuple[int, int]:
    def even_env(name: str, default: int) -> int:
        try:
            value = int(os.environ.get(name, str(default)))
        except ValueError:
            logger.warning("Invalid %s; using %d", name, default)
            value = default
        value = max(320, value)
        return value if value % 2 == 0 else value - 1

    return even_env("OUTPUT_WIDTH", 1920), even_env("OUTPUT_HEIGHT", 1080)


def resolve_speed_accents(config: dict, bpm: float, style: str) -> tuple[float, float]:
    mode = config["speed_accents_mode"]
    if mode == "off":
        return 0.0, 1.0
    if mode == "manual":
        return config["speed_accents_amount"], config["speed_accents_speed"]
    if style == "phonk" and bpm >= 125:
        return 0.30, 1.35
    if style == "house" or bpm < 105:
        return 0.10, 1.15
    if style == "phonk" or bpm >= 120:
        return 0.20, 1.25
    return 0.20, 1.25


def apply_speed_accents(sequence: list, config: dict, bpm: float, style: str) -> tuple[float, float]:
    amount, speed = resolve_speed_accents(config, bpm, style)
    for clip in sequence:
        clip["_speed"] = 1.0
    if amount <= 0 or speed <= 1.0:
        logger.info("Speed accents disabled")
        return amount, speed

    eligible = [
        index for index, clip in enumerate(sequence)
        if clip["duration"] + 0.001 >= clip["_segment_duration"] * speed
    ]
    random.shuffle(eligible)
    desired_count = max(1, round(len(sequence) * amount))
    selected = []
    for index in eligible:
        if any(abs(index - chosen) <= 1 for chosen in selected):
            continue
        selected.append(index)
        if len(selected) >= desired_count:
            break
    for index in selected:
        sequence[index]["_speed"] = speed
    logger.info(
        "Speed accents mode=%s resolved_amount=%.2f resolved_speed=%.2f eligible=%d applied=%s",
        config["speed_accents_mode"], amount, speed, len(eligible), selected,
    )
    return amount, speed


def motion_fx_config(config: dict) -> dict:
    return {key: config.get(key, "off") for key in MOTION_FX_KEYS}


def assign_motion_fx(sequence: list, config: dict) -> dict:
    for clip in sequence:
        clip["_motion_fx"] = "none"
        clip["_motion_fx_level"] = "off"
        clip["_motion_filter"] = ""
        clip["_motion_filter_applied"] = False

    enabled = [(key, config.get(key, "off")) for key in MOTION_FX_KEYS if config.get(key, "off") != "off"]
    debug_force = os.environ.get("MOTION_FX_DEBUG_FORCE", "0") == "1"
    logger.info(
        "Motion FX config: zoom_pulse=%s punch_zoom=%s bass_shake=%s flash_hit=%s debug_force=%s",
        config.get("zoom_pulse", "off"), config.get("punch_zoom", "off"),
        config.get("bass_shake", "off"), config.get("flash_hit", "off"), debug_force,
    )

    if debug_force:
        forced = enabled or [(key, "strong") for key in MOTION_FX_KEYS]
        for index, clip in enumerate(sequence[:10]):
            effect, _ = forced[index % len(forced)]
            clip["_motion_fx"] = effect
            clip["_motion_fx_level"] = "strong"
            clip["_motion_fx_debug_forced"] = True
    else:
        available = list(range(len(sequence)))
        assigned = set()
        for effect, level in enabled:
            desired = min(len(available), max(1, round(len(sequence) * MOTION_FX_RATES[level])))
            candidates = list(available)
            random.shuffle(candidates)
            if effect == "zoom_pulse":
                candidates.sort(key=lambda index: sequence[index].get("_segment_duration", 0.0), reverse=True)
            elif effect in {"punch_zoom", "flash_hit"}:
                candidates.sort(key=lambda index: sequence[index].get("_segment_duration", 0.0))
            else:
                candidates.sort(
                    key=lambda index: (
                        sequence[index].get("_segment_duration", 0.0),
                        -sequence[index].get("motion", 0.0),
                    )
                )
            spaced = [
                index for index in candidates
                if not any(abs(index - previous) <= 1 for previous in assigned)
            ]
            selected = (spaced + [index for index in candidates if index not in spaced])[:desired]
            selected_set = set(selected)
            available = [index for index in available if index not in selected_set]
            assigned.update(selected_set)
            for index in selected:
                sequence[index]["_motion_fx"] = effect
                sequence[index]["_motion_fx_level"] = level

    counts = {"none": 0, **{key: 0 for key in MOTION_FX_KEYS}}
    for clip in sequence:
        counts[clip["_motion_fx"]] = counts.get(clip["_motion_fx"], 0) + 1
    percentages = {
        key: round(count / max(1, len(sequence)) * 100, 2)
        for key, count in counts.items()
    }
    logger.info(
        "Motion FX assignment: total_segments=%d zoom_pulse=%d punch_zoom=%d bass_shake=%d flash_hit=%d none=%d "
        "pct_zoom_pulse=%.2f pct_punch_zoom=%.2f pct_bass_shake=%.2f pct_flash_hit=%.2f pct_none=%.2f",
        len(sequence), counts["zoom_pulse"], counts["punch_zoom"], counts["bass_shake"],
        counts["flash_hit"], counts["none"], percentages["zoom_pulse"], percentages["punch_zoom"],
        percentages["bass_shake"], percentages["flash_hit"], percentages["none"],
    )
    for index, clip in enumerate(sequence[:20], start=1):
        logger.info(
            "Motion FX first20 segment=%03d source=%s beat_count=%s duration=%.3f speed=%.2f variant=%s selected_motion_fx=%s level=%s",
            index, os.path.basename(clip["path"]), clip.get("_beat_count"), clip.get("_segment_duration", 0.0),
            clip.get("_speed", 1.0), clip.get("_variant"), clip.get("_motion_fx", "none"),
            clip.get("_motion_fx_level", "off"),
        )
    return {
        "total_segments": len(sequence),
        "motion_fx_config": motion_fx_config(config),
        "selected_counts": counts,
        "selected_percentages": percentages,
        "debug_force": debug_force,
    }


def write_motion_fx_report(tmpdir: str, report: dict, sequence: list) -> str:
    report_path = os.path.join(tmpdir, "motion_fx_report.json")
    payload = {
        **report,
        "segments": [
            {
                "index": index,
                "source": os.path.basename(clip["path"]),
                "beat_count": clip.get("_beat_count"),
                "duration": round(float(clip.get("_segment_duration", 0.0)), 3),
                "speed": round(float(clip.get("_speed", 1.0)), 3),
                "trim_start": clip.get("_trim_start"),
                "variant": clip.get("_variant"),
                "motion_fx": clip.get("_motion_fx", "none"),
                "motion_fx_level": clip.get("_motion_fx_level", "off"),
                "ffmpeg_filter_applied": bool(clip.get("_motion_filter_applied", False)),
                "ffmpeg_filter": clip.get("_motion_filter", ""),
            }
            for index, clip in enumerate(sequence, start=1)
        ],
    }
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(payload, report_file, ensure_ascii=False, indent=2)
    logger.info("Motion FX report path=%s", report_path)
    logger.info("Motion FX report JSON: %s", json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return report_path


def allowed_variants(config: dict) -> list:
    variants = ["normal"]
    if config["allow_mirror"]:
        variants.append("mirror")
    if config["allow_reverse"]:
        variants.append("reverse")
    if config["allow_mirror_reverse"]:
        variants.append("mirror_reverse")
    return variants


def with_ffmpeg_nostdin(cmd: list) -> list:
    if cmd and cmd[0] == "ffmpeg" and "-nostdin" not in cmd:
        return ["ffmpeg", "-nostdin", *cmd[1:]]
    return cmd


def dir_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def remove_files(paths: list[str]) -> None:
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("Could not remove temp file %s: %s", path, exc)


def ffmpeg_timeout(env_name: str, default_seconds: int = 21600) -> int:
    value = os.environ.get(env_name, str(default_seconds))
    try:
        timeout = int(value)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", env_name, value, default_seconds)
        return default_seconds
    return max(60, timeout)


def run_ffmpeg(cmd: list, task: str, timeout: int, cancel_check=None):
    cmd = with_ffmpeg_nostdin(cmd)
    logger.info("%s ffmpeg start: %s", task, " ".join(cmd))
    start_time = time.time()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        while True:
            if cancel_check and cancel_check():
                logger.warning("%s canceled; terminating ffmpeg pid=%s", task, proc.pid)
                proc.terminate()
                try:
                    proc.communicate(timeout=8)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.communicate()
                raise RenderCanceled("Render canceled")
            if proc.poll() is not None:
                _, stderr = proc.communicate()
                result = subprocess.CompletedProcess(cmd, proc.returncode, stderr=(stderr or b"")[-2000:])
                break
            if time.time() - start_time > timeout:
                logger.error("%s timed out after %ss; terminating ffmpeg pid=%s", task, timeout, proc.pid)
                proc.terminate()
                try:
                    _, stderr = proc.communicate(timeout=8)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    _, stderr = proc.communicate()
                result = subprocess.CompletedProcess(cmd, 124, stderr=(stderr or b"")[-2000:])
                break
            time.sleep(2)
    except RenderCanceled:
        raise
    except Exception as exc:
        logger.exception("%s ffmpeg process failed: %s", task, exc)
        return subprocess.CompletedProcess(cmd, 1, stderr=str(exc).encode())
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")[-1500:]
        logger.error("%s ffmpeg command failed: %s\nstderr: %s", task, " ".join(cmd), stderr)
    else:
        logger.info("%s ffmpeg completed in %.1fs", task, time.time() - start_time)
    return result


def build_motion_fx_filters(effect: str, level: str, width: int, height: int) -> list[str]:
    filters = []
    if effect == "zoom_pulse":
        amount = ZOOM_PULSE_AMOUNTS[level]
        filters.extend([
            (
                "scale="
                f"w='iw*(1+{amount:.4f}*(0.5+0.5*sin(2*PI*t*1.25)))':"
                f"h='ih*(1+{amount:.4f}*(0.5+0.5*sin(2*PI*t*1.25)))':eval=frame"
            ),
            f"crop={width}:{height}:x='(iw-ow)/2':y='(ih-oh)/2'",
        ])

    elif effect == "punch_zoom":
        amount = PUNCH_ZOOM_AMOUNTS[level]
        filters.extend([
            (
                "scale="
                f"w='iw*(1+{amount:.4f}*exp(-8*t))':"
                f"h='ih*(1+{amount:.4f}*exp(-8*t))':eval=frame"
            ),
            f"crop={width}:{height}:x='(iw-ow)/2':y='(ih-oh)/2'",
        ])

    elif effect == "bass_shake":
        pixels = BASS_SHAKE_PIXELS[level]
        filters.extend([
            f"scale={width + pixels * 2}:{height + pixels * 2}",
            (
                f"crop={width}:{height}:"
                f"x='(iw-ow)/2+{pixels}*sin(2*PI*t*9)':"
                f"y='(ih-oh)/2+{pixels}*cos(2*PI*t*11)'"
            ),
        ])

    elif effect == "flash_hit":
        amount = FLASH_HIT_BRIGHTNESS[level]
        filters.append(f"eq=brightness='{amount:.3f}*exp(-10*t)':contrast='1+{amount:.3f}*exp(-10*t)':eval=frame")

    return filters


def trim_start_for_use(clip: dict, source_duration: float, random_trim: bool) -> float:
    available_start = max(0.0, float(clip["duration"]) - source_duration)
    if not random_trim or available_start <= 0.25:
        return 0.0
    use_ordinal = max(1, int(clip.get("_source_use_ordinal", 1)))
    use_total = max(1, int(clip.get("_source_use_total", 1)))
    if use_total == 1:
        return random.uniform(0, available_start)
    position = ((use_ordinal - 1) % use_total) / max(1, use_total - 1)
    base_start = position * available_start
    jitter = min(0.20, available_start * 0.08)
    return min(available_start, max(0.0, base_start + random.uniform(-jitter, jitter)))


def prepare_clip_variant(clip: dict, tmpdir: str, variant: str, target_duration: float,
                         random_trim: bool, width: int, height: int, index: int,
                         speed: float = 1.0, cancel_check=None) -> str:
    src = clip["path"]
    out = os.path.join(tmpdir, f"variant_{index:04d}_{variant}.mp4")
    source_duration = target_duration * speed
    if speed > 1.0 and clip["duration"] + 0.001 < source_duration:
        logger.info(
            "Speed accent skipped for segment %d path=%s: source %.3fs < required %.3fs",
            index, src, clip["duration"], source_duration,
        )
        speed = 1.0
        source_duration = target_duration
    if source_duration > clip["duration"] + 0.001:
        raise RuntimeError(
            f"Source clip is too short: duration={clip['duration']:.3f}s required={source_duration:.3f}s"
        )
    start = trim_start_for_use(clip, source_duration, random_trim)
    clip["_trim_start"] = round(start, 3)
    filters = [
        f"trim=duration={source_duration:.3f}",
        f"setpts=(PTS-STARTPTS)/{speed:.2f}" if speed > 1.0 else "setpts=PTS-STARTPTS",
    ]
    if "mirror" in variant:
        filters.append("hflip")
    if "reverse" in variant:
        filters.append("reverse")
    filters.extend([
        f"scale={width}:{height}:force_original_aspect_ratio=decrease",
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
    ])
    motion_fx = clip.get("_motion_fx", "none")
    motion_level = clip.get("_motion_fx_level", "off")
    motion_filters = (
        build_motion_fx_filters(motion_fx, motion_level, width, height)
        if motion_fx != "none" and motion_level != "off"
        else []
    )
    if motion_filters:
        clip["_motion_filter"] = ",".join(motion_filters)
        clip["_motion_filter_applied"] = True
        logger.info("Segment %03d motion_fx=%s level=%s filter=%s", index, motion_fx, motion_level, clip["_motion_filter"])
    else:
        clip["_motion_filter"] = ""
        clip["_motion_filter_applied"] = False
        logger.info("Segment %03d motion_fx=none", index)
    filters.extend(motion_filters)
    filters.extend([
        "setsar=1",
        "format=yuv420p",
    ])
    input_seek = ["-ss", f"{start:.3f}"] if start > 0.001 else []
    cmd = [
        "ffmpeg", "-y", *input_seek, "-i", src,
        "-vf", ",".join(filters), "-an", "-r", "30",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-t", f"{target_duration:.3f}", out,
    ]
    result = run_ffmpeg(cmd, f"Variant segment {index}", 240, cancel_check=cancel_check)
    if result.returncode != 0:
        raise RuntimeError("Ошибка подготовки видеоклипа")
    logger.info(
        "Prepared segment %d output=%s trim_start=%.3f source_duration=%.3f size_bytes=%d",
        index, out, start, source_duration, os.path.getsize(out),
    )
    return out


def build_effects_filter(preset: dict, effects: dict, intensity: str = "normal") -> str:
    if not any(effects.values()):
        return "null"
    filters = [
        EFFECT_FILTERS[effect_id][intensity]
        for effect_id, details in EFFECTS_REGISTRY.items()
        if effects.get(effect_id) and details["stable"]
    ]
    return ",".join(filters) if filters else "null"


def visualizer_layout(config: dict, width: int, height: int) -> dict:
    """Resolve polished overlay dimensions while retaining legacy full-width presets."""
    vis_type = config["type"]
    margin_x = max(20, round(width * 0.035))
    margin_y = max(20, round(height * 0.045))
    vis_h = config["height"]
    layout = {"render_width": width, "height": vis_h, "x": 0, "label": False}

    if vis_type == "minimal_corner_bars":
        layout.update(render_width=max(180, round(width * 0.30)), height=max(42, round(vis_h * 0.70)), x=margin_x)
    elif vis_type == "label_bars":
        icon_size = max(42, round(vis_h * 0.72))
        layout.update(
            render_width=max(170, round(width * 0.25)),
            height=icon_size,
            x=margin_x + icon_size + 12,
            label=True,
            icon_size=icon_size,
            panel_x=margin_x,
        )
    elif vis_type == "thin_waveform":
        render_width=max(360, round(width * 0.68))
        layout.update(render_width=render_width, height=max(24, round(vis_h * 0.38)), x=(width - render_width) // 2)
    elif vis_type == "compact_waveform":
        layout.update(render_width=max(260, round(width * 0.44)), height=max(32, round(vis_h * 0.55)), x=margin_x)

    vis_h = layout["height"]
    if vis_type in {"minimal_corner_bars", "label_bars", "compact_waveform"}:
        content_width = layout["render_width"] + (layout.get("icon_size", 0) + 12 if layout["label"] else 0)
        left_x = margin_x + (layout.get("icon_size", 0) + 12 if layout["label"] else 0)
        right_x = max(margin_x, width - margin_x - layout["render_width"])
        if layout["label"]:
            right_x = max(margin_x, width - margin_x - content_width + layout["icon_size"] + 12)
            layout["panel_x"] = max(margin_x, width - margin_x - content_width) if config["position"].endswith("right") else margin_x
        layout["x"] = right_x if config["position"].endswith("right") else left_x
        overlay_y = margin_y if config["position"].startswith("top") else max(0, height - vis_h - margin_y)
    elif config["position"] == "top":
        overlay_y = 18
    else:
        overlay_y = max(0, height - vis_h - 18)
    layout["y"] = overlay_y
    return layout


def generate_visualizer(audio_path: str, output_path: str, preset: dict, config: dict,
                        width: int, height: int, duration: float, cancel_check=None) -> bool:
    vis_type = config["type"]
    base_type = VISUALIZER_TYPES[vis_type]
    layout = visualizer_layout(config, width, height)
    vis_width = layout["render_width"]
    vis_h = layout["height"]
    color = VISUALIZER_COLORS[config["color"]]
    amplitude_scale = VISUALIZER_AMPLITUDE[config["intensity"]][base_type]
    if base_type == "bars":
        vis_filter = (
            f"showfreqs=s={vis_width}x{vis_h}:win_size=1024:ascale={amplitude_scale}:"
            f"fscale=log:colors=0x{color}|0x{color}:mode=bar:cmode=combined"
        )
    else:
        vis_filter = f"showwaves=s={vis_width}x{vis_h}:mode=cline:colors=0x{color}:scale={amplitude_scale}"
    logger.info("Selected visualizer preset=%s filter=%s", vis_type, vis_filter)
    cmd = [
        "ffmpeg", "-y", "-i", audio_path, "-filter_complex", f"[0:a]{vis_filter}[vis]",
        "-map", "[vis]", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-t", str(duration), output_path,
    ]
    return run_ffmpeg(cmd, "Visualizer", 300, cancel_check=cancel_check).returncode == 0


def build_visualizer_overlay_filter(config: dict, width: int, height: int) -> str:
    """Create a transparent full-frame overlay chain for a generated visualizer."""
    layout = visualizer_layout(config, width, height)
    vis_h = layout["height"]
    vis_width = layout["render_width"]
    overlay_x = layout["x"]
    overlay_y = layout["y"]

    alpha = {"none": 0.0, "soft": 0.18, "medium": 0.34}[config["background_opacity"]]
    box_x = max(0, overlay_x - 8)
    box_y = max(0, overlay_y - 6)
    box_w = min(width - box_x, vis_width + 16)
    box_h = min(height - box_y, vis_h + 12)
    if layout["label"]:
        box_x = layout["panel_x"] - 8
        box_w = min(width - box_x, vis_width + layout["icon_size"] + 28)

    parts = []
    if alpha:
        parts.append(
            f"[0:v]drawbox=x={box_x}:y={box_y}:w={box_w}:h={box_h}:color=black@{alpha:.2f}:t=fill[base]"
        )
    else:
        parts.append("[0:v]null[base]")
    if layout["label"]:
        icon_alpha = 0.65 if config["background_opacity"] == "none" else 0.32
        icon_thickness = "2" if config["background_opacity"] == "none" else "fill"
        parts.append(
            f"[base]drawbox=x={layout['panel_x']}:y={overlay_y}:w={layout['icon_size']}:h={layout['icon_size']}:"
            f"color=0x{VISUALIZER_COLORS[config['color']]}@{icon_alpha:.2f}:t={icon_thickness}[labelbase]"
        )
        base_label = "[labelbase]"
    else:
        base_label = "[base]"
    parts.append("[1:v]format=rgba,colorkey=0x000000:0.18:0.06[vis]")

    if config["glow"] == "off":
        parts.append(f"{base_label}[vis]overlay={overlay_x}:{overlay_y}[out]")
    else:
        sigma, glow_alpha = (5, 0.45) if config["glow"] == "soft" else (10, 0.65)
        parts.append("[vis]split[vmain][vglow]")
        parts.append(f"[vglow]gblur=sigma={sigma},colorchannelmixer=aa={glow_alpha:.2f}[glow]")
        parts.append(f"{base_label}[glow]overlay={overlay_x}:{overlay_y}[withglow]")
        parts.append(f"[withglow][vmain]overlay={overlay_x}:{overlay_y}[out]")
    overlay_filter = ";".join(parts)
    logger.info("Selected visualizer preset=%s overlay_filter=%s", config["type"], overlay_filter)
    return overlay_filter


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
                       raw_video: str, fallback_name: str | None = None,
                       cancel_check=None) -> tuple[bool, str]:
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
    chunk_timeout = ffmpeg_timeout("CHUNK_ASSEMBLY_TIMEOUT_SECONDS")
    result = run_ffmpeg(cmd, f"Raw video ({fallback_name or 'primary'})", chunk_timeout, cancel_check=cancel_check)
    stderr = result.stderr.decode(errors="replace") if result.stderr else ""
    return result.returncode == 0, stderr


def concat_video_files(paths: list[str], tmpdir: str, output_path: str,
                       duration: float, cancel_check=None) -> tuple[bool, str]:
    concat_list = os.path.join(tmpdir, "concat_chunks.txt")
    with open(concat_list, "w", encoding="utf-8") as concat_file:
        for path in paths:
            concat_file.write(f"file '{path}'\n")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
        "-c", "copy", "-t", str(duration), output_path,
    ]
    raw_timeout = ffmpeg_timeout("RAW_ASSEMBLY_TIMEOUT_SECONDS")
    result = run_ffmpeg(cmd, "Final raw chunk concat", raw_timeout, cancel_check=cancel_check)
    stderr = result.stderr.decode(errors="replace") if result.stderr else ""
    return result.returncode == 0, stderr


def mux_chunk_files_with_audio(paths: list[str], tmpdir: str, audio_path: str,
                               output_path: str, cancel_check=None) -> tuple[bool, str]:
    concat_list = os.path.join(tmpdir, "concat_chunks_final_output.txt")
    with open(concat_list, "w", encoding="utf-8") as concat_file:
        for path in paths:
            concat_file.write(f"file '{path}'\n")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list, "-i", audio_path,
        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k",
        "-shortest", output_path,
    ]
    final_timeout = ffmpeg_timeout("FINAL_RENDER_TIMEOUT_SECONDS")
    logger.info(
        "Final render direct chunk mux: chunks=%d audio=%s output=%s timeout=%ss tmp_size_bytes=%d",
        len(paths), audio_path, output_path, final_timeout, dir_size_bytes(tmpdir),
    )
    result = run_ffmpeg(cmd, "Final render direct chunk mux", final_timeout, cancel_check=cancel_check)
    stderr = result.stderr.decode(errors="replace") if result.stderr else ""
    return result.returncode == 0, stderr


def build_video(clips: list, audio_path: str, style: str, user_overrides: dict | None,
                tmpdir: str, output_path: str, progress_callback=None,
                montage_config: dict | None = None, visualizer_config: dict | None = None,
                effects_config: dict | None = None, effects_intensity: str | None = None,
                cancel_check=None, status_callback=None) -> dict:
    preset = STYLE_PRESETS.get(style, STYLE_PRESETS["phonk"])
    montage = normalize_montage_config(montage_config)
    visualizer = normalize_visualizer_config(visualizer_config, preset)
    if effects_intensity is None and isinstance(effects_config, dict):
        effects_intensity = effects_config.get("intensity")
    effects_intensity = normalize_intensity(effects_intensity)
    effects = normalize_effects_config(effects_config if effects_config is not None else user_overrides, preset)
    logger.info("Selected montage_config: %s", montage)
    logger.info("Selected visualizer_config: %s", visualizer)
    logger.info("Selected effects_config: %s intensity=%s", effects, effects_intensity)

    clips = [clip for clip in clips if clip]
    if not clips:
        raise ValueError("Не удалось проанализировать видеоклипы")

    bpm, beat_times, audio_duration = detect_beats(audio_path)
    logger.info("Audio duration %.2fs, BPM %.2f, detected beats %d", audio_duration, bpm, len(beat_times))
    if progress_callback:
        progress_callback(f"30% Audio and BPM analyzed: {bpm:.0f} BPM, {audio_duration:.1f} sec.")

    beat_duration = 60.0 / bpm if bpm > 0 else 0.5
    width, height = output_dimensions()
    logger.info(
        "Output normalized to %dx%d at 30fps; source resolutions=%s",
        width, height, sorted({f"{clip['width']}x{clip['height']}" for clip in clips}),
    )
    target_energy = float(np.mean([clip.get("visual_energy", 0) for clip in clips]))
    sequence = []
    recent_paths = []
    recent_beat_counts = []
    usage_counts = {clip["path"]: 0 for clip in clips}
    current = random.choice(clips)
    elapsed = 0.0
    overlap = 0.35 if montage["transition_style"] == "crossfade" else 0.0
    while elapsed < audio_duration:
        selected = current if not sequence else choose_next_clip(
            current, clips, recent_paths, usage_counts, montage["clip_order_mode"], target_energy
        )
        beat_count, target_duration, duration_fallback = duration_aware_segment(
            selected, montage["beat_cut_mode"], style, bpm, recent_beat_counts
        )
        needed_duration = audio_duration - elapsed + (overlap if sequence else 0.0)
        minimum_segment = overlap + 0.10 if sequence and overlap else 0.25
        segment_duration = min(target_duration, max(minimum_segment, needed_duration))
        selected = {
            **selected,
            "_segment_duration": segment_duration,
            "_beat_count": beat_count,
            "_duration_fallback": duration_fallback,
            "_source_use_ordinal": usage_counts.get(selected["path"], 0) + 1,
            "_variant": random.choice(allowed_variants(montage)),
        }
        sequence.append(selected)
        recent_paths.append(selected["path"])
        recent_beat_counts.append(beat_count)
        usage_counts[selected["path"]] = usage_counts.get(selected["path"], 0) + 1
        logger.info(
            "Segment %d beat_count=%d duration=%.3f source_duration=%.3f duration_fallback=%s "
            "source=%s source_use=%d variant=%s recent_sources=%s",
            len(sequence), beat_count, segment_duration, selected["duration"], duration_fallback,
            selected["path"], selected["_source_use_ordinal"], selected["_variant"], recent_paths[-4:],
        )
        current = selected
        elapsed += segment_duration - (overlap if len(sequence) > 1 else 0.0)
    for selected in sequence:
        selected["_source_use_total"] = usage_counts[selected["path"]]
    selected_paths = {selected["path"] for selected in sequence}
    unused_paths = sorted(os.path.basename(clip["path"]) for clip in clips if clip["path"] not in selected_paths)
    beat_count_summary = {
        beat_count: sum(1 for selected in sequence if selected["_beat_count"] == beat_count)
        for beat_count in (4, 8, 12, 16)
    }
    duration_fallbacks = sum(1 for selected in sequence if selected.get("_duration_fallback"))
    logger.info(
        "Final sequence length=%d unique_sources=%d/%d repeated_segments=%d beat_duration=%.3fs mode=%s unused=%s",
        len(sequence), len(selected_paths), len(clips), len(sequence) - len(selected_paths),
        beat_duration, montage["beat_cut_mode"], unused_paths,
    )
    logger.info(
        "Segment rhythm summary: beat_counts=%s duration_fallbacks=%d",
        beat_count_summary, duration_fallbacks,
    )
    apply_speed_accents(sequence, montage, bpm, style)
    motion_fx_report = assign_motion_fx(sequence, montage)
    effects_filter = build_effects_filter(preset, effects, effects_intensity)
    raw_video = os.path.join(tmpdir, "raw_video.mp4")
    chunk_paths = []
    last_stderr = ""
    rendered_source_paths = set()
    rendered_segments = 0
    chunk_size = 40
    total_chunks = max(1, (len(sequence) + chunk_size - 1) // chunk_size)
    for chunk_number, start_index in enumerate(range(0, len(sequence), chunk_size), start=1):
        if cancel_check and cancel_check():
            raise RenderCanceled("Render canceled")
        chunk = sequence[start_index:start_index + chunk_size]
        prepared = []
        durations = []
        chunk_source_paths = set()
        chunk_rendered_segments = 0
        logger.info(
            "Prepare chunk %d/%d segments=%d tmp_size_bytes=%d",
            chunk_number, total_chunks, len(chunk), dir_size_bytes(tmpdir),
        )
        if status_callback:
            status_callback(
                "preparing_segments",
                35 + int(25 * (chunk_number - 1) / total_chunks),
                f"Preparing chunk {chunk_number}/{total_chunks}",
                current_segment=start_index + 1,
                total_segments=len(sequence),
                chunk=chunk_number,
                total_chunks=total_chunks,
            )
        for offset, clip in enumerate(chunk, start=1):
            index = start_index + offset
            durations.append(clip["_segment_duration"])
            logger.info(
                "Preparing segment %d/%d beat_count=%d source=%s speed=%.2fx variant=%s motion_fx=%s tmp_size_bytes=%d",
                index, len(sequence), clip["_beat_count"], clip["path"], clip["_speed"], clip["_variant"],
                clip.get("_motion_fx", "none"), dir_size_bytes(tmpdir),
            )
            try:
                prepared.append(prepare_clip_variant(
                    clip, tmpdir, clip["_variant"], clip["_segment_duration"],
                    montage["allow_random_trim"], width, height, index, clip["_speed"],
                    cancel_check=cancel_check,
                ))
                chunk_source_paths.add(clip["path"])
                chunk_rendered_segments += 1
            except Exception as exc:
                if clip.get("_motion_fx", "none") != "none":
                    logger.warning("Motion FX failed for segment %d, retrying without Motion FX: %s", index, exc)
                else:
                    logger.warning("Segment %d primary variant failed: %s; retrying normal/no trim", index, exc)
                try:
                    retry_clip = {
                        **clip,
                        "_motion_fx": "none",
                        "_motion_fx_level": "off",
                        "_motion_filter": "",
                        "_motion_filter_applied": False,
                    }
                    prepared.append(prepare_clip_variant(
                        retry_clip, tmpdir, "normal", clip["_segment_duration"],
                        False, width, height, index, 1.0, cancel_check=cancel_check,
                    ))
                    chunk_source_paths.add(clip["path"])
                    chunk_rendered_segments += 1
                    clip["_motion_filter_applied"] = False
                    clip["_motion_filter"] = ""
                except Exception as retry_exc:
                    logger.error("Segment %d skipped after retry failure: %s", index, retry_exc, exc_info=True)
                    durations.pop()
            if status_callback:
                status_callback(
                    "preparing_segments",
                    35 + int(25 * index / max(1, len(sequence))),
                    f"Prepared segment {index}/{len(sequence)}",
                    current_segment=index,
                    total_segments=len(sequence),
                    chunk=chunk_number,
                    total_chunks=total_chunks,
                )
            if progress_callback and (index == len(sequence) or index % 25 == 0):
                progress_callback(f"Preparing segments {index}/{len(sequence)}.")
        if not prepared:
            logger.warning("Chunk %d had no prepared segments; skipping", chunk_number)
            continue
        chunk_duration = sum(durations)
        if montage["transition_style"] == "crossfade" and len(durations) > 1:
            chunk_duration -= 0.35 * (len(durations) - 1)
        chunk_output = os.path.join(tmpdir, f"chunk_{chunk_number:04d}.mp4")
        logger.info("Assembling chunk %d/%d prepared=%d duration=%.3f", chunk_number, total_chunks, len(prepared), chunk_duration)
        if status_callback:
            status_callback(
                "assembling_raw_video",
                60 + int(15 * (chunk_number - 1) / total_chunks),
                f"Assembling chunk {chunk_number}/{total_chunks}",
                chunk=chunk_number,
                total_chunks=total_chunks,
            )
        raw_ok, last_stderr = assemble_raw_video(
            prepared, durations, montage["transition_style"], effects_filter,
            chunk_duration, tmpdir, chunk_output, f"chunk_{chunk_number}", cancel_check=cancel_check,
        )
        if not raw_ok and montage["transition_style"] != "cut":
            raw_ok, last_stderr = assemble_raw_video(
                prepared, durations, "cut", effects_filter,
                chunk_duration, tmpdir, chunk_output, f"chunk_{chunk_number}_cut", cancel_check=cancel_check,
            )
        if not raw_ok:
            raw_ok, last_stderr = assemble_raw_video(
                prepared, durations, "cut", "null",
                chunk_duration, tmpdir, chunk_output, f"chunk_{chunk_number}_no_effects", cancel_check=cancel_check,
            )
        remove_files(prepared)
        logger.info("Cleaned chunk %d variants tmp_size_bytes=%d", chunk_number, dir_size_bytes(tmpdir))
        if not raw_ok:
            break
        chunk_paths.append(chunk_output)
        rendered_source_paths.update(chunk_source_paths)
        rendered_segments += chunk_rendered_segments
    unique_clips_used = len(rendered_source_paths)
    repeated_segments = max(0, rendered_segments - unique_clips_used)
    unused_source_clips = sorted(
        os.path.basename(clip["path"]) for clip in clips if clip["path"] not in rendered_source_paths
    )
    logger.info(
        "Render coverage: accepted_sources=%d unique_sources_used=%d segments_rendered=%d "
        "repeated_segments=%d unused_sources=%s",
        len(clips), unique_clips_used, rendered_segments, repeated_segments, unused_source_clips,
    )
    motion_fx_report_path = write_motion_fx_report(tmpdir, motion_fx_report, sequence)
    raw_ok = bool(chunk_paths)
    if not raw_ok:
        stderr_summary = (last_stderr.strip() or "ffmpeg did not return an error message")[-800:]
        raise RuntimeError(f"?????? ?????? ?????: {stderr_summary}")

    if not visualizer["enabled"]:
        if progress_callback:
            progress_callback("75% Chunks assembled. Visualizer skipped.")
            progress_callback("90% Final render.")
        if status_callback:
            status_callback("final_render", 90, "Final render direct chunk mux started")
        final_render_start = time.time()
        raw_ok, last_stderr = mux_chunk_files_with_audio(
            chunk_paths, tmpdir, audio_path, output_path, cancel_check=cancel_check
        )
        if not raw_ok:
            stderr_summary = (last_stderr.strip() or "ffmpeg did not return an error message")[-1200:]
            raise RuntimeError(f"Final render direct chunk mux failed: {stderr_summary}")
        logger.info(
            "Final render direct chunk mux completed output_size_bytes=%d elapsed_seconds=%.1f",
            os.path.getsize(output_path), time.time() - final_render_start,
        )
        file_size = os.path.getsize(output_path) / (1024 * 1024 * 1024)
        minutes, seconds = divmod(int(audio_duration), 60)
        return {
            "duration": f"{minutes}:{seconds:02d}",
            "bpm": round(bpm),
            "clips_used": unique_clips_used,
            "unique_clips_used": unique_clips_used,
            "source_clips_available": len(clips),
            "segments_rendered": rendered_segments,
            "repeated_segments": repeated_segments,
            "unused_source_clips": unused_source_clips,
            "output_resolution": f"{width}x{height}",
            "output_fps": 30,
            "duration_fallbacks": duration_fallbacks,
            "beat_count_summary": beat_count_summary,
            "file_size_gb": round(file_size, 2),
            "output": output_path,
            "segment_duration": round(beat_duration, 3),
            "motion_fx_report": motion_fx_report_path,
        }

    if len(chunk_paths) == 1:
        raw_video = chunk_paths[0]
    else:
        raw_ok, last_stderr = concat_video_files(chunk_paths, tmpdir, raw_video, audio_duration, cancel_check=cancel_check)
    if not raw_ok:
        stderr_summary = (last_stderr.strip() or "ffmpeg did not return an error message")[-800:]
        raise RuntimeError(f"?????? ?????? ?????: {stderr_summary}")
    if progress_callback:
        progress_callback("75% Raw video assembled.")
    vis_video = os.path.join(tmpdir, "visualizer.mp4")
    vis_ok = False
    if visualizer["enabled"]:
        if status_callback:
            status_callback("final_render", 78, "Generating visualizer overlay")
        vis_ok = generate_visualizer(audio_path, vis_video, preset, visualizer, width, height, audio_duration, cancel_check=cancel_check)
        if progress_callback:
            progress_callback("80% Visualizer generated." if vis_ok else "80% Visualizer unavailable; continuing without it.")
    else:
        logger.info("Visualizer disabled; generation skipped")
        if progress_callback:
            progress_callback("80% Visualizer skipped.")

    if progress_callback:
        progress_callback("90% Final render.")
    if status_callback:
        status_callback("final_render", 90, "Final render started")
    final_render_timeout = ffmpeg_timeout("FINAL_RENDER_TIMEOUT_SECONDS")
    final_render_start = time.time()
    raw_video_size = os.path.getsize(raw_video) if os.path.exists(raw_video) else 0
    if vis_ok and os.path.exists(vis_video):
        overlay_filter = build_visualizer_overlay_filter(visualizer, width, height)
        final_mode = "re-encode video with visualizer overlay"
        final_cmd = [
            "ffmpeg", "-y", "-i", raw_video, "-i", vis_video, "-i", audio_path,
            "-filter_complex", overlay_filter,
            "-map", "[out]", "-map", "2:a", "-c:v", "libx264", "-preset", "fast",
            "-crf", "17", "-c:a", "aac", "-b:a", "320k", "-t", str(audio_duration), output_path,
        ]
    else:
        final_mode = "copy video and mux audio"
        final_cmd = [
            "ffmpeg", "-y", "-i", raw_video, "-i", audio_path, "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-shortest", output_path,
        ]
    logger.info(
        "Final render mode=%s visualizer_enabled=%s visualizer_generated=%s timeout=%ss "
        "raw_video=%s raw_size_bytes=%d audio=%s output=%s",
        final_mode, visualizer["enabled"], vis_ok, final_render_timeout,
        raw_video, raw_video_size, audio_path, output_path,
    )
    final_result = run_ffmpeg(final_cmd, "Final render", final_render_timeout, cancel_check=cancel_check)
    if final_result.returncode != 0:
        stderr_tail = final_result.stderr.decode(errors="replace")[-1200:] if final_result.stderr else ""
        if final_result.returncode == 124:
            raise RuntimeError(f"Final render timed out after {final_render_timeout} seconds: {stderr_tail}")
        raise RuntimeError(f"Final render ffmpeg failed: {stderr_tail or 'no stderr output'}")
    logger.info(
        "Final render completed output_size_bytes=%d elapsed_seconds=%.1f",
        os.path.getsize(output_path), time.time() - final_render_start,
    )

    file_size = os.path.getsize(output_path) / (1024 * 1024 * 1024)
    minutes, seconds = divmod(int(audio_duration), 60)
    return {
        "duration": f"{minutes}:{seconds:02d}",
        "bpm": round(bpm),
        "clips_used": unique_clips_used,
        "unique_clips_used": unique_clips_used,
        "source_clips_available": len(clips),
        "segments_rendered": rendered_segments,
        "repeated_segments": repeated_segments,
        "unused_source_clips": unused_source_clips,
        "output_resolution": f"{width}x{height}",
        "output_fps": 30,
        "duration_fallbacks": duration_fallbacks,
        "beat_count_summary": beat_count_summary,
        "file_size_gb": round(file_size, 2),
        "output": output_path,
        "segment_duration": round(beat_duration, 3),
        "motion_fx_report": motion_fx_report_path,
    }
