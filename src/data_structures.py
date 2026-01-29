"""Core data structures for Cutter video editor automation system."""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


class Sensitivity(Enum):
    """Sensitivity levels for beat detection and cut generation."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CutPattern(Enum):
    """Cut pattern styles for edit generation."""
    BALANCED = "balanced"
    LOW = "low"
    HIGH = "high"


class TransitionStyle(Enum):
    """Transition style categories."""
    GEOMETRIC = "geometric"
    ORGANIC = "organic"
    CAMERA = "camera"
    DISTORTION = "distortion"
    DYNAMIC = "dynamic"


class ParticleType(Enum):
    """Particle effect types."""
    CONFETTI = "confetti"
    SPARKLES = "sparkles"
    RAIN = "rain"
    SNOW = "snow"
    EXPLOSION = "explosion"
    BUBBLES = "bubbles"


class ColorGradePreset(Enum):
    """Color grading preset profiles."""
    CINEMATIC = "cinematic"
    CYBERPUNK = "cyberpunk"
    VINTAGE = "vintage"
    VHS = "vhs"
    NEON = "neon"
    SEPIA = "sepia"
    NOIR = "noir"
    COOL = "cool"
    WARM = "warm"
    SATURATED = "saturated"


class Platform(Enum):
    """Target platform presets."""
    TIKTOK = "tiktok"
    INSTAGRAM_REELS = "instagram_reels"
    YOUTUBE_SHORTS = "youtube_shorts"
    YOUTUBE = "youtube"
    WEB = "web"


# Platform resolution presets
PLATFORM_RESOLUTIONS = {
    Platform.TIKTOK: (1080, 1920),
    Platform.INSTAGRAM_REELS: (1080, 1920),
    Platform.YOUTUBE_SHORTS: (1080, 1920),
    Platform.YOUTUBE: (1920, 1080),
    Platform.WEB: (1920, 1080),
}


@dataclass
class VideoMetadata:
    """Metadata for a video file."""
    path: str
    width: int
    height: int
    fps: float
    duration: float  # seconds
    codec: str
    total_frames: int
    file_size: int  # bytes

    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio."""
        return self.width / self.height if self.height > 0 else 0

    @property
    def is_portrait(self) -> bool:
        """Check if video is portrait orientation."""
        return self.height > self.width


@dataclass
class AudioMetadata:
    """Metadata for an audio file."""
    path: str
    sample_rate: int
    duration: float  # seconds
    channels: int


@dataclass
class BeatInfo:
    """Beat detection results from audio analysis."""
    beat_times: List[float]  # Seconds when beats occur
    bpm: float  # Beats per minute
    beat_strength: List[float]  # 0-1 confidence per beat
    energy_peaks: List[float]  # Times of energy peaks
    energy_curve: Optional[List[float]] = None  # Energy over time

    @property
    def num_beats(self) -> int:
        """Return total number of detected beats."""
        return len(self.beat_times)

    def get_nearest_beat(self, time: float) -> tuple:
        """Get the nearest beat to a given time.

        Returns:
            Tuple of (beat_index, beat_time, distance)
        """
        if not self.beat_times:
            return (-1, 0.0, float('inf'))

        min_dist = float('inf')
        nearest_idx = 0
        for i, bt in enumerate(self.beat_times):
            dist = abs(bt - time)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        return (nearest_idx, self.beat_times[nearest_idx], min_dist)


@dataclass
class FrameQuality:
    """Quality scores for a single frame."""
    frame_index: int
    timestamp: float  # seconds
    motion_score: float  # 0-1
    face_score: float  # 0-1
    composition_score: float  # 0-1
    contrast_score: float  # 0-1
    lighting_score: float  # 0-1
    composite_score: float  # 0-1 weighted average

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "motion_score": self.motion_score,
            "face_score": self.face_score,
            "composition_score": self.composition_score,
            "contrast_score": self.contrast_score,
            "lighting_score": self.lighting_score,
            "composite_score": self.composite_score,
        }


@dataclass
class Shot:
    """A contiguous shot (sequence of frames) in the video."""
    start_frame: int
    end_frame: int
    start_time: float  # seconds
    end_time: float  # seconds
    avg_quality: float  # average composite score
    peak_quality: float  # max composite score
    peak_frame: int  # frame with highest quality

    @property
    def duration(self) -> float:
        """Duration of the shot in seconds."""
        return self.end_time - self.start_time

    @property
    def frame_count(self) -> int:
        """Number of frames in the shot."""
        return self.end_frame - self.start_frame + 1


@dataclass
class Cut:
    """A single cut in the edit timeline."""
    source_start: float  # Source video start (seconds)
    source_end: float  # Source video end (seconds)
    timeline_start: float  # Output timeline start (seconds)
    timeline_end: float  # Output timeline end (seconds)
    quality_score: float  # Quality 0-1
    beat_alignment: str  # "strong_beat", "energy_peak", "transition", "filler"
    beat_index: int  # Which beat this aligns to (-1 if none)
    shot_index: int  # Reference to source shot

    @property
    def duration(self) -> float:
        """Duration of the cut in seconds."""
        return self.timeline_end - self.timeline_start

    @property
    def source_duration(self) -> float:
        """Duration of source footage used."""
        return self.source_end - self.source_start

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "type": "video_clip",
            "source_start": self.source_start,
            "source_end": self.source_end,
            "timeline_start": self.timeline_start,
            "timeline_end": self.timeline_end,
            "quality_score": self.quality_score,
            "beat_alignment": self.beat_alignment,
            "beat_index": self.beat_index,
            "shot_index": self.shot_index,
        }


@dataclass
class Transition:
    """A transition between cuts."""
    style: str  # wipe, zoom, dissolve, etc.
    duration: float  # seconds
    timeline_start: float  # When transition begins
    easing: str = "ease-in-out"  # CSS easing function
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "type": "transition",
            "style": self.style,
            "duration": self.duration,
            "timeline_start": self.timeline_start,
            "easing": self.easing,
            "params": self.params,
        }


@dataclass
class TextAnimation:
    """Text overlay with animation."""
    text: str
    start_time: float  # seconds
    duration: float  # seconds
    entrance_animation: str  # fade_in, slide_in, etc.
    exit_animation: str  # fade_out, slide_out, etc.
    font_family: str = "Arial"
    font_size: int = 96
    font_color: str = "#FFFFFF"
    position: tuple = (0.5, 0.5)  # normalized x, y (0-1)
    glow_enabled: bool = False
    glow_intensity: float = 0.5
    drop_shadow: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "type": "text",
            "text": self.text,
            "start_time": self.start_time,
            "duration": self.duration,
            "entrance": self.entrance_animation,
            "exit": self.exit_animation,
            "font_family": self.font_family,
            "font_size": self.font_size,
            "font_color": self.font_color,
            "position_x": self.position[0],
            "position_y": self.position[1],
            "glow_enabled": self.glow_enabled,
            "glow_intensity": self.glow_intensity,
            "drop_shadow": self.drop_shadow,
        }


@dataclass
class ParticleEffect:
    """Particle effect specification."""
    particle_type: str  # confetti, sparkles, etc.
    start_time: float  # seconds
    duration: float  # seconds
    density: str  # low, medium, high
    colors: List[str] = field(default_factory=list)
    audio_reactive: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "type": "particles",
            "particle_type": self.particle_type,
            "start_time": self.start_time,
            "duration": self.duration,
            "density": self.density,
            "colors": self.colors,
            "audio_reactive": self.audio_reactive,
        }


@dataclass
class ColorGrade:
    """Color grading settings."""
    preset: Optional[str] = None  # cinematic, cyberpunk, etc.
    lut_file: Optional[str] = None
    lut_strength: float = 0.75  # 0-1
    saturation: float = 1.0  # 0-2
    contrast: float = 1.0  # 0-2
    brightness: float = 1.0  # 0-2
    hue_shift: float = 0.0  # -180 to 180 degrees
    blur: float = 0.0
    glow: float = 0.0
    grain: float = 0.0
    chromatic_aberration: float = 0.0
    vignette: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "preset": self.preset,
            "lut_file": self.lut_file,
            "lut_strength": self.lut_strength,
            "saturation": self.saturation,
            "contrast": self.contrast,
            "brightness": self.brightness,
            "hue_shift": self.hue_shift,
            "blur": self.blur,
            "glow": self.glow,
            "grain": self.grain,
            "chromatic_aberration": self.chromatic_aberration,
            "vignette": self.vignette,
        }


@dataclass
class Edit:
    """A complete edit with timeline and effects."""
    timeline: List[Cut]  # Cuts in order
    transitions: List[Transition]  # Transitions between cuts
    text_animations: List[TextAnimation]
    particle_effects: List[ParticleEffect]
    color_grade: ColorGrade
    total_duration: float  # seconds
    generation_params: Dict[str, Any]  # Sensitivity, pattern, etc.
    quality_score: float  # Average quality 0-1
    variation_id: int = 0  # Variation number

    @property
    def num_cuts(self) -> int:
        """Return total number of cuts."""
        return len(self.timeline)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "timeline": [cut.to_dict() for cut in self.timeline],
            "transitions": [t.to_dict() for t in self.transitions],
            "text_animations": [t.to_dict() for t in self.text_animations],
            "particle_effects": [p.to_dict() for p in self.particle_effects],
            "color_grading": self.color_grade.to_dict(),
            "total_duration": self.total_duration,
            "generation_params": self.generation_params,
            "quality_score": self.quality_score,
            "variation_id": self.variation_id,
        }


@dataclass
class Timeline:
    """Complete timeline for rendering, including all metadata."""
    metadata: Dict[str, Any]
    source_files: Dict[str, str]
    edit: Edit
    beat_info: BeatInfo

    def to_json_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "metadata": self.metadata,
            "source_files": self.source_files,
            "timeline": [cut.to_dict() for cut in self.edit.timeline],
            "transitions": [t.to_dict() for t in self.edit.transitions],
            "animations": {
                "text": [t.to_dict() for t in self.edit.text_animations],
                "particles": [p.to_dict() for p in self.edit.particle_effects],
            },
            "color_grading": self.edit.color_grade.to_dict(),
            "beat_info": {
                "beat_times": self.beat_info.beat_times,
                "bpm": self.beat_info.bpm,
                "energy_peaks": self.beat_info.energy_peaks,
            },
            "generation_params": self.edit.generation_params,
        }
