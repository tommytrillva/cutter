"""CapCut project export for Cutter."""
import json
import logging
import os
import uuid
from typing import Dict, Any, List

from ..data_structures import Edit, BeatInfo, VideoMetadata

logger = logging.getLogger(__name__)


class CapCutExporter:
    """Export edits to CapCut project format."""

    def __init__(self):
        """Initialize CapCut exporter."""
        pass

    def export_to_capcut(
        self,
        edit: Edit,
        video_metadata: VideoMetadata,
        audio_path: str,
        output_path: str,
        beat_info: BeatInfo = None,
    ) -> str:
        """Export edit to CapCut project format.

        Note: This creates a simplified CapCut-compatible JSON structure.
        CapCut uses a proprietary format that may not be fully compatible.

        Args:
            edit: Edit object to export.
            video_metadata: Source video metadata.
            audio_path: Path to audio file.
            output_path: Output file path (.capcut or .json).
            beat_info: Optional beat information.

        Returns:
            Path to output file.
        """
        # Generate unique IDs
        project_id = str(uuid.uuid4())
        video_material_id = str(uuid.uuid4())
        audio_material_id = str(uuid.uuid4())

        # Build CapCut project structure
        project = {
            "version": "1.0.0",
            "id": project_id,
            "name": os.path.splitext(os.path.basename(output_path))[0],
            "created_at": self._get_timestamp(),
            "duration": int(edit.total_duration * 1000000),  # microseconds
            "fps": 30,
            "width": video_metadata.width,
            "height": video_metadata.height,

            # Materials (source files)
            "materials": {
                "videos": [
                    {
                        "id": video_material_id,
                        "path": video_metadata.path,
                        "duration": int(video_metadata.duration * 1000000),
                        "width": video_metadata.width,
                        "height": video_metadata.height,
                        "type": "video",
                    }
                ],
                "audios": [
                    {
                        "id": audio_material_id,
                        "path": audio_path,
                        "type": "audio",
                    }
                ],
            },

            # Tracks
            "tracks": self._build_tracks(
                edit, video_material_id, audio_material_id, beat_info
            ),

            # Beat markers
            "beat_markers": self._build_beat_markers(beat_info) if beat_info else [],

            # Effects
            "effects": self._build_effects(edit),

            # Transitions
            "transitions": self._build_transitions(edit),
        }

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Write project file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(project, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported CapCut project to {output_path}")
        return output_path

    def _get_timestamp(self) -> int:
        """Get current timestamp in milliseconds."""
        import time
        return int(time.time() * 1000)

    def _build_tracks(
        self,
        edit: Edit,
        video_material_id: str,
        audio_material_id: str,
        beat_info: BeatInfo = None,
    ) -> List[Dict[str, Any]]:
        """Build track structure.

        Args:
            edit: Edit object.
            video_material_id: Video material ID.
            audio_material_id: Audio material ID.
            beat_info: Beat information.

        Returns:
            List of track dictionaries.
        """
        tracks = []

        # Video track
        video_track = {
            "id": str(uuid.uuid4()),
            "type": "video",
            "segments": [],
        }

        for i, cut in enumerate(edit.timeline):
            segment = {
                "id": str(uuid.uuid4()),
                "material_id": video_material_id,
                "source_start": int(cut.source_start * 1000000),
                "source_end": int(cut.source_end * 1000000),
                "timeline_start": int(cut.timeline_start * 1000000),
                "timeline_end": int(cut.timeline_end * 1000000),
                "speed": 1.0,
                "volume": 0.0,  # Mute video audio, use separate audio track
                "beat_alignment": cut.beat_alignment,
                "quality_score": cut.quality_score,
            }
            video_track["segments"].append(segment)

        tracks.append(video_track)

        # Audio track
        audio_track = {
            "id": str(uuid.uuid4()),
            "type": "audio",
            "segments": [
                {
                    "id": str(uuid.uuid4()),
                    "material_id": audio_material_id,
                    "source_start": 0,
                    "source_end": int(edit.total_duration * 1000000),
                    "timeline_start": 0,
                    "timeline_end": int(edit.total_duration * 1000000),
                    "volume": 1.0,
                }
            ],
        }
        tracks.append(audio_track)

        # Text track
        if edit.text_animations:
            text_track = {
                "id": str(uuid.uuid4()),
                "type": "text",
                "segments": [],
            }

            for text_anim in edit.text_animations:
                segment = {
                    "id": str(uuid.uuid4()),
                    "content": text_anim.text,
                    "timeline_start": int(text_anim.start_time * 1000000),
                    "timeline_end": int((text_anim.start_time + text_anim.duration) * 1000000),
                    "style": {
                        "font_family": text_anim.font_family,
                        "font_size": text_anim.font_size,
                        "font_color": text_anim.font_color,
                        "entrance_animation": text_anim.entrance_animation,
                        "exit_animation": text_anim.exit_animation,
                        "glow_enabled": text_anim.glow_enabled,
                    },
                }
                text_track["segments"].append(segment)

            tracks.append(text_track)

        return tracks

    def _build_beat_markers(self, beat_info: BeatInfo) -> List[Dict[str, Any]]:
        """Build beat marker list.

        Args:
            beat_info: Beat information.

        Returns:
            List of beat marker dictionaries.
        """
        markers = []

        for i, (time, strength) in enumerate(
            zip(beat_info.beat_times, beat_info.beat_strength)
        ):
            marker = {
                "id": str(uuid.uuid4()),
                "time": int(time * 1000000),
                "type": "beat",
                "strength": strength,
                "index": i,
            }
            markers.append(marker)

        # Add energy peak markers
        for time in beat_info.energy_peaks:
            marker = {
                "id": str(uuid.uuid4()),
                "time": int(time * 1000000),
                "type": "energy_peak",
            }
            markers.append(marker)

        return markers

    def _build_effects(self, edit: Edit) -> List[Dict[str, Any]]:
        """Build effects list.

        Args:
            edit: Edit object.

        Returns:
            List of effect dictionaries.
        """
        effects = []

        # Color grading effect
        if edit.color_grade:
            cg = edit.color_grade
            effect = {
                "id": str(uuid.uuid4()),
                "type": "color_grade",
                "preset": cg.preset,
                "adjustments": {
                    "saturation": cg.saturation,
                    "contrast": cg.contrast,
                    "brightness": cg.brightness,
                    "hue_shift": cg.hue_shift,
                },
                "lut": {
                    "file": cg.lut_file,
                    "strength": cg.lut_strength,
                } if cg.lut_file else None,
            }
            effects.append(effect)

        # Particle effects
        for particle in edit.particle_effects:
            effect = {
                "id": str(uuid.uuid4()),
                "type": "particles",
                "particle_type": particle.particle_type,
                "start_time": int(particle.start_time * 1000000),
                "duration": int(particle.duration * 1000000),
                "density": particle.density,
                "colors": particle.colors,
                "audio_reactive": particle.audio_reactive,
            }
            effects.append(effect)

        return effects

    def _build_transitions(self, edit: Edit) -> List[Dict[str, Any]]:
        """Build transitions list.

        Args:
            edit: Edit object.

        Returns:
            List of transition dictionaries.
        """
        transitions = []

        for trans in edit.transitions:
            transition = {
                "id": str(uuid.uuid4()),
                "style": trans.style,
                "duration": int(trans.duration * 1000000),
                "timeline_start": int(trans.timeline_start * 1000000),
                "easing": trans.easing,
            }
            transitions.append(transition)

        return transitions


def export_edits_to_capcut(
    edits: List[Edit],
    video_metadata: VideoMetadata,
    audio_path: str,
    output_dir: str,
    beat_info: BeatInfo = None,
    project_name: str = "video",
) -> List[str]:
    """Export multiple edits to CapCut projects.

    Args:
        edits: List of Edit objects.
        video_metadata: Video metadata.
        audio_path: Audio file path.
        output_dir: Output directory.
        beat_info: Beat information.
        project_name: Base project name.

    Returns:
        List of output file paths.
    """
    exporter = CapCutExporter()
    os.makedirs(output_dir, exist_ok=True)

    output_paths = []

    for i, edit in enumerate(edits):
        filename = f"{project_name}_variation_{i + 1}.capcut"
        output_path = os.path.join(output_dir, filename)

        exporter.export_to_capcut(
            edit, video_metadata, audio_path, output_path, beat_info
        )
        output_paths.append(output_path)

    logger.info(f"Exported {len(output_paths)} CapCut projects")
    return output_paths
