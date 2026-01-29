"""Timeline serialization for Cutter."""
import json
import logging
import os
from typing import Dict, Any, Optional

from ..data_structures import Edit, BeatInfo, Timeline, VideoMetadata

logger = logging.getLogger(__name__)


class TimelineSerializer:
    """Serialize edit timelines to JSON format."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize serializer.

        Args:
            config: Configuration dictionary.
        """
        self.config = config

    def edit_to_timeline(
        self,
        edit: Edit,
        beat_info: BeatInfo,
        video_metadata: VideoMetadata,
        audio_path: str,
    ) -> Timeline:
        """Convert Edit to Timeline object.

        Args:
            edit: Edit object to convert.
            beat_info: Beat information.
            video_metadata: Source video metadata.
            audio_path: Path to audio file.

        Returns:
            Timeline object.
        """
        export_config = self.config.get("export", {})
        video_config = export_config.get("video", {})

        metadata = {
            "project_name": self.config.get("project_name", "untitled"),
            "total_duration": edit.total_duration,
            "fps": video_config.get("fps", 30),
            "width": video_metadata.width,
            "height": video_metadata.height,
            "variation_id": edit.variation_id,
            "quality_score": edit.quality_score,
            "generation_params": edit.generation_params,
        }

        source_files = {
            "video": video_metadata.path,
            "audio": audio_path,
        }

        return Timeline(
            metadata=metadata,
            source_files=source_files,
            edit=edit,
            beat_info=beat_info,
        )

    def timeline_to_json(
        self,
        timeline: Timeline,
        output_path: str,
        pretty: bool = True,
    ) -> str:
        """Serialize timeline to JSON file.

        Args:
            timeline: Timeline object.
            output_path: Output file path.
            pretty: Use pretty printing.

        Returns:
            Path to output file.
        """
        data = timeline.to_json_dict()

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            if pretty:
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(data, f, ensure_ascii=False)

        logger.info(f"Exported timeline to {output_path}")
        return output_path

    def edit_to_json(
        self,
        edit: Edit,
        beat_info: BeatInfo,
        video_metadata: VideoMetadata,
        audio_path: str,
        output_path: str,
    ) -> str:
        """Convert edit to JSON and save.

        Args:
            edit: Edit object.
            beat_info: Beat information.
            video_metadata: Video metadata.
            audio_path: Audio file path.
            output_path: Output file path.

        Returns:
            Path to output file.
        """
        timeline = self.edit_to_timeline(
            edit, beat_info, video_metadata, audio_path
        )
        return self.timeline_to_json(timeline, output_path)

    def load_timeline(self, json_path: str) -> Dict[str, Any]:
        """Load timeline from JSON file.

        Args:
            json_path: Path to JSON file.

        Returns:
            Timeline data dictionary.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Loaded timeline from {json_path}")
        return data

    def export_all_variations(
        self,
        edits: list,
        beat_info: BeatInfo,
        video_metadata: VideoMetadata,
        audio_path: str,
        output_dir: str,
    ) -> list:
        """Export all edit variations to JSON.

        Args:
            edits: List of Edit objects.
            beat_info: Beat information.
            video_metadata: Video metadata.
            audio_path: Audio file path.
            output_dir: Output directory.

        Returns:
            List of output file paths.
        """
        os.makedirs(output_dir, exist_ok=True)

        output_paths = []
        project_name = self.config.get("project_name", "video")

        for i, edit in enumerate(edits):
            filename = f"{project_name}_variation_{i + 1}.json"
            output_path = os.path.join(output_dir, filename)

            self.edit_to_json(
                edit, beat_info, video_metadata, audio_path, output_path
            )
            output_paths.append(output_path)

        logger.info(f"Exported {len(output_paths)} timeline variations")
        return output_paths
