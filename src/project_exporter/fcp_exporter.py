"""Final Cut Pro XML export for Cutter."""
import logging
import os
import uuid
from typing import List, Optional
from xml.etree import ElementTree as ET
from xml.dom import minidom

from ..data_structures import Edit, BeatInfo, VideoMetadata

logger = logging.getLogger(__name__)


class FCPExporter:
    """Export edits to Final Cut Pro XML format."""

    def __init__(self):
        """Initialize FCP exporter."""
        pass

    def export_to_fcp_xml(
        self,
        edit: Edit,
        video_metadata: VideoMetadata,
        audio_path: str,
        output_path: str,
        project_name: str = "Cutter Project",
    ) -> str:
        """Export edit to Final Cut Pro XML format.

        Creates FCPXML 1.9 compatible file that can be imported into
        Final Cut Pro 10.4+ and DaVinci Resolve.

        Args:
            edit: Edit object to export.
            video_metadata: Source video metadata.
            audio_path: Path to audio file.
            output_path: Output file path (.fcpxml).
            project_name: Project name.

        Returns:
            Path to output file.
        """
        # Create root element
        fcpxml = ET.Element("fcpxml", version="1.9")

        # Resources section
        resources = ET.SubElement(fcpxml, "resources")

        # Format resource
        format_id = "r1"
        fps = int(video_metadata.fps)
        frame_duration = f"1/{fps}s"

        format_elem = ET.SubElement(resources, "format")
        format_elem.set("id", format_id)
        format_elem.set("name", f"FFVideoFormat{video_metadata.height}p{fps}")
        format_elem.set("frameDuration", frame_duration)
        format_elem.set("width", str(video_metadata.width))
        format_elem.set("height", str(video_metadata.height))

        # Video asset
        video_asset_id = "r2"
        video_asset = ET.SubElement(resources, "asset")
        video_asset.set("id", video_asset_id)
        video_asset.set("name", os.path.basename(video_metadata.path))
        video_asset.set("src", f"file://{os.path.abspath(video_metadata.path)}")
        video_asset.set("duration", self._format_time(video_metadata.duration, fps))
        video_asset.set("format", format_id)
        video_asset.set("hasVideo", "1")
        video_asset.set("hasAudio", "1")

        # Audio asset
        audio_asset_id = "r3"
        audio_asset = ET.SubElement(resources, "asset")
        audio_asset.set("id", audio_asset_id)
        audio_asset.set("name", os.path.basename(audio_path))
        audio_asset.set("src", f"file://{os.path.abspath(audio_path)}")
        audio_asset.set("duration", self._format_time(edit.total_duration, fps))
        audio_asset.set("hasAudio", "1")

        # Library
        library = ET.SubElement(fcpxml, "library")
        library.set("location", f"file://{os.path.dirname(os.path.abspath(output_path))}/")

        # Event
        event = ET.SubElement(library, "event")
        event.set("name", project_name)

        # Project
        project = ET.SubElement(event, "project")
        project.set("name", project_name)

        # Sequence
        sequence = ET.SubElement(project, "sequence")
        sequence.set("format", format_id)
        sequence.set("duration", self._format_time(edit.total_duration, fps))
        sequence.set("tcStart", "0s")
        sequence.set("tcFormat", "NDF")

        # Spine (main timeline)
        spine = ET.SubElement(sequence, "spine")

        # Add video clips
        for cut in edit.timeline:
            clip = ET.SubElement(spine, "clip")
            clip.set("name", f"Clip {cut.shot_index + 1}")
            clip.set("offset", self._format_time(cut.timeline_start, fps))
            clip.set("duration", self._format_time(cut.duration, fps))
            clip.set("start", self._format_time(cut.source_start, fps))

            # Asset clip reference
            asset_clip = ET.SubElement(clip, "asset-clip")
            asset_clip.set("ref", video_asset_id)
            asset_clip.set("offset", "0s")
            asset_clip.set("name", os.path.basename(video_metadata.path))
            asset_clip.set("duration", self._format_time(cut.source_duration, fps))
            asset_clip.set("start", self._format_time(cut.source_start, fps))

            # Add marker for beat alignment
            if cut.beat_alignment != "none":
                marker = ET.SubElement(clip, "marker")
                marker.set("start", self._format_time(cut.timeline_start, fps))
                marker.set("duration", "1/30s")
                marker.set("value", cut.beat_alignment)

        # Add audio track
        audio_clip = ET.SubElement(spine, "asset-clip")
        audio_clip.set("ref", audio_asset_id)
        audio_clip.set("offset", "0s")
        audio_clip.set("name", os.path.basename(audio_path))
        audio_clip.set("duration", self._format_time(edit.total_duration, fps))
        audio_clip.set("lane", "-1")  # Audio lane

        # Pretty print
        xml_string = ET.tostring(fcpxml, encoding="unicode")
        dom = minidom.parseString(xml_string)
        pretty_xml = dom.toprettyxml(indent="  ")

        # Remove extra blank lines
        lines = [line for line in pretty_xml.split('\n') if line.strip()]
        pretty_xml = '\n'.join(lines)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Write file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(pretty_xml)

        logger.info(f"Exported FCP XML to {output_path}")
        return output_path

    def _format_time(self, seconds: float, fps: int) -> str:
        """Format time for FCPXML.

        Args:
            seconds: Time in seconds.
            fps: Frames per second.

        Returns:
            FCPXML time string (e.g., "100/30s").
        """
        frames = int(seconds * fps)
        return f"{frames}/{fps}s"

    def export_markers_to_fcp(
        self,
        beat_info: BeatInfo,
        output_path: str,
        duration: float,
        fps: int = 30,
    ) -> str:
        """Export beat markers to FCP XML.

        Creates a marker-only XML that can be imported into FCP
        to add beat markers to existing projects.

        Args:
            beat_info: Beat information.
            output_path: Output file path.
            duration: Project duration.
            fps: Frames per second.

        Returns:
            Path to output file.
        """
        fcpxml = ET.Element("fcpxml", version="1.9")

        resources = ET.SubElement(fcpxml, "resources")

        format_id = "r1"
        format_elem = ET.SubElement(resources, "format")
        format_elem.set("id", format_id)
        format_elem.set("name", f"FFVideoFormat1080p{fps}")
        format_elem.set("frameDuration", f"1/{fps}s")
        format_elem.set("width", "1920")
        format_elem.set("height", "1080")

        library = ET.SubElement(fcpxml, "library")
        event = ET.SubElement(library, "event")
        event.set("name", "Beat Markers")

        project = ET.SubElement(event, "project")
        project.set("name", "Beat Markers")

        sequence = ET.SubElement(project, "sequence")
        sequence.set("format", format_id)
        sequence.set("duration", self._format_time(duration, fps))

        spine = ET.SubElement(sequence, "spine")

        # Create a gap with markers
        gap = ET.SubElement(spine, "gap")
        gap.set("offset", "0s")
        gap.set("duration", self._format_time(duration, fps))

        # Add beat markers
        for i, (time, strength) in enumerate(
            zip(beat_info.beat_times, beat_info.beat_strength)
        ):
            marker = ET.SubElement(gap, "marker")
            marker.set("start", self._format_time(time, fps))
            marker.set("duration", f"1/{fps}s")
            if strength > 0.6:
                marker.set("value", f"Strong Beat {i + 1}")
            else:
                marker.set("value", f"Beat {i + 1}")

        # Add energy peak markers
        for time in beat_info.energy_peaks:
            marker = ET.SubElement(gap, "marker")
            marker.set("start", self._format_time(time, fps))
            marker.set("duration", f"1/{fps}s")
            marker.set("value", "Energy Peak")

        # Pretty print and write
        xml_string = ET.tostring(fcpxml, encoding="unicode")
        dom = minidom.parseString(xml_string)
        pretty_xml = dom.toprettyxml(indent="  ")

        lines = [line for line in pretty_xml.split('\n') if line.strip()]
        pretty_xml = '\n'.join(lines)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(pretty_xml)

        logger.info(f"Exported FCP markers to {output_path}")
        return output_path


def export_edits_to_fcp(
    edits: List[Edit],
    video_metadata: VideoMetadata,
    audio_path: str,
    output_dir: str,
    project_name: str = "video",
) -> List[str]:
    """Export multiple edits to FCP XML files.

    Args:
        edits: List of Edit objects.
        video_metadata: Video metadata.
        audio_path: Audio file path.
        output_dir: Output directory.
        project_name: Base project name.

    Returns:
        List of output file paths.
    """
    exporter = FCPExporter()
    os.makedirs(output_dir, exist_ok=True)

    output_paths = []

    for i, edit in enumerate(edits):
        filename = f"{project_name}_variation_{i + 1}.fcpxml"
        output_path = os.path.join(output_dir, filename)

        exporter.export_to_fcp_xml(
            edit, video_metadata, audio_path, output_path,
            project_name=f"{project_name} - Variation {i + 1}"
        )
        output_paths.append(output_path)

    logger.info(f"Exported {len(output_paths)} FCP XML files")
    return output_paths
