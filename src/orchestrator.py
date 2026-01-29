"""Main orchestrator for Cutter video editor automation system."""
import asyncio
import logging
import os
import tempfile
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from .config import ConfigLoader, ConfigValidator, setup_logging
from .input_handler import VideoLoader, AudioLoader, FormatValidator
from .audio_analyzer import BeatDetector
from .shot_quality import QualityScorer
from .edit_generator import ShotSelector, EditVariationGenerator
from .rendering_engine import FFmpegEncoder
from .rendering_engine.browser_renderer import SimpleBrowserRenderer
from .project_exporter import TimelineSerializer, CapCutExporter, FCPExporter
from .data_structures import VideoMetadata, BeatInfo, Edit, Shot

logger = logging.getLogger(__name__)


class ProjectManager:
    """Orchestrate the complete video editing pipeline."""

    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict] = None):
        """Initialize project manager.

        Args:
            config_path: Path to YAML configuration file.
            config: Configuration dictionary (alternative to config_path).
        """
        self.config_loader = ConfigLoader(config_path)

        if config:
            self.config = config
        elif config_path:
            self.config = self.config_loader.load_config()
        else:
            self.config = self.config_loader.load_config()

        # Setup logging
        setup_logging(self.config)

        # Initialize components
        self.video_loader = VideoLoader()
        self.audio_loader = AudioLoader()
        self.format_validator = FormatValidator()

        # State
        self.video_metadata: Optional[VideoMetadata] = None
        self.beat_info: Optional[BeatInfo] = None
        self.shots: List[Shot] = []
        self.edits: List[Edit] = []

    def validate_config(self) -> bool:
        """Validate configuration.

        Returns:
            True if configuration is valid.
        """
        validator = ConfigValidator(self.config)
        return validator.validate()

    def run_pipeline(self, render_videos: bool = True) -> List[str]:
        """Run complete video editing pipeline.

        Args:
            render_videos: Whether to render output videos.

        Returns:
            List of output file paths.
        """
        start_time = time.time()
        output_paths = []

        try:
            logger.info("=" * 60)
            logger.info("CUTTER - Video Editor Automation Pipeline")
            logger.info("=" * 60)

            # Validate config
            if not self.validate_config():
                raise ValueError("Configuration validation failed")

            # Create output directory
            output_dir = self.config.get("output_directory", "./output")
            os.makedirs(output_dir, exist_ok=True)

            # Step 1: Load and validate inputs
            logger.info("\n[1/6] Loading video and audio...")
            self._load_inputs()

            # Step 2: Analyze audio (beat detection)
            logger.info("\n[2/6] Analyzing audio...")
            self._analyze_audio()

            # Step 3: Score shot quality
            logger.info("\n[3/6] Scoring shot quality...")
            self._score_shots()

            # Step 4: Generate edit variations
            logger.info("\n[4/6] Generating edit variations...")
            self._generate_edits()

            # Step 5: Export project files
            logger.info("\n[5/6] Exporting project files...")
            project_paths = self._export_projects(output_dir)
            output_paths.extend(project_paths)

            # Step 6: Render videos (optional)
            if render_videos:
                logger.info("\n[6/6] Rendering videos...")
                video_paths = self._render_videos(output_dir)
                output_paths.extend(video_paths)
            else:
                logger.info("\n[6/6] Skipping video rendering")

            elapsed = time.time() - start_time
            logger.info("\n" + "=" * 60)
            logger.info(f"Pipeline completed in {elapsed:.1f} seconds")
            logger.info(f"Generated {len(self.edits)} variations")
            logger.info(f"Output files: {len(output_paths)}")
            logger.info("=" * 60)

            return output_paths

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

    def _load_inputs(self):
        """Load and validate video and audio inputs."""
        input_video = self.config.get("input_video")
        audio_track = self.config.get("audio_track")

        if not input_video:
            raise ValueError("input_video is required in configuration")
        if not audio_track:
            raise ValueError("audio_track is required in configuration")

        # Load video metadata
        self.video_metadata = self.video_loader.get_metadata(input_video)
        logger.info(
            f"Video: {self.video_metadata.width}x{self.video_metadata.height}, "
            f"{self.video_metadata.fps:.2f}fps, {self.video_metadata.duration:.2f}s"
        )

        # Validate video
        valid, errors = self.format_validator.validate_video(self.video_metadata)
        if not valid:
            logger.warning(f"Video validation warnings: {errors}")

        # Load audio metadata
        audio_metadata = self.audio_loader.get_metadata(audio_track)
        logger.info(f"Audio: {audio_metadata.duration:.2f}s, {audio_metadata.sample_rate}Hz")

    def _analyze_audio(self):
        """Analyze audio for beat detection."""
        audio_track = self.config.get("audio_track")
        audio_config = self.config.get("audio_analysis", {})

        sensitivity = audio_config.get("audio_sensitivity", "medium")
        threshold = audio_config.get("beat_detection_threshold", 0.5)

        # Load audio
        audio, sr = self.audio_loader.load_audio(audio_track)
        logger.info(f"Loaded {len(audio)} audio samples at {sr}Hz")

        # Detect beats
        detector = BeatDetector(sensitivity=sensitivity, custom_threshold=threshold)
        self.beat_info = detector.detect_beats(audio, sr)

        logger.info(
            f"Detected {len(self.beat_info.beat_times)} beats at {self.beat_info.bpm:.1f} BPM"
        )
        logger.info(f"Found {len(self.beat_info.energy_peaks)} energy peaks")

    def _score_shots(self):
        """Score video frames for quality and detect shots."""
        input_video = self.config.get("input_video")
        quality_config = self.config.get("shot_quality", {})

        # Initialize scorer
        scorer = QualityScorer(
            motion_weight=quality_config.get("motion_weight", 0.25),
            face_weight=quality_config.get("face_detection_weight", 0.25),
            composition_weight=quality_config.get("composition_weight", 0.15),
            contrast_weight=quality_config.get("contrast_weight", 0.25),
            lighting_weight=quality_config.get("lighting_weight", 0.10),
        )

        # Sample frames
        sample_count = min(int(self.video_metadata.total_frames), 500)
        frame_generator = self.video_loader.sample_frames(input_video, sample_count)

        # Score frames
        frame_qualities = scorer.score_all_frames(
            frame_generator, self.video_metadata
        )

        # Detect shots
        self.shots = scorer.detect_shots(frame_qualities)

        # Get summary
        summary = scorer.get_quality_summary(frame_qualities)
        logger.info(
            f"Scored {summary['total_frames']} frames, "
            f"avg quality: {summary['composite']['mean']:.3f}"
        )
        logger.info(f"Detected {len(self.shots)} shots")

        scorer.close()

    def _generate_edits(self):
        """Generate edit variations."""
        cut_config = self.config.get("cut_timing", {})
        var_config = self.config.get("variations", {})

        target_duration = cut_config.get("target_video_length", 60)
        num_variations = var_config.get("count", 3)
        sensitivities = var_config.get("sensitivities", ["low", "medium", "high"])

        # Initialize generator
        generator = EditVariationGenerator(
            base_config=self.config,
            num_variations=num_variations,
            sensitivities=sensitivities,
        )

        # Generate variations
        self.edits = generator.generate_variations(
            self.shots,
            self.beat_info,
            target_duration,
        )

        for i, edit in enumerate(self.edits):
            logger.info(
                f"Variation {i + 1}: {len(edit.timeline)} cuts, "
                f"{edit.total_duration:.2f}s, quality: {edit.quality_score:.3f}"
            )

    def _export_projects(self, output_dir: str) -> List[str]:
        """Export project files.

        Args:
            output_dir: Output directory.

        Returns:
            List of exported file paths.
        """
        audio_track = self.config.get("audio_track")
        project_name = self.config.get("project_name", "video")
        export_config = self.config.get("export", {})

        output_paths = []

        # Export JSON timelines
        serializer = TimelineSerializer(self.config)
        json_paths = serializer.export_all_variations(
            self.edits,
            self.beat_info,
            self.video_metadata,
            audio_track,
            os.path.join(output_dir, "timelines"),
        )
        output_paths.extend(json_paths)

        # Export CapCut projects
        if export_config.get("capcut_projects", True):
            from .project_exporter.capcut_exporter import export_edits_to_capcut
            capcut_paths = export_edits_to_capcut(
                self.edits,
                self.video_metadata,
                audio_track,
                os.path.join(output_dir, "capcut"),
                self.beat_info,
                project_name,
            )
            output_paths.extend(capcut_paths)

        # Export FCP XML
        if export_config.get("fcp_xml", False):
            from .project_exporter.fcp_exporter import export_edits_to_fcp
            fcp_paths = export_edits_to_fcp(
                self.edits,
                self.video_metadata,
                audio_track,
                os.path.join(output_dir, "fcpxml"),
                project_name,
            )
            output_paths.extend(fcp_paths)

        return output_paths

    def _render_videos(self, output_dir: str) -> List[str]:
        """Render output videos.

        Args:
            output_dir: Output directory.

        Returns:
            List of rendered video paths.
        """
        audio_track = self.config.get("audio_track")
        input_video = self.config.get("input_video")
        project_name = self.config.get("project_name", "video")
        export_config = self.config.get("export", {})

        video_config = export_config.get("video", {})
        platform = export_config.get("platform", "tiktok")

        # Initialize encoder
        encoder = FFmpegEncoder(
            codec=video_config.get("codec", "h264"),
            quality=video_config.get("quality", 18),
            fps=video_config.get("fps", 30),
        )

        # Initialize simple renderer
        renderer = SimpleBrowserRenderer(
            width=self.video_metadata.width,
            height=self.video_metadata.height,
            fps=video_config.get("fps", 30),
        )

        output_paths = []
        videos_dir = os.path.join(output_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)

        # Load timeline JSON files
        timelines_dir = os.path.join(output_dir, "timelines")
        serializer = TimelineSerializer(self.config)

        for i, edit in enumerate(self.edits):
            logger.info(f"Rendering variation {i + 1}/{len(self.edits)}...")

            # Create temporary frame directory
            with tempfile.TemporaryDirectory() as frames_dir:
                # Load timeline data
                timeline_path = os.path.join(
                    timelines_dir, f"{project_name}_variation_{i + 1}.json"
                )
                timeline_data = serializer.load_timeline(timeline_path)

                # Render frames
                renderer.render_placeholder_frames(
                    timeline_data,
                    frames_dir,
                    input_video,
                )

                # Encode to video
                output_path = os.path.join(
                    videos_dir, f"{project_name}_variation_{i + 1}.mp4"
                )

                encoder.encode_with_preset(
                    frames_dir,
                    output_path,
                    platform,
                    audio_track,
                )

                output_paths.append(output_path)
                logger.info(f"Rendered: {output_path}")

        return output_paths

    def run_analysis_only(self) -> Dict[str, Any]:
        """Run analysis without rendering.

        Returns:
            Dictionary with analysis results.
        """
        self._load_inputs()
        self._analyze_audio()
        self._score_shots()
        self._generate_edits()

        return {
            "video_metadata": {
                "path": self.video_metadata.path,
                "width": self.video_metadata.width,
                "height": self.video_metadata.height,
                "fps": self.video_metadata.fps,
                "duration": self.video_metadata.duration,
            },
            "beat_info": {
                "bpm": self.beat_info.bpm,
                "num_beats": len(self.beat_info.beat_times),
                "num_energy_peaks": len(self.beat_info.energy_peaks),
            },
            "shots": [
                {
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "avg_quality": s.avg_quality,
                }
                for s in self.shots
            ],
            "edits": [
                {
                    "variation_id": e.variation_id,
                    "num_cuts": len(e.timeline),
                    "total_duration": e.total_duration,
                    "quality_score": e.quality_score,
                }
                for e in self.edits
            ],
        }


def create_project_manager(config_path: str) -> ProjectManager:
    """Factory function to create ProjectManager.

    Args:
        config_path: Path to configuration file.

    Returns:
        Initialized ProjectManager instance.
    """
    return ProjectManager(config_path=config_path)
