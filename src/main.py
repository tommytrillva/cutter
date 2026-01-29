"""Main entry point for Cutter video editor automation system."""
import argparse
import logging
import sys
from pathlib import Path

from .orchestrator import ProjectManager

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="cutter",
        description="Cutter - AI-powered video editing automation system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with config file
  python -m src.main --config config.yaml

  # Run analysis only (no rendering)
  python -m src.main --config config.yaml --analysis-only

  # Render from existing timeline
  python -m src.main --timeline timeline.json --output output.mp4

  # Override specific settings
  python -m src.main --config config.yaml --target-duration 30 --variations 5
        """,
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--timeline", "-t",
        type=str,
        help="Path to timeline JSON file (for rendering only)",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory or file path",
    )

    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Run analysis without rendering videos",
    )

    parser.add_argument(
        "--target-duration",
        type=float,
        help="Override target video duration (seconds)",
    )

    parser.add_argument(
        "--variations",
        type=int,
        help="Override number of variations to generate",
    )

    parser.add_argument(
        "--platform",
        choices=["tiktok", "instagram_reels", "youtube_shorts", "youtube", "web"],
        help="Target platform for export preset",
    )

    parser.add_argument(
        "--sensitivity",
        choices=["low", "medium", "high"],
        help="Beat detection sensitivity",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-error output",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="Cutter 1.0.0",
    )

    return parser.parse_args()


def apply_overrides(config: dict, args) -> dict:
    """Apply command-line overrides to configuration.

    Args:
        config: Base configuration dictionary.
        args: Parsed arguments.

    Returns:
        Modified configuration.
    """
    if args.output:
        config["output_directory"] = args.output

    if args.target_duration:
        if "cut_timing" not in config:
            config["cut_timing"] = {}
        config["cut_timing"]["target_video_length"] = args.target_duration

    if args.variations:
        if "variations" not in config:
            config["variations"] = {}
        config["variations"]["count"] = args.variations

    if args.platform:
        if "export" not in config:
            config["export"] = {}
        config["export"]["platform"] = args.platform

    if args.sensitivity:
        if "audio_analysis" not in config:
            config["audio_analysis"] = {}
        config["audio_analysis"]["audio_sensitivity"] = args.sensitivity

    # Logging level
    if args.verbose:
        if "logging" not in config:
            config["logging"] = {}
        config["logging"]["level"] = "DEBUG"
    elif args.quiet:
        if "logging" not in config:
            config["logging"] = {}
        config["logging"]["level"] = "ERROR"

    return config


def run_from_timeline(timeline_path: str, output_path: str, audio_path: str = None):
    """Render video from existing timeline.

    Args:
        timeline_path: Path to timeline JSON.
        output_path: Output video path.
        audio_path: Optional audio file path.
    """
    import json
    import tempfile
    import os

    from .rendering_engine import FFmpegEncoder
    from .rendering_engine.browser_renderer import SimpleBrowserRenderer

    # Load timeline
    with open(timeline_path, 'r') as f:
        timeline = json.load(f)

    metadata = timeline.get("metadata", {})
    source_files = timeline.get("source_files", {})

    source_video = source_files.get("video")
    if audio_path is None:
        audio_path = source_files.get("audio")

    if not source_video:
        raise ValueError("Timeline does not specify source video")

    # Initialize renderer
    renderer = SimpleBrowserRenderer(
        width=metadata.get("width", 1080),
        height=metadata.get("height", 1920),
        fps=metadata.get("fps", 30),
    )

    # Initialize encoder
    encoder = FFmpegEncoder(fps=metadata.get("fps", 30))

    # Render frames
    with tempfile.TemporaryDirectory() as frames_dir:
        renderer.render_placeholder_frames(
            timeline,
            frames_dir,
            source_video,
        )

        # Encode to video
        encoder.encode_from_frames(
            frames_dir,
            output_path,
            audio_path,
        )

    logger.info(f"Rendered video to {output_path}")


def main():
    """Main entry point."""
    args = parse_args()

    # Handle timeline-only rendering
    if args.timeline:
        if not args.output:
            args.output = "output.mp4"

        run_from_timeline(args.timeline, args.output)
        return 0

    # Require config for full pipeline
    if not args.config:
        print("Error: --config is required for full pipeline")
        print("Use --help for usage information")
        return 1

    # Check config file exists
    if not Path(args.config).exists():
        print(f"Error: Configuration file not found: {args.config}")
        return 1

    try:
        # Create project manager
        manager = ProjectManager(config_path=args.config)

        # Apply command-line overrides
        manager.config = apply_overrides(manager.config, args)

        # Run pipeline
        if args.analysis_only:
            results = manager.run_analysis_only()

            print("\n=== Analysis Results ===")
            print(f"Video: {results['video_metadata']['path']}")
            print(f"  Resolution: {results['video_metadata']['width']}x{results['video_metadata']['height']}")
            print(f"  Duration: {results['video_metadata']['duration']:.2f}s")
            print(f"\nBeat Analysis:")
            print(f"  BPM: {results['beat_info']['bpm']:.1f}")
            print(f"  Beats: {results['beat_info']['num_beats']}")
            print(f"  Energy peaks: {results['beat_info']['num_energy_peaks']}")
            print(f"\nShots: {len(results['shots'])}")
            print(f"\nEdit Variations: {len(results['edits'])}")
            for edit in results['edits']:
                print(f"  Variation {edit['variation_id'] + 1}: "
                      f"{edit['num_cuts']} cuts, "
                      f"{edit['total_duration']:.2f}s, "
                      f"quality: {edit['quality_score']:.3f}")
        else:
            output_paths = manager.run_pipeline()

            print(f"\n=== Output Files ===")
            for path in output_paths:
                print(f"  {path}")

        return 0

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
