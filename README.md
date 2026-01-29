# Cutter - AI-Powered Video Editor Automation System

Cutter is a comprehensive video editing automation system that analyzes raw footage, detects beats in audio, scores shot quality, and generates professional-grade video edits with motion graphics.

## Features

- **Audio Analysis**: Beat detection, BPM estimation, energy peak detection using librosa
- **Shot Quality Scoring**: Motion detection, face detection, composition analysis, contrast and lighting evaluation
- **Edit Generation**: Automatic timeline building with beat synchronization, multiple variation generation
- **Motion Graphics**: Browser-based rendering with text animations, transitions, particles, and color grading
- **Multi-Format Export**: MP4 video, CapCut projects, Final Cut Pro XML, JSON timelines

## Installation

### Requirements

- Python 3.8+
- FFmpeg (for video encoding)
- Chromium (for browser-based rendering, optional)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/your-repo/cutter.git
cd cutter

# Install Python dependencies
pip install -r requirements.txt

# Install Playwright and Chromium (optional, for browser rendering)
pip install playwright
playwright install chromium
```

### System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download FFmpeg from https://ffmpeg.org/download.html

## Quick Start

### 1. Create Configuration

Copy the template configuration:

```bash
cp config.template.yaml config.yaml
```

Edit `config.yaml` with your settings:

```yaml
project_name: "my_video"
input_video: "path/to/raw_footage.mp4"
audio_track: "path/to/music.mp3"
output_directory: "./output"

cut_timing:
  target_video_length: 60  # seconds

variations:
  count: 3
  sensitivities: ["low", "medium", "high"]
```

### 2. Run the Pipeline

```bash
python -m src.main --config config.yaml
```

### 3. Output

The pipeline generates:
- `output/videos/` - Rendered MP4 videos
- `output/timelines/` - JSON timeline files
- `output/capcut/` - CapCut project files

## Usage

### Command Line

```bash
# Full pipeline
python -m src.main --config config.yaml

# Analysis only (no rendering)
python -m src.main --config config.yaml --analysis-only

# Render from existing timeline
python -m src.main --timeline timeline.json --output output.mp4

# Override settings
python -m src.main --config config.yaml --target-duration 30 --variations 5

# Verbose output
python -m src.main --config config.yaml --verbose
```

### Python API

```python
from src.orchestrator import ProjectManager

# Run full pipeline
manager = ProjectManager('config.yaml')
output_paths = manager.run_pipeline()

# Analysis only
results = manager.run_analysis_only()
print(f"Detected {results['beat_info']['num_beats']} beats")
print(f"Found {len(results['shots'])} shots")
```

## Configuration Reference

### Audio Analysis

```yaml
audio_analysis:
  audio_sensitivity: "medium"  # low, medium, high
  beat_detection_threshold: 0.5  # 0.0-1.0
```

### Shot Quality

```yaml
shot_quality:
  motion_weight: 0.25
  face_detection_weight: 0.25
  composition_weight: 0.15
  contrast_weight: 0.25
  lighting_weight: 0.10
  min_quality_score: 0.4
```

### Cut Timing

```yaml
cut_timing:
  target_video_length: 60  # seconds
  min_cut_duration: 0.5    # seconds
  max_cut_duration: 5.0    # seconds
  cut_pattern: "balanced"  # low, balanced, high
```

### Motion Graphics

```yaml
motion_graphics:
  enabled: true
  intensity: "high"

  text:
    enabled: true
    entrance_animation: "slide_in"
    font_size: 96
    glow_enabled: true

  transitions:
    enabled: true
    style: "dynamic"  # geometric, organic, camera, distortion, dynamic
    duration: 0.3

  particles:
    enabled: true
    type: "confetti"  # confetti, sparkles, rain, snow, explosion, bubbles
    density: "medium"
```

### Color Grading

```yaml
color_grading:
  enabled: true
  preset_profile: "cinematic"  # cinematic, cyberpunk, vintage, VHS, neon, etc.
  saturation: 1.0
  contrast: 1.0
  brightness: 1.0
```

### Export

```yaml
export:
  formats: ["mp4"]
  capcut_projects: true
  fcp_xml: false
  platform: "tiktok"  # tiktok, instagram_reels, youtube_shorts, youtube, web

  video:
    codec: "h264"
    quality: 18  # CRF 0-51, lower = better
    fps: 30
```

## Architecture

```
src/
├── input_handler/       # Video and audio loading
├── audio_analyzer/      # Beat detection
├── shot_quality/        # Frame quality scoring
├── edit_generator/      # Timeline building
├── rendering_engine/    # Video rendering
├── project_exporter/    # Export formats
├── config/              # Configuration management
├── data_structures.py   # Core data types
├── orchestrator.py      # Pipeline orchestration
└── main.py              # CLI entry point
```

## Pipeline Flow

1. **Input Loading**: Load video and audio files, extract metadata
2. **Audio Analysis**: Detect beats, estimate BPM, find energy peaks
3. **Shot Quality Scoring**: Analyze frames for motion, faces, composition
4. **Shot Detection**: Identify scene boundaries and best shots
5. **Edit Generation**: Build timelines aligned to beats, create variations
6. **Rendering**: Apply motion graphics, color grading, encode video
7. **Export**: Generate MP4, CapCut, FCP XML files

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_beat_detector.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Performance

Typical processing times for a 60-second output video:

| Stage | Time |
|-------|------|
| Analysis | 5-8 minutes |
| Rendering (per variation) | 5-10 minutes |
| Export | 1-2 minutes |
| **Total (3 variations)** | **25-35 minutes** |

## Platform Presets

| Platform | Resolution | Aspect Ratio |
|----------|------------|--------------|
| TikTok | 1080x1920 | 9:16 |
| Instagram Reels | 1080x1920 | 9:16 |
| YouTube Shorts | 1080x1920 | 9:16 |
| YouTube | 1920x1080 | 16:9 |
| Web | 1920x1080 | 16:9 |

## Troubleshooting

### FFmpeg not found
Install FFmpeg and ensure it's in your PATH:
```bash
ffmpeg -version
```

### MediaPipe import error
MediaPipe may have issues on some platforms. The system falls back to OpenCV Haar cascades for face detection.

### Out of memory
For long videos, the system uses chunked processing. Reduce `sample_count` in the config if issues persist.

## License

MIT License

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting PRs.
