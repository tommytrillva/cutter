# Windows Setup Guide - RTX 4090 Optimized

## Prerequisites

### 1. Install Python 3.10+
Download from https://python.org or use Windows Store

### 2. Install NVIDIA Drivers
- Download latest Game Ready or Studio drivers from https://nvidia.com/drivers
- Restart after installation

### 3. Install FFmpeg with NVENC
```powershell
# Option A: Using Chocolatey (recommended)
choco install ffmpeg

# Option B: Using Scoop
scoop install ffmpeg

# Option C: Manual install
# Download from https://github.com/BtbN/FFmpeg-Builds/releases
# Get: ffmpeg-master-latest-win64-gpl.zip
# Extract to C:\ffmpeg and add C:\ffmpeg\bin to PATH
```

### 4. Verify NVENC is available
```powershell
ffmpeg -encoders 2>&1 | findstr nvenc
# Should show: h264_nvenc, hevc_nvenc
```

## Installation

### 1. Clone the repository
```powershell
git clone https://github.com/yourusername/cutter.git
cd cutter
```

### 2. Create virtual environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```powershell
pip install -r requirements.txt
playwright install chromium
```

### 4. Verify GPU detection
```powershell
python scripts/check_gpu.py
```

You should see:
```
[OK] NVIDIA GPU: NVIDIA GeForce RTX 4090
[OK] NVENC H.264 encoder available
[OK] NVDEC H.264 decoder available
```

## Usage

### 1. Configure your project
Edit `config.windows.yaml`:
```yaml
input_video_folder: C:\Videos\Raw
audio_track: C:\Videos\music.mp3
output_directory: C:\Videos\Output
```

### 2. Run the pipeline
```powershell
python -m src.main --config config.windows.yaml
```

### 3. Output
Videos will be in `C:\Videos\Output\videos\`

## Performance Expectations (RTX 4090)

| Stage | Time |
|-------|------|
| Analysis (200 frames) | ~10-15 seconds |
| Rendering (60s video) | ~2-3 minutes |
| **Total (3 variations)** | **~8-10 minutes** |

## Troubleshooting

### "NVENC not found"
- Update NVIDIA drivers
- Reinstall FFmpeg with NVENC support
- Check `ffmpeg -encoders | findstr nvenc`

### "CUDA out of memory"
- Close other GPU applications
- Reduce `parallel_frames` in config
- Process shorter videos

### Slow rendering
- Verify NVENC is being used (check log for "Using NVENC")
- Check GPU usage with Task Manager or `nvidia-smi`
- Ensure power management is set to "Prefer maximum performance"

## GPU Optimization Tips

### Set Windows power plan
```powershell
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
```

### Monitor GPU usage
```powershell
nvidia-smi -l 1  # Updates every second
```

### Expected GPU utilization
- NVENC: 30-50% (dedicated encoder chip)
- CUDA: 0-20% (if using CUDA frame processing)
- Memory: 2-4GB
