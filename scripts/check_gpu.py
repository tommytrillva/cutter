#!/usr/bin/env python3
"""GPU capability check for Cutter video editor.

Run this script to verify your NVIDIA GPU setup is working correctly.
"""
import subprocess
import sys


def print_header(text):
    print(f"\n{'='*60}")
    print(f" {text}")
    print('='*60)


def print_status(name, status, details=None):
    icon = "[OK]" if status else "[!!]"
    color_start = "\033[92m" if status else "\033[91m"
    color_end = "\033[0m"
    print(f"  {color_start}{icon}{color_end} {name}")
    if details:
        print(f"      {details}")


def check_nvidia_smi():
    """Check nvidia-smi availability and GPU info."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,cuda_version",
             "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            return True, {
                "name": parts[0] if len(parts) > 0 else "Unknown",
                "vram": parts[1] if len(parts) > 1 else "Unknown",
                "driver": parts[2] if len(parts) > 2 else "Unknown",
                "cuda": parts[3] if len(parts) > 3 else "Unknown",
            }
        return False, None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, None


def check_ffmpeg():
    """Check FFmpeg availability."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            return True, version
        return False, None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, None


def check_nvenc():
    """Check NVENC encoder availability in FFmpeg."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        encoders = {}
        if "h264_nvenc" in result.stdout:
            encoders["h264_nvenc"] = True
        if "hevc_nvenc" in result.stdout:
            encoders["hevc_nvenc"] = True
        return len(encoders) > 0, encoders
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, {}


def check_nvdec():
    """Check NVDEC decoder availability in FFmpeg."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-decoders"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        decoders = {}
        if "h264_cuvid" in result.stdout:
            decoders["h264_cuvid"] = True
        if "hevc_cuvid" in result.stdout:
            decoders["hevc_cuvid"] = True
        return len(decoders) > 0, decoders
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, {}


def check_opencv_cuda():
    """Check OpenCV CUDA support."""
    try:
        import cv2
        cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
        if cuda_count > 0:
            return True, f"{cuda_count} CUDA device(s)"
        return False, "No CUDA devices found"
    except ImportError:
        return False, "OpenCV not installed"
    except AttributeError:
        return False, "OpenCV built without CUDA"
    except Exception as e:
        return False, str(e)


def check_python_packages():
    """Check required Python packages."""
    packages = {}

    try:
        import numpy
        packages["numpy"] = numpy.__version__
    except ImportError:
        packages["numpy"] = None

    try:
        import cv2
        packages["opencv"] = cv2.__version__
    except ImportError:
        packages["opencv"] = None

    try:
        import librosa
        packages["librosa"] = librosa.__version__
    except ImportError:
        packages["librosa"] = None

    try:
        import PIL
        packages["pillow"] = PIL.__version__
    except ImportError:
        packages["pillow"] = None

    try:
        import yaml
        packages["pyyaml"] = yaml.__version__
    except ImportError:
        packages["pyyaml"] = None

    return packages


def run_nvenc_test():
    """Run a quick NVENC encoding test."""
    import tempfile
    import os

    # Create a simple test pattern
    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate 10 test frames
        try:
            import numpy as np
            import cv2

            for i in range(10):
                img = np.zeros((1080, 1920, 3), dtype=np.uint8)
                cv2.putText(img, f"Frame {i}", (800, 540), cv2.FONT_HERSHEY_SIMPLEX,
                           3, (255, 255, 255), 5)
                cv2.imwrite(os.path.join(tmpdir, f"frame_{i:06d}.png"), img)

            output = os.path.join(tmpdir, "test.mp4")

            # Test NVENC encoding
            cmd = [
                "ffmpeg", "-y",
                "-framerate", "30",
                "-i", os.path.join(tmpdir, "frame_%06d.png"),
                "-c:v", "h264_nvenc",
                "-preset", "p4",
                "-cq", "20",
                output,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0 and os.path.exists(output):
                size = os.path.getsize(output)
                return True, f"Test file: {size} bytes"
            return False, result.stderr[:200] if result.stderr else "Unknown error"

        except Exception as e:
            return False, str(e)


def main():
    print("\n" + "="*60)
    print("  CUTTER GPU CAPABILITY CHECK")
    print("  Optimized for NVIDIA RTX 4090")
    print("="*60)

    # Check NVIDIA GPU
    print_header("NVIDIA GPU Detection")
    gpu_ok, gpu_info = check_nvidia_smi()
    print_status("NVIDIA GPU", gpu_ok)
    if gpu_info:
        print(f"      GPU: {gpu_info['name']}")
        print(f"      VRAM: {gpu_info['vram']}")
        print(f"      Driver: {gpu_info['driver']}")
        print(f"      CUDA: {gpu_info['cuda']}")

    # Check FFmpeg
    print_header("FFmpeg")
    ffmpeg_ok, ffmpeg_version = check_ffmpeg()
    print_status("FFmpeg installed", ffmpeg_ok, ffmpeg_version)

    # Check NVENC
    nvenc_ok, nvenc_encoders = check_nvenc()
    print_status("NVENC encoders", nvenc_ok)
    if nvenc_encoders:
        for enc in nvenc_encoders:
            print(f"      - {enc}")

    # Check NVDEC
    nvdec_ok, nvdec_decoders = check_nvdec()
    print_status("NVDEC decoders", nvdec_ok)
    if nvdec_decoders:
        for dec in nvdec_decoders:
            print(f"      - {dec}")

    # Check OpenCV CUDA
    print_header("OpenCV CUDA")
    cv_cuda_ok, cv_cuda_info = check_opencv_cuda()
    print_status("OpenCV CUDA", cv_cuda_ok, cv_cuda_info)

    # Check Python packages
    print_header("Python Packages")
    packages = check_python_packages()
    for pkg, version in packages.items():
        print_status(pkg, version is not None, version or "Not installed")

    # Run NVENC test
    if nvenc_ok:
        print_header("NVENC Encoding Test")
        test_ok, test_info = run_nvenc_test()
        print_status("NVENC encoding test", test_ok, test_info)

    # Summary
    print_header("Summary")

    all_ok = gpu_ok and ffmpeg_ok and nvenc_ok

    if all_ok:
        print("\n  \033[92mGPU acceleration is fully configured!\033[0m")
        print("\n  Your RTX 4090 will be used for:")
        print("    - NVENC H.264/HEVC encoding (10-20x faster)")
        print("    - NVDEC hardware video decoding")
        if cv_cuda_ok:
            print("    - CUDA-accelerated image processing")
        print("\n  Expected encoding speed: 200-400+ FPS for 1080p")
        print("  Expected encoding speed: 80-150+ FPS for 4K")
    else:
        print("\n  \033[93mSome GPU features are not available:\033[0m")
        if not gpu_ok:
            print("    - Install NVIDIA drivers: https://www.nvidia.com/drivers")
        if not ffmpeg_ok:
            print("    - Install FFmpeg: sudo apt install ffmpeg")
        if not nvenc_ok:
            print("    - Reinstall FFmpeg with NVENC support")
            print("      Ubuntu: sudo apt install ffmpeg")
            print("      Or build from source with --enable-nvenc")

    # Missing packages
    missing = [pkg for pkg, ver in packages.items() if ver is None]
    if missing:
        print(f"\n  Missing Python packages: {', '.join(missing)}")
        print(f"  Install with: pip install {' '.join(missing)}")

    print("\n" + "="*60 + "\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
