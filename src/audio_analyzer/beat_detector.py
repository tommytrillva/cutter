"""Beat detection and audio analysis for Cutter."""
import logging
from typing import List, Tuple, Optional
import numpy as np

try:
    import librosa
except ImportError:
    librosa = None

from ..data_structures import BeatInfo, Sensitivity

logger = logging.getLogger(__name__)


class BeatDetector:
    """Detect beats and analyze audio energy."""

    # Sensitivity presets for beat detection
    SENSITIVITY_PARAMS = {
        Sensitivity.LOW: {
            "hop_length": 1024,
            "beat_strength_threshold": 0.6,
            "energy_peak_threshold": 0.7,
        },
        Sensitivity.MEDIUM: {
            "hop_length": 512,
            "beat_strength_threshold": 0.4,
            "energy_peak_threshold": 0.5,
        },
        Sensitivity.HIGH: {
            "hop_length": 256,
            "beat_strength_threshold": 0.2,
            "energy_peak_threshold": 0.3,
        },
    }

    def __init__(
        self,
        sensitivity: str = "medium",
        custom_threshold: Optional[float] = None,
    ):
        """Initialize beat detector.

        Args:
            sensitivity: Detection sensitivity - 'low', 'medium', or 'high'.
            custom_threshold: Override threshold (0.0-1.0).
        """
        if librosa is None:
            raise ImportError(
                "librosa is required for beat detection. "
                "Install with: pip install librosa"
            )

        self.sensitivity = Sensitivity(sensitivity.lower())
        self.params = self.SENSITIVITY_PARAMS[self.sensitivity].copy()

        if custom_threshold is not None:
            self.params["beat_strength_threshold"] = custom_threshold

    def detect_beats(
        self, audio: np.ndarray, sr: int
    ) -> BeatInfo:
        """Detect beats in audio.

        Args:
            audio: Audio samples as numpy array.
            sr: Sample rate.

        Returns:
            BeatInfo object with beat times, BPM, strengths, and energy peaks.
        """
        logger.info(f"Detecting beats with {self.sensitivity.value} sensitivity...")

        # Estimate BPM first
        bpm = self.estimate_bpm(audio, sr)
        logger.info(f"Estimated BPM: {bpm:.2f}")

        # Detect beat frames
        tempo, beat_frames = librosa.beat.beat_track(
            y=audio,
            sr=sr,
            hop_length=self.params["hop_length"],
            start_bpm=bpm,
            units="frames",
        )

        # Convert frames to times
        beat_times = librosa.frames_to_time(
            beat_frames,
            sr=sr,
            hop_length=self.params["hop_length"],
        ).tolist()

        # Calculate beat strengths (onset strength at each beat)
        onset_env = librosa.onset.onset_strength(
            y=audio,
            sr=sr,
            hop_length=self.params["hop_length"],
        )

        # Normalize onset envelope
        onset_env_norm = onset_env / (onset_env.max() + 1e-6)

        # Get strength at each beat frame
        beat_strength = []
        for frame in beat_frames:
            if frame < len(onset_env_norm):
                beat_strength.append(float(onset_env_norm[frame]))
            else:
                beat_strength.append(0.0)

        # Detect energy peaks
        energy_peaks = self._detect_energy_peaks(
            onset_env_norm, sr, self.params["hop_length"]
        )

        # Calculate energy curve for visualization/sync
        energy_curve = onset_env_norm.tolist()

        beat_info = BeatInfo(
            beat_times=beat_times,
            bpm=float(tempo) if np.isscalar(tempo) else float(tempo[0]),
            beat_strength=beat_strength,
            energy_peaks=energy_peaks,
            energy_curve=energy_curve,
        )

        logger.info(
            f"Detected {len(beat_times)} beats, "
            f"{len(energy_peaks)} energy peaks"
        )

        return beat_info

    def estimate_bpm(self, audio: np.ndarray, sr: int) -> float:
        """Estimate BPM of audio.

        Args:
            audio: Audio samples.
            sr: Sample rate.

        Returns:
            Estimated BPM.
        """
        # Use librosa's tempo estimation
        tempo, _ = librosa.beat.beat_track(
            y=audio, sr=sr, hop_length=self.params["hop_length"]
        )

        if np.isscalar(tempo):
            return float(tempo)
        return float(tempo[0])

    def calculate_energy(
        self, audio: np.ndarray, sr: int
    ) -> Tuple[np.ndarray, List[float]]:
        """Calculate energy envelope over time.

        Args:
            audio: Audio samples.
            sr: Sample rate.

        Returns:
            Tuple of (energy envelope array, time points).
        """
        # Calculate RMS energy
        frame_length = 2048
        hop_length = self.params["hop_length"]

        rms = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length,
        )[0]

        # Normalize
        rms_norm = rms / (rms.max() + 1e-6)

        # Get time points
        times = librosa.frames_to_time(
            np.arange(len(rms)),
            sr=sr,
            hop_length=hop_length,
        ).tolist()

        return rms_norm, times

    def _detect_energy_peaks(
        self,
        onset_env: np.ndarray,
        sr: int,
        hop_length: int,
    ) -> List[float]:
        """Detect energy peaks (drops, rises, high-energy moments).

        Args:
            onset_env: Onset strength envelope.
            sr: Sample rate.
            hop_length: Hop length used.

        Returns:
            List of peak times in seconds.
        """
        threshold = self.params["energy_peak_threshold"]

        # Find local maxima above threshold
        peaks = []
        for i in range(1, len(onset_env) - 1):
            if onset_env[i] > threshold:
                if onset_env[i] > onset_env[i - 1] and onset_env[i] > onset_env[i + 1]:
                    time = librosa.frames_to_time(
                        i, sr=sr, hop_length=hop_length
                    )
                    peaks.append(float(time))

        # Filter peaks that are too close together (< 0.25 seconds)
        filtered_peaks = []
        for peak in peaks:
            if not filtered_peaks or peak - filtered_peaks[-1] > 0.25:
                filtered_peaks.append(peak)

        return filtered_peaks

    def detect_drops_and_rises(
        self, audio: np.ndarray, sr: int
    ) -> Tuple[List[float], List[float]]:
        """Detect drops (sudden energy decreases) and rises.

        Args:
            audio: Audio samples.
            sr: Sample rate.

        Returns:
            Tuple of (drop times, rise times).
        """
        # Get energy envelope
        rms, times = self.calculate_energy(audio, sr)

        # Calculate energy gradient
        gradient = np.gradient(rms)

        drops = []
        rises = []

        threshold = 0.05  # Minimum change to detect

        for i in range(1, len(gradient)):
            time = times[i]
            if gradient[i] < -threshold and gradient[i - 1] >= -threshold:
                # Drop detected
                drops.append(time)
            elif gradient[i] > threshold and gradient[i - 1] <= threshold:
                # Rise detected
                rises.append(time)

        logger.debug(f"Detected {len(drops)} drops, {len(rises)} rises")
        return drops, rises

    def get_beat_at_time(
        self, beat_info: BeatInfo, time: float
    ) -> Tuple[int, float, float]:
        """Get nearest beat to a given time.

        Args:
            beat_info: BeatInfo from detect_beats.
            time: Time in seconds.

        Returns:
            Tuple of (beat_index, beat_time, distance_from_time).
        """
        return beat_info.get_nearest_beat(time)

    def quantize_to_beat(
        self,
        beat_info: BeatInfo,
        time: float,
        snap_threshold: float = 0.1,
    ) -> float:
        """Quantize a time to nearest beat.

        Args:
            beat_info: BeatInfo from detect_beats.
            time: Time to quantize.
            snap_threshold: Maximum distance to snap (seconds).

        Returns:
            Quantized time (original if no beat within threshold).
        """
        idx, beat_time, distance = beat_info.get_nearest_beat(time)

        if distance <= snap_threshold:
            return beat_time
        return time

    def get_strong_beats(
        self,
        beat_info: BeatInfo,
        strength_threshold: float = 0.6,
    ) -> List[Tuple[int, float, float]]:
        """Get beats above strength threshold.

        Args:
            beat_info: BeatInfo from detect_beats.
            strength_threshold: Minimum strength (0-1).

        Returns:
            List of (beat_index, time, strength) tuples.
        """
        strong_beats = []

        for i, (time, strength) in enumerate(
            zip(beat_info.beat_times, beat_info.beat_strength)
        ):
            if strength >= strength_threshold:
                strong_beats.append((i, time, strength))

        return strong_beats

    def analyze_sections(
        self, audio: np.ndarray, sr: int
    ) -> List[Tuple[float, float, str]]:
        """Detect musical sections (verse, chorus, bridge).

        Uses structural segmentation to find section boundaries.

        Args:
            audio: Audio samples.
            sr: Sample rate.

        Returns:
            List of (start_time, end_time, section_label) tuples.
        """
        # Compute chromagram
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)

        # Compute self-similarity matrix
        R = librosa.segment.recurrence_matrix(chroma, mode='affinity')

        # Find segment boundaries
        boundaries = librosa.segment.agglomerative(chroma, k=6)
        boundary_times = librosa.frames_to_time(boundaries, sr=sr).tolist()

        # Create section list
        sections = []
        labels = ["intro", "verse", "chorus", "verse", "chorus", "outro"]

        for i in range(len(boundary_times)):
            start = boundary_times[i]
            end = boundary_times[i + 1] if i + 1 < len(boundary_times) else (
                len(audio) / sr
            )
            label = labels[i] if i < len(labels) else f"section_{i}"
            sections.append((start, end, label))

        return sections
