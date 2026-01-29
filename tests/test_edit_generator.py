"""Tests for edit generation."""
import pytest
import numpy as np

from src.edit_generator import ShotSelector, EditVariationGenerator
from src.edit_generator.timeline_builder import TimelineBuilder, EditBuilder
from src.data_structures import Shot, BeatInfo, Edit, Cut


class TestShotSelector:
    """Tests for ShotSelector class."""

    @pytest.fixture
    def sample_shots(self):
        """Create sample shots for testing."""
        return [
            Shot(0, 90, 0.0, 3.0, 0.9, 0.95, 45),
            Shot(100, 190, 3.33, 6.33, 0.7, 0.8, 145),
            Shot(200, 290, 6.67, 9.67, 0.5, 0.6, 245),
            Shot(300, 390, 10.0, 13.0, 0.85, 0.9, 345),
            Shot(400, 490, 13.33, 16.33, 0.3, 0.4, 445),
        ]

    @pytest.fixture
    def sample_beat_info(self):
        """Create sample beat info for testing."""
        return BeatInfo(
            beat_times=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            bpm=120.0,
            beat_strength=[0.8, 0.6, 0.8, 0.6, 0.8, 0.6, 0.8, 0.6, 0.8],
            energy_peaks=[1.0, 3.0],
        )

    def test_initialization(self):
        """Test ShotSelector initialization."""
        selector = ShotSelector()
        assert selector.min_quality == 0.4

        selector_strict = ShotSelector(min_quality=0.7)
        assert selector_strict.min_quality == 0.7

    def test_select_shots(self, sample_shots, sample_beat_info):
        """Test basic shot selection."""
        selector = ShotSelector(min_quality=0.5)
        selected = selector.select_shots(sample_shots, 10.0, sample_beat_info)

        assert len(selected) > 0
        for shot, start, end in selected:
            assert end > start
            assert end <= 10.0

    def test_quality_filtering(self, sample_shots, sample_beat_info):
        """Test that low quality shots are filtered."""
        selector = ShotSelector(min_quality=0.6)
        selected = selector.select_shots(sample_shots, 10.0, sample_beat_info)

        for shot, _, _ in selected:
            assert shot.avg_quality >= 0.6 or len([s for s in sample_shots if s.avg_quality >= 0.6]) == 0

    def test_duration_filling(self, sample_shots, sample_beat_info):
        """Test that selection fills target duration."""
        selector = ShotSelector(min_quality=0.3)
        target_duration = 5.0
        selected = selector.select_shots(sample_shots, target_duration, sample_beat_info)

        if selected:
            total_duration = sum(end - start for _, start, end in selected)
            # Should be close to target (within shot duration variance)
            assert total_duration <= target_duration + 0.5


class TestTimelineBuilder:
    """Tests for TimelineBuilder class."""

    @pytest.fixture
    def sample_selected_shots(self):
        """Create sample selected shots."""
        shots = [
            Shot(0, 90, 0.0, 3.0, 0.9, 0.95, 45),
            Shot(100, 190, 3.33, 6.33, 0.7, 0.8, 145),
        ]
        return [
            (shots[0], 0.0, 2.0),
            (shots[1], 2.0, 4.0),
        ]

    @pytest.fixture
    def sample_beat_info(self):
        """Create sample beat info."""
        return BeatInfo(
            beat_times=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            bpm=120.0,
            beat_strength=[0.8, 0.6, 0.8, 0.6, 0.8, 0.6, 0.8, 0.6, 0.8],
            energy_peaks=[2.0],
        )

    def test_initialization(self):
        """Test TimelineBuilder initialization."""
        builder = TimelineBuilder()
        assert builder.min_cut_duration == 0.5
        assert builder.max_cut_duration == 5.0

    def test_build_timeline(self, sample_selected_shots, sample_beat_info):
        """Test building timeline from selected shots."""
        builder = TimelineBuilder()
        cuts = builder.build_timeline(sample_selected_shots, sample_beat_info, 4.0)

        assert len(cuts) == 2
        for cut in cuts:
            assert isinstance(cut, Cut)
            assert cut.timeline_end > cut.timeline_start

    def test_align_to_beats(self, sample_beat_info):
        """Test cut alignment to beats."""
        builder = TimelineBuilder()

        cuts = [
            Cut(0, 2, 0.02, 2.02, 0.8, "beat", 0, 0),  # Slightly off beat
            Cut(2, 4, 2.02, 4.02, 0.7, "beat", 4, 1),
        ]

        aligned = builder.align_to_beats(cuts, sample_beat_info, snap_threshold=0.1)

        # First cut should snap to 0.0
        assert aligned[0].timeline_start == 0.0

    def test_create_transitions(self, sample_beat_info):
        """Test transition creation."""
        builder = TimelineBuilder(transition_style="geometric")

        cuts = [
            Cut(0, 2, 0.0, 2.0, 0.8, "beat", 0, 0),
            Cut(2, 4, 2.0, 4.0, 0.7, "beat", 4, 1),
        ]

        transitions = builder.create_transitions(cuts, sample_beat_info)

        assert len(transitions) == 1
        assert transitions[0].duration > 0


class TestEditVariationGenerator:
    """Tests for EditVariationGenerator class."""

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration."""
        return {
            "project_name": "test",
            "cut_timing": {
                "min_cut_duration": 0.5,
                "max_cut_duration": 5.0,
            },
            "motion_graphics": {
                "transitions": {"style": "dynamic"},
                "particles": {"density": "medium"},
                "text": {"enabled": False},
            },
            "color_grading": {"enabled": True, "preset_profile": "cinematic"},
        }

    @pytest.fixture
    def sample_shots(self):
        """Create sample shots."""
        return [
            Shot(0, 90, 0.0, 3.0, 0.9, 0.95, 45),
            Shot(100, 190, 3.33, 6.33, 0.7, 0.8, 145),
            Shot(200, 290, 6.67, 9.67, 0.6, 0.7, 245),
            Shot(300, 390, 10.0, 13.0, 0.85, 0.9, 345),
        ]

    @pytest.fixture
    def sample_beat_info(self):
        """Create sample beat info."""
        return BeatInfo(
            beat_times=list(np.arange(0, 10, 0.5)),
            bpm=120.0,
            beat_strength=[0.7] * 20,
            energy_peaks=[2.0, 5.0, 8.0],
        )

    def test_initialization(self, sample_config):
        """Test EditVariationGenerator initialization."""
        generator = EditVariationGenerator(sample_config, num_variations=3)
        assert generator.num_variations == 3

    def test_generate_variations(self, sample_config, sample_shots, sample_beat_info):
        """Test generating multiple variations."""
        generator = EditVariationGenerator(sample_config, num_variations=3)
        variations = generator.generate_variations(sample_shots, sample_beat_info, 8.0)

        assert len(variations) == 3
        for edit in variations:
            assert isinstance(edit, Edit)
            assert edit.total_duration > 0

    def test_different_sensitivities(self, sample_config, sample_shots, sample_beat_info):
        """Test that different sensitivities produce different results."""
        generator = EditVariationGenerator(
            sample_config,
            num_variations=3,
            sensitivities=["low", "medium", "high"],
        )

        variations = generator.generate_variations(sample_shots, sample_beat_info, 8.0)

        # Check that variations have different parameters
        params = [v.generation_params.get("sensitivity") for v in variations]
        assert "low" in params
        assert "medium" in params
        assert "high" in params

    def test_custom_variation(self, sample_config, sample_shots, sample_beat_info):
        """Test generating custom variation."""
        generator = EditVariationGenerator(sample_config, num_variations=1)

        custom_params = {
            "cut_timing": {"min_cut_duration": 1.0},
            "min_quality": 0.5,
        }

        edit = generator.generate_custom_variation(
            sample_shots, sample_beat_info, 8.0, custom_params
        )

        assert isinstance(edit, Edit)
