"""Live heatmap tests."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.geometry import Rect, square_rect
from src.visualization.heatmap import (GazeHeatmap, MODE_BOARD, MODE_BOTH,
                                       MODE_SCREEN, square_colour)

SCREEN = Rect(0, 0, 1920, 1080)
BOARD = Rect(400, 100, 800, 800)


@pytest.fixture()
def heatmap() -> GazeHeatmap:
    return GazeHeatmap(SCREEN, grid_width=96, sigma_px=55.0)


class TestAccumulation:
    def test_starts_empty(self, heatmap):
        assert heatmap.is_empty
        assert heatmap.total_samples == 0.0

    def test_a_sample_lands_in_the_right_place(self, heatmap):
        """The hotspot must correspond to where the user actually looked."""
        heatmap.add(1920 * 0.75, 1080 * 0.25)
        x, y = heatmap.peak_position()
        assert x == pytest.approx(1920 * 0.75, abs=30)
        assert y == pytest.approx(1080 * 0.25, abs=30)

    def test_repeated_looks_build_up(self, heatmap):
        for _ in range(20):
            heatmap.add(960, 540)
        first = heatmap.total_samples
        for _ in range(20):
            heatmap.add(960, 540)
        assert heatmap.total_samples > first
        assert not heatmap.is_empty

    def test_two_hotspots_are_both_visible(self, heatmap):
        for _ in range(30):
            heatmap.add(300, 300)
        for _ in range(30):
            heatmap.add(1600, 800)
        grid = heatmap.normalised_grid()
        left = grid[:, :heatmap.grid_width // 2].max()
        right = grid[:, heatmap.grid_width // 2:].max()
        assert left > 0.3 and right > 0.3

    def test_corner_samples_are_not_discarded(self, heatmap):
        """The Gaussian must clip against the edge, not be dropped."""
        heatmap.add(2, 2)
        assert heatmap.total_samples > 0
        heatmap.clear()
        heatmap.add(1918, 1078)
        assert heatmap.total_samples > 0

    def test_samples_outside_the_screen_do_not_crash(self, heatmap):
        heatmap.add(-5000, -5000)
        heatmap.add(99999, 99999)
        heatmap.add(float("nan"), 100)
        assert heatmap.total_samples >= 0.0

    def test_weighting_by_confidence(self, heatmap):
        heatmap.add(500, 500, weight=1.0)
        confident = heatmap.total_samples
        heatmap.clear()
        heatmap.add(500, 500, weight=0.2)
        assert heatmap.total_samples < confident

    def test_zero_weight_is_ignored(self, heatmap):
        heatmap.add(500, 500, weight=0.0)
        assert heatmap.is_empty

    def test_clear_resets_everything(self, heatmap):
        heatmap.add(500, 500)
        heatmap.add_square_time("e4", 2.0)
        heatmap.clear()
        assert heatmap.is_empty
        assert heatmap.square_seconds == {}


class TestBoardAccumulation:
    def test_square_time_adds_up(self, heatmap):
        for _ in range(10):
            heatmap.add_square_time("e4", 0.1)
        heatmap.add_square_time("d5", 0.4)
        assert heatmap.square_seconds["e4"] == pytest.approx(1.0)
        assert heatmap.square_seconds["d5"] == pytest.approx(0.4)

    def test_empty_square_is_ignored(self, heatmap):
        heatmap.add_square_time(None, 1.0)
        heatmap.add_square_time("", 1.0)
        heatmap.add_square_time("e4", 0.0)
        assert heatmap.square_seconds == {}

    def test_board_and_screen_maps_agree(self, heatmap):
        """Looking at e4 should light up e4 and the matching screen location."""
        centre = square_rect("e4", BOARD.x, BOARD.y, BOARD.width, BOARD.height).center
        for _ in range(30):
            heatmap.add(*centre)
            heatmap.add_square_time("e4", 0.05)

        assert max(heatmap.square_seconds, key=heatmap.square_seconds.get) == "e4"
        x, y = heatmap.peak_position()
        assert (x, y) == pytest.approx(centre, abs=30)


class TestDecay:
    def test_no_decay_by_default(self, heatmap):
        heatmap.add(500, 500, timestamp=0.0)
        before = heatmap.total_samples
        heatmap.add(500, 500, timestamp=600.0)
        assert heatmap.total_samples > before

    def test_half_life_fades_older_data(self):
        heatmap = GazeHeatmap(SCREEN, half_life_seconds=10.0)
        heatmap.add(500, 500, timestamp=0.0)
        heatmap.add_square_time("e4", 4.0)
        initial = heatmap.total_samples

        # 10 s later, with a negligible new sample, roughly half should remain.
        heatmap.add(1900, 1070, weight=1e-9, timestamp=10.0)
        assert heatmap.total_samples == pytest.approx(initial * 0.5, rel=0.05)
        assert heatmap.square_seconds["e4"] == pytest.approx(2.0, rel=0.05)

    def test_time_going_backwards_is_ignored(self):
        heatmap = GazeHeatmap(SCREEN, half_life_seconds=10.0)
        heatmap.add(500, 500, timestamp=100.0)
        total = heatmap.total_samples
        heatmap.add(500, 500, weight=1e-9, timestamp=50.0)
        assert heatmap.total_samples == pytest.approx(total, rel=1e-6)


class TestRendering:
    def test_normalisation_is_bounded(self, heatmap):
        for _ in range(200):
            heatmap.add(960, 540)
        grid = heatmap.normalised_grid()
        assert grid.min() >= 0.0 and grid.max() <= 1.0

    def test_a_single_hotspot_does_not_flatten_everything(self, heatmap):
        """Percentile normalisation keeps moderate areas visible."""
        for _ in range(500):
            heatmap.add(300, 300)
        for _ in range(20):
            heatmap.add(1600, 800)
        grid = heatmap.normalised_grid()
        assert grid[:, heatmap.grid_width // 2:].max() > 0.2

    def test_rgba_shape_and_transparency(self, heatmap):
        heatmap.add(960, 540)
        rgba = heatmap.to_rgba(opacity=0.6)
        assert rgba.shape == (heatmap.grid_height, heatmap.grid_width, 4)
        assert rgba.dtype == np.uint8
        # Untouched corners must be fully transparent, not tinted blue.
        assert rgba[0, 0, 3] == 0
        assert rgba[..., 3].max() > 0

    def test_empty_heatmap_renders_fully_transparent(self, heatmap):
        assert heatmap.to_rgba()[..., 3].max() == 0

    def test_opacity_scales_alpha(self, heatmap):
        for _ in range(50):
            heatmap.add(960, 540)
        assert heatmap.to_rgba(opacity=0.2)[..., 3].max() < \
            heatmap.to_rgba(opacity=0.9)[..., 3].max()

    def test_colour_gradient_runs_cold_to_hot(self):
        cold = square_colour(0.0)
        hot = square_colour(1.0)
        assert cold[2] > cold[0], "low values should be blue-dominant"
        assert hot[0] > hot[2], "high values should be red-dominant"

    def test_colour_input_is_clamped(self):
        assert square_colour(-5.0) == square_colour(0.0)
        assert square_colour(99.0) == square_colour(1.0)


class TestGeometryChanges:
    def test_resizing_rebuilds_the_grid(self, heatmap):
        heatmap.add(960, 540)
        heatmap.set_rect(Rect(0, 0, 1280, 720))
        assert heatmap.is_empty, "a monitor change must not keep stale data"
        assert heatmap.grid_height == pytest.approx(96 * 720 / 1280, abs=1)

    def test_a_second_monitor_offset_is_respected(self):
        heatmap = GazeHeatmap(Rect(1920, 0, 1920, 1080), grid_width=96)
        heatmap.add(1920 + 960, 540)
        x, _y = heatmap.peak_position()
        assert x == pytest.approx(1920 + 960, abs=30)

    def test_peak_is_none_when_empty(self):
        heatmap = GazeHeatmap(Rect(0, 0, 1920, 1080))
        assert heatmap.peak_cell() is None
        assert heatmap.peak_position() is None

    def test_dirty_flag_tracks_updates(self, heatmap):
        heatmap.mark_clean()
        assert not heatmap.dirty
        heatmap.add(500, 500)
        assert heatmap.dirty


class TestModes:
    def test_mode_constants(self):
        assert (MODE_SCREEN, MODE_BOARD, MODE_BOTH) == ("screen", "board", "both")
