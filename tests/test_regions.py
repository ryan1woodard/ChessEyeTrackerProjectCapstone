"""Region manager and gaze classifier tests."""

from __future__ import annotations

import pytest

from src.screen.classifier import GazeRegionClassifier, OFF_SCREEN, UNKNOWN
from src.screen.regions import (KIND_BOARD, KIND_CLOCK, Region, RegionManager,
                                suggested_clock_regions)
from src.utils.geometry import BLACK_BOTTOM, Rect

SCREEN = Rect(0, 0, 1920, 1080)
BOARD = Rect(400, 100, 800, 800)


@pytest.fixture()
def manager() -> RegionManager:
    regions = RegionManager()
    regions.set_board(BOARD)
    regions.add(Region("White Clock", 1250, 700, 200, 60, priority=50, kind=KIND_CLOCK))
    regions.add(Region("Side Panel", 1220, 100, 400, 800, priority=10))
    return regions


@pytest.fixture()
def classifier(manager: RegionManager) -> GazeRegionClassifier:
    return GazeRegionClassifier(manager, SCREEN, off_screen_margin=60)


class TestRegionManager:
    def test_board_is_registered_with_top_priority(self, manager):
        board = manager.board
        assert board is not None and board.kind == KIND_BOARD
        assert board.priority == 100

    def test_higher_priority_wins_when_regions_overlap(self, manager):
        # The clock sits inside the side panel.
        assert manager.classify(1300, 720).name == "White Clock"
        assert manager.classify(1300, 300).name == "Side Panel"

    def test_smaller_region_wins_on_a_priority_tie(self):
        regions = RegionManager()
        regions.add(Region("Big", 0, 0, 500, 500, priority=5))
        regions.add(Region("Small", 100, 100, 50, 50, priority=5))
        assert regions.classify(120, 120).name == "Small"

    def test_disabled_regions_are_skipped(self, manager):
        manager.get("White Clock").enabled = False
        assert manager.classify(1300, 720).name == "Side Panel"

    def test_points_outside_every_region(self, manager):
        assert manager.classify(50, 50) is None

    def test_setting_the_board_replaces_the_previous_one(self, manager):
        manager.set_board(Rect(0, 0, 100, 100))
        assert len([r for r in manager.regions if r.kind == KIND_BOARD]) == 1
        assert manager.board.rect == Rect(0, 0, 100, 100)

    def test_remove(self, manager):
        assert manager.remove("Side Panel") is True
        assert manager.remove("Side Panel") is False

    def test_persistence_round_trip(self, manager, tmp_path):
        path = tmp_path / "regions.json"
        manager.save(path)
        restored = RegionManager()
        assert restored.load(path) is True
        assert {r.name for r in restored.regions} == {r.name for r in manager.regions}
        assert restored.board.rect == BOARD

    def test_loading_a_missing_file_is_not_an_error(self, tmp_path):
        assert RegionManager().load(tmp_path / "nope.json") is False

    def test_loading_a_corrupt_file_is_handled(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        assert RegionManager().load(path) is False

    def test_suggested_clocks_sit_beside_the_board(self):
        clocks = suggested_clock_regions(BOARD, SCREEN)
        assert [c.name for c in clocks] == ["Black Clock", "White Clock"]
        for clock in clocks:
            assert clock.rect.left >= BOARD.right
            assert clock.kind == KIND_CLOCK


class TestClassifier:
    def test_board_hit_reports_the_square(self, classifier):
        result = classifier.classify(450, 150)
        assert result.on_board and result.region_name == RegionManager.BOARD_NAME
        assert result.square == "a8"

    def test_clock_hit(self, classifier):
        result = classifier.classify(1300, 720)
        assert result.region_name == "White Clock"
        assert result.kind == KIND_CLOCK
        assert result.square is None

    def test_unmapped_screen_area(self, classifier):
        assert classifier.classify(50, 1000).region_name == UNKNOWN

    def test_off_screen_beyond_the_margin(self, classifier):
        assert classifier.classify(-200, 500).region_name == OFF_SCREEN
        assert classifier.classify(500, 2000).on_screen is False

    def test_just_off_screen_still_counts_as_on_screen(self, classifier):
        """A little overshoot is estimation error, not looking away."""
        assert classifier.classify(-30, 500).on_screen is True

    def test_orientation_change_flips_squares(self, classifier):
        assert classifier.classify(450, 150).square == "a8"
        classifier.set_orientation(BLACK_BOTTOM)
        assert classifier.classify(450, 150).square == "h1"

    def test_hysteresis_holds_the_square_across_a_boundary(self, classifier):
        # 100 px squares; boundary between a8 and b8 at x = 500.
        assert classifier.classify(450, 150).square == "a8"
        assert classifier.classify(505, 150).square == "a8"   # inside the margin
        assert classifier.classify(560, 150).square == "b8"   # clearly past it

    def test_leaving_the_board_clears_the_remembered_square(self, classifier):
        classifier.classify(450, 150)
        classifier.classify(50, 1000)
        assert classifier.classify(505, 150).square == "b8"
