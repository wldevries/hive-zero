"""Tests for shared.lr_scheduler."""

import pytest
from shared.lr_scheduler import LRScheduler, parse_lr_schedule, lr_scheduler_from_string


class TestParseSchedule:
    def test_none(self):
        mode, schedule = parse_lr_schedule(None)
        assert mode == "linear"
        assert schedule is None

    def test_no_prefix_defaults_linear(self):
        mode, schedule = parse_lr_schedule("0:0.02,100:0.005")
        assert mode == "linear"
        assert schedule == [(0, 0.02), (100, 0.005)]

    def test_step_prefix(self):
        mode, schedule = parse_lr_schedule("s:0:0.02,30:0.01")
        assert mode == "step"
        assert schedule == [(0, 0.02), (30, 0.01)]

    def test_linear_prefix(self):
        mode, schedule = parse_lr_schedule("l:0:0.1,50:0.01")
        assert mode == "linear"
        assert schedule == [(0, 0.1), (50, 0.01)]

    def test_sorts_waypoints(self):
        _, schedule = parse_lr_schedule("50:0.01,0:0.1")
        assert schedule == [(0, 0.1), (50, 0.01)]


class TestLinearMode:
    @pytest.fixture()
    def sched(self):
        return LRScheduler([(0, 0.02), (100, 0.002)], mode="linear")

    def test_at_start(self, sched):
        assert sched.get_scheduled_lr(0) == 0.02

    def test_at_end(self, sched):
        assert sched.get_scheduled_lr(100) == 0.002

    def test_midpoint(self, sched):
        assert sched.get_scheduled_lr(50) == pytest.approx(0.011)

    def test_before_start_clamps(self, sched):
        assert sched.get_scheduled_lr(-5) == 0.02

    def test_after_end_clamps(self, sched):
        assert sched.get_scheduled_lr(200) == 0.002

    def test_three_waypoints(self):
        sched = LRScheduler([(0, 0.1), (50, 0.05), (100, 0.01)], mode="linear")
        assert sched.get_scheduled_lr(25) == pytest.approx(0.075)
        assert sched.get_scheduled_lr(75) == pytest.approx(0.03)


class TestStepMode:
    @pytest.fixture()
    def sched(self):
        return LRScheduler([(0, 0.02), (30, 0.01), (60, 0.005)], mode="step")

    def test_at_first_waypoint(self, sched):
        assert sched.get_scheduled_lr(0) == 0.02

    def test_holds_before_step(self, sched):
        assert sched.get_scheduled_lr(29) == 0.02

    def test_steps_at_waypoint(self, sched):
        assert sched.get_scheduled_lr(30) == 0.01

    def test_holds_second_segment(self, sched):
        assert sched.get_scheduled_lr(59) == 0.01

    def test_at_last_waypoint(self, sched):
        assert sched.get_scheduled_lr(60) == 0.005

    def test_after_end_clamps(self, sched):
        assert sched.get_scheduled_lr(999) == 0.005


class TestFromString:
    def test_none_returns_none(self):
        assert lr_scheduler_from_string(None) is not None
        assert lr_scheduler_from_string(None).get_scheduled_lr(0) is None

    def test_step_roundtrip(self):
        sched = lr_scheduler_from_string("s:0:0.02,30:0.01")
        assert sched.get_scheduled_lr(15) == 0.02
        assert sched.get_scheduled_lr(30) == 0.01

    def test_linear_roundtrip(self):
        sched = lr_scheduler_from_string("0:0.1,100:0.01")
        assert sched.get_scheduled_lr(50) == pytest.approx(0.055)
