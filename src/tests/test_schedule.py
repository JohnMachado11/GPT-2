import math

import pytest

from gpt2.train.schedule import learning_rate_at


def schedule(step):
    # max 1.0, 10 warmup steps, 110 steps in total, decaying to a floor of 0.1
    return learning_rate_at(step, max_learning_rate=1.0, warmup_steps=10, total_steps=110, min_learning_rate=0.1)


def test_warmup_starts_small_and_climbs_in_a_straight_line_to_the_max():
    assert schedule(0) == pytest.approx(0.1)                 # max / warmup_steps -- not 0
    assert schedule(4) == pytest.approx(0.5)
    assert schedule(9) == pytest.approx(1.0)                 # the last warmup step reaches the max
    assert all(schedule(s) < schedule(s + 1) for s in range(9))


def test_decay_starts_at_the_max_right_after_warmup():
    assert schedule(10) == pytest.approx(1.0)


def test_decay_follows_a_cosine_not_a_straight_line():
    floor, peak = 0.1, 1.0
    cosine = lambda progress: floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * progress))
    assert schedule(35) == pytest.approx(cosine(0.25))       # a quarter of the way: 0.868, not linear's 0.775
    assert schedule(60) == pytest.approx((peak + floor) / 2)  # halfway: exactly the midpoint
    assert schedule(85) == pytest.approx(cosine(0.75))


def test_decay_never_goes_up_and_ends_at_the_floor():
    assert all(schedule(s) >= schedule(s + 1) for s in range(10, 200))
    assert schedule(110) == pytest.approx(0.1)
    assert schedule(500) == pytest.approx(0.1)               # past the end it stays at the floor


def test_no_warmup_starts_decaying_from_the_max_immediately():
    assert learning_rate_at(0, 1.0, 0, 100, 0.1) == pytest.approx(1.0)
