import numpy as np
import pytest

from a2c_research.metrics import (
    bootstrap_mean,
    curve_metrics,
    difference_ci,
    success_interval,
)


def test_auc_integrates_actual_uneven_timesteps():
    result = curve_metrics([0, 10, 100], [0, 500, 500], 100)
    assert result["auc"] == pytest.approx(0.95)
    assert result["auc"] != pytest.approx(2 / 3)


def test_confirmed_success_and_later_collapse():
    result = curve_metrics([0, 10, 20, 30, 40], [20, 480, 500, 100, 490], 40)
    assert result["solved"] == 1
    assert result["steps_to_threshold"] == 10
    assert result["confirmation_steps"] == 20
    assert result["post_solve_drawdown"] == 400
    assert result["post_solve_below_threshold_fraction"] == pytest.approx(1 / 3)
    assert result["right_censored"] == 0


def test_single_good_checkpoint_does_not_solve_and_is_censored():
    result = curve_metrics([0, 10, 20], [20, 500, 20], 20)
    assert result["solved"] == 0
    assert result["confirmation_steps"] is None
    assert result["restricted_confirmation_steps"] == 20
    assert result["right_censored"] == 1
    assert result["post_solve_drawdown"] is None


@pytest.mark.parametrize("t,y,budget", [
    ([0, 10, 10], [1, 2, 3], 10),
    ([0, 5], [1, 2], 10),
    ([1, 10], [1, 2], 10),
    ([0, 10], [1, np.nan], 10),
    ([0, 10], [1, 501], 10),
])
def test_invalid_curves_rejected(t, y, budget):
    with pytest.raises(ValueError):
        curve_metrics(t, y, budget)


def test_bootstrap_reproducible_and_single_run_has_no_interval():
    assert bootstrap_mean([0, 1, 4], 200, 42) == bootstrap_mean([0, 1, 4], 200, 42)
    single = bootstrap_mean([3], 100)
    assert single["mean"] == 3
    assert single["ci_low"] is None and single["std"] is None
    diff = difference_ci([5, 5], [2, 2], 100)
    assert diff == {"difference_a2c_minus_ppo": 3, "ci_low": 3, "ci_high": 3}


def test_all_successes_do_not_imply_certain_population_success():
    low, high = success_interval(10, 10)
    assert 0 < low < 1 and high == pytest.approx(1)
