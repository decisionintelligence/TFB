import numpy as np
import pandas as pd
import pytest

from ts_benchmark.evaluation.strategy.rolling_forecast import (
    RollingForecastEvalBatchMaker,
    RollingForecastPredictBatchMaker,
)


@pytest.fixture
def series_and_exog():
    timestamps = pd.date_range("2024-01-01", periods=6, freq="h")
    series = pd.DataFrame({"target": np.arange(6)}, index=timestamps)
    exog = pd.DataFrame({"external": np.arange(100, 106)}, index=timestamps)
    return series, exog


@pytest.mark.parametrize("origins", [[2], [4, 2]])
def test_batch_rejects_insufficient_history(series_and_exog, origins):
    series, exog = series_and_exog
    maker = RollingForecastEvalBatchMaker(series, origins, {"exog": exog})
    predictor = RollingForecastPredictBatchMaker(maker)

    with pytest.raises(ValueError, match="starts before the beginning"):
        predictor.make_batch(batch_size=len(origins), win_size=3)

    # An invalid request must leave every origin available for a valid retry.
    batch = predictor.make_batch(batch_size=len(origins), win_size=2)
    np.testing.assert_array_equal(
        batch["input"], [[[origin - 2], [origin - 1]] for origin in origins]
    )
    assert not predictor.has_more_batches()


@pytest.mark.parametrize(
    ("origins", "win_size", "history"),
    [
        ([3], 3, [[0, 1, 2]]),
        ([4, 5], 3, [[1, 2, 3], [2, 3, 4]]),
        ([1], 1, [[0]]),
        ([5], 5, [[0, 1, 2, 3, 4]]),
    ],
)
def test_batch_preserves_history_alignment(series_and_exog, origins, win_size, history):
    series, exog = series_and_exog
    maker = RollingForecastEvalBatchMaker(series, origins, {"exog": exog})

    batch = RollingForecastPredictBatchMaker(maker).make_batch(len(origins), win_size)
    rows = np.array(history)
    np.testing.assert_array_equal(batch["input"], rows[..., None])
    np.testing.assert_array_equal(batch["time_stamps"], series.index.to_numpy()[rows])
    np.testing.assert_array_equal(batch["covariates"]["exog"], (100 + rows)[..., None])


def test_evaluation_windows_start_at_the_forecast_origin(series_and_exog):
    series, exog = series_and_exog
    maker = RollingForecastEvalBatchMaker(series, [3, 4], {"exog": exog})
    predictor = RollingForecastPredictBatchMaker(maker)
    predictor.make_batch(batch_size=2, win_size=3)

    evaluation = maker.make_batch_eval(horizon=2)
    np.testing.assert_array_equal(evaluation["target"], [[[3], [4]], [[4], [5]]])
    np.testing.assert_array_equal(
        evaluation["covariates"]["exog"], [[[103], [104]], [[104], [105]]]
    )


def test_no_origins_has_no_batches(series_and_exog):
    series, _ = series_and_exog
    maker = RollingForecastEvalBatchMaker(series, [])
    assert not RollingForecastPredictBatchMaker(maker).has_more_batches()


def test_window_larger_than_series_is_rejected(series_and_exog):
    series, _ = series_and_exog
    predictor = RollingForecastPredictBatchMaker(
        RollingForecastEvalBatchMaker(series, [len(series) - 1])
    )
    with pytest.raises(ValueError):
        predictor.make_batch(batch_size=1, win_size=len(series) + 1)
