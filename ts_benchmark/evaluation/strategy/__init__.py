# -*- coding: utf-8 -*-
from ts_benchmark.evaluation.strategy.fixed_forecast import FixedForecast
from ts_benchmark.evaluation.strategy.anomaly_detect import FixedDetectScore, FixedDetectLabel, UnFixedDetectScore, \
    UnFixedDetectLabel, AllDetectScore, AllDetectLabel
from ts_benchmark.evaluation.strategy.rolling_forecast import RollingForecast

STRATEGY = {
    "fixed_forecast": FixedForecast,
    "rolling_forecast": RollingForecast,
    "fixed_detect_score": FixedDetectScore,
    "fixed_detect_label": FixedDetectLabel,
    "unfixed_detect_score": UnFixedDetectScore,
    "unfixed_detect_label": UnFixedDetectLabel,
    "all_detect_score": AllDetectScore,
    "all_detect_label": AllDetectLabel,
}