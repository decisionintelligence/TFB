# -*- coding: utf-8 -*-
from ts_benchmark.evaluation.metrics import regression_metrics

REGRESSION_METRICS = {
    k: getattr(regression_metrics, k) for k in regression_metrics.__all__
}
METRICS = {
    **REGRESSION_METRICS,
}