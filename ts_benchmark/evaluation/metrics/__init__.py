# -*- coding: utf-8 -*-
from ts_benchmark.evaluation.metrics import regression_metrics
from ts_benchmark.evaluation.metrics import classification_metrics_score
from ts_benchmark.evaluation.metrics import classification_metrics_label

REGRESSION_METRICS = {
    k: getattr(regression_metrics, k) for k in regression_metrics.__all__
}
CLASSIFICATION_METRICS_SCORE = {
    k: getattr(classification_metrics_score, k)
    for k in classification_metrics_score.__all__
}
CLASSIFICATION_METRICS_LABEL = {
    k: getattr(classification_metrics_label, k)
    for k in classification_metrics_label.__all__
}
METRICS = {
    **REGRESSION_METRICS,
    **CLASSIFICATION_METRICS_SCORE,
    **CLASSIFICATION_METRICS_LABEL,
}