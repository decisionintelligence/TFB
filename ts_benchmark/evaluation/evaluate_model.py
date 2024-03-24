# -*- coding: utf-8 -*-
import functools
import json
import logging
import traceback
from typing import Callable, Tuple, List, Generator

import pandas as pd
import tqdm

from ts_benchmark.evaluation.evaluator import Evaluator
from ts_benchmark.evaluation.strategy import STRATEGY
from ts_benchmark.evaluation.strategy.constants import FieldNames
from ts_benchmark.evaluation.strategy.strategy import Strategy
from ts_benchmark.models.get_model import ModelFactory
from ts_benchmark.utils.parallel import ParallelBackend

logger = logging.getLogger(__name__)


def _safe_execute(fn: Callable, args: Tuple, get_default_result: Callable):
    """
    make sure execution does not crash even if there are exceptions
    """
    try:
        return fn(*args)
    except Exception as e:
        log = f"{traceback.format_exc()}\n{e}"
        return get_default_result(**{FieldNames.LOG_INFO: log})


def eval_model(
    model_factory: ModelFactory, series_list: list, model_eval_config: dict
) -> Generator[pd.DataFrame, None, None]:
    """
    Evaluate the performance of the model on time series data.
    Evaluate the model based on the provided model factory, time series list, and evaluation configuration, and return the DataFrame of the evaluation results.

    :param model_factory: Model factory object used to create model instances.
    :param series_list: A list containing time series names.
    :param model_eval_config: Evaluate configuration information, including strategies, evaluation metrics, etc.
    :return: The DataFrame containing the evaluation results.
    """
    # 获取策略类
    strategy_class = STRATEGY.get(model_eval_config["strategy_args"]["strategy_name"])
    if strategy_class is None:
        raise RuntimeError("strategy_class is none")

    # 解析评价指标配置
    metric = model_eval_config["metrics"]
    if metric == "all":
        metric = list(strategy_class.accepted_metrics())
    elif isinstance(metric, (str, dict)):
        metric = [metric]

    metric = [
        {"name": metric_info} if isinstance(metric_info, str) else metric_info
        for metric_info in metric
    ]

    # Check if the evaluation indicators are legal
    invalid_metrics = [
        m.get("name")
        for m in metric
        if m.get("name") not in strategy_class.accepted_metrics()
    ]
    if invalid_metrics:
        raise RuntimeError(
            "The evaluation index to be evaluated does not exist: {}".format(
                invalid_metrics
            )
        )

    # Create an evaluator instance
    evaluator = Evaluator(metric)

    strategy = strategy_class(model_eval_config["strategy_args"], evaluator)  # 创建评估策略对象

    eval_backend = ParallelBackend()
    result_list = []
    for series_name in tqdm.tqdm(series_list, desc="scheduling..."):
        # TODO: refactor data model to optimize communication cost in parallel mode
        result_list.append(
            eval_backend.schedule(strategy.execute, (series_name, model_factory))
        )
        # result_list.append(single_series_results)

    collector = strategy.get_collector()
    for i, result in enumerate(tqdm.tqdm(result_list, desc="collecting...")):
        collector.add(
            _safe_execute(
                result.result,
                (),
                functools.partial(
                    strategy.get_default_result,
                    **{FieldNames.FILE_NAME: series_list[i]},
                ),
            )
        )
        if collector.get_size() > 100000:
            result_df = build_result_df(collector.collect(), model_factory, strategy)
            yield result_df
            collector.reset()

    if collector.get_size() > 0:
        result_df = build_result_df(collector.collect(), model_factory, strategy)
        yield result_df


def build_result_df(
    result_list: List, model_factory: ModelFactory, strategy: Strategy
) -> pd.DataFrame:
    result_df = pd.DataFrame(result_list, columns=strategy.field_names)
    if FieldNames.MODEL_PARAMS not in result_df.columns:
        # allow models to do hyper-param search and return independent model_params for each search
        result_df.insert(
            0,
            FieldNames.MODEL_PARAMS,
            json.dumps(model_factory.model_hyper_params, sort_keys=True),
        )
    result_df.insert(0, FieldNames.STRATEGY_ARGS, strategy.get_config_str())
    result_df.insert(0, FieldNames.MODEL_NAME, model_factory.model_name)

    missing_fields = set(FieldNames.all_fields()) - set(result_df.columns)
    if missing_fields:
        raise ValueError(
            "These required fields are missing in the result df: {}".format(
                missing_fields
            )
        )
    return result_df
