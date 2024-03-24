# -*- coding: utf-8 -*-
from typing import List

from ts_benchmark.data_loader.data_pool import DataPool, filter_data
from ts_benchmark.evaluation.evaluate_model import eval_model
from ts_benchmark.models.get_model import get_model
from ts_benchmark.report.save_log import save_log
from ts_benchmark.utils.parallel import ParallelBackend


def pipeline(
    data_loader_config: dict, model_config: dict, model_eval_config: dict, save_path: str
) -> List[str]:
    """
    Execute the benchmark pipeline process, including loading data, building models, evaluating models, and generating reports.

    :param data_loader_config: Configuration for data loading.
    :param model_config: Configuration for model construction.
    :param model_eval_config: Configuration for model evaluation.
    :param save_path: The relative path for saving evaluation results, relative to the result folder.
    """

    # Obtain data pool instances and load data
    data_pool = DataPool()
    series_list = filter_data(data_loader_config)
    data_pool.prepare_data(series_list)
    data_pool.share_data(ParallelBackend().shared_storage)
    ParallelBackend().notify_data_shared()

    # modeling
    model_factory_list = get_model(model_config)

    # Loop through each model
    log_file_names = []
    for index, model_factory in enumerate(model_factory_list):
        # evaluation model
        for i, result_df in enumerate(
            eval_model(model_factory, series_list, model_eval_config)
        ):
            # Name of the model being evaluated
            model_name = model_config["models"][index]["model_name"].split(".")[-1]
            # Reporting
            log_file_names.append(
                save_log(
                    save_path,
                    result_df,
                    model_name if i == 0 else f"{model_name}_{i}",
                )
            )

    return log_file_names
