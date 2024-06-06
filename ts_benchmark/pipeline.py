# -*- coding: utf-8 -*-
from dataclasses import dataclass
from functools import reduce
from operator import and_
from typing import List, Dict, Type, Optional

import pandas as pd

from ts_benchmark.data.data_source import (
    LocalForecastingDataSource,
    DataSource,
)
from ts_benchmark.data.suites.global_storage import GlobalStorageDataServer
from ts_benchmark.evaluation.evaluate_model import eval_model
from ts_benchmark.models import get_models
from ts_benchmark.report.save_log import save_log
from ts_benchmark.utils.parallel import ParallelBackend


@dataclass
class DatasetInfo:
    # the possible values of the meta-info field 'size'
    size_value: List
    # the class of data source for this dataset
    datasrc_class: Type[DataSource]


PREDEFINED_DATASETS = {
    "large_forecast": DatasetInfo(
        size_value=["large", "medium", "small"],
        datasrc_class=LocalForecastingDataSource,
    ),
    "medium_forecast": DatasetInfo(
        size_value=["medium", "small"], datasrc_class=LocalForecastingDataSource
    ),
    "small_forecast": DatasetInfo(
        size_value=["small"], datasrc_class=LocalForecastingDataSource
    ),
}


def filter_data(
    metadata: pd.DataFrame, size_value: List[str], feature_dict: Optional[Dict] = None
) -> List[str]:
    """
    Filters the dataset based on given filters

    :param metadata: The meta information DataFrame.
    :param size_value: The allowed values of the 'size' meta-info field.
    :param feature_dict: A dictionary of filters where each key is a meta-info field
        and the corresponding value is the field value to keep. If None is given,
        no extra filter is applied.
    :return:
    """
    # Remove items with a value of None in feature_dict
    feature_dict = {k: v for k, v in feature_dict.items() if v is not None}

    # Use the reduce and and_ functions to filter data file names that meet the criteria
    filt_metadata = metadata
    if feature_dict is not None:
        filt_metadata = metadata[
            reduce(and_, (metadata[k] == v for k, v in feature_dict.items()))
        ]
    filt_metadata = filt_metadata[filt_metadata["size"].isin(size_value)]

    return filt_metadata["file_name"].tolist()


def pipeline(
    data_config: dict,
    model_config: dict,
    evaluation_config: dict,
    save_path: str,
) -> List[str]:
    """
    Execute the benchmark pipeline process

    The pipline includes loading data, building models, evaluating models, and generating reports.

    :param data_config: Configuration for data loading.
    :param model_config: Configuration for model construction.
    :param evaluation_config: Configuration for model evaluation.
    :param save_path: The relative path for saving evaluation results, relative to the result folder.
    """
    # prepare data
    # TODO: move these code into the data module, after the pipeline interface is unified
    dataset_name = data_config.get("data_set_name", "small_forecast")
    if dataset_name not in PREDEFINED_DATASETS:
        raise ValueError(f"Unknown dataset {dataset_name}.")
    data_src: DataSource = PREDEFINED_DATASETS[dataset_name].datasrc_class()
    data_name_list = data_config.get("data_name_list", None)
    if not data_name_list:
        size_value = PREDEFINED_DATASETS[dataset_name].size_value
        feature_dict = data_config.get("feature_dict", None)
        data_name_list = filter_data(
            data_src.dataset.metadata, size_value, feature_dict=feature_dict
        )
    if not data_name_list:
        raise ValueError("No dataset specified.")
    data_src.load_series_list(data_name_list)
    data_server = GlobalStorageDataServer(data_src, ParallelBackend())
    data_server.start_async()

    # modeling
    model_factory_list = get_models(model_config)

    # Loop through each model
    log_file_names = []
    for index, model_factory in enumerate(model_factory_list):
        # evaluation model
        for i, result_df in enumerate(
            eval_model(model_factory, data_name_list, evaluation_config)
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
