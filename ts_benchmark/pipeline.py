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
from ts_benchmark.recording import save_log
from ts_benchmark.utils.parallel import ParallelBackend


@dataclass
class DatasetInfo:
    # the possible values of the meta-info field 'size'
    size_value: List
    # the class of data source for this dataset
    datasrc_class: Type[DataSource]


PREDEFINED_DATASETS = {
    "large_forecast": DatasetInfo(
        size_value=["large", "small"],
        datasrc_class=LocalForecastingDataSource,
    ),
    "small_forecast": DatasetInfo(
        size_value=["small"], datasrc_class=LocalForecastingDataSource
    ),
    "user_forecast": DatasetInfo(
        size_value=["user"], datasrc_class=LocalForecastingDataSource
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


def _get_model_names(model_names: List[str]):
    """
    Rename models if there exists duplications.

    If a model A appears multiple times in the list, each appearance will be renamed to
    `A`, `A_1`, `A_2`, ...

    :param model_names: A list of model names.
    :return: The renamed list of model names.
    """
    s = pd.Series(model_names)
    cumulative_counts = s.groupby(s).cumcount()
    return [
        f"{model_name}_{cnt}" if cnt > 0 else model_name
        for model_name, cnt in zip(model_names, cumulative_counts)
    ]


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
    dataset_name_list = data_config.get("data_set_name", ["small_forecast"])
    if not dataset_name_list:
        dataset_name_list = ["small_forecast"]
    if isinstance(dataset_name_list, str):
        dataset_name_list = [dataset_name_list]
    for dataset_name in dataset_name_list:
        if dataset_name not in PREDEFINED_DATASETS:
            raise ValueError(f"Unknown dataset {dataset_name}.")

    data_src_type = PREDEFINED_DATASETS[dataset_name_list[0]].datasrc_class
    if not all(
        PREDEFINED_DATASETS[dataset_name].datasrc_class is data_src_type
        for dataset_name in dataset_name_list
    ):
        raise ValueError("Not supporting different types of data sources.")

    data_src: DataSource = PREDEFINED_DATASETS[dataset_name_list[0]].datasrc_class()
    data_name_list = data_config.get("data_name_list", None)
    if not data_name_list:
        data_name_list = []
        for dataset_name in dataset_name_list:
            size_value = PREDEFINED_DATASETS[dataset_name].size_value
            feature_dict = data_config.get("feature_dict", None)
            data_name_list.extend(
                filter_data(
                    data_src.dataset.metadata, size_value, feature_dict=feature_dict
                )
            )
    data_name_list = list(set(data_name_list))
    if not data_name_list:
        raise ValueError("No dataset specified.")
    data_src.load_series_list(data_name_list)
    data_server = GlobalStorageDataServer(data_src, ParallelBackend())
    data_server.start_async()

    # modeling
    model_factory_list = get_models(model_config)

    result_list = [
        eval_model(model_factory, data_name_list, evaluation_config)
        for model_factory in model_factory_list
    ]
    model_save_names = [
        it.split(".")[-1]
        for it in _get_model_names(
            [model_factory.model_name for model_factory in model_factory_list]
        )
    ]

    log_file_names = []
    for model_factory, result_itr, model_save_name in zip(
        model_factory_list, result_list, model_save_names
    ):
        for i, result_df in enumerate(result_itr.collect()):
            log_file_names.append(
                save_log(
                    result_df,
                    save_path,
                    model_save_name if i == 0 else f"{model_save_name}-{i}",
                )
            )

    return log_file_names
