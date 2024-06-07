# -*- coding: utf-8 -*-
import itertools
import os
import logging

import pandas as pd

from ts_benchmark.common.constant import ROOT_PATH
from ts_benchmark.report.utils import write_log_file
from ts_benchmark.utils.get_file_name import get_log_file_name

# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_log(
    save_path,
    result_df: pd.DataFrame,
    model_name: str,
    compress_method: str = "gz",
) -> str:
    """
    Save log data.
    Save the evaluation results, model hyperparameters, model evaluation configuration, and model name to a log file.
    :param save_path: saved path.
    :param result_df: dataframe of the evaluation result.
    :param model_name: model name
    :param compress_method: the compression method for the output file.
    """

    if result_df["log_info"].any():
        error_itr = filter(None, result_df["log_info"])
        for error in itertools.islice(error_itr, 3):
            logger.info(error)
        if any(error_itr):
            logger.info(
                "-------------More error messages can be found in the record files!-------------"
            )

    if save_path is not None:
        result_path = os.path.join(ROOT_PATH, "result", save_path)
    else:
        result_path = os.path.join(ROOT_PATH, "result")
    os.makedirs(result_path, exist_ok=True)

    log_filename = get_log_file_name()
    file_path = os.path.join(result_path, model_name + log_filename)

    return write_log_file(result_df, file_path, compress_method)
