# -*- coding: utf-8 -*-
import os

import pandas as pd

from ts_benchmark.common.constant import ROOT_PATH
from ts_benchmark.report.utils import write_log_file
from ts_benchmark.utils.get_file_name import get_log_file_name


def save_log(
    save_path,
    result_df: pd.DataFrame,
    model_name: str,
    compress_method: str = "gz",
) -> str:
    """
    保存日志数据。

    将评估结果、模型超参数、模型评估配置和模型名称保存到日志文件中。

    :param result_df: 评估结果的 DataFrame。
    :param model_name: 模型名称。
    :param compress_method: 输出文件压缩方式。
    """
    if save_path is not None:
        result_path = os.path.join(ROOT_PATH, "result", save_path)
    else:
        result_path = os.path.join(ROOT_PATH, "result")
    os.makedirs(result_path, exist_ok=True)

    log_filename = get_log_file_name()
    file_path = os.path.join(result_path, model_name + log_filename)
    # print("------------------------------2-------------------------------")
    # # result_df = pd.concat([result_df] * 600)
    # repeated_values = result_df.loc[0, 'actual_data'] * 2
    # repeated_values1 = result_df.loc[0, 'inference_data'] * 2
    #
    # # 更新DataFrame中该列的值
    # result_df['actual_data'] = repeated_values
    # result_df['inference_data'] = repeated_values1
    # print(result_df)
    #
    # print("------------------------------2.5-------------------------------")
    return write_log_file(result_df, file_path, compress_method)
