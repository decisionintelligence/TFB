"""
@article{paparrizos2022volume,
  title={{Volume Under the Surface: A New Accuracy Evaluation Measure for Time-Series Anomaly Detection}},
  author={Paparrizos, John and Boniol, Paul and Palpanas, Themis and Tsay, Ruey S and Elmore, Aaron and Franklin, Michael J},
  journal={Proceedings of the VLDB Endowment},
  volume={15},
  number={11},
  pages={2774--2787},
  year={2022},
  publisher={VLDB Endowment}
}

"""

# -*- coding: utf-8 -*-
from typing import List

from statsmodels.tsa.stattools import acf
from scipy.signal import argrelextrema
import numpy as np


def find_length(data: np.ndarray) -> int:
    """
    自动计算时间序列数据的合适周期长度。

    :param data: 时间序列数据。
    :return: 自动计算的周期长度。
    """
    if len(data.shape) > 1:
        return 0

    # 取前 20000 个数据点进行计算
    data = data[: min(20000, len(data))]

    base = 3
    auto_corr = acf(data, nlags=400, fft=True)[base:]

    local_max = argrelextrema(auto_corr, np.greater)[0]
    try:
        max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
        if local_max[max_local_max] < 3 or local_max[max_local_max] > 300:
            return 125
        return local_max[max_local_max] + base
    except:
        return 125


def get_list_anomaly(labels: np.ndarray) -> List[int]:
    """
    获取时间序列标签中的异常间隔长度列表。

    :param labels: 时间序列标签列表，1 表示异常，0 表示正常。
    :return: 异常间隔长度列表。
    """
    # results = []
    # start = 0
    # anom = False
    # for i, val in enumerate(labels):
    #     if val == 1:
    #         anom = True
    #     else:
    #         if anom:
    #             results.append(i - start)
    #             anom = False
    #     if not anom:
    #         start = i
    # return results

    end_pos = np.diff(np.array(labels, dtype=int), append=0) < 0
    return np.diff(np.cumsum(labels)[end_pos], prepend=0)


# label = [1,1,1,0,0,0,1,1,0,0,0,1,1,1,0]
# print(get_list_anomaly(label))
