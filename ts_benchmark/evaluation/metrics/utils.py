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
    Automatically calculate the appropriate period length for time series data.

    :param data: Time series data.
    :return: Automatically calculated period length.
    """
    if len(data.shape) > 1:
        return 0

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
    Get a list of lengths of anomalous intervals in the time series labels.

    :param labels: List of anomaly label series, where 1 indicates anomalous and 0 indicates normal.
    :return: List of lengths of anomalous intervals.
    """
    end_pos = np.diff(np.array(labels, dtype=int), append=0) < 0
    return np.diff(np.cumsum(labels)[end_pos], prepend=0)

