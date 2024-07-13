# -*- coding: utf-8 -*-

from typing import Tuple

import pandas as pd


def split_before(data: pd.DataFrame, index: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into two parts at the specified index.

    :param data: Time series data to be segmented.
    :param index: Split index position.
    :return: Split the first and second half of the data.
    """
    return data.iloc[:index, :], data.iloc[index:, :]
