# -*- coding: utf-8 -*-
import abc
from typing import Optional

import pandas as pd


class DataPoolImpl(metaclass=abc.ABCMeta):
    """
    Base class for data pool implementations

    This class acts as a data client in each worker that feeds data to the pipeline.
    Techniques such as local caching may be adopted to improve performance.
    """

    @abc.abstractmethod
    def get_series(self, name: str) -> Optional[pd.DataFrame]:
        """
        Gets time series data by name

        The returned DataFrame follows the OTB protocol.

        :param name: The name of the series to get.
        :return: Time series data in DataFrame format. If the time series is not available,
            return None.
        """

    @abc.abstractmethod
    def get_series_meta_info(self, name: str) -> Optional[pd.Series]:
        """
        Gets the meta information of time series by name

        We do not return the meta information of unexisting series even if
        the meta information itself is available.

        :param name: The name of the series to get.
        :return: Meta information data in Series format. If the meta information or the
            corresponding series is not available, return None.
        """
