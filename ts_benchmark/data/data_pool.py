# -*- coding: utf-8 -*-
from typing import NoReturn

from ts_benchmark.data.data_pool_impl_base import DataPoolImpl
from ts_benchmark.utils.design_pattern import Singleton


class DataPool(metaclass=Singleton):
    """
    The global interface of data pools
    """

    def __init__(self):
        self.pool = None

    def set_pool(self, pool: DataPoolImpl) -> NoReturn:
        """
        Set the global data pool object

        :param pool: a DataPoolImpl object.
        """
        self.pool = pool

    def get_pool(self) -> DataPoolImpl:
        """
        Get the global data pool object
        """
        return self.pool
