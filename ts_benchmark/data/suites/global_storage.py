# -*- coding: utf-8 -*-
import importlib
import logging
from typing import NoReturn, Optional, Dict

import pandas as pd

from ts_benchmark.data.data_server_base import DataServer
from ts_benchmark.data.data_pool_impl_base import DataPoolImpl
from ts_benchmark.data.data_pool import DataPool
from ts_benchmark.data.data_source import DataSource
from ts_benchmark.utils.parallel import ParallelBackend, SharedStorage


logger = logging.getLogger(__name__)


class GlobalStorageDataServer(DataServer):
    """
    A fake data server that share all available data at once

    Users must ensure all target data are loaded in the data source
    before starting this server.
    """

    def __init__(self, data_src: DataSource, backend: ParallelBackend) -> NoReturn:
        """
        Initializer

        :param data_src: A DataSource object where the data is read from.
        :param backend: A parallel backend that manages the global storage.
        """
        self.data_src = data_src
        self.backend = backend

    def start_async(self) -> NoReturn:
        """
        make the current data shareable across multiple workers (if exists)
        """
        logger.info("Data server starting...")
        logger.info("Start sending data to the global storage.")
        storage = self.backend.shared_storage
        storage.put("dataset_state", self.data_src.dataset.get_state())
        storage.put("dataset_class_module", self.data_src.DATASET_CLASS.__module__)
        storage.put("dataset_class_name", self.data_src.DATASET_CLASS.__name__)
        logger.info("Notifying all workers to sync data from the global storage.")
        self.backend.add_worker_initializer(sync_data)
        self.backend.execute_on_workers(sync_data)
        logger.info("Data server started.")


class GlobalStorageDataPool(DataPoolImpl):
    """
    A simple DataPool that retrieves all data from the globals storage at once
    """

    def __init__(self, storage: SharedStorage):
        """
        Initializer

        :param storage: The global storage object where data is stored.
        """
        self.storage = storage
        self._global_dataset = None

    def get_series(self, name: str) -> Optional[pd.DataFrame]:
        """
        Gets the time series from the global storage

        NOTE the data in the global storage is considered READ-ONLY,
        DO NOT perform inplace operations on the return value.

        :param name: The name of the series to get.
        :return: A time series in DataFrame format.
        """
        if self._global_dataset is None:
            raise ValueError("Data is not ready.")
        return self._global_dataset.get_series(name)

    def get_series_meta_info(self, name: str) -> Optional[pd.Series]:
        """
        Gets the time series meta-info from the global storage

        NOTE the data in the global storage is considered READ-ONLY,
        DO NOT perform inplace operations on the return value.

        :param name: The name of the series to get.
        :return: The meta-info of a time series in Series format.
        """
        if self._global_dataset is None:
            raise ValueError("Data is not ready.")
        return self._global_dataset.get_series_meta_info(name)

    def sync_data(self) -> NoReturn:
        """
        Synchronizes data from the global storage
        """
        self._global_dataset = self._build_dataset()

    def _build_dataset(self) -> DataSource:
        """
        Builds a DataSet object with the data in the global storage

        :return: The reconstructed DataSource object.
        """
        dataset_class_module = self.storage.get("dataset_class_module")
        dataset_class_name = self.storage.get("dataset_class_name")
        dataset_class = getattr(importlib.import_module(dataset_class_module), dataset_class_name)
        dataset = dataset_class()
        dataset.set_state(self.storage.get("dataset_state"))
        return dataset


def sync_data(env: Dict) -> NoReturn:
    """
    Sets the global data pool for the worker and synchronizes data from the global storage

    :param env: The environment dictionary of the parallel backend, which contains the handle
        to the global storage.
    """
    pool = GlobalStorageDataPool(env["storage"])
    pool.sync_data()
    DataPool().set_pool(pool)
