# -*- coding: utf-8 -*-
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Union, NoReturn, List

import pandas as pd

from ts_benchmark.common.constant import FORECASTING_DATASET_PATH
from ts_benchmark.common.constant import ANOMALY_DETECT_DATASET_PATH
from ts_benchmark.common.constant import META_DETECTION_DATA_PATH
from ts_benchmark.common.constant import META_FORECAST_DATA_PATH
from ts_benchmark.utils.data_processing import read_data
from ts_benchmark.utils.design_pattern import Singleton
from ts_benchmark.utils.parallel import SharedStorage

from functools import reduce
from operator import and_


SIZE = {
    "large_forecast": ["large", "medium", "small"],
    "medium_forecast": ["medium", "small"],
    "small_forecast": ["small"],
    "large_detect": ["large", "medium", "small"],
    "medium_detect": ["medium", "small"],
    "small_detect": ["small"],
}


def filter_data(data_loader_config: dict) -> List[str]:
    """
    Load a list of data file names and filter file names based on configuration.

    :param data_loader_config: Configuration for data loading.
    :return: List of data file names that meet the filtering criteria.
    :raises RuntimeError: If feature_dict is None.
    """
    feature_dict = data_loader_config.get("feature_dict", None)

    typical_data_name_list = data_loader_config.get("data_name_list", None)
    if typical_data_name_list is not None:
        return typical_data_name_list

    if feature_dict is None:
        raise RuntimeError("feature_dict is None")

    # Remove items with a value of None in feature_dict
    feature_dict = {k: v for k, v in feature_dict.items() if v is not None}
    data_set_name = data_loader_config.get("data_set_name", "small_forecast")

    if data_set_name in [
        "large_forecast",
        "medium_forecast",
        "small_forecast",
    ]:
        META_DATA_PATH = META_FORECAST_DATA_PATH
    elif data_set_name in [
        "large_detect",
        "medium_detect",
        "small_detect",
    ]:
        META_DATA_PATH = META_DETECTION_DATA_PATH
    else:
        raise ValueError("Please enter the correct data_setname")

    data_meta = pd.read_csv(META_DATA_PATH)

    data_size = SIZE[data_set_name]
    # Use the reduce and and_ functions to filter data file names that meet the criteria
    data_name_list = data_meta[
        reduce(and_, (data_meta[k] == v for k, v in feature_dict.items()))
    ][data_meta["size"].isin(data_size)]["file_name"].tolist()

    return data_name_list


class DataPool(metaclass=Singleton):
    """
    The DataPool class is used to create a data pool and accelerate the reading of multiple data files.
    """

    _DATA_KEY = "file_name"

    def __init__(self):
        """
        Constructor, initialize DataPool instance.
        """
        self._forecast_data_meta = None
        self._detect_data_meta = None
        self.data_pool = {}  # Create a dictionary for storing data

    @property
    def forecast_data_meta(self) -> pd.DataFrame:
        if self._forecast_data_meta is None:
            self._forecast_data_meta = pd.read_csv(META_FORECAST_DATA_PATH)
            self._forecast_data_meta.set_index(self._DATA_KEY, drop=False, inplace=True)
        return self._forecast_data_meta

    @property
    def detect_data_meta(self) -> pd.DataFrame:
        if self._detect_data_meta is None:
            self._detect_data_meta = pd.read_csv(META_DETECTION_DATA_PATH)
            self._detect_data_meta.set_index(self._DATA_KEY, drop=False, inplace=True)
        return self._detect_data_meta

    def _load_meta_info(self, series_name: str) -> Union[pd.Series, None]:
        """
        Prepare metadata information for the specified series.

        :param series_name: The name of the series to search for metadata.
        :return: Pandas Series containing metadata information.
        :raises ValueError: If metadata information for the specified series name is not found.
        """
        if series_name in self.forecast_data_meta.index:
            return self.forecast_data_meta.loc[[series_name]]
        elif series_name in self.detect_data_meta.index:
            return self.detect_data_meta.loc[[series_name]]
        else:
            raise ValueError("do not have {}'s meta data".format(series_name))

    def prepare_data(self, list_of_files: list) -> None:
        """
        Load multiple data files in parallel to the data pool.

        :param list_of_files: A list of data files to be loaded.
        """
        self._forecast_data_meta = pd.read_csv(META_FORECAST_DATA_PATH)
        self._forecast_data_meta.set_index(self._DATA_KEY, drop=False, inplace=True)

        self._detect_data_meta = pd.read_csv(META_DETECTION_DATA_PATH)
        self._detect_data_meta.set_index(self._DATA_KEY, drop=False, inplace=True)

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._load_data, series_name)
                for series_name in list_of_files
            ]
            for future, series_name in zip(futures, list_of_files):
                self.data_pool[series_name] = future.result()

    def _load_data(self, series_name: str) -> tuple:
        """
        Load a single data file and return the file name and data.

        :param series_name: Data file name.
        :return: A binary containing the file name and metadata.
        """
        try:
            datafile_path = os.path.join(FORECASTING_DATASET_PATH, series_name)
            data = read_data(datafile_path)
        except:
            datafile_path = os.path.join(ANOMALY_DETECT_DATASET_PATH, series_name)
            data = read_data(datafile_path)
        return data, self._load_meta_info(series_name)

    def get_series(self, series_name: str) -> pd.DataFrame:
        """
        Retrieve data from the data pool based on the file name.

        :param series_name: Data file name.
        :return: Corresponding data.
        :raises ValueError: If the data file is not in the data pool.
        """
        if series_name not in self.data_pool:
            self.data_pool[series_name] = self._load_data(series_name)
        return self.data_pool[series_name][0]

    def get_series_meta_info(self, series_name: str) -> pd.DataFrame:
        """
        Retrieve data from the data pool based on the file name.

        :param series_name: Data file name.
        :return: The corresponding data meta information.
        :raises ValueError: If the data file is not in the data pool.
        """
        if series_name not in self.data_pool:
            self.data_pool[series_name] = self._load_data(series_name)
        return self.data_pool[series_name][1]

    def share_data(self, storage: SharedStorage) -> NoReturn:
        """
        make the current data shareable across multiple workers (if exists)

        :param storage: the storage to share data with
        """
        storage.put("forecast_meta", self.forecast_data_meta)
        storage.put("detect_meta", self.detect_data_meta)
        storage.put("data_pool", self.data_pool)

    def sync_data(self, storage: SharedStorage) -> NoReturn:
        """
        retrieve the data shared by the main process

        :param storage: the shared storage to get data from
        """
        self._forecast_data_meta = storage.get("forecast_meta")
        self._detect_data_meta = storage.get("detect_meta")
        self.data_pool.update(storage.get("data_pool", {}))
