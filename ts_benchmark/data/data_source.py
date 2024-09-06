# -*- coding: utf-8 -*-
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, NoReturn, List

import pandas as pd

from ts_benchmark.common.constant import FORECASTING_DATASET_PATH
from ts_benchmark.data.dataset import Dataset
from ts_benchmark.data.utils import load_series_info, read_data


logger = logging.getLogger(__name__)


class DataSource:
    """
    A class that manages and reads from data sources

    A data source is responsible for loading data into the internal dataset object,
    as well as detecting and updating data in the source storage.
    """

    # The class for the internal dataset object
    DATASET_CLASS = Dataset

    def __init__(
        self,
        data_dict: Optional[Dict[str, pd.DataFrame]] = None,
        metadata: Optional[pd.DataFrame] = None,
    ):
        """
        initializer

        :param data_dict: A dictionary of time series, where the keys are the names and
            the values are DataFrames following the OTB protocol.
        :param metadata: A DataFrame where the index contains series names and columns
            contains meta-info fields.
        """
        self._dataset = self.DATASET_CLASS()
        self._dataset.set_data(data_dict, metadata)

    @property
    def dataset(self) -> Dataset:
        """
        Returns the internally maintained dataset object

        This dataset is where the DataSource loads data into.
        """
        return self._dataset

    def load_series_list(self, series_list: List[str]) -> NoReturn:
        """
        Loads a list of time series from the source

        The series data and (optionally) meta information are loaded into the internal dataset.

        :param series_list: The list of series names.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support loading series at runtime.")


class LocalDataSource(DataSource):
    """
    The data source that manages data files in a local directory
    """

    #: index column name of the metadata
    _INDEX_COL = "file_name"

    def __init__(self, local_data_path: str, metadata_file_name: str):
        """
        initializer

        Only the metadata is loaded during initialization, while all series data are
        loaded on demand.

        :param local_data_path: the directory that contains csv data files and metadata.
        :param metadata_file_name: name of the metadata file.
        """
        self.local_data_path = local_data_path
        self.metadata_path = os.path.join(local_data_path, metadata_file_name)
        metadata = self.update_meta_index()
        super().__init__({}, metadata)

    def update_meta_index(self) -> pd.DataFrame:
        """
        Check if there are any user-added dataset files in the dataset folder
        Attempt to register them in the metadata and load metadata from the metadata file
        :return: metadata
        :rtype: pd.DataFrame
        """

        metadata = self._load_metadata()
        csv_files = {
            f
            for f in os.listdir(self.local_data_path)
            if f.endswith(".csv") and f != os.path.basename(self.metadata_path)
        }
        user_csv_files = set(csv_files).difference(metadata.index)
        if not user_csv_files:
            return metadata
        data_info_list = []
        for user_csv in user_csv_files:
            try:
                data_info_list.append(
                    load_series_info(os.path.join(self.local_data_path, user_csv))
                )
            except Exception as e:
                raise RuntimeError(f"Error loading series info from {user_csv}: {e}")
        new_metadata = pd.DataFrame(data_info_list)
        new_metadata.set_index(self._INDEX_COL, drop=False, inplace=True)
        metadata = pd.concat([metadata, new_metadata])
        with open(self.metadata_path, "w", newline="", encoding="utf-8") as csvfile:
            metadata.to_csv(csvfile, index=False)
        logger.info(
            "Detected %s new user datasets, registered in the metadata",
            len(user_csv_files),
        )
        return metadata

    def load_series_list(self, series_list: List[str]) -> NoReturn:
        logger.info("Start loading %s series in parallel", len(series_list))
        data_dict = {}
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._load_series, series_name)
                for series_name in series_list
            ]
        for future, series_name in zip(futures, series_list):
            data_dict[series_name] = future.result()
        logger.info("Data loading finished.")
        self.dataset.update_data(data_dict)

    def _load_metadata(self) -> pd.DataFrame:
        """
        Loads metadata from a local csv file
        """
        metadata = pd.read_csv(self.metadata_path)
        metadata.set_index(self._INDEX_COL, drop=False, inplace=True)
        return metadata

    def _load_series(self, series_name: str) -> pd.DataFrame:
        """
        Loads a time series from a single data file

        :param series_name: Series name.
        :return: A time series in DataFrame format.
        """
        datafile_path = os.path.join(self.local_data_path, series_name)
        data = read_data(datafile_path)
        return data


class LocalForecastingDataSource(LocalDataSource):
    """
    The local data source of the forecasting task
    """

    def __init__(self):
        super().__init__(
            FORECASTING_DATASET_PATH,
            "FORECAST_META.csv"
        )



