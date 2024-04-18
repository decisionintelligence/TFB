from typing import Optional, Dict, NoReturn

import pandas as pd


class Dataset:
    """
    A class that stores data information and meta information

    Any integrity checks and special update rules should be encapsulated under this interface.
    """

    def __init__(self):
        self._metadata = None
        self._data_dict = {}

    @property
    def metadata(self) -> Optional[pd.DataFrame]:
        """
        Returns the full metadata in DataFrame format

        If the metadata is not available, returns None.

        DO NOT perform inplace operations on the return value.
        """
        return self._metadata

    def set_data(
        self,
        data_dict: Optional[Dict[str, pd.DataFrame]] = None,
        metadata: Optional[pd.DataFrame] = None,
    ) -> NoReturn:
        """
        Sets the series data and the meta information

        :param data_dict: A dictionary of DataFrames where the keys are the names of the
            series, and the values are DataFrames following the OTB protocol. If None is
            given, the data dictionary is not set.
        :param metadata: A DataFrame of meta information where the index contains series names
            and the columns contains meta-info fields. If None is given, the metadata is not
            set.
        """
        new_metadata = metadata if metadata is not None else self._metadata
        new_data_dict = data_dict if data_dict is not None else self._data_dict

        self._validate_data(new_data_dict, new_metadata)

        self._metadata = new_metadata
        self._data_dict = new_data_dict

    def _validate_data(
        self, data_dict: Dict[str, pd.DataFrame], metadata: Optional[pd.DataFrame]
    ) -> NoReturn:
        """
        Validates if the given data_dict and metadata are compatible

        Currently, we do not enforce any checks on the data.

        :param data_dict: A dictionary of DataFrames where the keys are the names of the
            series, and the values are DataFrames following the OTB protocol.
        :param metadata: A DataFrame of meta information where the index contains series names
            and the columns contains meta-info fields. The value might be None when the
            metadata is not available.
        :return:
        """

    def update_data(self, inc_data_dict: Dict[str, pd.DataFrame]) -> NoReturn:
        """
        Updates the data dictionary

        :param inc_data_dict: The incremental dictionary of data.
        """
        self._validate_update_data(inc_data_dict)
        self._data_dict.update(inc_data_dict)

    def _validate_update_data(self, inc_data_dict: Dict[str, pd.DataFrame]) -> NoReturn:
        """
        Validates if the incremental data is compatible with the current data

        Currently, we do not enforce any checks on the incremental data.

        :param inc_data_dict: The incremental update of the data dictionary.
        """

    def clear_data(self) -> NoReturn:
        """
        Clear all data stored in this dataset
        """
        self._metadata = None
        self._data_dict = {}

    def get_series(self, name: str) -> Optional[pd.DataFrame]:
        """
        Gets a single time series by name

        :param name: The name of the series to get.
        :return: A time series in DataFrame format. If the time series is not available,
            return None.
        """
        return self._data_dict.get(name, None)

    def get_series_meta_info(self, name: str) -> Optional[pd.Series]:
        """
        Gets the meta information of time series by name

        We do not return the meta information of unexisting series even if
        the meta information itself is available.

        :param name: The name of the series to get.
        :return: Meta information data in Series format. If the meta information or the
            corresponding series is not available, return None.
        """
        if name not in self._data_dict:
            return None
        if self._metadata is None or name not in self._metadata.index:
            return None
        return self._metadata.loc[name]

    def has_series(self, name: str) -> bool:
        """
        Check if a series is available in the dataset

        :param name: The name of the series.
        :return: True if the series is contained in the dataset, False otherwise.
        """
        return name in self._data_dict

    def has_series_meta_info(self, name: str) -> bool:
        """
        Check if the meta-info of a series is available in the dataset

        :param name: The name of the series.
        :return: True if the series and its meta-info are both contained in the dataset,
            False otherwise.
        """
        return self.has_series(name) and self._metadata is not None and name in self._metadata.index

    def get_state(self) -> Dict:
        """
        Get a serializable state

        :return: A dictionary of state which is composed of simple types as well as numpy
            or pandas array types.
        """
        return {
            "metadata": self._metadata,
            "data_dict": self._data_dict,
        }

    def set_state(self, state: Dict) -> NoReturn:
        """
        Restores from a serializable state

        :param state: A dictionary of state which is composed of simple types as well as numpy
            or pandas array types.
        """
        self._metadata = state["metadata"]
        self._data_dict = state["data_dict"]
