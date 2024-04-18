# -*- coding: utf-8 -*-
import abc
from typing import NoReturn

from ts_benchmark.data.data_source import DataSource


class DataServer(metaclass=abc.ABCMeta):
    """
    Base class for data servers

    Data servers are responsible for sharing data to the workers
    through shared storage, message queue, etc.
    """

    @abc.abstractmethod
    def start_async(self) -> NoReturn:
        """
        Start the data server in non-blocking mode
        """
