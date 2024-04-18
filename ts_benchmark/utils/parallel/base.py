# -*- coding: utf-8 -*-

from __future__ import absolute_import

import abc
from typing import Any, NoReturn


class TaskResult(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def result(self) -> Any:
        """
        block until result is available
        """

    @abc.abstractmethod
    def put(self, value: Any) -> NoReturn:
        """
        set value of the result
        """


class SharedStorage(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def put(self, name: str, value: Any) -> NoReturn:
        """
        store variable into storage
        """

    @abc.abstractmethod
    def get(self, name: str, default_value: Any = None) -> Any:
        """
        get variable from storage
        """
