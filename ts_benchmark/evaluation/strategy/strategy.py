# -*- coding: utf-8 -*-
import abc
import json
import logging
from functools import cached_property
from typing import Any, NoReturn, List, Dict
from sklearn.preprocessing import StandardScaler

import numpy as np

from ts_benchmark.evaluation.evaluator import Evaluator
from ts_benchmark.models.get_model import ModelFactory


class ResultCollector:
    """
    测试结果收集工具

    用于帮助 strategy 自定义结果返回方式
    """

    def __init__(self):
        self.results = []

    def add(self, result: Any) -> NoReturn:
        self.results.append(result)

    def collect(self) -> List:
        return self.results

    def reset(self) -> NoReturn:
        self.results = []

    def get_size(self) -> int:
        """
        返回当前已收集的测试结果数量
        """
        return len(self.results)


class Strategy(metaclass=abc.ABCMeta):
    """
    策略基类，用于定义时间序列预测策略的通用结构。
    """

    REQUIRED_FIELDS = []
    STRATEGY_NAME = "strategy_name"

    def __init__(self, strategy_config: dict, evaluator: Evaluator):
        """
        初始化策略对象。

        :param strategy_config: 模型评估配置。
        """
        self.strategy_config = strategy_config
        self.evaluator = evaluator
        self.scaler = StandardScaler()

    @abc.abstractmethod
    def execute(self, series_name: str, model_factory: ModelFactory) -> Any:
        """
        执行策略的具体预测过程。

        """
        pass

    def get_config_str(self):
        """
        获取配置信息的字符串表示。

        :return: 配置信息的 JSON 格式字符串。
        """
        provided_args = sorted(list(self.strategy_config.keys()))
        required_args = ["strategy_name"] + self.REQUIRED_FIELDS

        if provided_args != sorted(required_args):
            missing_args = [
                arg for arg in self.REQUIRED_FIELDS if arg not in provided_args
            ]
            extra_args = [arg for arg in provided_args if arg not in required_args]
            config_args = {
                arg: self.strategy_config[arg]
                for arg in provided_args
                if arg in required_args
            }

            if missing_args:
                error_message = f"缺少参数: {', '.join(missing_args)} "
                raise RuntimeError(error_message)
            if extra_args:
                error_message = f"多出参数: {', '.join(extra_args)} "
                logging.warning(error_message)

            return json.dumps(config_args, sort_keys=True)
        else:
            return json.dumps(self.strategy_config, sort_keys=True)

    def get_collector(self) -> ResultCollector:
        return ResultCollector()

    @staticmethod
    @abc.abstractmethod
    def accepted_metrics() -> List[str]:
        """
        获取当前 strategy 支持的指标列表
        """

    @property
    @abc.abstractmethod
    def field_names(self) -> List[str]:
        """
        获取当前 strategy 返回结果的字段名列表
        """

    @cached_property
    def _field_name_to_idx(self) -> Dict:
        return {k: i for i, k in enumerate(self.field_names)}

    def get_default_result(self, **kwargs) -> List:
        """
        获取当前 strategy 返回结果的默认值

        :param kwargs: key 为 FieldNames 中定义的字段名，value 为想要将该字段替换为什么值。
        """
        ret = self.evaluator.default_result()
        ret += [np.nan] * (len(self.field_names) - len(ret))
        for k, v in kwargs.items():
            if k not in self._field_name_to_idx:
                raise ValueError(f"Unknown field name {k}")
            ret[self._field_name_to_idx[k]] = v
        return ret
