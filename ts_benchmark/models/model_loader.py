# -*- coding: utf-8 -*-
import importlib
import logging
from typing import Any, Union, Dict, Callable, List

from ts_benchmark.baselines import ADAPTER

logger = logging.getLogger(__name__)


def _import_attribute(attr_path: str) -> Any:
    """
    import attribute according to a fully qualified path.

    :param attr_path: A dot-separated path.
    :return: If the target attribute exists, the attribute is returned, otherwise return None.
    """
    package_name, name = attr_path.rsplit(".", 1)
    package = importlib.import_module(package_name)
    return getattr(package, name, None)


def import_model_info(model_path: str) -> Union[Dict, Callable]:
    """
    Import model information.

    We first clarify some concepts before defining model information:

    - required hyperparameters: This is a specially designed mechanism to enable models to relinquish the settings
      of some hyperparameters to the benchmark.
      For example, if a model cannot automatically decide the best input window size
      (corresponding hyperparameter `input_window_size`), it can leave the decision to the benchmark, so that
      the benchmark is able to use a globally recommended setting (corresponding hyperparameter `input_chunk_length`)
      to produce a fair comparison between different models;.
      In this example, to enable this mechanism properly, the model is required to provide a
      `required_hyper_params` field in dictionary `{"input_window_size": "input_chunk_length"}`.

    Model information should be either:

    - A dictionary containing these fields:

        - model_factory: Callable. A callable that accepts hyperparameters as kwargs;
        - model_hyper_params: Dictionary, optional; A dictionary containing hyperparameters for the model.
          These hyperparameters overwrite the ones specified by recommended hyperparameters;
        - required_hyper_params: Dictionary, optional; A dictionary of hyperparameters to be filled
          by the benchmark, in format `{model_param_name: std_param_name}`.
        - model_name: str, optional; The name of the model that is recorded in the output logs.

    - A callable that returns an instance compatible with :class:`ModelBase` interface when called with
      hyperparameters as keyword arguments. This callable may optionally support the following features:

        - attribute required_hyper_params: Dictionary, optional; A dictionary of hyperparameters to be
          filled by the benchmark, in format `{model_param_name: std_param_name}`.

    :param model_path: The fully qualified path to the model information.
    :return: The imported model information.
    """
    model_info = _import_attribute(model_path)

    if not isinstance(model_info, (Dict, Callable)):
        raise ValueError(
            f"Unsupported model info with type {type(model_info).__name__}"
        )

    return model_info


def get_model_info(model_config: Dict) -> Union[Dict, Callable]:
    """
    Obtain model information based on model configuration.

    :param model_config: A dictionary that contains model configuration information. The supported fields are:

        - model_name: str. The path to the model information, the following paths are searched in order to
          find the model information:

            - `{model_name[7:]}` if model_name.startswith("global.")
            - `ts_benchmark.baselines.{model_name}`
            - `{model_name}`

        - adapter: str, optional. The adapter name to wrap the found model information.
          Must be one of the adapters defined in :mod:`ts_benchmark.baselines.__init__`;

    :return: The model information corresponding to the config.
    :raises ImportError: If the specified model package cannot be imported.
    :raises AttributeError: If the specified `model_name` cannot be found in the imported module.
    """
    model_name_candidates = [
        model_config["model_name"][7:] if model_config["model_name"].startswith("global.") else None,
        "ts_benchmark.baselines." + model_config["model_name"],
        model_config["model_name"],
    ]
    model_name_candidates = list(filter(None, model_name_candidates))

    model_info = None
    for model_name in model_name_candidates:
        try:
            logger.info("Trying to load model %s", model_name)
            model_info = import_model_info(model_name)
        except (ImportError, AttributeError):
            logger.info("Loading model %s failed", model_name)
            continue
        else:
            break

    adapter_name = model_config.get("adapter")
    if adapter_name is not None:
        if adapter_name not in ADAPTER:
            raise ValueError(f"Unknown adapter {adapter_name}")
        model_info = _import_attribute(ADAPTER[adapter_name])(model_info)

    return model_info


def get_model_hyper_params(
    recommend_model_hyper_params: Dict, required_hyper_params: Dict, model_config: Dict
) -> Dict:
    """
    Obtain the hyperparameters of the model.

    The hyperparameter dictionary is constructed following these steps:

    - Fill in the recommended hyperparameters;
    - Update the hyperparameters with those specified in the model_config;

    :param recommend_model_hyper_params: A dictionary of hyperparameters recommended by the benchmark.
    :param required_hyper_params: A dictionary of hyperparameters to be filled by the benchmark,
        in format `{model_param_name: std_param_name}`. Please refer to :func:`import_model_info` for
        details about this argument.
    :param model_config: Model configuration, the supported fields are:

        - model_hyper_params: dictionary, optional; This dictionary specifies the hyperparameters used
          in the corresponding model;

    :return: The constructed model hyperparameter dictionary.
    :raises ValueError: If there are unfilled hyperparameters.
    """
    model_hyper_params = {
        arg_name: recommend_model_hyper_params[arg_std_name]
        for arg_name, arg_std_name in required_hyper_params.items()
        if arg_std_name in recommend_model_hyper_params
    }
    model_hyper_params.update(model_config.get("model_hyper_params", {}))
    missing_hp = set(required_hyper_params) - set(model_hyper_params)
    if missing_hp:
        raise ValueError("These hyper parameters are missing : {}".format(missing_hp))
    return model_hyper_params


class ModelFactory:
    """
    Model factory, the standard type to instantiate models in the pipeline.
    """

    def __init__(
        self,
        model_name: str,
        model_factory: Callable,
        model_hyper_params: dict,
    ):
        """
        Initialize the ModelFactory object.

        :param model_name: Model name.
        :param model_factory: A model factory (classes or factory functions) used to create model instances.
        :param model_hyper_params: The hyperparameter dictionary used to instantiate the model instance.
        """
        self.model_name = model_name
        self.model_factory = model_factory
        self.model_hyper_params = model_hyper_params

    def __call__(self) -> Any:
        """
        Instantiate the model.

        :return: A model instance that is compatible with the :class:`ModelBase` interface.
        """
        return self.model_factory(**self.model_hyper_params)


def get_models(all_model_config: Dict) -> List[ModelFactory]:
    """
    Obtain a list of ModelFactory objects based on model configuration.

    :param all_model_config: A dictionary that contains all model configuration information, supported fields are:

        - models: list. A list of model information, where each item is a dictionary.
          The supported fields in each dictionary are:

            - model_name: str. The path to the model information. Please refer to :func:`get_model_info` for
              the details about the model searching strategy;
            - adapter: str, optional. The adapter name to wrap the found model information.
              Must be one of the adapters defined in :mod:`ts_benchmark.baselines.__init__`;

        - recommend_model_hyper_params: dictionary, optional; A dictionary of globally recommended hyperparameters
          that the benchmark supplies to all models;

    :return: List of model factories used to instantiate different models.
    """
    model_factory_list = []  # Store a list of model factories
    # Traverse each model configuration
    for model_config in all_model_config["models"]:
        model_info = get_model_info(model_config)  # Obtain model information
        fallback_model_name = model_config["model_name"].split(".")[-1]

        # Analyze model information
        if isinstance(model_info, Dict):
            model_factory = model_info.get("model_factory")
            if model_factory is None:
                raise ValueError("model_factory is none")
            required_hyper_params = model_info.get("required_hyper_params", {})
            model_name = model_info.get("model_name", fallback_model_name)
        elif isinstance(model_info, Callable):
            model_factory = model_info
            required_hyper_params = {}
            if hasattr(model_factory, "required_hyper_params"):
                required_hyper_params = model_factory.required_hyper_params()
            model_name = fallback_model_name
        else:
            raise ValueError(f"Unexpected model info type {type(model_info).__name__}")

        model_hyper_params = get_model_hyper_params(
            all_model_config.get("recommend_model_hyper_params", {}),
            required_hyper_params,
            model_config,
        )
        # Add Model Factory to List
        model_factory_list.append(
            ModelFactory(model_name, model_factory, model_hyper_params)
        )
    return model_factory_list
