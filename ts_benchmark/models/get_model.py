# -*- coding: utf-8 -*-
import importlib
from typing import Type, Any

from ts_benchmark.baselines import ADAPTER


def get_model_info(model_config: dict) -> Any:
    """
    Obtain model information based on model configuration.
    Retrieve and return the corresponding model class based on the provided model configuration information.

    :param model_config: A dictionary that contains model configuration information.
    :return: The model class corresponding to the specified modelname.
    :raises ImportError: If the specified model package cannot be imported.
    :raises AttributeError: If the specified classname cannot be found in the imported module.
    """

    def import_model_info(model_class: str) -> Any:
        """
        Import model information.

        Import and return the model class from the corresponding module based on the provided model class name.

        :param model_class: The fully qualified name of the model class to be imported, such as' package. module. ' ModelClassName.

        :return: The imported model class.
        """
        model_package, class_name = model_class.rsplit(".", 1)
        # Import the specified model package
        mod = importlib.import_module(model_package)
        # Retrieve and return the specified class from the imported module
        model_info = getattr(mod, class_name)
        return model_info

    model_name = model_config["model_name"]
    try:
        model_class = model_name
        model_info = import_model_info(model_class)

    except (ImportError, AttributeError):
        model_class = (
            "ts_benchmark.baselines." + model_config["model_name"]
        )
        model_info = import_model_info(model_class)

    adapter_name = model_config.get("adapter")
    if adapter_name is not None:
        if adapter_name not in ADAPTER:
            raise ValueError(f"Unknown adapter {adapter_name}")
        model_info = import_model_info(ADAPTER[adapter_name])(model_info)

    return model_info


def get_model_hyper_params(
    recommend_model_hyper_params: dict, required_hyper_params: dict, model_config: dict
) -> dict:
    """
    Obtain the hyperparameters of the model.

    Based on the recommended model hyperparameters, required hyperparameter mapping, and model configuration, return the merged model hyperparameters.

    :param recommend_model_hyper_params: Recommended model hyperparameters.
    :param required_hyper_params: The required hyperparameter mapping, in the format of {parameter name: standard parameter name}.
    :param model_config: Model configuration, including the model_hyperparams field.

    :return: The merged model hyperparameters.

    :raises ValueError: If there are missing hyperparameters.
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
    Model factory class, used to instantiate models.
    """

    def __init__(
        self,
        model_name: str,
        model_factory: Type,
        model_hyper_params: dict,
    ):
        """
        Initialize the ModelFactory object.

        :param model_name: Model name.
        :param model_factory: The actual model factory class used to create model instances.
        :param model_hyper_params: The hyperparameter dictionary required by the model, which includes standard name mapping.
        """
        self.model_name = model_name
        self.model_factory = model_factory
        self.model_hyper_params = model_hyper_params

    def __call__(self) -> Any:
        """
        Instantiate the model by calling the actual model factory class.

        :return: Instantiated model object.
        """

        return self.model_factory(**self.model_hyper_params)


def get_model(all_model_config: dict) -> list:
    """
    Obtain a list of model factories based on model configuration.
    Create a model factory list for instantiating the model based on all provided model configuration information.

    :param all_model_config: A dictionary that contains all model configuration information.
    :return: List of model factories used to instantiate different models.
    """
    model_factory_list = []  # Store a list of model factories
    # Traverse each model configuration
    for model_config in all_model_config["models"]:
        model_info = get_model_info(model_config)  # Obtain model information
        fallback_model_name = model_config["model_name"].split(".")[-1]

        # Analyze model information
        if isinstance(model_info, dict):
            model_factory = model_info.get("model_factory")
            if model_factory is None:
                raise ValueError("model_factory is none")
            required_hyper_params = model_info.get("required_hyper_params", {})
            model_name = model_info.get("model_name", fallback_model_name)
        elif isinstance(model_info, type):
            model_factory = model_info
            required_hyper_params = {}
            if hasattr(model_factory, "required_hyper_params"):
                required_hyper_params = model_factory.required_hyper_params()
            model_name = fallback_model_name
        else:
            model_factory = model_info
            required_hyper_params = {}
            model_name = fallback_model_name

        model_hyper_params = get_model_hyper_params(
            all_model_config["recommend_model_hyper_params"],
            required_hyper_params,
            model_config,
        )
        # Add Model Factory to List
        model_factory_list.append(
            ModelFactory(model_name, model_factory, model_hyper_params)
        )
    return model_factory_list
