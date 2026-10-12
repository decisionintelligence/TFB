import json
import os
from typing import List
import optuna
from ts_benchmark.pipeline import pipeline
from .search_space import sample_params


def evaluate_params(params, config_data, data_name_list, model_name, strategy_args):
    # 在此处进行评估
    data_config = config_data["data_config"]
    model_config = config_data["model_config"]
    evaluation_config = config_data["evaluation_config"]

    model_config["models"] = [{
        "adapter": None,
        "model_name": model_name,
        "model_hyper_params": params
    }]

    # 进行模型训练并评估
    try:
        log_files = pipeline(data_config, model_config, evaluation_config)
    except Exception as e:
        print(f"Error: {e}")
        return float("inf")

    # 假设我们从日志文件中提取损失值（这里我们假设返回0.0作为占位符）
    return 0.0  # 在这里替换为实际计算的验证损失值


def run_optuna_search(config_path: str, data_name_list: List[str], model_name: str, save_path: str, n_trials: int = 10,
                      seed: int = None):
    # 加载配置文件
    with open(config_path, "r") as f:
        config_data = json.load(f)

    study = optuna.create_study(direction="minimize", study_name="hyperparameter_optimization")
    study.optimize(
        lambda trial: evaluate_params(sample_params(model_name, trial), config_data, data_name_list, model_name, {}),
        n_trials=n_trials)

    # 保存最优超参数
    best_params = study.best_params
    best_value = study.best_value

    best_params_json = {
        "model_name": model_name,
        "series_name": data_name_list[0],  # 假设只有一个数据集
        "objective": "val_loss",  # 假设优化目标是 val_loss
        "best_value": best_value,
        "best_params": best_params
    }

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, f"{model_name}_{data_name_list[0]}_best_params.json"), "w") as f:
        json.dump(best_params_json, f, indent=2)

    return best_params_json