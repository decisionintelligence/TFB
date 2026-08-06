import optuna

def sample_params(model_name: str, trial: optuna.trial.Trial) -> dict:
    if model_name == 'olinear.OLinear':
        return {
            "lr": trial.suggest_loguniform("lr", 1e-5, 1e-2),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "d_model": trial.suggest_categorical("d_model", [64, 128, 256, 512]),
            "d_ff": trial.suggest_categorical("d_ff", [128, 256, 512, 1024]),
            "e_layers": trial.suggest_int("e_layers", 1, 3)
        }
    else:
        raise NotImplementedError(f"Model {model_name} not implemented in search space.")