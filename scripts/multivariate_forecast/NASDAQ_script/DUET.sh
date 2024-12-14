python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NASDAQ.csv" --strategy-args '{"horizon": 24}' --model-name "duet.DUET" --model-hyper-params '{"CI": 1, "batch_size": 8, "d_ff": 1024, "d_model": 512, "dropout": 0.2, "e_layers": 2, "factor": 3, "horizon": 24, "k": 2, "loss": "MAE", "lr": 0.0005, "lradj": "type1", "n_heads": 1, "norm": true, "num_epochs": 100, "num_experts": 4, "patch_len": 48, "patience": 5, "seq_len": 104}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "NASDAQ/DUET"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NASDAQ.csv" --strategy-args '{"horizon": 36}' --model-name "duet.DUET" --model-hyper-params '{"CI": 1, "batch_size": 8, "d_ff": 1024, "d_model": 512, "dropout": 0.2, "e_layers": 2, "factor": 3, "fc_dropout": 0, "horizon": 36, "k": 2, "loss": "MAE", "lr": 0.0005, "lradj": "type1", "n_heads": 1, "norm": true, "num_epochs": 100, "num_experts": 4, "patch_len": 48, "patience": 5, "seq_len": 104}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "NASDAQ/DUET"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NASDAQ.csv" --strategy-args '{"horizon": 48}' --model-name "duet.DUET" --model-hyper-params '{"CI": 1, "batch_size": 8, "d_ff": 1024, "d_model": 512, "dropout": 0.3, "e_layers": 2, "factor": 3, "fc_dropout": 0.3, "horizon": 48, "k": 2, "loss": "MAE", "lr": 0.0005, "lradj": "type1", "n_heads": 1, "norm": true, "num_epochs": 100, "num_experts": 4, "patch_len": 48, "patience": 5, "seq_len": 104}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "NASDAQ/DUET"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NASDAQ.csv" --strategy-args '{"horizon": 60}' --model-name "duet.DUET" --model-hyper-params '{"CI": 1, "batch_size": 8, "d_ff": 1024, "d_model": 512, "dropout": 0.2, "e_layers": 2, "factor": 3, "fc_dropout": 0.6, "horizon": 60, "k": 2, "loss": "MAE", "lr": 0.0005, "lradj": "type1", "n_heads": 1, "norm": true, "num_epochs": 100, "num_experts": 4, "patch_len": 48, "patience": 5, "seq_len": 104}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "NASDAQ/DUET"
