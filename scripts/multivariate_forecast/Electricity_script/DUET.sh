python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon": 96}' --model-name "duet.DUET" --model-hyper-params '{"CI": 1, "batch_size": 16, "d_ff": 1024, "d_model": 512, "dropout": 0.5, "e_layers": 4, "factor": 3, "fc_dropout": 0, "horizon": 96, "k": 2, "loss": "MAE", "lr": 0.0005, "lradj": "type1", "n_heads": 1, "norm": true, "num_epochs": 100, "num_experts": 4, "patch_len": 48, "patience": 5, "seq_len": 512}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Electricity/DUET"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon": 192}' --model-name "duet.DUET" --model-hyper-params '{"CI": 1, "batch_size": 16, "d_ff": 1024, "d_model": 512, "dropout": 0.5, "e_layers": 4, "factor": 3, "fc_dropout": 0, "horizon": 192, "k": 2, "loss": "MAE", "lr": 0.0005, "lradj": "type1", "n_heads": 1, "norm": true, "num_epochs": 100, "num_experts": 4, "patch_len": 48, "patience": 5, "seq_len": 512}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Electricity/DUET"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon": 336}' --model-name "duet.DUET" --model-hyper-params '{"CI": 1, "batch_size": 16, "d_ff": 1024, "d_model": 512, "dropout": 0.5, "e_layers": 4, "factor": 3, "fc_dropout": 0, "horizon": 336, "k": 2, "loss": "MAE", "lr": 0.0005, "lradj": "type1", "n_heads": 1, "norm": true, "num_epochs": 100, "num_experts": 4, "patch_len": 48, "patience": 5, "seq_len": 512}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Electricity/DUET"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon": 720}' --model-name "duet.DUET" --model-hyper-params '{"CI": 1, "batch_size": 16, "d_ff": 1024, "d_model": 512, "dropout": 0.5, "e_layers": 4, "factor": 3, "fc_dropout": 0, "horizon": 720, "k": 2, "loss": "MAE", "lr": 0.0005, "lradj": "type1", "n_heads": 1, "norm": true, "num_epochs": 100, "num_experts": 4, "patch_len": 48, "patience": 5, "seq_len": 512}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Electricity/DUET"
