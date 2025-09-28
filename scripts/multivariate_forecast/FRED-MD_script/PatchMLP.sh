python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "FRED-MD.csv" --strategy-args '{"horizon": 24}' --model-name "patchmlp.PatchMLP" --model-hyper-params '{"batch_size": 64, "d_model": 2048, "dropout": 0.1, "e_layers": 3, "horizon": 24, "lr": 0.001, "norm": true, "seq_len": 104}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FRED-MD/PatchMLP"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "FRED-MD.csv" --strategy-args '{"horizon": 36}' --model-name "patchmlp.PatchMLP" --model-hyper-params '{"batch_size": 64, "d_model": 2048, "dropout": 0.1, "e_layers": 1, "horizon": 36, "lr": 0.001, "norm": true, "seq_len": 104}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FRED-MD/PatchMLP"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "FRED-MD.csv" --strategy-args '{"horizon": 48}' --model-name "patchmlp.PatchMLP" --model-hyper-params '{"batch_size": 64, "d_model": 2048, "dropout": 0.1, "e_layers": 2, "horizon": 48, "lr": 0.001, "norm": true, "seq_len": 104}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FRED-MD/PatchMLP"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "FRED-MD.csv" --strategy-args '{"horizon": 60}' --model-name "patchmlp.PatchMLP" --model-hyper-params '{"batch_size": 64, "d_model": 2048, "dropout": 0.1, "e_layers": 2, "horizon": 60, "lr": 0.001, "norm": true, "seq_len": 104}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FRED-MD/PatchMLP"

