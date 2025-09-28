python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"horizon": 96}' --model-name "patchmlp.PatchMLP" --model-hyper-params '{"batch_size": 64, "d_model": 1024, "dropout": 0.1, "e_layers": 1, "horizon": 96, "lr": 0.001, "norm": true, "seq_len": 336}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTh1/PatchMLP"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"horizon": 192}' --model-name "patchmlp.PatchMLP" --model-hyper-params '{"batch_size": 64, "d_model": 1024, "dropout": 0.1, "e_layers": 1, "horizon": 192, "lr": 0.0005, "norm": true, "seq_len": 336}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTh1/PatchMLP"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"horizon": 336}' --model-name "patchmlp.PatchMLP" --model-hyper-params '{"batch_size": 64, "d_model": 1024, "dropout": 0.1, "e_layers": 1, "horizon": 336, "lr": 0.0001, "norm": true, "seq_len": 336}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTh1/PatchMLP"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"horizon": 720}' --model-name "patchmlp.PatchMLP" --model-hyper-params '{"batch_size": 64, "d_model": 1024, "dropout": 0.1, "e_layers": 1, "horizon": 720, "lr": 0.0001, "norm": true, "seq_len": 336}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTh1/PatchMLP"

