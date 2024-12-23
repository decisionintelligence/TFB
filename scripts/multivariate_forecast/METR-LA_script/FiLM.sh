python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "METR-LA.csv" --strategy-args '{"horizon": 96}' --model-name "time_series_library.FiLM" --model-hyper-params '{"d_ff": 512, "d_model": 256, "horizon": 96, "norm": true, "seq_len": 96}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "METR-LA/FiLM" --adapter "transformer_adapter"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "METR-LA.csv" --strategy-args '{"horizon":192}' --model-name "time_series_library.FiLM" --model-hyper-params '{"dropout": 0.05, "factor": 3, "moving_avg": 24, "num_epochs": 15, "patience": 15, "horizon": 192, "seq_len": 720, "d_ff": 2048, "d_model": 512}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "METR-LA/FiLM"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "METR-LA.csv" --strategy-args '{"horizon": 336}' --model-name "time_series_library.FiLM" --model-hyper-params '{"batch_size": 2, "d_ff": 2048, "d_model": 512, "dropout": 0.05, "factor": 3, "horizon": 336, "moving_avg": 24, "norm": true, "num_epochs": 15, "patience": 15, "seq_len": 512}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "METR-LA/FiLM" --adapter "transformer_adapter"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "METR-LA.csv" --strategy-args '{"horizon": 720}' --model-name "time_series_library.FiLM" --model-hyper-params '{"batch_size": 8, "d_ff": 2048, "d_model": 512, "dropout": 0.05, "factor": 3, "horizon": 720, "moving_avg": 24, "norm": true, "num_epochs": 15, "patience": 15, "seq_len": 512}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "METR-LA/FiLM" --adapter "transformer_adapter"

