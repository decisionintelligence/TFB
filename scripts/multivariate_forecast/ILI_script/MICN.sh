python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args '{"horizon": 24}' --model-name "time_series_library.MICN" --model-hyper-params '{"conv_kernel": [18, 12], "d_ff": 512, "d_model": 256, "dropout": 0.05, "horizon": 24, "lr": 0.001, "moving_avg": 25, "norm": true, "num_epochs": 15, "seq_len": 104}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "ILI/MICN" --adapter "transformer_adapter"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args '{"horizon": 36}' --model-name "time_series_library.MICN" --model-hyper-params '{"conv_kernel": [18, 12], "d_ff": 512, "d_model": 512, "dropout": 0.05, "horizon": 36, "lr": 0.001, "moving_avg": 25, "norm": true, "num_epochs": 15, "seq_len": 104}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "ILI/MICN" --adapter "transformer_adapter"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args '{"horizon": 48}' --model-name "time_series_library.MICN" --model-hyper-params '{"conv_kernel": [18, 12], "d_ff": 512, "d_model": 256, "dropout": 0.05, "horizon": 48, "lr": 0.001, "moving_avg": 25, "norm": true, "num_epochs": 15, "seq_len": 104}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "ILI/MICN" --adapter "transformer_adapter"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args '{"horizon": 60}' --model-name "time_series_library.MICN" --model-hyper-params '{"conv_kernel": [18, 12], "d_ff": 768, "d_model": 768, "factor": 3, "horizon": 60, "label_len": 36, "norm": true, "seq_len": 104}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "ILI/MICN" --adapter "transformer_adapter"

