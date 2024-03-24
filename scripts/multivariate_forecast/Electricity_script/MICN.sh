python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"pred_len":96}' --model-name "time_series_library.MICN" --model-hyper-params '{"d_model": 512, "dropout": 0.05, "lr": 0.001, "moving_avg": 24, "num_epochs": 15, "pred_len": 96, "seq_len": 96, "d_ff": 2048}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Electricity/MICN"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"pred_len":192}' --model-name "time_series_library.MICN" --model-hyper-params '{"d_ff": 512, "d_model": 256, "pred_len": 192, "seq_len": 96}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Electricity/MICN"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"pred_len":336}' --model-name "time_series_library.MICN" --model-hyper-params '{"d_ff": 512, "d_model": 256, "pred_len": 336, "seq_len": 96}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Electricity/MICN"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"pred_len":720}' --model-name "time_series_library.MICN" --model-hyper-params '{"d_ff": 512, "d_model": 256, "pred_len": 720, "seq_len": 96}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Electricity/MICN"

