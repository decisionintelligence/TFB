python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NYSE.csv" --strategy-args '{"horizon":24}' --model-name "time_series_library.DLinear" --model-hyper-params '{"d_ff": 64, "d_model": 32, "lr": 0.01, "horizon": 24, "seq_len": 36}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "NYSE/DLinear"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NYSE.csv" --strategy-args '{"horizon":36}' --model-name "time_series_library.DLinear" --model-hyper-params '{"d_ff": 64, "d_model": 32, "lr": 0.01, "horizon": 36, "seq_len": 36}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "NYSE/DLinear"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NYSE.csv" --strategy-args '{"horizon":48}' --model-name "time_series_library.DLinear" --model-hyper-params '{"d_ff": 64, "d_model": 32, "lr": 0.01, "horizon": 48, "seq_len": 36}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "NYSE/DLinear"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NYSE.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.DLinear" --model-hyper-params '{"d_ff": 64, "d_model": 32, "lr": 0.01, "horizon": 60, "seq_len": 36}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "NYSE/DLinear"

