python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NYSE.csv" --strategy-args '{"pred_len":24}' --model-name "time_series_library.Linear" --model-hyper-params '{"d_ff": 64, "d_model": 32, "lr": 0.01, "pred_len": 24, "seq_len": 104}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "NYSE/Linear"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NYSE.csv" --strategy-args '{"pred_len":36}' --model-name "time_series_library.Linear" --model-hyper-params '{"d_ff": 64, "d_model": 32, "lr": 0.01, "pred_len": 36, "seq_len": 36}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "NYSE/Linear"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NYSE.csv" --strategy-args '{"pred_len":48}' --model-name "time_series_library.Linear" --model-hyper-params '{"d_ff": 64, "d_model": 32, "lr": 0.01, "pred_len": 48, "seq_len": 104}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "NYSE/Linear"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NYSE.csv" --strategy-args '{"pred_len":60}' --model-name "time_series_library.Linear" --model-hyper-params '{"batch_size": 32, "lr": 0.01, "pred_len": 60, "seq_len": 104, "d_ff": 2048, "d_model": 512}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "NYSE/Linear"

