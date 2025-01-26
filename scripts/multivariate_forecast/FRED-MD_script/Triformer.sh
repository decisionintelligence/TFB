python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "FRED-MD.csv" --strategy-args '{"horizon": 24}' --model-name "time_series_library.Triformer" --model-hyper-params '{"d_ff": 64, "d_model": 32, "horizon": 24, "norm": true, "seq_len": 36}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "FRED-MD/Triformer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "FRED-MD.csv" --strategy-args '{"horizon": 36}' --model-name "time_series_library.Triformer" --model-hyper-params '{"d_ff": 64, "d_model": 32, "horizon": 36, "norm": true, "seq_len": 36}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "FRED-MD/Triformer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "FRED-MD.csv" --strategy-args '{"horizon": 48}' --model-name "time_series_library.Triformer" --model-hyper-params '{"d_ff": 64, "d_model": 32, "horizon": 48, "norm": true, "seq_len": 36}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "FRED-MD/Triformer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "FRED-MD.csv" --strategy-args '{"horizon": 60}' --model-name "time_series_library.Triformer" --model-hyper-params '{"d_ff": 64, "d_model": 32, "horizon": 60, "norm": true, "seq_len": 104}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "FRED-MD/Triformer"

