python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "FRED-MD.csv" --strategy-args '{"horizon": 24}' --adapter "transformer_adapter" --model-name "time_series_library.Informer" --model-hyper-params '{"d_ff": 256, "d_model": 128, "horizon": 24, "norm": true, "seq_len": 36}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FRED-MD/Informer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "FRED-MD.csv" --strategy-args '{"horizon": 36}' --adapter "transformer_adapter" --model-name "time_series_library.Informer" --model-hyper-params '{"d_ff": 256, "d_model": 128, "horizon": 36, "norm": true, "seq_len": 36}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FRED-MD/Informer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "FRED-MD.csv" --strategy-args '{"horizon": 48}' --adapter "transformer_adapter" --model-name "time_series_library.Informer" --model-hyper-params '{"d_ff": 256, "d_model": 128, "horizon": 48, "norm": true, "seq_len": 36}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FRED-MD/Informer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "FRED-MD.csv" --strategy-args '{"horizon": 60}' --adapter "transformer_adapter" --model-name "time_series_library.Informer" --model-hyper-params '{"d_ff": 256, "d_model": 128, "horizon": 60, "norm": true, "seq_len": 36}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FRED-MD/Informer"

