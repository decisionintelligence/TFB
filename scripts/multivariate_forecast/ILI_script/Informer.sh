python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args '{"horizon":24}' --model-name "time_series_library.Informer" --model-hyper-params '{"factor": 3, "horizon": 24, "seq_len": 104, "d_ff": 2048, "d_model": 512}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ILI/Informer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args '{"horizon":36}' --model-name "time_series_library.Informer" --model-hyper-params '{"d_ff": 512, "d_model": 256, "horizon": 36, "seq_len": 104}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ILI/Informer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args '{"horizon":48}' --model-name "time_series_library.Informer" --model-hyper-params '{"factor": 3, "horizon": 48, "seq_len": 104, "d_ff": 2048, "d_model": 512}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ILI/Informer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.Informer" --model-hyper-params '{"factor": 3, "horizon": 60, "seq_len": 36, "d_ff": 2048, "d_model": 512}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ILI/Informer"

