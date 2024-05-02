python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "AQWan.csv" --strategy-args '{"horizon":96}' --model-name "time_series_library.Informer" --model-hyper-params '{"d_ff": 512, "d_model": 256, "horizon": 96, "seq_len": 96}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "AQWan/Informer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "AQWan.csv" --strategy-args '{"horizon":192}' --model-name "time_series_library.Informer" --model-hyper-params '{"d_ff": 64, "d_model": 32, "horizon": 192, "seq_len": 512}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "AQWan/Informer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "AQWan.csv" --strategy-args '{"horizon":336}' --model-name "time_series_library.Informer" --model-hyper-params '{"d_ff": 64, "d_model": 32, "horizon": 336, "seq_len": 512}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "AQWan/Informer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "AQWan.csv" --strategy-args '{"horizon":720}' --model-name "time_series_library.Informer" --model-hyper-params '{"d_ff": 64, "d_model": 32, "horizon": 720, "seq_len": 512}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "AQWan/Informer"

