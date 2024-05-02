python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Traffic.csv" --strategy-args '{"horizon":96}' --model-name "time_series_library.Triformer" --model-hyper-params '{"d_ff": 64, "d_model": 64, "horizon": 96, "seq_len": 96}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Traffic/Triformer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Traffic.csv" --strategy-args '{"horizon":192}' --model-name "time_series_library.Triformer" --model-hyper-params '{"d_ff": 64, "d_model": 64, "horizon": 192, "seq_len": 96}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Traffic/Triformer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Traffic.csv" --strategy-args '{"horizon":336}' --model-name "time_series_library.Triformer" --model-hyper-params '{"d_ff": 64, "d_model": 64, "horizon": 336, "seq_len": 96}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Traffic/Triformer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Traffic.csv" --strategy-args '{"horizon":720}' --model-name "time_series_library.Triformer" --model-hyper-params '{"batch_size": 4, "d_ff": 64, "d_model": 64, "horizon": 720, "seq_len": 96}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Traffic/Triformer"

