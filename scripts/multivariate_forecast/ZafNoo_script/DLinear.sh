python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ZafNoo.csv" --strategy-args '{"horizon":96}' --model-name "time_series_library.DLinear" --model-hyper-params '{"batch_size": 16, "lr": 0.005, "horizon": 96, "seq_len": 336, "d_ff": 2048, "d_model": 512}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZafNoo/DLinear"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ZafNoo.csv" --strategy-args '{"horizon":192}' --model-name "time_series_library.DLinear" --model-hyper-params '{"d_ff": 64, "d_model": 32, "lr": 0.005, "horizon": 192, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZafNoo/DLinear"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ZafNoo.csv" --strategy-args '{"horizon":336}' --model-name "time_series_library.DLinear" --model-hyper-params '{"d_ff": 64, "d_model": 32, "lr": 0.005, "horizon": 336, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZafNoo/DLinear"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ZafNoo.csv" --strategy-args '{"horizon":720}' --model-name "time_series_library.DLinear" --model-hyper-params '{"d_ff": 64, "d_model": 32, "lr": 0.005, "horizon": 720, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ZafNoo/DLinear"

