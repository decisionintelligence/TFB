python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS04.csv" --strategy-args '{"pred_len":96}' --model-name "time_series_library.DLinear" --model-hyper-params '{"d_ff": 512, "d_model": 256, "pred_len": 96, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "PEMS04/DLinear"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS04.csv" --strategy-args '{"pred_len":192}' --model-name "time_series_library.DLinear" --model-hyper-params '{"batch_size": 16, "lr": 0.005, "pred_len": 192, "seq_len": 336, "d_ff": 2048, "d_model": 512}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "PEMS04/DLinear"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS04.csv" --strategy-args '{"pred_len":336}' --model-name "time_series_library.DLinear" --model-hyper-params '{"batch_size": 16, "lr": 0.005, "pred_len": 336, "seq_len": 336, "d_ff": 2048, "d_model": 512}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "PEMS04/DLinear"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS04.csv" --strategy-args '{"pred_len":720}' --model-name "time_series_library.DLinear" --model-hyper-params '{"d_ff": 512, "d_model": 256, "pred_len": 720, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "PEMS04/DLinear"

