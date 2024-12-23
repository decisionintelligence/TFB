python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --strategy-args '{"horizon": 96}' --model-name "time_series_library.DLinear" --model-hyper-params '{"batch_size": 8, "d_ff": 2048, "d_model": 512, "horizon": 96, "lr": 0.005, "norm": true, "seq_len": 336}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Exchange/DLinear" --adapter "transformer_adapter"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --strategy-args '{"horizon": 192}' --model-name "time_series_library.DLinear" --model-hyper-params '{"batch_size": 64, "d_ff": 2048, "d_model": 512, "factor": 3, "horizon": 192, "norm": true, "period_len": 6, "seq_len": 96, "station_lr": 0.001}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Exchange/DLinear" --adapter "transformer_adapter"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --strategy-args '{"horizon": 336}' --model-name "time_series_library.DLinear" --model-hyper-params '{"d_ff": 2048, "d_model": 512, "factor": 3, "horizon": 336, "norm": true, "seq_len": 96}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Exchange/DLinear" --adapter "transformer_adapter"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --strategy-args '{"horizon": 720}' --model-name "time_series_library.DLinear" --model-hyper-params '{"batch_size": 64, "d_ff": 64, "d_model": 32, "horizon": 720, "norm": true, "period_len": 6, "seq_len": 96, "station_lr": 0.001}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Exchange/DLinear" --adapter "transformer_adapter"

