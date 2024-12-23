python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS-BAY.csv" --strategy-args '{"horizon": 96}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"batch_size": 128, "d_ff": 256, "d_model": 128, "factor": 3, "horizon": 96, "norm": true, "seq_len": 512, "top_k": 5}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "PEMS-BAY/TimesNet" --adapter "transformer_adapter"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS-BAY.csv" --strategy-args '{"horizon": 192}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"batch_size": 128, "d_ff": 256, "d_model": 128, "factor": 3, "horizon": 192, "norm": true, "seq_len": 512, "top_k": 5}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "PEMS-BAY/TimesNet" --adapter "transformer_adapter"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS-BAY.csv" --strategy-args '{"horizon": 336}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"batch_size": 64, "d_ff": 512, "d_model": 256, "factor": 3, "horizon": 336, "norm": true, "seq_len": 512, "top_k": 5}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "PEMS-BAY/TimesNet" --adapter "transformer_adapter"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS-BAY.csv" --strategy-args '{"horizon": 720}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"batch_size": 128, "d_ff": 256, "d_model": 256, "factor": 3, "horizon": 720, "norm": true, "seq_len": 512, "top_k": 5}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "PEMS-BAY/TimesNet" --adapter "transformer_adapter"

