python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Traffic.csv" --strategy-args '{"horizon": 96}' --adapter "transformer_adapter" --model-name "time_series_library.Informer" --model-hyper-params '{"d_ff": 2048, "d_model": 512, "horizon": 96, "norm": true, "seq_len": 96}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Traffic/Informer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Traffic.csv" --strategy-args '{"horizon": 192}' --adapter "transformer_adapter" --model-name "time_series_library.Informer" --model-hyper-params '{"d_ff": 2048, "d_model": 512, "horizon": 512, "norm": true, "seq_len": 96}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Traffic/Informer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Traffic.csv" --strategy-args '{"horizon": 336}' --adapter "transformer_adapter" --model-name "time_series_library.Informer" --model-hyper-params '{"d_ff": 2048, "d_model": 512, "horizon": 336, "norm": true, "seq_len": 96}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Traffic/Informer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Traffic.csv" --strategy-args '{"horizon": 720}' --adapter "transformer_adapter" --model-name "time_series_library.Informer" --model-hyper-params '{"batch_size": 4, "d_ff": 2048, "d_model": 512, "horizon": 512, "norm": true, "seq_len": 96}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Traffic/Informer"

