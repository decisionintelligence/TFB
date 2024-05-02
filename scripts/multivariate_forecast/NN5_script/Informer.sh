python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon":24}' --model-name "time_series_library.Informer" --model-hyper-params '{"d_ff": 512, "d_model": 256, "horizon": 24, "seq_len": 104}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "NN5/Informer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon":36}' --model-name "time_series_library.Informer" --model-hyper-params '{"d_ff": 512, "d_model": 256, "horizon": 36, "seq_len": 104}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "NN5/Informer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon":48}' --model-name "time_series_library.Informer" --model-hyper-params '{"d_ff": 512, "d_model": 256, "horizon": 48, "seq_len": 104}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "NN5/Informer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.Informer" --model-hyper-params '{"d_ff": 512, "d_model": 256, "horizon": 60, "seq_len": 104}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "NN5/Informer"

