python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon":24}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"factor": 3, "lr": 0.0005, "horizon": 24, "seq_len": 36, "d_ff": 2048, "d_model": 512}' --adapter "transformer_adapter"  --gpus 3  --num-workers 1  --timeout 60000  --save-path "Wike2000/iTransformer"&

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon":36}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"d_ff": 2048, "lr": 0.0005, "d_model": 512, "horizon": 36, "seq_len": 36}' --adapter "transformer_adapter"  --gpus 4  --num-workers 1  --timeout 60000  --save-path "Wike2000/iTransformer"&

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon":48}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"batch_size": 8, "factor": 3, "e_layers": 2, "lr": 0.0005, "horizon": 48, "seq_len": 36, "d_ff": 2048, "d_model": 512}' --adapter "transformer_adapter"  --gpus 5  --num-workers 1  --timeout 60000  --save-path "Wike2000/iTransformer"&

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"factor": 3, "lr": 0.0005, "horizon": 60, "seq_len": 104, "d_ff": 2048, "d_model": 512}' --adapter "transformer_adapter"  --gpus 6  --num-workers 1  --timeout 60000  --save-path "Wike2000/iTransformer"&



