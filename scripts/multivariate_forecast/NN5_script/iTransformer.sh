python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon": 24}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"d_ff": 2048, "d_model": 512, "factor": 3, "horizon": 24, "lr": 0.0005, "norm": true, "seq_len": 104}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "NN5/iTransformer" --adapter "transformer_adapter"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon": 36}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"d_ff": 2048, "d_model": 512, "horizon": 36, "lr": 0.0005, "norm": true, "seq_len": 104}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "NN5/iTransformer" --adapter "transformer_adapter"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon": 48}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"d_ff": 2048, "d_model": 512, "factor": 3, "horizon": 48, "lr": 0.0005, "norm": true, "seq_len": 104}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "NN5/iTransformer" --adapter "transformer_adapter"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon": 60}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"d_ff": 2048, "d_model": 512, "factor": 3, "horizon": 60, "lr": 0.0005, "norm": true, "seq_len": 104}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "NN5/iTransformer" --adapter "transformer_adapter"

