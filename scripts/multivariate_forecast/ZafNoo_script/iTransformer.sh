python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ZafNoo.csv" --strategy-args '{"horizon": 96}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"batch_size": 64, "d_ff": 512, "d_model": 16, "e_layers": 1, "horizon": 96, "lr": 0.005, "norm": true, "seq_len": 336}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "ZafNoo/iTransformer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ZafNoo.csv" --strategy-args '{"horizon": 192}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"batch_size": 64, "d_ff": 1024, "d_model": 16, "e_layers": 1, "horizon": 192, "lr": 0.005, "norm": true, "seq_len": 336}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "ZafNoo/iTransformer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ZafNoo.csv" --strategy-args '{"horizon": 336}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"batch_size": 64, "d_ff": 256, "d_model": 16, "e_layers": 1, "horizon": 336, "lr": 0.005, "norm": true, "seq_len": 336}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "ZafNoo/iTransformer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ZafNoo.csv" --strategy-args '{"horizon": 720}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"d_ff": 512, "d_model": 16, "e_layers": 1, "horizon": 720, "lr": 0.005, "norm": true, "seq_len": 512}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "ZafNoo/iTransformer"

