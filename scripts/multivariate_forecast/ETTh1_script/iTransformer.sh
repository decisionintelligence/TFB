python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"horizon":96}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"d_ff": 256, "d_model": 256, "e_layers": 2, "horizon": 96, "seq_len": 96}' --adapter "transformer_adapter"  --gpus 3  --num-workers 1  --timeout 60000  --save-path "ETTh1/iTransformer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"horizon":192}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"d_ff": 256, "d_model": 256, "e_layers": 2, "horizon": 192, "seq_len": 512}' --adapter "transformer_adapter"  --gpus 4  --num-workers 1  --timeout 60000  --save-path "ETTh1/iTransformer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"horizon":336}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"d_ff": 128, "d_model": 128, "e_layers": 2, "horizon": 336, "seq_len": 512}' --adapter "transformer_adapter"  --gpus 5  --num-workers 1  --timeout 60000  --save-path "ETTh1/iTransformer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"horizon":720}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"d_ff": 512, "d_model": 512, "e_layers": 2, "horizon": 720, "seq_len": 96}' --adapter "transformer_adapter"  --gpus 7  --num-workers 1  --timeout 60000  --save-path "ETTh1/iTransformer"






