python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Solar.csv" --strategy-args '{"horizon":96}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"d_ff": 256, "lr": 0.0005, "d_model": 256, "e_layers": 1, "horizon": 96, "seq_len": 336}' --adapter "transformer_adapter"  --gpus 5  --num-workers 1  --timeout 60000  --save-path "Solar/iTransformer"&

#python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Solar.csv" --strategy-args '{"horizon":192}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"d_ff": 256, "d_model": 256, "e_layers": 1, "horizon": 192, "seq_len": 512}' --adapter "transformer_adapter"  --gpus 4  --num-workers 1  --timeout 60000  --save-path "Solar/iTransformer"&
#
#python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Solar.csv" --strategy-args '{"horizon":336}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"d_ff": 256, "lr": 0.0005, "d_model": 256, "e_layers": 1, "horizon": 336, "seq_len": 512}' --adapter "transformer_adapter"  --gpus 5  --num-workers 1  --timeout 60000  --save-path "Solar/iTransformer"&
#
#python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Solar.csv" --strategy-args '{"horizon":720}' --model-name "time_series_library.iTransformer" --model-hyper-params '{"d_ff": 256, "d_model": 256, "e_layers": 1, "horizon": 720, "seq_len": 512}' --adapter "transformer_adapter"  --gpus 1  --num-workers 1  --timeout 60000  --save-path "Solar/iTransformer"&


