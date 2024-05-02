python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --strategy-args '{"horizon":96}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"d_ff": 256, "d_model": 128, "horizon": 96, "seq_len": 96}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Exchange/PatchTST"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --strategy-args '{"horizon":192}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"d_ff": 256, "d_model": 128, "horizon": 192, "seq_len": 96}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Exchange/PatchTST"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --strategy-args '{"horizon":336}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"e_layers": 2, "factor": 3, "num_epochs": 1, "horizon": 336, "seq_len": 96, "d_ff": 2048, "d_model": 512}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Exchange/PatchTST"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --strategy-args '{"horizon":720}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"d_ff": 256, "d_model": 128, "horizon": 720, "seq_len": 96}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Exchange/PatchTST"

