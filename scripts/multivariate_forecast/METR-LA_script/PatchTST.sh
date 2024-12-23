python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "METR-LA.csv" --strategy-args '{"horizon": 96}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"d_ff": 1024, "d_model": 512, "horizon": 96, "norm": true, "seq_len": 512}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "METR-LA/PatchTST" --adapter "transformer_adapter"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "METR-LA.csv" --strategy-args '{"horizon": 192}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"d_ff": 512, "d_model": 256, "horizon": 192, "norm": true, "seq_len": 512}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "METR-LA/PatchTST" --adapter "transformer_adapter"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "METR-LA.csv" --strategy-args '{"horizon": 336}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"batch_size": 64, "d_ff": 512, "d_model": 128, "horizon": 336, "lr": 0.0001, "norm": true, "seq_len": 512}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "METR-LA/PatchTST" --adapter "transformer_adapter"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "METR-LA.csv" --strategy-args '{"horizon": 720}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"batch_size": 64, "d_ff": 512, "d_model": 256, "horizon": 720, "lr": 0.0005, "norm": true, "seq_len": 512}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "METR-LA/PatchTST" --adapter "transformer_adapter"

