python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon": 96}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"d_ff": 64, "d_model": 32, "horizon": 96, "norm": true, "seq_len": 336}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTm2/PatchTST"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon": 192}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"d_ff": 64, "d_model": 32, "horizon": 192, "norm": true, "seq_len": 336}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTm2/PatchTST"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon": 336}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"batch_size": 64, "d_ff": 512, "d_model": 128, "horizon": 336, "lr": 0.0001, "norm": true, "seq_len": 512}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTm2/PatchTST"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon": 720}' --model-name "time_series_library.PatchTST" --model-hyper-params '{"d_ff": 64, "d_model": 32, "horizon": 720, "norm": true, "seq_len": 336}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTm2/PatchTST"

