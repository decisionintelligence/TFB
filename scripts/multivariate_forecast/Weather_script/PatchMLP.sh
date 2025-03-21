python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Weather.csv" --strategy-args '{"horizon": 96}' --model-name "patchmlp.PatchMLP" --model-hyper-params '{"data": "custom", "features": "M", "seq_len": 96, "enc_in": 21}' --deterministic "full" --gpus 0 --num-workers 1 --timeout 60000 --save-path "Weather/PatchMLP"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Weather.csv" --strategy-args '{"horizon": 192}' --model-name "patchmlp.PatchMLP" --model-hyper-params '{"data": "custom", "features": "M", "seq_len": 96, "enc_in": 21}' --deterministic "full" --gpus 0 --num-workers 1 --timeout 60000 --save-path "Weather/PatchMLP"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Weather.csv" --strategy-args '{"horizon": 336}' --model-name "patchmlp.PatchMLP" --model-hyper-params '{"data": "custom", "features": "M", "seq_len": 96, "enc_in": 21}' --deterministic "full" --gpus 0 --num-workers 1 --timeout 60000 --save-path "Weather/PatchMLP"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Weather.csv" --strategy-args '{"horizon": 720}' --model-name "patchmlp.PatchMLP" --model-hyper-params '{"data": "custom", "features": "M", "seq_len": 96, "enc_in": 21}' --deterministic "full" --gpus 0 --num-workers 1 --timeout 60000 --save-path "Weather/PatchMLP"

