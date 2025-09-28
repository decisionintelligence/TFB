python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon": 96}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 0, "batch_size": 32, "hidden_size": 64, "horizon": 96, "lr": 0.02, "norm": true, "seq_len": 96}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTh2/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon": 192}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 0, "batch_size": 32, "hidden_size": 64, "horizon": 192, "lr": 0.0005, "norm": true, "seq_len": 512}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTh2/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon": 336}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 0, "batch_size": 32, "hidden_size": 64, "horizon": 336, "lr": 0.0005, "norm": true, "seq_len": 336}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTh2/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon": 720}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 0, "batch_size": 32, "hidden_size": 64, "horizon": 720, "lr": 0.0001, "norm": true, "seq_len": 512}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTh2/Amplifier"

