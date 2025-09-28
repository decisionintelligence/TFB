python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 24}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 0, "batch_size": 32, "hidden_size": 128, "horizon": 24, "lr": 0.001, "norm": true, "seq_len": 36}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Wike2000/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 36}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 0, "batch_size": 32, "hidden_size": 64, "horizon": 36, "lr": 0.001, "norm": true, "seq_len": 36}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Wike2000/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 48}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 0, "batch_size": 32, "hidden_size": 64, "horizon": 48, "lr": 0.001, "norm": true, "seq_len": 36}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Wike2000/Amplifier"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 60}' --model-name "amplifier.Amplifier" --model-hyper-params '{"SCI": 0, "batch_size": 32, "hidden_size": 64, "horizon": 60, "lr": 0.001, "norm": true, "seq_len": 36}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Wike2000/Amplifier"

