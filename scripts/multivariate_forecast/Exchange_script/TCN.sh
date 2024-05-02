python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --strategy-args '{"horizon":96}' --model-name "darts.TCNModel" --model-hyper-params '{"input_chunk_length": 512, "n_epochs": 10, "output_chunk_length": 96}' --gpus 0  --num-workers 1  --timeout 60000  --save-path "Exchange/TCN"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --strategy-args '{"horizon":192}' --model-name "darts.TCNModel" --model-hyper-params '{"input_chunk_length": 512, "n_epochs": 10, "output_chunk_length": 192}' --gpus 0  --num-workers 1  --timeout 60000  --save-path "Exchange/TCN"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --strategy-args '{"horizon":336}' --model-name "darts.TCNModel" --model-hyper-params '{"input_chunk_length": 512, "n_epochs": 100, "output_chunk_length": 336}' --gpus 0  --num-workers 1  --timeout 60000  --save-path "Exchange/TCN"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --strategy-args '{"horizon":720}' --model-name "darts.TCNModel" --model-hyper-params '{"input_chunk_length": 960, "n_epochs": 100, "output_chunk_length": 720}' --gpus 0  --num-workers 1  --timeout 60000  --save-path "Exchange/TCN"

