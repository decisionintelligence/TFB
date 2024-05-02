python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon":24}' --model-name "darts.TCNModel" --model-hyper-params '{"input_chunk_length": 104, "n_epochs": 100, "output_chunk_length": 24}' --gpus 0  --num-workers 1  --timeout 60000  --save-path "NN5/TCN"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon":36}' --model-name "darts.TCNModel" --model-hyper-params '{"input_chunk_length": 104, "n_epochs": 100, "output_chunk_length": 36}' --gpus 0  --num-workers 1  --timeout 60000  --save-path "NN5/TCN"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon":48}' --model-name "darts.TCNModel" --model-hyper-params '{"input_chunk_length": 104, "n_epochs": 100, "output_chunk_length": 48}' --gpus 0  --num-workers 1  --timeout 60000  --save-path "NN5/TCN"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon":60}' --model-name "darts.TCNModel" --model-hyper-params '{"input_chunk_length": 104, "n_epochs": 100, "output_chunk_length": 60}' --gpus 0  --num-workers 1  --timeout 60000  --save-path "NN5/TCN"

