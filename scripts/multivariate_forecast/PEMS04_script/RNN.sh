python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS04.csv" --strategy-args '{"horizon":96}' --model-name "darts.RNNModel" --gpus 0  --num-workers 1  --timeout 60000  --save-path "PEMS04/RNN"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS04.csv" --strategy-args '{"horizon":192}' --model-name "darts.RNNModel" --gpus 0  --num-workers 1  --timeout 60000  --save-path "PEMS04/RNN"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS04.csv" --strategy-args '{"horizon":336}' --model-name "darts.RNNModel" --gpus 0  --num-workers 1  --timeout 60000  --save-path "PEMS04/RNN"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS04.csv" --strategy-args '{"horizon":720}' --model-name "darts.RNNModel" --gpus 0  --num-workers 1  --timeout 60000  --save-path "PEMS04/RNN"

