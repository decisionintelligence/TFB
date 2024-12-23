python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 24}' --model-name "darts.RNNModel" --model-hyper-params '{"input_chunk_length": 96, "norm": true, "output_chunk_length": 336}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Wike2000/RNNModel"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 36}' --model-name "darts.RNNModel" --model-hyper-params '{"input_chunk_length": 96, "norm": true, "output_chunk_length": 336}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Wike2000/RNNModel"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 48}' --model-name "darts.RNNModel" --model-hyper-params '{"input_chunk_length": 96, "norm": true, "output_chunk_length": 336}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Wike2000/RNNModel"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon": 60}' --model-name "darts.RNNModel" --model-hyper-params '{"input_chunk_length": 96, "norm": true, "output_chunk_length": 336}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Wike2000/RNNModel"

