python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon": 96}' --model-name "darts.RNNModel" --model-hyper-params '{"input_chunk_length": 96, "norm": true, "output_chunk_length": 336}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTh2/RNNModel"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon": 192}' --model-name "darts.RNNModel" --model-hyper-params '{"input_chunk_length": 96, "norm": true, "output_chunk_length": 336}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTh2/RNNModel"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon": 336}' --model-name "darts.RNNModel" --model-hyper-params '{"input_chunk_length": 96, "norm": true, "output_chunk_length": 336}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTh2/RNNModel"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh2.csv" --strategy-args '{"horizon": 720}' --model-name "darts.RNNModel" --model-hyper-params '{"input_chunk_length": 96, "norm": true, "output_chunk_length": 336}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTh2/RNNModel"

