python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon": 96}' --model-name "darts.RegressionModel" --model-hyper-params '{"lags": 96, "norm": true, "output_chunk_length": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "PEMS08/RegressionModel"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon": 192}' --model-name "darts.RegressionModel" --model-hyper-params '{"lags": 96, "norm": true, "output_chunk_length": 192}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "PEMS08/RegressionModel"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon": 336}' --model-name "darts.RegressionModel" --model-hyper-params '{"lags": 96, "norm": true, "output_chunk_length": 336}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "PEMS08/RegressionModel"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon": 720}' --model-name "darts.RegressionModel" --model-hyper-params '{"lags": 96, "norm": true, "output_chunk_length": 720}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "PEMS08/RegressionModel"

