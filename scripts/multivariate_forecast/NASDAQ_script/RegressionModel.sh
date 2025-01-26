python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NASDAQ.csv" --strategy-args '{"horizon": 24}' --model-name "darts.RegressionModel" --model-hyper-params '{"lags": 96, "norm": true, "output_chunk_length": 24}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "NASDAQ/RegressionModel"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NASDAQ.csv" --strategy-args '{"horizon": 36}' --model-name "darts.RegressionModel" --model-hyper-params '{"lags": 96, "norm": true, "output_chunk_length": 36}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "NASDAQ/RegressionModel"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NASDAQ.csv" --strategy-args '{"horizon": 48}' --model-name "darts.RegressionModel" --model-hyper-params '{"lags": 96, "norm": true, "output_chunk_length": 48}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "NASDAQ/RegressionModel"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NASDAQ.csv" --strategy-args '{"horizon": 60}' --model-name "darts.RegressionModel" --model-hyper-params '{"lags": 96, "norm": true, "output_chunk_length": 60}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "NASDAQ/RegressionModel"

