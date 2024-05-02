python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon":24}' --model-name "darts.RegressionModel" --model-hyper-params '{"output_chunk_length": 1}' --gpus 0  --num-workers 1  --timeout 60000  --save-path "NN5/regressionmodel"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon":36}' --model-name "darts.RegressionModel" --model-hyper-params '{"output_chunk_length": 1}' --gpus 0  --num-workers 1  --timeout 60000  --save-path "NN5/regressionmodel"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon":48}' --model-name "darts.RegressionModel" --model-hyper-params '{"output_chunk_length": 1}' --gpus 0  --num-workers 1  --timeout 60000  --save-path "NN5/regressionmodel"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "NN5.csv" --strategy-args '{"horizon":60}' --model-name "darts.RegressionModel" --model-hyper-params '{"output_chunk_length": 1}' --gpus 0  --num-workers 1  --timeout 60000  --save-path "NN5/regressionmodel"

