python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon":96}' --model-name "darts.RegressionModel"  --model-hyper-params '{"output_chunk_length": 96}' --gpus 0  --num-workers 1  --timeout 60000  --save-path "Electricity/regressionmodel"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon":192}' --model-name "darts.RegressionModel" --model-hyper-params '{"output_chunk_length": 192}' --gpus 0  --num-workers 1  --timeout 60000  --save-path "Electricity/regressionmodel"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon":336}' --model-name "darts.RegressionModel" --model-hyper-params '{"output_chunk_length": 1}' --gpus 0  --num-workers 1  --timeout 60000  --save-path "Electricity/regressionmodel"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Electricity.csv" --strategy-args '{"horizon":720}' --model-name "darts.RegressionModel" --model-hyper-params '{"output_chunk_length": 1}' --gpus 0  --num-workers 1  --timeout 60000  --save-path "Electricity/regressionmodel"

