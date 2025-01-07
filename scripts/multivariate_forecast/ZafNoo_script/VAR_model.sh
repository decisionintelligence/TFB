python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ZafNoo.csv" --strategy-args '{"horizon": 96}' --model-name "self_impl.VAR_model" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "ZafNoo/VAR_model"

