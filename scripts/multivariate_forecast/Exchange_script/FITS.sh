python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --strategy-args '{"horizon": 96}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 6, "batch_size": 32, "horizon": 96, "loss": "MSE", "lr": 0.005, "norm": true, "patience": 10, "seq_len": 96}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Exchange/FITS"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --strategy-args '{"horizon": 192}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 6, "batch_size": 64, "horizon": 192, "loss": "MSE", "lr": 0.005, "norm": true, "patience": 10, "seq_len": 96}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Exchange/FITS"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --strategy-args '{"horizon": 336}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 6, "batch_size": 64, "horizon": 336, "loss": "MSE", "lr": 0.005, "norm": true, "patience": 10, "seq_len": 96}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Exchange/FITS"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --strategy-args '{"horizon": 720}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 6, "batch_size": 32, "horizon": 720, "loss": "MSE", "lr": 0.005, "norm": true, "patience": 20, "seq_len": 96}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Exchange/FITS"

