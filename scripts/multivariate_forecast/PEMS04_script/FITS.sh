python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS04.csv" --strategy-args '{"horizon": 96}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 6, "batch_size": 64, "horizon": 96, "loss": "MSE", "lr": 0.0001, "norm": true, "patience": 5, "seq_len": 512}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "PEMS04/FITS"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS04.csv" --strategy-args '{"horizon": 192}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 6, "batch_size": 64, "horizon": 192, "loss": "MSE", "lr": 0.0001, "norm": true, "patience": 5, "seq_len": 512}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "PEMS04/FITS"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS04.csv" --strategy-args '{"horizon": 336}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 6, "batch_size": 64, "horizon": 336, "loss": "MSE", "lr": 0.0001, "norm": true, "patience": 5, "seq_len": 512}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "PEMS04/FITS"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS04.csv" --strategy-args '{"horizon": 720}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 6, "batch_size": 64, "horizon": 720, "loss": "MSE", "lr": 0.0001, "norm": true, "patience": 5, "seq_len": 512}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "PEMS04/FITS"

