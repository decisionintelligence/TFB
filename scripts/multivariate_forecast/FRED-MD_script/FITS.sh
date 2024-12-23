python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "FRED-MD.csv" --strategy-args '{"horizon": 24}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 6, "batch_size": 64, "horizon": 24, "loss": "MSE", "lr": 0.005, "norm": true, "patience": 20, "seq_len": 104}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FRED-MD/FITS"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "FRED-MD.csv" --strategy-args '{"horizon": 36}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 6, "batch_size": 64, "horizon": 36, "loss": "MSE", "lr": 0.005, "norm": true, "patience": 20, "seq_len": 104}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FRED-MD/FITS"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "FRED-MD.csv" --strategy-args '{"horizon": 48}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 6, "batch_size": 64, "horizon": 48, "loss": "MSE", "lr": 0.005, "norm": true, "patience": 20, "seq_len": 104}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FRED-MD/FITS"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "FRED-MD.csv" --strategy-args '{"horizon": 60}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 6, "batch_size": 64, "horizon": 60, "loss": "MSE", "lr": 0.005, "norm": true, "patience": 20, "seq_len": 104}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "FRED-MD/FITS"

