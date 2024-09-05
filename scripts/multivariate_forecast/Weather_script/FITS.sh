python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Weather.csv" --strategy-args '{"horizon":96}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 12, "base_T": 144, "loss": "MSE", "batch_size": 128, "horizon": 96, "seq_len": 512, "patience": 10, "lr": 0.005}'   --gpus 1  --num-workers 1  --timeout 60000  --save-path "Weather/FITS"&

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Weather.csv" --strategy-args '{"horizon":192}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 12, "base_T": 144, "loss": "MSE", "batch_size": 128, "horizon": 192, "seq_len": 512, "patience": 10, "lr": 0.005}'   --gpus 1  --num-workers 1  --timeout 60000  --save-path "Weather/FITS"&

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Weather.csv" --strategy-args '{"horizon":336}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 8, "base_T": 144, "loss": "MSE", "batch_size": 128, "horizon": 336, "seq_len": 512, "patience": 10, "lr": 0.005}'   --gpus 3  --num-workers 1  --timeout 60000  --save-path "Weather/FITS"&

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Weather.csv" --strategy-args '{"horizon":720}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 12, "base_T": 144, "loss": "MSE", "batch_size": 128, "horizon": 720, "seq_len": 512, "patience": 10, "lr": 0.005}'   --gpus 3  --num-workers 1  --timeout 60000  --save-path "Weather/FITS"&

