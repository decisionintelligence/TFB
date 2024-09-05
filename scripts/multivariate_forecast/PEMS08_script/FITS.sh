python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":96}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 6, "loss": "MSE", "batch_size": 64,  "horizon": 96, "seq_len": 512, "patience": 10, "lr": 0.0005}'  --gpus 1  --num-workers 1  --timeout 60000  --save-path "PEMS08/FITS"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":192}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 6, "loss": "MSE", "batch_size": 64,  "horizon": 192, "seq_len": 512, "patience": 10, "lr": 0.0005}'  --gpus 4  --num-workers 1  --timeout 60000  --save-path "PEMS08/FITS"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":336}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 6, "loss": "MSE", "batch_size": 64,  "horizon": 336, "seq_len": 512, "patience": 10, "lr": 0.0005}'  --gpus 5  --num-workers 1  --timeout 60000  --save-path "PEMS08/FITS"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS08.csv" --strategy-args '{"horizon":720}' --model-name "fits.FITS" --model-hyper-params '{"H_order": 6, "loss": "MSE", "batch_size": 64,  "horizon": 720, "seq_len": 512, "patience": 10, "lr": 0.0005}'  --gpus 6  --num-workers 1  --timeout 60000  --save-path "PEMS08/FITS"

