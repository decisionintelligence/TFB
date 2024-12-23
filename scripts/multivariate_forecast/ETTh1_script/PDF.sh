python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"horizon": 96}' --model-name "pdf.PDF" --model-hyper-params '{"batch_size": 128, "d_ff": 128, "d_model": 16, "dropout": 0.25, "e_layers": 3, "fc_dropout": 0.15, "horizon": 96, "kernel_list": [3, 7, 11], "n_head": 4, "norm": true, "patch_len": [1], "patience": 10, "pc_start": 0.2, "period": [24], "seq_len": 512, "stride": [1], "train_epochs": 100}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTh1/PDF"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"horizon": 192}' --model-name "pdf.PDF" --model-hyper-params '{"batch_size": 128, "d_ff": 128, "d_model": 16, "dropout": 0.25, "e_layers": 3, "fc_dropout": 0.15, "horizon": 192, "kernel_list": [3, 7, 11], "n_head": 4, "norm": true, "patch_len": [1], "patience": 10, "pc_start": 0.2, "period": [24], "seq_len": 512, "stride": [1], "train_epochs": 100}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTh1/PDF"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"horizon": 336}' --model-name "pdf.PDF" --model-hyper-params '{"batch_size": 128, "d_ff": 128, "d_model": 16, "dropout": 0.5, "e_layers": 3, "fc_dropout": 0.25, "horizon": 336, "kernel_list": [3, 7, 11], "learning_rate": 0.00025, "n_head": 4, "norm": true, "patch_len": [1], "patience": 10, "period": [24], "seq_len": 512, "stride": [1], "train_epochs": 100}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTh1/PDF"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"horizon": 720}' --model-name "pdf.PDF" --model-hyper-params '{"batch_size": 64, "d_ff": 256, "d_model": 128, "dropout": 0.3, "e_layers": 1, "fc_dropout": 0.15, "horizon": 720, "kernel_list": [3, 7, 9, 11], "n_head": 32, "norm": true, "patch_len": [1], "patience": 5, "pc_start": 0.2, "period": [24], "seq_len": 96, "stride": [1], "train_epochs": 20}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTh1/PDF"

