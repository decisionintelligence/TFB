python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"horizon": 96}'  --model-name "timekan.TimeKAN"   --model-hyper-params '{"batch_size": 128, "d_ff": 32, "d_model": 16, "down_sampling_layer": 2, "down_sampling_window": 2, "e_layers": 2, "horizon": 96, "lr": 0.01, "norm": true, "num_epochs": 10, "patience": 10, "seq_len": 96, "begin_order": 0}' --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTh1/TimeKAN"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"horizon": 192}' --model-name "timekan.TimeKAN" --model-hyper-params '{"batch_size": 128, "d_ff": 32, "d_model": 16, "down_sampling_layer": 1, "down_sampling_window": 2, "e_layers": 2, "horizon": 192, "lr": 0.01, "norm": true, "num_epochs": 10, "patience": 10, "seq_len": 96, "begin_order": 0}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTh1/TimeKAN"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"horizon": 336}' --model-name "timekan.TimeKAN" --model-hyper-params '{"batch_size": 128, "d_ff": 32, "d_model": 16, "down_sampling_layer": 2, "down_sampling_window": 2, "e_layers": 2, "horizon": 336, "lr": 0.01, "norm": true, "num_epochs": 10, "patience": 10, "seq_len": 96, "begin_order": 0}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTh1/TimeKAN"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"horizon": 720}' --model-name "timekan.TimeKAN" --model-hyper-params '{"batch_size": 128, "d_ff": 32, "d_model": 16, "down_sampling_layer": 3, "down_sampling_window": 2, "e_layers": 2, "horizon": 720, "lr": 0.01, "norm": true, "num_epochs": 10, "patience": 10, "seq_len": 96, "begin_order": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTh1/TimeKAN"



