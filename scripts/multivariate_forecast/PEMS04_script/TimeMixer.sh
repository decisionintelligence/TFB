#python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS04.csv" --strategy-args '{"horizon":96}' --model-name "time_series_library.TimeMixer" --model-hyper-params '{"batch_size":8, "d_ff": 64, "d_model": 32, "e_layers": 3, "lr": 0.01,  "num_epochs": 20, "horizon": 96, "seq_len": 336,"down_sampling_layer": 3,"down_sampling_window": 2,"patience":10}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "PEMS04/TimeMixer"

#python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS04.csv" --strategy-args '{"horizon":192}' --model-name "time_series_library.TimeMixer" --model-hyper-params '{"batch_size":8, "d_ff": 64, "d_model": 32, "e_layers": 3, "lr": 0.01,  "num_epochs": 20, "horizon": 192, "seq_len": 336,"down_sampling_layer": 3,"down_sampling_window": 2,"patience":10}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "PEMS04/TimeMixer"

#python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS04.csv" --strategy-args '{"horizon":336}' --model-name "time_series_library.TimeMixer" --model-hyper-params '{"batch_size":8, "d_ff": 64, "d_model": 32, "e_layers": 3, "lr": 0.01,  "num_epochs": 20, "horizon": 336, "seq_len": 336,"down_sampling_layer": 3,"down_sampling_window": 2,"patience":10}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "PEMS04/TimeMixer"

#python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "PEMS04.csv" --strategy-args '{"horizon":720}' --model-name "time_series_library.TimeMixer" --model-hyper-params '{"batch_size":8, "d_ff": 64, "d_model": 32, "e_layers": 3, "lr": 0.01,  "num_epochs": 20, "horizon": 720, "seq_len": 512,"down_sampling_layer": 3,"down_sampling_window": 2,"patience":10}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "PEMS04/TimeMixer"