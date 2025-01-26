python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Covid-19.csv" --strategy-args '{"horizon": 24}' --model-name "time_series_library.Nonstationary_Transformer" --model-hyper-params '{"d_ff": 2048, "d_model": 512, "dropout": 0.05, "factor": 3, "horizon": 24, "norm": true, "p_hidden_dims": [32, 32], "p_hidden_layers": 2, "seq_len": 36}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "Covid-19/Nonstationary_Transformer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Covid-19.csv" --strategy-args '{"horizon": 36}' --model-name "time_series_library.Nonstationary_Transformer" --model-hyper-params '{"d_ff": 256, "d_model": 128, "dropout": 0.05, "factor": 3, "horizon": 36, "norm": true, "p_hidden_dims": [32, 32], "p_hidden_layers": 2, "seq_len": 36}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "Covid-19/Nonstationary_Transformer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Covid-19.csv" --strategy-args '{"horizon": 48}' --model-name "time_series_library.Nonstationary_Transformer" --model-hyper-params '{"d_ff": 256, "d_model": 128, "dropout": 0.05, "factor": 3, "horizon": 48, "norm": true, "p_hidden_dims": [32, 32], "p_hidden_layers": 2, "seq_len": 36}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "Covid-19/Nonstationary_Transformer"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Covid-19.csv" --strategy-args '{"horizon": 60}' --model-name "time_series_library.Nonstationary_Transformer" --model-hyper-params '{"d_ff": 256, "d_model": 128, "dropout": 0.05, "factor": 3, "horizon": 60, "norm": true, "p_hidden_dims": [32, 32], "p_hidden_layers": 2, "seq_len": 36}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "Covid-19/Nonstationary_Transformer"

