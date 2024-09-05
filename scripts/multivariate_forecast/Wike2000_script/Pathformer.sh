python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon":24}' --model-name "pathformer.Pathformer" --model-hyper-params '{"k": 1, "horizon": 24, "seq_len": 104, "d_ff": 8, "d_model": 4,"num_nodes":2000, "learning_rate":0.0002, "batch_size":16, "gpu":6, "patch_size_list":[[13, 4, 8, 2],[13, 8, 4, 2], [4, 8, 8, 4]], "lradj":"TST", "residual_connection":0}'  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Wike2000/PathFormer"


python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon":36}' --model-name "pathformer.Pathformer" --model-hyper-params '{"k": 1, "horizon": 36, "seq_len": 104, "d_ff": 8, "d_model": 4,"num_nodes":2000, "learning_rate":0.0002, "batch_size":16, "gpu":6, "patch_size_list":[[13, 4, 8, 2],[13, 8, 4, 2], [4, 8, 8, 4]],"lradj":"TST", "residual_connection":0}'  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Wike2000/PathFormer"


python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon":48}' --model-name "pathformer.Pathformer" --model-hyper-params '{"k": 1, "horizon": 48, "seq_len": 104, "d_ff": 8, "d_model": 4,"num_nodes":2000, "learning_rate":0.0002, "batch_size":16, "gpu":6, "patch_size_list":[[13, 4, 8, 2],[13, 8, 4, 2], [4, 8, 8, 4]], "lradj":"TST", "residual_connection":0}'  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Wike2000/PathFormer"


python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Wike2000.csv" --strategy-args '{"horizon":60}' --model-name "pathformer.Pathformer" --model-hyper-params '{"k": 1, "horizon": 60, "seq_len": 104, "d_ff": 8, "d_model": 4,"num_nodes":2000, "learning_rate":0.0002, "batch_size":16, "gpu":6, "patch_size_list":[[13, 4, 8, 2],[13, 8, 4, 2], [4, 8, 8, 4]], "lradj":"TST", "residual_connection":0}'  --gpus 0  --num-workers 1  --timeout 60000  --save-path "Wike2000/PathFormer"



