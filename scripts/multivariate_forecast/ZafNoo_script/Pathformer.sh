python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ZafNoo.csv" --strategy-args '{"horizon":96}' --model-name "pathformer.Pathformer" --model-hyper-params '{"k": 2, "horizon": 96, "seq_len": 336, "d_ff": 64, "d_model": 8,"num_nodes":11, "learning_rate":0.0005, "batch_size":256, "gpu":6, "patch_size_list":[[42, 24, 12, 16],[42, 28, 16, 12], [16, 28, 12, 42]]}'   --gpus 6  --num-workers 1  --timeout 60000  --save-path "ZafNoo/PathFormer"


python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ZafNoo.csv" --strategy-args '{"horizon":192}' --model-name "pathformer.Pathformer" --model-hyper-params '{"k": 2, "horizon": 192, "seq_len": 336, "d_ff": 64, "d_model": 8,"num_nodes":11, "learning_rate":0.0005, "batch_size":256, "gpu":6, "patch_size_list":[[42, 24, 12, 16],[42, 28, 16, 12], [16, 28, 12, 42]] }'   --gpus 6  --num-workers 1  --timeout 60000  --save-path "ZafNoo/PathFormer"


python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ZafNoo.csv" --strategy-args '{"horizon":336}' --model-name "pathformer.Pathformer" --model-hyper-params '{"k": 2, "horizon": 336, "seq_len": 336, "d_ff": 64, "d_model": 8,"num_nodes":11, "learning_rate":0.0005, "batch_size":256, "gpu":6, "patch_size_list":[[42, 24, 12, 16],[42, 28, 16, 12], [16, 28, 12, 42]] }'   --gpus 6  --num-workers 1  --timeout 60000  --save-path "ZafNoo/PathFormer"


python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ZafNoo.csv" --strategy-args '{"horizon":720}' --model-name "pathformer.Pathformer" --model-hyper-params '{"k": 2, "horizon": 720, "seq_len": 336, "d_ff": 64, "d_model": 8,"num_nodes":11, "learning_rate":0.0005, "batch_size":256, "gpu":6, "patch_size_list":[[42, 24, 12, 16],[42, 28, 16, 12], [16, 28, 12, 42]] }'   --gpus 6  --num-workers 1  --timeout 60000  --save-path "ZafNoo/PathFormer"

