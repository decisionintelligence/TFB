
python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_weekly.json"  --data-name-list "m4_weekly_dataset_180.csv" "m4_weekly_dataset_72.csv"   --model-name  "darts.LinearRegressionModel"   --model-hyper-params  '{"output_chunk_length":1}' --gpus 2  --num-workers 1 --timeout 60000  --save-path "for_validation_final"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_weekly.json"  --data-name-list "m4_weekly_dataset_180.csv" "m4_weekly_dataset_72.csv"   --model-name  "darts.LinearRegressionModel"    --gpus 2  --num-workers 1 --timeout 60000  --save-path "for_validation_final"


python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_weekly.json"  --data-name-list "m4_weekly_dataset_180.csv" "m4_weekly_dataset_72.csv"   --model-name  "time_series_library.Triformer"  --model-hyper-params '{"d_model":32,"d_ff":64, "seq_len":96,"pred_len":96, "batch_size":16, "lr":0.001, "num_epochs":5}'  --adapter "transformer_adapter" --gpus 2  --num-workers 1 --timeout 60000  --save-path "for_validation_final"


python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_weekly.json"  --data-name-list "m4_weekly_dataset_180.csv" "m4_weekly_dataset_72.csv"  --model-name  "darts.TCNModel" --model-hyper-params  '{"n_epochs":100}' --gpus 2  --num-workers 1 --timeout 60000  --save-path "for_validation_final"


python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json"  --data-name-list "ILI.csv" --strategy-args '{"pred_len":24}' --model-name "time_series_library.Triformer"   "time_series_library.DLinear"  "time_series_library.TimesNet"   --model-hyper-params '{"d_model":32, "batch_size":32,"d_ff":64, "seq_len":96,"pred_len":96}' '{"d_model":32, "batch_size":32,"d_ff":64, "seq_len":96,"pred_len":96}' '{"d_model":32, "batch_size":32,"d_ff":64, "seq_len":96,"pred_len":96}' --adapter "transformer_adapter"  "transformer_adapter" "transformer_adapter" --gpus 7  --num-workers 1 --timeout 6000  --save-path "for_validation_final"


python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json"  --data-name-list "ILI.csv" --strategy-args '{"pred_len":24}' --model-name  "darts.TCNModel" --model-hyper-params  '{"n_epochs":100,"input_chunk_length":96,"output_chunk_length":24}'  --gpus 2  --num-workers 1 --timeout 60000  --save-path "for_validation_final"


python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json"  --data-name-list "ILI.csv"  "NN5.csv"  --strategy-args '{"pred_len":60}' --model-name  "self_implementation.VAR_model" --gpus 6 --num-workers 1 --timeout 100000 --save-path "for_validation_final"


python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args '{"pred_len":24}' --model-name "darts.RegressionModel" --gpus 0  --num-workers 1  --timeout 60000  --save-path "for_validation_final"



python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_weekly.json"  --data-name-list "m4_weekly_dataset_180.csv" "m4_weekly_dataset_72.csv" "m4_weekly_dataset_330.csv" "m4_weekly_dataset_332.csv" "m4_weekly_dataset_333.csv" "m4_weekly_dataset_334.csv" "m4_weekly_dataset_335.csv" "m4_weekly_dataset_337.csv" "m4_weekly_dataset_338.csv" "m4_weekly_dataset_339.csv"  --model-name  "darts.NBEATSModel" "darts.NHiTSModel" "darts.StatsForecastAutoETS" "darts.StatsForecastAutoCES" "darts.StatsForecastAutoTheta" "darts.AutoARIMA"  "darts.LinearRegressionModel"  "darts.RandomForest" "darts.KalmanForecaster" "darts.XGBModel"  "darts.NaiveMean"  "darts.NaiveSeasonal" "darts.NaiveDrift" "darts.NaiveMovingAverage"  "darts.RNNModel" "darts.BlockRNNModel" "darts.TiDEModel"  --save-path "weekly" --gpus 0  --num-workers 1 --timeout 60000


python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_weekly.json"  --data-name-list "m4_weekly_dataset_180.csv" "m4_weekly_dataset_72.csv" "m4_weekly_dataset_330.csv" "m4_weekly_dataset_332.csv" "m4_weekly_dataset_333.csv" "m4_weekly_dataset_334.csv" "m4_weekly_dataset_335.csv" "m4_weekly_dataset_337.csv" "m4_weekly_dataset_338.csv" "m4_weekly_dataset_339.csv"  --model-name  "darts.NBEATSModel" "darts.NHiTSModel" "darts.StatsForecastAutoETS" "darts.StatsForecastAutoCES" "darts.StatsForecastAutoTheta" "darts.AutoARIMA"  "darts.LinearRegressionModel"  "darts.RandomForest" "darts.KalmanForecaster" "darts.XGBModel"  "darts.NaiveMean"  "darts.NaiveSeasonal" "darts.NaiveDrift" "darts.NaiveMovingAverage"  "darts.RNNModel" "darts.BlockRNNModel" "darts.TiDEModel"  --save-path "weekly"  --eval-backend "ray" --gpus 0 1 2 3 4 5 6 7  --num-workers 8 --timeout 60000


python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json"  --data-name-list "S4-ADL2.test.csv@79.csv" --model-name "time_series_library.PatchTST"   --model-hyper-params '{"batch_size":128, "seq_len":100,"d_model":8, "d_ff":8, "e_layers":3, "num_epochs":3, "pred_len":0}'  --adapter "transformer_adapter" --report-method csv --gpus 1 --num-workers 1 --timeout 60000  --save-path "for_validation7"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-name-list "S4-ADL2.test.csv@79.csv" --model-name "self_implementation.DCdetector" --model-hyper-params '{"batch_size":16, "win_size":35,"anormly_ratio":1, "patch_size":[5,7]}' --report-method csv --gpus 1 --num-workers 1 --save-path "for_validation7"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_weekly.json"  --data-name-list "m4_weekly_dataset_180.csv" "m4_weekly_dataset_72.csv"   --model-name "time_series_library.Triformer"    --model-hyper-params '{"d_model":32,"d_ff":64, "seq_len":96,"pred_len":96, "batch_size":16, "lr":0.001, "num_epochs":5}'  --adapter "transformer_adapter"  --gpus 4  --num-workers 1 --timeout 60000  --save-path "for_validation7"
