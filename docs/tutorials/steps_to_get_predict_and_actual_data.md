## Steps to get the model's predicted values and the target values

1. If you want to save the model's predicted values and the target values, you should set `--save-true-pred` to `True` when running the benchmark.
Such as:
```shell
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args '{"horizon":60}' --model-name "time_series_library.DLinear" --model-hyper-params '{"batch_size": 64, "d_ff": 512, "d_model": 256, "lr": 0.01, "horizon": 60, "seq_len": 104}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "your_result_path" --save-true-pred True
```
2. After the benchmark finishes running, the predicted values and target values are saved in the result file. For example, if you set `--save-path "your_result_path"`, the result file will be located at `/result/your_result_path`.
3. The result file is in `.tar.gz` format. You need to extract this file to access the results. 
4. The predicted values are stored in the `"inference_data"` column, while the target values are in the `"actual_data"` column. However, you may notice that they appear as garbled code because they are encoded in base64. You can use the function below to get the decoded predicted and target values.

```python
import base64
import pickle

import numpy as np
import pandas as pd


def decode_data(filepath: str) -> pd.DataFrame:
    """
    Load the result file, decode base64-encoded inference and actual data columns.

    :param filepath: The path to the result data.
    :return: The decoded data.
    """
    data = pd.read_csv(filepath)
    for index, row in data.iterrows():
        decoded_inference_data = base64.b64decode(row["inference_data"])
        decoded_actual_data = base64.b64decode(row["actual_data"])
        data.at[index, "inference_data"] = pickle.loads(decoded_inference_data)
        data.at[index, "actual_data"] = pickle.loads(decoded_actual_data)
    return data


'''
If you want to save the decoded data as a CSV file. Please follow the steps below.

your_result_path = r"your_result_path/your_result.csv"
decoded_result = decode_data(your_result_path)
pd.set_option('display.width', None)  # Avoid ellipses in the data.
np.set_printoptions(threshold=np.inf)  # Avoid ellipses in the data.
decoded_result.to_csv("decoded_result.csv", index=None)
'''
```
