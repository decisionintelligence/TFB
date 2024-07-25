## Steps to evaluating your own time series

1. Process your dataset into three columns in [TFB format](#TFB-data-format) using code like [here](#Example-code).  
2. Put the processed dataset under: **./dataset/forecasting** folder.
3. Run the following shell commands to evaluate on your own datasets：

- Evaluate on several user-defined forecasting datasets,  **please set the "--data-name-list" parameter to "your own dataset name"；**If you want to evaluate more than one series, separate the dataset names with commas.

```shell
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "your series name1" "your series name2" "your series name3" ...
```

A detailed example is provided below:

```shell
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "your series name1" "your series name2" "your series name3" --strategy-args '{"horizon":24}' --model-name "time_series_library.DLinear" --model-hyper-params '{"batch_size": 16, "d_ff": 512, "d_model": 256, "lr": 0.01, "horizon": 24, "seq_len": 104}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "ILI/DLinear"
```

- Evaluate on all user-defined forecasting datasets,  **please set the "--data-set-namet" parameter to "user_forecast"**；At this point, all the datasets you uploaded will be evaluated.

```shell
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-set-name "user_forecast"
```

A detailed example is provided below:

```shell
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-set-name "user_forecast" --strategy-args '{"horizon":24}' --model-name "time_series_library.DLinear" --model-hyper-params '{"batch_size": 16, "d_ff": 512, "d_model": 256, "lr": 0.01, "horizon": 24, "seq_len": 104}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "ILI/DLinear"
```



## TFB data format

TFB stores time series in a format of three column long tables,  which we will introduce below：

### Format Introduction

**First column: date** (the exact column name is required, the same applies below.)

- The columns stores the time information in the time series, which can be in either of the following formats:
  - Timestamps in string, datetime, or other types that are compatible with pd.to_datetime;
  - Integers starting from 1, e.g. 1, 2, 3, 4, 5, ...
  

**Second column: data**

- This column stores the series values corresponding to the timestamps.

**Third column: cols**

- This column stores the column name (variable name).

  


### Multivariate time series example:

**A common time series in wide table format:**

| date | channel1 | channel2 | channel3 |
| ---- | -------- | -------- | -------- |
| 1    | 0.1      | 1        | 10       |
| 2    | 0.2      | 2        | 20       |
| 3    | 0.3      | 3        | 30       |

**Convert to our format:**

| date | data | cols     |
| ---- | ---- | -------- |
| 1    | 0.1  | channel1 |
| 2    | 0.2  | channel1 |
| 3    | 0.3  | channel1 |
| 1    | 1    | channel2 |
| 2    | 2    | channel2 |
| 3    | 3    | channel2 |
| 1    | 10   | channel3 |
| 2    | 20   | channel3 |
| 3    | 30   | channel3 |



###  Univariate time series example:

**A common time series in wide table format:**

| date                | channel1 |
| ------------------- | -------- |
| 2012-09-26 12:00:00 | 0.1      |
| 2012-09-27 12:00:00 | 0.2      |
| 2012-09-28 12:00:00 | 0.3      |
| 2012-09-29 12:00:00 | 0.4      |
| 2012-09-30 12:00:00 | 0.5      |

**Convert to our format:**

| date                | data | cols     |
| ------------------- | ---- | -------- |
| 2012-09-26 12:00:00 | 0.1  | channel1 |
| 2012-09-27 12:00:00 | 0.2  | channel1 |
| 2012-09-28 12:00:00 | 0.3  | channel1 |
| 2012-09-29 12:00:00 | 0.4  | channel1 |
| 2012-09-30 12:00:00 | 0.5  | channel1 |





## Example code

```python
import pandas as pd


def convert_to_tfb_series(data):
    data = data.set_index("date")
    melted_df = data.melt(value_name="data", var_name="cols", ignore_index=False)
    return melted_df.reset_index()[['date', 'data', 'cols']]

data = pd.read_csv(r"./ori-Exchange.csv")
melted_df = convert_to_tfb_series(data)
melted_df.to_csv(r"./converted-Exchange.csv", index=False)

```

