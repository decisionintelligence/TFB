# Steps to predict only a subset of input variables

This tutorial will guide you on how to predict only a subset of input variables. You can follow the steps below:
### 1. Function Overview

This feature supports multivariate input and allows users to flexibly specify the target variables to be predicted through the `target_channel` parameter. Columns not designated as target variables will automatically act as covariates (exogenous variables) and participate in the model's training and prediction processes. **However, when calculating the training loss during training and evaluation metrics, only the target variables specified by the user are used for calculation.**

---

### 2. Detailed Explanation of the `target_channel` Parameter

The `target_channel` is the core parameter of this function, used to specify the target variables that need to be predicted. Columns not listed in `target_channel` will serve as covariates to assist in prediction. Below are the detailed explanations:

- **Type**: `Optional[List]`
- **Function**: Define the column indices of the target variables. The system divides the data into:
  - **Target Variables**: Columns specified by `target_channel`, which are the objects for model prediction.
  - **Covariates**: Other columns in the time series, which act as exogenous variables to support prediction.

- **Supported Formats**:
  - **`[]` (empty list)**: Indicates that no target variables are specified, and all columns are treated as target variables with no covariates (the model predicts all input variables).
  - **Single integer**:
    - `[0]`: Selects the first column as the target variable.
    - `[-1]`: Selects the last column as the target variable.
  - **List of integers**:
    - `[0, 1]`: Selects the first and second columns as target variables.
    - `[-2, -1]`: Selects the last two columns as target variables.
  - **List of tuples**:
    - `[[0, 3]]`: Selects the first to third columns (**excluding the fourth column**) as target variables.
  - **`None`**: All columns are treated as target variables with no covariates (the model predicts all input variables).

- **Notes**:
  - The indexing method is similar to Python slicing: positive indices start from 0, and negative indices start from -1 **(for example, `0` represents the first column, `-1` represents the last column, and `-2` represents the penultimate column).**
  - If `target_channel` is not specified or is set to `None`, the system defaults to treating all columns as target variables.

### 3. Usage Examples

Assume your dataset is `ETTh1.csv`, which contains 7 columns (column indices range from 0 to 6), and you are using the DUET model. Below are several usage examples of `target_channel`.

- **Configuration Methods**:

  - **Command Line**: Specify `target_channel` through `--strategy-args`.
  - **Configuration File**: Define it in the `"strategy_args"` section of a json file (e.g., `rolling_forecast_config.json`).
  - **Priority**: Command line arguments will override the settings in the configuration file.

- **Basic Command Line Example**:

  ```bash
  python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"horizon": 96, "target_channel":[-1]}' --model-name "duet.DUET" --model-hyper-params '{"CI": 1, "batch_size": 32, "d_ff": 512, "d_model": 512, "dropout": 0.5, "e_layers": 1, "factor": 3, "fc_dropout": 0.1, "horizon": 96, "k": 1, "loss": "MAE", "lr": 0.0005, "lradj": "type1", "n_heads": 1, "norm": true, "num_epochs": 100, "num_experts": 2, "patch_len": 48, "patience": 5, "seq_len": 512}' --deterministic "full" --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTh1/DUET"
  ```

- **Basic Configuration File Example** (`rolling_forecast_config.json`):

  ```json
  {
    "strategy_args": {
      "horizon": 96,
      "target_channel": [-1]
    }
  }
  ```

#### Example 1: Predict the 1st and 2nd Columns

- **Configuration**:

  - **Command Line**:

    ```bash
    --strategy-args "{"horizon":96,"target_channel":[0,1]}"
    ```

  - **Configuration File**:

  ```json
  {
    "strategy_args": {
      "horizon": 96,
      "target_channel": [0,1]
    }
  }
  ```

- **Result**:

  - **Target Variables**: The 1st and 2nd columns of the ETTh1 dataset.
  - **Covariates**: The 3rd, 4th, 5th, 6th, and 7th columns of the ETTh1 dataset.

#### Example 2: Predict the 3rd Column

- **Configuration**:

  - **Command Line**:

  ```bash
  --strategy-args "{"horizon":96,"target_channel":[2]}"
  ```

  - **Configuration File**:

  ```json
  {
    "strategy_args": {
      "horizon": 96,
      "target_channel": [2]
    }
  }
  ```

- **Result**:

  - **Target Variable**: The 3rd column of the ETTh1 dataset.
  - **Covariates**: The 1st, 2nd, 4th, 5th, 6th, and 7th columns of the ETTh1 dataset.

#### Example 3: Predict the 2nd to 4th Columns

- **Configuration**:

  - **Command Line**:

  ```bash
  --strategy-args "{"horizon":96,"target_channel":[[1,4]]}"
  ```

  - **Configuration File**:

  ```json
  {	
    "strategy_args": {
      "horizon": 96,
      "target_channel": [[1,4]]
    }
  }
  ```

- **Result**:

  - **Target Variables**: The 2nd, 3rd, and 4th columns of the ETTh1 dataset.

  - **Covariates**: The 1st, 5th, 6th, and 7th columns of the ETTh1 dataset.

#### Example 4: Predict the Last Two Columns (Using Negative Indexing)

- **Configuration**:

  - **Command Line**:

    ```bash
    --strategy-args "{"horizon":96,"target_channel":[-2,-1]}"
    ```

  - **Configuration File**:

  ```json
  {
    "strategy_args": {
      "horizon": 96,
      "target_channel": [-2, -1]
    }
  }
  ```

- **Result**:

  - **Target Variables**: The 6th and 7th columns (last two columns) of the ETTh1 dataset.
  - **Covariates**: The 1st, 2nd, 3rd, 4th, and 5th columns of the ETTh1 dataset.

#### Example 5: Predict All Columns

- **Configuration**:

  - **Command Line** (Omit `target_channel` or set it to `null`):

    ```bash
    --strategy-args "{"horizon":96}"
    ```

  - **Configuration File**:

  ```json
  {
    "strategy_args": {
      "horizon": 96,
      "target_channel": null
    }
  }
  ```

- **Result**:

  - **Target Variables**: All columns (the 1st to 7th columns of the ETTh1 dataset).
  - **Covariates**: None.

---

### 4. Notes

- **Index Range**: Ensure that the indices specified in `target_channel` are within the column range of the dataset. For example, for a dataset with 7 columns, the valid range for negative indices is -7 to -1.
- **Negative Indices**: `-1` corresponds to the 7th column (last column), `-7` corresponds to the 1st column (first column).
- **Tuple Format**: When using `[[start, end]]`, the `end` index is not included **(for example, `[1, 4]` includes the 2nd, 3rd, and 4th columns, but not the 5th column).**
- **Default Behavior**: If `target_channel` is `None` or not specified, the system will predict all columns, and there will be no covariates.