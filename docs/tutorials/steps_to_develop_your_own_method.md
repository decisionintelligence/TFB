# Steps to develop your own method 

## Ⅰ. New implimentation for deep learning methods

### 1. Create a "your_model.py" (or whatever name you prefer, but remember to update the commands in the subsequent sections accordingly) under the ./ts_benchmark/baselines/your_model directory.

### 2. Define your model

- The user-implemented model class should inherit the class **DeepForecastingModelBase**, here
we use DUET as an example

  ```python
    from ts_benchmark.baselines.duet.models.duet_model import DUETModel
    from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase

    # model hyper params
    MODEL_HYPER_PARAMS = {
        "enc_in": 1,
        "dec_in": 1,
        "c_out": 1,
        "e_layers": 2,
        "d_layers": 1,
        "d_model": 512,
        "d_ff": 2048,
        "hidden_size": 256,
        "freq": "h",
        "factor": 1,
        "n_heads": 8,
        "seg_len": 6,
        "win_size": 2,
        "activation": "gelu",
        "output_attention": 0,
        "patch_len": 16,
        "stride": 8,
        "period_len": 4,
        "dropout": 0.2,
        "fc_dropout": 0.2,
        "moving_avg": 25,
        "batch_size": 256,
        "lradj": "type3",
        "lr": 0.02,
        "num_epochs": 100,
        "num_workers": 0,
        "loss": "huber",
        "patience": 10,
        "num_experts": 4,
        "noisy_gating": True,
        "k": 1,
        "CI": True,
        "parallel_strategy": "DP"
    }


    class DUET(DeepForecastingModelBase):
        """
        DUET adapter class.

        Attributes:
            model_name (str): Name of the model for identification purposes.
            _init_model: Initializes an instance of the DUETModel.
            _process: Executes the model's forward pass and returns the output.
        """

        def __init__(self, **kwargs):
            super(DUET, self).__init__(MODEL_HYPER_PARAMS, **kwargs)
  ```

- Implement  **model_name**  method to name your model

  ```python
    @property
    def model_name(self):
        return "DUET"
  ```

- Implement **_init_model** method to init the model that contains logic process

  ```python
    def _init_model(self):
        return DUETModel(self.config)
  ```

- Implement **_process** method to process the train or validate steps

    ```python
        def _process(self, input, target, input_mark, target_mark):
            output, loss_importance = self.model(input)
            out_loss = {"output": output}
            if self.model.training:
                out_loss["additional_loss"] = loss_importance
            return out_loss
    ```

- Besides these methods you can also implement six other methods: **_adjust_lr**, **save_checkpoint**, **_init_criterion_and_optimizer**, **_post_process** and  **_init_early_stopping**. For details of the method, please refer to **DeepForecastingModelBase**

**Now, let's put it all together:**

```python
from ts_benchmark.baselines.duet.models.duet_model import DUETModel
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase

# model hyper params
MODEL_HYPER_PARAMS = {
    "enc_in": 1,
    "dec_in": 1,
    "c_out": 1,
    "e_layers": 2,
    "d_layers": 1,
    "d_model": 512,
    "d_ff": 2048,
    "hidden_size": 256,
    "freq": "h",
    "factor": 1,
    "n_heads": 8,
    "seg_len": 6,
    "win_size": 2,
    "activation": "gelu",
    "output_attention": 0,
    "patch_len": 16,
    "stride": 8,
    "period_len": 4,
    "dropout": 0.2,
    "fc_dropout": 0.2,
    "moving_avg": 25,
    "batch_size": 256,
    "lradj": "type3",
    "lr": 0.02,
    "num_epochs": 100,
    "num_workers": 0,
    "loss": "huber",
    "patience": 10,
    "num_experts": 4,
    "noisy_gating": True,
    "k": 1,
    "CI": True,
    "parallel_strategy": "DP"
}


class DUET(DeepForecastingModelBase):
    """
    DUET adapter class.

    Attributes:
        model_name (str): Name of the model for identification purposes.
        _init_model: Initializes an instance of the DUETModel.
        _adjust_lr：Adjusts the learning rate of the optimizer based on the current epoch and configuration.
        _process: Executes the model's forward pass and returns the output.
    """

    def __init__(self, **kwargs):
        super(DUET, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "DUET"

    def _init_model(self):
        return DUETModel(self.config)

    def _process(self, input, target, input_mark, target_mark):
        output, loss_importance = self.model(input)
        out_loss = {"output": output}
        if self.model.training:
            out_loss["additional_loss"] = loss_importance
        return out_loss

```


### 3. Choose your configuration file

  - Please select a config file from the **./config** directory based on your needs, such as choosing the **./config/rolling_forecast_config.json**.
  
### 4. Run it

**"your_model"** is the name of the Python module you created. The pipeline will search for this module relative to **./ts_benchmark/baselines**.

```shell
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args '{"horizon": 24}' --model-name "your_model.DUET" --model-hyper-params '{"CI": 1, "batch_size": 8, "d_ff": 512, "d_model": 256, "dropout": 0.1, "e_layers": 2, "factor": 3, "fc_dropout": 0, "horizon": 24, "k": 2, "loss": "MAE", "lr": 0.001, "lradj": "type1", "n_heads": 1, "norm": true, "num_epochs": 100, "num_experts": 2, "patch_len": 48, "patience": 5, "seq_len": 104}' --deterministic "full" --gpus 0 --num-workers 1 --timeout 60000 --save-path "ILI/DUET"
```

## Ⅱ. Old implementation (for the new implementation based on deep learning, please refer to Ⅰ)

This is a tutorial for developing your own method using **ModelBase**.  You can refer to the following steps:


### 1. Create a "your_model.py" (or whatever name you prefer, but remember to update the commands in the subsequent sections accordingly) under the ./ts_benchmark/baselines/ directory.



### 2. Define your model

- The user-implemented model class should inherit the class **ModelBase**

  ```python
  import numpy as np
  import pandas as pd
  from sklearn.preprocessing import StandardScaler
  from statsmodels.tsa.api import VAR as VARModel
  
  from ts_benchmark.models.model_base import ModelBase
  
  
  class VAR(ModelBase):
      """
      VAR class.
  
      This class encapsulates a process of using VAR models for time series prediction.
      """
  
      def __init__(
          self, lags=13
      ):
          self.scaler = StandardScaler()
          self.lags = lags
          self.results = None
  ```

- Implement  **forecast_fit**  method to train your model

  ```python
  def forecast_fit(
      self, train_data: pd.DataFrame, *, train_ratio_in_tv: float = 1.0, **kwargs
  ) -> "ModelBase":
      """
      Train the model.
  
      :param train_data: Time series data used for training.
      :param train_ratio_in_tv: Represents the splitting ratio of the training set validation set.
          If it is equal to 1, it means that the validation set is not partitioned.
  		
      :return: The fitted model object.
      """
  
      self.scaler.fit(train_data.values)
      train_data_value = pd.DataFrame(
          self.scaler.transform(train_data.values),
          columns=train_data.columns,
          index=train_data.index,
      )
      model = VARModel(train_data_value)
      self.results = model.fit(self.lags)
  
      return self
  ```

- Implement **forecast** method to inference with your model

  ```python
  def forecast(self, horizon: int, series: pd.DataFrame, **kwargs) -> np.ndarray:
      """
      Make predictions.
  
      :param horizon: The predicted length.
      :param series: Time series data used for prediction.
      :return: An array of predicted results.
      """
      train = pd.DataFrame(
          self.scaler.transform(series.values),
          columns=series.columns,
          index=series.index,
      )
      z = self.results.forecast(train.values, steps=horizon)
  
      predict = self.scaler.inverse_transform(z)
  
      return predict
  ```

- Because VAR does not support **batch_forecast**, this method is not implemented in this tutorial.

- Implement **required_hyper_params** method.  This method only needs to be overwritten if your model requires this mechanism. It is provided here as an instructional example.

  > **About required_hyper_params**
  >
  > This is a specially designed mechanism to enable models to relinquish the settings of some hyperparameters to the benchmark (We do not enforce the model to adhere to these parameter values). The method should return a key-value dictionary where the key is the model's hyperparameter name and the value is the parameter name defined globally in **recommend_model_hyper_params**. 
  >
  > For example, if a model cannot automatically decide the best input window size (corresponding hyperparameter **input_window_size**), it can leave the decision to the benchmark, so that the benchmark can use a globally recommended setting (corresponding hyperparameter **input_chunk_length**) to produce a fair comparison between different models;
  >  In this example, to enable this mechanism properly, the model is required to provide a **required_hyper_params** field in dictionary **{"input_window_size": "input_chunk_length"}**.

  ```python
  def required_hyper_params() -> dict:
      """
      Return the hyperparameters required by VAR.
  
      :return: An empty dictionary indicating that VAR does not require additional hyperparameters.
      """
      return {}
  ```

- Implement **model_name** method which returns a string representation of the model name

  ```python
  def model_name(self):
      """
      Returns the name of the model.
      """
      return "VAR"
  ```



**Now, let's put it all together:**

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR as VARModel

from ts_benchmark.models.model_base import ModelBase


class VAR(ModelBase):
    """
    VAR class.

    This class encapsulates a process of using VAR models for time series prediction.
    """

    def __init__(self, lags=13):
        self.scaler = StandardScaler()
        self.lags = lags
        self.results = None

    @property
    def model_name(self):
        """
        Returns the name of the model.
        """
        return "VAR"

    @staticmethod
    def required_hyper_params() -> dict:
        """
        Return the hyperparameters required by VAR.

        :return: An empty dictionary indicating that VAR does not require additional hyperparameters.
        """
        return {}

    def forecast_fit(
        self, train_data: pd.DataFrame, *, train_ratio_in_tv: float = 1.0, **kwargs
    ) -> "ModelBase":
        """
        Train the model.

        :param train_data: Time series data used for training.
        :param train_ratio_in_tv: Represents the splitting ratio of the training set validation set. If it is equal to 1, it means that the validation set is not partitioned.
        :return: The fitted model object.
        """

        self.scaler.fit(train_data.values)
        train_data_value = pd.DataFrame(
            self.scaler.transform(train_data.values),
            columns=train_data.columns,
            index=train_data.index,
        )
        model = VARModel(train_data_value)
        self.results = model.fit(self.lags)

        return self

    def forecast(self, horizon: int, series: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Make predictions.

        :param horizon: The predicted length.
        :param series: Time series data used for prediction.
        :return: An array of predicted results.
        """
        train = pd.DataFrame(
            self.scaler.transform(series.values),
            columns=series.columns,
            index=series.index,
        )
        z = self.results.forecast(train.values, steps=horizon)

        predict = self.scaler.inverse_transform(z)

        return predict
```



### 3. Choose your configuration file

  - Please select a config file from the **./config** directory based on your needs, such as choosing the **./config/rolling_forecast_config.json**.

  - TODO:  There will be a dedicated tutorial on how to write your own config file in the future.

    

### 4. Run it

Make sure to set the value of the --model-name parameter to **"your_model.VAR\"**.

**"your_model"** is the name of the Python module you created. The pipeline will search for this module relative to **./ts_benchmark/baselines**.

```shell
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args '{"horizon":24}' --model-name "your_model.VAR" --num-workers 1  --timeout 60000  --save-path "saved_path"
```

