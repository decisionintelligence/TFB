## Steps to develop your own method 

This is a tutorial for developing your own method.  You can refer to the following steps:



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

