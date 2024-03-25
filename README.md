# Time seires Forecasting Benchmark (TFB)

**TFB is an open-source library designed for time series researchers.**

**We provide a clean codebase for end-to-end evaluation of time series forecasting models, comparing their performance with baseline algorithms under various evaluation strategies and metrics.**



## Quickstart

### Installation

Given a python environment (**note**: this project is fully tested under python 3.8), install the dependencies with the following command:

```
pip install -r requirements.txt
```

### Data preparation

Prepare Data. You can obtained the well pre-processed datasets from [Google Drive](https://drive.google.com/drive/folders/1NFPGjCIX-aV1D0qeME4yQNGuv3fUVz7Y?usp=sharing).Then place the downloaded data under the folder `./dataset`. 

### Train and evaluate model.

We provide the experiment scripts for all benchmarks under the folder ./scripts/multivariate_forecast and ./scripts/univariate_forecast. For example，you can reproduce a experiment result as the following:

```
sh ./scripts/multivariate_forecast/AQShunyi_script/Triformer.sh
```

### Steps to develop your own method

- **Define you model class or factory**
  
  - For different strategies, the user implemented model should implement the following functions in order to adapt to this benchmark.
  - For all strategies，required_hyper_params function is optional，__repr__ functions is necessary.
  - The matching relationship between other functions and policies is shown in the table below:
  
  |  strategy_name   | Strategic implications                                       | forecast_fit | forecast |
  | :--------------: | :----------------------------------------------------------- | :----------: | :------: |
  |  fixed_forecast  | Fixed_forecast, with a total of n time points. If the defined prediction step size is f time points, then (n-f) time points are used as training data to predict future f time points. |      √       |    √     |
  | rolling_forecast | Rolling_forecast mirrors the cross-validation approach commonly utilized in machine learning. Here, the term 'origin' pertains to the training set within the time series, which is gradually expanded. In simpler terms, this technique enables the generation of multiple forecasts, each produced using an increasingly larger training set extracted from a single time series. |      √       |    √     |
  - **The function prototype is as follows：**
  
    - required_hyper_params  function:
  
      ```
      """
      Return the hyperparameters required by the model
      This function is optional and static
      
      :return: A dictionary that represents the hyperparameters required by the model
      :rtype: dict
      """
      ```
    
    - forecast_fit  function training model
    
      ```
      """
      Fitting models on time series data
      
      :param series: time series data
      :type series: pd.DataFrame
      """
      ```
    
    - forecast function predicts the model
    
      ```
      """
      Use models for prediction
      
      :param pred_len: Predict length
      :type pred_len: int
      :param train: Training data used to fit the model
      :type train: pd.DataFrame
      
      :return: Prediction results
      :rtype: np.ndarray
      """
      ```
    
    - __repr __ string representation of function model name
    
      ```
      """
      Returns a string representation of the model name
      
      :return: Returns a string representation of the model name
      :rtype: str
      """
      ```
    

- **Configure your Configuration File**

  - modify the corresponding config under the folder `./ts_benchmark/config/`.

  - modify the contents in  `./scripts/run_benchmark.py/`.

  - **We strongly recommend using the pre-defined configurations in `./ts_benchmark/config/`. Create your own  configuration file only when you have a clear understanding of the configuration items.**

- **The benchmark can be run in the following format：**

```
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"pred_len":96}' --model-name "time_series_library.Triformer" --model-hyper-params '{"d_ff": 64, "d_model": 32, "pred_len": 96, "seq_len": 96}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTh1/Triformer"
```



## Example Usage

- **Define the model class or factory**
  - We demonstrated what functions need to be implemented for time series forecasting  using the VAR algorithm. You can find the complete code in `./ts_benchmark/baselines/self_implementation/VAR/VAR.py`.

```
class VAR_model:
    """
    VAR class.

    This class encapsulates a process of using VAR models for time series forecasting.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.model_args = {}

    @staticmethod
    def required_hyper_params() -> dict:
        """
        Return the hyperparameters required by VAR.

        :return: An empty dictionary indicating that VAR does not require additional hyperparameters.
        """
        
        return {}

    def forecast_fit(self, train_data: pd.DataFrame, train_val_ratio: float):
        """
        Train the model.

        :param train_data: Time series data used for training.
        :param train_val_ratio: Represents the splitting ratio of the training set validation set. If it is equal to 1, it means 					that the validation set is not partitioned.
        :return: The fitted model object.
        """

        self.scaler.fit(train_data.values)
        train_data_value = pd.DataFrame(
            self.scaler.transform(train_data.values),
            columns=train_data.columns,
            index=train_data.index,
        )
        model = VAR(train_data_value)
        self.results = model.fit(13)

    def forecast(self, pred_len: int, testdata: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        :param pred_len: The predicted length.
        :param testdata: Time series data used for prediction.
        :return: An array of predicted results.
        """
        
        train = pd.DataFrame(
            self.scaler.transform(testdata.values),
            columns=testdata.columns,
            index=testdata.index,
        )
        z = self.results.forecast(train.values, steps=pred_len)

        predict = self.scaler.inverse_transform(z)
        return predict

    def __repr__(self) -> str:
        """
        Returns a string representation of the model name.
        """
        
        return self.model_name

```

- **Run benchmark using VAR**

  ```
  python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"pred_len":96}' --model-name "self_implementation.VAR_model" --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTh1/VAR_model"
  ```



## Citation

If you find this repo useful, please cite our paper.

```

```



## Contact

If you have any questions or suggestions, feel free to contact:

- Xiangfei Qiu ( xfqiu@stu.ecnu.edu.cn)
- Xingjian Wu (xjwu@stu.ecnu.edu.cn)
- Buang Zhang (buazhang@stu.ecnu.edu.cn)

Or describe it in Issues.
