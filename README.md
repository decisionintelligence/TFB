# Time seires Forecasting Benchmark（TFB）

**OTB is an open-source library designed for time series researchers.**

**We provide a clean codebase for end-to-end evaluation of time series models, comparing their performance with baseline algorithms under various evaluation strategies and metrics.**

## Quickstart

### Installation

Given a python environment (**note**: this project is fully tested under python 3.8), install the dependencies with the following command:

```
pip install -r requirements.txt
```

### Data preparation

Prepare Data. You can obtained the well pre-processed datasets from [Google Drive](https://drive.google.com/file/d/1xph1pkaZPYxAGNV_ljTrHdbRio4wiMAJ/view?usp=drive_link).Then place the downloaded data under the folder `./dataset`. 

### Example Usage

#### Forecasting Example:

- **Define the model class or factory**
  - We demonstrated what functions need to be implemented for time series forecasting based on **fixed_forecast strategy** using the LSTM algorithm. You can find the complete code in ` ./ts_benchmark/baseline/lstm.py`.
  - The algorithm expects input data in the form of a `pd.DataFrame`, where the time column serves as the index.

```
class TimeSeriesLSTM:

    def forecast_fit(self, train_data: pd.DataFrame):
        """
        training model

        :param train_data: time series data for training
        :type train_data: pd.DataFrame
        """
		pass

    def forecast(self, pred_len: int, testdata: pd.DataFrame) -> np.ndarray:
        """
        Making Predictions

        :param pred_len: Predicted length
        :type pred_len: int
        :param testdata: time series data for prediction
        :type testdata: pd.DataFrame
        
        :return: array of predicted results
        :rtype: np.ndarray
        """
        
        return output

    def __repr__(self) -> str:
        """
        returns a string representation of the model name
        """
        
        return "LSTM"

```

- **Run benchmark with fixed_forecaststrategy**

  ```
  python ./scripts/run_benchmark.py --config-path "fixed_forecast_config.json" --data-set-name "small_forecast" --model-name "lstm.TimeSeriesLSTM"
  ```

## User guide

### Data Format

The algorithm expects input data in the form of a `pd.DataFrame`, where the time column serves as the index. If the time values are integers of type `int`, they will be retained in this format. However, if the time values are in the standard timestamp format, they will be converted to a `pd.DatetimeIndex` type.

#### Example

- **The time values are integers of type `int`，they will be retained in this format.**

```
           col_1  col_2  col_3  col_4  ...  col_53  
date                                   ...                               
1       2.146646    0.0    0.0    0.0  ...     0.0    
2       2.146646    0.0    0.0    0.0  ...     0.0    
3       2.146646    0.0    0.0    0.0  ...     0.0     
4       2.151326    0.0    0.0    0.0  ...     0.0     
5       2.163807    0.0    0.0    0.0  ...     0.0    
...          ...    ...    ...    ...  ...     ...     
132042  0.499149    0.0    0.0    0.0  ...     0.0  
132043  0.501221    0.0    0.0    0.0  ...     0.0     
132044  0.501221    0.0    0.0    0.0  ...     0.0     
132045  0.501221    0.0    0.0    0.0  ...     0.0    
132046 -0.954212    0.0    0.0    0.0  ...     0.0     
```

- **The time values are in the standard timestamp format, they will be converted to a `pd.DatetimeIndex` type.**

```
                           col_1
date                            
2012-09-28 12:00:00  2074.503844
2012-09-29 12:00:00  3024.346943
2012-09-30 12:00:00  3088.428014
2012-10-01 12:00:00  3103.715163
2012-10-02 12:00:00  3123.547161
...                          ...
2016-05-03 12:00:00  9033.287169
2016-05-04 12:00:00  9055.950486
2016-05-05 12:00:00  9202.848984
2016-05-06 12:00:00  9180.724092
2016-05-07 12:00:00  9132.311537
```

### Folder Description

```
- baselines：Store the baseline model. Including third-party library models and models replicated in this warehouse

- common：Store some constants, such as configuration file path: PROFIG_ PATH

- config：Store configuration files under different evaluation strategies

- data_loader：Storing files for data crawling and data loading

- evaluation：Store evaluation strategy classes, implementation of evaluation indicators, and files for running evaluation models

- models：Store and return the model factory based on the model path entered by the user

- report：File for evaluating algorithm and baseline algorithm performance comparison in storage presentation

- utils：Store some tool files

- pipeline：Store files connected to the entire benchmark pipeline
```



### Steps to Evaluate Your Model

- **Define you model class or factory**
  - For different strategies, the user implemented model should implement the following functions in order to adapt to this benchmark.
  - For all strategies，required_hyper_params function is optional，__repr__ functions is necessary.
  - The matching relationship between other functions and policies is shown in the table below:
  
  |  strategy_name   | Strategic implications                                       | forecast_fit | detect_fit | forecast | detect_label | detect_score |
  | :--------------: | :----------------------------------------------------------- | :----------: | :--------: | :------: | :----------: | :----------: |
  |  fixed_forecast  | Fixed_forecast, with a total of n time points. If the defined prediction step size is f time points, then (n-f) time points are used as training data to predict future f time points. |      √       |            |    √     |              |              |
  | rolling_forecast | Rolling_forecast mirrors the cross-validation approach commonly utilized in machine learning. Here, the term 'origin' pertains to the training set within the time series, which is gradually expanded. In simpler terms, this technique enables the generation of multiple forecasts, each produced using an increasingly larger training set extracted from a single time series. |      √       |            |    √     |              |              |
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
    
    - __repr __ String representation of function model name
    
      ```
      """
      Returns a string representation of the model name
      
      :return: Returns a string representation of the model name
      :rtype: str
      """
      ```
    
    
  
- **Configure your Configuration File**

  - modify the corresponding config under the folder `./ts_benchmark/config/`.

  - modify the contents in run_benchmark_demo.py.
  
  - **We strongly recommend using the pre-defined configurations in `./ts_benchmark_config/`. Create your own  configuration file only when you have a clear understanding of the configuration items.**

- **The benchmark can be run in the following format：**

```
python ./scripts/run_benchmark.py --config-path "fixed_forecast_config.json" --data-set-name "small_forecast" --adapter None "statistics_darts_model_adapter" --model-name "darts_models.darts_arima" "darts.models.forecasting.arima.ARIMA" --model-hyper-params "{\"p\":7}" "{}" 
```



