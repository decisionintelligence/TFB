# <ins>T</ins>ime Series <ins>F</ins>orecasting <ins>B</ins>enchmark (TFB)

**TFB is an open-source library designed for time series researchers.**

**We provide a clean codebase for end-to-end evaluation of time series forecasting models, comparing their performance with baseline algorithms under various evaluation strategies and metrics.**

**We are further optimizing our code and welcome any suggestions for modifications.**


## Quickstart

### Installation

Given a python environment (**note**: this project is fully tested under python 3.8), install the dependencies with the following command:

```shell
pip install -r requirements.txt
```

### Data preparation

Prepare Data. You can obtained the well pre-processed datasets from [Google Drive](https://drive.google.com/drive/folders/1NFPGjCIX-aV1D0qeME4yQNGuv3fUVz7Y?usp=sharing).Then place the downloaded data under the folder `./dataset`. 

### Train and evaluate model.

We provide the experiment scripts for all benchmarks under the folder ./scripts/multivariate_forecast and ./scripts/univariate_forecast. For example，you can reproduce a experiment result as the following:

```shell
sh ./scripts/multivariate_forecast/ILI_script/DLinear.sh
```

### Steps to develop your own method

1. **Define you model or adapter class**

  - The user-implemented model or adapter class should implement the following functions in order to adapt to this benchmark.
  - required_hyper_params function is optional，__repr__ functions is necessary.

  - **The function prototype is as follows：**

    - required_hyper_params  function:

      ```python
      """
      Return the hyperparameters required by the model
      This function is optional and static
      
      :return: A dictionary that represents the hyperparameters required by the model
      :rtype: dict
      """
      # For example
      @staticmethod
      def required_hyper_params() -> dict:
          """
          An empty dictionary indicating that model does not require
          additional hyperparameters.
          """
          return {}
      ```
    
    - forecast_fit  function training model
    
      ```python
      """
      Train the model.
      
      :param train_valid_data: Time series data used for training.
      :param train_val_ratio: Represents the splitting ratio of the training
      set validation set. If it is equal to 1, it means that the validation
      set is not partitioned.
      """
      # For example
      def forecast_fit(self, train_valid_data: pd.DataFrame, train_val_ratio: float):
          pass
      ```
    
    - forecast function utilizing the model for inference
    
      ```python
      """
      Use models for forecasting
      
      :param pred_len: Predict length
      :type pred_len: int
      :param train: Training data used to fit the model
      :type train: pd.DataFrame
      
      :return: Forecasting results
      :rtype: np.ndarray
      """
      # For example
      def forecast(self, pred_len: int, train: pd.DataFrame) -> np.ndarray:
          pass
      ```
    
    - __repr __ string representation of function model name
    
      ```python
      """
      Returns a string representation of the model name
      
      :return: Returns a string representation of the model name
      :rtype: str
      """
      # For example
      def __repr__(self) -> str:
          return self.model_name
      ```
    

2. **Configure your Configuration File**

  - modify the corresponding config under the folder `./ts_benchmark/config/`.

  - modify the contents in  `./scripts/run_benchmark.py/`.

  - **We strongly recommend using the pre-defined configurations in `./ts_benchmark/config/`. Create your own  configuration file only when you have a clear understanding of the configuration items.**

3. **The benchmark can be run in the following format：**

```shell
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"pred_len":96}' --model-name "time_series_library.Triformer" --model-hyper-params '{"d_ff": 64, "d_model": 32, "pred_len": 96, "seq_len": 96}' --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTh1/Triformer"
```



## Example Usage

- **Define the model class or factory**
  - We demonstrated what functions need to be implemented for time series forecasting  using the VAR algorithm. You can find the complete code in `./ts_benchmark/baselines/self_implementation/VAR/VAR.py`.

```python
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
      
       :param train_valid_data: Time series data used for training.
       :param train_val_ratio: Represents the splitting ratio of the training set validation set. If it is equal to 1, it means that the validation set is not partitioned.
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
      Use models for forecasting
      
      :param pred_len: Predict length
      :type pred_len: int
      :param train: Training data used to fit the model
      :type train: pd.DataFrame
      
      :return: Forecasting results
      :rtype: np.ndarray
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

  ```shell
  python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"pred_len":96}' --model-name "self_implementation.VAR_model" --gpus 0  --num-workers 1  --timeout 60000  --save-path "ETTh1/VAR_model"
  ```



## Citation

If you find this repo useful, please cite our paper.

```
@article{qiu2024tfb,
    title={TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods},
    author={Xiangfei Qiu and Jilin Hu and Lekui Zhou and Xingjian Wu and Junyang Du and Buang Zhang and Chenjuan Guo and Aoying Zhou and Christian S. Jensen and Zhenli Sheng and Bin Yang},
    year={2024},
    eprint={2403.20150},
    archivePrefix={arXiv}
}
```


## Acknowledgement

The development of this library has been supported by **Huawei Cloud**, and we would like to acknowledge their contribution and assistance.


## Contact

If you have any questions or suggestions, feel free to contact:

- Xiangfei Qiu (xfqiu@stu.ecnu.edu.cn)
- Lekui Zhou (zhoulekui@huawei.com)
- Xingjian Wu (xjwu@stu.ecnu.edu.cn)
- Buang Zhang (buazhang@stu.ecnu.edu.cn)
- Junyang Du (jydu818@issbd2014.com)


Or describe it in Issues.
