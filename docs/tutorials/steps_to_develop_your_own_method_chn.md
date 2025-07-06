# 开发自己的方法的步骤



## 一. 深度学习方法的实现-------以DUET算法为例

### 1. 创建模型目录结构

- 在`./ts_benchmark/baselines/`下新建您的模型目录（如`your_model`），可以包含以下子目录：

  - `layers/`：基础网络层实现

  - `models/`：核心模型架构

  - `utils/`：辅助工具函数

- 参考实现：[DUET目录结构示例](https://github.com/decisionintelligence/TFB/tree/master/ts_benchmark/baselines/duet)

- 对应TFB代码里面的: 

  - [./ts_benchmark/baselines/duet/layers](https://github.com/decisionintelligence/TFB/tree/master/ts_benchmark/baselines/duet/layers)

  - [./ts_benchmark/baselines/duet/models](https://github.com/decisionintelligence/TFB/tree/master/ts_benchmark/baselines/duet/models)

  - [./ts_benchmark/baselines/duet/utils](https://github.com/decisionintelligence/TFB/tree/master/ts_benchmark/baselines/duet/utils) 

- 在以下流程中`your_model`都指代duet

  

### 2. 创建模型适配文件

- 在模型目录下创建适配文件（如`your_model.py`），该文件负责将您的模型接入TFB框架。参考示例：[duet.py实现](https://github.com/decisionintelligence/TFB/blob/master/ts_benchmark/baselines/duet/duet.py)

- 对应在./ts_benchmark/baselines/duet目录下创建一个[duet.py](https://github.com/decisionintelligence/TFB/blob/master/ts_benchmark/baselines/duet/duet.py)。duet.py即是训练，验证，测试DUET这个算法的适配类。



### 3. 实现模型适配类

- 用户实现的**模型适配类**应继承 **[DeepForecastingModelBase](https://github.com/decisionintelligence/TFB/blob/master/ts_benchmark/baselines/deep_forecasting_model_base.py)** 类，此处 我们以 [duet.py](https://github.com/decisionintelligence/TFB/blob/master/ts_benchmark/baselines/duet/duet.py) 为例

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
      DUET 模型适配器类
  
      属性说明：
          model_name (str): 模型标识名称，用于识别不同模型
          _init_model: 初始化 DUETModel 模型实例的方法
          _process: 执行模型前向计算并返回输出结果
      """
      def __init__(self, **kwargs):
          super(DUET, self).__init__(MODEL_HYPER_PARAMS, **kwargs)
  ```
  
  
  
- 实施**model_name**方法来命名模型

  ```python
  @property
  def model_name(self):
      return "DUET"
  ```

  

- 实现**_init_model**方法来初始化自定义的模型

  ```python
  def _init_model(self):
      return DUETModel(self.config)
  ```

  

- 实施**_process**方法以处理训练、验证、测试步骤

  ```python
  def _process(self, input, target, input_mark, target_mark):
      """
      本方法作为模板方法，定义了数据处理和建模的标准流程，以及计算附加损失的规范。应根据自身需求实现具体的处理和计算逻辑。
  
      参数:
      - input: 输入数据，具体形式和含义取决于子类实现
      - target: 目标数据，与输入数据配合用于处理和损失计算  
      - input_mark: 输入数据的标记/元数据，辅助数据处理或模型训练
      - target_mark: 目标数据的标记/元数据，同样辅助数据处理或模型训练
  
      返回:
      - dict: 包含以下至少一个键的字典:
          - 'output' (必需): 模型输出张量
          - 'additional_loss' (可选): 存在的附加损失值
  
      异常:
      - NotImplementedError: 如果子类未实现本方法，调用时将抛出该异常
      """
      output, loss_importance = self.model(input)
      out_loss = {"output": output}
      if self.model.training:
          out_loss["additional_loss"] = loss_importance
      return out_loss
  ```

  

- 除了这些方法，您还可以实现其他六种方法：**_adjust_lr**、**save_checkpoint**、**_init_criterion_and_optimizer**、**_post_process** 和 **_init_early_stopping**。有关该方法的详细信息，请参阅 **[DeepForecastingModelBase](https://github.com/decisionintelligence/TFB/blob/master/ts_benchmark/baselines/deep_forecasting_model_base.py)**

  

**现在，让我们把它们放在一起：**

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
    DUET 模型适配器类

    属性说明：
        model_name (str): 模型标识名称，用于识别不同模型
        _init_model: 初始化 DUETModel 模型实例的方法
        _process: 执行模型前向计算并返回输出结果
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



### 4. 定义适配类接口

- 在`__init__.py`中声明公开接口：
- 对应 [./ts_benchmark/baselines/duet/__init__.py](https://github.com/decisionintelligence/TFB/blob/master/ts_benchmark/baselines/duet/__init__.py).

```python
__all__ = [
    "DUET"
]

from ts_benchmark.baselines.duet.duet import DUET
```



### 5. 选择您的配置文件

- 请根据自己的需求从 **./config** 目录下选择一个配置文件，例如选择 **[./config/rolling_forecast_config.json](https://github.com/decisionintelligence/TFB/blob/master/config/rolling_forecast_config.json)**。

- TODO： 将来会有一个专门的教程来介绍如何编写自己的配置文件。

  

### 6. 运行

```bash
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args '{"horizon": 24}' --model-name "your_model.DUET" --model-hyper-params '{"CI": 1, "batch_size": 8, "d_ff": 512, "d_model": 256, "dropout": 0.1, "e_layers": 2, "factor": 3, "fc_dropout": 0, "horizon": 24, "k": 2, "loss": "MAE", "lr": 0.001, "lradj": "type1", "n_heads": 1, "norm": true, "num_epochs": 100, "num_experts": 2, "patch_len": 48, "patience": 5, "seq_len": 104}' --deterministic "full" --gpus 0 --num-workers 1 --timeout 60000 --save-path "ILI/DUET"
```

关键参数说明：

- `--model-name`：格式为"模块名.适配类名", 对应教程里面的`duet.DUET`

  

  

## 二、旧的实现（基于深度学习的新实现请参考 I.）

这是使用 **ModelBase** 开发您自己的方法的教程。您可以参考以下步骤：

### 1. 在 ./ts_benchmark/baselines/ 目录下创建一个 “your_model.py” （或您喜欢的任何名称，但请记住相应地更新后续部分中的命令）。



### 2. 定义模型

- 用户实现的模型类应继承 **ModelBase** 类

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

  

- 实施 **forecast_fit** 方法来训练模型

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

  

- 实施**预测**方法以使用模型进行推理

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

  

- 由于 VAR 不支持 **batch_forecast**，因此本教程中未实现此方法。

- 实现**required_hyper_params**方法。仅当您的模型需要此机制时，才需要覆盖此方法。此处将其作为说明性示例提供。

  > **关于 required_hyper_params**
  >
  > 这是一种专门设计的机制，使模型能够将某些超参数的设置交给基准测试（我们不会强制模型遵守这些参数值）。该方法应返回一个键值字典，其中 key 是模型的超参数名称，value 是在 **recommend_model_hyper_params** 中全局定义的参数名称。
  >
  > 例如，如果模型无法自动确定最佳输入窗口大小（相应的超参数**input_window_size**），则可以将决策留给基准测试，以便基准测试可以使用全局推荐的设置（相应的超参数**input_chunk_length**）来生成不同模型之间的公平比较; 在此示例中，要正确启用此机制，模型需要在 dictionary **{“input_window_size”： “input_chunk_length”}** 中提供**required_hyper_params**字段。

  ```python
  def required_hyper_params() -> dict:
      """
      Return the hyperparameters required by VAR.
  
      :return: An empty dictionary indicating that VAR does not require additional hyperparameters.
      """
      return {}
  ```

  

- 实现**model_name**返回模型名称的字符串表示形式的方法

  ```python
  def model_name(self):
      """
      Returns the name of the model.
      """
      return "VAR"
  ```

  

**现在，让我们把它们放在一起：**

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



### 3. 选择您的配置文件

- 请根据自己的需求从 **./config** 目录下选择一个配置文件，例如选择 **./config/rolling_forecast_config.json**。

- TODO： 将来会有一个专门的教程来介绍如何编写自己的配置文件。

  

### 4. 运行

确保将 --model-name 参数的值设置为 **“your_model.VAR”。**

**“your_model”** 是您创建的 Python 模块的名称。管道将相对于 **./ts_benchmark/baselines** 搜索此模块。

```bash
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args '{"horizon":24}' --model-name "your_model.VAR" --num-workers 1  --timeout 60000  --save-path "saved_path"
```