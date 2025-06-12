## 1. 安装环境

- 创建conda环境, 相关环境为：python 3.8，R 4.3.1

```shell
conda create -n char_extractor python=3.8 r-base=4.3.1
conda activate char_extractor
```

- 设置 R 语言的环境变量：

```shell
export R_HOME=$CONDA_PREFIX/lib/R
export PATH=$PATH:$R_HOME/bin
```

- 安装必要的 R 包

```shell
conda install -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge r-tidyverse r-Rcatch22 r-forecast r-tsfeatures -y
```

- 安装必要的Python包

```shell
pip install rpy2==3.5.16 pandas==1.5.3 scipy==1.10.1 numpy==1.24.4 statsmodels==0.14.1 scikit_learn==1.3.2
```



## 2. 输入

- 数据格式

输入的数据格式需要是TFB，支持的3列长表的格式，具体请见[这里](https://github.com/decisionintelligence/TFB/blob/master/docs/tutorials/steps_to_evaluate_your_own_time_series.md).

- 输入格式

  - 支持输入指定一个时间序列文件的路径，会计算该时间序列的特征，例如`./DemoDatasets/Exchange.csv`.

  - 支持输入指定文件夹路径，会遍历计算文件夹内所有时间序列文件的时间序列特征, 例如`./DemoDatasets`.



## 3. 输出

- 输出的格式

  - 对于单变量时间序列文件而言，例如输入的文件名为: `m4_hourly_dataset_69.csv`，会在用户指定的输出文件夹下（默认为characteristics文件夹）输出2个对应文件

    - `All_characteristics_m4_hourly_dataset_69.csv`: 包含本代码内计算的所有时间序列特征

    - `TFB_characteristics_m4_hourly_dataset_69.csv`: 包含TFB论文内使用的时间序列特征

  - 对于多变量时间序列文件而言，例如输入的文件名为: `Exchange.csv`，会在用户指定的输出文件夹下（默认为characteristics文件夹）输出4个对应文件

    - `All_characteristics_Exchange.csv`: 包含本代码内计算的所有时间序列特征, 每一行代表多变量中的一个变量对应的时间序列特征

    - `TFB_characteristics_Exchange.csv`: 包含TFB论文内使用的时间序列特征, 每一行代表多变量中的一个变量对应的时间序列特征

    - `mean_All_characteristics_Exchange.csv`: 包含本代码内计算的所有时间序列特征, 将多变量中所有变量对应的每一个时间序列特征求均值

    - `mean_TFB_characteristics_Exchange.csv`: 包含TFB论文内使用的时间序列特征, 将多变量中所有变量对应的每一个时间序列特征求均值



## 4. TFB特征介绍

详情请见[这里](https://github.com/decisionintelligence/TFB/blob/master/docs/tutorials/introduction_and_pseudocode_for_time_series_characteristics.md).



## 5. 代码文件

点击[这里](https://github.com/decisionintelligence/TFB/tree/master/characteristics_extractor/Characteristics_Extractor.py)看代码.
