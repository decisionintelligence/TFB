## 1. Environment Setup

- Create a Conda environment with the following specifications: Python 3.8, R 4.3.1.

```shell
conda create -n char_extractor python=3.8 r-base=4.3.1
conda activate char_extractor
```

- Set environment variables for R:

```shell
export R_HOME=$CONDA_PREFIX/lib/R
export PATH=$PATH:$R_HOME/bin
```

- Install necessary R packages:

```shell
conda install -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge r-tidyverse r-Rcatch22 r-forecast r-tsfeatures -y
```

- Install required Python packages:

```shell
pip install rpy2==3.5.16 pandas==1.5.3 scipy==1.10.1 numpy==1.24.4 statsmodels==0.14.1 scikit_learn==1.3.2
```



## 2. Input

- Data Format

The input data format needs to be TFB, supporting a three-column long table format. For more details, please refer to [here](https://github.com/decisionintelligence/TFB/blob/master/docs/tutorials/steps_to_evaluate_your_own_time_series.md).

- Input Format

  - You can provide the path to a single time series file, and the tool will compute the characteristics of that time series. For example: `./DemoDatasets/Exchange.csv`.

  - Alternatively, you can specify a folder path, and the tool will recursively calculate the time series features for all files within the folder. For example: `./DemoDatasets`.



## 3. Output

- Output Format

  - For univariate time series files, for example, if the input file is named `m4_hourly_dataset_69.csv`, two corresponding files will be generated in the user-specified output directory (default is the "characteristics" folder):

    - `All_characteristics_m4_hourly_dataset_69.csv`: Contains all time series features calculated by this code.

    - `TFB_characteristics_m4_hourly_dataset_69.csv`: Contains the time series features used in the TFB paper.

  - For multivariate time series files, for example, if the input file is named `Exchange.csv`, four corresponding files will be generated in the user-specified output directory (default is the "characteristics" folder):

    - `All_characteristics_Exchange.csv`: Contains all time series features calculated by this code. Each row corresponds to the time series features of one variable in the multivariate series.

    - `TFB_characteristics_Exchange.csv`: Contains the time series features used in the TFB paper. Each row corresponds to the time series features of one variable in the multivariate series.

    - `mean_All_characteristics_Exchange.csv`: Contains all time series features calculated by this code, with each feature averaged across all variables in the multivariate series.

    - `mean_TFB_characteristics_Exchange.csv`: Contains the time series features used in the TFB paper, with each feature averaged across all variables in the multivariate series.



## 4. Introduction to TFB Features

For more details, please refer to [here](https://github.com/decisionintelligence/TFB/blob/master/docs/tutorials/introduction_and_pseudocode_for_time_series_characteristics.md).



## 5. Code File

Click [here](https://github.com/decisionintelligence/TFB/tree/master/characteristics_extractor/Characteristics_Extractor.py) to view the code.
