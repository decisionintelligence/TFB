
<div align="center">
  <img src="docs/figures/TFB-LOGO.png" width="80%">
  <h2>TFB: 全面且公平的时间序列预测方法评测基准</h2>
</div>

<div align="center">

[**English**](./README.md) **|**
[**简体中文**](./README_CN.md)

</div>

---

<div align="center">

[![PVLDB](https://img.shields.io/badge/PVLDB'24-TFB-orange)](https://arxiv.org/pdf/2403.20150.pdf)  [![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)  [![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-blue)](https://pytorch.org/)  ![Stars](https://img.shields.io/github/stars/decisionintelligence/TFB)  

</div>

> [!IMPORTANT]
> 1、如果您觉得这个项目有帮助，请不要忘记给它一个 ⭐ 星标以表示支持。谢谢！
> 
> 2、我们重新测试了一些算法的结果，这些结果可能与TFB论文中的结果有所不同。您可以在[scripts](https://github.com/decisionintelligence/TFB/tree/master/scripts)文件夹中找到我们最终为每个数据集上的每个算法选择的超参数，以及它们对应的算法测试结果可以在[OpenTS](https://decisionintelligence.github.io/OpenTS/leaderboards/multivariate_time_series/)上找到！

🚩 **新闻** (2025.06) **我们还开源了时间序列异常检测基准(TAB)和时间序列基础模型基准(TSFM-Bench)。**

🚩 **新闻** (2025.04) **TFB已开源用于计算时间序列特征的[代码](https://github.com/decisionintelligence/TFB/blob/master/characteristics_extractor/Characteristics_Extractor.py)，如趋势、季节性、平稳性、移动、转换、相关性等。提供了[中文](https://github.com/decisionintelligence/TFB/blob/master/characteristics_extractor/Readme_chn.md)和[英文](https://github.com/decisionintelligence/TFB/blob/master/characteristics_extractor/Readme_en.md)文档。**

🚩 **新闻** (2025.04) **[DUET](https://arxiv.org/pdf/2412.10859)发布了统一超参数的长期预测任务结果，其中输入长度固定为96。点击[这里](https://github.com/decisionintelligence/DUET/blob/main/figures/DUET_unified_seq_len_96.pdf)查看结果，点击[这里](https://github.com/decisionintelligence/DUET/blob/main/scripts/multivariate_forecast/DUET_unified_seq_len_96.sh)查看重现结果的脚本。**

🚩 **新闻** (2025.04) **TFB新增了两个数据集：PEMS03和PEMS07，总计达到27个多变量数据集**。

🚩 **新闻** (2025.03) **TFB新增了一个实用功能：支持仅预测输入变量的子集。提供了[中文](https://github.com/decisionintelligence/TFB/blob/master/docs/tutorials/steps_to_predict_only_a_subset_of_input_variables_chn.pdf)和[英文](./docs/tutorials/steps_to_predict_only_a_subset_of_input_variables.md)文档。**

🚩 **新闻** (2025.03) **我们维护了一个[微信群](./docs/figures/QR.png)来促进关于TFB和[OpenTS](https://decisionintelligence.github.io/OpenTS/)的讨论**。

🚩 **新闻** (2024.09) **您可以在[这里](https://tfb-docs.readthedocs.io/en/latest/index.html)找到详细的API文档**。

🚩 **新闻** (2024.08) **介绍视频（中文）：[bilibili](https://www.bilibili.com/video/BV1fYH4eQEPv/?spm_id_from=333.337.search-card.all.click)。**

🚩 **新闻** (2024.08) **TFB在PVLDB 2024中获得🌟最佳论文提名🌟**。

🚩 **新闻** (2024.08) **我们为时间序列预测创建了一个排行榜，称为[OpenTS](https://decisionintelligence.github.io/OpenTS/)。**

🚩 **新闻** (2024.05) **一些介绍（中文）：[介绍1](https://mp.weixin.qq.com/s/5BscuAWIn-tzla2rzW1IsQ)，[介绍2](https://mp.weixin.qq.com/s/IPY2QwJ68YIrclMi2JtkMA)，[介绍3](https://mp.weixin.qq.com/s/D4SBwwVjHvuksaQ0boXjNw)，[介绍4](https://mp.weixin.qq.com/s/OfZJtd3H3-TCkvBGATt0mA)，[介绍5](https://mp.weixin.qq.com/s/pjTN15vHL5UxjL1mhJxguw)，[介绍6](https://mp.weixin.qq.com/s/ghJ3xN38pB-sDb0hiWjW7w)，和[介绍7](https://mp.weixin.qq.com/s/J8SRsN4W0FNMtlLpULhwKg)。**

**新增的基线方法。** ☑ 表示其代码已经包含在此仓库中，且其性能结果已包含在[OpenTS](https://decisionintelligence.github.io/OpenTS/)排行榜中。☒ 表示仅其代码已包含在此仓库中。
  - [ ] **TimeKAN** - TimeKAN: 基于KAN的频率分解学习架构用于长期时间序列预测 [[ICLR 2025]](https://arxiv.org/pdf/2502.06910), [[代码]](https://github.com/decisionintelligence/TFB/tree/master/ts_benchmark/baselines/timekan)。
        
  - [ ] **xPatch** - xPatch: 具有指数季节-趋势分解的双流时间序列预测 [[AAAI 2025]](https://arxiv.org/pdf/2412.17323), [[代码]](https://github.com/decisionintelligence/TFB/tree/master/ts_benchmark/baselines/xpatch)。

  - [ ] **HDMixer** - HDMixer: 用于多变量时间序列预测的具有可扩展补丁的分层依赖 [[AAAI 2024]](https://ojs.aaai.org/index.php/AAAI/article/view/29155), [[代码]](https://github.com/decisionintelligence/TFB/tree/master/ts_benchmark/baselines/hdmixer)。

  - [ ] **PatchMLP** - PatchMLP: 释放补丁的力量：基于补丁的MLP用于长期时间序列预测 [[AAAI 2025]](https://arxiv.org/pdf/2405.13575), [[代码]](https://github.com/decisionintelligence/TFB/tree/master/ts_benchmark/baselines/patchmlp)。
    
  - [ ] **Amplifier** - Amplifier: 在时间序列预测中关注被忽视的低能量成分 [[AAAI 2025]](https://arxiv.org/pdf/2501.17216), [[代码]](https://github.com/decisionintelligence/TFB/tree/master/ts_benchmark/baselines/amplifier)。
    
  - [x] **DUET** - DUET: 双聚类增强的多变量时间序列预测 [[KDD 2025]](https://arxiv.org/pdf/2412.10859), [[代码]](https://github.com/decisionintelligence/TFB/tree/master/ts_benchmark/baselines/duet)。

  - [x] **PDF** - 长期序列预测的周期性解耦框架 [[ICLR 2024]](https://openreview.net/pdf?id=dp27P5HBBt), [[代码]](https://github.com/decisionintelligence/TFB/tree/master/ts_benchmark/baselines/pdf)。

  - [x] **Pathformer** - Pathformer: 用于时间序列预测的具有自适应路径的多尺度变换器 [[ICLR 2024]](https://arxiv.org/pdf/2402.05956), [[代码]](https://github.com/decisionintelligence/TFB/tree/master/ts_benchmark/baselines/pathformer)。

  - [x] **FITS** - FITS: 用10k参数建模时间序列 [[ICLR 2024]](https://arxiv.org/pdf/2307.03756), [[代码]](https://github.com/decisionintelligence/TFB/tree/master/ts_benchmark/baselines/fits)。

## 目录

1. [介绍](#介绍)
1. [快速开始](#快速开始)
1. [开发自己方法的步骤](#开发自己方法的步骤)
1. [在自己的时间序列上评估的步骤](#在自己的时间序列上评估的步骤)
1. [时间序列代码漏洞drop-last说明](#时间序列代码漏洞drop-last说明)
1. [常见问题](#常见问题)
1. [引用](#引用)
1. [致谢](#致谢)
1. [联系](#联系)

## 介绍

TFB是一个专为时间序列预测研究者设计的开源库。

我们提供了一个干净的代码库，用于端到端评估时间序列预测模型，在各种评估策略和指标下将它们的性能与基线算法进行比较。

下图提供了TFB流水线的可视化概述。

<div align="center">
<img alt="Logo" src="docs/figures/Pipeline.png" width="80%"/>
</div>

下表提供了TFB的关键功能与其他时间序列预测库的比较的可视化概述。

![image-20240514151134923](docs/figures/Comparison_with_Related_Libraries.png)

## 快速开始
> [!IMPORTANT]
> 本项目在python 3.8下完全测试通过，建议您将Python版本设置为3.8。

1. 安装：

- 从PyPI

在给定的python环境中（**注意**：本项目在**python 3.8**下完全测试通过），使用以下命令安装依赖：

```shell
pip install -r requirements.txt
```

> [!IMPORTANT]
> 如果您想重现[scripts](https://github.com/decisionintelligence/TFB/tree/master/scripts)中的结果，请使用`requirements-docker.txt`文件而不是`requirements.txt`。这是因为`requirements-docker.txt`锁定了包的版本，而`requirements.txt`提供版本范围，这可能导致不同的依赖版本并影响重现的准确性。
> ```shell
> pip install -r requirements-docker.txt
> ```

- 从Docker

我们还为您提供了一个[Dockerfile](https://github.com/decisionintelligence/TFB/blob/master/Dockerfile)。要使此设置工作，您需要安装Docker服务。您可以在[Docker网站](https://docs.docker.com/get-docker/)获取它。

```shell
docker build . -t tfb:latest
```

```shell
docker run -it -v $(pwd)/:/app/ tfb:latest bash
```

2. 数据准备：

您可以从[Google Drive](https://drive.google.com/file/d/1vgpOmAygokoUt235piWKUjfwao6KwLv7/view?usp=drive_link)或[百度网盘](https://pan.baidu.com/s/1ycq7ufOD2eFOjDkjr0BfSg?pwd=bpry)获取预处理好的数据集。然后将下载的数据放在文件夹`./dataset`下。

3. 训练和评估模型：

我们在文件夹`./scripts/multivariate_forecast`和`./scripts/univariate_forecast`下提供了所有基准的实验脚本。例如，您可以按以下方式重现实验结果：

```shell
sh ./scripts/multivariate_forecast/ILI_script/DLinear.sh
```

## 开发自己方法的步骤
我们提供了关于如何开发自己方法的教程，您可以[点击这里](./docs/tutorials/steps_to_develop_your_own_method.md)。

## 在自己的时间序列上评估的步骤
我们提供了关于如何在自己的时间序列上评估的教程，您可以[点击这里](./docs/tutorials/steps_to_evaluate_your_own_time_series.md)。

## 时间序列代码漏洞drop-last说明
现有方法的实现经常在测试阶段采用`Drop Last技巧`。为了加速测试，通常将数据分成批次。但是，如果我们丢弃最后一个实例少于批大小的不完整批次，这会导致不公平的比较。例如，在图4中，ETTh2的测试序列长度为2,880，我们需要使用大小为512的回看窗口预测336个未来时间步。如果我们选择批大小为32、64和128，最后一个批次的样本数分别为17、49和113。**除非所有方法使用相同的批大小，否则丢弃最后一批测试样本是不公平的，因为测试集的实际使用长度不一致。**表2显示了PatchTST、DLinear和FEDformer在ETTh2上使用不同批大小并打开"Drop Last"技巧的测试结果。**我们观察到，当改变批大小时，方法的性能会发生变化。**

**因此，TFB呼吁测试过程避免使用drop-last操作以确保公平性，TFB在测试期间也没有使用drop-last操作。**

<div align="center">
<img alt="Logo" src="docs/figures/Drop-last.png" width="70%"/>
</div>

## 常见问题

1. 如何使用Pycharm运行代码？

在pycharm下运行时，请转义双引号，删除空格，并删除开头和结尾的单引号。

例如：**'{"d_ff": 512, "d_model": 256, "horizon": 24}' ---> {\\"d_ff\\":512,\\"d_model\\":256,\\"horizon\\":24}**

```shell
--config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args {\"horizon\":24} --model-name "time_series_library.DLinear" --model-hyper-params {\"batch_size\":16,\"d_ff\":512,\"d_model\":256,\"lr\":0.01,\"horizon\":24,\"seq_len\":104} --adapter "transformer_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "ILI/DLinear"
```

2. 如何获取模型的预测值和目标值？

我们提供了关于如何获取模型的预测值和目标值的教程，您可以[点击这里](./docs/tutorials/steps_to_get_predict_and_actual_data.md)。

3. 脚本编写示例。

如果您想并行运行数据集、测试多个数据集或测试多个算法等，您可以[点击这里](./docs/tutorials/other_usage.sh)。

4. 多变量预测评估结果使用的回看窗口有多长？

您可以在[这里](https://github.com/decisionintelligence/TFB/issues/40)找到答案。

5. 如何使用DataParallel在多个GPU上训练模型？

您可以在[这里](./docs/tutorials/steps_to_train_models_with_multi_gpus_using_dp.md)找到答案。

6. 如何仅预测输入变量的子集？

您可以在[这里](./docs/tutorials/steps_to_predict_only_a_subset_of_input_variables.md)找到答案。

7. TFB的代码库中是否有计算数据集特征的代码？

TFB已开源用于计算时间序列特征的[代码](https://github.com/decisionintelligence/TFB/blob/master/characteristics_extractor/Characteristics_Extractor.py)，如趋势、季节性、平稳性、移动、转换、相关性等。提供了[中文](https://github.com/decisionintelligence/TFB/blob/master/characteristics_extractor/Readme_chn.md)和[英文](https://github.com/decisionintelligence/TFB/blob/master/characteristics_extractor/Readme_en.md)文档。

## 引用

如果您觉得这个仓库有用，请引用我们的论文。

```
@article{qiu2024tfb,
  title   = {TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods},
  author  = {Xiangfei Qiu and Jilin Hu and Lekui Zhou and Xingjian Wu and Junyang Du and Buang Zhang and Chenjuan Guo and Aoying Zhou and Christian S. Jensen and Zhenli Sheng and Bin Yang},
  journal = {Proc. {VLDB} Endow.},
  volume  = {17},
  number  = {9},
  pages   = {2363--2377},
  year    = {2024}
}

@inproceedings{qiu2025duet,
  title     = {DUET: Dual Clustering Enhanced Multivariate Time Series Forecasting},
  author    = {Xiangfei Qiu and Xingjian Wu and Yan Lin and Chenjuan Guo and Jilin Hu and Bin Yang},
  booktitle = {SIGKDD},
  pages     = {1185-1196},
  year      = {2025}
}
```

## 致谢

这个库的开发得到了**华为云**的支持，我们要感谢他们的贡献和帮助。

## 联系

如果您有任何问题或建议，请随时联系：

- [邱翔飞](https://qiu69.github.io/) (xfqiu@stu.ecnu.edu.cn)
- [吴行健](https://ccloud0525.github.io/) (xjwu@stu.ecnu.edu.cn)
- 李正宇 (lizhengyu@stu.ecnu.edu.cn)
- 陆骏凯 (jklu@stu.ecnu.edu.cn)
- 裘王辉 (onehui@stu.ecnu.edu.cn)

或者在Issues中描述。

我们邀请您加入微信上的OpenTS社区。我们在微信上建立了一个群聊，您可以通过扫描[二维码](./docs/figures/QR.png)获得访问权限。通过加入社区，您可以获得OpenTS的最新更新，分享您的想法，并与其他成员讨论。

希望加入的朋友可以先扫描[二维码](./docs/figures/QR.png)通过微信联系我。**添加时请在备注中注明您的姓名和研究方向**。申请通过后，我们会邀请您加入群聊。进群后，请将群昵称更新为“**姓名+学校/机构+研究方向**”。一周内未更新备注的成员将被管理员定期清理。