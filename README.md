# 用 EEG 区分赌博任务中的两类试次

本项目用于在 OpenNeuro 的 ds004511 数据集上比较两种 EEG 分类方法。`agfl` 使用提取好的频带功率，`eegnet` 直接读取按事件对齐的 EEG 波形。两者都可以在同一组受试者和试次上比较加入图传播前后的结果。[公开数据集](https://github.com/OpenNeuroDatasets/ds004511)

预测目标来自赌博任务行为表的 `Trial_Condition` 列。代码将 `Probable_Truth` 编为 0，将 `Probable_Lie` 编为 1，其它条件写入剔除记录。结果反映模型对这两个任务标签的区分能力。数据集没有为每个试次提供独立核实的欺骗意图标签。

## 已有数据包时，直接启动 EEGNet

在已配置依赖的训练机器上，进入 `lies` 的上一级目录。将输入路径换成包含 `trials.csv`、`epochs.npy` 和 `metadata.json` 的目录，并使用新的输出目录。

```bash
python -m lies eegnet --input /path/to/prepared_bundle --output runs/eegnet-01
```

此命令默认使用 CPU，完成普通 EEGNet、带 AGFL 的 EEGNet 和零跳对照的留一受试者评测。最终的配对结果写入 `runs/eegnet-01/metrics.json`。从原始 BIDS 数据开始时，先按下文生成对齐表和数据包。本次修改的原理、公式及已知限制见[中文更新记录](upd.txt)。

## 在服务器或集群上准备环境

以下命令均在 **`lies` 的上一级目录** 执行。如果当前目录是仓库根目录，先运行 `cd ..`。`lies` 本身就是 Python 包，目录内直接包含 `__main__.py` 和 `cli.py`。把修改后的整个目录复制到服务器并保留目录名 `lies`。若你的包目录名为 `deception`，以下命令中的 `python -m lies` 相应改为 `python -m deception`。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas scipy "scikit-learn>=1.2" mne torch
python -m lies --help
```

示例使用 Bash，可选 Python 3.11 建立环境。已有包含这些依赖的集群环境时，可以直接激活该环境。准备 EEG 需要 MNE，两条神经网络评测命令都需要 PyTorch。使用 NVIDIA GPU 时，按集群驱动和软件环境安装支持 CUDA 的 PyTorch，并在分配到 GPU 的计算节点上运行 `eegnet --device cuda`。

## 用合成数据检查安装

这组命令生成 4 名虚拟受试者的 EEG，每人 8 个试次，然后对两条路径各跑一轮训练。以后重跑时请使用新的输出目录。

```bash
python -m lies mock --subjects 4 --trials 8 --output work/mock
python -m lies agfl --input work/mock --output work/mock-agfl --allow-synthetic --epochs 1
python -m lies eegnet --input work/mock --output work/mock-eegnet --allow-synthetic --epochs 1
```

`mock` 默认注入与标签相关的模拟信号，加上 `--noise-only` 可生成纯噪声对照。这些运行在结果中标为 `synthetic_check`。

## 准备公开数据

将 ds004511 原始 BIDS 数据放到服务器或挂载的数据盘。下面以 `/data/ds004511` 为例，请将 `BIDS_ROOT` 替换为实际根目录。程序读取 `sub-*/ses-01` 下 `task-GG_run-01` 的行为表、原始 EEG、通道表、事件表和 EEG 元数据。原始数据与逐试次对齐文件需另外提供。

```bash
BIDS_ROOT=/data/ds004511
python -m lies alignment-template --dataset-root "$BIDS_ROOT" --output work/alignment_eeg.csv
```

打开 `work/alignment_eeg.csv`，保留每行的 `Subject_ID` 和 `Trial_ID`。根据已经核实的实验时间记录填写以下两列，再继续运行。

| 列 | 填写方法 |
| --- | --- |
| `EEG_Onset` | 相对于该受试者 EEG 第一个采样点的秒数。 |
| `Event_Phase` | 明确写出选定的事件，例如正确性报告提示出现的时刻。所有受试者及试次必须使用同一事件阶段。 |

公开 EEG 的同一试次有多个同名 `Sync` 标记。模板只填写试次标识，时间由使用者核对。程序校验试次键、时间顺序和事件阶段，并记录对齐时间与最近事件的距离供质控。若只处理部分受试者，在生成模板和准备数据时均可加 `--subjects` 指定受试者。

```bash
python -m lies prepare --dataset-root "$BIDS_ROOT" --alignment work/alignment_eeg.csv --output work/ds004511-eeg
python -m lies audit --input work/ds004511-eeg
```

默认只准备 EEG 信号。程序保留受试者间共有的合格 EEG 通道，在目标时间窗两侧各取 5 秒滤波上下文，按条件执行 50 Hz 陷波，进行 1–45 Hz 零相位带通滤波，重采样至 256 Hz，再截取事件前 0.2 秒至事件后 1.5 秒。因 EEG 缺失、越界或坏段等原因剔除的试次记录在 `dropped_trials.csv`。这种使用完整时间窗和双向滤波的分析适用于离线评测。

准备好的目录包含 `trials.csv`（逐试次标签、特征和质控信息）、`epochs.npy`（试次 × 通道 × 采样点）及 `metadata.json`（通道顺序、采样率、来源和文件校验值）。若已有旧版数据包且保留了 `epochs.npy`，可重新计算当前版本的频带功率，并在后续命令中改用新目录。

```bash
python -m lies features --input work/old-bundle --output work/refeatured-eeg
```

## 运行模型

两条命令都在一次评测中完成训练、验证和测试。下面的例子明确写出默认的验证轮数和模型配置，便于核对作业参数。

```bash
python -m lies agfl --input work/ds004511-eeg --output runs/agfl-eeg \
  --configs none inter --grouping loso --inner-splits 3 --validation-epochs 20 40 60

python -m lies eegnet --input work/ds004511-eeg --output runs/eegnet-cpu \
  --configs none inter k0 --grouping loso --inner-splits 3 --validation-epochs 20 40 60 \
  --device cpu --batch-size 32
```

| 命令 | 输入和配置含义 |
| --- | --- |
| `agfl` | 每个试次按 4 个时间窗和 θ、α、β、γ 组合成 16 个 EEG 节点，每个节点汇集各通道的频带功率。默认比较共用训练折编码和 PCA 的 `none`（无图层）与 `inter`（在所选节点之间传播）。还可指定 `k0`、`intra`、`combined`。 |
| `eegnet` | 从 `epochs.npy` 小批量读取 EEG，利用训练折统计量逐通道标准化。EEGNet-8,2 将每个池化后的时间位置编码成 16 维向量。`none` 为纯 EEGNet，`inter` 在时间节点间传播，`k0` 保留零跳图变换。三种配置使用相同结构的卷积编码器和分类头。 |

**默认跨受试者评测。** 外层每次留出一名受试者测试（LOSO）。内层只在其余受试者中按人分折，用汇总的验证集平衡准确率为每个配置和种子选择训练轮数，同分时选较少轮数。随后在整个外层训练折重新训练并预测留出者。嵌套验证至少需要 3 名受试者，并且训练折需要足够的两类试次。

| 参数 | 用途 |
| --- | --- |
| `--validation-epochs 20 40 60`、`--inner-splits 3` | 默认候选训练轮数及内层验证折数。 |
| `--epochs 40` | 预先固定训练轮数，跳过内层选轮。与 `--validation-epochs` 二选一，仍执行外层测试。 |
| `--grouping within --epochs 40` | 在同一受试者的试次内进行分层交叉验证，回答针对已知受试者的预测问题。 |
| `--seeds 2026 2027 2028` | 用多个模型随机种子重复评测。默认一个种子 2026。`--seed` 指定划分种子，默认也为 2026。 |
| `--shuffle-labels` | 在每个外层训练受试者内部打乱标签，测试者保留原始标签，用作负对照。 |
| `--lr`、`--weight-decay` | 两个命令的默认学习率均为 0.001。特征版 `agfl` 的默认权重衰减为 0.01，`eegnet` 为 0。 |

在已分配的 NVIDIA GPU 上运行波形模型时，使用新输出目录并显式选择 CUDA。

```bash
python -m lies eegnet --input work/ds004511-eeg --output runs/eegnet-gpu --device cuda --batch-size 32
```

## 可选的 EEG 与肌电联合评测

特征版 `agfl` 支持 EEG 加肌电。先生成包含 `Physio_Onset` 的模板，并人工填写两路记录中同一事件阶段的时间。`Physio_Onset` 采用 BIDS 生理记录的参考时钟，以秒为单位。生理文件第一行的时间由 JSON 的 `StartTime` 指定。

```bash
python -m lies alignment-template --dataset-root "$BIDS_ROOT" --include-physiology --output work/alignment_physio.csv
# 核对并填写 EEG_Onset、Physio_Onset、Event_Phase 后继续
python -m lies prepare --dataset-root "$BIDS_ROOT" --alignment work/alignment_physio.csv \
  --include-physiology --output work/ds004511-physio
python -m lies agfl --input work/ds004511-physio --input-mode eeg_emg --output runs/agfl-eeg-emg
```

`prepare --include-physiology` 读取生理 TSV 及其 JSON，按同一事件窗计算已识别通道的均值、标准差、均方根、最小值、最大值和斜率。`--input-mode eeg_emg` 只加入 EMG 特征，`eeg_physiology` 加入所有可用生理组。EEGNet 始终只读取 EEG 波形。若研究问题涉及反应时，`prepare --include-behavior` 可另外提取答题后才可获得的反应时特征，供支持行为输入的评测命令使用。

## 查看结果

每次评测使用独立的空输出目录。下表对应示例中的 `runs/eegnet-cpu/`。

| 文件 | 内容 |
| --- | --- |
| `predictions.csv` | 每个外层测试试次的受试者、试次号、真实标签、配置、种子、预测和正类概率。 |
| `folds.csv`、`selection.csv`、`validation.csv` | 外层训练及测试受试者、各配置的入选轮数和状态、内层各候选轮数的表现。使用固定轮数时，内层验证表无数据行。 |
| `paired_comparisons.csv` | 有可比较配置时生成。列出每位受试者、每个种子的配对平衡准确率差值。默认 EEGNet 配置下可比较 `inter − none`、`k0 − none` 和 `inter − k0`。 |
| `metrics.json` | 包含覆盖率、准确率、平衡准确率、宏平均 F1、ROC AUC、逐人指标和配对汇总。 |
| `run.json`、`eegnet_config.json` 或 `agfl_config.json` | 记录命令参数、数据来源、依赖版本、代码及表格校验值，以及模型的实际配置。 |

判断图传播是否改善 EEGNet 时，优先查看 `metrics.json` 中 `paired_comparisons` 的 `inter − none` 和 `inter − k0`。均值先在每名受试者内部跨种子计算，再对受试者等权平均。结合 `participants_scored`、`participants_total` 和覆盖率阅读结果。某人的测试试次只有一种标签时，其个人平衡准确率无定义。

## 其他入口

`benchmark` 比较传统表格分类器，`spatial` 使用 CSP 或 xDAWN，`fusion` 支持概率融合与堆叠，`neural` 支持注意力、对比学习和蒸馏。`llm` 可将所选数值特征发送到用户配置的外部模型服务，需安装 `openai` 并提供模型名和 API 凭据。CatBoost 和 XGBoost 分别需要额外安装 `catboost`、`xgboost`。

只有旧 CSV 时，可先用 `audit` 检查最基本的试次信息。对于来源未经核实、但已含相应特征的表格，表格分类命令可通过 `--allow-unverified` 进行诊断。EEGNet 需要带实际波形的已对齐数据包。

```bash
python -m lies benchmark --help
python -m lies eegnet --help
python -m unittest lies.test_preparation_eeg lies.test_agfl_evaluation lies.test_eegnet lies.test_paired_metrics
```
