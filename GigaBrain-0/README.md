
# GigaBrain-0 for PyTorch

## 目录

- [GigaBrain-0 for PyTorch](#gigabrain-0-for-pytorch)
  - [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [准备训练环境](#准备训练环境)
  - [安装昇腾环境](#安装昇腾环境)
  - [安装模型环境](#安装模型环境)
  - [模型准备](#模型准备)
  - [数据集下载](#数据集下载)
- [快速开始](#快速开始)
  - [训练结果展示](#训练结果展示)
- [版本说明](#版本说明)
  - [变更](#变更)
  - [FAQ](#faq)

# 简介

## 模型介绍

GigaBrain 是面向具身智能的视觉-语言-动作（VLA）基座模型。本样例基于 GigaBrain-0.7 发布代码适配昇腾 NPU：GigaBrain-0.7 采用“三系统”架构统一理解、预测与动作生成，基于超过 37000 小时的异构具身数据预训练，并通过单阶段对齐训练联合优化视觉语言理解与多本体动作生成，在零样本泛化、语言条件指令跟随以及后训练任务成功率上均有明显提升。本样例覆盖官方两套主线后训练配置：

- `gb07_pg2_pick_and_place_piper_30k.py`：AgileX PiPER 机器人抓取-放置后训练；
- `gb07_pg2_push_buttons_h01_30k.py`：Maker H01 机器人按键后训练。

## 支持任务列表

本仓已经支持以下模型任务类型。如下列表中Released为Y的表示已经过测试验证，N的表示开发自验通过。

|    模型     | 任务列表 | 数据集 | 是否支持 | Released |
| :---------: | :------: | :------: | :------: | :------: |
|  GigaBrain-0 |  SFT后训练  |  H01/PiPER  |    ✔    |    N     |

## 代码实现

- 参考实现（主仓）：

```shell
url=https://github.com/open-gigaai/giga-brain-0.git
commit_id=b3b0cb2a5095f0f6e9f23837b5abafcba51e11f0
```

- 参考实现（依赖仓 giga-models，含 VLA 模型代码）：

```shell
url=https://github.com/open-gigaai/giga-models.git
commit_id=eb8da91c490a8b36f315acb1638f123fb41bef56   # tag 1.1.0
```

- 适配昇腾 AI 处理器的实现：

```shell
url=https://gitcode.com/Ascend/DrivingSDK.git
code_path=model_examples/GigaBrain-0
```

# 准备训练环境

## 安装昇腾环境

请参考昇腾社区中《[PyTorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/Pytorch/latest/installguide/swinstall/docs/zh/installation_guide)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1**  昇腾软件版本支持表

|     软件类型      | 首次支持版本 |
| :---------------: | :------: |
| TorchNPU | 2.7.1.post8  |
|       CANN        |  9.0.0  |

## 安装模型环境

当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

**表 2**  版本支持表

|      三方库       |  支持版本  |
|:--------------:|:------:|
|    Python      | 3.11.10 |
|    PyTorch     | 2.7.1 |
|    TorchNPU    | 2.7.1.post8 |
|    accelerate  | 1.13.0 |
|    transformers | 4.54.1 |

0. 激活 CANN 环境（例如：source /usr/local/Ascend/ascend-toolkit/set_env.sh）

1. 创建环境

    ```sh
    conda create -n gigabrain07 python=3.11.10 -y
    conda activate gigabrain07
    ```

2. 准备主仓源码，应用 Patch

    参考原仓下载 Driving SDK 加速库：<https://gitcode.com/Ascend/DrivingSDK>

    ```sh
    git clone https://gitcode.com/Ascend/DrivingSDK.git
    cd ./DrivingSDK/model_examples/GigaBrain-0
    ```

    在 GigaBrain-0 目录下，克隆主仓，切换至指定 commit 并应用 patch

    ```sh
    git clone https://github.com/open-gigaai/giga-brain-0.git
    cd giga-brain-0
    git checkout b3b0cb2a5095f0f6e9f23837b5abafcba51e11f0
    cp ../giga_brain_0.patch .
    git apply --check giga_brain_0.patch
    git apply giga_brain_0.patch
    cd ..
    ```

    > **说明：** 该 patch 中的 NPU 适配包括：关闭 `torch.compile/inductor`（NPU 上使用 eager 模式）、训练进程内提前导入 `torch_npu` 并限制 torch 线程池以规避 aarch64 fork DataLoader 的并行死锁、注释 requirements 中 CUDA 相关的依赖锁定（`torch==2.7.1`、`torchvision`、`torchcodec`、`triton`、`decord`）。数据集路径、`gpu_ids`、`batch_size_per_gpu` 等运行参数按官方说明由用户按环境编辑（见下文[快速开始](#快速开始)），测试脚本会临时注入 `batch_size_per_gpu`/`max_steps`/`log_interval`/`resume`/`gpu_ids` 并在退出时自动还原。

3. 安装主仓依赖

    ```sh
    cd giga-brain-0
    python -m pip install -r requirements-paligemma2.txt
    python -m pip install "giga-datasets==1.1.0"
    python -m pip install "giga-train==1.1.0"
    cd ..
    ```

4. 准备依赖仓 giga-models 并应用 Patch

    giga-models 包含 VLA 模型实现，NPU 适配（`torch.autocast` 设备类型、时间步嵌入 dtype 转换）以 patch 形式提供：

    ```sh
    git clone https://github.com/open-gigaai/giga-models.git
    cd giga-models
    git checkout eb8da91c490a8b36f315acb1638f123fb41bef56
    cp ../giga_models.patch .
    git apply --check giga_models.patch
    git apply giga_models.patch
    pip install -e . --no-deps
    cd ..
    ```

5. 设置环境变量

    ```sh
    export GIGA_BRAIN_ROOT="$(pwd)/giga-brain-0"
    export PYTHONPATH="$GIGA_BRAIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    ```

## 模型准备

这一步同时是训练和推理的前置项。

- 下载 GigaBrain-0.7 预训练权重与分词器，放到 `giga-brain-0/checkpoints` 下（路径对应训练 config 中的 `PRETRAINED_PALIGEMMA_PATH`、`PALIGEMMA2_TOKENIZER_PATH`、`FAST_TOKENIZER_PATH`）：

| 资源 | 链接 | 说明 |
| :---: | :---: | :--- |
| GigaBrain-0.7-3.5B-Base | https://huggingface.co/open-gigaai/GigaBrain-0.7-3.5B-Base | 后训练的预训练基座（PaliGemma2 3.5B） |
| PaliGemma2 tokenizer | https://huggingface.co/google/paligemma2-3b-pt-224 | 对应 `models--google--paligemma2-3b-pt-224` |
| Fast tokenizer | https://huggingface.co/physical-intelligence/fast | 对应 `models--physical-intelligence--fast` |

## 数据集下载

GigaBrain-0.7 示例数据已发布到：

- https://huggingface.co/datasets/open-gigaai/GigaBrain-0.7-SampleData

在 `giga-brain-0/data/GigaBrain-0.7-SampleData` 下下载并解压后，目录结构应为：

```text
giga-brain-0/data/GigaBrain-0.7-SampleData/
├── pick_and_place_piper_v30/
└── push_button_h01_v30/
    ├── push_button_red/
    ├── push_button_blue/
    └── push_button_yellow/
```

训练数据为 LeRobot 格式。使用 `scripts/compute_norm_stats_fast.py` 生成每个任务的归一化统计：

```sh
cd giga-brain-0
python scripts/compute_norm_stats_fast.py \
  --data-paths data/GigaBrain-0.7-SampleData/push_button_h01_v30/push_button_red \
    data/GigaBrain-0.7-SampleData/push_button_h01_v30/push_button_blue \
    data/GigaBrain-0.7-SampleData/push_button_h01_v30/push_button_yellow \
  --output-path data/norm_stats/push_buttons_h01_norm_stats.json \
  --embodiment-id 7 \
  --sample-rate 1.0 \
  --action-chunk 50 \
  --action-dim 32 \
  --num-workers 64
```

> **说明：** 各 config 中的 `PROJECT_ROOT`、`DATA_PATHS`、`NORM_STATS_PATH`、`PRETRAINED_PALIGEMMA_PATH`、`PALIGEMMA2_TOKENIZER_PATH`、`FAST_TOKENIZER_PATH` 为示例内部路径，运行前请按本机实际路径编辑。

# 快速开始

进入 DrivingSDK/model_examples/GigaBrain-0 目录：

>**说明：**
>Ascend 950训练前需：
>
>1、注释 `test/train_full.sh` 和 `test/train_performance.sh` 中的 `export TORCH_HCCL_ZERO_COPY=1`。
>
>2、取消注释 `test/train_full.sh` 和 `test/train_performance.sh` 中的 `export HCCL_OP_EXPANSION_MODE="CCU_SCHED"`。
>

运行脚本支持命令行参数（支持默认值+关键字参数+位置参数）
  - `--gpu-ids`：使用的NPU卡id（逗号分隔，如 0,1,2,3,4,5,6,7），默认值0-7；
  - `--nproc-per-node`：每节点NPU卡数，默认值8（未指定 `--gpu-ids` 时生成 0..N-1）；
  - `--task`：运行任务名称，可选 `gb07_pg2_push_buttons_h01_30k` / `gb07_pg2_pick_and_place_piper_30k`，默认值gb07_pg2_push_buttons_h01_30k；
  - `--batch-size`：每卡batch size大小，默认值8；
  - `--max-steps`：最大运行步数；
  - `--log-interval`：日志打印间隔，默认值1；

- 单机8卡精度训练

  快速开始（参数均为默认值）

  ```shell
  bash test/train_full.sh
  ```

  参数配置

  ```shell
  bash test/train_full.sh --gpu-ids=0,1,2,3,4,5,6,7 --task=gb07_pg2_push_buttons_h01_30k --batch-size=8 --max-steps=3000
  ```

- 单机8卡性能训练

  快速开始（参数均为默认值）

  ```shell
  bash test/train_performance.sh
  ```

  参数配置

  ```shell
  bash test/train_performance.sh --gpu-ids=0,1,2,3,4,5,6,7 --task=gb07_pg2_push_buttons_h01_30k --batch-size=8 --max-steps=500
  ```

  也可在仓库根目录直接用 `python scripts/train.py --config configs/gb07_pg2_push_buttons_h01_30k.py` 启动（需自行按环境编辑 config）。

## 训练结果展示

**表 3**  训练结果展示表（H01 后训练）

|     任务      |     芯片      | 卡数 | batch size | max steps | Final loss | SPS |
| :-----------: | :-----------: | :--: | :---------------: | :--------------------: | :--------------------:  | :--------------------: |
| push_buttons_h01 | 竞品A |  8p  |         8       | 1000 |  0.0226 |  44.54 |
| push_buttons_h01 | Atlas 800T A3 |  8p-16die  |         8       | 1000 |  0.0194 |  81.18 |

**表 4** 结果字段说明

|     字段      | 说明 |
| :-----------: | :---------------: |
|     卡数     |    训练使用的昇腾芯片数量      |
| batch size  |    每次训练迭代每卡处理的样本总数      |
| SPS |     Samples Per Second，平均每秒处理的样本总数      |

# 版本说明

## 变更

2026.10: 首次发布。

## FAQ

Q: 设置环境变量 `TORCH_HCCL_ZERO_COPY=1` 报错？

A：当前该环境变量仅支持 A3，910b 和 950 不能开启，请在运行脚本前注释 `test/train_full.sh` 和 `test/train_performance.sh` 中的 `export TORCH_HCCL_ZERO_COPY=1`。

Q: 设置环境变量 `HCCL_OP_EXPANSION_MODE="CCU_SCHED"` 报错？

A：该环境变量仅支持 950，910b 和 A3 不能开启，请保持 `test/train_full.sh` 和 `test/train_performance.sh` 中 `export HCCL_OP_EXPANSION_MODE="CCU_SCHED"` 前的注释。

Q: 训练时报数据集路径不存在？

A：config 中的 `PROJECT_ROOT`、`DATA_PATHS`、`NORM_STATS_PATH`、`PRETRAINED_PALIGEMMA_PATH` 等为示例内部路径，请按本机实际下载路径编辑 `configs/gb07_pg2_push_buttons_h01_30k.py`（或 piper 配置）后重新运行。
