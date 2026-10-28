
# FasterWAM for PyTorch

## 目录

- [FasterWAM for PyTorch](#fasterwam-for-pytorch)
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

FasterWAM 是一个面向机器人具身控制的 World Action Model（WAM，世界动作模型）训练与推理框架。训练时利用视频生成/预测学习世界表征，推理时不再生成未来视频，而是基于 one-pass future cache 模式、根据当前观测和任务指令直接预测动作，从而在保持较强控制性能的同时显著降低推理延迟。

## 支持任务列表

本仓已经支持以下模型任务类型。如下列表中Released为Y的表示已经过测试验证，N的表示开发自验通过。

|    模型     | 任务列表 | 数据集 | 是否支持 | Released |
| :---------: | :------: | :------: | :------: | :------: |
|  FasterWAM  |  SFT训练  |  LIBERO  |    ✔    |    N     |

## 代码实现

- 参考实现：

```shell
url=https://github.com/hustvl/FasterWAM.git
commit_id=83667817df0d4f823f39d90700e61ea2f432ac45
```

- 适配昇腾 AI 处理器的实现：

```shell
url=https://gitcode.com/Ascend/DrivingSDK.git
code_path=model_examples/FasterWAM
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
|    Python      | 3.10 |
|    PyTorch     | 2.7.1 |
|    TorchNPU    | 2.7.1.post8 |
|    DeepSpeed   | 0.18.5 |

0. 激活 CANN 环境（例如：source /usr/local/Ascend/ascend-toolkit/set_env.sh）

1. 创建环境

    创建conda环境

    ```sh
    conda create -n fasterwam python=3.10 -y
    conda activate fasterwam
    ```

2. 准备模型源码，应用 Patch 并安装依赖

    参考原仓下载 Driving SDK 加速库：<https://gitcode.com/Ascend/DrivingSDK>

    ```sh
    git clone https://gitcode.com/Ascend/DrivingSDK.git
    cd ./DrivingSDK/model_examples/FasterWAM
    ```

    在 FasterWAM 目录下，克隆原始仓，切换至指定 commit 并应用 patch

    ```sh
    git clone https://github.com/hustvl/FasterWAM.git
    cd FasterWAM
    git checkout 83667817df0d4f823f39d90700e61ea2f432ac45
    cp ../fasterwam.patch .
    git apply --check fasterwam.patch
    git apply fasterwam.patch
    pip install -U pip
    pip install -e .
    ```

3. 安装torchvision

    ```sh
    pip install torchvision==0.22.1
    ```

4. 安装ffmpeg

    - 推荐基于conda安装

    ```sh
    # 安装ffmpeg
    conda install -c conda-forge ffmpeg=4.4.2
    export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
    ```

    - 若采用源码安装，则步骤如下

    ```sh
    # 下载源码
    wget https://ffmpeg.org/releases/ffmpeg-4.4.2.tar.bz2
    tar -xvf ffmpeg-4.4.2.tar.bz2
    cd ffmpeg-4.4.2
    export PKG_CONFIG_PATH=/usr/local/lib64/pkgconfig:/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
    # 执行此步时环境中可能需要手动下载部分依赖包
    ./configure --prefix=/usr/local/ffmpeg --disable-doc --disable-openssl --enable-avresample --enable-demuxer=dash --enable-hardcoded-tables --enable-libfreetype --enable-libfontconfig --enable-libopenh264 --enable-gnutls --enable-libmp3lame --enable-libvpx --enable-pthreads --enable-gpl --enable-libx264 --enable-libx265 --enable-libaom --enable-libsvtav1 --enable-libdav1d --enable-libxml2 --enable-pic --enable-shared --disable-static --enable-version3 --enable-zlib
    make -j 64
    make install
    cd ..

    # 编辑全局配置文件
    vim /etc/profile.d/ffmpeg.sh

    # 添加以下内容
    export PATH="/usr/local/ffmpeg/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/ffmpeg/lib:$LD_LIBRARY_PATH"

    # 使配置立即生效
    source /etc/profile
    # 运行命令后应正常输出相关配置等信息
    ffmpeg
    ```

5. 安装torchcodec（可选）

    训练默认使用 `pyav` 作为视频解码器，不安装 torchcodec 也能正常运行；如需安装，可参考如下源码方式：

    ```sh
    # 安装pybind11
    conda install -c conda-forge pybind11
    # 源码安装torchcodec
    git clone https://github.com/meta-pytorch/torchcodec.git
    cd torchcodec
    git checkout v0.5.0
    python -m pip install PyYAML
    pip install -e . --no-build-isolation
    cd ..
    ```

## 模型准备

这一步同时是训练和推理的前置项。

- 第一步，先在 DrivingSDK/model_examples/FasterWAM/FasterWAM 目录下设置 Wan 模型目录（可选，默认 `./checkpoints`）：

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
```

- 第二步，预生成 FasterWAM 的 SparseActionDiT backbone（从 Wan22 DiT 插值）：

```bash
python scripts/preprocess_sparse_action_dit_backbone.py \
  --model-config configs/model/fasterwam.yaml \
  --output checkpoints/SparseActionDiT_cond_0_4_8_12_16_20_24_28_Wan22_alphascale_1024hdim.pt \
  --device npu \
  --dtype bfloat16
```

## 数据集下载

FasterWAM 使用与 Fast-WAM 相同的 LIBERO 预处理数据，已发布到：

- https://huggingface.co/datasets/yuanty/LIBERO-fastwam

先下载全部压缩包，在 DrivingSDK/model_examples/FasterWAM/FasterWAM 目录下全部解压：

```bash
mkdir -p data/libero_mujoco3.3.2
cd data/libero_mujoco3.3.2

# 下载 4 个 tar.gz 文件后执行
for f in *.tar.gz; do
  tar -xzf "$f"
done
```

解压后目录结构应为：

```text
data/libero_mujoco3.3.2/
├── libero_10_no_noops_lerobot/
├── libero_goal_no_noops_lerobot/
├── libero_object_no_noops_lerobot/
└── libero_spatial_no_noops_lerobot/
```

# 快速开始

1.预训练

使用 `scripts/precompute_text_embeds.py`，按训练 task 预计算：

进入 DrivingSDK/model_examples/FasterWAM/FasterWAM 目录

```bash
python scripts/precompute_text_embeds.py task=libero_fasterwam_2cam224_1e-4
```

如有多卡可用：

```bash
torchrun --standalone --nproc_per_node=8 scripts/precompute_text_embeds.py task=libero_fasterwam_2cam224_1e-4
```

2.训练

进入 DrivingSDK/model_examples/FasterWAM 目录：

>**说明：**
>Ascend 950训练前需：
>
>1、注释 `test/train_full.sh` 和 `test/train_performance.sh` 中的 `export TORCH_HCCL_ZERO_COPY=1`。
>
>2、取消注释 `test/train_full.sh` 和 `test/train_performance.sh` 中的 `export HCCL_OP_EXPANSION_MODE="CCU_SCHED"`。
>

- 单机8卡精度训练

  运行脚本支持命令行参数（支持默认值+关键字参数+位置参数）
  - `--nproc_per_node`：每节点NPU卡数，默认值8；
  - `--task`：运行任务名称，默认值libero_fasterwam_2cam224_1e-4；
  - `--batch-size`：每卡batch size大小，默认值8；
  - `--max-steps`：最大运行步数，默认值1000；
  - `--log-every`：日志打印间隔，默认值1；

  快速开始（参数均为默认值）

  ```shell
  bash test/train_full.sh
  ```

  参数配置

  ```shell
  bash test/train_full.sh --nproc_per_node=8 --task=libero_fasterwam_2cam224_1e-4 --batch-size=8 --max-steps=1000
  ```

- 单机8卡性能训练

  运行脚本支持命令行参数（支持默认值+关键字参数+位置参数）
  - `--nproc_per_node`：每节点NPU卡数，默认值8；
  - `--task`：运行任务名称，默认值libero_fasterwam_2cam224_1e-4；
  - `--batch-size`：每卡batch size大小，默认值8；
  - `--max-steps`：最大运行步数，默认值500；
  - `--log-every`：日志打印间隔，默认值1；

  快速开始（参数均为默认值）

  ```shell
  bash test/train_performance.sh
  ```

  参数配置

  ```shell
  bash test/train_performance.sh --nproc_per_node=8 --task=libero_fasterwam_2cam224_1e-4 --batch-size=8 --max-steps=500
  ```

## 训练结果展示

**表 3**  训练结果展示表

|     芯片      | 卡数 | batch size | max steps | Final loss | SPS |
| :-----------: | :--: | :---------------: | :--------------------: | :--------------------:  | :--------------------: |
| 竞品A |  8p  |         8       | 1000 |  0.3128 |  52.22 |
| Atlas 800T A3 |  8p-16die  |         8       | 1000 |  0.3105 |  92.28 |

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
