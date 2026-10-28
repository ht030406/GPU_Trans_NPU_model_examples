# VPP2 for PyTorch

## 目录

- [VPP2 for PyTorch](#vpp2-for-pytorch)
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

Video Prediction Policy 2（VPP2）是基于 Wan2.1-14B 视频扩散主干构建的世界-动作模型：在视频 DiT（40 层，hidden 5120）之外并联一个 Action DiT（40 层，hidden 1024），通过 Mixture-of-Transformers 混合注意力把视频生成与动作预测联合于同一个前向，采用 flow-matching 回归训练（MSE）。训练配方：历史条件 Video-10k 预训练后，0-100k 步联合训练视频与随机初始化的动作头（详见仓库文档）。

## 支持任务列表

本仓已经支持以下模型任务类型。如下列表中Released为Y的表示已经过测试验证，N的表示开发自验通过。

|    模型     | 任务列表 | 数据集 | 是否支持 | Released |
| :---------: | :------: | :------: | :------: | :------: |
|    VPP2     |  训练（RoboDojo joint Video+Action）  |  RoboDojo  |    ✔    |    N     |

## 代码实现

- 参考实现（主仓）：

```shell
url=https://github.com/roboterax/video-prediction-policy-2.git
commit_id=47199cd0cd157fd859520e119b04a9cf0d6c0308
```

- 适配昇腾 AI 处理器的实现：

```shell
url=https://gitcode.com/Ascend/DrivingSDK.git
code_path=model_examples/VPP2
```

# 准备训练环境

## 安装昇腾环境

请参考昇腾社区中《[PyTorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/Pytorch/latest/installguide/swinstall/docs/zh/installation_guide)》文档搭建昇腾环境，本仓已支持表1中软件版本。实测清单（CANN 与 torch_npu 配对本交付验证过）：

```sh
# 1) CANN（root）: ascend-toolkit 9.0.0
ls /usr/local/Ascend/ascend-toolkit/set_env.sh
# 2) conda env + CPU 版 torch（NPU 上不能装 GPU 版 torch）
conda create -n vpp2_npu python=3.10 -y && conda activate vpp2_npu
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch_npu==2.7.1.post10
# 3) 验证（source CANN 后）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"
```

**表 1**  昇腾软件版本支持表

|     软件类型      | 首次支持版本 |
| :---------------: | :------: |
|      TorchNPU       |  26.0.0  |
|       CANN        |  9.0.0  |

## 安装模型环境

当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

**表 2**  版本支持表

|      三方库       |  支持版本  |
|:--------------:|:------:|
|    Python      |  3.10  |
|    PyTorch     |  2.7.1  |
|    torch_npu   | 2.7.1.post10 |
|    deepspeed   | 0.18.7 |

0. 激活 CANN 环境（例如：source /usr/local/Ascend/ascend-toolkit/set_env.sh）

1. 创建环境

    ```sh
    conda create -n vpp2_npu python=3.10 -y
    conda activate vpp2_npu
    ```

2. 准备主仓源码，应用 Patch

    参考原仓下载 Driving SDK 加速库：<https://gitcode.com/Ascend/DrivingSDK>

    ```sh
    git clone https://gitcode.com/Ascend/DrivingSDK.git
    cd ./DrivingSDK/model_examples/VPP2
    ```

    在 VPP2 目录下，克隆主仓，切换至指定 commit 并应用 patch（顺序不能反：先 `git apply` 打补丁，再覆盖交付文件）

    ```sh
    git clone https://github.com/roboterax/video-prediction-policy-2.git
    cd video-prediction-policy-2
    git checkout 47199cd0cd157fd859520e119b04a9cf0d6c0308
    cp -f ../VPP2.patch ./
    git apply --check VPP2.patch    # 1. 先在干净 checkout 上打补丁
    git apply VPP2.patch
    cp -rf ../test/ ./              # 2. 再覆盖交付文件
    cd ..
    ```

3. 安装模型依赖

    ```sh
    cd video-prediction-policy-2
    pip install -e . --no-build-isolation
    pip install -r requirements.txt
    ```

## 模型准备

训练前置权重（Video-10k 历史条件主干 + Wan2.1 共享资产），共 44.7GB 量级。

1. 下载预训练权重（Hugging Face，需 `python -m pip install -U huggingface_hub`）：

   ```sh
   hf download Haodong082399/VPP2 --local-dir weights \
     --include 'checkpoints/initialization/**' --include 'checkpoints/Wan2.1-I2V-14B-480P/**'
   ```

   ModelScope 镜像（需有访问权限的账号）：

   ```sh
   modelscope download --model haodong123/VPP2_preview --local_dir weights \
     --include 'checkpoints/initialization/**' 'checkpoints/Wan2.1-I2V-14B-480P/**'
   ```

   下载后布局：`weights/checkpoints/initialization/robodojo_his10k.pt`（历史条件 Video-10k 主干，对应 `paths.video_init`）、`weights/checkpoints/Wan2.1-I2V-14B-480P/`（共享 VAE fp32 / CLIP fp16 / 文本编码器资产，对应 `paths.wan`）。

2. 生成动作头初始权重（上游 release 不直接提供，由 `init-action` 从 video 主干导出）：

   ```sh
   bash scripts/robodojo/init_action.sh \
     --video weights/checkpoints/initialization/robodojo_his10k.pt \
     --output weights/init/action2b_from_his10k.pt
   ```

   产出 `weights/init/action2b_from_his10k.pt` 对应 `paths.action_init`。

3. 校验：`sha256sum weights/checkpoints/initialization/robodojo_his10k.pt` 与上游 release 对照。
4. 布局：`configs/robodojo/train_100k.yaml` 的 paths 节（`paths.wan` / `paths.video_init` / `paths.action_init`）与 `config.json` 中的嵌套绝对路径字段（如 `backbone`）需按本机路径修改；数据路径通过 `test/*.sh` 的 `--dataset_path` / `--media_path` 传入。

## 数据集下载

数据源为 RoboDojo 公共 EE16 LeRobot v3.0 导出（3500 集 / 1,856,102 帧 @25Hz，由 RoboDojo 上游配置下载），随后按序执行三段转换：

```sh
bash scripts/robodojo/convert.sh \
  --source /abs/path/RoboDojo_ee_lerobot_v30_video \
  --output /abs/path/robodojo_media --workers 4

bash scripts/robodojo/prepare.sh \
  --metadata /abs/path/robodojo_media/full_episode_metadata.csv \
  --media-root /abs/path/robodojo_media \
  --stats configs/robodojo/normalization_ee16.json \
  --output data/robodojo

bash scripts/robodojo/text_cache.sh \
  --data data/robodojo \
  --wan weights/checkpoints/Wan2.1-I2V-14B-480P
```

- `convert` 把 Parquet+MP4 转成原生帧 episode 文件（`--limit 2` 可做小样本冒烟，但不能通过全量 split 校验）；
- `prepare` 固定 3466 训练 / 34 留出切分并生成 critical_index；
- `text_cache` 预抽取 CLIP text embeddings（`--device` 默认 cuda，NPU 上由补丁的 transfer_to_npu 透明接管）。

最终目录结构：

```text
data/
├── robodojo/            # 原始媒体与转换产物（由 vpp2 prepare 消费）
│   ├── train.csv
│   └── videos_original/
└── prepared/            # vpp2 prepare 输出（训练直接消费）
    ├── train.csv        # 3466 条
    ├── val.csv          # 34 条
    ├── critical_index/
    ├── dataset_stats.json
    ├── prompts.json
    └── text_embeds/     # vpp2 text-cache 输出（CLIP embeddings）
```

# 快速开始

本任务主要提供**单机 8 卡**的训练脚本（`test/`，多机由 `NNODES`/`WORLD_SIZE` 覆盖）。运行前 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 并激活 `vpp2_npu` 环境启动各脚本；脚本会自动 source `test/env_npu.sh`（NPU 训练环境变量基线，含 CPU 绑核、RMSNorm 融合开关等）。

- 精度训练（跑满 max_steps，报 FinalLoss）：

  ```shell
  bash test/train_full_8p.sh \
      --dataset_path=/abs/data/prepared \
      --media_path=/abs/data/robodojo \
      [--max_steps=100000] [--num_npus=8] [--output_dir=...]
  ```

- 性能训练（确定性开关关闭，末 N 步均值算 SPS）：

  ```shell
  bash test/train_performance_8p.sh \
      --dataset_path=/abs/data/prepared \
      --media_path=/abs/data/robodojo \
      [--max_steps=4] [--perf_steps=2] [--num_npus=8] [--output_dir=...]
  ```

- 冒烟：小步数验证链路：

  ```shell
  bash test/train_performance_8p.sh --max_steps=2 --perf_steps=1 --dataset_path=... --media_path=...
  ```

> ⚠️ 两个主脚本已固定传 `eval_at_start=false`（与基线口径一致），不要移除它：`train_100k.yaml` 默认 `eval_at_start: true`，不加会触发 HCCL 不支持的 fp64 allreduce 崩溃（`Unsupported data type at::kDouble`；eval 指标汇总的 fp64→fp32 修复已包含在 VPP2.patch 的 `src/vpp2/trainer.py` 中）。

## 训练结果展示

**表 3**  训练结果展示表（global batch 288 = micro1 × 8 卡 × grad_accum 36）

|      任务      |      芯片      | 卡数 | global batch size | Precision | max steps | loss | SPS |
| :-----------: | :-----------: | :--: | :---------------: | :-------: | :-------: | :---: | :-: |
| RoboDojo joint 训练 | 竞品A | 8p | 288 | bf16 mixed | 150 | 0.4791 | 0.05 |
| RoboDojo joint 训练 | Atlas 800T A3 | 8p-16die | 288 | bf16 mixed | 150 | 0.4808 | 0.15 |

**表 4** 结果字段说明

|     字段      | 说明 |
| :-----------: | :---------------: |
|     卡数     |    训练使用的昇腾芯片数量      |
| global batch size |    单次迭代全局样本数（micro × 卡数 × grad_accum）      |
|      SPS     |     Samples Per Second，平均每秒处理的样本总数      |

# 版本说明

## 变更

2026.10：首次发布。

## FAQ

Q: 8 卡显存是否够用？

A：8×910B 每卡 61.3GB 可用；基线配置峰值 allocated 36.8GiB / reserved 49.5GiB。`test/env_npu.sh` 默认 `TASK_QUEUE_ENABLE=1`（与全部基线/精度/性能跑一致；A/B 实测 TQE=2 在模型形状链上零收益且官方注明有内存代价，故收敛回 1）。

Q: 训练权重是否进 git？

A：不进。`weights/`、`data/`、`runs/` 由 `.gitignore` 排除。
