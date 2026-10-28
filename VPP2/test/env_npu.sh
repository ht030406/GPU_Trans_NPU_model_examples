#!/bin/bash
# ============================================================================
# VPP2 昇腾 NPU 训练环境变量基线（交付件）
#
# 收录判据：① 实测通用；② 偏离官方默认值（写默认值是 no-op，不收）。
# 变量含义/出处：MindSpeed-Agent skills vla-model-adaptation-optimization
#   references/appendix-env.md。
# ============================================================================

# ---- 算子下发二级流水（官方建议值 2）----
# NOTE(VPP2): 本模型显存吃紧（8×910B 61.3GB，前向激活峰值 ~49GB），实测
#   TASK_QUEUE_ENABLE=2 在 micro_batch=2 时会抬升内存峰值导致 OOM（官方注明
#   =2 有内存代价）。B 期 A/B 判据：模型形状链基准（40 段 gelu+rmsnorm+cast+
#   matmul）TQE 1 vs 2 = 6.3 vs 6.3 ms/iter **零收益**，按「中性回退」规则收敛
#   为基线值 1（= 全部基线/精度/性能跑所用值，与 README 表 3 数字口径一致）；
#   若目标机器显存宽裕且想再试 =2，需先跑 perf+prec 双臂 A/B 有量化收益才采纳。
export TASK_QUEUE_ENABLE=1

# ---- CPU 绑核（粗粒度）----
export CPU_AFFINITY_CONF=1

# ---- RMSNorm 融合（C1 采纳项，mode=2 bf16 累积域）----
# NOTE(VPP2): wan_video_dit.py WanRMSNorm 在 NPU 上用 torch_npu.npu_rms_norm
#   替换 7-kernel fp32 链，省掉 [M,5120] fp32 中间张量写读。C1 双臂 A/B
#   （2026-10-14，16 卡全空 node-gate 干净窗口）：
#     mode=1（fp32 累积域）: 最干净步 1894s ≈ 基线 1897.3s = 中性 → 回退。
#     mode=2（bf16 累积域）: 1884/1880/1881s vs 基线 1895/1900/1897s
#       = **-0.83%**（t=8.31, p=0.0007，6.2× 窗口噪声下限 0.13%）；
#       prec loss step2 2.3670 与基线逐位一致，step1 Δ=4e-4 → 采纳。
#     代码默认 "0" = 基线路径逐位一致；本文件统一置 "2" 为训练默认。
export VPP2_NPU_RMSNORM=2

# ---- 显存：扩展段（长序列模型建议开）----
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# ---- 杂项 ----
export TOKENIZERS_PARALLELISM=false   # 关 tokenizer 多进程告警
export WANDB_MODE=offline             # 离线环境

# ---------------------------------------------------------------------------
# 仅精度跑开（有性能代价）——性能跑脚本会显式覆盖为关闭
# ---------------------------------------------------------------------------
# export HCCL_DETERMINISTIC=true
# export CLOSE_MATMUL_K_SHIFT=1

echo "[env_npu.sh] NPU environment configured."

# ============================================================================
# ⚠️ ASCEND_LAUNCH_BLOCKING=1 只允许 debug 阶段临时用（同步下发，性能大幅下降，
#    且会让 TASK_QUEUE_ENABLE 整体失效）。交付件里必须删掉。自查：
#        grep -rn "ASCEND_LAUNCH_BLOCKING" test/ && echo "残留了调试变量！"
# ============================================================================
