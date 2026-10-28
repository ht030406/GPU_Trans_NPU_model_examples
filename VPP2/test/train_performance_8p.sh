#!/bin/bash
# ============================================================================
# VPP2 —— 单机 8 卡训练（性能跑）
#
# 用法：
#   bash test/train_performance_8p.sh \
#       --dataset_path=/path/to/data/prepared \
#       --media_path=/path/to/data/robodojo \
#       [--max_steps=4] [--perf_steps=2] [--num_npus=8] [--output_dir=...]
#
# 口径（与 2-optimize.md §9 / 档案卡一致）：
#   - FPS = global_batch_size / step_time；step_time 取日志最后 perf_steps 个
#     optimizer step 的墙钟均值（跳过 warmup）。本模型一个 step = 36 个
#     micro step（~30min/step，见 README 结果表），默认 4 步、末 2 步统计。
#   - 性能跑关闭确定性开关；精度验证请跑 test/train_full_8p.sh（两次独立运行）。
#   - step_time 从主进程日志 [train] 行的时间戳差分得到（训练循环不打印
#     StepTime 字段，此处按日志时间戳适配）。
# ============================================================================

set -uo pipefail

CUR_DIR=$(cd "$(dirname "$0")" && pwd)
if [ -f "${CUR_DIR}/env_npu.sh" ]; then
    source "${CUR_DIR}/env_npu.sh"
fi

# 性能跑：确定性开关必须关闭（与精度跑分开，两次独立运行）
unset HCCL_DETERMINISTIC CLOSE_MATMUL_K_SHIFT 2>/dev/null || true

# ---------------------------------------------------------------------------
# 默认参数
# ---------------------------------------------------------------------------
Network="VPP2"
ROOT_DIR=$(pwd)

num_npus=8
micro_batch_size=1
grad_accum=36                 # global_batch = micro × npus × grad_accum = 288
global_batch_size=$((micro_batch_size * num_npus * grad_accum))
max_steps=5
perf_steps=2                  # 取最后 2 个 step 的墙钟均值
seed=100008

dataset_path=""
media_path=""
output_dir=${ROOT_DIR}/runs/train_performance_8p

nnodes=${NNODES:-1}
node_rank=${NODE_RANK:-0}
master_addr=${MASTER_ADDR:-127.0.0.1}
master_port=${MASTER_PORT:-29500}

# ---------------------------------------------------------------------------
# 解析 --key=value
# ---------------------------------------------------------------------------
for para in "$@"; do
    case "$para" in
        --num_npus*)      num_npus="${para#*=}" ;;
        --max_steps*)     max_steps="${para#*=}" ;;
        --perf_steps*)    perf_steps="${para#*=}" ;;
        --seed*)          seed="${para#*=}" ;;
        --dataset_path*)  dataset_path="${para#*=}" ;;
        --media_path*)    media_path="${para#*=}" ;;
        --output_dir*)    output_dir="${para#*=}" ;;
        --nnodes*)        nnodes="${para#*=}" ;;
        --node_rank*)     node_rank="${para#*=}" ;;
        --master_addr*)   master_addr="${para#*=}" ;;
        --master_port*)   master_port="${para#*=}" ;;
    esac
done

if [[ "$dataset_path" == "" || ! -d "$dataset_path" ]]; then
    echo "[Error] para \"dataset_path\" must point to an existing prepared data dir."; exit 1
fi
if [[ "$media_path" == "" || ! -d "$media_path" ]]; then
    echo "[Error] para \"media_path\" must point to an existing robodojo media dir."; exit 1
fi
if [[ "${max_steps}" =~ ^[1-9][0-9]*$ ]] && [[ "${perf_steps}" =~ ^[1-9][0-9]*$ ]]; then :; else
    echo "[Error] max_steps / perf_steps must be positive integers."; exit 1
fi
if [ "${perf_steps}" -gt "${max_steps}" ]; then
    echo "[Warn] perf_steps(${perf_steps}) > max_steps(${max_steps})：实际只有 max_steps 步参与统计"
fi

WORLD_SIZE=$((num_npus * nnodes))
CaseName=${Network}_bs${global_batch_size}_${WORLD_SIZE}p
LOG_DIR=${CUR_DIR}/output/${CaseName}
mkdir -p "${LOG_DIR}" "$output_dir"
LOG_FILE=${LOG_DIR}/train.log

echo "=========================================="
echo " Network        : ${Network}"
echo " NPUs           : ${num_npus} x ${nnodes} nodes = ${WORLD_SIZE}"
echo " global batch   : ${global_batch_size} (micro ${micro_batch_size} x ${num_npus} x grad_accum ${grad_accum})"
echo " max steps      : ${max_steps}  (perf 取末 ${perf_steps} 步均值)"
echo " seed           : ${seed}"
echo " dataset        : ${dataset_path}"
echo " media          : ${media_path}"
echo " log            : ${LOG_FILE}"
echo "=========================================="

start_time=$(date +%s)

NNODES=${nnodes} NPROC_PER_NODE=${num_npus} NODE_RANK=${node_rank} \
MASTER_ADDR=${master_addr} MASTER_PORT=${master_port} \
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$(seq -s, 0 $((num_npus - 1)))}" \
bash "${ROOT_DIR}/scripts/robodojo/train.sh" \
    --allow-batch-change \
    "paths.data=${dataset_path}" \
    "paths.media=${media_path}" \
    "max_steps=${max_steps}" \
    "batch_size=${micro_batch_size}" \
    "gradient_accumulation_steps=${grad_accum}" \
    "seed=${seed}" \
    "log_every=1" \
    "save_every=0" \
    save_final_checkpoint=false \
    "eval_at_start=false" \
    "output_dir=${output_dir}" \
    2>&1 | tee "${LOG_FILE}"
status=${PIPESTATUS[0]}

end_time=$(date +%s)
e2e_time=$((end_time - start_time))

if [ ! -f "${LOG_FILE}" ]; then
    echo "Log Error: 日志文件 ${LOG_FILE} 未生成"; exit 1
fi

# ---------------------------------------------------------------------------
# step_time：主进程 [train] 行时间戳（HH:MM:SS）差分，取末 perf_steps 个的均值
#   NH：rich logger 会把 "[train]" 标记当样式标签剥掉（日志里是
#   ">>  epoch=.. step=../.. loss=.."），grep "\[train\]" 匹配不到行；
#   改按同一物理行上的 "step=<i>/<n> loss=" 特征匹配。
# ---------------------------------------------------------------------------
TRAIN_LINE_RE='step=[0-9]+/[0-9]+ loss='
# 末 perf_steps 个 step 的均值需要 perf_steps+1 个时间戳（N+1 个点得 N 个差分）
step_time=$(grep -aE "${TRAIN_LINE_RE}" "${LOG_FILE}" \
    | grep -aoE "\[[0-9]{2}:[0-9]{2}:[0-9]{2}\]" \
    | tail -n $((perf_steps + 1)) \
    | awk -F'[:\[\]]' '{t = $2*3600 + $3*60 + $4; if (NR > 1) {d = t - prev; sum += d; n++} prev = t} END {if (n > 0) printf "%.3f", sum / n}')

if [ -z "$step_time" ]; then
    echo "Log Error: 未找到可用的 [train] 时间戳记录（需至少 2 行 log_every=1 的 [train] 日志）"
    exit 1
fi

FPS=$(awk 'BEGIN{printf "%.4f\n", '${global_batch_size}' / '${step_time}'}')
TrainingTime=$(awk 'BEGIN{printf "%.1f\n", '${max_steps}' * '${step_time}'}')

# --- Loss（附属信息；本脚本只统计，判定以精度跑为准）---
FinalLoss=$(grep -aE "${TRAIN_LINE_RE}" "${LOG_FILE}" | grep -aoE "loss=[0-9.eE+-]+" | tail -n 1 | awk -F= '{print $2}')

# ---------------------------------------------------------------------------
# 汇总（ModelZoo 标准字段，评审工具按名抓取，名字不要改）
# ---------------------------------------------------------------------------
RESULT_FILE=${LOG_DIR}/${CaseName}_result.log
{
    echo "Network       = ${Network}"
    echo "DeviceType    = $(uname -m)"
    echo "CaseName      = ${CaseName}"
    echo "BatchSize     = ${global_batch_size}"
    echo "WorldSize     = ${WORLD_SIZE}"
    echo "MaxSteps      = ${max_steps}"
    echo "ActualFPS     = ${FPS}"
    echo "TrainingTime  = ${TrainingTime}"
    echo "ActualLoss    = ${FinalLoss:-N/A}"
    echo "E2ETrainingTime = ${e2e_time}"
    echo "StepTime      = ${step_time}"
} | tee "${RESULT_FILE}"

echo ""
echo "结果已写入: ${RESULT_FILE}"
echo "训练退出码: ${status}"
exit ${status}
