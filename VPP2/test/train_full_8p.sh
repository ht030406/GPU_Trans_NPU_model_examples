#!/bin/bash
# ============================================================================
# VPP2 —— 单机 8 卡训练（精度跑）
#
# 用法：
#   bash test/train_full_8p.sh \
#       --dataset_path=/path/to/data/prepared \
#       --media_path=/path/to/data/robodojo \
#       [--max_steps=100000] [--num_npus=8] [--output_dir=...]
#
# 说明：
#   - 精度跑：确定性开关（HCCL_DETERMINISTIC / CLOSE_MATMUL_K_SHIFT）开启，
#     与性能跑（test/train_performance_8p.sh）互斥且必须分开跑。
#   - 配置契约：reference_global_batch_size=288，本脚本固定
#     micro_batch=1 × 8 卡 × grad_accum=36 = 288（global batch 口径，见文档）。
#   - FinalLoss 取日志末行 [train] ... loss= 的值；脚本退出码反映训练是否成功。
# ============================================================================

set -uo pipefail

CUR_DIR=$(cd "$(dirname "$0")" && pwd)
if [ -f "${CUR_DIR}/env_npu.sh" ]; then
    source "${CUR_DIR}/env_npu.sh"
fi

# 精度跑：确定性开关（性能跑保持关闭；二者为两次独立运行）
export HCCL_DETERMINISTIC=true
export CLOSE_MATMUL_K_SHIFT=1

# ---------------------------------------------------------------------------
# 默认参数
# ---------------------------------------------------------------------------
Network="VPP2"
ROOT_DIR=$(pwd)

num_npus=8
micro_batch_size=1
grad_accum=36                 # global_batch = micro × npus × grad_accum = 288
global_batch_size=$((micro_batch_size * num_npus * grad_accum))
max_steps=100000
seed=100008

dataset_path=""
media_path=""
output_dir=${ROOT_DIR}/output

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
if [[ "${max_steps}" =~ ^[1-9][0-9]*$ ]]; then :; else
    echo "[Error] max_steps must be a positive integer (got \"${max_steps}\")."; exit 1
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
echo " max steps      : ${max_steps}"
echo " seed           : ${seed}"
echo " dataset        : ${dataset_path}"
echo " media          : ${media_path}"
echo " log            : ${LOG_FILE}"
echo " deterministic  : HCCL_DETERMINISTIC=true CLOSE_MATMUL_K_SHIFT=1"
echo "=========================================="

start_time=$(date +%s)

NNODES=${nnodes} NPROC_PER_NODE=${num_npus} NODE_RANK=${node_rank} \
MASTER_ADDR=${master_addr} MASTER_PORT=${master_port} \
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$(seq -s, 0 $((num_npus - 1)))}" \
bash "${ROOT_DIR}/scripts/robodojo/train.sh" \
    "paths.data=${dataset_path}" \
    "paths.media=${media_path}" \
    "max_steps=${max_steps}" \
    "batch_size=${micro_batch_size}" \
    "gradient_accumulation_steps=${grad_accum}" \
    "seed=${seed}" \
    "log_every=10" \
    "eval_at_start=false" \
    "output_dir=${output_dir}" \
    2>&1 | tee "${LOG_FILE}"
status=${PIPESTATUS[0]}

end_time=$(date +%s)
e2e_time=$((end_time - start_time))

if [ ! -f "${LOG_FILE}" ]; then
    echo "Log Error: 日志文件 ${LOG_FILE} 未生成"; exit 1
fi

# --- Loss：取末行 [train] ... loss= ---
# NH：项目 logger（rich）会把 "[train]" 标记当作样式标签剥掉（日志里显示为
#   ">>  epoch=.. step=../.. loss=.."），grep "\[train\]" 会匹配不到任何行。
#   这里改按真实行特征匹配：同一物理行上的 "step=<i>/<n> loss="。
TRAIN_LINE_RE='step=[0-9]+/[0-9]+ loss='
FinalLoss=$(grep -aE "${TRAIN_LINE_RE}" "${LOG_FILE}" | grep -aoE "loss=[0-9.eE+-]+" | tail -n 1 | awk -F= '{print $2}')
if [ -z "$FinalLoss" ]; then
    echo "Log Error: 未找到 [train] loss 记录（max_steps 太小或 log_every 未及）"
    exit 1
fi
MeanLoss=$(grep -aE "${TRAIN_LINE_RE}" "${LOG_FILE}" | grep -aoE "loss=[0-9.eE+-]+" \
    | awk -F= '{sum += $2; cnt += 1} END {if (cnt > 0) printf "%.6f", sum / cnt}')

# ---------------------------------------------------------------------------
# 汇总（ModelZoo 标准字段：ActualFPS / TrainingTime / ActualLoss 供评审抓取）
# 精度跑不报 FPS（FPS 属于性能跑口径），这里只报 Loss 与收尾字段。
# ---------------------------------------------------------------------------
RESULT_FILE=${LOG_DIR}/${CaseName}_result.log
{
    echo "Network       = ${Network}"
    echo "DeviceType    = $(uname -m)"
    echo "CaseName      = ${CaseName}"
    echo "BatchSize     = ${global_batch_size}"
    echo "WorldSize     = ${WORLD_SIZE}"
    echo "MaxSteps      = ${max_steps}"
    echo "ActualLoss    = ${FinalLoss}"
    echo "MeanLoss      = ${MeanLoss}"
    echo "E2ETrainingTime = ${e2e_time}"
} | tee "${RESULT_FILE}"

echo ""
echo "结果已写入: ${RESULT_FILE}"
echo "训练退出码: ${status}"
exit ${status}
