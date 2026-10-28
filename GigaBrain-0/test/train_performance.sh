#!/usr/bin/env bash
set -euo pipefail

export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TORCH_HCCL_ZERO_COPY=1  # 910b和950不支持，运行前注释
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# export HCCL_OP_EXPANSION_MODE="CCU_SCHED" # 仅支持950
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_root}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIGA_BRAIN_DIR="${GIGA_BRAIN_DIR:-${SCRIPT_DIR}/../giga-brain-0}"
TRAIN_SCRIPT="${GIGA_BRAIN_DIR}/scripts/train.py"

# 默认训练参数，可通过脚本参数覆盖
DEFAULT_NPROC_PER_NODE=8
DEFAULT_TASK_NAME="gb07_pg2_push_buttons_h01_30k"
GPU_IDS=""   # 默认按 --nproc_per_node 生成 0..N-1
BATCH_SIZE=8
MAX_STEPS=500
LOG_INTERVAL=1

# 先使用默认值，后面根据参数覆盖
NPROC_PER_NODE="${DEFAULT_NPROC_PER_NODE}"
TASK_NAME="${DEFAULT_TASK_NAME}"

# 标记是否已经显式设置过，兼容位置参数写法
NPROC_SET=0
TASK_SET=0

# 统一解析参数：同时兼容位置参数和 --key=value / --key value 格式
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nproc_per_node)
      NPROC_PER_NODE="${2:?Error: --nproc_per_node requires a value}"
      NPROC_SET=1
      shift 2
      ;;

    --nproc_per_node=*)
      NPROC_PER_NODE="${1#*=}"
      NPROC_SET=1
      shift
      ;;

    --nproc-per-node)
      NPROC_PER_NODE="${2:?Error: --nproc-per-node requires a value}"
      NPROC_SET=1
      shift 2
      ;;

    --nproc-per-node=*)
      NPROC_PER_NODE="${1#*=}"
      NPROC_SET=1
      shift
      ;;

    --gpu-ids)
      GPU_IDS="${2:?Error: --gpu-ids requires a value}"
      shift 2
      ;;

    --gpu-ids=*)
      GPU_IDS="${1#*=}"
      shift
      ;;

    --task)
      TASK_NAME="${2:?Error: --task requires a value}"
      TASK_SET=1
      shift 2
      ;;

    --task=*)
      TASK_NAME="${1#*=}"
      TASK_SET=1
      shift
      ;;

    --batch-size)
      BATCH_SIZE="${2:?Error: --batch-size requires a value}"
      shift 2
      ;;

    --batch-size=*)
      BATCH_SIZE="${1#*=}"
      shift
      ;;

    --max-steps)
      MAX_STEPS="${2:?Error: --max-steps requires a value}"
      shift 2
      ;;

    --max-steps=*)
      MAX_STEPS="${1#*=}"
      shift
      ;;

    --log-interval)
      LOG_INTERVAL="${2:?Error: --log-interval requires a value}"
      shift 2
      ;;

    --log-interval=*)
      LOG_INTERVAL="${1#*=}"
      shift
      ;;

    *)
      # 兼容旧写法：bash test/train_full.sh gb07_pg2_push_buttons_h01_30k
      if [[ "${TASK_SET}" -eq 0 && "$1" != *=* && "$1" != --* ]]; then
        TASK_NAME="$1"
        TASK_SET=1
      else
        echo "Error: unknown argument: $1" >&2
        exit 1
      fi
      shift
      ;;
  esac
done

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "Error: ${TRAIN_SCRIPT} does not exist." >&2
  echo "Set GIGA_BRAIN_DIR to the giga-brain-0 repository root." >&2
  exit 1
fi

CONFIG="${GIGA_BRAIN_DIR}/configs/${TASK_NAME}.py"
if [[ ! -f "${CONFIG}" ]]; then
  echo "Error: ${CONFIG} does not exist." >&2
  echo "Available tasks: gb07_pg2_push_buttons_h01_30k / gb07_pg2_pick_and_place_piper_30k" >&2
  exit 1
fi

if [[ -z "${RUN_ID:-}" ]]; then
  RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
fi

export RUN_ID
export GIGA_BRAIN_ROOT="${GIGA_BRAIN_DIR}"
export PYTHONPATH="${GIGA_BRAIN_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# 确定 gpu_ids：
#   1) 显式 --gpu-ids 优先（进程数 = 卡id个数）；
#   2) 否则按 --nproc_per_node 生成 0..N-1。
# config 中 launch.gpu_ids 的长度即实际启动的进程数。
if [[ -z "${GPU_IDS}" ]]; then
  GPU_IDS="$(seq 0 $((NPROC_PER_NODE - 1)) | paste -sd, -)"
fi
NPROC_PER_NODE="$(echo "${GPU_IDS}" | tr ',' '\n' | wc -l)"
GPU_IDS_SED="$(echo "${GPU_IDS}" | tr ',' ' ' | sed 's/ /, /g')"

echo "[launch] gpu_ids=${GPU_IDS} nproc_per_node=${NPROC_PER_NODE} task=${TASK_NAME} run_id=${RUN_ID}"

LOG_DIR="$(pwd)/logs"
mkdir -p "${LOG_DIR}"

LOG="${LOG_DIR}/${TASK_NAME}_${RUN_ID}.log"
echo "[log] ${LOG}"

CONFIG_BAK="${CONFIG}.bak_${RUN_ID}"

# 训练开始时间
start_time=$(date +%s)

# 备份 config
cp "${CONFIG}" "${CONFIG_BAK}"

restore_config() {
  if [[ -f "${CONFIG_BAK}" ]]; then
    mv -f "${CONFIG_BAK}" "${CONFIG}"
  fi
}

cleanup() {
  restore_config
}

# 设置信号捕获，如果训练
#   正常退出（EXIT）
#   用户中断（SIGINT）
#   Kill终止请求（SIGTERM）
#   命令执行失败（ERR）
# 可以自动还原对config文件的修改
trap restore_config EXIT SIGINT SIGTERM ERR
# ##################################################################

# 修改 config 中的运行参数（数据路径等仍需在 config 中按环境编辑）
sed -i -E "s/^([[:space:]]*gpu_ids=).*/\1[${GPU_IDS_SED}],/" "${CONFIG}"
sed -i -E "s/^([[:space:]]*batch_size_per_gpu=).*/\1${BATCH_SIZE},/" "${CONFIG}"
sed -i -E "s/^([[:space:]]*max_steps=).*/\1${MAX_STEPS},/" "${CONFIG}"
sed -i -E "s/^([[:space:]]*log_interval=).*/\1${LOG_INTERVAL},/" "${CONFIG}"
sed -i -E "s/^([[:space:]]*resume=).*/\1False,/" "${CONFIG}"

set +e

(
  cd "${GIGA_BRAIN_DIR}"

  python "${TRAIN_SCRIPT}" \
    --config "configs/${TASK_NAME}.py"
) 2>&1 \
  | tee "${LOG}" \
  | grep --line-buffered -Ei "speed: [0-9.]+|total_loss:|loss=|error|runtimeerror|traceback|oom|out of memory|killed"

TRAIN_STATUS=${PIPESTATUS[0]}

set -e

# 训练结束时间
end_time=$(date +%s)
e2e_time=$((end_time - start_time))

echo "------------------ Final result ------------------"

# 单迭代训练时长：取最后 8 个 speed 的平均倒数 * 全局 batch size
GLOBAL_BATCH=$((BATCH_SIZE * NPROC_PER_NODE))
TrainingTime=$(
  grep -oE "speed: [0-9.]+" "${LOG}" \
    | tail -n 8 \
    | sed "s/speed: //" \
    | awk -v gb="${GLOBAL_BATCH}" '{sum += 1 / $1} END {if (NR > 0) printf "%.6f", sum / NR * gb; else print "NA"}' \
    || true
)

if [[ -z "${TrainingTime}" ]]; then
  TrainingTime="NA"
fi

# 吞吐量：取最后 3 个 speed 的平均值（speed 即为全局 samples/s）
ActualSPS=$(
  grep -oE "speed: [0-9.]+" "${LOG}" \
    | tail -n 3 \
    | sed "s/speed: //" \
    | awk '{sum += $1} END {if (NR > 0) printf "%.6f", sum / NR; else print "NA"}' \
    || true
)

if [[ -z "${ActualSPS}" ]]; then
  ActualSPS="NA"
fi

# loss：取最后 3 个 total_loss 的平均值
ActualLoss=$(
  grep -oE "total_loss: [0-9.eE+-]+" "${LOG}" \
    | sed "s/total_loss: //" \
    | tail -n 3 \
    | awk '{sum += $1} END {if (NR > 0) printf "%.6f", sum / NR; else print "NA"}' \
    || true
)

if [[ -z "${ActualLoss}" ]]; then
  ActualLoss="NA"
fi

echo "Training Time per step : ${TrainingTime} s"
echo "E2E Training Time : ${e2e_time} s"
echo "Final Performance images/sec : ${ActualSPS}"
echo "Final Train Loss : ${ActualLoss}"

exit "${TRAIN_STATUS}"
