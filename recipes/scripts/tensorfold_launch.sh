#!/usr/bin/env bash
# Launch one TensorFold rank for the GLM-5.3-Flash long-context/tuning follow-up round.
# Mirrors the NCCL/mount config of this fleet's real vLLM production containers so the
# comparison stays apples-to-apples (see VALIDATION.md's TensorFold A/B section).
#
# Usage: tensorfold_launch.sh <rank 0|1> <master-ip> [context] [drafter-policy] [prefill-rows]
#   rank 0 = head (serves HTTP on :8080), rank 1 = worker. Start rank 1 first.
#   prefill-rows (optional): patches the installed package's hardcoded prefill_rows=64
#   (engine.py's GlmEngine.__init__ -> Engine(...) call) before serving. Experimental --
#   never tested upstream past the default; verify exactness before trusting speed numbers.
set -euo pipefail

RANK="${1:?rank required (0 or 1)}"
MASTER_IP="${2:?master ip required}"
CONTEXT="${3:-}"
DRAFTER="${4:-auto}"
PREFILL_ROWS="${5:-}"
NAME="tensorfold_rank${RANK}"

docker rm -f "$NAME" >/dev/null 2>&1 || true

EXTRA_ARGS=(--tp 2 --rank "$RANK" --master "$MASTER_IP" --no-thinking --drafter "$DRAFTER")
if [[ "$RANK" == "0" ]]; then
  EXTRA_ARGS+=(--host 0.0.0.0 --port 8080)
fi
if [[ -n "$CONTEXT" ]]; then
  EXTRA_ARGS+=(--context "$CONTEXT")
fi

# Build the in-container command as a plain bash script (no nested-quoting gymnastics).
# The prefill_rows patch step is only appended when PREFILL_ROWS is set.
INNER=/tmp/tensorfold_inner_${RANK}.sh
{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  echo 'pip install -q git+https://github.com/ashhart/TensorFold.git'
  if [[ -n "$PREFILL_ROWS" ]]; then
    echo 'F=$(python3 -c "import tensorfold.families.glm5_next.cuda.engine as e; print(e.__file__)")'
    echo 'grep -q "prefill_rows=64," "$F" || { echo "patch target not found, aborting"; exit 1; }'
    echo "sed -i 's/prefill_rows=64,/prefill_rows=${PREFILL_ROWS},/' \"\$F\""
    echo 'grep -n "prefill_rows=" "$F"'
  fi
  printf 'exec tensorfold serve Vontra/GLM-5.3-Flash-MLX-4bit-MTP'
  printf ' %q' "${EXTRA_ARGS[@]}"
  printf '\n'
} > "$INNER"
chmod +x "$INNER"

docker run -d --name "$NAME" \
  --network host --ipc host --shm-size 32g \
  --gpus all --device /dev/infiniband \
  --ulimit memlock=-1:-1 --ulimit nofile=65535:65535 \
  -v /models:/models \
  -v "$INNER":/tmp/tensorfold_inner.sh \
  -e HF_HUB_CACHE=/models \
  -e HF_HUB_OFFLINE=1 \
  -e NCCL_CROSS_NIC=1 \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_SOCKET_IFNAME=enP7s7,enp1s0f0np0,enP2p1s0f0np0 \
  -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 \
  -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_NET=IB \
  -e NCCL_DEBUG=WARN \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 \
  nvcr.io/nvidia/pytorch:26.07-py3 \
  bash /tmp/tensorfold_inner.sh

echo "started $NAME (rank $RANK, context=${CONTEXT:-default}, drafter=$DRAFTER, prefill_rows=${PREFILL_ROWS:-default(64)})"
