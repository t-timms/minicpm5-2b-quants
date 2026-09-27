#!/usr/bin/env bash
# MiniCPM5-2B quant A/B: BF16 vs NVFP4-W4A16 vs NVFP4-AWQ-Lite vs INT4-AWQ.
# Mirrors ~/kat_ab/run_kat_ab_eval_eager.sh. Repeated draws (nondeterministic
# harness, see eval_repeat.sh header). enforce_eager for the coherence-safe path;
# swap to PIECEWISE later once a build is proven correct.
set -uo pipefail

REPEAT="/home/ttimm/kat-coder-16gb/scripts/eval/eval_repeat.sh"
REPEAT_MC="/home/ttimm/minicpm-quant/eval_repeat_mc.sh"
AGG="/home/ttimm/kat-coder-16gb/scripts/eval/aggregate_repeats.py"
PY="/home/ttimm/vllm-env/bin/python"
OUTROOT="/home/ttimm/minicpm-quant/eval"
TASKS=(humaneval_plus_instruct mbpp_plus_instruct)
REPEATS="${REPEATS:-5}"
PER_RUN_TIMEOUT="${PER_RUN_TIMEOUT:-5400}"   # 90 min ceiling per eval_repeat.sh call

declare -A MODELS=(
  [bf16]=/home/ttimm/models/MiniCPM5-2B
  [nvfp4_w4a16]=/home/ttimm/models/minicpm5-2b-nvfp4-w4a16
  [nvfp4_awqlite]=/home/ttimm/models/minicpm5-2b-nvfp4-awqlite
  [int4_awq]=/home/ttimm/models/minicpm5-2b-int4-awq
)
ARMS="${ARMS:-bf16 nvfp4_w4a16 nvfp4_awqlite int4_awq}"

# MiniCPM is vanilla llama with NO vision tower -> drop language_model_only.
# dtype=auto so vLLM reads the quant config from the quantized checkpoints;
# bf16 source falls back to its own dtype. enforce_eager for correctness on sm_120.
if [ ! -f "$REPEAT_MC" ] || [ "$REPEAT" -nt "$REPEAT_MC" ]; then
  sed -e 's#pretrained=${MODEL},dtype=bfloat16,max_model_len=2048,gpu_memory_utilization=0.92,max_num_seqs=4,language_model_only=True,trust_remote_code=True#pretrained=${MODEL},dtype=auto,max_model_len=2048,gpu_memory_utilization=0.90,max_num_seqs=8,trust_remote_code=True,enforce_eager=True#' \
      "$REPEAT" > "$REPEAT_MC"
  chmod +x "$REPEAT_MC"
  echo "built $REPEAT_MC"
  grep -c "enforce_eager=True" "$REPEAT_MC" || { echo "PATCH FAILED - check the sed anchor against $REPEAT"; exit 1; }
fi

mkdir -p "$OUTROOT"
echo "=== MiniCPM5-2B quant A/B start $(date -Iseconds) ==="
echo "    arms: $ARMS   tasks: ${TASKS[*]}   repeats: $REPEATS"

for arm in $ARMS; do
  m="${MODELS[$arm]}"
  if [ ! -e "$m" ]; then echo "  skip $arm - $m missing"; continue; fi
  for task in "${TASKS[@]}"; do
    out="$OUTROOT/$arm/$task"
    echo; echo "########## $(date -Iseconds)  arm=$arm task=$task ##########"
    if timeout --signal=KILL "$PER_RUN_TIMEOUT" \
         env MODEL="$m" TASK="$task" REPEATS="$REPEATS" OUT="$out" bash "$REPEAT_MC"; then
      echo "  ok: $arm/$task"
    else
      rc=$?
      echo "  !! $arm/$task exited $rc ($([ $rc -eq 137 ] && echo TIMEOUT-KILLED || echo error)) - continuing"
      pkill -9 -f "lm_eval run --model vllm" 2>/dev/null
      sleep 20
    fi
  done
done

echo; echo "=== AGGREGATE $(date -Iseconds) ==="
for arm in $ARMS; do
  for task in "${TASKS[@]}"; do
    echo "--- $arm / $task ---"
    "$PY" "$AGG" "$OUTROOT/$arm/$task" 2>&1 | tail -8
  done
done
echo; echo "MINICPM_AB_COMPLETE $(date -Iseconds)"
