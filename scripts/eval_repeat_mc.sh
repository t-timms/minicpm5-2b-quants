#!/usr/bin/env bash
# Repeated-draw wrapper around a single lm-eval task.
#
# Why this exists. eval_suite.sh runs each task exactly once, and its header
# claims the scores are "deterministic and re-runnable rather than sampled
# estimates" because decoding is greedy (do_sample: false, repeats: 1, seed 1234).
# That claim is false, and we have the counterexample:
#
#   2026-09-05, /home/ttimm/models/kat-w4a4-published, identical model_args,
#   identical seed, humaneval_instruct:
#       fulleval 04:44  ->  96.34%   (158/164)
#       repeat 3 05:03  ->  92.07%   (151/164)
#
#   Same checkpoint, same flags, same seed. A 4.27 pp / 7-problem spread.
#
# The randomness is not in token selection -- greedy is greedy. It is in the
# execution path: vLLM's continuous batching composes batches differently run to
# run, and FP4 kernel reductions are not order-invariant, so logits differ in the
# last bits and greedy argmax occasionally tips to a different token. A seed
# cannot remove this because the seed does not control batch composition.
#
# Consequence: a single draw is not a score, and any A/B built on single draws
# (this build vs that build, ours vs theirs) cannot separate a real effect from
# this spread. Published numbers need the spread reported alongside them.
#
# This script runs one task R times into rep1..repR under OUT, draining VRAM
# between runs, and leaves the arithmetic to aggregate_repeats.py.
#
# The lm_eval invocation below is a byte-for-byte copy of the one in
# eval_suite.sh. Keep it that way. If the two drift, repeat runs stop being
# comparable to the single-draw numbers already published on the model cards,
# which is the whole point of running them.
#
# Usage:
#   MODEL=/home/ttimm/models/kat-w4a4-published \
#   TASK=humaneval_instruct \
#   REPEATS=5 \
#   OUT=/home/ttimm/abwork/repeat-he \
#   ./eval_repeat.sh
#
# HumanEval executes model-generated code. Both gates are required: the
# --confirm_run_unsafe_code flag AND HF_ALLOW_CODE_EVAL=1.

set -uo pipefail

export HF_ALLOW_CODE_EVAL=1

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LM="${HOME}/vllm-env/bin/lm_eval"
MODEL="${MODEL:?set MODEL to the checkpoint path}"
TASK="${TASK:-humaneval_instruct}"
REPEATS="${REPEATS:-5}"
OUT="${OUT:-${HOME}/abwork/repeat-${TASK}}"
TASKS_DIR="${REPO}/tasks"
LIMIT="${LIMIT:-}"

# Free-VRAM floor before a run is allowed to start, in MiB. The model is ~12.5
# GiB and vLLM asks for gpu_memory_utilization=0.92 of a 15.89 GiB card
# (14.62 GiB), so a previous engine that has not finished releasing memory makes
# the next run die at init with either
#   "Free memory on device cuda:0 ... is less than desired GPU memory utilization"
# or "No available memory for the cache blocks".
# Both were observed on 2026-09-05; that is why repeat 2 produced NO RESULT.
DRAIN_FLOOR_MIB="${DRAIN_FLOOR_MIB:-15000}"
DRAIN_TIMEOUT_S="${DRAIN_TIMEOUT_S:-180}"

mkdir -p "${OUT}"

# Same install step eval_suite.sh performs: humaneval_plus_instruct composes
# lm-eval's instruct framing with EvalPlus's tests, and its `include:` /
# `!function` references only resolve from inside lm-eval's own humaneval task
# directory. --include_path is not sufficient.
HE_DIR="$("${HOME}/vllm-env/bin/python" -c 'import lm_eval, os; print(os.path.join(os.path.dirname(lm_eval.__file__), "tasks", "humaneval"))')"
cp "${TASKS_DIR}/humaneval_plus_instruct.yaml" "${HE_DIR}/humaneval_plus_instruct.yaml"

# Wait for the GPU to actually give the memory back. nvidia-smi reports total
# used across every process including the Windows desktop compositor, so this
# waits on free memory rather than assuming our own process is the only tenant.
drain_gpu() {
  local waited=0 free
  while [ "${waited}" -lt "${DRAIN_TIMEOUT_S}" ]; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -z "${free}" ] && { echo "    drain: nvidia-smi unavailable, proceeding blind"; return 0; }
    if [ "${free}" -ge "${DRAIN_FLOOR_MIB}" ]; then
      echo "    drain: ${free} MiB free (floor ${DRAIN_FLOOR_MIB}), ok after ${waited}s"
      return 0
    fi
    sleep 5
    waited=$(( waited + 5 ))
  done
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  echo "    drain: TIMEOUT after ${DRAIN_TIMEOUT_S}s, only ${free} MiB free (floor ${DRAIN_FLOOR_MIB})"
  echo "    drain: starting anyway -- the run will fail at init and be recorded as a miss,"
  echo "           which is the honest outcome. Do not lower the floor to make it pass."
  return 1
}

echo "=== repeated-draw eval, $(date -Iseconds) ==="
echo "    model:   ${MODEL}"
echo "    task:    ${TASK}"
echo "    repeats: ${REPEATS}"
echo "    out:     ${OUT}"
[ -n "${LIMIT}" ] && echo "    LIMIT=${LIMIT} (pilot mode, NOT a real score)"

for i in $(seq 1 "${REPEATS}"); do
  dir="${OUT}/rep${i}"

  if [ -n "$(find "${dir}" -name 'results_*.json' 2>/dev/null | head -1)" ]; then
    echo
    echo "--- rep${i}: skip, results already present"
    continue
  fi

  mkdir -p "${dir}"
  echo
  echo "--- rep${i} @ $(date -Iseconds) ---"
  drain_gpu

  extra=()
  [ -n "${LIMIT}" ] && extra=(--limit "${LIMIT}")

  start=$(date +%s)
  "${LM}" run \
    --model vllm \
    --model_args "pretrained=${MODEL},dtype=auto,max_model_len=2048,gpu_memory_utilization=0.90,max_num_seqs=8,trust_remote_code=True,enforce_eager=True" \
    --tasks "${TASK}" \
    --include_path "${TASKS_DIR}" \
    --batch_size auto \
    "${extra[@]}" \
    --log_samples \
    --output_path "${dir}" \
    --apply_chat_template \
    --confirm_run_unsafe_code \
    --seed 1234 \
    > "${dir}/eval.log" 2>&1
  rc=$?
  elapsed=$(( $(date +%s) - start ))

  res=$(find "${dir}" -name 'results_*.json' 2>/dev/null | head -1)
  if [ -n "${res}" ]; then
    # lm-eval spells the metric two ways depending on the task's filter:
    # humaneval reports "pass@1,create_test", mbpp reports "pass_at_1,extract_code".
    # Matching only the first silently yields an empty MBPP+ score with rc=0.
    score=$(grep -oE '"pass(@|_at_)1,[a-z_]*": *[0-9.]+' "${res}" | head -1 | grep -oE '[0-9.]+$')
    printf '    rep%-3s pass@1 = %s  (%ds, rc=%d)\n' "${i}" "${score}" "${elapsed}" "${rc}"
  else
    printf '    rep%-3s NO RESULT (rc=%d, %ds)\n' "${i}" "${rc}" "${elapsed}"
    grep -aoE "ValueError: [^\"]{0,140}|Error: [^\"]{0,120}" "${dir}/eval.log" \
      | grep -v "Engine core init" | head -2
  fi
done

echo
echo "=== aggregate ==="
"${HOME}/vllm-env/bin/python" "$(dirname "${BASH_SOURCE[0]}")/aggregate_repeats.py" "${OUT}"
agg_rc=$?
echo "=== done $(date -Iseconds) ==="
exit "${agg_rc}"
