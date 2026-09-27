#!/usr/bin/env bash
# Spark-X2.5-4B full eval: BF16 vs NVFP4-W4A16 {RTN, GPTQ} x {humaneval_instruct, mbpp} x 3 draws.
# Resumable - reuses existing spark_eval/ HumanEval results for bf16 + nvfp4_rtn.
set -uo pipefail
export VLLM_USE_V2_MODEL_RUNNER=0 HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false
D=/home/ttimm/minicpm-quant
LM=/home/ttimm/vllm-env/bin/lm_eval
VP=/home/ttimm/vllm-env/bin/python
AGG=/home/ttimm/kat-coder-16gb/scripts/eval/aggregate_repeats.py
ED=$D/spark_eval
L=$D/spark_eval2.log
say(){ echo "[$(date -Iseconds)] $*" | tee -a "$L"; }
drain(){ local w=0 f; while [ $w -lt 240 ]; do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null|head -1)
  [ -n "$f" ]&&[ "$f" -ge 13000 ]&&return 0; sleep 5; w=$((w+5)); done; say "  drain timeout ${f:-?}"; }

declare -A ARM=(
  [bf16]=/home/ttimm/models/Spark-X2.5-4B
  [nvfp4_rtn]=/home/ttimm/models/Spark-X2.5-4B-NVFP4-rtn
  [nvfp4_gptq]=/home/ttimm/models/Spark-X2.5-4B-NVFP4-gptq
)
MA="dtype=auto,max_model_len=8192,gpu_memory_utilization=0.88,enforce_eager=True,trust_remote_code=True"

say "=== spark_eval2 start ==="
for arm in bf16 nvfp4_rtn nvfp4_gptq; do
  ck="${ARM[$arm]}"; [ -f "$ck/config.json" ] || { say "  $arm MISSING $ck - skip"; continue; }
  for task in humaneval_instruct mbpp; do
    for i in 1 2 3; do
      dir=$ED/$arm/$task/rep$i
      [ -n "$(find "$dir" -name 'results_*.json' 2>/dev/null|head -1)" ] && { say "  $arm/$task rep$i done"; continue; }
      mkdir -p "$dir"; drain; say "  $arm/$task rep$i @ $(date +%H:%M)"
      timeout --signal=KILL 5400 "$LM" run --model vllm --model_args "pretrained=$ck,$MA" \
        --tasks "$task" --batch_size auto --seed 1234 \
        $([ "$task" != mbpp ] && echo --apply_chat_template) \
        --gen_kwargs max_gen_toks=2048 --confirm_run_unsafe_code --log_samples --output_path "$dir" > "$dir/eval.log" 2>&1
      r=$(find "$dir" -name 'results_*.json' 2>/dev/null|head -1)
      [ -n "$r" ] && say "    -> $(grep -oE '"(pass(@|_at_)1)[^"]*": *[0-9.]+' "$r"|head -1)" || say "    -> NO RESULT rc=$?"
    done
  done
done

say "STAGE summary"
{
  echo "===== Spark-X2.5-4B  BF16 vs NVFP4-W4A16 {RTN, GPTQ}  $(date -Iseconds) ====="
  for arm in bf16 nvfp4_rtn nvfp4_gptq; do
    for task in humaneval_instruct mbpp; do
      d=$ED/$arm/$task; [ -d "$d" ] || continue
      echo "--- $arm / $task ---"; "$VP" "$AGG" "$d" 2>&1 | tail -8; echo
    done
  done
} | tee "$D/SPARK_EVAL2_SUMMARY.txt" | tee -a "$L"
say "SPARK_EVAL2_DONE"
