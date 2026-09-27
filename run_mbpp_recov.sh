#!/usr/bin/env bash
set -uo pipefail
export VLLM_USE_V2_MODEL_RUNNER=0 HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false
D=/home/ttimm/minicpm-quant
LM=/home/ttimm/vllm-env/bin/lm_eval
VP=/home/ttimm/vllm-env/bin/python
AGG=/home/ttimm/kat-coder-16gb/scripts/eval/aggregate_repeats.py
L=$D/mbpp_recov.log; : > "$L"
say(){ echo "[$(date -Iseconds)] $*" | tee -a "$L"; }
declare -A CKPT=(
  [bf16]=/home/ttimm/models/MiniCPM5-2B
  [nvfp4_rtn]=/home/ttimm/models/minicpm5-2b-nvfp4-w4a16
  [nvfp4_gptq]=/home/ttimm/models/minicpm5-2b-nvfp4a16-gptq
  [fp8]=/home/ttimm/models/minicpm5-2b-fp8 )
MA="dtype=auto,max_model_len=2048,gpu_memory_utilization=0.85,max_num_seqs=8,enforce_eager=True,trust_remote_code=True"
drain(){ local w=0 f; while [ $w -lt 180 ]; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null|head -1); [ -n "$f" ]&&[ "$f" -ge 13500 ]&&return 0; sleep 5; w=$((w+5)); done; }
for arm in bf16 nvfp4_rtn nvfp4_gptq fp8; do
  ck=${CKPT[$arm]}; [ -f "$ck/config.json" ] || { say "$arm MISSING"; continue; }
  for i in 1 2 3; do
    dir=$D/recov/$arm/mbpp/rep$i
    [ -n "$(find "$dir" -name results_*.json 2>/dev/null|head -1)" ] && { say "$arm rep$i done"; continue; }
    mkdir -p "$dir"; drain; say "$arm/mbpp rep$i @ $(date +%H:%M)"
    timeout --signal=KILL 2700 "$LM" run --model vllm --model_args "pretrained=$ck,$MA" \
      --tasks mbpp --batch_size auto --seed 1234 --confirm_run_unsafe_code \
      --log_samples --output_path "$dir" > "$dir/eval.log" 2>&1
    r=$(find "$dir" -name results_*.json 2>/dev/null|head -1)
    [ -n "$r" ] && say "  -> $(grep -oE "\"pass(@|_at_)1[^\"]*\": *[0-9.]+" "$r"|head -1)" || { say "  -> NO RESULT rc=$?"; tail -3 "$dir/eval.log"|tee -a "$L"; }
  done
done
say "=== MBPP recovery summary ==="
{ echo "== MiniCPM5-2B MBPP (base, 3-shot) recovery  $(date -Iseconds) =="; \
  for arm in bf16 nvfp4_rtn nvfp4_gptq fp8; do echo "--- $arm ---"; "$VP" "$AGG" "$D/recov/$arm/mbpp" 2>&1|tail -9; echo; done; } \
  | tee "$D/MBPP_RECOVERY_SUMMARY.txt" | tee -a "$L"
say "MBPP_RECOV_DONE"
