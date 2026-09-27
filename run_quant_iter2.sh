#!/usr/bin/env bash
set -uo pipefail
export VLLM_USE_V2_MODEL_RUNNER=0 HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false
D=/home/ttimm/minicpm-quant; QP=~/quant-env/bin/python; LM=~/vllm-env/bin/lm_eval; VP=~/vllm-env/bin/python
AGG=/home/ttimm/kat-coder-16gb/scripts/eval/aggregate_repeats.py
L=$D/iter2.log; : > "$L"; say(){ echo "[$(date -Iseconds)] $*"|tee -a "$L"; }
declare -A CKPT=( [nvfp4_ar]=~/models/minicpm5-2b-nvfp4a16-ar [mixed]=~/models/minicpm5-2b-mixed-nvfp4-fp8 )
declare -A METH=( [nvfp4_ar]=autoround_nvfp4a16 [mixed]=mixed_nvfp4_fp8 )
MA="dtype=auto,max_model_len=2048,gpu_memory_utilization=0.85,max_num_seqs=8,enforce_eager=True,trust_remote_code=True"
drain(){ local w=0 f; while [ $w -lt 180 ]; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null|head -1); [ -n "$f" ]&&[ "$f" -ge 13500 ]&&return 0; sleep 5; w=$((w+5)); done; }
ARMS=()
for a in nvfp4_ar mixed; do
  ck=${CKPT[$a]}
  if [ ! -f "$ck/config.json" ]; then drain; say "quant $a (${METH[$a]})"
    "$QP" "$D/quantize_llmc.py" --method "${METH[$a]}" --samples 512 --out "$ck" >> "$L" 2>&1; say "  rc=$?"; fi
  drain
  if "$VP" "$D/coherence_check.py" "$ck" > "$D/coh_$a.log" 2>&1; then say "$a coherence PASS"; ARMS+=("$a")
  else say "$a coherence FAIL -> $(grep -aE "FAIL|Error" "$D/coh_$a.log"|head -1)"; fi
done
for arm in "${ARMS[@]}"; do
  for task in humaneval_instruct mbpp; do
    for i in 1 2 3; do
      dir=$D/recov/$arm/$task/rep$i
      [ -n "$(find "$dir" -name results_*.json 2>/dev/null|head -1)" ] && continue
      mkdir -p "$dir"; drain; say "$arm/$task rep$i @ $(date +%H:%M)"
      timeout --signal=KILL 2700 "$LM" run --model vllm --model_args "pretrained=${CKPT[$arm]},$MA" \
        --tasks "$task" --batch_size auto --seed 1234 $([ "$task" = humaneval_instruct ] && echo --apply_chat_template) \
        --confirm_run_unsafe_code --log_samples --output_path "$dir" > "$dir/eval.log" 2>&1
      r=$(find "$dir" -name results_*.json 2>/dev/null|head -1)
      [ -n "$r" ] && say "  -> $(grep -oE "\"pass(@|_at_)1[^\"]*\": *[0-9.]+" "$r"|head -1)" || say "  -> NO RESULT rc=$?"
    done
  done
done
{ echo "== iter2  $(date -Iseconds) =="; for arm in "${ARMS[@]}"; do for t in humaneval_instruct mbpp; do
  echo "--- $arm / $t ---"; du -sh ${CKPT[$arm]}/*.safetensors 2>/dev/null; "$VP" "$AGG" "$D/recov/$arm/$t" 2>&1|tail -8; echo; done; done; } | tee "$D/ITER2_SUMMARY.txt" | tee -a "$L"
say "ITER2_DONE"
