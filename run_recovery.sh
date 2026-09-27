#!/usr/bin/env bash
# MiniCPM5-2B recovery run: quant GPTQ-NVFP4A16 + FP8, then eval
# {bf16, nvfp4_rtn(ModelOpt), nvfp4_gptq(llmc), fp8} x humaneval_instruct x 3 draws.
# Decision gate: does GPTQ land within ~1-2pp of BF16?
set -uo pipefail
export VLLM_USE_V2_MODEL_RUNNER=0 HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false
D=/home/ttimm/minicpm-quant
QP=/home/ttimm/quant-env/bin/python
LM=/home/ttimm/vllm-env/bin/lm_eval
VP=/home/ttimm/vllm-env/bin/python
AGG=/home/ttimm/kat-coder-16gb/scripts/eval/aggregate_repeats.py
PLOG=$D/recovery.log; : > "$PLOG"
say(){ echo "[$(date -Iseconds)] $*" | tee -a "$PLOG"; }

declare -A CKPT=(
  [bf16]=/home/ttimm/models/MiniCPM5-2B
  [nvfp4_rtn]=/home/ttimm/models/minicpm5-2b-nvfp4-w4a16
  [nvfp4_gptq]=/home/ttimm/models/minicpm5-2b-nvfp4a16-gptq
  [fp8]=/home/ttimm/models/minicpm5-2b-fp8
)
MA="dtype=auto,max_model_len=2048,gpu_memory_utilization=0.85,max_num_seqs=8,enforce_eager=True,trust_remote_code=True"
drain(){ local w=0 f; while [ $w -lt 180 ]; do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null|head -1)
  [ -n "$f" ]&&[ "$f" -ge 13500 ]&&return 0; sleep 5; w=$((w+5)); done
  say "  drain timeout ${f:-?} MiB"; }

# ---- quant ----
say "STAGE quant"
if [ ! -f "${CKPT[nvfp4_gptq]}/config.json" ]; then
  drain; say "  GPTQ-NVFP4A16 (512 calib)"
  "$QP" "$D/quantize_llmc.py" --method gptq_nvfp4a16 --samples 512 --out "${CKPT[nvfp4_gptq]}" >> "$PLOG" 2>&1
  say "  GPTQ rc=$?"
else say "  GPTQ already done"; fi
if [ ! -f "${CKPT[fp8]}/config.json" ]; then
  drain; say "  FP8-dynamic"
  "$QP" "$D/quantize_llmc.py" --method fp8 --samples 8 --out "${CKPT[fp8]}" >> "$PLOG" 2>&1
  say "  FP8 rc=$?"
else say "  FP8 already done"; fi

# ---- coherence (only the 2 new) ----
say "STAGE coherence"
ARMS=(bf16 nvfp4_rtn)   # both proven in run 1
for name in nvfp4_gptq fp8; do
  ck=${CKPT[$name]}
  [ -f "$ck/config.json" ] || { say "  $name MISSING - skip"; continue; }
  drain
  if "$VP" "$D/coherence_check.py" "$ck" > "$D/coh_$name.log" 2>&1; then
    say "  $name PASS"; ARMS+=("$name")
  else say "  $name FAIL rc=$? -> $(grep -aE 'FAIL|DEGENERATE|Error' "$D/coh_$name.log"|head -1)"; fi
done
say "arms: ${ARMS[*]}"

# ---- eval: humaneval_instruct x 3 ----
say "STAGE eval (humaneval_instruct x3)"
for arm in "${ARMS[@]}"; do
  ck=${CKPT[$arm]}
  for i in 1 2 3; do
    dir=$D/recov/$arm/humaneval_instruct/rep$i
    [ -n "$(find "$dir" -name results_*.json 2>/dev/null|head -1)" ] && { say "  $arm rep$i done"; continue; }
    mkdir -p "$dir"; drain; say "  $arm rep$i @ $(date +%H:%M)"
    timeout --signal=KILL 2400 "$LM" run --model vllm --model_args "pretrained=$ck,$MA" \
      --tasks humaneval_instruct --batch_size auto --seed 1234 --apply_chat_template \
      --confirm_run_unsafe_code --log_samples --output_path "$dir" > "$dir/eval.log" 2>&1
    r=$(find "$dir" -name results_*.json 2>/dev/null|head -1)
    [ -n "$r" ] && say "    -> $(grep -oE '"pass(@|_at_)1[^"]*": *[0-9.]+' "$r"|head -1)" \
      || { say "    -> NO RESULT rc=$?"; grep -aoE 'Error:[^"]{0,140}|RuntimeError:[^"]{0,140}' "$dir/eval.log"|head -2|tee -a "$PLOG"; }
  done
done

# ---- summary ----
say "STAGE summary"
{
  echo "===== MiniCPM5-2B RECOVERY  $(date -Iseconds) ====="
  echo "arms: ${ARMS[*]}  |  task: humaneval_instruct  |  3 draws greedy"
  echo "nvfp4_rtn = ModelOpt baseline NVFP4A16 | nvfp4_gptq = llm-compressor GPTQ-NVFP4A16 (512 code calib)"
  echo
  for arm in "${ARMS[@]}"; do
    echo "--- $arm ---"; "$VP" "$AGG" "$D/recov/$arm/humaneval_instruct" 2>&1 | tail -10; echo
  done
  echo "RUN1 reference: bf16 HE 85.98% median / nvfp4_rtn 81.10%"
} | tee "$D/RECOVERY_SUMMARY.txt" | tee -a "$PLOG"
say "RECOVERY_DONE"
