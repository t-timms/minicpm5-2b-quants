#!/usr/bin/env bash
# v2 MiniCPM5-2B coding LoRA pipeline. RESUMABLE. Preserves all v1 artifacts.
#   data (skip if present) -> LoRA train (resume) -> merge -> quant FP8
#   -> dual-eval {humaneval_instruct, mbpp, ifeval} base-vs-tuned -> summary + gate
set -uo pipefail
export VLLM_USE_V2_MODEL_RUNNER=0 HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false
D=/home/ttimm/minicpm-quant
QP=/home/ttimm/quant-env/bin/python
LM=/home/ttimm/vllm-env/bin/lm_eval
VP=/home/ttimm/vllm-env/bin/python
AGG=/home/ttimm/kat-coder-16gb/scripts/eval/aggregate_repeats.py
L=$D/finetune_v2.log
say(){ echo "[$(date -Iseconds)] $*" | tee -a "$L"; }
drain(){ local w=0 f; while [ $w -lt 240 ]; do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null|head -1)
  [ -n "$f" ]&&[ "$f" -ge 13500 ]&&return 0; sleep 5; w=$((w+5)); done; say "  drain timeout ${f:-?}"; }

ADAPTER=/home/ttimm/models/minicpm5-2b-coding2-lora
MERGED=/home/ttimm/models/minicpm5-2b-coding2-merged
MERGED_FP8=/home/ttimm/models/minicpm5-2b-coding2-fp8
BASE=/home/ttimm/models/MiniCPM5-2B
ED=$D/ft_eval_v2
MA="dtype=auto,max_model_len=2048,gpu_memory_utilization=0.85,max_num_seqs=8,enforce_eager=True,trust_remote_code=True"

say "=== run_finetune_v2 start (resumable) ==="

# 1. data
if [ ! -f "$D/ft_data_v2/train.parquet" ]; then
  say "STAGE build data (OpenCodeInstruct test-passing + evol + alpaca-cleaned)"
  "$QP" "$D/ft_build_data_v2.py" >> "$L" 2>&1
  say "data rc=$?"
fi
[ -f "$D/ft_data_v2/train.parquet" ] || { say "FATAL no ft_data_v2"; exit 1; }
say "data ok: $(du -sh "$D/ft_data_v2" 2>/dev/null | cut -f1)"

# 2. train (+resume) + merge
if [ ! -f "$MERGED/config.json" ]; then
  drain
  say "STAGE train v2 (LR2e-5, r32/a32, 1ep, maxlen2048, eager, resume=auto)"
  FT_ADAPTER_OUT="$ADAPTER" FT_MERGED_OUT="$MERGED" "$QP" "$D/ft_train_v2.py" >> "$L" 2>&1
  say "train+merge rc=$?"
fi
[ -f "$MERGED/config.json" ] || { say "no merged model yet (VM restart mid-train?) - re-run this script"; exit 2; }
say "merged: $(du -sh "$MERGED"/*.safetensors 2>/dev/null | cut -f1)"

# 3. quant merged -> FP8
if [ ! -f "$MERGED_FP8/config.json" ]; then
  drain; say "STAGE quant merged -> FP8"
  "$QP" "$D/quantize_llmc.py" --method fp8 --samples 8 --model "$MERGED" --out "$MERGED_FP8" >> "$L" 2>&1
  say "fp8 rc=$?"
fi

# 4. dual-eval
say "STAGE dual-eval"
declare -A ARM=( [base]="$BASE" [tuned_bf16]="$MERGED" [tuned_fp8]="$MERGED_FP8" )
for arm in base tuned_bf16 tuned_fp8; do
  ck="${ARM[$arm]}"; [ -f "$ck/config.json" ] || { say "  $arm MISSING - skip"; continue; }
  for task in humaneval_instruct mbpp ifeval; do
    reps=3; [ "$task" = ifeval ] && reps=1
    for i in $(seq 1 $reps); do
      dir=$ED/$arm/$task/rep$i
      [ -n "$(find "$dir" -name 'results_*.json' 2>/dev/null|head -1)" ] && { say "  $arm/$task rep$i done"; continue; }
      mkdir -p "$dir"; drain; say "  $arm/$task rep$i @ $(date +%H:%M)"
      timeout --signal=KILL 3000 "$LM" run --model vllm --model_args "pretrained=$ck,$MA" \
        --tasks "$task" --batch_size auto --seed 1234 \
        $([ "$task" != mbpp ] && echo --apply_chat_template) \
        --confirm_run_unsafe_code --log_samples --output_path "$dir" > "$dir/eval.log" 2>&1
      r=$(find "$dir" -name 'results_*.json' 2>/dev/null|head -1)
      [ -n "$r" ] && say "    -> $(grep -oE '"(pass(@|_at_)1|prompt_level_strict_acc|inst_level_strict_acc)[^"]*": *[0-9.]+' "$r"|head -2|tr '\n' ' ')" || say "    -> NO RESULT rc=$?"
    done
  done
done

# 5. summary + gate
say "STAGE summary"
{
  echo "===== MiniCPM5-2B coding LoRA v2 - dual eval  $(date -Iseconds) ====="
  echo "base=MiniCPM5-2B  tuned=+coding2 LoRA (OpenCodeInstruct test-passing 11k + evol 2k + alpaca 1.5k, LR2e-5 r32/a32 1ep)"
  echo "SHIP GATE: coding (HE/MBPP) UP vs base  AND  ifeval not down >2pp"
  echo "v1 ref (regressed): tuned HE 63.4 / MBPP 48.2 ; base HE ~86 / MBPP ~50.6"
  echo
  for arm in base tuned_bf16 tuned_fp8; do
    for task in humaneval_instruct mbpp; do
      d=$ED/$arm/$task; [ -d "$d" ] || continue
      echo "--- $arm / $task ---"; "$VP" "$AGG" "$d" 2>&1 | tail -8; echo
    done
    for j in $ED/$arm/ifeval/rep1/**/results_*.json; do
      [ -f "$j" ] && echo "--- $arm / ifeval ---" && grep -oE '"(prompt_level_strict_acc|inst_level_strict_acc)[^"]*": *[0-9.]+' "$j" && echo
    done
  done
} | tee "$D/FINETUNE_V2_SUMMARY.txt" | tee -a "$L"
say "FINETUNE_V2_DONE"
