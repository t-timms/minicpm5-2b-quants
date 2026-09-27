#!/usr/bin/env bash
set -uo pipefail
QP=~/quant-env/bin/python
LOG=~/minicpm-quant/quants.log
: > "$LOG"
for pair in "w4a16_nvfp4:minicpm5-2b-nvfp4-w4a16" "nvfp4_awq_lite:minicpm5-2b-nvfp4-awqlite" "int4_awq:minicpm5-2b-int4-awq"; do
  cfg="${pair%%:*}"; out="$HOME/models/${pair##*:}"
  echo "=== $(date -Iseconds) START $cfg -> $out ===" | tee -a "$LOG"
  if [ -f "$out/config.json" ]; then echo "  already done, skip" | tee -a "$LOG"; continue; fi
  "$QP" quantize_modelopt.py --cfg "$cfg" --out "$out" --samples 512 >> "$LOG" 2>&1
  rc=$?
  echo "=== $(date -Iseconds) END $cfg rc=$rc ===" | tee -a "$LOG"
  ls -la "$out" 2>/dev/null | tee -a "$LOG"
done
echo "ALL_QUANTS_DONE $(date -Iseconds)" | tee -a "$LOG"
