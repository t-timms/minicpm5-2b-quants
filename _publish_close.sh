#!/usr/bin/env bash
set -uo pipefail
D=/home/ttimm/minicpm-quant
SP=/mnt/c/Users/ttimm/AppData/Local/Temp/claude/C--Users-ttimm/d1c4304a-c66f-4834-9e0a-47dc1bde6c15/scratchpad
cd "$D"

echo "############ 1. GitHub repo ############"
if git remote get-url origin >/dev/null 2>&1; then
  echo "origin already set: $(git remote get-url origin)"; git push -u origin main
else
  gh repo create t-timms/minicpm5-2b-quants --public --source=. --remote=origin \
    --description="First FP8 + NVFP4 quants of openbmb/MiniCPM5-2B: measured quant-quality comparison + sm_120 serving notes" \
    --push
fi
echo "repo: $(gh repo view t-timms/minicpm5-2b-quants --json url -q .url 2>/dev/null)"

echo
echo "############ 2. HF card: FP8 ############"
HF="/home/ttimm/quant-env/bin/hf"
[ -x "$HF" ] || HF="/home/ttimm/vllm-env/bin/hf"
cp "$SP/card_fp8.md" /tmp/README_fp8.md
"$HF" upload Ttimms/MiniCPM5-2B-FP8 /tmp/README_fp8.md README.md --commit-message "docs: full quant comparison table + serving notes + writeup link"

echo
echo "############ 3. HF card: NVFP4 ############"
cp "$SP/card_nvfp4.md" /tmp/README_nvfp4.md
"$HF" upload Ttimms/MiniCPM5-2B-NVFP4 /tmp/README_nvfp4.md README.md --commit-message "docs: full quant comparison table + serving notes + writeup link"

echo
echo "############ DONE ############"
