#!/usr/bin/env bash
set -euo pipefail
D=/home/ttimm/minicpm-quant
SP=/mnt/c/Users/ttimm/AppData/Local/Temp/claude/C--Users-ttimm/d1c4304a-c66f-4834-9e0a-47dc1bde6c15/scratchpad
cd "$D"

cp "$SP/minicpm_README.md" README.md

mkdir -p scripts results
cp -f quantize_llmc.py quantize_modelopt.py coherence_check.py audit_arch.py \
      eval_repeat_mc.sh analyze_he.py rescore_mbpp.py \
      ft_build_data_v2.py ft_train_v2.py run_finetune_v2.sh scripts/ 2>/dev/null || true
cp -f RECOVERY_SUMMARY.txt MBPP_RECOVERY_SUMMARY.txt MIXED_SUMMARY.txt \
      SUMMARY.txt FINETUNE_SUMMARY.txt FINETUNE_V2_SUMMARY.txt results/ 2>/dev/null || true

cat > .gitignore <<'EOF'
*.log
__pycache__/
ft_data/
ft_data_v2/
ft_eval/
ft_eval_v2/
eval/
recov/
*.bak-*
card_*.md
push_quants.sh
katq_*.json
peek_v2.py
# staged copies live in scripts/ and results/
/quantize_llmc.py
/quantize_modelopt.py
/coherence_check.py
/audit_arch.py
/eval_repeat_mc.sh
/analyze_he.py
/rescore_mbpp.py
/ft_build_data_v2.py
/ft_train_v2.py
/ft_build_data.py
/ft_train.py
/run_finetune.sh
/run_finetune_v2.sh
/run_all.sh
/rerun_mbpp.sh
/eval_mixed.sh
/*.txt
EOF

git init -q
git add README.md .gitignore scripts/ results/
git -c user.name="Tremayne Timms" -c user.email="ttimmsinternational@gmail.com" \
    commit -q -m "MiniCPM5-2B consumer-Blackwell quant study: FP8 vs NVFP4-W4A16 + fine-tune negative result"
git branch -M main
echo "=== staged tree (local commit made; NOT pushed) ==="
git ls-files
echo
echo "=== commit ==="
git log --oneline -1
echo
echo "Next (manual): gh repo create t-timms/minicpm5-2b-quants --public --source=. --remote=origin --push"
