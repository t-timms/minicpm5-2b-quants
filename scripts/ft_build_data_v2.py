"""v2 MiniCPM5-2B coding SFT data. SOTA source = nvidia/OpenCodeInstruct
(CC-BY-4.0, built+validated for 1B-7B on HumanEval/MBPP), quality-filtered to
unit-test-passing samples. + evol-codealpaca (diversity) + alpaca-cleaned
(small general rehearsal). All chat-templated, <=2048 tok, longest 10% dropped.
Run in ~/quant-env. Writes ~/minicpm-quant/ft_data_v2/.
"""
from __future__ import annotations
import random
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

MODEL = "/home/ttimm/models/MiniCPM5-2B"
OUT = "/home/ttimm/minicpm-quant/ft_data_v2"
N_OCI = 11000        # OpenCodeInstruct, test-passing
N_EVOL = 2000        # evol-codealpaca diversity
N_REHEARSAL = 1500   # general instruction-following rehearsal
MAX_TOK = 2048
DROP_LONGEST_FRAC = 0.10
SEED = 1234

tok = AutoTokenizer.from_pretrained(MODEL)
random.seed(SEED)


def render(msgs):
    if len(msgs) < 2 or msgs[-1]["role"] != "assistant":
        return None
    if not all(m.get("content") for m in msgs):
        return None
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    except Exception:
        return None
    n = len(tok(text)["input_ids"])
    if n < 24 or n > MAX_TOK:
        return None
    return text, n


def _f(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


# ---- OpenCodeInstruct: stream, keep only unit-test-passing / high-score ----
print("=== OpenCodeInstruct (nvidia, CC-BY-4.0) — streaming, test-passing only ===")
oci_rows, seen, scanned = [], 0, 0
MAX_SCAN = 500000
ds = load_dataset("nvidia/OpenCodeInstruct", split="train", streaming=True)
for r in ds:
    scanned += 1
    if len(oci_rows) >= N_OCI or scanned >= MAX_SCAN:
        break
    # tests_execution_status is a JSON list-string like ["pass","pass",...];
    # average_test_score is a stringified float. Keep only all-pass (score==1)
    # with no explicit "fail" in the status list.
    status = str(r.get("tests_execution_status", "")).lower()
    score = _f(r.get("average_test_score"))
    if score < 0.999 or "fail" in status:
        continue
    inp = (r.get("input") or "").strip()
    out = (r.get("output") or "").strip()
    if not inp or not out or len(out) < 20:
        continue
    got = render([{"role": "user", "content": inp}, {"role": "assistant", "content": out}])
    if got:
        oci_rows.append({"text": got[0], "_ntok": got[1]})
    if scanned % 20000 == 0:
        print(f"  scanned {scanned}, kept {len(oci_rows)}")
print(f"  OpenCodeInstruct: kept {len(oci_rows)} from {scanned} scanned")

# ---- evol-codealpaca: diversity (cached) ----
print("=== evol-codealpaca-v1 (diversity) ===")
evol_rows = []
ds = load_dataset("theblackcat102/evol-codealpaca-v1", split="train", streaming=True)
for r in ds:
    if len(evol_rows) >= N_EVOL:
        break
    inp = (r.get("instruction") or r.get("prompt") or "").strip()
    out = (r.get("output") or r.get("response") or "").strip()
    if not inp or not out or len(out) < 20:
        continue
    got = render([{"role": "user", "content": inp}, {"role": "assistant", "content": out}])
    if got:
        evol_rows.append({"text": got[0], "_ntok": got[1]})
print(f"  evol-codealpaca: kept {len(evol_rows)}")

# ---- alpaca-cleaned: small general rehearsal (cached) ----
print("=== alpaca-cleaned (general rehearsal) ===")
reh_rows = []
ds = load_dataset("yahma/alpaca-cleaned", split="train", streaming=True)
for r in ds:
    if len(reh_rows) >= N_REHEARSAL:
        break
    instr = (r.get("instruction") or "").strip()
    ctx = (r.get("input") or "").strip()
    out = (r.get("output") or "").strip()
    if not instr or not out:
        continue
    user = instr if not ctx else f"{instr}\n\n{ctx}"
    got = render([{"role": "user", "content": user}, {"role": "assistant", "content": out}])
    if got:
        reh_rows.append({"text": got[0], "_ntok": got[1]})
print(f"  alpaca-cleaned: kept {len(reh_rows)}")

# ---- combine, drop longest 10%, shuffle, split ----
allrows = oci_rows + evol_rows + reh_rows
allrows.sort(key=lambda r: r["_ntok"])
keep_n = int(len(allrows) * (1.0 - DROP_LONGEST_FRAC))
dropped = len(allrows) - keep_n
allrows = allrows[:keep_n]
maxtok_kept = allrows[-1]["_ntok"] if allrows else 0
for r in allrows:
    r.pop("_ntok", None)
random.shuffle(allrows)

ds = Dataset.from_list(allrows)
ds = ds.train_test_split(test_size=min(400, len(ds) // 25), seed=SEED)
import os
os.makedirs(OUT, exist_ok=True)
ds["train"].to_parquet(f"{OUT}/train.parquet")
ds["test"].to_parquet(f"{OUT}/eval.parquet")
print(f"\nDONE: {len(ds['train'])} train / {len(ds['test'])} eval")
print(f"  sources: {len(oci_rows)} OpenCodeInstruct + {len(evol_rows)} evol + {len(reh_rows)} rehearsal")
print(f"  dropped longest {dropped} ({DROP_LONGEST_FRAC:.0%}); max kept = {maxtok_kept} tok")
print(f"  -> {OUT}/")
