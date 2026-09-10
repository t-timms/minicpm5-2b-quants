#!/usr/bin/env python
"""MiniCPM5-2B recovery quant via llm-compressor. One method per invocation.

  quantize_llmc.py --method gptq_nvfp4a16 --out ~/models/minicpm5-2b-nvfp4a16-gptq
  quantize_llmc.py --method autoround_nvfp4a16 --out ~/models/minicpm5-2b-nvfp4a16-ar
  quantize_llmc.py --method fp8              --out ~/models/minicpm5-2b-fp8

Smoke:  --method gptq_nvfp4a16 --samples 8 --out /home/ttimm/minicpm-quant/_smoke_llmc
Run in ~/quant-env. Output = HF checkpoint servable by vLLM (compressed-tensors).
BF16 baseline + the ModelOpt NVFP4-RTN build already exist; this adds the recovered
NVFP4 + an FP8 arm.
"""
from __future__ import annotations

import argparse
import pathlib

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier

MODEL = "/home/ttimm/models/MiniCPM5-2B"
MAX_LEN = 2048
IGNORE = ["lm_head", "re:.*embed_tokens"]

# code-heavy calibration, cached locally (KAT used it), disjoint from HumanEval/MBPP
CALIB_DS = "theblackcat102/evol-codealpaca-v1"


def build_calib(tok, n):
    ds = load_dataset(CALIB_DS, split=f"train[:{n * 3}]")
    from datasets import Dataset

    rows = []
    for r in ds:
        instr = r.get("instruction") or ""
        out = r.get("output") or ""
        t = (instr + "\n\n" + out).strip()
        if len(t) < 40:
            continue
        rows.append({"text": t})
        if len(rows) >= n:
            break
    d = Dataset.from_list(rows)
    return d.map(lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN),
                 remove_columns=d.column_names)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["gptq_nvfp4a16", "autoround_nvfp4a16", "awq_gptq_nvfp4a16", "fp8", "mixed_nvfp4_fp8"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda")

    if a.method == "fp8":
        recipe = [QuantizationModifier(scheme="FP8_DYNAMIC", targets="Linear", ignore=IGNORE)]
        ds = None  # FP8 dynamic needs no calibration
    else:
        ds = build_calib(tok, a.samples)
        if a.method == "gptq_nvfp4a16":
            recipe = [GPTQModifier(scheme="NVFP4A16", targets="Linear", ignore=IGNORE,
                                   dampening_frac=0.1)]
        elif a.method == "autoround_nvfp4a16":
            recipe = [AutoRoundModifier(scheme="NVFP4A16", targets="Linear", ignore=IGNORE)]
        elif a.method == "mixed_nvfp4_fp8":
            recipe = [
                GPTQModifier(scheme="NVFP4A16", targets=["re:.*mlp.*"], ignore=IGNORE, dampening_frac=0.1),
                QuantizationModifier(scheme="FP8_DYNAMIC", targets=["re:.*self_attn.*"], ignore=IGNORE),
            ]
        elif a.method == "awq_gptq_nvfp4a16":
            from llmcompressor.modifiers.awq import AWQModifier
            recipe = [
                AWQModifier(scheme="NVFP4A16", targets="Linear", ignore=IGNORE),
                GPTQModifier(scheme="NVFP4A16", targets="Linear", ignore=IGNORE, dampening_frac=0.1),
            ]

    out = str(pathlib.Path(a.out).expanduser())
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_LEN,
        num_calibration_samples=a.samples if ds is not None else 1,
        output_dir=out,
    )
    tok.save_pretrained(out)
    print(f"[llmc] {a.method} exported -> {out}", flush=True)


if __name__ == "__main__":
    main()
