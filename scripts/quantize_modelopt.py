#!/usr/bin/env python
"""Stage 1 - ModelOpt PTQ of MiniCPM5-2B. One format per invocation.

  quantize_modelopt.py --cfg w4a16_nvfp4   --out ~/models/minicpm5-2b-nvfp4-w4a16
  quantize_modelopt.py --cfg nvfp4_awq_lite --out ~/models/minicpm5-2b-nvfp4-awqlite
  quantize_modelopt.py --cfg int4_awq       --out ~/models/minicpm5-2b-int4-awq

Smoke test first:  --cfg w4a16_nvfp4 --samples 8 --out /tmp/mc_smoke
Run in ~/quant-env.  Outputs an HF checkpoint servable by vLLM (sm_120).
BF16 baseline needs no quant - serve ~/models/MiniCPM5-2B directly.
"""

import argparse
import pathlib

import torch
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.utils.dataset_utils import create_forward_loop, get_dataset_dataloader
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/home/ttimm/models/MiniCPM5-2B"

CFGS = {
    "w4a16_nvfp4": mtq.W4A16_NVFP4_CFG,       # NVFP4 weights, FP16 activations  (the recommended SOTA format)
    "nvfp4_awq_lite": mtq.NVFP4_AWQ_LITE_CFG,  # NVFP4 weights + AWQ scale search (best-of-both)
    "int4_awq": mtq.INT4_AWQ_CFG,             # INT4 AWQ weight-only            (the AWQ-Marlin competitor)
}
CALIB = ["open_code_reasoning", "nemotron-sft-swe-v2"]  # code + SWE, disjoint from eval sets
MAX_LEN = 2048


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, choices=list(CFGS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=512, help="total calib samples (use 8 for a smoke test)")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    half = a.samples // 2
    dl = get_dataset_dataloader(
        dataset_name=CALIB,
        tokenizer=tok,
        num_samples=[half, a.samples - half],
        max_sample_length=MAX_LEN,
        batch_size=1,
        device="cuda",
    )
    forward_loop = create_forward_loop(model=model, dataloader=dl)

    print(f"[quantize] cfg={a.cfg} calib={a.samples} samples", flush=True)
    model = mtq.quantize(model, CFGS[a.cfg], forward_loop)
    mtq.print_quant_summary(model)

    out = pathlib.Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    export_hf_checkpoint(model, export_dir=str(out))
    tok.save_pretrained(str(out))
    print(f"[quantize] exported -> {out}", flush=True)


if __name__ == "__main__":
    main()
