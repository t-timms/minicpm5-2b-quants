#!/usr/bin/env python
"""Spark-X2.5-4B quantization via llm-compressor. One method per invocation.
Run in ~/quant-env. Output = HF checkpoint (compressed-tensors), served on vLLM
via the out-of-tree Spark2_5 plugin.

  quantize_spark.py --method nvfp4_rtn  --out ~/models/Spark-X2.5-4B-NVFP4-rtn   --samples 0
  quantize_spark.py --method nvfp4_gptq --out ~/models/Spark-X2.5-4B-NVFP4       --samples 512
  quantize_spark.py --method fp8        --out ~/models/Spark-X2.5-4B-FP8-ours    --samples 0

Ignore list matches XHToken's own INT8 config: lm_head + every self_attn.g_proj
(the head-wise attention output gate; tied embeddings also excluded).
"""
from __future__ import annotations

import argparse
import pathlib

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier

# Spark's vendored modeling_spark.py (transformers 4.57) sets _tied_weights_keys as
# a list; transformers 5.14 (llm-compressor pin) requires a {target: source} dict.
# Coerce it at post_init so HF loading of the custom class succeeds.
import transformers  # noqa: E402

_ORIG_POST_INIT = transformers.PreTrainedModel.post_init


def _post_init_coerce_tied(self):
    twk = getattr(type(self), "_tied_weights_keys", None)
    if isinstance(twk, list):
        # Spark ties lm_head to model.embedding (not embed_tokens)
        type(self)._tied_weights_keys = {k: "model.embedding.weight" for k in twk}
    return _ORIG_POST_INIT(self)


transformers.PreTrainedModel.post_init = _post_init_coerce_tied

MODEL = "/home/ttimm/models/Spark-X2.5-4B"
MAX_LEN = 2048
IGNORE = ["lm_head", "re:.*embed_tokens", "re:.*self_attn\\.g_proj"]
CALIB_DS = "theblackcat102/evol-codealpaca-v1"


def build_calib(tok, n):
    ds = load_dataset(CALIB_DS, split=f"train[:{n * 3}]")
    rows = []
    for r in ds:
        t = ((r.get("instruction") or "") + "\n\n" + (r.get("output") or "")).strip()
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
    ap.add_argument("--method", required=True, choices=["nvfp4_rtn", "nvfp4_gptq", "fp8"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)

    # Spark's vendored modeling calls create_causal_mask(input_embeds=..., cache_position=...);
    # transformers 5.14 renamed the kwarg to `inputs_embeds` and dropped `cache_position`.
    # Wrap the two mask builders inside the dynamically-loaded Spark module (needed only
    # for GPTQ/AWQ, which run calibration forwards; RTN is data-free and never hits this).
    import sys as _sys

    def _wrap_mask(fn):
        def inner(*args, **kw):
            if "input_embeds" in kw:
                kw["inputs_embeds"] = kw.pop("input_embeds")
            kw.pop("cache_position", None)
            return fn(*args, **kw)
        return inner

    for _name, _mod in list(_sys.modules.items()):
        if "modeling_spark" in _name and _mod is not None:
            for _fn in ("create_causal_mask", "create_sliding_window_causal_mask"):
                if hasattr(_mod, _fn):
                    setattr(_mod, _fn, _wrap_mask(getattr(_mod, _fn)))

    if a.method == "fp8":
        recipe = [QuantizationModifier(scheme="FP8_DYNAMIC", targets="Linear", ignore=IGNORE)]
        ds = None
    elif a.method == "nvfp4_rtn":
        recipe = [QuantizationModifier(scheme="NVFP4A16", targets="Linear", ignore=IGNORE)]
        ds = None
    else:  # nvfp4_gptq
        ds = build_calib(tok, a.samples)
        recipe = [GPTQModifier(scheme="NVFP4A16", targets="Linear", ignore=IGNORE,
                               dampening_frac=0.1)]

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
    # carry the custom modeling files so the checkpoint is self-contained
    import shutil
    for f in ("configuration_spark.py", "modeling_spark.py", "chat_template.jinja"):
        src = pathlib.Path(a.model) / f
        if src.exists():
            shutil.copy2(src, pathlib.Path(out) / f)
    print(f"[llmc] {a.method} exported -> {out}", flush=True)


if __name__ == "__main__":
    main()
