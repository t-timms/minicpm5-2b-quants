"""v2 MiniCPM5-2B coding LoRA SFT — gentler than v1 (which regressed HE ~21pp).
LR 2e-5, r32/alpha32 (1:1), 1 epoch, cosine, max_length 2048, eager attn.
bf16 base, no bnb/unsloth. Run in ~/quant-env.
-> $FT_ADAPTER_OUT (adapter) + $FT_MERGED_OUT (merged bf16).
"""
from __future__ import annotations
import glob
import os
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL = "/home/ttimm/models/MiniCPM5-2B"
DATA = "/home/ttimm/minicpm-quant/ft_data_v2"
ADAPTER_OUT = os.environ.get("FT_ADAPTER_OUT", "/home/ttimm/models/minicpm5-2b-coding2-lora")
MERGED_OUT = os.environ.get("FT_MERGED_OUT", "/home/ttimm/models/minicpm5-2b-coding2-merged")
_MAX_STEPS = int(os.environ.get("FT_MAX_STEPS", "-1"))  # -1 => full 1 epoch

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token if isinstance(tok.eos_token, str) else "<|im_end|>"

model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager"
)
model.config.use_cache = False

train = load_dataset("parquet", data_files=f"{DATA}/train.parquet", split="train")
evald = load_dataset("parquet", data_files=f"{DATA}/eval.parquet", split="train")

peft_cfg = LoraConfig(
    r=32, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

sft_cfg = SFTConfig(
    output_dir=ADAPTER_OUT,
    num_train_epochs=1,
    max_steps=_MAX_STEPS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,
    eval_strategy="no",
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_length=2048,
    packing=False,
    dataset_text_field="text",
    report_to="none",
    seed=1234,
)

trainer = SFTTrainer(
    model=model,
    args=sft_cfg,
    train_dataset=train,
    eval_dataset=evald,
    peft_config=peft_cfg,
    processing_class=tok,
)

_resume = bool(glob.glob(os.path.join(ADAPTER_OUT, "checkpoint-*")))
print(f"[ft2] resume_from_checkpoint={_resume}  train_rows={len(train)}", flush=True)
trainer.train(resume_from_checkpoint=_resume)
trainer.save_model(ADAPTER_OUT)
print(f"[ft2] adapter -> {ADAPTER_OUT}", flush=True)

from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
merged = PeftModel.from_pretrained(base, ADAPTER_OUT).merge_and_unload()
merged.save_pretrained(MERGED_OUT, safe_serialization=True)
tok.save_pretrained(MERGED_OUT)
print(f"[ft2] merged -> {MERGED_OUT}", flush=True)
print("FT_TRAIN_DONE", flush=True)
