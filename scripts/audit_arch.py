"""Stage 0.2 - MiniCPM5-2B architecture audit for ModelOpt NVFP4/AWQ.
Meta-device load (no weights, no GPU): walk the module tree, confirm every
quantizable Linear is standard nn.Linear, flag anything ModelOpt won't recognize.
"""

import collections

import torch
from transformers import AutoConfig, AutoModelForCausalLM

MODEL = "/home/ttimm/models/MiniCPM5-2B"

cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=False)
print(f"arch={cfg.architectures} model_type={cfg.model_type}")
print(f"layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} "
      f"heads={cfg.num_attention_heads}/{cfg.num_key_value_heads} "
      f"head_dim={getattr(cfg, 'head_dim', None)} vocab={cfg.vocab_size} "
      f"tie_emb={cfg.tie_word_embeddings}")

with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=False)

by_type = collections.Counter(type(m).__name__ for m in model.modules())
linear_names = collections.Counter()
non_std_in_attn_mlp = []
for name, m in model.named_modules():
    if isinstance(m, torch.nn.Linear):
        # bucket by the last two path components (q_proj, gate_proj, ...)
        linear_names[name.split(".")[-1]] += 1
    # anything custom sitting where a Linear normally would be
    if any(k in name for k in (".self_attn.", ".mlp.")) and not isinstance(
        m, (torch.nn.Linear, torch.nn.Module.__mro__[0])
    ):
        pass

print("\nmodule types:", dict(by_type))
print("\nLinear layers by role:", dict(linear_names))
total_linear = sum(linear_names.values())
print(f"total nn.Linear: {total_linear}")

# what ModelOpt/llm-compressor would quantize vs skip
skip = {"lm_head"}
quantizable = total_linear - sum(v for k, v in linear_names.items() if k in skip)
print(f"quantizable (excl {skip}): {quantizable}")

# sanity: standard llama has per layer q,k,v,o + gate,up,down = 7 linears
expected = cfg.num_hidden_layers * 7 + (0 if cfg.tie_word_embeddings else 1)
print(f"expected for vanilla llama ({cfg.num_hidden_layers}*7 + lm_head): {expected}")
print("MATCH" if total_linear == expected else f"MISMATCH (diff {total_linear - expected})")

# custom / non-Linear modules anywhere
custom = [t for t in by_type if t not in {
    "MiniCPM5ForCausalLM", "LlamaForCausalLM", "LlamaModel", "LlamaDecoderLayer",
    "LlamaAttention", "LlamaMLP", "LlamaRMSNorm", "LlamaRotaryEmbedding",
    "Linear", "Embedding", "ModuleList", "SiLU", "ACT2FN", "SiLUActivation",
}]
print("\nnon-standard module types (investigate if any):", custom or "none")
