#!/usr/bin/env python
"""Post-quant coherence gate. Run in ~/vllm-env.
Serves a checkpoint on vLLM (sm_120, enforce_eager) and checks it produces real
text, not pad-token collapse. Inspects token IDs + finish_reason, not "looks empty"
(ZAYA1 lesson). Exit 0 = coherent, 2 = degenerate.

  coherence_check.py /home/ttimm/models/minicpm5-2b-nvfp4-w4a16
"""

import sys

from vllm import LLM, SamplingParams

PROMPTS = [
    "Write a Python function that returns the nth Fibonacci number.",
    "Explain what a hash map is in two sentences.",
    "What does John 3:16 say?",
]


def main() -> int:
    ckpt = sys.argv[1]
    llm = LLM(
        model=ckpt,
        dtype="auto",
        max_model_len=2048,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        trust_remote_code=True,
    )
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=128)
    msgs = [[{"role": "user", "content": p}] for p in PROMPTS]
    outs = llm.chat(msgs, sp)

    bad = 0
    for p, o in zip(PROMPTS, outs):
        c = o.outputs[0]
        ids = list(c.token_ids)
        uniq = len(set(ids))
        # MiniCPM pad_token_id (1) is also an eos id, so a trailing pad is normal
        # termination, not collapse. Only flag pad-dominance in a long output.
        pad_frac = ids.count(tok.pad_token_id) / max(len(ids), 1) if tok.pad_token_id is not None else 0.0
        txt = c.text.strip()
        degenerate = (uniq <= 3 and len(ids) > 10) or (pad_frac > 0.5 and len(ids) > 10) or len(txt) < 5
        flag = "  <-- DEGENERATE" if degenerate else ""
        bad += degenerate
        print(f"\n[{p[:50]}]")
        print(f"  finish={c.finish_reason} n_tok={len(ids)} uniq_ids={uniq} pad_frac={pad_frac:.2f}{flag}")
        print(f"  text: {txt[:200]!r}")

    print(f"\n{'PASS' if bad == 0 else f'FAIL ({bad}/{len(PROMPTS)} degenerate)'}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
