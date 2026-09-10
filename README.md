# MiniCPM5-2B on consumer Blackwell: FP8 vs NVFP4-W4A16, measured

The first FP8 and first NVFP4 quantizations of
[`openbmb/MiniCPM5-2B`](https://huggingface.co/openbmb/MiniCPM5-2B) (a 2.5 B dense
`LlamaForCausalLM`, Apache-2.0), with a measured quant-quality comparison, serving
notes for an RTX 5070 Ti (Blackwell, SM120), and a fine-tune experiment that did
not pan out — written up so the negative result is on record.

## Releases

| repo | format | size | use it when |
|---|---|---:|---|
| [`Ttimms/MiniCPM5-2B-FP8`](https://huggingface.co/Ttimms/MiniCPM5-2B-FP8) | FP8-dynamic (compressed-tensors) | 2.84 GiB | you want maximum quality retention |
| [`Ttimms/MiniCPM5-2B-NVFP4`](https://huggingface.co/Ttimms/MiniCPM5-2B-NVFP4) | NVFP4 W4A16, GPTQ rounding | 2.03 GiB | you need to fit near 2 GB |

## Method

`lm-evaluation-harness`, vLLM 0.26 backend, greedy decoding. **Every number is the
median of 3 independent draws, with the range**: the harness is non-deterministic
run-to-run even at greedy (batch scheduling, kernel reductions), so a single draw
is not a reproducible score. Differences smaller than the observed spread are
treated as unresolved. HumanEval-instruct is `pass@1` with `create_test`
(n = 164); MBPP is the base 3-shot task (n = 500). All arms measured in one
session against the released checkpoint.

## Results

| build | HumanEval-inst | MBPP (3-shot) | size | Δ HE / MBPP vs bf16 |
|---|---:|---:|---:|---:|
| bf16 base | 86.59 % (85.98–86.59) | 50.60 % (50.40–51.00) | 4.68 GiB | — |
| **FP8-dynamic** | **84.76 %** (84.15–85.37) | **48.80 %** (48.80–49.00) | 2.84 GiB | −1.8 / −1.8 pp |
| **NVFP4-W4A16 (GPTQ)** | **84.15 %** (81.71–84.15) | **45.80 %** (45.60–46.40) | 2.03 GiB | −2.4 / −4.8 pp |
| NVFP4-W4A16 (RTN) | 79.88 % (77.44–79.88) | 41.20 % (41.20–41.80) | 2.03 GiB | −6.7 / −9.4 pp |
| mixed (MLP-NVFP4 + attn-FP8) | 84.15 % (82.93–85.98) | 46.60 % (46.00–46.80) | 2.20 GiB | −2.4 / −4.0 pp |

**Reading it:**

- **FP8 is effectively lossless** — both deltas sit inside the eval's own
  run-to-run spread. It is the recommended default.
- **NVFP4-W4A16 with GPTQ rounding** costs ~2.4 pp HumanEval / ~4.8 pp MBPP for
  57 % less disk. GPTQ recovers ~4–5 pp over plain RTN, which is not usable.
- **The mixed build gives no advantage** over pure NVFP4-GPTQ — same HumanEval,
  slightly better MBPP, but 0.17 GiB larger. Not shipped.
- The "2–4 % NVFP4 quality cost" figure quoted for large models **does not hold
  for a 2.5 B dense model** — there is less redundancy to absorb 4-bit weights,
  and MBPP in particular takes a real hit even with Hessian-aware rounding. NVFP4
  pays off here only if the extra ~0.8 GiB (vs FP8) matters for your KV budget.

## Serving on SM120 (RTX 5070 Ti)

- **`--kv-cache-dtype fp8`** — roughly halves KV cache, no measurable quality cost
  on these tasks.
- **NVFP4-W4A16 decodes through the Marlin kernel to a bf16 GEMM** on SM120 — no
  native FP4 arithmetic on the weight-only path. The benefit is footprint, not
  raw speed. (Native FP4 compute needs W4A4, which is a separate story and is not
  what these builds are.)
- **On WSL**, set `VLLM_USE_V2_MODEL_RUNNER=0` — the V2 runner needs UVA, which WSL
  disables (`pin_memory=False`), and the engine dies at init otherwise.
- **`enforce_eager`** was used for all evals here for determinism of the serving
  path; weight-only NVFP4-W4A16 also runs correctly with CUDA graphs.

```bash
vllm serve Ttimms/MiniCPM5-2B-FP8 --max-model-len 32768 --kv-cache-dtype fp8
```

## Does fine-tuning the base improve its coding? No.

Two LoRA SFT runs on top of the released instruct checkpoint, deliberately
different:

| run | data | LoRA | LR | HumanEval | MBPP | IFEval (prompt-strict) |
|---|---|---|---:|---:|---:|---:|
| base | — | — | — | 87.8 % | 50.6 % | 41.4 % |
| v1 | self-oss-instruct 12k + ultrachat | r32 / α64 | 1e-4, 2 ep | 63.4 % | 48.2 % | 44.7 % |
| v2 | OpenCodeInstruct (test-passing) 11k + evol + alpaca | r32 / α32 | 2e-5, 1 ep | 73.2 % | 48.6 % | 50.3 % |

*(base re-measured in the v2 harness for a like-for-like comparison; 3 draws each.)*

Both runs **regress HumanEval** (−24 pp, then −15 pp), hold MBPP, and **improve
IFEval instruction-following** (+3 pp, +9 pp). The gentler v2 config removed the
degeneration failure mode v1 had (repetition loops) but not the capability drop —
v2's failures are clean code that misses edge cases.

The base's coding sits at a sharp optimum from openbmb's own SFT + RL pipeline
(`UltraData-SFT` + RL per the model card). Additional SFT on a different
distribution, without RL, partially overwrites the edge-case defensiveness that
HumanEval rewards. Beating the base on coding would need training from
`openbmb/MiniCPM5-2B-Base` or RL with execution feedback — not a light adapter.
**The quants target the released checkpoint. The fine-tunes are not published.**

## Repro

- `scripts/quantize_llmc.py` — FP8 / NVFP4-W4A16 (GPTQ or RTN) via
  [llm-compressor](https://github.com/vllm-project/llm-compressor) 0.13
- `scripts/coherence_check.py` — post-quant token-id + `finish_reason` gate
- `scripts/eval_repeat_mc.sh` — the repeated-draw eval wrapper
- `scripts/ft_*.py`, `scripts/run_finetune_v2.sh` — the fine-tune runs
- `results/` — the raw per-run summaries the tables above are built from

## Licenses / attribution

- Models: Apache-2.0, inherited from `openbmb/MiniCPM5-2B`.
- NVFP4 GPTQ calibration: `theblackcat102/evol-codealpaca-v1` (code, disjoint from
  the eval sets).
- Fine-tune data (not shipped): `nvidia/OpenCodeInstruct`
  ([arXiv:2504.04030](https://arxiv.org/abs/2504.04030), CC-BY-4.0),
  `theblackcat102/evol-codealpaca-v1`, `yahma/alpaca-cleaned`.
