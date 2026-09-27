"""Spark-X2.5-4B serving-gate coherence check on sm_120 via the out-of-tree plugin.
Run in ~/vllm-env. arg1 = model path, arg2 (optional) = 'graphs' to allow CUDA graphs.
Checks token ids + finish_reason, not "looks empty" (ZAYA1 lesson). Exit 0 = coherent.
"""
import sys


def main() -> int:
    from vllm import LLM, SamplingParams

    ckpt = sys.argv[1]
    eager = not (len(sys.argv) > 2 and sys.argv[2] == "graphs")

    prompts = [
        "Write a Python function that returns the nth Fibonacci number.",
        "Explain what a hash map is in two sentences.",
        "Reverse the string 'hello world' in Python and show the output.",
        "What is 17 * 23? Show your work.",
    ]

    llm = LLM(
        model=ckpt,
        dtype="auto",
        max_model_len=4096,
        gpu_memory_utilization=0.88,
        enforce_eager=eager,
        trust_remote_code=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=200)
    outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], sp)

    bad = 0
    for p, o in zip(prompts, outs):
        c = o.outputs[0]
        ids = list(c.token_ids)
        uniq = len(set(ids))
        txt = c.text.strip()
        degenerate = (uniq <= 3 and len(ids) > 10) or len(txt) < 5
        bad += degenerate
        print(f"\n[{p[:55]}]")
        print(f"  finish={c.finish_reason} n_tok={len(ids)} uniq_ids={uniq}"
              f"{'  <-- DEGENERATE' if degenerate else ''}")
        print(f"  text: {txt[:280]!r}")

    print(f"\n{'PASS' if bad == 0 else f'FAIL ({bad}/{len(prompts)})'}  "
          f"(mode={'eager' if eager else 'graphs'})")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
