"""Re-score logged mbpp_plus_instruct samples with a def-preserving extractor.
Runs extracted code + the sample's target asserts in a subprocess (15s timeout).
No GPU - uses only already-generated outputs.
"""
import glob
import json
import re
import subprocess
import sys

FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)
DEFLINE = re.compile(r"\s*def \w+\s*\(")
STOP = re.compile(r"\s*(#\s*Test|#\s*Example|assert |print\()")


def extract(raw: str) -> str:
    m = FENCE.search(raw)
    body = m.group(1) if m else raw
    lines = body.splitlines()
    start = 0
    for i, l in enumerate(lines):
        if DEFLINE.match(l):
            start = i
            break
    out = []
    for l in lines[start:]:
        if STOP.match(l) and out:
            break
        out.append(l)
    return "\n".join(out)


def passes(code: str, tests: str) -> bool:
    src = code + "\n\n" + tests + "\nprint('OK')\n"
    try:
        p = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, timeout=15)
        return p.returncode == 0 and "OK" in p.stdout
    except Exception:
        return False


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


for arm in ("bf16", "nvfp4_w4a16"):
    reps = sorted(glob.glob("/home/ttimm/minicpm-quant/eval/%s/mbpp_plus_instruct/rep*" % arm))
    scores = []
    ntot = 0
    for rep in reps:
        sfiles = glob.glob(rep + "/**/samples_*.jsonl", recursive=True)
        if not sfiles:
            continue
        rows = [json.loads(l) for l in open(sfiles[0])]
        ntot = len(rows)
        ok = 0
        for r in rows:
            resp = r["resps"][0]
            raw = resp[0] if isinstance(resp, list) else resp
            tgt = r["target"]
            tgt = tgt if isinstance(tgt, str) else tgt[0]
            if passes(extract(raw), tgt):
                ok += 1
        scores.append(ok / ntot)
        print("  %s %s: %d/%d = %.2f%%" % (arm, rep.split("/")[-1], ok, ntot, ok / ntot * 100))
    if scores:
        med = median(scores)
        print("  %s MBPP+ (re-scored): median %.2f%%  range %.2f-%.2f%%  (n=%d, draws=%d)\n"
              % (arm, med * 100, min(scores) * 100, max(scores) * 100, ntot, len(scores)))
