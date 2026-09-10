import json, collections, sys

f = sys.argv[1]
rows = [json.loads(l) for l in open(f)]
c = collections.Counter(r["pass@1"] for r in rows)
print("n=", len(rows), "pass@1 dist:", dict(c))

def as_str(x):
    while isinstance(x, list) and x:
        x = x[0]
    return x if isinstance(x, str) else ""

fails = [r for r in rows if not r["pass@1"]]
degen = 0
for r in fails:
    fr = as_str(r.get("filtered_resps"))
    lines = [l.strip() for l in fr.split("\n") if l.strip()]
    if lines:
        mx = collections.Counter(lines).most_common(1)[0][1]
        if mx >= 8:
            degen += 1
print("fails with >=8x repeated line (degenerate loop): %d / %d fails" % (degen, len(fails)))

for r in rows:
    if r["doc"]["task_id"] == "HumanEval/9":
        print("\n=== HumanEval/9 (looked correct) ===")
        print("pass@1:", r["pass@1"])
        print("filtered code:\n", as_str(r.get("filtered_resps")))
        print("--- target/test (first 500) ---\n", str(r["target"])[:500])
