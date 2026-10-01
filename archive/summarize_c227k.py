import json
from collections import defaultdict
from pathlib import Path

p = Path(r"C:/Users/kanal/Projects/fff-upstream/archive/paired-c227k.jsonl")
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []

def med(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2

G = defaultdict(lambda: defaultdict(list))
for r in rows:
    k = (r["query"], r["threads"])
    G[k][r["subject"]].append(r["us_per_op"])
    G[k][r["subject"] + ".ck"].append(r["checksum"])

print(f"{'workload':<22} {'upstream':>10} {'keepers':>10} {'delta':>8}  ck")
for k in sorted(G):
    u = med(G[k]["upstream"])
    c = med(G[k]["keepers"])
    ck = "ok" if G[k]["upstream.ck"] == G[k]["keepers.ck"] else "DIFF"
    print(f"{k[0] + ' t=' + str(k[1]):<22} {u:10.1f} {c:10.1f} {100 * (c / u - 1):+7.1f}%  {ck}")
print("rows", len(rows))
