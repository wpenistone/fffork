import json
from pathlib import Path

d = json.loads(Path(r"C:/Users/kanal/Projects/fff-upstream/archive/compare-quiet-keepers.json").read_text(encoding="utf-8-sig"))
cells = d.get("cell_deltas", [])
print("=== by CPU delta (neg = keepers better) ===")
for c in sorted(cells, key=lambda x: x.get("cpu_total_us_p95_rel") or 0):
    cpu = c.get("cpu_total_us_p95_rel") or 0
    wall = c.get("wall_us_p95_rel") or 0
    cap = c.get("cap")
    key = (c.get("cell_key") or "")[-75:]
    print("cpu=%+.2f wall=%+.3f cap=%s %s" % (cpu, wall, cap, key))

imp_w = [c for c in cells if (c.get("wall_us_p95_rel") or 0) < -0.02]
reg_w = [c for c in cells if (c.get("wall_us_p95_rel") or 0) > 0.02]
imp_c = [c for c in cells if (c.get("cpu_total_us_p95_rel") or 0) < -0.02]
reg_c = [c for c in cells if (c.get("cpu_total_us_p95_rel") or 0) > 0.02]
print("wall improved=%d regressed=%d | cpu improved=%d regressed=%d" % (len(imp_w), len(reg_w), len(imp_c), len(reg_c)))
fails = [c for c in cells if c.get("cap") == "fail"]
print("cap fails=%d" % len(fails))
for c in fails:
    print("  FAIL wall=%+.3f %s" % (c.get("wall_us_p95_rel") or 0, (c.get("cell_key") or "")[-70:]))
