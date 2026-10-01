import sqlite3, os, re, json

base = r"C:/Users/kanal/.gemini/antigravity/conversations"
# July 15 cluster (atlas day) + large July 6
names = [
    "0e44b06c-fab4-4ae1-8f52-67f3f9d25c25.db",
    "458809d6-685f-4558-ae3e-ec34185c8fb6.db",
    "74192778-467f-4a52-b6c2-6b6ce3699e73.db",
    "83fa5acb-1bd7-4469-8413-d861b97ae68e.db",
    "4676ac17-3635-43be-ad01-4ed815d2b436.db",
    "65c961a7-8f1d-4132-889d-c3a989f6dbb6.db",
]
pat = re.compile(r"fff-bench|process-runs|sumo_fork|warm.index|mega-corpus|fff-optimization|git am|walk_collect|timeout_secs|open_corpus", re.I)

for name in names:
    p = os.path.join(base, name)
    if not os.path.exists(p):
        continue
    c = sqlite3.connect("file:%s?mode=ro" % p, uri=True)
    print("========", name)
    hits = 0
    for idx, payload, fmt in c.execute("SELECT idx, step_payload, step_format FROM steps ORDER BY idx"):
        if payload is None:
            continue
        if isinstance(payload, bytes):
            try:
                text = payload.decode("utf-8", "replace")
            except Exception:
                continue
        else:
            text = str(payload)
        if not pat.search(text):
            continue
        hits += 1
        # print a compact snippet around each match
        for m in pat.finditer(text):
            a = max(0, m.start() - 120)
            b = min(len(text), m.end() + 200)
            snip = text[a:b].replace("\n", " ")
            print("step", idx, fmt, ":", snip[:320])
            if hits > 40:
                break
        if hits > 40:
            break
    print("hits", hits)
    c.close()
