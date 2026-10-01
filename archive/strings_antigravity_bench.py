import sqlite3, os, re

base = r"C:/Users/kanal/.gemini/antigravity/conversations"
names = [
    "0e44b06c-fab4-4ae1-8f52-67f3f9d25c25.db",
    "458809d6-685f-4558-ae3e-ec34185c8fb6.db",
    "74192778-467f-4a52-b6c2-6b6ce3699e73.db",
    "83fa5acb-1bd7-4469-8413-d861b97ae68e.db",
    "4676ac17-3635-43be-ad01-4ed815d2b436.db",
    "65c961a7-8f1d-4132-889d-c3a989f6dbb6.db",
    "c81323d7-68d3-4c39-84c2-26e0a658ae85.db",
    "f9d6f969-b7a2-4f60-8ff6-ba611ad851eb.db",
]
key = re.compile(
    rb"fff-bench|process-runs|sumo_fork|mega-corpus|warm.index|git am|open_corpus|fff-optimization|timeout-secs|compare-run|fff-main|WalkBuilder|scan timed",
    re.I,
)
# printable runs >= 8
str_re = re.compile(rb"[\x20-\x7e]{8,}")

for name in names:
    p = os.path.join(base, name)
    if not os.path.exists(p):
        continue
    c = sqlite3.connect("file:%s?mode=ro" % p, uri=True)
    print("========", name)
    hits = 0
    for (idx, payload) in c.execute("SELECT idx, step_payload FROM steps"):
        if not payload:
            continue
        blob = payload if isinstance(payload, bytes) else str(payload).encode()
        for s in str_re.findall(blob):
            if key.search(s):
                hits += 1
                text = s.decode("ascii", "replace")
                print("step", idx, ":", text[:400])
                if hits >= 50:
                    break
        if hits >= 50:
            break
    print("total hits printed", hits)
    c.close()
