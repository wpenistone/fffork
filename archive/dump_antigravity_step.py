import sqlite3, os

p = r"C:/Users/kanal/.gemini/antigravity/conversations/0e44b06c-fab4-4ae1-8f52-67f3f9d25c25.db"
c = sqlite3.connect("file:%s?mode=ro" % p, uri=True)
print("sample steps:")
for idx, st, payload, fmt in c.execute("SELECT idx, step_type, step_payload, step_format FROM steps LIMIT 8"):
    pl = payload
    if isinstance(pl, bytes):
        print(idx, st, fmt, "bytes", len(pl), pl[:80])
    else:
        s = str(pl) if pl is not None else ""
        print(idx, st, fmt, type(pl).__name__, len(s), s[:120].replace("\n"," "))

print("\nsample gen_metadata:")
for idx, data, size in c.execute("SELECT idx, data, size FROM gen_metadata LIMIT 5"):
    d = data
    if isinstance(d, bytes):
        print(idx, "bytes", len(d), d[:60])
    else:
        s = str(d) if d is not None else ""
        print(idx, size, s[:150].replace("\n"," "))

print("\nstep_type counts")
for r in c.execute("SELECT step_type, count(*) FROM steps GROUP BY 1"):
    print(r)
c.close()
