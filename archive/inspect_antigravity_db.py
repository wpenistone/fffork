import sqlite3, os, sys
base = r"C:/Users/kanal/.gemini/antigravity/conversations"
names = [
    "0e44b06c-fab4-4ae1-8f52-67f3f9d25c25.db",
    "458809d6-685f-4558-ae3e-ec34185c8fb6.db",
    "83fa5acb-1bd7-4469-8413-d861b97ae68e.db",
    "99b8ef63-307c-4f4f-9a54-f73cb90f5e93.db",
    "74192778-467f-4a52-b6c2-6b6ce3699e73.db",
    "0d4ce217-6bd4-4583-8d78-ef8b45457206.db",
]
for name in names:
    p = os.path.join(base, name)
    if not os.path.exists(p):
        print("missing", name)
        continue
    try:
        c = sqlite3.connect("file:%s?mode=ro" % p, uri=True)
        tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        print("===", name, "tables", tables)
        for t in tables:
            cols = [r[1] for r in c.execute("PRAGMA table_info(%s)" % t)]
            n = c.execute("SELECT count(*) FROM %s" % t).fetchone()[0]
            print(" ", t, "rows", n, "cols", cols[:15])
        c.close()
    except Exception as e:
        print(name, "ERR", e)
