#!/usr/bin/env python3
"""Generate the Chromium-shaped fill for the G-01 500k corpus.

The real trees are copied first and measured through the release driver; this
script creates exactly `target - base_indexed` files under `chromium-gen/`.
Everything is deterministic from --seed, including mtimes.
"""

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

IGNORED_COMPONENTS = {
    "node_modules", "__pycache__", "venv", ".venv", ".git",
    "target", "debug", "release", "bin", "obj", "Program Files",
}

TOP_TREES = [
    ("src/third_party/blink/renderer", 0.34),
    ("src/out/Release/gen", 0.20),
    ("src/components", 0.14),
    ("src/chrome/browser", 0.10),
    ("src/v8/src", 0.08),
    ("src/third_party/icu/source", 0.05),
    ("src/tools", 0.05),
    ("src/net", 0.04),
]

MODULES = [
    "core", "modules", "platform", "dom", "html", "css", "layout", "paint",
    "loader", "network", "storage", "service", "worker", "media", "canvas",
    "events", "bindings", "devtools", "extensions", "gpu", "mojo", "base",
    "metrics", "profiler", "scheduler", "security", "testing", "ui",
]

WORDS = [
    "score", "controller", "file_picker", "index", "netconvert", "query",
    "parser", "render", "frame", "resource", "loader", "session", "history",
    "profile", "package", "manifest", "config", "widget", "window", "layout",
    "pipeline", "dispatch", "handler", "registry", "factory", "observer",
]

EXTS = [
    (".cc", 0.17), (".h", 0.17), (".js", 0.12), (".ts", 0.10),
    (".json", 0.10), (".py", 0.08), (".md", 0.05), (".txt", 0.05),
    (".xml", 0.04), (".java", 0.03), (".mojom", 0.02), (".o", 0.03),
    (".a", 0.01), (".png", 0.01), (".bin", 0.02),
]
BINARY_EXTS = {".o", ".a", ".png", ".bin"}

SIZE_BUCKETS = [
    (0, 4096, 0.45),
    (4096, 16384, 0.25),
    (16384, 65536, 0.18),
    (65536, 262144, 0.08),
    (262144, 1048576, 0.03),
    (1048576, 8388608, 0.01),
]

UNICODE_SEGMENTS = ["naïve", "über", "日本語", "данные", "café"]


def weighted(rng, pairs):
    total = sum(w for _, w in pairs)
    x = rng.random() * total
    for value, weight in pairs:
        x -= weight
        if x <= 0:
            return value
    return pairs[-1][0]


def make_relpath(rng, i):
    top = weighted(rng, TOP_TREES)
    parts = [p for p in top.split("/") if p]

    depth = rng.choices([2, 3, 4, 5, 6], weights=[0.15, 0.30, 0.30, 0.18, 0.07])[0]
    for d in range(depth):
        mod = rng.choice(MODULES)
        if rng.random() < 0.002:
            mod = rng.choice(UNICODE_SEGMENTS)
        if rng.random() < 0.25:
            mod = f"{mod}_{rng.choice(WORDS)}"
        if mod.lower() in IGNORED_COMPONENTS:
            mod = "core"
        parts.append(mod)

    ext = weighted(rng, EXTS)
    stem = f"{rng.choice(WORDS)}_{i:06d}"
    parts.append(stem + ext)
    return "/".join(parts)


def size_for(rng):
    total = sum(w for _, _, w in SIZE_BUCKETS)
    x = rng.random() * total
    for lo, hi, weight in SIZE_BUCKETS:
        x -= weight
        if x <= 0:
            return rng.randint(lo, hi)
    lo, hi, _ = SIZE_BUCKETS[-1]
    return rng.randint(lo, hi)


def write_text(path, size, rel, i):
    header = (
        f"// {rel}\n"
        f"// generated chromium-shaped fill {i}\n"
        f"// score controller file_picker index netconvert __init__\n"
    ).encode("utf-8")
    block = (
        b"score controller file_picker index netconvert __init__ "
        b"render frame resource loader session history profile package\n"
    ) * 16
    with open(path, "wb") as f:
        if size <= len(header):
            f.write(header[:size])
            return
        f.write(header)
        remaining = size - len(header)
        while remaining > 0:
            chunk = block[: min(remaining, len(block))]
            f.write(chunk)
            remaining -= len(chunk)


def write_binary(path, size, rng):
    block = rng.randbytes(min(65536, max(1, size)))
    with open(path, "wb") as f:
        remaining = size
        while remaining > 0:
            chunk = block[: min(remaining, len(block))]
            f.write(chunk)
            remaining -= len(chunk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--target", type=int, default=500_000)
    ap.add_argument("--base-indexed", type=int, required=True)
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--summary", required=True)
    args = ap.parse_args()

    n = args.target - args.base_indexed
    if n <= 0:
        raise SystemExit(f"base {args.base_indexed} already reaches target {args.target}")

    root = Path(args.root)
    fill_root = root / "chromium-gen"
    fill_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    base_mtime = 1_767_225_600  # 2026-01-01T00:00:00Z

    by_ext = Counter()
    by_bucket = Counter()
    total_bytes = 0
    created = 0
    for i in range(n):
        rel = make_relpath(rng, i)
        path = fill_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        size = size_for(rng)
        ext = path.suffix.lower() or "<none>"
        if ext in BINARY_EXTS:
            write_binary(path, size, rng)
        else:
            write_text(path, size, "chromium-gen/" + rel, i)
        stamp = base_mtime + i * 60
        os.utime(path, (stamp, stamp))
        total_bytes += size
        by_ext[ext] += 1
        for lo, hi, _ in SIZE_BUCKETS:
            if lo <= size < hi:
                by_bucket[f"{lo}-{hi}"] += 1
                break
        created += 1
        if created % 1000 == 0:
            print(f"generated {created}/{n}", flush=True)

    summary = {
        "root": str(root),
        "target": args.target,
        "base_indexed": args.base_indexed,
        "generated": created,
        "seed": args.seed,
        "total_bytes": total_bytes,
        "by_ext": dict(sorted(by_ext.items())),
        "by_size_bucket": dict(sorted(by_bucket.items())),
        "fill_root": str(fill_root),
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
