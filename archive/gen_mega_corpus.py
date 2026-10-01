"""Generate a ~225k-file monorepo-shaped corpus for FFF E2E benches."""
import os
import random
from pathlib import Path

ROOT = Path(r"C:/Users/kanal/Projects/fff-upstream/.fff-bench/cache/mega-repo")
random.seed(225_000)

PACKAGES = [
    "core", "search", "index", "grep", "query", "path", "score", "watch",
    "bench", "cli", "mcp", "nvim", "c-api", "python", "node", "bun",
    "docs", "examples", "tools", "vendor", "internal", "shared",
    "auth", "billing", "catalog", "checkout", "inventory", "shipping",
    "analytics", "ml", "pipeline", "storage", "queue", "cache",
]
DIRS = [
    "src", "tests", "benches", "docs", "examples", "fixtures", "scripts",
    "src/handlers", "src/models", "src/utils", "src/api", "src/db",
    "tests/unit", "tests/integration", "tests/e2e",
    "packages/lib/src", "packages/lib/tests",
    "crates/mod/src", "crates/mod/tests",
    "deep/nested/level/one/two/three",
]
EXTS = ["rs", "ts", "tsx", "js", "py", "go", "java", "md", "toml", "json", "yml", "c", "h", "lua"]
NAMES = [
    "index", "main", "lib", "mod", "util", "utils", "helper", "service",
    "handler", "controller", "model", "schema", "config", "types", "error",
    "client", "server", "router", "store", "cache", "queue", "worker",
    "parser", "lexer", "ast", "score", "match", "search", "grep", "walk",
    "file_picker", "path_utils", "background_watcher", "simd_path", "bigram",
]

def body(i: int, name: str) -> bytes:
    kind = i % 7
    if kind == 0:
        return f"pub fn {name}_{i}() -> i32 {{ {i} }}\n".encode()
    if kind == 1:
        return f"export function {name}{i}(x: number): number {{ return x + {i}; }}\n".encode()
    if kind == 2:
        return f"def {name}_{i}():\n    return {i}\n".encode()
    if kind == 3:
        return f"// {name} module {i}\nmodule {name}_{i};\n".encode()
    if kind == 4:
        return f"# {name} {i}\n\nDocs for {name}_{i}.\n".encode()
    if kind == 5:
        return f"fn main() {{ println!(\"{name}{i}\"); }}\n".encode()
    return f"package {name}\n\nfunc {name}{i}() int {{ return {i} }}\n".encode()

def main() -> None:
    if ROOT.exists():
        print(f"exists: {ROOT} ({sum(1 for _ in ROOT.rglob('*') if _.is_file())} files)")
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    written = 0
    target = 225_000
    # Realistic monorepo: many packages x dirs x files
    pkg_i = 0
    while written < target:
        pkg = PACKAGES[pkg_i % len(PACKAGES)]
        pkg_i += 1
        pkg_dir = ROOT / f"{pkg}-{pkg_i:04d}"
        for d in DIRS:
            if written >= target:
                break
            dd = pkg_dir / d
            dd.mkdir(parents=True, exist_ok=True)
            nfiles = 40 if "deep" not in d else 20
            for j in range(nfiles):
                if written >= target:
                    break
                name = NAMES[(written + j) % len(NAMES)]
                ext = EXTS[(written + j) % len(EXTS)]
                path = dd / f"{name}_{written % 10000}.{ext}"
                path.write_bytes(body(written, name))
                written += 1
        if pkg_i % 20 == 0:
            print(f"  {written} files...")
    print(f"wrote {written} files under {ROOT}")

if __name__ == "__main__":
    main()
