#!/usr/bin/env python3
"""Adversarial corpus for G11/P1b (deterministic, idempotent).

Layout under C:\\Users\\kanal\\fff-corpus:
  edge/       unicode, longpath, deep, wide, specials, case, hidden,
              content shapes, one safe junction, filler (total > 10k)
  edge-git/   gitignore negation repo + known git-dirt repo
  edge-links/ reserved for the junction cycle; created empty for now
              (a walk that follows a cyclic junction visits unboundedly;
              the cycle test stays manual until the scanner's junction
              policy is known).

Writes edge.json manifest with per-category counts and expected git-dirt
sets. Junctions need `_winapi.CreateJunction` (no admin needed).
"""

import ctypes
import json
import os
import random
import subprocess
from pathlib import Path

ROOT = Path(r"C:\Users\kanal\fff-corpus")
EDGE = ROOT / "edge"
EDGE_GIT = ROOT / "edge-git"
EDGE_LINKS = ROOT / "edge-links"
FILLER = 12000
WIDE = 5000


def touch(p: Path, data: bytes = b"") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_bytes(data)


def sh(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(list(args), cwd=str(cwd) if cwd else None,
                   capture_output=True, check=False)


def build_edge() -> None:
    uni = EDGE / "unicode"
    for nm in [
        "控制器_Main.rs", "日本語_ファイル_01.txt", "データ_02_sumo.xml",
        "тест_кириллица_03.txt", "test_עברית_04.txt", "café_05.txt",
        "cafe\u0301_06_combining.txt", "naïve_07.txt", "ümlaut_08.txt",
        "emoji_🚀_09.txt", "🚀rocket🎯_10.txt", "Δέλτα_11.txt",
        "ファイル名に日本語を含む長い名前のテスト_012.txt",
    ]:
        touch(uni / nm)

    lp = EDGE / "longpath"
    seg = "d" + "x" * 20
    cur = lp
    while len(str(cur)) < 235:
        cur = cur / seg
    touch(cur / ("very_long_file_name_" + "y" * 120 + ".txt"))
    touch(cur / "short.txt")
    cur2 = lp / ("e" + "z" * 30)
    while len(str(cur2)) < 240:
        cur2 = cur2 / seg
    touch(cur2 / ("another_very_long_name_" + "q" * 140 + ".txt"))

    dp = EDGE / "deep"
    cur = dp
    for i in range(50):
        cur = cur / f"lvl{i:02d}"
    touch(cur / "bottom.txt")
    touch(dp / "top.txt")

    wd = EDGE / "wide"
    for i in range(WIDE):
        touch(wd / f"w_{i:05d}.dat")

    sp = EDGE / "specials"
    for nm in [
        "with space.txt", "hash#tag.txt", "percent%20name.txt",
        "amp&and.txt", "brackets[1].txt", "brackets(2).txt",
        "plus+minus-.txt", "comma,semi;.txt", "equals=at@.txt",
        "bang!quest.txt", "caret^tilde~.txt", "dollar$brace{}.txt",
    ]:
        assert not any(c in nm for c in '<>:"/\\|?*'), nm
        touch(sp / nm)

    cs = EDGE / "case"
    touch(cs / "dir_a" / "CaseName.txt")
    touch(cs / "dir_b" / "casename.txt")
    touch(cs / "dir_c" / "CASENAME.TXT")
    touch(cs / "dir_d" / "CaseName.TXT")

    hd = EDGE / "hidden"
    touch(hd / ".hidden_dot_file.txt")
    touch(hd / ".hidden_dir" / "inside.txt")
    h = hd / "attrs_hidden.txt"
    touch(h)
    if os.name == "nt" and h.exists():
        ctypes.windll.kernel32.SetFileAttributesW(str(h), 0x02)

    cd = EDGE / "content"
    touch(cd / "zero_bytes.txt")
    touch(cd / "one_byte.txt", b"A")
    touch(cd / "crlf_text.txt", b"line one\r\nline two\r\n")
    touch(cd / "utf16le_text.txt", "hello utf16".encode("utf-16"))
    touch(cd / "binary_blob.bin", random.Random(7).randbytes(65536))
    touch(cd / "one_mb.bin", random.Random(9).randbytes(1024 * 1024))
    touch(cd / "no_extension")
    touch(cd / "many.dots.in.name.tar.gz")
    touch(cd / "UPPER.TXT")

    # One safe junction: edge/junction_to_content -> edge/content.
    if os.name == "nt":
        j = EDGE / "junction_to_content"
        if not j.exists():
            import _winapi
            _winapi.CreateJunction(str(cd), str(j))

    fd = EDGE / "filler"
    for i in range(FILLER):
        touch(fd / f"{i // 1000:04d}" / f"fill_{i:06}.dat")


def build_git() -> None:
    gi = EDGE_GIT / "gitignore"
    gi.mkdir(parents=True, exist_ok=True)
    if not (gi / ".git" / "HEAD").exists():
        sh("git", "init", "-q", str(gi))
    (gi / ".gitignore").write_text("*.log\n!keep.log\n", encoding="utf-8")
    touch(gi / "tracked.txt")
    touch(gi / "ignored.log")
    touch(gi / "keep.log")

    gd = EDGE_GIT / "gitdirt"
    if not (gd / ".git" / "HEAD").exists():
        gd.mkdir(parents=True, exist_ok=True)
        sh("git", "init", "-q", str(gd))
        sh("git", "-C", str(gd), "config", "user.email", "bench@example.com")
        sh("git", "-C", str(gd), "config", "user.name", "bench")
        touch(gd / "unmodified_one.txt")
        touch(gd / "unmodified_two.txt")
        touch(gd / "modified_one.txt")
        touch(gd / "staged_one.txt")
        sh("git", "-C", str(gd), "add", ".")
        sh("git", "-C", str(gd), "commit", "-q", "-m", "base")
        (gd / "modified_one.txt").write_text("changed", encoding="utf-8")
        (gd / "staged_one.txt").write_text("staged", encoding="utf-8")
        sh("git", "-C", str(gd), "add", "staged_one.txt")
        touch(gd / "untracked_one.txt")
        touch(gd / "untracked_two.txt")


def count_tree(d: Path, skip: set[str] = frozenset()) -> int:
    n = 0
    for dirpath, dirnames, filenames in os.walk(d):
        dirnames[:] = [x for x in dirnames
                       if not (Path(dirpath) / x) in skip]
        n += len(filenames)
    return n


def main() -> None:
    EDGE_LINKS.mkdir(parents=True, exist_ok=True)
    build_edge()
    build_git()

    counts = {}
    for cat in sorted(p.name for p in EDGE.iterdir() if p.is_dir()):
        counts[cat] = count_tree(EDGE / cat)
    total = count_tree(EDGE)
    manifest = {
        "edge_root": str(EDGE),
        "counts": counts,
        "edge_total_files": total,
        "filler": FILLER,
        "wide": WIDE,
        "gitdirt_expected": {
            "modified": 1, "staged": 1, "untracked": 2, "unmodified": 2,
        },
        "gitignore_expected": {
            "indexed": ["tracked.txt", "keep.log"],
            "ignored_present_on_disk": ["ignored.log"],
        },
        "notes": [
            "junction_to_content points at content/; observe whether the "
            "scan counts content/ twice",
            "edge-links/ is reserved for the cyclic-junction test; do not "
            "walk it until the junction policy is known",
        ],
    }
    out = ROOT / "edge.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items()
                      if k in ("counts", "edge_total_files")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
