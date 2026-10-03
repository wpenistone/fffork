// Standalone access-pattern lab for the FFF chunked path arena.
//
// Question: the gate is latency-bound, not compute-bound (SSE2 did
// nothing; threads flat past 2; prefetch +5-8%). Its cost is a 3-level
// dependent load chain over a multi-MB working set:
//
//   index_offset -> indices[idx] -> chunks[arena_idx*CH] -> bytes
//
// This isolates THAT, from compute, by streaming every path byte through
// each layout doing only trivial per-byte work (fold + accumulate). No
// LCS here on purpose: any LCS I wrote would be a different kernel and
// would confound the comparison. fff's own gate already proved the
// compute side is not the bottleneck.
//
//   A  u32 index + deduped chunks   (today)
//   B  u16 index + deduped chunks   (only feasible <64k unique chunks)
//   C  contiguous, padded, no index/dedup  (pure sequential stream)
//   D  like A but first chunk stored INLINE per path (kills hop 1 for the
//      first 32 bytes, where most short needles live)
//
// Build: rustc -O arena_lab.rs -o arena_lab.exe

use std::hint::black_box;
use std::time::Instant;

const CH: usize = 32;

#[inline(always)]
fn fold(b: u8) -> u8 { if b.is_ascii_uppercase() { b + 32 } else { b } }

struct A { chunks: Vec<[u8; CH]>, indices: Vec<u32>, off: Vec<u32>, len: Vec<u16> }
struct B { chunks: Vec<[u8; CH]>, indices: Vec<u16>, off: Vec<u32>, len: Vec<u16> }
struct C { buf: Vec<u8>, off: Vec<u32>, len: Vec<u16> }
struct D { inl: Vec<[u8; CH]>, chunks: Vec<[u8; CH]>, indices: Vec<u32>, off: Vec<u32>, len: Vec<u16> }

fn push_dedup(chunks: &mut Vec<[u8; CH]>, dedup: &mut std::collections::HashMap<[u8; CH], u32>, c: &[u8]) -> u32 {
    let mut b = [0u8; CH]; b[..c.len()].copy_from_slice(c);
    *dedup.entry(b).or_insert_with(|| { chunks.push(b); (chunks.len()-1) as u32 })
}

fn build_a(p: &[Vec<u8>]) -> A {
    let (mut chunks, mut dedup) = (Vec::new(), std::collections::HashMap::new());
    let (mut indices, mut off, mut len) = (Vec::new(), Vec::new(), Vec::new());
    for q in p {
        off.push(indices.len() as u32); len.push(q.len() as u16);
        for c in q.chunks(CH) { indices.push(push_dedup(&mut chunks, &mut dedup, c)); }
    }
    A { chunks, indices, off, len }
}

fn build_b(p: &[Vec<u8>]) -> B {
    let (mut chunks, mut dedup) = (Vec::new(), std::collections::HashMap::new());
    let (mut indices, mut off, mut len) = (Vec::new(), Vec::new(), Vec::new());
    for q in p {
        off.push(indices.len() as u32); len.push(q.len() as u16);
        for c in q.chunks(CH) { indices.push(push_dedup(&mut chunks, &mut dedup, c) as u16); }
    }
    B { chunks, indices, off, len }
}

fn build_c(p: &[Vec<u8>]) -> C {
    let (mut buf, mut off, mut len) = (Vec::new(), Vec::new(), Vec::new());
    for q in p {
        off.push(buf.len() as u32); len.push(q.len() as u16);
        let s = buf.len();
        buf.resize(s + ((q.len() + CH - 1) / CH) * CH, 0);
        buf[s..s + q.len()].copy_from_slice(q);
    }
    C { buf, off, len }
}

fn build_d(p: &[Vec<u8>]) -> D {
    let (mut chunks, mut dedup) = (Vec::new(), std::collections::HashMap::new());
    let (mut indices, mut off, mut len, mut inl) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for q in p {
        off.push(indices.len() as u32); len.push(q.len() as u16);
        let mut first = [0u8; CH];
        first[..q.len().min(CH)].copy_from_slice(&q[..q.len().min(CH)]);
        inl.push(first);
        for c in q.chunks(CH).skip(1) { indices.push(push_dedup(&mut chunks, &mut dedup, c)); }
    }
    D { inl, chunks, indices, off, len }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).expect("paths.bin");
    let n = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let mut paths = Vec::with_capacity(n);
    let mut i = 4usize;
    for _ in 0..n {
        let l = u16::from_le_bytes(data[i..i+2].try_into().unwrap()) as usize; i += 2;
        paths.push(data[i..i+l].to_vec()); i += l;
    }
    let bytes: usize = paths.iter().map(|p| p.len()).sum();
    eprintln!("paths={n} total_bytes={bytes} avg={:.1}", bytes as f64 / n as f64);

    let A = build_a(&paths);
    let Bc = build_b(&paths);
    let Cc = build_c(&paths);
    let D = build_d(&paths);
    eprintln!("unique_chunks A={} -> u16 fits={}", A.chunks.len(), A.chunks.len() < 65536);
    eprintln!("footprint: A={:.2}MB B={:.2}MB C={:.2}MB D={:.2}MB",
        (A.chunks.len()*CH + A.indices.len()*4) as f64/1e6,
        (Bc.chunks.len()*CH + Bc.indices.len()*2) as f64/1e6,
        Cc.buf.len() as f64/1e6,
        (D.inl.len()*CH + D.chunks.len()*CH + D.indices.len()*4) as f64/1e6);

    let reps = 50;
    let mut res: Vec<(&str, f64)> = Vec::new();

    macro_rules! bench { ($nm:expr, $body:expr) => {{
        let t = Instant::now();
        let mut last = 0u64;
        for _ in 0..reps { last = black_box($body); }
        let ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
        res.push(($nm, ms));
        eprintln!("  {}: {:.3} ms (chk {})", $nm, ms, last & 0xff);
    }}};

    bench!("A_u32_index_dedup", {
        let mut acc = 0u64;
        for k in 0..A.len.len() {
            let total = A.len[k] as usize; let base = A.off[k] as usize;
            let mut pos = 0usize;
            while pos < total {
                let take = CH.min(total - pos); let blk = pos / CH; let skip = pos % CH;
                let idx = A.indices[base + blk] as usize;
                for j in skip..take { acc = acc.wrapping_add(fold(A.chunks[idx][j]) as u64); }
                pos += take;
            }
        }
        acc
    });

    bench!("B_u16_index_dedup", {
        let mut acc = 0u64;
        for k in 0..Bc.len.len() {
            let total = Bc.len[k] as usize; let base = Bc.off[k] as usize;
            let mut pos = 0usize;
            while pos < total {
                let take = CH.min(total - pos); let blk = pos / CH; let skip = pos % CH;
                let idx = Bc.indices[base + blk] as usize;
                for j in skip..take { acc = acc.wrapping_add(fold(Bc.chunks[idx][j]) as u64); }
                pos += take;
            }
        }
        acc
    });

    bench!("C_contiguous_noindex", {
        let mut acc = 0u64;
        for k in 0..Cc.len.len() {
            let total = Cc.len[k] as usize; let base = Cc.off[k] as usize;
            let mut pos = 0usize;
            while pos < total {
                let take = CH.min(total - pos); let blk = pos / CH; let skip = pos % CH;
                for j in skip..take { acc = acc.wrapping_add(fold(Cc.buf[base + blk*CH + j]) as u64); }
                pos += take;
            }
        }
        acc
    });

    bench!("D_inline_first_chunk", {
        let mut acc = 0u64;
        for k in 0..D.len.len() {
            let total = D.len[k] as usize; let base = D.off[k] as usize;
            let mut pos = 0usize;
            while pos < total {
                let take = CH.min(total - pos); let blk = pos / CH; let skip = pos % CH;
                if blk == 0 {
                    for j in skip..take { acc = acc.wrapping_add(fold(D.inl[k][j]) as u64); }
                } else {
                    let idx = D.indices[base + blk - 1] as usize;
                    for j in skip..take { acc = acc.wrapping_add(fold(D.chunks[idx][j]) as u64); }
                }
                pos += take;
            }
        }
        acc
    });

    println!("--- results (avg of {} reps) ---", reps);
    let base = res[0].1;
    for (nm, ms) in &res { println!("{nm:24} {ms:8.3} ms   {:5.2}x", base / ms); }
}