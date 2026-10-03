// Memory-bound probe: same four layouts, but per-CHUNK work instead of
// per-byte, so the only cost left is resolving the dependent load chain.
//
// arena_lab.rs answered "does layout matter" but was compute-bound: 11.7 ms
// for 19.7 MB is ~1.7 bytes/ns, i.e. scalar byte-loop throughput. It saw
// no memory difference because ALU dominated.
//
// Here each layout resolves every chunk pointer and touches ONE byte per
// chunk. That is ~2.7 dependent loads per path (avg 86.5 B / 32 B chunks)
// against ~87 bytes of work removed, so the residual signal is memory.
//
// Layout E is the control: contiguous storage walked the same way. If the
// indirection is real, E must beat A. If A ~= E, the chain is free.
//
// Build: rustc -O --edition 2021 mem_probe.rs -o mem_probe.exe

use std::hint::black_box;
use std::time::Instant;

const CH: usize = 32;

fn read_all(p: &[u8]) -> Vec<Vec<u8>> {
    let n = u32::from_le_bytes(p[0..4].try_into().unwrap()) as usize;
    let mut out = Vec::with_capacity(n);
    let mut i = 4usize;
    for _ in 0..n {
        let l = u16::from_le_bytes(p[i..i+2].try_into().unwrap()) as usize; i += 2;
        out.push(p[i..i+l].to_vec()); i += l;
    }
    out
}

fn build_indexed(p: &[Vec<u8>]) -> (Vec<[u8; CH]>, Vec<u32>, Vec<u32>, Vec<u16>) {
    let (mut chunks, mut dedup) = (Vec::new(), std::collections::HashMap::new());
    let (mut idx, mut off, mut len) = (Vec::new(), Vec::new(), Vec::new());
    for q in p {
        off.push(idx.len() as u32); len.push(q.len() as u16);
        for c in q.chunks(CH) {
            let mut b = [0u8; CH]; b[..c.len()].copy_from_slice(c);
            let v = *dedup.entry(b).or_insert_with(|| { chunks.push(b); (chunks.len()-1) as u32 });
            idx.push(v);
        }
    }
    (chunks, idx, off, len)
}

fn build_contig(p: &[Vec<u8>]) -> (Vec<[u8; CH]>, Vec<u32>, Vec<u32>, Vec<u16>) {
    let (mut chunks, mut off, mut len) = (Vec::new(), Vec::new(), Vec::new());
    for q in p {
        off.push(chunks.len() as u32); len.push(q.len() as u16);
        for c in q.chunks(CH) {
            let mut b = [0u8; CH]; b[..c.len()].copy_from_slice(c);
            chunks.push(b);
        }
    }
    (chunks, off.clone(), off, len)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).expect("paths.bin");
    let paths = read_all(&data);
    let chunks_total: usize = paths.iter().map(|q| q.len().div_ceil(CH)).sum();
    eprintln!("paths={} chunk_reads={} avg_path={:.1}", paths.len(), chunks_total,
        paths.iter().map(|p| p.len()).sum::<usize>() as f64 / paths.len() as f64);

    let (achunks, aidx, aoff, alen) = build_indexed(&paths);
    let (cchunks, _coff2, coff, clen) = build_contig(&paths);
    eprintln!("indexed: unique_chunks={} index_entries={} footprint={:.2}MB",
        achunks.len(), aidx.len(), (achunks.len()*CH + aidx.len()*4) as f64/1e6);
    eprintln!("contig : slots={} footprint={:.2}MB", cchunks.len(), cchunks.len() as f64 * CH as f64 / 1e6);

    let reps = 200;
    let mut res: Vec<(&str, f64)> = Vec::new();

    // A: resolve index -> chunk, touch one byte per chunk.
    let t = Instant::now();
    let mut acc = 0u64;
    for _ in 0..reps {
        let mut a = 0u64;
        for k in 0..alen.len() {
            let total = alen[k] as usize; let base = aoff[k] as usize;
            let mut pos = 0usize;
            while pos < total {
                let blk = pos / CH;
                let ci = aidx[base + blk] as usize;
                a = a.wrapping_add(achunks[ci][0] as u64);
                pos += CH;
            }
        }
        acc = black_box(a);
    }
    let ta = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
    res.push(("A_indexed_u32_dedup", ta));

    // E: same walk over contiguous storage (control).
    let t = Instant::now();
    let mut acc2 = 0u64;
    for _ in 0..reps {
        let mut a = 0u64;
        for k in 0..clen.len() {
            let total = clen[k] as usize; let base = coff[k] as usize;
            let mut pos = 0usize;
            while pos < total {
                a = a.wrapping_add(cchunks[base + pos / CH][0] as u64);
                pos += CH;
            }
        }
        acc2 = black_box(a);
    }
    let te = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
    res.push(("E_contiguous_control", te));

    // F: read the SAME bytes sequentially, no path structure at all.
    let flat: Vec<u8> = paths.concat();
    let t = Instant::now();
    let mut acc3 = 0u64;
    for _ in 0..reps {
        let mut a = 0u64;
        let mut i = 0usize;
        while i < flat.len() { a = a.wrapping_add(flat[i] as u64); i += CH; }
        acc3 = black_box(a);
    }
    let tf = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
    res.push(("F_flat_sequential", tf));

    println!("--- memory-bound probe ({reps} reps) ---");
    eprintln!("checks: A={} E={} F={}", acc & 0xff, acc2 & 0xff, acc3 & 0xff);
    let base = res[0].1;
    for (nm, ms) in &res {
        println!("{nm:24} {ms:8.3} ms   {:5.2}x", base / ms);
    }
    println!("\nchunk-read throughput (A): {:.2} Mchunk/s",
        chunks_total as f64 / (ta / 1000.0) / 1e6);
}