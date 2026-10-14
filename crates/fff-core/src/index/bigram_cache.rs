// Persisted post-scan bigram index: skip rebuilding file-content
// bitsets on every startup when the corpus is unchanged. Fingerprint
// (paths, sizes, mtimes in index order + build constants) invalidates
// on any change; corruption degrades to a rebuild, never a panic.
use crate::index::bigram_filter::BigramFilter;
use crate::simd_path::ArenaPtr;
use crate::types::FileItem;

const MAGIC: u64 = 0x6666726269677261;
const CODEC: u32 = 1;
const MAX_SKIP_DEPTH: usize = 4;

fn u64le(v: u64, out: &mut Vec<u8>) {
    out.extend_from_slice(&v.to_le_bytes());
}

struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn bytes(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        let s = self.b.get(self.pos..end)?;
        self.pos = end;
        Some(s)
    }

    fn u64(&mut self) -> Option<u64> {
        let s = self.bytes(8)?;
        Some(u64::from_le_bytes(s.try_into().ok()?))
    }

    fn u8(&mut self) -> Option<u8> {
        let s = self.bytes(1)?;
        Some(s[0])
    }

    fn vec_u16(&mut self) -> Option<Vec<u16>> {
        let n = self.u64()? as usize;
        let s = self.bytes(n.checked_mul(2)?)?;
        Some(s.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
    }

    fn vec_u64(&mut self) -> Option<Vec<u64>> {
        let n = self.u64()? as usize;
        let s = self.bytes(n.checked_mul(8)?)?;
        Some(
            s.chunks_exact(8)
                .map(|c| {
                    u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                })
                .collect(),
        )
    }

    fn vec_u32(&mut self) -> Option<Vec<u32>> {
        let n = self.u64()? as usize;
        let s = self.bytes(n.checked_mul(4)?)?;
        Some(s.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    fn vec_u8(&mut self) -> Option<Vec<u8>> {
        let n = self.u64()? as usize;
        self.bytes(n).map(<[u8]>::to_vec)
    }

    fn hash32(&mut self) -> Option<[u8; 32]> {
        self.bytes(32)?.try_into().ok()
    }
}

fn write_filter(f: &BigramFilter, out: &mut Vec<u8>) {
    let (lookup, dense, dense_count, offsets, sparse, words, file_count, populated, skip) =
        f.parts();
    u64le(lookup.len() as u64, out);
    for v in lookup {
        out.extend_from_slice(&v.to_le_bytes());
    }
    u64le(dense.len() as u64, out);
    for v in dense {
        out.extend_from_slice(&v.to_le_bytes());
    }
    u64le(dense_count as u64, out);
    u64le(offsets.len() as u64, out);
    for v in offsets {
        out.extend_from_slice(&v.to_le_bytes());
    }
    u64le(sparse.len() as u64, out);
    out.extend_from_slice(sparse);
    u64le(words as u64, out);
    u64le(file_count as u64, out);
    u64le(populated as u64, out);
    match skip {
        Some(s) => {
            out.push(1);
            write_filter(s, out);
        }
        None => out.push(0),
    }
}

fn read_filter(r: &mut Reader, depth: usize) -> Option<BigramFilter> {
    if depth > MAX_SKIP_DEPTH {
        return None;
    }
    let filter = BigramFilter::from_parts(
        r.vec_u16()?,
        r.vec_u64()?,
        r.u64()? as usize,
        r.vec_u32()?,
        r.vec_u8()?,
        r.u64()? as usize,
        r.u64()? as usize,
        r.u64()? as usize,
        None,
    )?;
    let skip = match r.u8()? {
        0 => None,
        1 => Some(read_filter(r, depth + 1)?),
        _ => return None,
    };
    if skip.is_some() {
        let mut filter = filter;
        let (lookup, dense, dense_count, offsets, sparse, words, file_count, populated, _) =
            filter.parts();
        let rebuilt = BigramFilter::from_parts(
            lookup.to_vec(),
            dense.to_vec(),
            dense_count,
            offsets.to_vec(),
            sparse.to_vec(),
            words,
            file_count,
            populated,
            skip,
        )?;
        return Some(rebuilt);
    }
    Some(filter)
}

// Corpus fingerprint: blake3 over index-order (path, size, mtime) plus
// build constants. Order matters (indices must align); any content,
// rename, add, or remove changes it.
pub(crate) fn fingerprint(
    files: &[FileItem],
    base_path: &std::path::Path,
    arena: ArenaPtr,
) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(b"fff-bigram-cache-v1");
    h.update(&(files.len() as u64).to_le_bytes());
    h.update(&(crate::constants::MAX_INDEXABLE_FILE_SIZE as u64).to_le_bytes());
    h.update(&crate::index::bigram_filter::SKIP_INDEX_MIN_DENSITY_PCT.to_le_bytes());
    let mut path_buf = [0u8; crate::simd_path::PATH_BUF_SIZE];
    for f in files {
        let abs = f.write_absolute_path(arena, base_path, &mut path_buf);
        h.update(abs.as_os_str().as_encoded_bytes());
        h.update(&f.size.to_le_bytes());
        h.update(&f.modified.to_le_bytes());
    }
    *h.finalize().as_bytes()
}

// Cache file per corpus (blake3 of the base path), under the OS cache dir.
pub(crate) fn cache_path(base_path: &std::path::Path) -> Option<std::path::PathBuf> {
    let dir = dirs::cache_dir()?.join("fff").join("bigram");
    let mut h = blake3::Hasher::new();
    h.update(base_path.as_os_str().as_encoded_bytes());
    let name = h.finalize().to_hex();
    Some(dir.join(format!("{name}.bin")))
}

// Masks (u64/file, index order), binary flags (bitset), fingerprint.
pub(crate) fn save(
    filter: &BigramFilter,
    masks: Option<(Vec<u64>, bool)>,
    binary: &[u64],
    fingerprint: &[u8; 32],
    path: &std::path::Path,
) {
    let mut out = Vec::new();
    u64le(MAGIC, &mut out);
    out.extend_from_slice(&CODEC.to_le_bytes());
    out.extend_from_slice(fingerprint);
    write_filter(filter, &mut out);
    match masks {
        Some((slots, has_data)) => {
            out.push(1);
            u64le(slots.len() as u64, &mut out);
            for v in &slots {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out.push(u8::from(has_data));
        }
        None => out.push(0),
    }
    u64le(binary.len() as u64, &mut out);
    for v in binary {
        out.extend_from_slice(&v.to_le_bytes());
    }
    let sum = blake3::hash(&out);
    out.extend_from_slice(sum.as_bytes());
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Err(e) = std::fs::write(path, &out) {
        tracing::warn!(path = ?path.display(), ?e, "bigram cache save failed");
    }
}

pub(crate) struct Loaded {
    pub filter: BigramFilter,
    pub masks: Option<(Vec<u64>, bool)>,
    pub binary: Vec<u64>,
    pub fingerprint: [u8; 32],
}

// Corrupt input degrades to None (caller rebuilds), never panics.
pub(crate) fn load(path: &std::path::Path) -> Option<Loaded> {
    let data = std::fs::read(path).ok()?;
    if data.len() < 8 + 4 + 32 + 32 {
        return None;
    }
    let (body, sum) = data.split_at(data.len() - 32);
    if blake3::hash(body).as_bytes() != sum {
        return None;
    }
    let mut r = Reader { b: body, pos: 0 };
    if r.u64()? != MAGIC || r.bytes(4)? != CODEC.to_le_bytes() {
        return None;
    }
    let fingerprint = r.hash32()?;
    let filter = read_filter(&mut r, 0)?;
    let masks = match r.u8()? {
        0 => None,
        1 => {
            let slots = r.vec_u64()?;
            let has_data = r.u8()? != 0;
            Some((slots, has_data))
        }
        _ => return None,
    };
    let binary = r.vec_u64()?;
    if r.pos != body.len() {
        return None;
    }
    Some(Loaded {
        filter,
        masks,
        binary,
        fingerprint,
    })
}

// Load-and-publish when the corpus matches a previous scan. Restores
// masks and binary flags; watcher overlay starts empty as usual.
// Returns true on a usable hit (caller skips build + sniff).
pub(crate) fn try_load(
    files: &[FileItem],
    base_path: &std::path::Path,
    arena: ArenaPtr,
    content_masks: Option<&std::sync::Arc<crate::file_picker::ContentMaskStore>>,
    set_index: impl FnOnce(BigramFilter),
) -> bool {
    let Some(path) = cache_path(base_path) else {
        return false;
    };
    let Some(loaded) = load(&path) else {
        return false;
    };
    if loaded.fingerprint != fingerprint(files, base_path, arena) {
        return false;
    }
    if loaded.filter.file_count() != files.len() {
        return false;
    }
    if let (Some(store), Some((slots, _))) = (content_masks, loaded.masks) {
        if slots.len() == files.len() {
            for (i, v) in slots.iter().enumerate() {
                if *v != 0 {
                    store.set(i, crate::types::ContentMask::from_raw(*v));
                }
            }
        }
    }
    for (i, chunk) in loaded.binary.iter().enumerate() {
        let mut word = *chunk;
        while word != 0 {
            let b = word.trailing_zeros() as usize;
            if let Some(f) = files.get(i * 64 + b) {
                f.set_binary(true);
            }
            word &= word - 1;
        }
    }
    set_index(loaded.filter);
    tracing::debug!(path = ?path.display(), "bigram cache hit");
    true
}

// Snapshot masks, binary flags, and fingerprint after a fresh build.
pub(crate) fn save_after(
    files: &[FileItem],
    base_path: &std::path::Path,
    arena: ArenaPtr,
    filter: &BigramFilter,
    content_masks: Option<&std::sync::Arc<crate::file_picker::ContentMaskStore>>,
) {
    let Some(path) = cache_path(base_path) else {
        return;
    };
    let masks = content_masks.and_then(|s| s.snapshot());
    let mut binary = vec![0u64; files.len().div_ceil(64)];
    for (i, f) in files.iter().enumerate() {
        if f.is_binary() {
            binary[i / 64] |= 1u64 << (i % 64);
        }
    }
    let fp = fingerprint(files, base_path, arena);
    save(filter, masks, &binary, &fp, &path);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::bigram_filter::BigramIndexBuilder;

    fn tiny_filter() -> BigramFilter {
        let consec = BigramIndexBuilder::new(3);
        let skip = BigramIndexBuilder::new(3);
        consec.add_file_content(&skip, 0, b"hello world");
        consec.add_file_content(&skip, 1, b"foo bar baz");
        consec.add_file_content(&skip, 2, b"neovim fast finder");
        let mut index = consec.compress(None);
        index.set_skip_index(skip.compress(Some(12)));
        index
    }

    fn dir() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn roundtrip_preserves_queries() {
        let index = tiny_filter();
        let dir = dir();
        let path = dir.path().join("c.bin");
        let fp = [7u8; 32];
        save(&index, Some((vec![1, 2, 3], true)), &[0b101], &fp, &path);
        let back = load(&path).expect("roundtrip loads");
        assert_eq!(back.fingerprint, fp);
        assert_eq!(back.masks, Some((vec![1, 2, 3], true)));
        assert_eq!(back.binary, vec![0b101]);
        for pat in [b"hello".as_slice(), b"foo", b"finder", b"vim", b"zzz"] {
            assert_eq!(
                index.query(pat),
                back.filter.query(pat),
                "query differs for {pat:?}"
            );
        }
    }

    #[test]
    fn corrupt_or_truncated_is_none() {
        let index = tiny_filter();
        let dir = dir();
        let path = dir.path().join("c.bin");
        save(&index, None, &[], &[9u8; 32], &path);
        let mut data = std::fs::read(&path).unwrap();
        for cut in [10usize, 100, data.len() - 33, data.len() - 1] {
            std::fs::write(&path, &data[..cut.min(data.len())]).unwrap();
            assert!(load(&path).is_none(), "truncated at {cut} loaded");
        }
        data[20] ^= 0xFF;
        std::fs::write(&path, &data).unwrap();
        assert!(load(&path).is_none(), "flipped byte loaded");
        data[8] ^= 0xFF;
        std::fs::write(&path, &data).unwrap();
        assert!(load(&path).is_none(), "bad codec loaded");
    }
}
