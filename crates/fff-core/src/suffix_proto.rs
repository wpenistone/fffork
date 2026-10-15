use crate::simd_path::ArenaPtr;
use crate::stable_vec::StableVec;
use crate::types::FileItem;

// Flat filename table for the gate suffix walk: filenames contiguous in
// index order, built once per scan. The walk reads linearly instead of
// chasing arena chunks. Same bytes in path order, so results are
// identical; the table only changes how they are fetched.
// Keyed by element address against the shared files allocation (which
// never moves), so every view type resolves correctly or falls back.
pub(crate) struct SuffixProto {
    flat: Vec<u8>,
    off: Vec<u32>,
    base: usize,
    base_count: usize,
    files: StableVec<FileItem>,
    generation: u64,
}

impl SuffixProto {
    pub(crate) fn build(
        files: &StableVec<FileItem>,
        arena: ArenaPtr,
        base_count: usize,
        generation: u64,
    ) -> Self {
        let n = files.len().min(base_count);
        let mut flat = Vec::new();
        let mut off = Vec::with_capacity(n);
        for f in files.iter().take(n) {
            let split = (f.path.filename_offset as usize).min(f.path.byte_len as usize);
            let len = f.path.byte_len as usize - split;
            off.push(flat.len() as u32);
            let start = flat.len();
            flat.resize(start + len, 0);
            if len > 0 {
                f.path.copy_window_to(
                    arena,
                    split as u32,
                    (split + len) as u32,
                    &mut flat[start..],
                );
            }
        }
        let base = files
            .first()
            .map(|f| f as *const FileItem as usize)
            .unwrap_or(0);
        Self {
            flat,
            off,
            base,
            base_count: n,
            files: files.clone(),
            generation,
        }
    }

    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    // Filename bytes for this file, or `None` to walk the arena. The
    // address check keeps every view correct: overflow and foreign files
    // miss the range and fall back.
    pub(crate) fn suffix_for(&self, file: &FileItem) -> Option<&[u8]> {
        let addr = file as *const FileItem as usize;
        let size = std::mem::size_of::<FileItem>();
        if addr < self.base {
            return None;
        }
        let k = addr.wrapping_sub(self.base) / size;
        if addr != self.base.wrapping_add(k.wrapping_mul(size)) {
            return None;
        }
        if k >= self.base_count {
            return None;
        }
        let split = (file.path.filename_offset as usize).min(file.path.byte_len as usize);
        let len = file.path.byte_len as usize - split;
        let o = *self.off.get(k)? as usize;
        self.flat.get(o..o.checked_add(len)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd_path::build_chunked_path_store_from_strings;
    use crate::stable_vec::StableVec;

    fn fixture(rel: &[&str]) -> (StableVec<FileItem>, crate::simd_path::ChunkedPathStore) {
        let files = vec![FileItem::new_raw(0, 0, 0, None, false); rel.len()];
        let owned: Vec<String> = rel.iter().map(|s| s.to_string()).collect();
        let (store, strings) = build_chunked_path_store_from_strings(&owned, &files);
        let mut vec: Vec<FileItem> = files.into_iter().collect();
        for ((file, s), r) in vec.iter_mut().zip(strings.iter()).zip(owned.iter()) {
            file.set_path(s.clone());
            file.path.filename_offset =
                (r.rfind('/').map(|i| i + 1).unwrap_or(0) as u16).min(file.path.byte_len);
        }
        (StableVec::from_vec_with_reserve(vec, 0), store)
    }

    #[test]
    fn suffix_bytes_match_path_tails() {
        let rel = [
            "src/alpha/one.rs",
            "src/alpha/two.rs",
            "x",
            "deep/nested/dir/with/many/parts/seven.rs",
        ];
        let (stable, store) = fixture(&rel);
        let arena = store.as_arena_ptr();
        let proto = SuffixProto::build(&stable, arena, stable.len(), 3);
        let files: &[FileItem] = &stable;
        for (file, r) in files.iter().zip(rel.iter()) {
            let split = r.rfind('/').map(|i| i + 1).unwrap_or(0);
            assert_eq!(
                proto.suffix_for(file),
                Some(r.as_bytes()[split..].as_ref()),
                "tail mismatch for {r:?}"
            );
        }
    }
}
