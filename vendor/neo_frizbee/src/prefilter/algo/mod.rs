mod ascii;
mod ascii_typos;
mod load;
mod unicode;
mod unicode_typos;

#[cfg(target_arch = "x86_64")]
pub(crate) use ascii::find_last_char_pos;
pub(crate) use load::{can_overread, load_window};

use super::{UnicodeChar, backend::Backend, case_needle, case_needle_unicode};
use alloc::vec::Vec;

#[derive(Debug, Clone, Copy)]
pub(crate) struct PathState<M> {
    pub needle_idx: usize,
    pub needle_mask: M,
}

#[derive(Debug)]
pub(crate) struct Prefilter<B: Backend> {
    needle_ascii: Vec<(B::Chunk, B::Chunk)>,
    needle_unicode: Vec<UnicodeChar>,
    paths: Vec<PathState<B::Mask>>,
    /// Case-folded needle bytes for the Myers LCS gate (≤64 chars).
    needle_folded: Vec<u8>,
    /// Cached PEQ table for LCS bit-parallel (2 KiB). Rebuilt only on new().
    lcs_peq: Option<Box<[u64; 256]>>,
    /// Adaptive LCS gate: disabled when the first samples show a high pass
    /// rate (bit-parallel LCS then costs more than the SIMD walker).
    lcs_disabled: std::sync::atomic::AtomicBool,
    lcs_samples: std::sync::atomic::AtomicU32,
    lcs_rejects: std::sync::atomic::AtomicU32,
}

impl<B: Backend> Clone for Prefilter<B> {
    fn clone(&self) -> Self {
        Self {
            needle_ascii: self.needle_ascii.clone(),
            needle_unicode: self.needle_unicode.clone(),
            paths: self.paths.clone(),
            needle_folded: self.needle_folded.clone(),
            lcs_peq: self.lcs_peq.clone(),
            lcs_disabled: std::sync::atomic::AtomicBool::new(
                self.lcs_disabled.load(std::sync::atomic::Ordering::Relaxed),
            ),
            lcs_samples: std::sync::atomic::AtomicU32::new(
                self.lcs_samples.load(std::sync::atomic::Ordering::Relaxed),
            ),
            lcs_rejects: std::sync::atomic::AtomicU32::new(
                self.lcs_rejects.load(std::sync::atomic::Ordering::Relaxed),
            ),
        }
    }
}

impl<B: Backend> Prefilter<B> {
    /// # Safety
    /// The backend's target features must be enabled.
    #[inline(always)]
    pub unsafe fn new(needle: &str, case_sensitive: bool) -> Self {
        let needle_ascii = case_needle(needle.as_bytes(), case_sensitive)
            .into_iter()
            .map(|(c1, c2)| unsafe { (B::splat(c1), B::splat(c2)) })
            .collect();
        let needle_unicode = case_needle_unicode(needle, case_sensitive);
        let needle_folded = if case_sensitive {
            needle.as_bytes().to_vec()
        } else {
            needle.bytes().map(crate::myers::fold).collect()
        };

        let lcs_peq = if needle_folded.len() <= crate::myers::MAX_NEEDLE_BITS {
            Some(Box::new(crate::myers::peq_table(&needle_folded)))
        } else {
            None
        };

        Self {
            needle_ascii,
            needle_unicode,
            paths: Vec::new(),
            needle_folded,
            lcs_peq,
            lcs_disabled: std::sync::atomic::AtomicBool::new(false),
            lcs_samples: std::sync::atomic::AtomicU32::new(0),
            lcs_rejects: std::sync::atomic::AtomicU32::new(0),
        }
    }

    /// True when the adaptive LCS gate should run for this haystack.
    #[inline]
    pub(crate) fn lcs_gate_active(&self) -> bool {
        !self.lcs_disabled.load(std::sync::atomic::Ordering::Relaxed)
    }

    #[inline]
    pub(crate) fn note_lcs_sample(&self, rejected: bool) {
        if self.lcs_disabled.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        self.lcs_samples
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if rejected {
            self.lcs_rejects
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let n = self.lcs_samples.load(std::sync::atomic::Ordering::Relaxed);
        if n == 512 {
            let r = self.lcs_rejects.load(std::sync::atomic::Ordering::Relaxed);
            // Keep LCS only when it rejects ≥90% of samples (long/selective
            // queries). At 77% reject (`file_picker_`) the bit-parallel DP
            // still costs more than the SIMD walker + SW on the survivors.
            if r.saturating_mul(10) < n.saturating_mul(9) {
                self.lcs_disabled
                    .store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    #[inline]
    pub(crate) fn lcs_peq(&self) -> Option<&[u64; 256]> {
        self.lcs_peq.as_deref()
    }

    #[inline]
    pub(crate) fn needle_folded(&self) -> &[u8] {
        &self.needle_folded
    }

    #[inline(always)]
    unsafe fn needle_unchecked(&self, idx: usize) -> (B::Chunk, B::Chunk) {
        unsafe { *self.needle_ascii.get_unchecked(idx) }
    }

    #[inline(always)]
    unsafe fn unicode_needle_unchecked(&self, idx: usize) -> &UnicodeChar {
        unsafe { self.needle_unicode.get_unchecked(idx) }
    }
}
