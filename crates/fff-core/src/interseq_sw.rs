// Frizbee defaults with fff overrides for lowercase queries.
#[derive(Clone, Copy)]
pub(crate) struct SwParams {
    pub match_score: u16,
    pub mismatch_penalty: u16,
    pub gap_open: u16,
    pub gap_extend: u16,
    pub prefix_bonus: u16,
    pub delimiter_bonus: u16,
    pub capitalization_bonus: u16,
    pub matching_case_bonus: u16,
}

impl SwParams {
    pub fn for_query(has_uppercase: bool) -> Self {
        Self {
            match_score: 12,
            mismatch_penalty: 6,
            gap_open: 5,
            gap_extend: 1,
            prefix_bonus: 12,
            delimiter_bonus: 4,
            capitalization_bonus: if has_uppercase { 8 } else { 0 },
            matching_case_bonus: if has_uppercase { 4 } else { 0 },
        }
    }
    pub fn u8_safe(&self, m: usize) -> bool {
        let per = self.match_score as u32
            + self.delimiter_bonus as u32
            + self.capitalization_bonus as u32
            + self.matching_case_bonus as u32;
        (m as u32)
            .saturating_mul(per)
            .saturating_add(self.prefix_bonus as u32)
            < 256
    }

    // Sound upper bound: one diagonal per column at most, so rows add
    // at most `per` each, plus prefix and exact once.
    #[allow(dead_code)]
    pub(crate) fn max_sw_score(&self, m: usize) -> i32 {
        let per = self.match_score as i32
            + self.delimiter_bonus as i32
            + self.capitalization_bonus as i32
            + self.matching_case_bonus as i32;
        (m as i32) * per + self.prefix_bonus as i32 + 8
    }
}

const CLASS_LOWER: u8 = 1;
const CLASS_UPPER: u8 = 2;
const CLASS_DELIM: u8 = 4;

// Low byte is the class. High byte is ASCII `A..=Z` folded to lowercase.
const BYTE_INFO: [u16; 256] = {
    let mut t = [0u16; 256];
    let mut i = 0u16;
    while i < 256 {
        let b = i as u8;
        let lower = b >= b'a' && b <= b'z';
        let upper = b >= b'A' && b <= b'Z';
        let digit = b >= b'0' && b <= b'9';
        let delim = b < 128 && !lower && !upper && !digit;
        let class = (lower as u8) | ((upper as u8) << 1) | ((delim as u8) << 2);
        let fold = if upper { b + 32 } else { b };
        t[i as usize] = class as u16 | ((fold as u16) << 8);
        i += 1;
    }
    t
};

#[inline(always)]
fn flip_case(b: u8) -> u8 {
    (BYTE_INFO[b as usize] >> 8) as u8
}

struct FlipCache {
    len: u16,
    src: [u8; 256],
    dst: [u8; 256],
}

// Same needle on the next window reuses the flipped bytes. `out` holds `needle`.
fn load_flipped(needle: &[u8], out: &mut [u8]) {
    debug_assert!(needle.len() <= 256 && out.len() >= needle.len());
    thread_local! {
        static CACHE: std::cell::RefCell<FlipCache> = const {
            std::cell::RefCell::new(FlipCache {
                len: 0,
                src: [0; 256],
                dst: [0; 256],
            })
        };
    }
    CACHE.with(|slot| {
        let cache = &mut *slot.borrow_mut();
        let n = needle.len();
        if cache.len as usize == n && &cache.src[..n] == needle {
            out[..n].copy_from_slice(&cache.dst[..n]);
            return;
        }
        cache.len = n as u16;
        cache.src[..n].copy_from_slice(needle);
        for (i, &b) in needle.iter().enumerate() {
            let flipped = flip_case(b);
            cache.dst[i] = flipped;
            out[i] = flipped;
        }
    });
}

// Per-worker DP rows for needles longer than 32. Match flags are bits.
// Row 0 starts zero; a column writes one row and reads the other.
#[derive(Default)]
struct LongScratch {
    h: [Vec<i32>; 2],
    e: [Vec<i32>; 2],
    matched: [Vec<u64>; 2],
}

impl LongScratch {
    fn ensure(&mut self, n: usize) {
        for row in [&mut self.h, &mut self.e] {
            row[0].resize(n, 0);
            row[1].resize(n, 0);
            row[0].fill(0);
        }
        // `n` is m+1 bits. Only the previous row is cleared.
        let words = n.div_ceil(64);
        self.matched[0].resize(words, 0);
        self.matched[1].resize(words, 0);
        self.matched[0].fill(0);
    }
}

thread_local! {
    static LONG_SCRATCH: std::cell::RefCell<LongScratch> =
        const { std::cell::RefCell::new(LongScratch {
            h: [Vec::new(), Vec::new()],
            e: [Vec::new(), Vec::new()],
            matched: [Vec::new(), Vec::new()],
        }) };
}

// Reference SW on a TRIMMED haystack; `include_prefix` iff trimmed start == 0.
// Returns (score, end_col_local): last-row max and its LAST column.
// No exact_match_bonus; caller adds it when `include_exact && needle == trimmed`.
pub(crate) fn score_one_scalar_full(
    needle: &[u8],
    haystack: &[u8],
    p: &SwParams,
    include_prefix: bool,
) -> (u16, u16) {
    let m = needle.len();
    if m == 0 || haystack.is_empty() {
        return (0, 0);
    }
    if m <= 256 {
        let mut buf = [0u8; 256];
        load_flipped(needle, &mut buf);
        return score_one_scalar_flipped(needle, &buf[..m], haystack, p, include_prefix);
    }
    let flipped: Vec<u8> = needle.iter().copied().map(flip_case).collect();
    score_one_scalar_flipped(needle, &flipped, haystack, p, include_prefix)
}

// `flipped` is the ASCII-lowercase needle, same length. The batch folds it once.
pub(crate) fn score_one_scalar_flipped(
    needle: &[u8],
    flipped: &[u8],
    haystack: &[u8],
    p: &SwParams,
    include_prefix: bool,
) -> (u16, u16) {
    debug_assert_eq!(needle.len(), flipped.len());
    if needle.is_empty() || haystack.is_empty() {
        return (0, 0);
    }
    if needle.len() <= 32 {
        score_one_scalar_stack(needle, flipped, haystack, p, include_prefix)
    } else {
        score_one_scalar_vec(needle, flipped, haystack, p, include_prefix)
    }
}

// Needle is non-empty, at most 32, and `flipped` matches it.
#[inline(always)]
pub(crate) fn score_one_scalar_short(
    needle: &[u8],
    flipped: &[u8],
    haystack: &[u8],
    p: &SwParams,
    include_prefix: bool,
) -> (u16, u16) {
    if haystack.is_empty() {
        return (0, 0);
    }
    score_one_scalar_stack(needle, flipped, haystack, p, include_prefix)
}

// Writes the flipped needle when it fits in 256 bytes.
pub(crate) fn prepare_scalar_flip(needle: &[u8], out: &mut [u8; 256]) -> bool {
    if needle.len() > 256 {
        return false;
    }
    load_flipped(needle, out);
    true
}

fn score_one_scalar_vec(
    needle: &[u8],
    flipped: &[u8],
    haystack: &[u8],
    p: &SwParams,
    include_prefix: bool,
) -> (u16, u16) {
    let m = needle.len();
    let n = haystack.len();
    debug_assert_eq!(flipped.len(), m);
    // Use i32 DP values to avoid saturation artifacts during computation.
    let gap_open = p.gap_open as i32;
    let gap_extend = p.gap_extend as i32;
    let gap_open_only = gap_open; // full open penalty when transitioning match->gap
    let open_extra = gap_open_only - gap_extend;
    let match_add = p.match_score as i32;
    let mismatch_pen = p.mismatch_penalty as i32;
    let case_bonus = p.matching_case_bonus as i32;
    let want_case = case_bonus != 0;
    let cap_unit = p.capitalization_bonus as i32;
    let use_cap = cap_unit != 0;
    LONG_SCRATCH.with(|c| {
        let mut scratch = c.borrow_mut();
        scratch.ensure(m + 1);
        let mut prev_i = 0usize;
        let mut prev_is_lower = false;
        let mut prev_is_delim = false;
        let mut last_row_max = 0i32;
        let mut end_col = 0u16;
        let needle_p = needle.as_ptr();
        let folded_p = flipped.as_ptr();

        if n > 0 {
            let mut j = 0usize;
            let mut hb = haystack[0];
            let mut info = BYTE_INFO[hb as usize];
            loop {
            let curr_i = prev_i ^ 1;
            // The next column's class does not depend on this column's cells.
            let col_more = j + 1 < n;
            let hb_next = if col_more { haystack[j + 1] } else { 0 };
            let info_next = if col_more { BYTE_INFO[hb_next as usize] } else { 0 };
            let class = info as u8;
            let hb_fold = (info >> 8) as u8;
            let hb_upper = class & CLASS_UPPER != 0;
            let hb_lower = class & CLASS_LOWER != 0;
            let hb_delim = class & CLASS_DELIM != 0;

            let cap_bonus = column_bonus(
                use_cap,
                hb_upper,
                prev_is_lower,
                cap_unit,
                prev_is_delim,
                hb_delim,
                p.delimiter_bonus as i32,
                0i32.wrapping_sub((include_prefix as i32) & ((j == 0) as i32))
                    & (p.prefix_bonus as i32),
            );
            let match_base = match_add + cap_bonus;
            let match_lift = match_base + mismatch_pen;

            // Rows hold `m + 1` cells. The cell loop adds the column index only.
            let hp = scratch.h[prev_i].as_ptr();
            let hc = scratch.h[curr_i].as_mut_ptr();
            let ep = scratch.e[prev_i].as_ptr();
            let ec = scratch.e[curr_i].as_mut_ptr();
            let mp = scratch.matched[prev_i].as_ptr();
            let mc = scratch.matched[curr_i].as_mut_ptr();
            unsafe {
                // Column 0 reads a zero row, so the diagonal is the cell score.
                if j == 0 {
                    let neg_gap = -gap_extend;
                    let mut f_prev = 0i32;
                    let mut f_was_match = false;
                    let mut h_col = 0i32;
                    let mut i = 1usize;
                    let mut word = 0usize;
                    let mut first_cell = true;
                    let mut fp = folded_p;
                    let mut np = needle_p;
                    let mut hc_i = hc.add(1);
                    let mut ec_i = ec.add(1);
                    while i <= m {
                        let mut bit_mask = if word == 0 { 2u64 } else { 1u64 };
                        let mut curr_bits = 0u64;
                        let limit = if word == 0 {
                            63.min(m)
                        } else {
                            ((word + 1) * 64 - 1).min(m)
                        };
                        let mut lower_nc = *fp;
                        let mut nb_cur = if want_case { *np } else { 0 };
                        loop {
                            // The next needle byte does not depend on this cell.
                            let more = i < limit;
                            if more {
                                fp = fp.add(1);
                                if want_case {
                                    np = np.add(1);
                                }
                            }
                            let lower_next = if more { *fp } else { 0 };
                            let nb_next = if more && want_case { *np } else { 0 };
                            // The folded compare does not read the gap.
                            let same_case = want_case && hb == nb_cur;
                            let is_match = hb_fold == lower_nc;
                            // The horizontal gap does not read this row's match.
                            let f_hit = f_was_match as i32;
                            let f_new = h_col - (gap_extend + (0i32.wrapping_sub(f_hit) & open_extra));
                            let f_here = f_new.max(f_prev - gap_extend);
                            let is_exact = is_match && same_case;
                            curr_bits |= bit_mask & (0u64.wrapping_sub(is_match as u64));
                            let s = -mismatch_pen
                                + (0i32.wrapping_sub(is_match as i32) & match_lift)
                                + (0i32.wrapping_sub(is_exact as i32) & case_bonus);
                            *ec_i = neg_gap;
                            let hv = if first_cell {
                                first_cell = false;
                                s.max(0)
                            } else {
                                f_prev = f_here;
                                s.max(f_here).max(0)
                            };
                            *hc_i = hv;
                            h_col = hv;
                            f_was_match = is_match;
                            bit_mask <<= 1;
                            i += 1;
                            hc_i = hc_i.add(1);
                            ec_i = ec_i.add(1);
                            if !more {
                                fp = fp.add(1);
                                if want_case {
                                    np = np.add(1);
                                }
                                break;
                            }
                            lower_nc = lower_next;
                            nb_cur = nb_next;
                        }
                        *mc.add(word) = curr_bits;
                        word += 1;
                    }
                    keep_best(h_col, &mut last_row_max, &mut end_col, j);
                } else {
                // F lives in this column only. Cell 0 is never read.
                let mut f_prev = 0i32;
                // Each 64-bit word is its own run. Cell 0 of the first word stays clear.
                let mut f_was_match;
                let mut h_col;
                let mut i;
                let mut word = 0usize;
                let mut fp = folded_p;
                let mut np = needle_p;
                let mut hp_i = hp.add(1);
                let mut ep_i = ep.add(1);
                let mut hc_i = hc.add(1);
                let mut ec_i = ec.add(1);
                // The carry into cell 1 is the zero boundary, so the diagonal is the cell score.
                let prev_bits0 = *mp;
                let lower_nc0 = *fp;
                fp = fp.add(1);
                let h_here0 = *hp_i;
                // The vertical gap does not read the needle compare.
                let e_hit0 = ((prev_bits0 & 2) != 0) as i32;
                let e_here0 = (h_here0 - (gap_extend + (0i32.wrapping_sub(e_hit0) & open_extra))).max(*ep_i - gap_extend);
                let is_match0 = hb_fold == lower_nc0;
                let is_exact0 = if want_case {
                    let nb0 = *np;
                    np = np.add(1);
                    is_match0 && hb == nb0
                } else {
                    false
                };
                let s0 = -mismatch_pen
                    + (0i32.wrapping_sub(is_match0 as i32) & match_lift)
                    + (0i32.wrapping_sub(is_exact0 as i32) & case_bonus);
                *ec_i = e_here0;
                h_col = s0.max(e_here0).max(0);
                *hc_i = h_col;
                f_was_match = is_match0;
                let mut h_carry = h_here0;
                let first_bits = 2u64 & (0u64.wrapping_sub(is_match0 as u64));
                i = 2;
                hp_i = hp_i.add(1);
                ep_i = ep_i.add(1);
                hc_i = hc_i.add(1);
                ec_i = ec_i.add(1);
                while i <= m {
                    let mut bit_mask = if word == 0 { 4u64 } else { 1u64 };
                    let mut curr_bits = if word == 0 { first_bits } else { 0 };
                    let prev_bits = *mp.add(word);
                    let limit = if word == 0 {
                        63.min(m)
                    } else {
                        ((word + 1) * 64 - 1).min(m)
                    };
                    let mut lower_nc = *fp;
                    let mut h_here = *hp_i;
                    let mut e_prev = *ep_i;
                    let mut nb_cur = if want_case { *np } else { 0 };
                    loop {
                        // The next cell's bytes do not depend on this recurrence.
                        let more = i < limit;
                        if more {
                            fp = fp.add(1);
                            hp_i = hp_i.add(1);
                            ep_i = ep_i.add(1);
                            if want_case {
                                np = np.add(1);
                            }
                        }
                        let lower_next = if more { *fp } else { 0 };
                        let h_next = if more { *hp_i } else { 0 };
                        let e_next = if more { *ep_i } else { 0 };
                        let nb_next = if more && want_case { *np } else { 0 };
                        // The folded compare does not read the gap.
                        let same_case = want_case && hb == nb_cur;
                        let is_match = hb_fold == lower_nc;
                        let diag_base = h_carry - mismatch_pen;
                        // Both gaps read the previous column and the previous row.
                        let e_hit = ((prev_bits & bit_mask) != 0) as i32;
                        let e_new = h_here - (gap_extend + (0i32.wrapping_sub(e_hit) & open_extra));
                        let e_ext = e_prev - gap_extend;
                        let e_here = e_new.max(e_ext);
                        let f_hit = f_was_match as i32;
                        let f_new = h_col - (gap_extend + (0i32.wrapping_sub(f_hit) & open_extra));
                        let f_here = f_new.max(f_prev - gap_extend);
                        let is_exact = is_match && same_case;
                        curr_bits |= bit_mask & (0u64.wrapping_sub(is_match as u64));
                        let diag = diag_base
                            + (0i32.wrapping_sub(is_match as i32) & match_lift)
                            + (0i32.wrapping_sub(is_exact as i32) & case_bonus);
                        h_carry = h_here;
                        *ec_i = e_here;
                        f_prev = f_here;
                        let hv = diag.max(e_here).max(f_here).max(0);
                        *hc_i = hv;
                        h_col = hv;
                        f_was_match = is_match;
                        bit_mask <<= 1;
                        i += 1;
                        hc_i = hc_i.add(1);
                        ec_i = ec_i.add(1);
                        if !more {
                            fp = fp.add(1);
                            hp_i = hp_i.add(1);
                            ep_i = ep_i.add(1);
                            if want_case {
                                np = np.add(1);
                            }
                            break;
                        }
                        lower_nc = lower_next;
                        h_here = h_next;
                        e_prev = e_next;
                        nb_cur = nb_next;
                    }
                    *mc.add(word) = curr_bits;
                    word += 1;
                }
                keep_best(h_col, &mut last_row_max, &mut end_col, j);
                }
            }

            prev_i = curr_i;
            prev_is_lower = hb_lower;
            prev_is_delim = hb_delim;
            if !col_more {
                break;
            }
            hb = hb_next;
            info = info_next;
            j += 1;
            }
        }

        (last_row_max as u16, end_col)
    })
}

// A higher column replaces the running best. An equal score keeps the earlier column.
#[inline(always)]
fn keep_best(h_col: i32, last_row_max: &mut i32, end_col: &mut u16, j: usize) {
    let better = (h_col > *last_row_max) as i32;
    let pick = 0i32.wrapping_sub(better);
    *last_row_max = last_row_max.wrapping_add(h_col.wrapping_sub(*last_row_max) & pick);
    let b = 0u16.wrapping_sub(better as u16);
    *end_col = (*end_col & !b) | ((j as u16) & b);
}

// Cap and delimiter are 0 or the unit. A lowercase query skips the cap test.
#[inline(always)]
fn column_bonus(
    use_cap: bool,
    upper: bool,
    prev_lower: bool,
    cap_unit: i32,
    prev_delim: bool,
    delim: bool,
    delim_unit: i32,
    prefix: i32,
) -> i32 {
    // A lowercase query skips this. A capital query has no branch on the previous letter.
    let cap = if use_cap {
        // Both flags set is all-bits, so the unit is an and.
        0i32.wrapping_sub((upper as i32) & (prev_lower as i32)) & cap_unit
    } else {
        0
    };
    let at_boundary = 0i32.wrapping_sub((prev_delim as i32) & (!delim as i32)) & delim_unit;
    cap + at_boundary + prefix
}

fn score_one_scalar_stack(
    needle: &[u8],
    flipped: &[u8],
    haystack: &[u8],
    p: &SwParams,
    include_prefix: bool,
) -> (u16, u16) {
    // Case and capital bonuses are fixed for the query. The column does not test them.
    if p.matching_case_bonus != 0 {
        if p.capitalization_bonus != 0 {
            stack_dp::<true, true>(needle, flipped, haystack, p, include_prefix)
        } else {
            stack_dp::<true, false>(needle, flipped, haystack, p, include_prefix)
        }
    } else if p.capitalization_bonus != 0 {
        stack_dp::<false, true>(needle, flipped, haystack, p, include_prefix)
    } else {
        stack_dp::<false, false>(needle, flipped, haystack, p, include_prefix)
    }
}

fn stack_dp<const WANT_CASE: bool, const USE_CAP: bool>(
    needle: &[u8],
    flipped: &[u8],
    haystack: &[u8],
    p: &SwParams,
    include_prefix: bool,
) -> (u16, u16) {
    let m = needle.len();
    let n = haystack.len();
    debug_assert!(m <= 32 && flipped.len() == m);
    // `i` stays inside the 33-cell rows. Release drops the index checks.
    unsafe { std::hint::assert_unchecked(m <= 32) };
    let gap_open = p.gap_open as i32;
    let gap_extend = p.gap_extend as i32;
    let gap_open_only = gap_open;
    let open_extra = gap_open_only - gap_extend;
    let match_add = p.match_score as i32;
    let mismatch_pen = p.mismatch_penalty as i32;
    let case_bonus = p.matching_case_bonus as i32;
    let cap_unit = p.capitalization_bonus as i32;
    // Only cells the first column reads start at zero. The other row is written first.
    // Match flag i is bit i. Needles here are at most 32, so one word holds the row.
    let mut h = [std::mem::MaybeUninit::<[i32; 33]>::uninit(); 2];
    let mut e = [std::mem::MaybeUninit::<[i32; 33]>::uninit(); 2];
    unsafe {
        std::ptr::write_bytes((h[0].as_mut_ptr() as *mut i32).add(1), 0, m);
        std::ptr::write_bytes((e[0].as_mut_ptr() as *mut i32).add(1), 0, m);
    }
    let mut matched = [0u64; 2];
    let mut prev_i = 0usize;
    let mut prev_is_lower = false;
    let mut prev_is_delim = false;
    let mut last_row_max = 0i32;
    let mut end_col = 0u16;
    let needle_p = needle.as_ptr();
    let folded_p = flipped.as_ptr();
    // Column 0 reads a zero row: the diagonal is the cell score and E is `-gap_extend`.
    if n > 0 {
        let curr_i = 1usize;
        let hb = haystack[0];
        let info = BYTE_INFO[hb as usize];
        let class = info as u8;
        let hb_fold = (info >> 8) as u8;
        let hb_delim = class & CLASS_DELIM != 0;
        let hb_lower = USE_CAP && class & CLASS_LOWER != 0;
        let cap_bonus = column_bonus(
            USE_CAP,
            USE_CAP && class & CLASS_UPPER != 0,
            false,
            cap_unit,
            false,
            hb_delim,
            p.delimiter_bonus as i32,
            if include_prefix { p.prefix_bonus as i32 } else { 0 },
        );
        let match_base = match_add + cap_bonus;
        let match_lift = match_base + mismatch_pen;
        let hc = h[curr_i].as_mut_ptr() as *mut i32;
        let ec = e[curr_i].as_mut_ptr() as *mut i32;
        let neg_gap = -gap_extend;
        unsafe {
            let mut curr_bits = 0u64;
            let mut bit_mask = 2u64;
            let mut fp = folded_p;
            let mut np = needle_p;
            let mut hc_i = hc.add(1);
            let mut ec_i = ec.add(1);
            let is_match0 = hb_fold == *fp;
            fp = fp.add(1);
            // The raw needle byte is only the case bonus.
            let is_exact0 = if WANT_CASE {
                let nb = *np;
                np = np.add(1);
                is_match0 && hb == nb
            } else {
                false
            };
            curr_bits |= bit_mask & (0u64.wrapping_sub(is_match0 as u64));
            let s0 = -mismatch_pen
                + (0i32.wrapping_sub(is_match0 as i32) & match_lift)
                + (0i32.wrapping_sub(is_exact0 as i32) & case_bonus);
            *ec_i = neg_gap;
            let mut h_col = s0.max(0);
            *hc_i = h_col;
            let mut f_was_match = is_match0;
            let mut f_prev = 0i32;
            bit_mask <<= 1;
            hc_i = hc_i.add(1);
            ec_i = ec_i.add(1);
            if m >= 2 {
                let mut i = 2usize;
                let mut lower_nc = *fp;
                let mut nb_cur = if WANT_CASE { *np } else { 0 };
                loop {
                    // The next needle byte does not depend on this cell.
                    let more = i < m;
                    if more {
                        fp = fp.add(1);
                        if WANT_CASE {
                            np = np.add(1);
                        }
                    }
                    let lower_next = if more { *fp } else { 0 };
                    let nb_next = if more && WANT_CASE { *np } else { 0 };
                    // The folded compare does not read the gap.
                    let same_case = if WANT_CASE { hb == nb_cur } else { false };
                    let is_match = hb_fold == lower_nc;
                    // The horizontal gap does not read this row's match.
                    let f_hit = f_was_match as i32;
                    let f_new = h_col - (gap_extend + (0i32.wrapping_sub(f_hit) & open_extra));
                    let f_here = f_new.max(f_prev - gap_extend);
                    let is_exact = WANT_CASE && is_match && same_case;
                    curr_bits |= bit_mask & (0u64.wrapping_sub(is_match as u64));
                    let s = -mismatch_pen
                        + (0i32.wrapping_sub(is_match as i32) & match_lift)
                        + (0i32.wrapping_sub(is_exact as i32) & case_bonus);
                    *ec_i = neg_gap;
                    f_prev = f_here;
                    h_col = s.max(f_here).max(0);
                    *hc_i = h_col;
                    f_was_match = is_match;
                    bit_mask <<= 1;
                    hc_i = hc_i.add(1);
                    ec_i = ec_i.add(1);
                    if !more {
                        break;
                    }
                    lower_nc = lower_next;
                    nb_cur = nb_next;
                    i += 1;
                }
            }
            matched[curr_i] = curr_bits;
            keep_best(h_col, &mut last_row_max, &mut end_col, 0);
        }
        prev_i = curr_i;
        if USE_CAP {
            prev_is_lower = hb_lower;
        }
        prev_is_delim = hb_delim;
    }
    #[allow(clippy::needless_range_loop)]
    if n > 1 {
        let mut j = 1usize;
        let mut hb = haystack[1];
        let mut info = BYTE_INFO[hb as usize];
        loop {
        let curr_i = prev_i ^ 1;
        // The next column's class does not depend on this column's cells.
        let more = j + 1 < n;
        let hb_next = if more { haystack[j + 1] } else { 0 };
        let info_next = if more { BYTE_INFO[hb_next as usize] } else { 0 };
        let class = info as u8;
        let hb_fold = (info >> 8) as u8;
        let hb_lower = USE_CAP && class & CLASS_LOWER != 0;
        let hb_delim = class & CLASS_DELIM != 0;
        let cap_bonus = column_bonus(
            USE_CAP,
            USE_CAP && class & CLASS_UPPER != 0,
            prev_is_lower,
            cap_unit,
            prev_is_delim,
            hb_delim,
            p.delimiter_bonus as i32,
            0,
        );
        let match_base = match_add + cap_bonus;
        let match_lift = match_base + mismatch_pen;
        // One base per row. The cell loop adds the column index only.
        let hp = h[prev_i].as_ptr() as *const i32;
        let hc = h[curr_i].as_mut_ptr() as *mut i32;
        let ep = e[prev_i].as_ptr() as *const i32;
        let ec = e[curr_i].as_mut_ptr() as *mut i32;
        let prev_bits = matched[prev_i];
        unsafe {
            // F lives in this column only. Cell 0 is never read.
            let mut f_prev = 0i32;
            let mut curr_bits = 0u64;
            let mut bit_mask = 2u64;
            let mut h_carry;
            let mut fp = folded_p;
            let mut np = needle_p;
            let mut hp_i = hp.add(1);
            let mut ep_i = ep.add(1);
            let mut hc_i = hc.add(1);
            let mut ec_i = ec.add(1);
            // The first needle cell has no row above it, so its vertical gap is zero.
            let lower_nc0 = *fp;
            fp = fp.add(1);
            let h_here0 = *hp_i;
            // The vertical gap does not read the needle compare.
            let e_hit0 = ((prev_bits & bit_mask) != 0) as i32;
            let e_here0 = (h_here0 - (gap_extend + (0i32.wrapping_sub(e_hit0) & open_extra))).max(*ep_i - gap_extend);
            let is_match0 = hb_fold == lower_nc0;
            let is_exact0 = if WANT_CASE {
                let nb0 = *np;
                np = np.add(1);
                is_match0 && hb == nb0
            } else {
                false
            };
            curr_bits |= bit_mask & (0u64.wrapping_sub(is_match0 as u64));
            let diag0 = -mismatch_pen
                + (0i32.wrapping_sub(is_match0 as i32) & match_lift)
                + (0i32.wrapping_sub(is_exact0 as i32) & case_bonus);
            h_carry = h_here0;
            *ec_i = e_here0;
            let mut h_col = diag0.max(e_here0).max(0);
            *hc_i = h_col;
            let mut f_was_match = is_match0;
            bit_mask <<= 1;
            hp_i = hp_i.add(1);
            ep_i = ep_i.add(1);
            hc_i = hc_i.add(1);
            ec_i = ec_i.add(1);
            if m >= 2 {
                let mut i = 2usize;
                let mut lower_nc = *fp;
                let mut h_here = *hp_i;
                let mut e_prev = *ep_i;
                let mut nb_cur = if WANT_CASE { *np } else { 0 };
                loop {
                    // The next cell's bytes do not depend on this recurrence.
                    let more = i < m;
                    if more {
                        fp = fp.add(1);
                        hp_i = hp_i.add(1);
                        ep_i = ep_i.add(1);
                        if WANT_CASE {
                            np = np.add(1);
                        }
                    }
                    let lower_next = if more { *fp } else { 0 };
                    let h_next = if more { *hp_i } else { 0 };
                    let e_next = if more { *ep_i } else { 0 };
                    let nb_next = if more && WANT_CASE { *np } else { 0 };
                    // The folded compare does not read the gap.
                    let same_case = if WANT_CASE { hb == nb_cur } else { false };
                    let is_match = hb_fold == lower_nc;
                    let diag_base = h_carry - mismatch_pen;
                    // Both gaps read the previous column and the previous row.
                    let e_hit = ((prev_bits & bit_mask) != 0) as i32;
                    let e_new = h_here - (gap_extend + (0i32.wrapping_sub(e_hit) & open_extra));
                    let e_ext = e_prev - gap_extend;
                    let e_here = e_new.max(e_ext);
                    let f_hit = f_was_match as i32;
                    let f_new = h_col - (gap_extend + (0i32.wrapping_sub(f_hit) & open_extra));
                    let f_here = f_new.max(f_prev - gap_extend);
                    let is_exact = WANT_CASE && is_match && same_case;
                    curr_bits |= bit_mask & (0u64.wrapping_sub(is_match as u64));
                    let diag = diag_base
                        + (0i32.wrapping_sub(is_match as i32) & match_lift)
                        + (0i32.wrapping_sub(is_exact as i32) & case_bonus);
                    h_carry = h_here;
                    *ec_i = e_here;
                    f_prev = f_here;
                    let hv = diag.max(e_here).max(f_here).max(0);
                    *hc_i = hv;
                    h_col = hv;
                    f_was_match = is_match;
                    bit_mask <<= 1;
                    i += 1;
                    hc_i = hc_i.add(1);
                    ec_i = ec_i.add(1);
                    if !more {
                        break;
                    }
                    lower_nc = lower_next;
                    h_here = h_next;
                    e_prev = e_next;
                    nb_cur = nb_next;
                }
            }
            matched[curr_i] = curr_bits;
            keep_best(h_col, &mut last_row_max, &mut end_col, j);
        }
        prev_i = curr_i;
        if USE_CAP {
            prev_is_lower = hb_lower;
        }
        prev_is_delim = hb_delim;
        if !more {
            break;
        }
        hb = hb_next;
        info = info_next;
        j += 1;
        }
    }
    (last_row_max as u16, end_col)
}

// Score-only wrapper; test oracle for the batch kernels.
#[cfg(test)]
pub(crate) fn score_one_scalar(
    needle: &[u8],
    haystack: &[u8],
    p: &SwParams,
    include_prefix: bool,
) -> u16 {
    score_one_scalar_full(needle, haystack, p, include_prefix).0
}

// Portable AoS-to-SoA transpose: out[j*stride+k] = stage[k*rows+j].
// Byte loop at memcpy speed; ARM uses this (no SSE2 tiles there).
// x86 keeps the shuffle tiles.
#[allow(dead_code)]
pub(crate) fn soa_from_stage_scalar(stage: &[u8], rows: usize, stride: usize, out: &mut [u8]) {
    for j in 0..rows {
        for k in 0..stride {
            out[j * stride + k] = stage[k * rows + j];
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2 {
    use super::*;
    use std::arch::x86_64::*;

    pub(crate) const LANES: usize = 16;

    // Test oracle SoA: soa[j*stride+p] = haystack[p][j], zero-padded.
    // `stride` must equal the kernel lane count.
    #[cfg(test)]
    pub(crate) fn build_soa(haystacks: &[&[u8]], max_len: usize, stride: usize) -> Vec<u8> {
        let n = haystacks.len().min(stride);
        let mut soa = vec![0u8; max_len * stride];
        for (p, h) in haystacks.iter().enumerate().take(stride) {
            for (j, &b) in h.iter().enumerate() {
                soa[j * stride + p] = b;
            }
        }
        let _ = n;
        soa
    }

    // 8x8 AoS-to-SoA transpose: dst[j*dst_stride+i] = src[i*src_stride+j].
    // Safety: `src` needs 8 rows of `src_stride` readable, `dst` 8 writable;
    // aliasing only when strides match and tiles coincide.
    #[inline]
    #[target_feature(enable = "sse2")]
    pub(crate) unsafe fn transpose_8x8(
        src: *const u8,
        src_stride: usize,
        dst: *mut u8,
        dst_stride: usize,
    ) {
        unsafe {
            let s = src_stride;
            let a0 = _mm_loadl_epi64(src as *const __m128i);
            let a1 = _mm_loadl_epi64(src.add(s) as *const __m128i);
            let a2 = _mm_loadl_epi64(src.add(s * 2) as *const __m128i);
            let a3 = _mm_loadl_epi64(src.add(s * 3) as *const __m128i);
            let a4 = _mm_loadl_epi64(src.add(s * 4) as *const __m128i);
            let a5 = _mm_loadl_epi64(src.add(s * 5) as *const __m128i);
            let a6 = _mm_loadl_epi64(src.add(s * 6) as *const __m128i);
            let a7 = _mm_loadl_epi64(src.add(s * 7) as *const __m128i);

            let p0 = _mm_unpacklo_epi8(a0, a1);
            let p1 = _mm_unpacklo_epi8(a2, a3);
            let p2 = _mm_unpacklo_epi8(a4, a5);
            let p3 = _mm_unpacklo_epi8(a6, a7);

            let q0 = _mm_unpacklo_epi16(p0, p1);
            let q1 = _mm_unpackhi_epi16(p0, p1);
            let q2 = _mm_unpacklo_epi16(p2, p3);
            let q3 = _mm_unpackhi_epi16(p2, p3);

            let r0 = _mm_unpacklo_epi32(q0, q2); // rows 0, 1
            let r1 = _mm_unpackhi_epi32(q0, q2); // rows 2, 3
            let r2 = _mm_unpacklo_epi32(q1, q3); // rows 4, 5
            let r3 = _mm_unpackhi_epi32(q1, q3); // rows 6, 7

            let d = dst_stride;
            _mm_storel_epi64(dst as *mut __m128i, r0);
            _mm_storel_epi64(dst.add(d) as *mut __m128i, _mm_srli_si128(r0, 8));
            _mm_storel_epi64(dst.add(d * 2) as *mut __m128i, r1);
            _mm_storel_epi64(dst.add(d * 3) as *mut __m128i, _mm_srli_si128(r1, 8));
            _mm_storel_epi64(dst.add(d * 4) as *mut __m128i, r2);
            _mm_storel_epi64(dst.add(d * 5) as *mut __m128i, _mm_srli_si128(r2, 8));
            _mm_storel_epi64(dst.add(d * 6) as *mut __m128i, r3);
            _mm_storel_epi64(dst.add(d * 7) as *mut __m128i, _mm_srli_si128(r3, 8));
        }
    }

    // Dual 8x8 to 8x16 step: lanes [0..8] fill the low 128 bits,
    // lanes [8..16] the high, one unpack chain across 256-bit lanes.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn transpose_16x8_avx2(
        src_lo: *const u8,
        src_hi: *const u8,
        src_stride: usize,
        dst: *mut u8,
        dst_stride: usize,
    ) {
        unsafe {
            let s = src_stride;
            macro_rules! pair_at {
                ($k:expr) => {{
                    let lo = _mm_loadl_epi64(src_lo.add(s * $k) as *const __m128i);
                    let hi = _mm_loadl_epi64(src_hi.add(s * $k) as *const __m128i);
                    _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi)
                }};
            }
            let a0 = pair_at!(0);
            let a1 = pair_at!(1);
            let a2 = pair_at!(2);
            let a3 = pair_at!(3);
            let a4 = pair_at!(4);
            let a5 = pair_at!(5);
            let a6 = pair_at!(6);
            let a7 = pair_at!(7);

            let p0 = _mm256_unpacklo_epi8(a0, a1);
            let p1 = _mm256_unpacklo_epi8(a2, a3);
            let p2 = _mm256_unpacklo_epi8(a4, a5);
            let p3 = _mm256_unpacklo_epi8(a6, a7);

            let q0 = _mm256_unpacklo_epi16(p0, p1);
            let q1 = _mm256_unpackhi_epi16(p0, p1);
            let q2 = _mm256_unpacklo_epi16(p2, p3);
            let q3 = _mm256_unpackhi_epi16(p2, p3);

            let r0 = _mm256_unpacklo_epi32(q0, q2);
            let r1 = _mm256_unpackhi_epi32(q0, q2);
            let r2 = _mm256_unpacklo_epi32(q1, q3);
            let r3 = _mm256_unpackhi_epi32(q1, q3);

            // Lane permute [0, 2, 1, 3]: low 128 hold the even row, high the odd.
            let s01 = _mm256_permute4x64_epi64::<0xD8>(r0);
            let s23 = _mm256_permute4x64_epi64::<0xD8>(r1);
            let s45 = _mm256_permute4x64_epi64::<0xD8>(r2);
            let s67 = _mm256_permute4x64_epi64::<0xD8>(r3);

            let d = dst_stride;
            _mm_storeu_si128(dst as *mut __m128i, _mm256_castsi256_si128(s01));
            _mm_storeu_si128(dst.add(d) as *mut __m128i, _mm256_extracti128_si256::<1>(s01));
            _mm_storeu_si128(dst.add(d * 2) as *mut __m128i, _mm256_castsi256_si128(s23));
            _mm_storeu_si128(dst.add(d * 3) as *mut __m128i, _mm256_extracti128_si256::<1>(s23));
            _mm_storeu_si128(dst.add(d * 4) as *mut __m128i, _mm256_castsi256_si128(s45));
            _mm_storeu_si128(dst.add(d * 5) as *mut __m128i, _mm256_extracti128_si256::<1>(s45));
            _mm_storeu_si128(dst.add(d * 6) as *mut __m128i, _mm256_castsi256_si128(s67));
            _mm_storeu_si128(dst.add(d * 7) as *mut __m128i, _mm256_extracti128_si256::<1>(s67));
        }
    }

    // Tiles path-major stage to position-major SoA; writes every byte.
    // Safety: `stage`/`out` cover stride*rows; AVX2 only, no racy aliasing.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn soa_from_stage(
        stage: &[u8],
        rows: usize,
        stride: usize,
        lanes: usize,
        out: &mut [u8],
    ) {
        debug_assert_eq!(rows % 8, 0);
        debug_assert_eq!(stride % 8, 0);
        debug_assert_eq!(lanes % 8, 0);
        debug_assert!(lanes <= stride);
        let stage_p = stage.as_ptr();
        let out_p = out.as_mut_ptr();
        let mut k0 = 0usize;
        while k0 + 16 <= lanes {
            // One tile is 8 path bytes. The next tile continues in the same lanes.
            let mut src_lo = unsafe { stage_p.add(k0 * rows) };
            let mut src_hi = unsafe { stage_p.add((k0 + 8) * rows) };
            let mut dst = unsafe { out_p.add(k0) };
            let mut left = rows;
            let step = stride * 8;
            while left >= 16 {
                // The next eight path bytes do not read this tile.
                unsafe {
                    transpose_16x8_avx2(src_lo, src_hi, rows, dst, stride);
                    transpose_16x8_avx2(
                        src_lo.add(8),
                        src_hi.add(8),
                        rows,
                        dst.add(step),
                        stride,
                    );
                    src_lo = src_lo.add(16);
                    src_hi = src_hi.add(16);
                    dst = dst.add(step * 2);
                }
                left -= 16;
            }
            if left > 0 {
                unsafe {
                    transpose_16x8_avx2(src_lo, src_hi, rows, dst, stride);
                }
            }
            k0 += 16;
        }
        // Narrow-stride tail keeps the old 8-wide contract.
        while k0 < lanes {
            let mut src = unsafe { stage_p.add(k0 * rows) };
            let mut dst = unsafe { out_p.add(k0) };
            let mut left = rows;
            let step = stride * 8;
            while left >= 16 {
                unsafe {
                    transpose_8x8(src, rows, dst, stride);
                    transpose_8x8(src.add(8), rows, dst.add(step), stride);
                    src = src.add(16);
                    dst = dst.add(step * 2);
                }
                left -= 16;
            }
            if left > 0 {
                unsafe {
                    transpose_8x8(src, rows, dst, stride);
                }
            }
            k0 += 8;
        }
    }

    // AVX2 SW over up to 16 pre-trimmed haystacks; per-lane (score, end_col).
    // Needle rows dynamic to 32; longer needles stay scalar.
    // Safety: AVX2 available; `out.len() >= haystacks.len()`.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn score_batch_avx2(
        needle: &[u8],
        soa: &[u8],
        max_len: usize,
        p: &SwParams,
        include_prefix_mask: u16,
        out: &mut [u16],
        end_cols: &mut [u16],
    ) {
        let m = needle.len().min(32);
        let n = max_len;
        if m == 0 || n == 0 {
            out.iter_mut().for_each(|s| *s = 0);
            end_cols.iter_mut().for_each(|s| *s = 0);
            return;
        }

        let set1 = |v: u16| _mm256_set1_epi16(v as i16);
        let zero = _mm256_setzero_si256();

        let gap_ext_v = set1(p.gap_extend);
        let gap_diff_v = set1(p.gap_open.wrapping_sub(p.gap_extend));
        let match_score_v = set1(p.match_score);
        let mismatch_v = set1(p.mismatch_penalty);
        let delim_v = set1(p.delimiter_bonus);
        let cap_v = set1(p.capitalization_bonus);
        let case_v = set1(p.matching_case_bonus);
        let prefix_v = set1(p.prefix_bonus);
        let bit20 = set1(0x20);
        let a_lo = set1(b'A' as u16 - 1);
        let z_hi = set1(b'Z' as u16 + 1);
        let la_lo = set1(b'a' as u16 - 1);
        let lz_hi = set1(b'z' as u16 + 1);
        let d0_lo = set1(b'0' as u16 - 1);
        let d9_hi = set1(b'9' as u16 + 1);
        let lt128 = set1(128);
        let col_one = set1(1);
        let need_case = p.capitalization_bonus != 0 || p.matching_case_bonus != 0;
        let mut n_fold_v = [zero; 32];
        let mut nc_v = [zero; 32];
        let mut i = 0usize;
        while i + 1 < m {
            let b0 = needle[i] as u16;
            let b1 = needle[i + 1] as u16;
            let f0 = if (b0 as u8).is_ascii_uppercase() { b0 + 32 } else { b0 };
            let f1 = if (b1 as u8).is_ascii_uppercase() { b1 + 32 } else { b1 };
            n_fold_v[i] = set1(f0);
            n_fold_v[i + 1] = set1(f1);
            if need_case {
                nc_v[i] = set1(b0);
                nc_v[i + 1] = set1(b1);
            }
            i += 2;
        }
        if i < m {
            let b = needle[i] as u16;
            n_fold_v[i] = set1(if (b as u8).is_ascii_uppercase() { b + 32 } else { b });
            if need_case {
                nc_v[i] = set1(b);
            }
        }

        let prefix_add = if include_prefix_mask == 0 {
            zero
        } else if include_prefix_mask == u16::MAX {
            // Every lane takes the prefix, so there is no per-lane mask.
            prefix_v
        } else {
            // Each mask bit becomes a 0xFFFF lane.
            let bits = _mm256_setr_epi16(
                0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80, 0x100, 0x200, 0x400, 0x800, 0x1000,
                0x2000, 0x4000, 0x8000u16 as i16,
            );
            let maskv = _mm256_set1_epi16(include_prefix_mask as i16);
            let lanes = _mm256_cmpeq_epi16(_mm256_and_si256(maskv, bits), bits);
            _mm256_and_si256(prefix_v, lanes)
        };

        // 1D state with register-carried f/mm.
        let mut h = [zero; 32];
        let mut e = [zero; 32];
        let mut match_prev = [zero; 32];

        let mut prev_lower = zero;
        let mut prev_delim;
        let mut last_row_max = zero;
        let mut end_col_v = zero;
        // End column increments; wraps like `j as u16`.
        let mut col_v = zero;
        // Column 0 has a zero left edge, so only the match bonus and the vertical gap run.
            let raw = unsafe { _mm_loadu_si128(soa.as_ptr() as *const __m128i) };
            let hay = _mm256_cvtepu8_epi16(raw);
            // Digit and the ASCII range do not read the case fold.
            let ge_0 = _mm256_cmpgt_epi16(hay, d0_lo);
            let le_9 = _mm256_cmpgt_epi16(d9_hi, hay);
            let below = _mm256_cmpgt_epi16(lt128, hay);
            let ge_a = _mm256_cmpgt_epi16(hay, a_lo);
            let le_z = _mm256_cmpgt_epi16(z_hi, hay);
            let is_upper = _mm256_and_si256(ge_a, le_z);
            let hay_fold = _mm256_add_epi16(hay, _mm256_and_si256(is_upper, bit20));
            let ge_la = _mm256_cmpgt_epi16(hay_fold, la_lo);
            let le_lz = _mm256_cmpgt_epi16(lz_hi, hay_fold);
            let is_letter = _mm256_and_si256(ge_la, le_lz);
            let is_digit = _mm256_and_si256(ge_0, le_9);
            let is_delim = _mm256_andnot_si256(_mm256_or_si256(is_letter, is_digit), below);
            let bonus_base = _mm256_add_epi16(match_score_v, prefix_add);
            if need_case {
                prev_lower = _mm256_andnot_si256(is_upper, is_letter);
            }
            let mut h_up;
            let mut f_open_extra;
            let mut f_running = zero;
            {
                let mm = _mm256_cmpeq_epi16(hay_fold, n_fold_v[0]);
                let opened = _mm256_add_epi8(_mm256_and_si256(mm, gap_diff_v), gap_ext_v);
                match_prev[0] = opened;
                let mut match_gain = _mm256_and_si256(bonus_base, mm);
                if need_case {
                    let eq_nc = _mm256_cmpeq_epi16(hay, nc_v[0]);
                    match_gain = _mm256_add_epi16(match_gain, _mm256_and_si256(eq_nc, case_v));
                }
                h[0] = match_gain;
                h_up = match_gain;
                f_open_extra = opened;
            }
            if m > 1 {
                // The next needle compare does not read this cell.
                let mut mm = _mm256_cmpeq_epi16(hay_fold, n_fold_v[1]);
                for i in 1..m {
                    let mm_here = mm;
                    if i + 1 < m {
                        mm = _mm256_cmpeq_epi16(hay_fold, n_fold_v[i + 1]);
                    }
                    // The horizontal gap does not read this row's match.
                    f_running = _mm256_max_epu16(
                        _mm256_subs_epu16(h_up, f_open_extra),
                        _mm256_subs_epu16(f_running, gap_ext_v),
                    );
                    let opened = _mm256_add_epi8(_mm256_and_si256(mm_here, gap_diff_v), gap_ext_v);
                    match_prev[i] = opened;
                    let mut match_gain = _mm256_and_si256(bonus_base, mm_here);
                    if need_case {
                        let eq_nc = _mm256_cmpeq_epi16(hay, nc_v[i]);
                        match_gain =
                            _mm256_add_epi16(match_gain, _mm256_and_si256(eq_nc, case_v));
                    }
                    let h_val = _mm256_max_epu16(match_gain, f_running);
                    h[i] = h_val;
                    h_up = h_val;
                    f_open_extra = opened;
                }
            }
            let next_col = _mm256_add_epi16(col_v, col_one);
            let gt_best = _mm256_cmpgt_epi16(h_up, last_row_max);
            end_col_v = _mm256_blendv_epi8(end_col_v, col_v, gt_best);
            last_row_max = _mm256_max_epu16(last_row_max, h_up);
            col_v = next_col;
            prev_delim = is_delim;

        let mut col_ptr = unsafe { soa.as_ptr().add(LANES) };
        macro_rules! classify16 {
            ($hay:expr) => {{
                // Digit and the ASCII range do not read the case fold.
                let ge_0 = _mm256_cmpgt_epi16($hay, d0_lo);
                let le_9 = _mm256_cmpgt_epi16(d9_hi, $hay);
                let below = _mm256_cmpgt_epi16(lt128, $hay);
                let ge_a = _mm256_cmpgt_epi16($hay, a_lo);
                let le_z = _mm256_cmpgt_epi16(z_hi, $hay);
                let is_upper = _mm256_and_si256(ge_a, le_z);
                let hay_fold = _mm256_add_epi16($hay, _mm256_and_si256(is_upper, bit20));
                let ge_la = _mm256_cmpgt_epi16(hay_fold, la_lo);
                let le_lz = _mm256_cmpgt_epi16(lz_hi, hay_fold);
                let is_letter = _mm256_and_si256(ge_la, le_lz);
                let is_digit = _mm256_and_si256(ge_0, le_9);
                let is_delim = _mm256_andnot_si256(_mm256_or_si256(is_letter, is_digit), below);
                (is_upper, hay_fold, is_letter, is_delim)
            }};
        }
        // One loop for a cased query and one for a lowercase query.
        macro_rules! u16_cols {
            ($cased:tt) => {
        let (mut hay, mut is_upper, mut hay_fold, mut is_letter, mut is_delim) = if n > 1 {
            let raw = unsafe { _mm_loadu_si128(col_ptr as *const __m128i) };
            col_ptr = unsafe { col_ptr.add(LANES) };
            let hay = _mm256_cvtepu8_epi16(raw);
            let (is_upper, hay_fold, is_letter, is_delim) = classify16!(hay);
            (hay, is_upper, hay_fold, is_letter, is_delim)
        } else {
            (zero, zero, zero, zero, zero)
        };
        for j in 1..n {
            let bonus_base;
            if $cased {
                // Capitalization does not read the delimiter bonus.
                let is_lower = _mm256_andnot_si256(is_upper, is_letter);
                let cap_add = _mm256_and_si256(_mm256_and_si256(is_upper, prev_lower), cap_v);
                let delim_bonus =
                    _mm256_and_si256(_mm256_andnot_si256(is_delim, prev_delim), delim_v);
                bonus_base =
                    _mm256_add_epi16(_mm256_add_epi16(match_score_v, delim_bonus), cap_add);
                prev_lower = is_lower;
            } else {
                let delim_bonus =
                    _mm256_and_si256(_mm256_andnot_si256(is_delim, prev_delim), delim_v);
                bonus_base = _mm256_add_epi16(delim_bonus, match_score_v);
            }
            // Class of the following column does not read this column's scores.
            let (hay_n, upper_n, fold_n, letter_n, delim_n) = if j + 1 < n {
                let raw = unsafe { _mm_loadu_si128(col_ptr as *const __m128i) };
                col_ptr = unsafe { col_ptr.add(LANES) };
                let hay_n = _mm256_cvtepu8_epi16(raw);
                let (upper_n, fold_n, letter_n, delim_n) = classify16!(hay_n);
                (hay_n, upper_n, fold_n, letter_n, delim_n)
            } else {
                (zero, zero, zero, zero, zero)
            };
            // Register-carried recurrence; no 2D state or f array.
            let mut h_diag;
            // Row 0 writes these before a later row reads them.
            let mut h_up;
            let mut f_open_extra;
            let mut f_running = zero;

            let row1_h = if m > 1 { h[1] } else { zero };
            let row1_e = if m > 1 { e[1] } else { zero };
            let row1_open = if m > 1 { match_prev[1] } else { zero };
            macro_rules! i_steps {
                ($case:expr) => {
                    {
                        let h_left = h[0];
                        let e0 = e[0];
                        let e_open_extra = match_prev[0];
                        // The vertical gap does not read the needle compare.
                        let e_val = _mm256_max_epu16(
                            _mm256_subs_epu16(h_left, e_open_extra),
                            _mm256_subs_epu16(e0, gap_ext_v),
                        );
                        let mm = _mm256_cmpeq_epi16(hay_fold, n_fold_v[0]);
                        let opened = _mm256_add_epi8(_mm256_and_si256(mm, gap_diff_v), gap_ext_v);
                        match_prev[0] = opened;
                        let mut match_gain = _mm256_and_si256(bonus_base, mm);
                        if $case {
                            let eq_nc = _mm256_cmpeq_epi16(hay, nc_v[0]);
                            match_gain =
                                _mm256_add_epi16(match_gain, _mm256_and_si256(eq_nc, case_v));
                        }
                        // `h_diag` is zero here, so the saturated subtract vanishes.
                        h_diag = h_left;
                        let diag = match_gain;
                        e[0] = e_val;
                        let h_val = _mm256_max_epu16(diag, e_val);
                        h[0] = h_val;
                        h_up = h_val;
                        f_open_extra = opened;
                    }
                    if m > 1 {
                        // The next needle compare does not read this cell.
                        let mut mm = _mm256_cmpeq_epi16(hay_fold, n_fold_v[1]);
                        let mut h_left = row1_h;
                        let mut e_cur = row1_e;
                        let mut e_open_extra = row1_open;
                        for i in 1..m {
                            let h_ahead = if i + 1 < m { h[i + 1] } else { zero };
                            let e_ahead = if i + 1 < m { e[i + 1] } else { zero };
                            let open_ahead = if i + 1 < m { match_prev[i + 1] } else { zero };
                            // The next needle compare does not read this cell.
                            let mm_here = mm;
                            if i + 1 < m {
                                mm = _mm256_cmpeq_epi16(hay_fold, n_fold_v[i + 1]);
                            }
                            // The case compare does not read this cell. A lowercase query drops it.
                            let eq_nc = if $case {
                                _mm256_cmpeq_epi16(hay, nc_v[i])
                            } else {
                                zero
                            };
                            // Mismatch and both gaps do not read the case bonus.
                            let miss = _mm256_subs_epu16(h_diag, mismatch_v);
                            // The base match add does not read the case bonus.
                            let mut hit = _mm256_add_epi16(h_diag, bonus_base);
                            let e_val = _mm256_max_epu16(
                                _mm256_subs_epu16(h_left, e_open_extra),
                                _mm256_subs_epu16(e_cur, gap_ext_v),
                            );
                            f_running = _mm256_max_epu16(
                                _mm256_subs_epu16(h_up, f_open_extra),
                                _mm256_subs_epu16(f_running, gap_ext_v),
                            );
                            if $case {
                                hit = _mm256_add_epi16(hit, _mm256_and_si256(eq_nc, case_v));
                            } else {
                                let _ = eq_nc;
                            }
                            let diag = _mm256_blendv_epi8(miss, hit, mm_here);

                            h_diag = h_left;
                            let opened = _mm256_add_epi8(_mm256_and_si256(mm_here, gap_diff_v), gap_ext_v);
                            match_prev[i] = opened;
                            e[i] = e_val;

                            let h_val = _mm256_max_epu16(_mm256_max_epu16(diag, e_val), f_running);
                            h[i] = h_val;
                            h_up = h_val;
                            f_open_extra = opened;
                            h_left = h_ahead;
                            e_cur = e_ahead;
                            e_open_extra = open_ahead;
                        }
                    }
                };
            }
            i_steps!($cased);

            let next_col = _mm256_add_epi16(col_v, col_one);
            let gt_best = _mm256_cmpgt_epi16(h_up, last_row_max);
            end_col_v = _mm256_blendv_epi8(end_col_v, col_v, gt_best);
            last_row_max = _mm256_max_epu16(last_row_max, h_up);
            col_v = next_col;

            prev_delim = is_delim;
            hay = hay_n;
            is_upper = upper_n;
            hay_fold = fold_n;
            is_letter = letter_n;
            is_delim = delim_n;
        }
            };
        }
        if need_case {
            u16_cols!(true);
        } else {
            u16_cols!(false);
        }

        let nout = out.len().min(end_cols.len()).min(16);
        unsafe {
            store_u16_lanes(last_row_max, &mut out[..nout]);
            store_u16_lanes(end_col_v, &mut end_cols[..nout]);
        }
    }

    // 32-lane u8 variant; same semantics, valid only when scores fit in u8.
    // Safety: AVX2 available; scores must fit in u8.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn score_batch_avx2_u8(
        needle: &[u8],
        soa: &[u8],
        max_len: usize,
        p: &SwParams,
        include_prefix_mask: u32,
        out: &mut [u16],
        end_cols: &mut [u16],
    ) {
        const LANES8: usize = 32;
        let m = needle.len().min(32);
        let n = max_len;
        if m == 0 || n == 0 {
            out.iter_mut().for_each(|s| *s = 0);
            end_cols.iter_mut().for_each(|s| *s = 0);
            return;
        }

        let set1 = |v: u8| _mm256_set1_epi8(v as i8);
        let zero = _mm256_setzero_si256();
        let ones = set1(u8::MAX);

        let gap_ext_v = set1(p.gap_extend as u8);
        let gap_diff_v = set1(p.gap_open.wrapping_sub(p.gap_extend) as u8);
        let match_score_v = set1(p.match_score as u8);
        let mismatch_v = set1(p.mismatch_penalty as u8);
        let delim_v = set1(p.delimiter_bonus as u8);
        let cap_v = set1(p.capitalization_bonus as u8);
        let case_v = set1(p.matching_case_bonus as u8);
        let prefix_v = set1(p.prefix_bonus as u8);
        let bit20 = set1(0x20);
        let add_upper = set1(127 - b'Z');
        let add_letter = set1(127 - b'z');
        let thresh_az = set1(101);
        let add_digit = set1(127 - b'9');
        let thresh_digit = set1(117);
        let col_one = set1(1);
        let need_case = p.capitalization_bonus != 0 || p.matching_case_bonus != 0;

        let prefix_add = if include_prefix_mask == 0 {
            zero
        } else if include_prefix_mask == u32::MAX {
            // Every lane takes the prefix, so there is no per-lane mask.
            prefix_v
        } else {
            // Each mask bit becomes a 0xFF lane. Same bytes as the scalar expand.
            let vmask = _mm256_set1_epi32(include_prefix_mask as i32);
            let shuffle = _mm256_setr_epi8(
                0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
                3, 3, 3, 3, 3, 3,
            );
            let bytes = _mm256_shuffle_epi8(vmask, shuffle);
            let bits = _mm256_set1_epi64x(0x8040_2010_0804_0201u64 as i64);
            let lanes = _mm256_cmpeq_epi8(_mm256_and_si256(bytes, bits), bits);
            _mm256_and_si256(prefix_v, lanes)
        };

        let mut n_fold_v = [zero; 32];
        let mut nc_v = [zero; 32];
        let mut i = 0usize;
        while i + 1 < m {
            let b0 = needle[i];
            let b1 = needle[i + 1];
            let f0 = if b0.is_ascii_uppercase() { b0 + 32 } else { b0 };
            let f1 = if b1.is_ascii_uppercase() { b1 + 32 } else { b1 };
            n_fold_v[i] = set1(f0);
            n_fold_v[i + 1] = set1(f1);
            if need_case {
                nc_v[i] = set1(b0);
                nc_v[i + 1] = set1(b1);
            }
            i += 2;
        }
        if i < m {
            let b = needle[i];
            n_fold_v[i] = set1(if b.is_ascii_uppercase() { b + 32 } else { b });
            if need_case {
                nc_v[i] = set1(b);
            }
        }

        // 1D state with register-carried f/mm.
        let mut h = [zero; 32];
        let mut e = [zero; 32];
        let mut match_prev = [zero; 32];

        let mut prev_lower = zero;
        let mut prev_delim = zero;
        let mut end_col_v = zero;
        // End column increments; wraps like `j as u8`.
        let mut col_v = zero;
        // Equal scores keep the earlier column.
        let mut best = zero;

        let mut col_ptr = soa.as_ptr();
        macro_rules! classify8 {
            ($hay:expr) => {{
                // Digit and the ASCII range do not read the case fold.
                let is_digit = _mm256_cmpgt_epi8(_mm256_add_epi8($hay, add_digit), thresh_digit);
                let below = _mm256_cmpgt_epi8($hay, ones);
                let is_upper = _mm256_cmpgt_epi8(_mm256_add_epi8($hay, add_upper), thresh_az);
                let hay_fold = _mm256_or_si256($hay, _mm256_and_si256(is_upper, bit20));
                let is_letter =
                    _mm256_cmpgt_epi8(_mm256_add_epi8(hay_fold, add_letter), thresh_az);
                let alnum = _mm256_or_si256(is_letter, is_digit);
                let is_delim = _mm256_andnot_si256(alnum, below);
                (is_upper, hay_fold, is_letter, is_delim)
            }};
        }
        let (mut hay, mut is_upper, mut hay_fold, mut is_letter, mut is_delim) = if n > 0 {
            let hay = unsafe { _mm256_loadu_si256(col_ptr as *const __m256i) };
            col_ptr = unsafe { col_ptr.add(LANES8) };
            let (is_upper, hay_fold, is_letter, is_delim) = classify8!(hay);
            (hay, is_upper, hay_fold, is_letter, is_delim)
        } else {
            (zero, zero, zero, zero, zero)
        };
        for j in 0..n {
            // All lanes at 255 exit; every 4th column is exact.
            if j & 3 == 0 && _mm256_movemask_epi8(_mm256_cmpeq_epi8(best, ones)) == -1 {
                break;
            }
            let mut bonus_base = if need_case {
                // Capitalization does not read the delimiter bonus.
                let is_lower = _mm256_andnot_si256(is_upper, is_letter);
                let cap_add = _mm256_and_si256(_mm256_and_si256(is_upper, prev_lower), cap_v);
                let delim_bonus = _mm256_and_si256(_mm256_andnot_si256(is_delim, prev_delim), delim_v);
                prev_lower = is_lower;
                _mm256_add_epi8(_mm256_add_epi8(match_score_v, delim_bonus), cap_add)
            } else {
                let delim_bonus = _mm256_and_si256(_mm256_andnot_si256(is_delim, prev_delim), delim_v);
                _mm256_add_epi8(delim_bonus, match_score_v)
            };
            if j == 0 {
                bonus_base = _mm256_add_epi8(bonus_base, prefix_add);
            }
            // Class of the following column does not read this column's scores.
            let (hay_n, upper_n, fold_n, letter_n, delim_n) = if j + 1 < n {
                let hay_n = unsafe { _mm256_loadu_si256(col_ptr as *const __m256i) };
                col_ptr = unsafe { col_ptr.add(LANES8) };
                let (upper_n, fold_n, letter_n, delim_n) = classify8!(hay_n);
                (hay_n, upper_n, fold_n, letter_n, delim_n)
            } else {
                (zero, zero, zero, zero, zero)
            };

            // Register-carried recurrence; no 2D state or f array.
            let mut h_diag = zero;
            let mut h_up = zero;
            let mut f_running = zero;
            let mut f_open_extra = zero;

            macro_rules! i_steps {
                ($case:expr) => {
                    if m > 0 {
                        // The next needle compare does not read this cell.
                        let mut mm = _mm256_cmpeq_epi8(hay_fold, n_fold_v[0]);
                        let mut h_left = h[0];
                        let mut e_cur = e[0];
                        let mut e_open_extra = match_prev[0];
                        for i in 0..m {
                            let h_ahead = if i + 1 < m { h[i + 1] } else { zero };
                            let e_ahead = if i + 1 < m { e[i + 1] } else { zero };
                            let open_ahead = if i + 1 < m { match_prev[i + 1] } else { zero };
                            // The next needle compare does not read this cell.
                            let mm_here = mm;
                            if i + 1 < m {
                                mm = _mm256_cmpeq_epi8(hay_fold, n_fold_v[i + 1]);
                            }
                            // The case compare does not read this cell. A lowercase query drops it.
                            let eq_nc = if $case {
                                _mm256_cmpeq_epi8(hay, nc_v[i])
                            } else {
                                zero
                            };
                            // Mismatch and both gaps do not read the case bonus.
                            let miss = _mm256_subs_epu8(h_diag, mismatch_v);
                            // The base match add does not read the case bonus.
                            let mut hit = _mm256_adds_epu8(h_diag, bonus_base);
                            let e_val = _mm256_max_epu8(
                                _mm256_subs_epu8(h_left, e_open_extra),
                                _mm256_subs_epu8(e_cur, gap_ext_v),
                            );
                            f_running = _mm256_max_epu8(
                                _mm256_subs_epu8(h_up, f_open_extra),
                                _mm256_subs_epu8(f_running, gap_ext_v),
                            );
                            if $case {
                                hit = _mm256_adds_epu8(hit, _mm256_and_si256(eq_nc, case_v));
                            } else {
                                let _ = eq_nc;
                            }
                            let diag = _mm256_blendv_epi8(miss, hit, mm_here);

                            h_diag = h_left;
                            let opened = _mm256_add_epi8(_mm256_and_si256(mm_here, gap_diff_v), gap_ext_v);
                            match_prev[i] = opened;
                            e[i] = e_val;

                            let h_val = _mm256_max_epu8(_mm256_max_epu8(diag, e_val), f_running);
                            h[i] = h_val;
                            h_up = h_val;
                            f_open_extra = opened;
                            h_left = h_ahead;
                            e_cur = e_ahead;
                            e_open_extra = open_ahead;
                        }
                    }
                };
            }
            if need_case {
                i_steps!(true);
            } else {
                i_steps!(false);
            }

            let next_col = _mm256_add_epi8(col_v, col_one);
            let new_best = _mm256_max_epu8(best, h_up);
            let keep_old = _mm256_cmpeq_epi8(new_best, best);
            end_col_v = _mm256_blendv_epi8(col_v, end_col_v, keep_old);
            best = new_best;
            col_v = next_col;

            prev_delim = is_delim;
            hay = hay_n;
            is_upper = upper_n;
            hay_fold = fold_n;
            is_letter = letter_n;
            is_delim = delim_n;
        }

        let nout = out.len().min(end_cols.len()).min(32);
        unsafe {
            store_u8_lanes(best, &mut out[..nout]);
            store_u8_lanes(end_col_v, &mut end_cols[..nout]);
        }
    }

    pub(crate) struct U8NeedleFold {
        n_fold: [core::arch::x86_64::__m256i; 32],
        nc: [core::arch::x86_64::__m256i; 32],
        gap_ext_v: core::arch::x86_64::__m256i,
        gap_diff_v: core::arch::x86_64::__m256i,
        match_score_v: core::arch::x86_64::__m256i,
        mismatch_v: core::arch::x86_64::__m256i,
        delim_v: core::arch::x86_64::__m256i,
        cap_v: core::arch::x86_64::__m256i,
        case_v: core::arch::x86_64::__m256i,
        prefix_v: core::arch::x86_64::__m256i,
        bit20: core::arch::x86_64::__m256i,
        add_upper: core::arch::x86_64::__m256i,
        add_letter: core::arch::x86_64::__m256i,
        thresh_az: core::arch::x86_64::__m256i,
        add_digit: core::arch::x86_64::__m256i,
        thresh_digit: core::arch::x86_64::__m256i,
        col_one: core::arch::x86_64::__m256i,
        ones: core::arch::x86_64::__m256i,
        need_case: bool,
    }

    // Needle and score broadcasts for every batch of this query.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn prepare_u8_needle(needle: &[u8], p: &SwParams) -> U8NeedleFold {
        use core::arch::x86_64::*;
        let set1 = |v: u8| _mm256_set1_epi8(v as i8);
        let zero = _mm256_setzero_si256();
        let mut n_fold = [zero; 32];
        let mut nc = [zero; 32];
        let need_case = p.capitalization_bonus != 0 || p.matching_case_bonus != 0;
        let m = needle.len().min(32);
        let mut i = 0usize;
        while i + 1 < m {
            let b0 = needle[i];
            let b1 = needle[i + 1];
            let f0 = if b0.is_ascii_uppercase() { b0 + 32 } else { b0 };
            let f1 = if b1.is_ascii_uppercase() { b1 + 32 } else { b1 };
            n_fold[i] = set1(f0);
            n_fold[i + 1] = set1(f1);
            if need_case {
                nc[i] = set1(b0);
                nc[i + 1] = set1(b1);
            }
            i += 2;
        }
        if i < m {
            let b = needle[i];
            n_fold[i] = set1(if b.is_ascii_uppercase() { b + 32 } else { b });
            if need_case {
                nc[i] = set1(b);
            }
        }
        let ones = set1(u8::MAX);
        U8NeedleFold {
            n_fold,
            nc,
            gap_ext_v: set1(p.gap_extend as u8),
            gap_diff_v: set1(p.gap_open.wrapping_sub(p.gap_extend) as u8),
            match_score_v: set1(p.match_score as u8),
            mismatch_v: set1(p.mismatch_penalty as u8),
            delim_v: set1(p.delimiter_bonus as u8),
            cap_v: set1(p.capitalization_bonus as u8),
            case_v: set1(p.matching_case_bonus as u8),
            prefix_v: set1(p.prefix_bonus as u8),
            bit20: set1(0x20),
            add_upper: set1(127 - b'Z'),
            add_letter: set1(127 - b'z'),
            thresh_az: set1(101),
            add_digit: set1(127 - b'9'),
            thresh_digit: set1(117),
            col_one: set1(1),
            ones,
            need_case,
        }
    }

    struct CachedU8Fold {
        needle: [u8; 32],
        nlen: u8,
        params: super::SwParams,
        fold: U8NeedleFold,
    }

    fn same_params(a: &super::SwParams, b: &super::SwParams) -> bool {
        a.match_score == b.match_score
            && a.mismatch_penalty == b.mismatch_penalty
            && a.gap_open == b.gap_open
            && a.gap_extend == b.gap_extend
            && a.prefix_bonus == b.prefix_bonus
            && a.delimiter_bonus == b.delimiter_bonus
            && a.capitalization_bonus == b.capitalization_bonus
            && a.matching_case_bonus == b.matching_case_bonus
    }

    // One fold per needle on this thread. Groups of one search share it.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn with_cached_u8_fold<R>(
        needle: &[u8],
        p: &super::SwParams,
        f: impl FnOnce(&U8NeedleFold) -> R,
    ) -> R {
        use std::cell::RefCell;
        thread_local! {
            static CACHE: RefCell<Option<CachedU8Fold>> = const { RefCell::new(None) };
        }
        CACHE.with(|slot| {
            let mut slot = slot.borrow_mut();
            let nlen = needle.len().min(32);
            let reuse = slot.as_ref().is_some_and(|cached| {
                cached.nlen as usize == nlen
                    && cached.needle[..nlen] == needle[..nlen]
                    && same_params(&cached.params, p)
            });
            if !reuse {
                let mut key = [0u8; 32];
                key[..nlen].copy_from_slice(&needle[..nlen]);
                *slot = Some(CachedU8Fold {
                    needle: key,
                    nlen: nlen as u8,
                    params: *p,
                    fold: unsafe { prepare_u8_needle(needle, p) },
                });
            }
            f(&slot.as_ref().unwrap().fold)
        })
    }

    // Dispatches common needle lengths to const-unrolled kernels; bit-identical.
    // Other lengths keep the general runtime-m loop.
    // Safety: AVX2 required (score.rs gates on simd_available).
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn score_batch_avx2_u8_dispatch<const CHECK_SAT: bool>(
        needle: &[u8],
        soa: &[u8],
        max_len: usize,
        p: &SwParams,
        include_prefix_mask: u32,
        out: &mut [u16],
        end_cols: &mut [u16],
        fold: Option<&U8NeedleFold>,
    ) {
        // `FFF_NO_UNROLL=1` forces the general runtime-m loop.
        static NO_UNROLL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *NO_UNROLL.get_or_init(|| std::env::var_os("FFF_NO_UNROLL").is_some()) {
            return unsafe {
                score_batch_avx2_u8(needle, soa, max_len, p, include_prefix_mask, out, end_cols)
            };
        }
        let owned = if fold.is_none() {
            Some(unsafe { prepare_u8_needle(needle, p) })
        } else {
            None
        };
        let fold = fold.unwrap_or_else(|| owned.as_ref().unwrap());
        let m = needle.len().min(32);
        macro_rules! unrolled {
            ($($v:literal),*) => {
                match m {
                    $( $v => unsafe { score_batch_avx2_u8_const::<$v, CHECK_SAT>(fold, needle, soa, max_len, p, include_prefix_mask, out, end_cols) }, )*
                    _ => unsafe { score_batch_avx2_u8(needle, soa, max_len, p, include_prefix_mask, out, end_cols) },
                }
            };
        }
        unrolled!(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32
        );
    }

    // Compile-time needle length: unrolled row loop, SROA state; bit-identical.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn score_batch_avx2_u8_const<const M: usize, const CHECK_SAT: bool>(
        fold: &U8NeedleFold,
        needle: &[u8],
        soa: &[u8],
        max_len: usize,
        _p: &SwParams,
        include_prefix_mask: u32,
        out: &mut [u16],
        end_cols: &mut [u16],
    ) {
        debug_assert_eq!(needle.len().min(32), M);
        let m = M;
        let n = max_len;
        if m == 0 || n == 0 {
            out.iter_mut().for_each(|s| *s = 0);
            end_cols.iter_mut().for_each(|s| *s = 0);
            return;
        }

        let zero = _mm256_setzero_si256();
        // Row-carried penalties stay live. Column class vectors are reloaded from the fold.
        let gap_ext_v = fold.gap_ext_v;
        let gap_diff_v = fold.gap_diff_v;
        let mismatch_v = fold.mismatch_v;
        let need_case = fold.need_case;

        let prefix_add = if include_prefix_mask == 0 {
            zero
        } else if include_prefix_mask == u32::MAX {
            // Every lane takes the prefix, so there is no per-lane mask.
            fold.prefix_v
        } else {
            // Each mask bit becomes a 0xFF lane. Same bytes as the scalar expand.
            let vmask = _mm256_set1_epi32(include_prefix_mask as i32);
            let shuffle = _mm256_setr_epi8(
                0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
                3, 3, 3, 3, 3, 3,
            );
            let bytes = _mm256_shuffle_epi8(vmask, shuffle);
            let bits = _mm256_set1_epi64x(0x8040_2010_0804_0201u64 as i64);
            let lanes = _mm256_cmpeq_epi8(_mm256_and_si256(bytes, bits), bits);
            _mm256_and_si256(fold.prefix_v, lanes)
        };

        // 1D state with register-carried f/mm; indices compile-time for SROA.
        let mut h = [zero; M];
        let mut e = [zero; M];
        let mut match_prev = [zero; M];

        let mut prev_lower = zero;
        let mut prev_delim;
        let mut end_col_v = zero;
        // End column increments; wraps like `j as u8`.
        let mut col_v = zero;
        // Equal scores keep the earlier column.
        let mut best = zero;
        // Needle lanes stay in the fold. A copy would compete with the DP rows.
        // Column 0 has a zero left edge, so only the match bonus and the vertical gap run.
            let hay = unsafe { _mm256_loadu_si256(soa.as_ptr() as *const __m256i) };
            // Digit and the ASCII range do not read the case fold.
            let is_digit = _mm256_cmpgt_epi8(_mm256_add_epi8(hay, fold.add_digit), fold.thresh_digit);
            let below = _mm256_cmpgt_epi8(hay, fold.ones);
            let is_upper = _mm256_cmpgt_epi8(_mm256_add_epi8(hay, fold.add_upper), fold.thresh_az);
            let hay_fold = _mm256_or_si256(hay, _mm256_and_si256(is_upper, fold.bit20));
            let is_letter =
                _mm256_cmpgt_epi8(_mm256_add_epi8(hay_fold, fold.add_letter), fold.thresh_az);
            let alnum = _mm256_or_si256(is_letter, is_digit);
            let is_delim = _mm256_andnot_si256(alnum, below);
            // No previous column, so the delimiter and capital bonuses are zero.
            let bonus_base = _mm256_add_epi8(fold.match_score_v, prefix_add);
            if need_case {
                prev_lower = _mm256_andnot_si256(is_upper, is_letter);
            }
            let mut h_up;
            let mut f_open_extra;
            let mut f_running = zero;
            {
                let mm = _mm256_cmpeq_epi8(hay_fold, fold.n_fold[0]);
                let opened = _mm256_add_epi8(_mm256_and_si256(mm, gap_diff_v), gap_ext_v);
                match_prev[0] = opened;
                let mut match_gain = _mm256_and_si256(bonus_base, mm);
                if need_case {
                    let eq_nc = _mm256_cmpeq_epi8(hay, fold.nc[0]);
                    match_gain = _mm256_add_epi8(match_gain, _mm256_and_si256(eq_nc, fold.case_v));
                }
                h[0] = match_gain;
                h_up = match_gain;
                f_open_extra = opened;
            }
            if M > 1 {
                // The next needle compare does not read this cell.
                let mut mm = _mm256_cmpeq_epi8(hay_fold, fold.n_fold[1]);
                for i in 1..M {
                    let mm_here = mm;
                    if i + 1 < M {
                        mm = _mm256_cmpeq_epi8(hay_fold, fold.n_fold[i + 1]);
                    }
                    // The horizontal gap does not read this row's match.
                    f_running = _mm256_max_epu8(
                        _mm256_subs_epu8(h_up, f_open_extra),
                        _mm256_subs_epu8(f_running, gap_ext_v),
                    );
                    let opened = _mm256_add_epi8(_mm256_and_si256(mm_here, gap_diff_v), gap_ext_v);
                    match_prev[i] = opened;
                    let mut match_gain = _mm256_and_si256(bonus_base, mm_here);
                    if need_case {
                        let eq_nc = _mm256_cmpeq_epi8(hay, fold.nc[i]);
                        match_gain =
                            _mm256_add_epi8(match_gain, _mm256_and_si256(eq_nc, fold.case_v));
                    }
                    let h_val = _mm256_max_epu8(match_gain, f_running);
                    h[i] = h_val;
                    h_up = h_val;
                    f_open_extra = opened;
                }
            }
            let next_col = _mm256_add_epi8(col_v, fold.col_one);
            let new_best = _mm256_max_epu8(best, h_up);
            let keep_old = _mm256_cmpeq_epi8(new_best, best);
            end_col_v = _mm256_blendv_epi8(col_v, end_col_v, keep_old);
            best = new_best;
            col_v = next_col;
            prev_delim = is_delim;

        // The next column is 32 bytes on. Its class is ready before this column's cells finish.
        let mut col_ptr = unsafe { soa.as_ptr().add(32) };
        macro_rules! classify {
            ($hay:expr) => {{
                // Digit and the ASCII range do not read the case fold.
                let is_digit =
                    _mm256_cmpgt_epi8(_mm256_add_epi8($hay, fold.add_digit), fold.thresh_digit);
                let below = _mm256_cmpgt_epi8($hay, fold.ones);
                let is_upper =
                    _mm256_cmpgt_epi8(_mm256_add_epi8($hay, fold.add_upper), fold.thresh_az);
                let hay_fold = _mm256_or_si256($hay, _mm256_and_si256(is_upper, fold.bit20));
                let is_letter =
                    _mm256_cmpgt_epi8(_mm256_add_epi8(hay_fold, fold.add_letter), fold.thresh_az);
                let alnum = _mm256_or_si256(is_letter, is_digit);
                let is_delim = _mm256_andnot_si256(alnum, below);
                (is_upper, hay_fold, is_letter, is_delim)
            }};
        }
        // One loop for a cased query and one for a lowercase query. The branch is not per column.
        macro_rules! score_cols {
            ($case:tt) => {
        let (mut hay, mut is_upper, mut hay_fold, mut is_letter, mut is_delim) = if n > 1 {
            let hay = unsafe { _mm256_loadu_si256(col_ptr as *const __m256i) };
            col_ptr = unsafe { col_ptr.add(32) };
            let (is_upper, hay_fold, is_letter, is_delim) = classify!(hay);
            (hay, is_upper, hay_fold, is_letter, is_delim)
        } else {
            (zero, zero, zero, zero, zero)
        };
        for j in 1..n {
            // Unreachable when the caller proved the score cannot reach 255.
            if CHECK_SAT && j & 3 == 0 && _mm256_movemask_epi8(_mm256_cmpeq_epi8(best, fold.ones)) == -1 {
                break;
            }
            let bonus_base;
            if $case {
                // Capitalization does not read the delimiter bonus.
                let is_lower = _mm256_andnot_si256(is_upper, is_letter);
                let cap_add = _mm256_and_si256(_mm256_and_si256(is_upper, prev_lower), fold.cap_v);
                let delim_bonus =
                    _mm256_and_si256(_mm256_andnot_si256(is_delim, prev_delim), fold.delim_v);
                bonus_base = _mm256_add_epi8(
                    _mm256_add_epi8(fold.match_score_v, delim_bonus),
                    cap_add,
                );
                prev_lower = is_lower;
            } else {
                let delim_bonus =
                    _mm256_and_si256(_mm256_andnot_si256(is_delim, prev_delim), fold.delim_v);
                bonus_base = _mm256_add_epi8(delim_bonus, fold.match_score_v);
            }
            // Class of the following column does not read this column's scores.
            let (hay_n, upper_n, fold_n, letter_n, delim_n) = if j + 1 < n {
                let hay_n = unsafe { _mm256_loadu_si256(col_ptr as *const __m256i) };
                col_ptr = unsafe { col_ptr.add(32) };
                let (upper_n, fold_n, letter_n, delim_n) = classify!(hay_n);
                (hay_n, upper_n, fold_n, letter_n, delim_n)
            } else {
                (zero, zero, zero, zero, zero)
            };
            // Register-carried recurrence; no 2D state or f array.
            let mut h_diag;
            // Row 0 writes these before a later row reads them.
            let mut h_up;
            let mut f_open_extra;
            let mut f_running = zero;

            // `case` is a literal so the bonus compare disappears on a lowercase query.
            // Row 0 has no cell above it, so the vertical gap stays zero.
            // Row 1's previous column does not depend on row 0's store.
            let row1_h = if M > 1 { h[1] } else { zero };
            let row1_e = if M > 1 { e[1] } else { zero };
            let row1_open = if M > 1 { match_prev[1] } else { zero };
            {
                        let h_left = h[0];
                        let e0 = e[0];
                        let e_open_extra = match_prev[0];
                        // The vertical gap does not read the needle compare.
                        let e_val = _mm256_max_epu8(
                            _mm256_subs_epu8(h_left, e_open_extra),
                            _mm256_subs_epu8(e0, gap_ext_v),
                        );
                        let mm = _mm256_cmpeq_epi8(hay_fold, fold.n_fold[0]);
                        let opened = _mm256_add_epi8(_mm256_and_si256(mm, gap_diff_v), gap_ext_v);
                        match_prev[0] = opened;
                        let mut match_gain = _mm256_and_si256(bonus_base, mm);
                        if $case {
                            let eq_nc = _mm256_cmpeq_epi8(hay, fold.nc[0]);
                            match_gain =
                                _mm256_add_epi8(match_gain, _mm256_and_si256(eq_nc, fold.case_v));
                        }
                        // `h_diag` is zero here, so the saturated subtract vanishes.
                        h_diag = h_left;
                        let diag = match_gain;
                        e[0] = e_val;
                        let h_val = _mm256_max_epu8(diag, e_val);
                        h[0] = h_val;
                        h_up = h_val;
                        f_open_extra = opened;
                    }
                    if M > 1 {
                        // The next needle compare does not read this cell.
                        let mut mm = _mm256_cmpeq_epi8(hay_fold, fold.n_fold[1]);
                        let mut h_left = row1_h;
                        let mut e_cur = row1_e;
                        let mut e_open_extra = row1_open;
                        for i in 1..M {
                            let h_ahead = if i + 1 < M { h[i + 1] } else { zero };
                            let e_ahead = if i + 1 < M { e[i + 1] } else { zero };
                            let open_ahead = if i + 1 < M { match_prev[i + 1] } else { zero };
                            // The next needle compare does not read this cell.
                            let mm_here = mm;
                            if i + 1 < M {
                                mm = _mm256_cmpeq_epi8(hay_fold, fold.n_fold[i + 1]);
                            }
                            // The case compare does not read this cell. A lowercase query drops it.
                            let eq_nc = if $case {
                                _mm256_cmpeq_epi8(hay, fold.nc[i])
                            } else {
                                zero
                            };
                            // Mismatch and both gaps do not read the case bonus.
                            let miss = _mm256_subs_epu8(h_diag, mismatch_v);
                            // The base match add does not read the case bonus.
                            let mut hit = _mm256_adds_epu8(h_diag, bonus_base);
                            let e_val = _mm256_max_epu8(
                                _mm256_subs_epu8(h_left, e_open_extra),
                                _mm256_subs_epu8(e_cur, gap_ext_v),
                            );
                            f_running = _mm256_max_epu8(
                                _mm256_subs_epu8(h_up, f_open_extra),
                                _mm256_subs_epu8(f_running, gap_ext_v),
                            );
                            if $case {
                                hit = _mm256_adds_epu8(hit, _mm256_and_si256(eq_nc, fold.case_v));
                            } else {
                                let _ = eq_nc;
                            }
                            let diag = _mm256_blendv_epi8(miss, hit, mm_here);

                            h_diag = h_left;
                            let opened = _mm256_add_epi8(_mm256_and_si256(mm_here, gap_diff_v), gap_ext_v);
                            match_prev[i] = opened;
                            e[i] = e_val;

                            let h_val = _mm256_max_epu8(_mm256_max_epu8(diag, e_val), f_running);
                            h[i] = h_val;
                            h_up = h_val;
                            f_open_extra = opened;
                            h_left = h_ahead;
                            e_cur = e_ahead;
                            e_open_extra = open_ahead;
                        }
                    }

            let next_col = _mm256_add_epi8(col_v, fold.col_one);
            let new_best = _mm256_max_epu8(best, h_up);
            let keep_old = _mm256_cmpeq_epi8(new_best, best);
            end_col_v = _mm256_blendv_epi8(col_v, end_col_v, keep_old);
            best = new_best;
            col_v = next_col;

            prev_delim = is_delim;
            hay = hay_n;
            is_upper = upper_n;
            hay_fold = fold_n;
            is_letter = letter_n;
            is_delim = delim_n;
        }
            };
        }
        if need_case {
            score_cols!(true);
        } else {
            score_cols!(false);
        }

        let nout = out.len().min(end_cols.len()).min(32);
        unsafe {
            store_u8_lanes(best, &mut out[..nout]);
            store_u8_lanes(end_col_v, &mut end_cols[..nout]);
        }
    }

    // One store for a full 16-lane batch. A short batch copies only those lanes.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn store_u16_lanes(bytes: __m256i, out: &mut [u16]) {
        let n = out.len().min(16);
        if n == 0 {
            return;
        }
        unsafe {
            if n == 16 {
                _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, bytes);
            } else {
                let mut tmp = [0u16; 16];
                _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, bytes);
                std::ptr::copy_nonoverlapping(tmp.as_ptr(), out.as_mut_ptr(), n);
            }
        }
    }

    // Zero-extend 32 score bytes into the u16 outputs. A full batch is four stores.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn store_u8_lanes(bytes: __m256i, out: &mut [u16]) {
        let n = out.len().min(32);
        if n == 0 {
            return;
        }
        unsafe {
            let lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(bytes));
            let hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(bytes));
            if n >= 16 {
                _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, lo);
            } else {
                let mut tmp = [0u16; 16];
                _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, lo);
                std::ptr::copy_nonoverlapping(tmp.as_ptr(), out.as_mut_ptr(), n);
                return;
            }
            if n >= 32 {
                _mm256_storeu_si256(out.as_mut_ptr().add(16) as *mut __m256i, hi);
            } else if n > 16 {
                let mut tmp = [0u16; 16];
                _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, hi);
                std::ptr::copy_nonoverlapping(tmp.as_ptr(), out.as_mut_ptr().add(16), n - 16);
            }
        }
    }
}

// 16-lane u8 SW kernel for AArch64 NEON: same recurrence and exact
// semantics as the AVX2 u8 kernel (factored gap form, 1D state with
// register-carried f/mm, folded-needle single compare). NEON compares
// are natively unsigned, so no sign-bias trick is needed.
//
// # Safety
// Caller must be AArch64 (NEON is baseline there). Scores must fit in
// u8 (`u8_safe` domain). Windows must be at most 256 bytes: end
// tracking is byte-wide, so longer windows stay scalar by dispatch.
#[cfg(target_arch = "aarch64")]
pub(crate) mod neon {
    use super::*;
    use std::arch::aarch64::*;

    pub(crate) const LANES8: usize = 16;

    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn score_batch_neon_u8(
        needle: &[u8],
        soa: &[u8],
        max_len: usize,
        p: &SwParams,
        include_prefix_mask: u16,
        out: &mut [u16],
        end_cols: &mut [u16],
    ) {
        let m = needle.len().min(32);
        let n = max_len;
        if m == 0 || n == 0 {
            out.iter_mut().for_each(|s| *s = 0);
            end_cols.iter_mut().for_each(|s| *s = 0);
            return;
        }

        let zero = vdupq_n_u8(0);
        let ones = vdupq_n_u8(0xFF);
        let gap_ext_v = vdupq_n_u8(p.gap_extend as u8);
        let gap_diff_v = vdupq_n_u8(p.gap_open.wrapping_sub(p.gap_extend) as u8);
        let match_score_v = vdupq_n_u8(p.match_score as u8);
        let mismatch_v = vdupq_n_u8(p.mismatch_penalty as u8);
        let delim_v = vdupq_n_u8(p.delimiter_bonus as u8);
        let cap_v = vdupq_n_u8(p.capitalization_bonus as u8);
        let case_v = vdupq_n_u8(p.matching_case_bonus as u8);
        let prefix_v = vdupq_n_u8(p.prefix_bonus as u8);
        let bit20 = vdupq_n_u8(0x20);
        // Signed-wrap range tests; bytes outside the range wrap or stay below.
        let add_upper = vdupq_n_u8(127 - b'Z');
        let add_letter = vdupq_n_u8(127 - b'z');
        let thresh_az = vdupq_n_u8(101);
        let add_digit = vdupq_n_u8(127 - b'9');
        let thresh_digit = vdupq_n_u8(117);
        let ascii_limit = vdupq_n_u8(128);
        let col_one = vdupq_n_u8(1);
        let need_case = p.capitalization_bonus != 0 || p.matching_case_bonus != 0;

        let prefix_lanes = {
            let mut arr = [0u8; 16];
            for (i, a) in arr.iter_mut().enumerate() {
                *a = if include_prefix_mask & (1u16 << i) != 0 {
                    0xFF
                } else {
                    0
                };
            }
            vld1q_u8(arr.as_ptr())
        };

        let mut n_fold_v = [zero; 32];
        let mut nc_v = [zero; 32];
        for i in 0..m {
            let b = needle[i];
            n_fold_v[i] = vdupq_n_u8(if b.is_ascii_uppercase() { b + 32 } else { b });
            if need_case {
                nc_v[i] = vdupq_n_u8(b);
            }
        }

        // 1D in-place state: indices are compile-time after unrolling,
        // so no ping-pong dimension is needed for e, h, or match masks.
        let mut h = [zero; 32];
        let mut e = [zero; 32];
        let mut match_prev = [zero; 32];
        let mut prev_lower = zero;
        let mut prev_delim = zero;
        let mut last_row_max = zero;
        let mut end_col_v = zero;
        // End column increments; wraps like `j as u8`.
        let mut col_v = zero;

        for j in 0..n {
            // Saturated lanes stay saturated; checking every 4th column is exact.
            if j & 3 == 0 && vmaxvq_u8(last_row_max) == 0xFF {
                break;
            }
            let hay = vld1q_u8(soa.as_ptr().add(j * LANES8));
            let is_upper = vcgtq_s8(
                vreinterpretq_s8_u8(vaddq_u8(hay, add_upper)),
                vreinterpretq_s8_u8(thresh_az),
            );
            let hay_fold = vorrq_u8(hay, vandq_u8(is_upper, bit20));
            let is_letter = vcgtq_s8(
                vreinterpretq_s8_u8(vaddq_u8(hay_fold, add_letter)),
                vreinterpretq_s8_u8(thresh_az),
            );
            let is_digit = vcgtq_s8(
                vreinterpretq_s8_u8(vaddq_u8(hay, add_digit)),
                vreinterpretq_s8_u8(thresh_digit),
            );
            let alnum = vorrq_u8(is_letter, is_digit);
            let lt_128 = vcltq_u8(hay, ascii_limit);
            let is_delim = vbicq_u8(lt_128, alnum);
            let dm = vbicq_u8(prev_delim, is_delim);
            let delim_bonus = vandq_u8(dm, delim_v);

            let mut bonus_base = vaddq_u8(delim_bonus, match_score_v);
            if need_case {
                let is_lower = vbicq_u8(is_letter, is_upper);
                let cap_mask = vandq_u8(is_upper, prev_lower);
                bonus_base = vaddq_u8(bonus_base, vandq_u8(cap_mask, cap_v));
                prev_lower = is_lower;
            }
            if j == 0 {
                bonus_base = vaddq_u8(bonus_base, vandq_u8(prefix_v, prefix_lanes));
            }

            let mut h_diag = zero;
            let mut h_up = zero;
            let mut f_running = zero;
            let mut f_open_extra = zero;

            macro_rules! i_steps {
                ($case:expr) => {
                    for i in 0..m {
                        let mm = vceqq_u8(hay_fold, n_fold_v[i]);
                        let e_open_extra = match_prev[i];
                        let mm_extra = vandq_u8(mm, gap_diff_v);
                        match_prev[i] = mm_extra;

                        let mut match_gain = vandq_u8(bonus_base, mm);
                        if $case {
                            let eq_nc = vceqq_u8(hay, nc_v[i]);
                            match_gain = vaddq_u8(match_gain, vandq_u8(eq_nc, case_v));
                        }
                        let mis_pen = vbicq_u8(mismatch_v, mm);
                        let diag = vqaddq_u8(vqsubq_u8(h_diag, mis_pen), match_gain);

                        let h_left = h[i];
                        h_diag = h_left;

                        let e_pre = vmaxq_u8(vqsubq_u8(h_left, e_open_extra), e[i]);
                        let e_val = vqsubq_u8(e_pre, gap_ext_v);
                        e[i] = e_val;

                        let f_pre = vmaxq_u8(vqsubq_u8(h_up, f_open_extra), f_running);
                        f_running = vqsubq_u8(f_pre, gap_ext_v);

                        let h_val = vmaxq_u8(vmaxq_u8(diag, e_val), f_running);
                        h[i] = h_val;
                        h_up = h_val;
                        f_open_extra = mm_extra;
                    }
                };
            }
            if need_case {
                i_steps!(true);
            } else {
                i_steps!(false);
            }

            let gt_best = vcgtq_u8(h_up, last_row_max);
            end_col_v = vbslq_u8(gt_best, col_v, end_col_v);
            last_row_max = vmaxq_u8(last_row_max, h_up);
            col_v = vaddq_u8(col_v, col_one);

            prev_delim = is_delim;
        }

        let mut sb = [0u8; 16];
        let mut eb = [0u8; 16];
        vst1q_u8(sb.as_mut_ptr(), last_row_max);
        vst1q_u8(eb.as_mut_ptr(), end_col_v);
        let nout = out.len().min(end_cols.len()).min(16);
        for i in 0..nout {
            out[i] = sb[i] as u16;
            end_cols[i] = eb[i] as u16;
        }
    }

    // Test-only SoA builder (mirrors the avx2 test oracle layout).
    #[cfg(test)]
    pub(crate) fn build_soa16(haystacks: &[&[u8]], max_len: usize) -> Vec<u8> {
        let mut soa = vec![0u8; max_len * 16];
        for (p, h) in haystacks.iter().enumerate().take(16) {
            for (j, &b) in h.iter().enumerate() {
                soa[j * 16 + p] = b;
            }
        }
        soa
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> SwParams {
        SwParams::for_query(false)
    }

    // Ground truth: frizbee Matcher under fff's scoring config.
    fn frizbee_score(needle: &str, haystack: &str) -> u16 {
        // Ensure skip_sw is OFF for the ground-truth measurement

        let has_upper = needle.chars().any(|c| c.is_uppercase());
        let cfg = neo_frizbee::Config {
            max_typos: Some(6),
            sort: neo_frizbee::SortStrategy::Unsorted,
            scoring: neo_frizbee::Scoring {
                capitalization_bonus: if has_upper { 8 } else { 0 },
                matching_case_bonus: if has_upper { 4 } else { 0 },
                ..Default::default()
            },
            ..Default::default()
        };
        let mut m = neo_frizbee::Matcher::new(needle, &cfg);
        let matches = m.match_list(std::slice::from_ref(&haystack));
        if matches.is_empty() {
            0
        } else {
            // Less exact_match_bonus, which score_haystack excludes.
            let exact = needle.eq_ignore_ascii_case(haystack);
            let bonus = if exact { 8u16 } else { 0 };
            matches[0].score.saturating_sub(bonus)
        }
    }

    #[test]
    fn scalar_matches_frizbee_on_basic_cases() {
        let p = params();
        let cases: &[(&str, &str)] = &[
            ("abc", "abc"),
            ("abc", "aabbcc"),
            ("ab", "aXc"),
            ("score", "crates/fff-core/src/score.rs"),
            ("abc", "xyz"),
            ("a", "abc"),
            ("abc", "ab"),
            ("test", "src/test/main.rs"),
            ("f", "foo/bar/baz.txt"),
            ("fb", "foo/bar/baz.txt"),
            ("fbz", "foo/bar/baz.txt"),
            ("abc", "a-b-c"),
            ("abc", "aXbXc"),
            ("file", "src/file_picker.rs"),
        ];
        for &(n, h) in cases {
            let mine = score_one_scalar(n.as_bytes(), h.as_bytes(), &p, true);
            let theirs = frizbee_score(n, h);
            assert_eq!(
                mine, theirs,
                "needle={n:?} haystack={h:?}: mine={mine} frizbee={theirs}"
            );
        }
    }

    #[test]
    fn max_sw_bound_holds_on_random() {
        // Any violation kills pre-SW pruning; parity lets scalar coverage transfer.
        let mut state = 0xBEEFu32;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 16) as u16
        };
        let byte = move |next: &mut dyn FnMut() -> u16| {
            let r = next() % 6;
            if r == 0 {
                b'/'
            } else if r == 1 {
                b'A' + (next() % 26) as u8
            } else {
                b'a' + (next() % 26) as u8
            }
        };
        for _ in 0..8000 {
            let m = 1 + (next() % 32) as usize;
            let n = 1 + (next() % 64) as usize;
            let needle: Vec<u8> = (0..m).map(|_| byte(&mut next)).collect();
            let hay: Vec<u8> = (0..n).map(|_| byte(&mut next)).collect();
            let p = SwParams::for_query(next() % 2 == 0);
            let bound = p.max_sw_score(m);
            for inc in [false, true] {
                let (sc, _) = score_one_scalar_full(&needle, &hay, &p, inc);
                assert!(
                    sc as i32 <= bound,
                    "m={m} n={n} inc={inc} sc={sc} bound={bound}"
                );
            }
        }
    }

    #[test]
    fn scalar_matches_frizbee_no_prefix() {
        let p = params();
        let cases: &[(&str, &str)] = &[("abc", "abc"), ("bc", "abc"), ("score", "src/score.rs")];
        for &(n, h) in cases {
            let mine = score_one_scalar(n.as_bytes(), h.as_bytes(), &p, false);
            let theirs = frizbee_score(n, h);
            // Without prefix, scores may differ; check upper bound only.
            assert!(
                mine <= theirs + 12,
                "needle={n} haystack={h}: mine={mine} theirs={theirs}"
            );
        }
    }

    #[test]
    fn exact_match_scores_match_score() {
        let p = params();
        let s = score_one_scalar(b"abc", b"abc", &p, true);
        assert_eq!(s, 48, "abc/abc");
    }

    #[cfg(target_arch = "x86_64")]
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_u8_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let p = params();
        let batches: &[&[&str]] = &[
            &[
                "abc",
                "aabbcc",
                "aXc",
                "xyz",
                "abcabc",
                "a",
                "ab",
                "a-b-c",
                "src/score.rs",
                "foo/bar.txt",
                "test.rs",
                "main",
                "aaa",
                "bbb",
                "ccc",
                "ddd",
                "eee",
                "fff",
                "ggg",
                "hhh",
                "iii",
                "jjj",
                "kkk",
                "lll",
                "mmm",
                "nnn",
                "ooo",
                "ppp",
                "qqq",
                "rrr",
                "sss",
                "ttt",
            ],
            &[
                "file_picker",
                "score",
                "netedit",
                "tl",
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
                "g",
                "h",
                "i",
                "j",
                "k",
                "l",
                "m",
                "n",
                "o",
                "p",
                "q",
                "r",
                "s",
                "t",
                "u",
                "v",
                "w",
                "x",
                "y",
                "z",
                "0",
                "1",
            ],
        ];
        for batch in batches {
            let max_len = batch.iter().map(|h| h.len()).max().unwrap();
            let hays: Vec<&[u8]> = batch.iter().map(|h| h.as_bytes()).collect();
            let soa = super::avx2::build_soa(&hays, max_len, 32);
            let mut simd_out = vec![0u16; batch.len()];
            let mut simd_ecol = vec![0u16; batch.len()];
            let prefix_mask: u32 = if batch.len() >= 32 {
                u32::MAX
            } else {
                (1u32 << batch.len()) - 1
            };
            unsafe {
                super::avx2::score_batch_avx2_u8(
                    b"abc",
                    &soa,
                    max_len,
                    &p,
                    prefix_mask,
                    &mut simd_out,
                    &mut simd_ecol,
                );
            }
            for (k, h) in batch.iter().enumerate() {
                let (scalar_score, scalar_ecol) =
                    score_one_scalar_full(b"abc", h.as_bytes(), &p, true);
                assert_eq!(
                    simd_out[k], scalar_score,
                    "u8 batch[{k}]={h:?}: simd_score={} scalar_score={}",
                    simd_out[k], scalar_score
                );
                assert_eq!(
                    simd_ecol[k], scalar_ecol,
                    "u8 batch[{k}]={h:?}: simd_ecol={} scalar_ecol={}",
                    simd_ecol[k], scalar_ecol
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_u8_dispatch_matches_general_unrolled() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let p = params();
        for m in 1usize..=32 {
            let needle: Vec<u8> = (0..m).map(|i| b'a' + (i % 26) as u8).collect();
            let batch: Vec<String> = (0..32)
                .map(|i| {
                    let len = 4 + (i * 3 + m) % 40;
                    let bytes: Vec<u8> = (0..len).map(|j| b'a' + ((i + j) % 26) as u8).collect();
                    String::from_utf8(bytes).unwrap()
                })
                .collect();
            let max_len = batch.iter().map(|h| h.len()).max().unwrap();
            let hays: Vec<&[u8]> = batch.iter().map(|h| h.as_bytes()).collect();
            let soa = super::avx2::build_soa(&hays, max_len, 32);
            let mut out_a = vec![0u16; 32];
            let mut eco_a = vec![0u16; 32];
            let mut out_b = vec![0u16; 32];
            let mut eco_b = vec![0u16; 32];
            unsafe {
                super::avx2::score_batch_avx2_u8(
                    &needle,
                    &soa,
                    max_len,
                    &p,
                    u32::MAX,
                    &mut out_a,
                    &mut eco_a,
                );
                super::avx2::score_batch_avx2_u8_dispatch::<true>(
                    &needle,
                    &soa,
                    max_len,
                    &p,
                    u32::MAX,
                    &mut out_b,
                    &mut eco_b,
                    None,
                );
            }
            assert_eq!(out_a, out_b, "m={m} scores differ general vs unrolled");
            assert_eq!(eco_a, eco_b, "m={m} end_cols differ general vs unrolled");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn u8_matches_scalar_on_long_needles() {
        // Out-of-domain long needles must still agree with scalar exactly.
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let p = params();
        let needles: &[&[u8]] = &[
            b"internationalization",
            b"characterization",
            b"controller",
            b"file_picker",
            b"rename VARIABLE42",
        ];
        let hays: &[&[u8]] = &[
            b"src/i18n/internationalization.rs",
            b"crates/fff-core/src/characterization_layer.rs",
            b"sumo/dom0/controllers/NetworkController.cpp",
            b"lua/fff/file_picker/init.lua",
            b"a",
            b"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            b"Characterization_of_VARIABLE42_in_DOCS.md",
        ];
        for needle in needles {
            // Pad lanes past 8; shorter tails run scalar in production.
            let mut hays32: Vec<&[u8]> = hays.to_vec();
            while hays32.len() < 12 {
                hays32.extend_from_within(..);
            }
            hays32.truncate(24);
            let n = hays32.len();
            let max_len = hays32.iter().map(|h| h.len()).max().unwrap();
            let soa = super::avx2::build_soa(&hays32, max_len, 32);
            let mut simd_out = vec![0u16; n];
            let mut simd_ecol = vec![0u16; n];
            let pmask: u32 = 0b101010101010101010101010;
            unsafe {
                super::avx2::score_batch_avx2_u8(
                    needle,
                    &soa,
                    max_len,
                    &p,
                    pmask,
                    &mut simd_out,
                    &mut simd_ecol,
                );
            }
            for (k, h) in hays32.iter().enumerate() {
                let inc = pmask >> k & 1 == 1;
                let (scalar_score, scalar_ecol) = score_one_scalar_full(needle, h, &p, inc);
                if simd_out[k] != scalar_score || simd_ecol[k] != scalar_ecol {
                    println!(
                        "U8DIVERGE needle={:?} hay[{k}]={:?}: u8=({},{}) scalar=({},{})",
                        needle, h, simd_out[k], simd_ecol[k], scalar_score, scalar_ecol
                    );
                }
            }
            // Dispatch must equal the general kernel (production calls dispatch).
            let mut disp_out = vec![0u16; n];
            let mut disp_eco = vec![0u16; n];
            unsafe {
                super::avx2::score_batch_avx2_u8_dispatch::<true>(
                    needle,
                    &soa,
                    max_len,
                    &p,
                    pmask,
                    &mut disp_out,
                    &mut disp_eco,
                    None,
                );
            }
            for (k, h) in hays32.iter().enumerate() {
                if disp_out[k] != simd_out[k] || disp_eco[k] != simd_ecol[k] {
                    println!(
                        "DISPDIVERGE needle={:?} hay[{k}]={:?}: disp=({},{}) general=({},{})",
                        needle, h, disp_out[k], disp_eco[k], simd_out[k], simd_ecol[k]
                    );
                }
            }
            // Same inputs through u16 isolate divergence to u8.
            let soa16 = super::avx2::build_soa(&hays32, max_len, 16);
            let mut out16 = vec![0u16; n];
            let mut eco16 = vec![0u16; n];
            unsafe {
                super::avx2::score_batch_avx2(
                    needle,
                    &soa16,
                    max_len,
                    &p,
                    0b1010101010101010u16,
                    &mut out16,
                    &mut eco16,
                );
            }
            for (k, h) in hays32.iter().enumerate() {
                let inc = (0b1010101010101010u32 >> k) & 1 == 1;
                let (scalar_score, scalar_ecol) = score_one_scalar_full(needle, h, &p, inc);
                assert_eq!(
                    out16[k], scalar_score,
                    "needle={:?} hay[{k}]={:?}: u16={} scalar={}",
                    needle, h, out16[k], scalar_score
                );
                assert_eq!(
                    eco16[k], scalar_ecol,
                    "needle={:?} hay[{k}]={:?}: u16_ecol={} scalar_ecol={}",
                    needle, h, eco16[k], scalar_ecol
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn u8_fuzz_long_needles_all_kernels_agree() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let p = params();
        let mut state = 0xC0FFEEu32;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 16) as u8
        };
        let alphabet: &[u8] = b"abcdefABCDEF0123456789/._- \x80\xff\x01\x7f";
        for round in 0..300 {
            let m = 16 + (next() as usize % 9);
            let needle: Vec<u8> = (0..m)
                .map(|_| alphabet[next() as usize % alphabet.len()])
                .collect();
            let nlanes = 32;
            let hays: Vec<Vec<u8>> = (0..nlanes)
                .map(|_| {
                    let len = 20 + (next() as usize % 230);
                    (0..len)
                        .map(|_| alphabet[next() as usize % alphabet.len()])
                        .collect()
                })
                .collect();
            let hrefs: Vec<&[u8]> = hays.iter().map(|h| h.as_slice()).collect();
            let max_len = hrefs.iter().map(|h| h.len()).max().unwrap();
            let soa32 = super::avx2::build_soa(&hrefs, max_len, 32);
            let soa16 = super::avx2::build_soa(&hrefs[..16], max_len, 16);
            let mut o8 = vec![0u16; nlanes];
            let mut e8 = vec![0u16; nlanes];
            let mut o16 = vec![0u16; 16];
            let mut e16 = vec![0u16; 16];
            unsafe {
                super::avx2::score_batch_avx2_u8(
                    &needle,
                    &soa32,
                    max_len,
                    &p,
                    u32::MAX,
                    &mut o8,
                    &mut e8,
                );
                super::avx2::score_batch_avx2(
                    &needle,
                    &soa16,
                    max_len,
                    &p,
                    u16::MAX,
                    &mut o16,
                    &mut e16,
                );
            }
            for k in 0..nlanes {
                let (ss, se) = score_one_scalar_full(&needle, hrefs[k], &p, true);
                assert_eq!(o8[k], ss, "round {round} lane {k}: u8 vs scalar");
                let _ = se;
            }
            for k in 0..16 {
                let (ss, _) = score_one_scalar_full(&needle, hrefs[k], &p, true);
                assert_eq!(o16[k], ss, "round {round} lane {k}: u16 vs scalar");
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn u8_end_col_survives_high_scores() {
        // In-domain exact needles must agree; the loop pins the boundary plus margin.
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let p = params();
        for m in [5usize, 10, 11, 13, 15, 16, 18] {
            let needle: Vec<u8> = b"internationalization"
                .iter()
                .cycle()
                .take(m)
                .cloned()
                .collect();
            let mut hay = needle.clone();
            hay.extend_from_slice(b"_trailing_junk_0123456789");
            let n = 16;
            let hrefs: Vec<&[u8]> = vec![hay.as_slice(); n];
            let max_len = hay.len();
            let soa32 = super::avx2::build_soa(&hrefs, max_len, 32);
            let mut o8 = vec![0u16; n];
            let mut e8 = vec![0u16; n];
            unsafe {
                super::avx2::score_batch_avx2_u8(
                    &needle,
                    &soa32,
                    max_len,
                    &p,
                    u32::MAX,
                    &mut o8,
                    &mut e8,
                );
            }
            let (ss, se) = score_one_scalar_full(&needle, &hay, &p, true);
            assert_eq!(o8[0], ss, "score diverged at m={m}");
            assert_eq!(e8[0], se, "end_col diverged at m={m}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn u8_transient_band_agrees_with_scalar() {
        // Diag intermediates can exceed the final, so in-domain cells can wrap; biased fuzz hunts that band.
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut state = 0x48u32;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 16) as u8
        };
        let bonus_alpha = b"aBcDeF_/012";
        for round in 0..500 {
            let upper = round % 2 == 0;
            let p = SwParams::for_query(upper);
            let m = if upper {
                6 + (next() % 3) as usize
            } else {
                13 + (next() % 3) as usize
            };
            if !p.u8_safe(m) {
                continue;
            }
            let needle: Vec<u8> = (0..m)
                .map(|_| bonus_alpha[next() as usize % bonus_alpha.len()])
                .collect();
            let mut hay = Vec::new();
            for _ in 0..(1 + next() % 3) {
                hay.extend_from_slice(&needle[..(1 + next() as usize % m)]);
            }
            hay.extend_from_slice(b"_junk0123456789");
            let n = 16;
            let hrefs: Vec<&[u8]> = vec![hay.as_slice(); n];
            let soa32 = super::avx2::build_soa(&hrefs, hay.len(), 32);
            let mut o8 = vec![0u16; n];
            let mut e8 = vec![0u16; n];
            unsafe {
                super::avx2::score_batch_avx2_u8(
                    &needle,
                    &soa32,
                    hay.len(),
                    &p,
                    u32::MAX,
                    &mut o8,
                    &mut e8,
                );
            }
            let (ss, se) = score_one_scalar_full(&needle, &hay, &p, true);
            assert_eq!(
                o8[0], ss,
                "score diverged round={round} m={m} upper={upper}"
            );
            assert_eq!(
                e8[0], se,
                "end_col diverged round={round} m={m} upper={upper}"
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn u8_late_best_end_col_agrees() {
        // u8 while best-ends fit a byte; longer windows take u16. Both must agree with scalar.
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let p = params();
        let needle = b"controller".to_vec();
        assert!(p.u8_safe(needle.len()));
        // In-contract: best-ends 209/249 fit the byte column tracker.
        for &off in &[200usize, 240] {
            let mut hay = vec![b'q'; off + needle.len() + 8];
            hay[off..off + needle.len()].copy_from_slice(&needle);
            let n = 16;
            let hrefs: Vec<&[u8]> = vec![hay.as_slice(); n];
            let soa32 = super::avx2::build_soa(&hrefs, hay.len(), 32);
            let mut o8 = vec![0u16; n];
            let mut e8 = vec![0u16; n];
            unsafe {
                super::avx2::score_batch_avx2_u8(
                    &needle,
                    &soa32,
                    hay.len(),
                    &p,
                    u32::MAX,
                    &mut o8,
                    &mut e8,
                );
            }
            let (ss, se) = score_one_scalar_full(&needle, &hay, &p, true);
            assert_eq!(o8[0], ss, "score diverged off={off}");
            assert_eq!(e8[0], se, "end_col diverged off={off}");
        }
        // Handoff: best-ends past byte 255 go u16, still exact.
        for &off in &[260usize, 300, 511] {
            let mut hay = vec![b'q'; off + needle.len() + 8];
            hay[off..off + needle.len()].copy_from_slice(&needle);
            let n = 16;
            let hrefs: Vec<&[u8]> = vec![hay.as_slice(); n];
            let soa16 = super::avx2::build_soa(&hrefs, hay.len(), 16);
            let mut o16 = vec![0u16; n];
            let mut e16 = vec![0u16; n];
            unsafe {
                super::avx2::score_batch_avx2(
                    &needle,
                    &soa16,
                    hay.len(),
                    &p,
                    u16::MAX,
                    &mut o16,
                    &mut e16,
                );
            }
            let (ss, se) = score_one_scalar_full(&needle, &hay, &p, true);
            assert_eq!(o16[0], ss, "u16 score diverged off={off}");
            assert_eq!(e16[0], se, "u16 end_col diverged off={off}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn u8_agrees_with_scalar_through_safe_domain() {
        // Wherever u8_safe admits, u8 must equal scalar on exact-prefix hays.
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        for upper in [false, true] {
            let p = SwParams::for_query(upper);
            let base: Vec<u8> = if upper {
                b"INTERNATIONALIZATION".to_vec()
            } else {
                b"internationalization".to_vec()
            };
            for m in 1..=32usize {
                if !p.u8_safe(m) {
                    continue;
                }
                let needle: Vec<u8> = base.iter().cycle().take(m).cloned().collect();
                let mut hay = needle.clone();
                hay.extend_from_slice(b"_trailing_junk_0123456789");
                let n = 16;
                let hrefs: Vec<&[u8]> = vec![hay.as_slice(); n];
                let soa32 = super::avx2::build_soa(&hrefs, hay.len(), 32);
                let mut o8 = vec![0u16; n];
                let mut e8 = vec![0u16; n];
                unsafe {
                    super::avx2::score_batch_avx2_u8(
                        &needle,
                        &soa32,
                        hay.len(),
                        &p,
                        u32::MAX,
                        &mut o8,
                        &mut e8,
                    );
                }
                let (ss, se) = score_one_scalar_full(&needle, &hay, &p, true);
                assert_eq!(o8[0], ss, "score diverged upper={upper} m={m}");
                assert_eq!(e8[0], se, "end_col diverged upper={upper} m={m}");
            }
        }
    }

    #[test]
    fn u8_bound_is_sound() {
        let lo = params();
        assert!(lo.u8_safe(3));
        assert!(lo.u8_safe(14));
        assert!(lo.u8_safe(15));
        assert!(!lo.u8_safe(16));
        let hi = SwParams::for_query(true);
        assert!(hi.u8_safe(3));
        assert!(hi.u8_safe(8));
        assert!(!hi.u8_safe(9));
        assert!(!hi.u8_safe(14));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_u8_agrees_with_scalar() {
        // First-class ARM parity: NEON u8 kernel vs scalar oracle over
        // the u8_safe domain, exact-prefix plus biased-bonus fuzz
        // shapes (transient band included). Runs on ARM hardware.
        let mut state = 0x9E01u32;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 16) as u8
        };
        let alpha = b"aBcDeF_/012";
        for round in 0..300 {
            let upper = round % 2 == 0;
            let p = SwParams::for_query(upper);
            let m = if upper {
                3 + (next() % 6) as usize
            } else {
                5 + (next() % 11) as usize
            };
            if !p.u8_safe(m) {
                continue;
            }
            let needle: Vec<u8> = (0..m)
                .map(|_| alpha[next() as usize % alpha.len()])
                .collect();
            let mut hay = Vec::new();
            for _ in 0..(1 + next() % 3) {
                hay.extend_from_slice(&needle[..(1 + next() as usize % m)]);
            }
            hay.extend_from_slice(b"_junk0123456789");
            let n = 16;
            let hrefs: Vec<&[u8]> = vec![hay.as_slice(); n];
            let soa = super::neon::build_soa16(&hrefs, hay.len());
            let mut o = vec![0u16; n];
            let mut e = vec![0u16; n];
            unsafe {
                super::neon::score_batch_neon_u8(
                    &needle,
                    &soa,
                    hay.len(),
                    &p,
                    u16::MAX,
                    &mut o,
                    &mut e,
                );
            }
            let (ss, se) = score_one_scalar_full(&needle, &hay, &p, true);
            assert_eq!(
                o[0], ss,
                "neon score diverged round={round} m={m} upper={upper}"
            );
            assert_eq!(
                e[0], se,
                "neon end_col diverged round={round} m={m} upper={upper}"
            );
        }
    }

    #[test]
    fn stack_scalar_matches_vec_path() {
        let p = params();
        let mut state = 0xBEEFu32;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 16) as u8
        };
        for m in [1usize, 7, 15, 16, 31, 32, 33, 40, 64, 100] {
            let needle: Vec<u8> = (0..m)
                .map(|i| b'a' + (next().wrapping_add(i as u8) % 5))
                .collect();
            let hay: Vec<u8> = (0..60).map(|_| b'a' + (next() % 5)).collect();
            let n = String::from_utf8_lossy(&needle).to_string();
            let h = String::from_utf8_lossy(&hay).to_string();
            let flipped: Vec<u8> = n.as_bytes().iter().copied().map(flip_case).collect();
            let stack = if m <= 32 {
                // Stack path only valid for m<=32 (fixed [T; 33] rows).
                score_one_scalar_stack(n.as_bytes(), &flipped, h.as_bytes(), &p, true)
            } else {
                score_one_scalar_full(n.as_bytes(), h.as_bytes(), &p, true)
            };
            let vec = score_one_scalar_vec(n.as_bytes(), &flipped, h.as_bytes(), &p, true);
            assert_eq!(stack.0, vec.0, "score m={m} needle={n:?}");
            assert_eq!(stack.1, vec.1, "end_col m={m} needle={n:?}");
        }
    }

    #[test]
    fn trimmed_no_prefix_matches_frizbee() {
        // Our trimmed SW must match frizbee's on frizbee's own prefilter windows.
        let p = params();
        let cases: &[(&str, &str)] = &[
            ("username.rs", "src/username.rs"),
            ("username.rs", "src/user_name_handler.rs"),
            ("item.rs", "src/models/item.rs"),
            ("item.rs", "src/item_sync/file.rs"),
            ("score", "crates/fff-core/src/score.rs"),
            ("score", "src/scoring/engine.rs"),
            ("file_picker_", "crates/fff-core/src/file_picker.rs"),
            ("usrnmea", "src/username/username.rs"),
            ("usrnmea", "src/username/handler.rs"),
            ("abc", "abc"),
            ("abc", "aabbcc"),
            ("test", "src/test/main.rs"),
        ];
        let has_upper = false;
        let cfg = neo_frizbee::Config {
            max_typos: Some(6),
            sort: neo_frizbee::SortStrategy::Unsorted,
            scoring: neo_frizbee::Scoring {
                capitalization_bonus: if has_upper { 8 } else { 0 },
                matching_case_bonus: if has_upper { 4 } else { 0 },
                ..Default::default()
            },
            ..Default::default()
        };
        for &(needle, full) in cases {
            let skip_matches = {
                let sm = neo_frizbee::with_skip_sw(|| {
                    let mut m = neo_frizbee::Matcher::new(needle, &cfg);
                    m.match_list(std::slice::from_ref(&full))
                });

                sm
            };

            let theirs = frizbee_score(needle, full);

            if skip_matches.is_empty() {
                assert_eq!(
                    theirs, 0,
                    "frizbee rejects {needle:?} on {full:?} but score={theirs}"
                );
                continue;
            }
            let start_adj = skip_matches[0].score as usize;
            let end_pos = skip_matches[0].end_col as usize;
            let end_clamped = end_pos.min(full.len());
            let start_clamped = start_adj.min(end_clamped);
            let trimmed = &full.as_bytes()[start_clamped..end_clamped];
            let include_prefix = start_clamped == 0;

            let mine = score_one_scalar(needle.as_bytes(), trimmed, &p, include_prefix);
            assert_eq!(
                mine,
                theirs,
                "needle={needle:?} full={full:?} window=({start_adj},{end_pos}) trimmed={:?} inc_prefix={include_prefix}: mine={mine} frizbee={theirs}",
                std::str::from_utf8(trimmed).unwrap_or("?")
            );
        }
    }

    #[test]
    fn no_match_scores_zero() {
        let p = params();
        let s = score_one_scalar(b"xyz", b"abc", &p, true);
        assert_eq!(s, 0, "xyz/abc");
    }

    #[cfg(target_arch = "x86_64")]
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let p = params();
        let batches: &[&[&str]] = &[
            &[
                "abc",
                "aabbcc",
                "aXc",
                "xyz",
                "abcabc",
                "a",
                "ab",
                "a-b-c",
                "src/score.rs",
                "foo/bar.txt",
                "test.rs",
                "main",
                "aaa",
                "bbb",
                "ccc",
                "ddd",
            ],
            &[
                "file_picker",
                "score",
                "netedit",
                "tl",
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
                "g",
                "h",
                "i",
                "j",
                "k",
                "l",
            ],
        ];
        for batch in batches {
            let max_len = batch.iter().map(|h| h.len()).max().unwrap();
            let hays: Vec<&[u8]> = batch.iter().map(|h| h.as_bytes()).collect();
            let soa = super::avx2::build_soa(&hays, max_len, 16);
            let mut simd_out = vec![0u16; batch.len()];
            let mut simd_ecol = vec![0u16; batch.len()];
            let prefix_mask: u16 = if batch.len() >= 16 {
                u16::MAX
            } else {
                (1u16 << batch.len()) - 1
            };
            unsafe {
                super::avx2::score_batch_avx2(
                    b"abc",
                    &soa,
                    max_len,
                    &p,
                    prefix_mask,
                    &mut simd_out,
                    &mut simd_ecol,
                );
            }
            for (k, h) in batch.iter().enumerate() {
                let (scalar_score, scalar_ecol) =
                    score_one_scalar_full(b"abc", h.as_bytes(), &p, true);
                assert_eq!(
                    simd_out[k], scalar_score,
                    "batch[{k}]={h:?}: simd_score={} scalar_score={}",
                    simd_out[k], scalar_score
                );
                assert_eq!(
                    simd_ecol[k], scalar_ecol,
                    "batch[{k}]={h:?}: simd_ecol={} scalar_ecol={}",
                    simd_ecol[k], scalar_ecol
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_u16_matches_scalar_long_needle() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let p = params();
        // m=20 exercises rows past 16.
        let needle = b"internationalization";
        let hays: &[&str] = &[
            "src/internationalization/mod.rs",
            "src/internationalizatio/mod.rs",
            "src/intenationalization/mod.rs",
            "xxxxxxxxxxxxxxxxxxxxxxxx",
            "src/other.rs",
        ];
        let max_len = hays.iter().map(|h| h.len()).max().unwrap();
        let hays_b: Vec<&[u8]> = hays.iter().map(|h| h.as_bytes()).collect();
        let soa = super::avx2::build_soa(&hays_b, max_len, 16);
        let mut simd_out = vec![0u16; hays.len()];
        let mut simd_ecol = vec![0u16; hays.len()];
        let prefix_mask: u16 = (1u16 << hays.len()) - 1;
        unsafe {
            super::avx2::score_batch_avx2(
                needle,
                &soa,
                max_len,
                &p,
                prefix_mask,
                &mut simd_out,
                &mut simd_ecol,
            );
        }
        for (k, h) in hays.iter().enumerate() {
            let (scalar_score, scalar_ecol) = score_one_scalar_full(needle, h.as_bytes(), &p, true);
            assert_eq!(
                simd_out[k], scalar_score,
                "lane[{k}]={h:?}: simd_score={} scalar_score={}",
                simd_out[k], scalar_score
            );
            assert_eq!(
                simd_ecol[k], scalar_ecol,
                "lane[{k}]={h:?}: simd_ecol={} scalar_ecol={}",
                simd_ecol[k], scalar_ecol
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn transpose_8x8_is_exact_matrix_transpose() {
        let mut src = [0u8; 8 * 11];
        for i in 0..8 {
            for j in 0..8 {
                src[i * 11 + j] = (i * 8 + j) as u8;
            }
        }
        let mut dst = [0xAAu8; 8 * 13];
        unsafe {
            super::avx2::transpose_8x8(src.as_ptr(), 11, dst.as_mut_ptr(), 13);
        }
        for i in 0..8 {
            for j in 0..8 {
                assert_eq!(
                    dst[j * 13 + i],
                    (i * 8 + j) as u8,
                    "dst[{j}][{i}] != src[{i}][{j}]"
                );
            }
        }
        assert_eq!(dst[8], 0xAA);
        assert_eq!(dst[12], 0xAA);
    }

    #[cfg(target_arch = "x86_64")]
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn soa_from_stage_matches_build_soa() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let cases: &[&[&str]] = &[
            &["a", "", "bb", "ccc", "dddd", "eeeee", "ffffff", "ggggggg"],
            &[
                "", "x", "yz", "1234", "56789", "abcdef", "ghijklm", "mnopqrs",
            ],
            &[
                "src/score.rs",
                "a",
                "",
                "very/long/path/name/that/spans/many/chunks/x",
                "zz",
            ],
            &[
                "01234567",
                "8",
                "89abcdef",
                "",
                "foo",
                "barbaz",
                "q",
                "wertyuiop",
            ],
        ];
        for batch in cases {
            for &stride in &[8usize, 16, 32] {
                let max_len = batch.iter().map(|h| h.len()).max().unwrap_or(0);
                let rows = max_len.div_ceil(8) * 8;
                if rows == 0 {
                    continue;
                }
                let mut stage = vec![0u8; stride * rows];
                for (k, h) in batch.iter().enumerate().take(stride) {
                    stage[k * rows..k * rows + h.len()].copy_from_slice(h.as_bytes());
                }
                let mut out = vec![0u8; rows * stride];
                unsafe {
                    super::avx2::soa_from_stage(&stage, rows, stride, stride, &mut out);
                }
                let hays: Vec<&[u8]> = batch.iter().map(|h| h.as_bytes()).collect();
                let expect = super::avx2::build_soa(&hays, rows, stride);
                assert_eq!(out, expect, "stride={stride} batch={batch:?}");
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn soa_from_stage_dense_roundtrip_layout() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let (rows, stride) = (16usize, 32usize);
        let mut stage = vec![0u8; rows * stride];
        for k in 0..stride {
            for j in 0..rows {
                stage[k * rows + j] = ((k * 16 + j) % 256) as u8;
            }
        }
        let mut out = vec![0u8; rows * stride];
        unsafe {
            super::avx2::soa_from_stage(&stage, rows, stride, stride, &mut out);
        }
        for k in 0..stride {
            for j in 0..rows {
                assert_eq!(
                    out[j * stride + k],
                    ((k * 16 + j) % 256) as u8,
                    "soa[{j}][{k}]"
                );
            }
        }
    }
}
