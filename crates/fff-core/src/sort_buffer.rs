use std::cell::RefCell;
use std::cmp::Ordering;
use std::mem::MaybeUninit;

thread_local! {
    static TLS_SORT_BUFFER: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static PREFIX_HEAP: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

// Stable glidesort. Heap scratch only after the 4KB stack path, capped like sort_by.
pub fn sort_with_buffer<T, F>(slice: &mut [T], compare: F)
where
    F: FnMut(&T, &T) -> Ordering,
{
    let n = slice.len();
    let size = std::mem::size_of::<T>();
    if size == 0 || n < SMALL_SORT || n / 2 <= STACK_SCRATCH_BYTES / size {
        glidesort::sort_by(slice, compare);
        return;
    }

    let elems = scratch_elems::<T>(n);
    TLS_SORT_BUFFER.with(|cell| {
        let mut buf = cell.borrow_mut();
        let align = std::mem::align_of::<T>();
        let bytes = elems.saturating_mul(size).saturating_add(align);
        let extra = bytes.saturating_sub(buf.capacity());
        if extra > 0 {
            buf.reserve(extra);
        }

        let ptr = buf.as_mut_ptr();
        let offset = ptr.align_offset(align);
        let typed = unsafe {
            // Capacity covers `offset + elems * size` with `offset < align`.
            std::slice::from_raw_parts_mut(ptr.add(offset).cast::<MaybeUninit<T>>(), elems)
        };
        glidesort::sort_with_buffer_by(slice, typed, compare);
    });
}

const SMALL_SORT: usize = 48;
const STACK_SCRATCH_BYTES: usize = 4096;
const FULL_ALLOC_MAX_BYTES: usize = 1024 * 1024;
const HALF_ALLOC_MAX_BYTES: usize = 1024 * 1024 * 1024;

fn scratch_elems<T>(n: usize) -> usize {
    let tlen = std::mem::size_of::<T>();
    let full_allowed = n.min(FULL_ALLOC_MAX_BYTES / tlen);
    let half_allowed = (n / 2).min(HALF_ALLOC_MAX_BYTES / tlen);
    let eighth_allowed = n / 8;
    full_allowed
        .max(half_allowed)
        .max(eighth_allowed)
        .max(SMALL_SORT)
}

const PREFIX_MAX: usize = 2048;

fn sorts_after<T, F>(i: u32, j: u32, slice: &[T], compare: &mut F) -> bool
where
    F: FnMut(&T, &T) -> Ordering,
{
    let (i, j) = (i as usize, j as usize);
    match compare(&slice[i], &slice[j]) {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => i > j,
    }
}

fn sift_worst<T, F>(heap: &mut [u32], mut i: usize, slice: &[T], compare: &mut F)
where
    F: FnMut(&T, &T) -> Ordering,
{
    loop {
        let left = i * 2 + 1;
        if left >= heap.len() {
            break;
        }
        let right = left + 1;
        let mut worst = left;
        if right < heap.len() && sorts_after(heap[right], heap[left], slice, compare) {
            worst = right;
        }
        if sorts_after(heap[worst], heap[i], slice, compare) {
            heap.swap(i, worst);
            i = worst;
        } else {
            break;
        }
    }
}

// First `k` match a full `sort_with_buffer` prefix; the tail is unspecified.
pub fn sort_prefix_with_buffer<T, F>(slice: &mut [T], k: usize, mut compare: F)
where
    F: FnMut(&T, &T) -> Ordering,
{
    let n = slice.len();
    let k = k.min(n);
    if k == 0 || n <= 1 {
        return;
    }
    if k == n || k > PREFIX_MAX || n > u32::MAX as usize {
        sort_with_buffer(slice, compare);
        return;
    }

    PREFIX_HEAP.with(|cell| {
        let mut heap = cell.borrow_mut();
        let heap_cap = heap.capacity();
        if heap_cap < k {
            heap.reserve(k - heap_cap);
        }
        heap.clear();
        heap.extend(0..k as u32);
        for i in (0..k / 2).rev() {
            sift_worst(&mut heap, i, slice, &mut compare);
        }
        for i in k..n {
            let i = i as u32;
            if sorts_after(heap[0], i, slice, &mut compare) {
                heap[0] = i;
                sift_worst(&mut heap, 0, slice, &mut compare);
            }
        }
        heap.sort_by(|&a, &b| {
            let (ia, ib) = (a as usize, b as usize);
            compare(&slice[ia], &slice[ib]).then(a.cmp(&b))
        });
        for out in 0..k {
            let src = heap[out] as usize;
            if src == out {
                continue;
            }
            slice.swap(out, src);
            for later in out + 1..k {
                if heap[later] as usize == out {
                    heap[later] = src as u32;
                    break;
                }
            }
        }
    });
}

// `order[i]` is the source index for slot `i`, a permutation of `0..len`.
pub(crate) fn apply_source_order<T>(items: &mut [T], order: &mut [u32]) {
    let n = items.len();
    debug_assert_eq!(order.len(), n);
    for start in 0..n {
        if order[start] as usize == start {
            continue;
        }
        // Each index is read once and written once; cycles of a permutation do not alias.
        unsafe {
            let temp = std::ptr::read(items.as_ptr().add(start));
            let mut dst = start;
            let mut src = order[start] as usize;
            while src != start {
                std::ptr::write(
                    items.as_mut_ptr().add(dst),
                    std::ptr::read(items.as_ptr().add(src)),
                );
                order[dst] = dst as u32;
                dst = src;
                src = order[src] as usize;
            }
            std::ptr::write(items.as_mut_ptr().add(dst), temp);
            order[dst] = dst as u32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_source_order_matches_gather() {
        let src = vec![10, 20, 30, 40, 50];
        let orders = [
            vec![0u32, 1, 2, 3, 4],
            vec![4, 3, 2, 1, 0],
            vec![2, 0, 1, 4, 3],
            vec![1, 2, 3, 4, 0],
        ];
        for order in orders {
            let mut items = src.clone();
            let expect: Vec<_> = order.iter().map(|&i| src[i as usize]).collect();
            let mut ord = order;
            apply_source_order(&mut items, &mut ord);
            assert_eq!(items, expect);
        }
    }

    #[test]
    fn test_sort_with_buffer() {
        let mut data = vec![5, 2, 8, 1, 9];
        sort_with_buffer(&mut data, |a, b| a.cmp(b));
        assert_eq!(data, vec![1, 2, 5, 8, 9]);
    }

    #[test]
    fn test_sort_by_key_with_buffer() {
        let mut data = vec![(1, 50), (2, 20), (3, 80), (4, 10), (5, 90)];
        sort_with_buffer(&mut data, |a, b| a.1.cmp(&b.1));
        assert_eq!(data, vec![(4, 10), (2, 20), (1, 50), (3, 80), (5, 90)]);
    }

    #[test]
    fn test_reverse_sort() {
        let mut data = vec![1, 2, 3, 4, 5];
        sort_with_buffer(&mut data, |a, b| b.cmp(a));
        assert_eq!(data, vec![5, 4, 3, 2, 1]);
    }

    #[test]
    fn test_empty_slice() {
        let mut data: Vec<i32> = vec![];
        sort_with_buffer(&mut data, |a, b| a.cmp(b));
        assert_eq!(data, Vec::<i32>::new());
    }

    #[test]
    fn test_single_element() {
        let mut data = vec![42];
        sort_with_buffer(&mut data, |a, b| a.cmp(b));
        assert_eq!(data, vec![42]);
    }

    #[test]
    fn test_with_duplicates() {
        let mut data = vec![3, 1, 4, 1, 5, 9, 2, 6, 5];
        sort_with_buffer(&mut data, |a, b| a.cmp(b));
        assert_eq!(data, vec![1, 1, 2, 3, 4, 5, 5, 6, 9]);
    }

    #[test]
    fn test_descending_order() {
        let mut data = vec![3, 1, 4, 1, 5, 9, 2, 6, 5];
        sort_with_buffer(&mut data, |a, b| b.cmp(a));
        assert_eq!(data, vec![9, 6, 5, 5, 4, 3, 2, 1, 1]);
    }

    #[test]
    fn test_simple_descending() {
        let mut data = vec![100, 300, 200];
        sort_with_buffer(&mut data, |a, b| b.cmp(a));
        assert_eq!(data[0], 300);
        assert_eq!(data[1], 200);
        assert_eq!(data[2], 100);
    }

    #[test]
    fn test_descending_tiebreak_large() {
        for n in [10usize, 50, 100, 101, 150, 200, 500] {
            let mut data: Vec<(i32, u64)> = (0..n)
                .map(|i| (0, (i * 7919 % (n.max(1))) as u64))
                .collect();
            sort_with_buffer(&mut data, |a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
            for w in data.windows(2) {
                assert!(
                    w[0].1 >= w[1].1,
                    "tie order broken at n={n}: {:?} before {:?}",
                    w[0],
                    w[1]
                );
            }
        }
    }
}
