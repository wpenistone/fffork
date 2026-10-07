use std::cell::RefCell;
use std::mem::MaybeUninit;

// Per-worker scratch buffers: bounded by the worker count, so concurrent
// sorts neither contend on a global mutex nor fall back to allocation.
thread_local! {
    static TLS_SORT_BUFFER: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

fn ensure_capacity(buf: &mut Vec<u8>, required: usize) {
    if buf.capacity() < required {
        buf.reserve(required.saturating_sub(buf.len()));
    }
}

pub fn sort_with_buffer<T, F>(slice: &mut [T], compare: F)
where
    F: FnMut(&T, &T) -> std::cmp::Ordering,
{
    TLS_SORT_BUFFER.with(|cell| {
        let mut buf = cell.borrow_mut();
        let align = std::mem::align_of::<MaybeUninit<T>>();
        let size = std::mem::size_of::<MaybeUninit<T>>();
        let required = slice.len().saturating_mul(size).saturating_add(align);
        ensure_capacity(&mut buf, required);

        let ptr = buf.as_mut_ptr();
        let offset = ptr.align_offset(align);
        let typed = unsafe {
            std::slice::from_raw_parts_mut(ptr.add(offset) as *mut MaybeUninit<T>, slice.len())
        };
        glidesort::sort_with_buffer_by(slice, typed, compare);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
