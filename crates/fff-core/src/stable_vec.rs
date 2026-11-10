use std::alloc::{self, Layout};
use std::cell::UnsafeCell;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Vector that guarantees no re-alloc happening at runtime
pub(crate) struct StableVec<T> {
    inner: Arc<StableBuf<T>>,
}

struct StableBuf<T> {
    ptr: NonNull<UnsafeCell<T>>,
    cap: usize,
    /// Acquire/Release pairs publish appended elements to readers.
    len: AtomicUsize,
}

// SAFETY: the buffer owns `T` values and may move them across threads.
unsafe impl<T: Send> Send for StableBuf<T> {}
// SAFETY: shared access reads elements and appends new ones; moving a `T` into
// the buffer from one thread and dropping it on another requires `Send`.
unsafe impl<T: Send + Sync> Sync for StableBuf<T> {}

impl<T> Drop for StableBuf<T> {
    fn drop(&mut self) {
        let len = *self.len.get_mut();
        unsafe {
            for i in 0..len {
                std::ptr::drop_in_place((*self.ptr.as_ptr().add(i)).get());
            }
            if self.cap > 0 {
                let layout = Layout::array::<UnsafeCell<T>>(self.cap).expect("layout");
                alloc::dealloc(self.ptr.as_ptr().cast(), layout);
            }
        }
    }
}

impl<T> StableVec<T> {
    pub fn from_vec_with_reserve(mut vec: Vec<T>, extra: usize) -> Self {
        vec.reserve(extra);
        let cap = vec.capacity();
        let len = vec.len();

        let inner = if cap == 0 {
            StableBuf {
                ptr: NonNull::dangling(),
                cap: 0,
                len: AtomicUsize::new(0),
            }
        } else {
            // Take ownership of the Vec's buffer without running element
            // drops; we hand them off to the StableBuf.
            let mut vec = std::mem::ManuallyDrop::new(vec);
            let ptr = NonNull::new(vec.as_mut_ptr().cast::<UnsafeCell<T>>()).expect("non-null");
            StableBuf {
                ptr,
                cap,
                len: AtomicUsize::new(len),
            }
        };

        Self {
            inner: Arc::new(inner),
        }
    }

    /// Append. Returns `false` if capacity is exhausted (item dropped).
    ///
    /// Safe to call via `&self` as long as the caller holds the outer
    /// picker write lock (single-writer invariant).
    #[inline]
    pub fn push(&self, item: T) -> bool {
        let cap = self.inner.cap;
        let len = self.inner.len.load(Ordering::Acquire);
        if len >= cap {
            debug_assert!(
                false,
                "StableVec: push would exceed capacity ({len} at capacity {cap})"
            );
            tracing::error!(
                len,
                capacity = cap,
                "StableVec: capacity exhausted — dropping item to prevent reallocation"
            );
            return false;
        }
        unsafe {
            std::ptr::write(self.inner.ptr.as_ptr().add(len), UnsafeCell::new(item));
        }
        self.inner.len.store(len + 1, Ordering::Release);
        true
    }

    // this method is specifically private because you probably need to use
    // live_count if you are trying to access this method
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len.load(Ordering::Acquire)
    }

    /// Mutable element access for in-place field updates. Never shifts.
    ///
    /// Panics if an Arc clone (e.g. a post-scan snapshot) still aliases the
    /// buffer: a `&mut T` would invalidate its shared readers.
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if Arc::strong_count(&self.inner) != 1 {
            return None;
        }
        let len = self.inner.len.load(Ordering::Acquire);
        if index >= len {
            return None;
        }
        unsafe { Some(&mut *(*self.inner.ptr.as_ptr().add(index)).get()) }
    }

    #[inline]
    pub fn last(&self) -> Option<&T> {
        let len = self.len();
        if len == 0 {
            None
        } else {
            unsafe { Some(&*self.inner.ptr.as_ptr().add(len - 1).cast::<T>()) }
        }
    }

    /// Iterate mutably for in-place field updates. Never shifts storage.
    ///
    /// Panics if an Arc clone still aliases the buffer.
    #[inline]
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        assert_unique(&self.inner);
        let len = self.inner.len.load(Ordering::Acquire);
        unsafe {
            std::slice::from_raw_parts_mut(self.inner.ptr.as_ptr().cast::<T>(), len).iter_mut()
        }
    }
}

#[inline]
fn assert_unique<T>(inner: &Arc<StableBuf<T>>) {
    assert!(
        Arc::strong_count(inner) == 1,
        "StableVec: mutable access while a snapshot aliases the buffer"
    );
}

impl<T> Clone for StableVec<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for StableVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("StableVec").field(&self.len()).finish()
    }
}

impl<T> std::ops::Deref for StableVec<T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        let len = self.len();
        unsafe { std::slice::from_raw_parts(self.inner.ptr.as_ptr().cast::<T>(), len) }
    }
}

impl<T> std::ops::DerefMut for StableVec<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        assert_unique(&self.inner);
        let len = self.inner.len.load(Ordering::Acquire);
        unsafe { std::slice::from_raw_parts_mut(self.inner.ptr.as_ptr().cast::<T>(), len) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_mut_is_none_while_a_clone_exists() {
        let mut v = StableVec::from_vec_with_reserve(vec![1, 2], 0);
        let clone = v.clone();
        assert!(v.get_mut(0).is_none());
        drop(clone);
        assert_eq!(v.get_mut(0), Some(&mut 1));
    }

    #[test]
    fn push_is_visible_to_existing_clones() {
        let v = StableVec::from_vec_with_reserve(Vec::new(), 2);
        assert!(v.push(7));
        let clone = v.clone();
        assert_eq!(&*clone, &[7]);
        assert!(v.push(9));
        assert_eq!(&*clone, &[7, 9]);
    }
}
