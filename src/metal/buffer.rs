//! [`MetalBuffer`]: a typed allocation in shared memory, and how the CPU
//! reads and writes it.

use std::marker::PhantomData;
use std::mem::{ManuallyDrop, size_of};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;

use super::device::{GPU, with_gpu};
use super::sync::sync_or_panic;

/// Copy `src` into the front of a shared buffer's storage.
pub(super) fn upload<T: Copy>(buffer: &ProtocolObject<dyn MTLBuffer>, src: &[T]) {
    let dst = buffer.contents().as_ptr() as *mut T;
    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
}

/// Read `len` floats out of the front of a shared buffer's storage.
pub(super) fn download(buffer: &ProtocolObject<dyn MTLBuffer>, len: usize) -> Vec<f32> {
    let src = buffer.contents().as_ptr() as *const f32;
    let mut out = vec![0.0f32; len];
    unsafe { std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), len) };
    out
}

/// A typed allocation in Apple-silicon shared memory.
///
/// Keeping intermediate values in this type avoids the upload/download copies
/// made by the convenience functions below. The CPU may read the allocation in
/// place with [`as_slice`](Self::as_slice) — shared storage is CPU-cached
/// memory, not a separate GPU pool — while matrix, elementwise, broadcast, and
/// FFT operations chain without ever leaving it.
///
/// This is the storage behind the [`Metal`](crate::tensors::Metal) tensor
/// backend, which pairs it with the extents to make a `Vector` or `Matrix`.
///
/// Metal objects are thread-affine, so this type intentionally is not `Send`.
/// Dropping one returns its allocation to the thread's buffer pool.
pub struct MetalBuffer<T = f32> {
    /// Returned to the pool by `Drop`, hence `ManuallyDrop`.
    pub(super) raw: ManuallyDrop<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub(super) len: usize,
    pub(super) marker: PhantomData<T>,
}

impl<T: Copy + 'static> MetalBuffer<T> {
    /// Allocate `len` values of shared storage, recycling a pooled allocation
    /// when one is big enough. The contents are unspecified, so every caller
    /// either uploads into it or has a kernel write every element.
    pub(crate) fn allocate(len: usize) -> Option<Self> {
        let bytes = len.checked_mul(size_of::<T>())?.max(1);
        with_gpu(|gpu| {
            let raw = gpu.acquire(bytes)?;
            Some(Self {
                raw: ManuallyDrop::new(raw),
                len,
                marker: PhantomData,
            })
        })
    }

    /// Allocate shared storage and initialize it from a CPU slice.
    pub fn from_slice(values: &[T]) -> Option<Self> {
        let buffer = Self::allocate(values.len())?;
        upload(&buffer.raw, values);
        Some(buffer)
    }

    /// A second allocation holding the same values, copied on the GPU timeline:
    /// queued behind any kernel still writing this one, and without a sync.
    pub(crate) fn duplicate(&self) -> Option<Self> {
        let copy = Self::allocate(self.len)?;
        if self.len != 0 {
            let bytes = self.len.checked_mul(size_of::<T>())?;
            with_gpu(|gpu| super::encode::encode_copy(gpu, &self.raw, &copy.raw, bytes))?;
        }
        Some(copy)
    }

    /// Number of stored values.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether this buffer contains no values.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Borrow the shared allocation as an ordinary slice, without copying.
    ///
    /// Operations are submitted without waiting, so this is one of the points
    /// where queued GPU work has to have actually happened; it blocks until it
    /// has. Every CPU read of a `Metal`-backed tensor comes through here.
    pub fn as_slice(&self) -> &[T] {
        with_gpu(|gpu| {
            sync_or_panic(gpu);
            Some(())
        })
        .expect("a Metal buffer cannot outlive its thread-local device");
        // SAFETY: `MTLStorageModeShared` memory is CPU-readable at
        // `contents()`, and holds `len` initialized values — a buffer is only
        // handed out after an upload or a kernel that writes every element. The
        // `sync` above drained every command buffer that could still be writing
        // it, and nothing on this thread can submit more while the borrow is
        // alive, so no GPU write is in flight.
        unsafe { std::slice::from_raw_parts(self.raw.contents().as_ptr().cast::<T>(), self.len) }
    }

    /// Copy the shared allocation into an ordinary CPU vector.
    pub fn to_vec(&self) -> Vec<T> {
        self.as_slice().to_vec()
    }

    /// The Metal allocation, for binding to a kernel.
    pub(crate) fn raw(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.raw
    }

    /// Borrow the shared allocation mutably, after waiting for every queued
    /// command buffer — any of which may still be reading or writing it.
    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        with_gpu(|gpu| {
            sync_or_panic(gpu);
            Some(())
        })
        .expect("a Metal buffer cannot outlive its thread-local device");
        // SAFETY: as for `as_slice`, and the exclusive borrow of `self` rules out
        // any other view of this allocation for as long as the slice lives.
        unsafe {
            std::slice::from_raw_parts_mut(self.raw.contents().as_ptr().cast::<T>(), self.len)
        }
    }
}

impl<T> Drop for MetalBuffer<T> {
    fn drop(&mut self) {
        // SAFETY: `raw` is live until here and this runs exactly once, so it is
        // never taken twice and nothing reads the field afterwards.
        let raw = unsafe { ManuallyDrop::take(&mut self.raw) };
        // Hand the allocation back for reuse. Metal objects are thread-affine
        // and this type is not `Send`, so this is the pool it came from.
        // `try_with`/`try_borrow_mut` cover the cases where the pool cannot take
        // it — thread-local teardown, or a drop during another allocation — and
        // then the allocation is simply released to Metal.
        let _ = GPU.try_with(|cell| {
            if let Some(Some(gpu)) = cell.get() {
                // Anything committed but not yet known to be finished may still
                // reference this allocation, so it cannot go straight back into
                // service.
                let fence = gpu.fence();
                if let Ok(mut pool) = gpu.pool.try_borrow_mut() {
                    pool.release(raw, fence);
                }
            }
        });
    }
}
