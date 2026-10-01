//! Work counters: how many kernels ran, how many bytes they moved and how many
//! fresh allocations they made.
//!
//! These exist to measure fusion, which trades exactly these three quantities
//! against each other. Every [`Kernels`](crate::tensors::Kernels) operation is
//! one *logical* kernel whichever backend runs it, so the counts from a `Host`
//! run and a `Metal` run of the same code agree; the Metal backend also counts
//! the command buffers it commits and the times it had to wait for them, which
//! is what a GPU launch really costs.
//!
//! Compiled only with the `counters` feature. Without it every recording call is
//! an empty inline function and [`snapshot`] does not exist, so production builds
//! pay nothing.
//!
//! The counters are per thread, like the Metal device, so a test measuring its
//! own work is not disturbed by tests running beside it.
//!
//! ```
//! # #[cfg(feature = "counters")] {
//! use tensorcrate::counters;
//! use tensorcrate::tensors::{Kernels, Host, Vector, BinaryOp};
//!
//! counters::reset();
//! let v = Vector::new(vec![1.0f32; 1024]);
//! let _ = Host::vector_elementwise(&v, &v, BinaryOp::Add);
//! let counts = counters::snapshot();
//! assert_eq!(counts.kernels, 1);
//! assert_eq!(counts.bytes, 3 * 1024 * 4); // two operands read, one result written
//! assert_eq!(counts.allocations, 1);
//! # }
//! ```

/// A reading of this thread's counters.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    /// Logical kernels: one per [`Kernels`](crate::tensors::Kernels) call, and
    /// one per fused program however many instructions it holds.
    pub kernels: u64,
    /// Bytes read from operands plus bytes written to results. Intermediates a
    /// fused program keeps in registers are not counted, since they never touch
    /// memory.
    pub bytes: u64,
    /// Result tensors allocated fresh, rather than written into an existing one.
    pub allocations: u64,
    /// Metal command buffers committed.
    pub command_buffers: u64,
    /// Times the CPU blocked on the GPU, whether to read a result or because
    /// the backlog of queued command buffers filled up.
    pub syncs: u64,
}

impl std::ops::Sub for Counts {
    type Output = Counts;

    fn sub(self, earlier: Counts) -> Counts {
        Counts {
            kernels: self.kernels - earlier.kernels,
            bytes: self.bytes - earlier.bytes,
            allocations: self.allocations - earlier.allocations,
            command_buffers: self.command_buffers - earlier.command_buffers,
            syncs: self.syncs - earlier.syncs,
        }
    }
}

#[cfg(feature = "counters")]
mod enabled {
    use std::cell::Cell;

    use super::Counts;

    thread_local! {
        static COUNTS: Cell<Counts> = const {
            Cell::new(Counts {
                kernels: 0,
                bytes: 0,
                allocations: 0,
                command_buffers: 0,
                syncs: 0,
            })
        };
    }

    fn update(f: impl FnOnce(&mut Counts)) {
        COUNTS.with(|cell| {
            let mut counts = cell.get();
            f(&mut counts);
            cell.set(counts);
        });
    }

    /// This thread's counts so far.
    pub fn snapshot() -> Counts {
        COUNTS.with(Cell::get)
    }

    /// Zero this thread's counts.
    pub fn reset() {
        COUNTS.with(|cell| cell.set(Counts::default()));
    }

    /// Run `f` and return what it did alongside its result.
    pub fn measure<R>(f: impl FnOnce() -> R) -> (R, Counts) {
        let before = snapshot();
        let result = f();
        (result, snapshot() - before)
    }

    pub(crate) fn kernel(bytes: usize, allocations: usize) {
        update(|counts| {
            counts.kernels += 1;
            counts.bytes += bytes as u64;
            counts.allocations += allocations as u64;
        });
    }

    #[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(dead_code))]
    pub(crate) fn command_buffer() {
        update(|counts| counts.command_buffers += 1);
    }

    #[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(dead_code))]
    pub(crate) fn sync() {
        update(|counts| counts.syncs += 1);
    }
}

#[cfg(feature = "counters")]
pub use enabled::{measure, reset, snapshot};
#[cfg(feature = "counters")]
#[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(unused_imports))]
pub(crate) use enabled::{command_buffer, kernel, sync};

#[cfg(not(feature = "counters"))]
mod disabled {
    #[inline(always)]
    pub(crate) fn kernel(_bytes: usize, _allocations: usize) {}

    #[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(dead_code))]
    #[inline(always)]
    pub(crate) fn command_buffer() {}

    #[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(dead_code))]
    #[inline(always)]
    pub(crate) fn sync() {}
}

#[cfg(not(feature = "counters"))]
#[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(unused_imports))]
pub(crate) use disabled::{command_buffer, kernel, sync};

/// Record one elementwise-shaped kernel: `inputs` operands of `len` `f32`s read
/// and one `len`-long result written into a fresh allocation.
#[inline(always)]
pub(crate) fn elementwise(len: usize, inputs: usize) {
    kernel((inputs + 1) * len * size_of::<f32>(), 1);
}
