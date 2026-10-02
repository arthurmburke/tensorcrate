//! A small mathematical language that compiles to statically-typed Rust.
//!
//! Write mathematics in a [`math!`] block and it expands, at compile time, into
//! ordinary Rust over the numeric types in [`numbers`] and [`tensors`]. There is
//! no interpreter and no dynamically-typed value: every expression has a
//! concrete Rust type that the compiler checks.
//!
//! ```
//! use tensorcrate::math;
//!
//! let z = math! {
//!     let x = 1 + 2i;
//!     let y = 1 - 2i;
//!     x * y
//! };
//! assert_eq!(z.to_string(), "5+0i");
//! ```

/// The `math!` macro: a small mathematical language that expands to
/// statically-typed Rust using [`numbers`] and [`tensors`]. Matrix and
/// matrix/vector products use `@`, vector `*` vector is a dot product, `.*` is
/// explicit elementwise multiplication, and analytic and ordering functions
/// map over tensors. A leading `backend = Metal;` selects resident `f32` Metal
/// tensors, and `dtype = f32;` (or `f16`, `bf16`) makes a block compute in that
/// type; `Host` with `f64` is the default, and Metal computes in `f32`, `f16`
/// or `bf16`.
///
/// Chains of elementwise tensor operations are fused into single kernels as the
/// block expands, and each kernel is optimized — regrouping associative chains
/// unless `reassociate = false;` asks for results identical to the unfused
/// block. `fuse = false;` turns fusion off for one block, and
/// [`fused::with_mode`](crate::tensors::fused::with_mode) at runtime. See the
/// fusion section of the README.
pub use tensorcrate_macros::math;

mod compact;
pub mod counters;
pub mod errors;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;
pub mod numbers;
pub mod optim;
mod parallel;
#[doc(hidden)]
pub use parallel::set_host_threads;
pub mod persist;
pub mod projections;
#[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
pub mod simd;
pub mod statistics;
pub mod tensors;
mod vmath;

/// Support for the code `math!` expands to. Not part of the public API: the
/// names here may change with the macro.
#[doc(hidden)]
pub mod __private {
    /// Record one kernel a `math!` block fused on the host, so the
    /// [`counters`](crate::counters) see it like any other kernel.
    #[inline(always)]
    pub fn record_kernel(bytes: usize, allocations: usize) {
        crate::counters::kernel(bytes, allocations);
    }

    /// Element-instructions worth giving a thread of their own.
    const PARALLEL_WORK: usize = 256 * 1024;

    /// The fewest elements worth giving a thread, for elements costing `work`
    /// instructions each.
    fn grain(work: usize) -> usize {
        (PARALLEL_WORK / work.max(1)).max(1024)
    }

    /// `[f(0), …, f(len - 1)]`, computed on several threads when it is worth
    /// it: `work` is roughly the instructions one element costs. A fused
    /// `math!` kernel with one result.
    pub fn generate<T: Send>(len: usize, work: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
        let grain = grain(work);
        if len < 2 * grain || crate::parallel::threads() <= 1 {
            return (0..len).map(f).collect();
        }
        let mut out = Vec::with_capacity(len);
        crate::parallel::for_slices(&mut out.spare_capacity_mut()[..len], grain, |start, window| {
            for (offset, slot) in window.iter_mut().enumerate() {
                slot.write(f(start + offset));
            }
        });
        // SAFETY: every element below `len` was written.
        unsafe { out.set_len(len) };
        out
    }

    /// [`generate`] for a kernel with `N` results of one type, each element
    /// computing one of each.
    pub fn generate_many<T: Copy + Default + Send, const N: usize>(
        len: usize,
        work: usize,
        f: impl Fn(usize) -> [T; N] + Sync,
    ) -> [Vec<T>; N] {
        let mut outputs: [Vec<T>; N] = std::array::from_fn(|_| vec![T::default(); len]);
        let write = |outputs: &[crate::parallel::Shared<T>; N], range: std::ops::Range<usize>| {
            for i in range {
                for (output, value) in outputs.iter().zip(f(i)) {
                    // SAFETY: `i` is below `len`, and in this range only.
                    unsafe { *output.at(i) = value };
                }
            }
        };
        let shared = outputs
            .each_mut()
            .map(|output| crate::parallel::Shared(output.as_mut_ptr()));
        let grain = grain(work);
        if len < 2 * grain {
            write(&shared, 0..len);
        } else {
            crate::parallel::for_ranges(len, grain, 16, |range| write(&shared, range));
        }
        outputs
    }

    /// The bounds check `clamp` makes, for a fused `clamp` — which must still
    /// panic where the unfused one would.
    #[track_caller]
    pub fn assert_clamp_bounds<T: PartialOrd>(low: &T, high: &T) {
        assert!(
            !matches!(low.partial_cmp(high), Some(core::cmp::Ordering::Greater)),
            "clamp: the lower bound exceeds the upper one"
        );
    }
}
