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
/// tensors, and `dtype = f32;` makes a host block compute in `f32`; `Host` with
/// `f64` is the default.
///
/// Chains of elementwise tensor operations are fused into single kernels as the
/// block expands, with identical results on the host; `fuse = false;` turns
/// that off for one block, and
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
pub mod persist;
pub mod projections;
#[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
pub mod simd;
pub mod statistics;
pub mod tensors;

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
