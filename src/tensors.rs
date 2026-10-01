//! Dynamically-shaped vectors and matrices.
//!
//! [`Vector<T>`] and [`Matrix<T>`] carry their dimensions as ordinary fields, so
//! a shape can be computed at runtime: reading a length from a file, sizing a
//! layer from a batch, or building a matrix whose extent nobody knows until the
//! program runs. Storage is a flat row-major [`Vec`], so the tensors live on the
//! heap, are [`Clone`] rather than [`Copy`], and hand out contiguous slices that
//! the SIMD and GPU kernels read directly.
//!
//! Shapes are checked where the operation happens. A matrix product
//! `(R×K)·(K′×C)` panics unless `K == K′`, and adding a `2×3` to a `3×2` panics
//! too — the message names both shapes. The operators `+ - * /` cannot return a
//! `Result` (their signatures are fixed by `std`), so every shape-dependent
//! operation is consistent with them and panics. The operations that can fail
//! for reasons other than shape keep returning [`Result`]: [`Matrix::inverse`],
//! because singularity is a property of the values, and [`chained_matmul`],
//! which validates a whole chain before multiplying anything.
//!
//! `+ - * /` are elementwise; the linear-algebra products are the named methods.
//! Each is implemented for owned and borrowed operands, so `&a + &b` leaves both
//! usable.
//!
//! Both types take a second parameter, the storage [`Backend`], which defaults
//! to [`Host`] — the `Vec` just described. On macOS with the `metal` feature,
//! tensors can instead be placed on the [`Metal`] backend, whose elements live
//! in GPU-shared memory so a chain of operations runs without copying between
//! CPU and GPU pools. The full GPU operation set is available for `f32`, `f16`
//! and `bf16`, each with its own compiled kernels. On the host every element
//! type works, and the layers built on [`Kernels`] — automatic
//! differentiation, optimizers, statistics, fused programs — are generic over
//! the floating-point ones (`f32`, `f64`, `f16`, `bf16`), with `f32` the default.
//!
//! `f16` and `bf16` are computed in, not just stored: an elementwise operation
//! rounds to the 16-bit type every time, while sums, dot and matrix products,
//! prefix sums and moments accumulate in `f32` and round once. That holds on
//! both backends.
//! [`Vector::to_backend`] and [`Matrix::to_backend`] move between the two; see
//! the [`backend`] module for the details.

pub mod analytic;
pub mod backend;
pub mod dual;
pub mod fused;
pub mod kernels;
pub mod tape;

mod chain;
mod fft;
mod matrix;
mod ops;
mod order;
mod shape;
mod vector;

#[cfg(target_os = "macos")]
mod accelerate_dispatch;
#[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
pub(crate) mod simd_dispatch;

/// The `Metal`-backed inherent operations. It declares no new types, so there is
/// nothing to re-export — naming the module is what puts the methods on the
/// tensors.
#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal_backend;

pub use analytic::Transcendental;
pub use backend::{Backend, Host};
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use backend::{Metal, MetalStorage};
pub use chain::{chained_matmul, chained_matmul_cost};
pub use dual::{
    DualMatrix, DualVector, gradient, gradient_wrt_matrix, jacobian, jacobian_wrt_matrix,
    matrix_gradient,
};
pub use kernels::{
    Analytic, Axis, BinaryOp, Compare, Family, Kernels, Ordered, Reduce, SortOrder, Statistic,
};
pub use matrix::Matrix;
pub use tape::{MatrixVar, ScalarVar, Tape, Var, VectorVar};
pub use vector::Vector;

/// Minimum scalar multiply-accumulates before a Host matrix product leaves the
/// generic loop. Accelerate and the architecture-specific SIMD kernels share
/// this gate so adding the macOS fast path does not change dispatch semantics.
const HOST_MATMUL_DISPATCH_OPS: usize = 512;
