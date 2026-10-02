//! Dynamically-shaped vectors, matrices and N-dimensional tensors.
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
//!
//! # N-dimensional tensors
//!
//! [`Tensor<T>`] holds up to [`MAX_RANK`] axes over the same flat row-major
//! storage a [`Vector`] uses, so converting between a tensor and a vector or
//! matrix — or reshaping one — moves the storage without copying it.
//! Reordering and slicing — `permute`, `transpose`, `narrow`, `slice`,
//! `select`, `split`, `chunk` — return a [`TensorView`]: the tensor's storage
//! read in place through a shape, strides and an offset. A view becomes a
//! tensor of its own with [`TensorView::contiguous`], one strided copy on the
//! tensor's backend; [`Tensor::concat`] and [`Tensor::stack`] assemble their
//! result from views the same way, and [`Tensor::write_slice`] writes one into
//! part of an existing tensor in place.
//!
//! The elementwise arithmetic, analytic functions and comparisons of
//! [`Kernels`] apply to tensors and views of one shape, and a tensor or a view
//! whose leading axes fold together is an input to a
//! [fused program](fused::Program). Shape errors panic, naming the operation
//! and the shapes, as everywhere in this module.
//!
//! ```
//! use tensorcrate::tensors::Tensor;
//!
//! let x = Tensor::from_vec(&[2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
//! let xt = x.transpose(0, 1); // a [3, 2] view: nothing copied
//! assert_eq!(xt.get(&[2, 1]), Some(6.0));
//! let y = &xt.contiguous() + xt; // the view is read in order for the kernel
//! assert_eq!(y.to_vec(), [2.0, 8.0, 4.0, 10.0, 6.0, 12.0]);
//! ```

pub mod analytic;
pub mod backend;
pub mod dual;
pub mod fused;
pub mod kernels;
pub mod tape;

mod chain;
mod fft;
pub(crate) mod layout;
mod matrix;
mod ops;
mod order;
mod shape;
mod tensor;
mod tensor_ops;
mod vector;
mod view;

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
    Transposed,
};
pub use layout::{AxisIndex, MAX_RANK};
pub use matrix::Matrix;
pub use tape::{MatrixVar, ScalarVar, Tape, Var, VectorVar};
pub use tensor::Tensor;
pub use vector::Vector;
pub use view::TensorView;

/// Minimum scalar multiply-accumulates before a Host matrix product leaves the
/// generic loop. Accelerate and the architecture-specific SIMD kernels share
/// this gate so adding the macOS fast path does not change dispatch semantics.
const HOST_MATMUL_DISPATCH_OPS: usize = 512;
