//! A small mathematical language that compiles to statically-typed Rust.
//!
//! Write mathematics in a [`math!`] block and it expands, at compile time, into
//! ordinary Rust over the numeric types in [`numbers`] and [`tensors`]. There is
//! no interpreter and no dynamically-typed value: every expression has a
//! concrete Rust type that the compiler checks.
//!
//! ```
//! use rinterp::math;
//!
//! let z = math! {
//!     let x = 1 + 2i;
//!     let y = 1 - 2i;
//!     x * y
//! };
//! assert_eq!(z.to_string(), "5+0i");
//! ```

#![feature(generic_const_exprs)]
#![allow(incomplete_features)] 

/// The `math!` macro: a small mathematical language that expands to
/// statically-typed Rust using [`numbers`] and [`tensors`]. Matrix and
/// matrix/vector products use `@`, vector `*` vector is a dot product, and
/// analytic functions map elementwise over tensors.
pub use rinterp_macros::math;

pub mod errors;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;
pub mod numbers;
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub mod simd;
pub mod tensors;
