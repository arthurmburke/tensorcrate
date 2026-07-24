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

/// The `math!` macro: a small mathematical language that expands to
/// statically-typed Rust using [`numbers`] and [`tensors`].
pub use rinterp_macros::math;

pub mod errors;
#[cfg(feature = "metal")]
pub mod metal;
pub mod numbers;
pub mod tensors;
