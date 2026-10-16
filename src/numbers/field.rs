//! Real and complex floating-point fields used by iterative solvers.

use std::ops::Neg;

use super::{Coefficient, Complex, Conj};

/// A floating-point scalar with a real magnitude and a conjugate.
///
/// Solvers intentionally exclude integer and dual coefficient domains.
pub trait Field: Coefficient + Conj<Output = Self> + Neg<Output = Self> {
    /// Embed a real scalar in this field.
    fn from_real(value: f64) -> Self;
    /// Machine epsilon of the underlying real type.
    fn epsilon() -> f64;
}

macro_rules! real_field {
    ($($t:ty),* $(,)?) => {$(
        impl Field for $t {
            fn from_real(value: f64) -> Self { value as Self }
            fn epsilon() -> f64 { <$t>::EPSILON as f64 }
        }
        impl Field for Complex<$t> {
            fn from_real(value: f64) -> Self { Self::new(value as $t, 0.0) }
            fn epsilon() -> f64 { <$t>::EPSILON as f64 }
        }
    )*};
}

real_field!(f32, f64);
