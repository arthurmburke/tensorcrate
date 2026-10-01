//! The numeric traits for the primitive integers and floats, and `f16` and
//! `bf16`.

use super::{
    Arccos, Arcsin, Arctan, Coefficient, Cos, Cosh, Csc, Exp, Ln, Power, Real, Recip, Sec, Sin,
    Sinh, Sqrt, Tan, Tanh, bf16, f16,
};

macro_rules! int_coefficient {
    ($($t:ty),+ $(,)?) => {$(
        impl Coefficient for $t {
            fn supports_fractional_division() -> bool { false }

            // Integer division already truncates toward zero.
            fn trunc_div(self, rhs: Self) -> Self { self / rhs }

            fn round(self) -> Self { self }

            fn magnitude(self) -> f64 { (self as f64).abs() }
        }
    )+};
}
int_coefficient!(i8, u8, i16, u16, i32, u32, i64, u64);

macro_rules! float_coefficient {
    ($($t:ty),+ $(,)?) => {$(
        impl Coefficient for $t {
            fn trunc_div(self, rhs: Self) -> Self { (self / rhs).trunc() }

            fn round(self) -> Self { self.round() }

            fn magnitude(self) -> f64 { (self as f64).abs() }
        }
    )+};
}
float_coefficient!(f32, f64);

macro_rules! half_coefficient {
    ($($t:ty),+ $(,)?) => {$(
        impl Coefficient for $t {
            fn trunc_div(self, rhs: Self) -> Self {
                Self::from_f32((f32::from(self) / f32::from(rhs)).trunc())
            }

            fn round(self) -> Self {
                Self::from_f32(f32::from(self).round())
            }

            fn magnitude(self) -> f64 {
                f32::from(self).abs() as f64
            }
        }
    )+};
}
half_coefficient!(f16, bf16);

macro_rules! real {
    ($($t:ty),+ $(,)?) => {$(
        impl Real for $t {
            fn total_order(&self, other: &Self) -> std::cmp::Ordering {
                self.total_cmp(other)
            }
        }
    )+};
}
real!(f32, f64, f16, bf16);

// ---- the analytic ops for every number type ---------------------------------
//
// Floats keep their precision (`Output = Self`); integers widen to `f64`, since
// a transcendental of an integer is not an integer. The functions themselves
// come from `vmath`, so an `f32` here is exactly what the tensor kernels give.

/// Implement a unary analytic op for every float and every integer type. The
/// op is written once as an expression over a bound value `$x`.
macro_rules! unary_number_impls {
    ($Trait:ident, $method:ident, $x:ident, $body:expr) => {
        impl $Trait for f32 {
            type Output = f32;
            fn $method(self) -> f32 { let $x = self; $body }
        }
        impl $Trait for f64 {
            type Output = f64;
            fn $method(self) -> f64 { let $x = self; $body }
        }
        unary_number_impls!(@half $Trait, $method, $x, $body, f16, bf16);
        unary_number_impls!(@int $Trait, $method, $x, $body, i8, u8, i16, u16, i32, u32, i64, u64);
    };
    // The compact floats compute in `f32` and round once on the way out, which
    // is how every half-precision library evaluates a transcendental.
    (@half $Trait:ident, $method:ident, $x:ident, $body:expr, $($t:ty),+) => {$(
        impl $Trait for $t {
            type Output = $t;
            fn $method(self) -> $t { let $x = f32::from(self); <$t>::from_f32($body) }
        }
    )+};
    (@int $Trait:ident, $method:ident, $x:ident, $body:expr, $($t:ty),+) => {$(
        impl $Trait for $t {
            type Output = f64;
            fn $method(self) -> f64 { let $x = self as f64; $body }
        }
    )+};
}

unary_number_impls!(Sin, sin, x, crate::vmath::Elementary::sin(x));
unary_number_impls!(Cos, cos, x, crate::vmath::Elementary::cos(x));
unary_number_impls!(Tan, tan, x, crate::vmath::Elementary::tan(x));
unary_number_impls!(Csc, csc, x, crate::vmath::Elementary::sin(x).recip());
unary_number_impls!(Sec, sec, x, crate::vmath::Elementary::cos(x).recip());
unary_number_impls!(Arcsin, arcsin, x, crate::vmath::Elementary::asin(x));
unary_number_impls!(Arccos, arccos, x, crate::vmath::Elementary::acos(x));
unary_number_impls!(Arctan, arctan, x, crate::vmath::Elementary::atan(x));
unary_number_impls!(Exp, exp, x, crate::vmath::Elementary::exp(x));
unary_number_impls!(Ln, ln, x, crate::vmath::Elementary::ln(x));
unary_number_impls!(Sinh, sinh, x, crate::vmath::Elementary::sinh(x));
unary_number_impls!(Cosh, cosh, x, crate::vmath::Elementary::cosh(x));
unary_number_impls!(Tanh, tanh, x, crate::vmath::Elementary::tanh(x));
unary_number_impls!(Sqrt, sqrt, x, x.sqrt());
unary_number_impls!(Recip, recip, x, x.recip());

impl Power for f32 {
    type Output = f32;
    fn power(self, exponent: f32) -> f32 {
        self.powf(exponent)
    }
}

impl Power for f64 {
    type Output = f64;
    fn power(self, exponent: f64) -> f64 {
        self.powf(exponent)
    }
}

macro_rules! power_half_impls {
    ($($t:ty),+) => {$(
        impl Power for $t {
            type Output = $t;
            fn power(self, exponent: $t) -> $t {
                <$t>::from_f32(f32::from(self).powf(f32::from(exponent)))
            }
        }
    )+};
}
power_half_impls!(f16, bf16);

macro_rules! power_int_impls {
    ($($t:ty),+) => {$(
        impl Power for $t {
            type Output = f64;
            fn power(self, exponent: $t) -> f64 { (self as f64).powf(exponent as f64) }
        }
    )+};
}
power_int_impls!(i8, u8, i16, u16, i32, u32, i64, u64);
