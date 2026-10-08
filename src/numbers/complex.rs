//! [`Complex`] numbers over any coefficient type.

use std::fmt::{self, Display};
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

use num_traits::{Float, Num, One, Zero};

use crate::numbers::Conj;

use super::{
    Arccos, Arcsin, Arctan, Coefficient, Cos, Cosh, Csc, Exp, Ln, Power, Recip, Sec, Sin, Sinh,
    Sqrt, Tan, Tanh,
};

/// A complex number `a + bi`, where `i² = -1`.
///
/// Complex numbers are the core machinery of rotations and much of
/// wave mechanics.
///
/// The coefficients may be any dtype ([`Coefficient`]); over integer
/// coefficients the arithmetic gives Gaussian integers (with Rust's standard
/// integer division/overflow behaviour). The analytic operations additionally
/// need floating-point coefficients.
// `repr(C)` fixes the field order to `real` then `im`, so `[Complex<f32>]` has
// the interleaved `[re, im, re, im, …]` layout the NEON FFT kernel reads.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Complex<T> {
    /// The real part, `a`.
    pub real: T,
    /// The imaginary part `b` — the coefficient of `i`.
    pub im: T,
}

impl<T> Complex<T> {
    /// Construct `real + im·i`.
    pub fn new(real: T, im: T) -> Self {
        Complex { real, im }
    }
}

impl<T: Coefficient> Complex<T> {
    /// A purely real number, `a + 0i`.
    pub fn constant(real: T) -> Self {
        Complex {
            real,
            im: T::zero(),
        }
    }

    /// A purely imaginary number, `0 + b·i`.
    pub fn imaginary(im: T) -> Self {
        Complex {
            real: T::zero(),
            im,
        }
    }

    /// The imaginary unit `i`.
    pub fn i() -> Self {
        Complex {
            real: T::zero(),
            im: T::one(),
        }
    }

    /// The squared magnitude `a² + b² = z·conj(z)`. Real-valued, and free of the
    /// square root that [`abs`](Complex::abs) needs.
    pub fn norm_sqr(self) -> T {
        self.real * self.real + self.im * self.im
    }
}

impl<T: Copy + Neg<Output = T>> Conj for Complex<T> {
    type Output = Self;

    /// The complex conjugate `a − b·i`.
    fn conj(&self) -> Self {
        Complex::new(self.real, -self.im)
    }
}

impl<T: Float + Coefficient> Complex<T> {
    /// The magnitude `|z| = √(a² + b²)`.
    pub fn abs(self) -> T {
        self.norm_sqr().sqrt()
    }

    /// The argument (phase) in radians, in `(−π, π]`.
    pub fn arg(self) -> T {
        self.im.atan2(self.real)
    }

    /// The principal square root: the root with non-negative real part, whose
    /// imaginary part takes the sign of `self.im`.
    pub fn sqrt(self) -> Self {
        let two = T::one() + T::one();
        let r = self.abs();
        // Clamp at zero: rounding can make these marginally negative.
        let re = ((r + self.real) / two).max(T::zero()).sqrt();
        let im = ((r - self.real) / two).max(T::zero()).sqrt();
        Complex::new(re, if self.im < T::zero() { -im } else { im })
    }
}

impl<T: Copy + Neg<Output = T>> Neg for Complex<T> {
    type Output = Complex<T>;

    /// `−(a + bi) = −a − bi`.
    fn neg(self) -> Self {
        Complex::new(-self.real, -self.im)
    }
}

impl<T: Coefficient> Add<T> for Complex<T> {
    type Output = Complex<T>;

    fn add(self, rhs: T) -> Self {
        self.add(Complex::constant(rhs))
    }
}

impl<T: Coefficient> Add for Complex<T> {
    type Output = Complex<T>;

    fn add(self, rhs: Self) -> Self {
        Complex::new(self.real + rhs.real, self.im + rhs.im)
    }
}

impl<T: Coefficient> Sub<T> for Complex<T> {
    type Output = Complex<T>;

    fn sub(self, rhs: T) -> Complex<T> {
        self.sub(Complex::constant(rhs))
    }
}

impl<T: Coefficient> Sub for Complex<T> {
    type Output = Complex<T>;

    /// `(a + bi) − (c + di) = (a − c) + (b − d)i`.
    fn sub(self, rhs: Self) -> Self {
        Complex::new(self.real - rhs.real, self.im - rhs.im)
    }
}

impl<T: Coefficient> Mul<T> for Complex<T> {
    type Output = Complex<T>;

    fn mul(self, rhs: T) -> Self {
        self.mul(Complex::constant(rhs))
    }
}

impl<T: Coefficient> Mul for Complex<T> {
    type Output = Complex<T>;

    /// `(a + bi)(c + di) = (ac − bd) + (ad + bc)i`, since `i² = −1`.
    fn mul(self, rhs: Self) -> Self {
        let real = self.real * rhs.real - self.im * rhs.im;
        let im = self.im * rhs.real + self.real * rhs.im;

        Complex::new(real, im)
    }
}

impl<T: Coefficient> Div<T> for Complex<T> {
    type Output = Complex<T>;

    fn div(self, rhs: T) -> Self {
        self.div(Complex::constant(rhs))
    }
}

impl<T: Coefficient> Div for Complex<T> {
    type Output = Complex<T>;

    /// Multiply through by the conjugate of the divisor:
    /// `(a + bi)/(c + di) = ((ac + bd) + (bc − ad)i) / (c² + d²)`.
    ///
    /// The denominator is `|rhs|² = c² + d²` (a sum — it is `rhs·conj(rhs)`).
    fn div(self, rhs: Self) -> Self::Output {
        let denom = rhs.norm_sqr();
        let real = (self.real * rhs.real + self.im * rhs.im) / denom;
        let im = (self.im * rhs.real - self.real * rhs.im) / denom;

        Complex::new(real, im)
    }
}

impl<T: Coefficient> Rem<T> for Complex<T> {
    type Output = Complex<T>;

    fn rem(self, rhs: T) -> Self::Output {
        self.rem(Complex::constant(rhs))
    }
}

impl<T: Coefficient> Rem for Complex<T> {
    type Output = Complex<T>;

    /// The remainder left by rounding the quotient to the nearest Gaussian
    /// integer: `self − round(self/rhs)·rhs`.
    ///
    /// Rounding (rather than truncating) keeps `|rem| ≤ |rhs|/√2`.
    fn rem(self, rhs: Self) -> Self {
        let q = self / rhs;
        let rounded = Complex::new(q.real.round(), q.im.round());

        self - rounded * rhs
    }
}

impl<T: Coefficient> Zero for Complex<T> {
    fn zero() -> Self {
        Complex::new(T::zero(), T::zero())
    }

    fn is_zero(&self) -> bool {
        self.real.is_zero() && self.im.is_zero()
    }
}

impl<T: Coefficient> One for Complex<T> {
    fn one() -> Self {
        Complex::new(T::one(), T::zero())
    }
}

impl<T: Coefficient> Num for Complex<T> {
    type FromStrRadixErr = T::FromStrRadixErr;

    /// Parses a purely real value; there is no standard literal syntax for a
    /// complex number in a given radix.
    fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        T::from_str_radix(s, radix).map(Complex::constant)
    }
}

/// Complex numbers are themselves valid coefficients, which is what lets a dual
/// number carry complex parts (`Dual<Complex<f64>>`).
impl<T: Coefficient> Coefficient for Complex<T> {
    fn supports_fractional_division() -> bool {
        T::supports_fractional_division()
    }

    /// The quotient rounded to the nearest Gaussian integer — the same
    /// locally-constant quotient [`Complex`]'s own `%` uses.
    fn trunc_div(self, rhs: Self) -> Self {
        (self / rhs).round()
    }

    fn round(self) -> Self {
        Complex::new(self.real.round(), self.im.round())
    }

    /// `|a + bi| = √(a² + b²)`, computed in `f64` so it works for any
    /// coefficient type.
    fn magnitude(self) -> f64 {
        self.real.magnitude().hypot(self.im.magnitude())
    }
}

impl<T: Float> Sin for Complex<T> {
    type Output = Complex<T>;
    /// `sin(a + bi) = sin(a)cosh(b) + i·cos(a)sinh(b)`.
    fn sin(self) -> Complex<T> {
        Complex::new(
            self.real.sin() * self.im.cosh(),
            self.real.cos() * self.im.sinh(),
        )
    }
}

impl<T: Float> Cos for Complex<T> {
    type Output = Complex<T>;
    /// `cos(a + bi) = cos(a)cosh(b) − i·sin(a)sinh(b)`.
    fn cos(self) -> Complex<T> {
        Complex::new(
            self.real.cos() * self.im.cosh(),
            -(self.real.sin() * self.im.sinh()),
        )
    }
}

impl<T: Float> Tan for Complex<T> {
    type Output = Complex<T>;
    /// `tan(a + bi) = (sin 2a + i·sinh 2b) / (cos 2a + cosh 2b)`.
    fn tan(self) -> Complex<T> {
        let two = T::one() + T::one();
        let (a2, b2) = (self.real * two, self.im * two);
        let denom = a2.cos() + b2.cosh();
        Complex::new(a2.sin() / denom, b2.sinh() / denom)
    }
}

impl<T: Float> Csc for Complex<T> {
    type Output = Complex<T>;
    /// `csc = 1/sin`. Multiplying by the conjugate,
    /// `csc(a + bi) = (2 sin(a)cosh(b) − 2i·cos(a)sinh(b)) / (cosh 2b − cos 2a)`.
    fn csc(self) -> Complex<T> {
        let two = T::one() + T::one();
        let denom = (self.im * two).cosh() - (self.real * two).cos();
        let real = (two * self.real.sin() * self.im.cosh()) / denom;
        let im = -(two * self.real.cos() * self.im.sinh()) / denom;
        Complex::new(real, im)
    }
}

impl<T: Float + Coefficient> Sec for Complex<T> {
    type Output = Complex<T>;

    fn sec(self) -> Complex<T> {
        Complex::<T>::one() / self.cos()
    }
}

impl<T: Float + Coefficient> Arcsin for Complex<T> {
    type Output = Complex<T>;

    fn arcsin(self) -> Complex<T> {
        let i = Complex::<T>::i();
        let one = Complex::<T>::one();

        -i * (i * self + (one - self * self).sqrt()).ln()
    }
}

impl<T: Float + Coefficient> Arccos for Complex<T> {
    type Output = Complex<T>;

    fn arccos(self) -> Complex<T> {
        let i = Complex::<T>::i();
        let one = Complex::<T>::one();

        -i * (self + i * (one - self * self).sqrt()).ln()
    }
}

impl<T: Float + Coefficient> Arctan for Complex<T> {
    type Output = Complex<T>;

    fn arctan(self) -> Complex<T> {
        let i = Complex::<T>::i();
        let one = Complex::<T>::one();
        let two = T::one() + T::one();

        let half_i = Complex::new(T::zero(), T::one() / two);

        half_i * ((one - i * self).ln() - (one + i * self).ln())
    }
}

impl<T: Float> Exp for Complex<T> {
    type Output = Complex<T>;

    fn exp(self) -> Complex<T> {
        let expx = self.real.exp();

        Complex::new(expx * self.im.cos(), expx * self.im.sin())
    }
}

impl<T: Float + Coefficient> Ln for Complex<T> {
    type Output = Complex<T>;

    fn ln(self) -> Complex<T> {
        Complex::new(self.abs().ln(), self.arg())
    }
}

impl<T: Float> Sinh for Complex<T> {
    type Output = Complex<T>;

    fn sinh(self) -> Complex<T> {
        Complex::new(
            self.real.sinh() * self.im.cos(),
            self.real.cosh() * self.im.sin(),
        )
    }
}

impl<T: Float> Cosh for Complex<T> {
    type Output = Complex<T>;

    fn cosh(self) -> Complex<T> {
        Complex::new(
            self.real.cosh() * self.im.cos(),
            self.real.sinh() * self.im.sin(),
        )
    }
}

impl<T: Float + Coefficient> Tanh for Complex<T> {
    type Output = Complex<T>;

    /// `tanh = sinh / cosh`.
    fn tanh(self) -> Complex<T> {
        self.sinh() / self.cosh()
    }
}

impl<T: Float + Coefficient> Sqrt for Complex<T> {
    type Output = Complex<T>;

    /// The principal square root; the inherent [`Complex::sqrt`] is the same
    /// root, reachable without importing this trait.
    fn sqrt(self) -> Complex<T> {
        Complex::sqrt(self)
    }
}

impl<T: Float + Coefficient> Power for Complex<T> {
    type Output = Complex<T>;

    /// `z^w = exp(w · ln z)` (principal branch, via [`Ln`]).
    fn power(self, exponent: Complex<T>) -> Complex<T> {
        (exponent * self.ln()).exp()
    }
}

impl<T: Float + Coefficient> Recip for Complex<T> {
    type Output = Complex<T>;

    fn recip(self) -> Complex<T> {
        let magnitude = self.real * self.real + self.im * self.im;
        let real = self.real / magnitude;
        let im = self.im.neg() / magnitude;
        Complex::new(real, im)
    }
}

impl<T: Display + PartialOrd + Zero> Display for Complex<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // A negative imaginary part already prints its own sign.
        if self.im >= T::zero() {
            write!(f, "{}+{}i", self.real, self.im)
        } else {
            write!(f, "{}{}i", self.real, self.im)
        }
    }
}
