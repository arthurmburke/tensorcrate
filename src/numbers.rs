//! The numeric types the language is built on.
//!
//! Everything here is a concrete Rust type — there is no runtime type tag. The
//! [`Coefficient`] trait is what a numeric element must satisfy, and it is
//! implemented for the primitive integers and floats, for [`Complex`], and for
//! [`Dual`]. That closure is deliberate: it lets the extensions nest, so
//! `Dual<Complex<f64>>` (a dual number with complex coefficients) and
//! `Matrix<Dual<f64>, R, C>` both work.

use std::fmt::{self, Display};
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

use num_traits::{Float, Num, One, Zero};

/// What a numeric element must provide: the primitive integers and floats,
/// [`Complex`], and [`Dual`] all implement it.
///
/// Because `Complex` and `Dual` are themselves coefficients, the extensions
/// nest — `Dual<Complex<f64>>` and `Vector<Dual<f64>, N>` are ordinary types.
pub trait Coefficient: Num + Copy {
    /// The quotient truncated toward zero, `trunc(self / rhs)`. This is the
    /// (locally constant) integer quotient underlying `%`.
    fn trunc_div(self, rhs: Self) -> Self;

    /// Round to the nearest whole value (the identity for integer types).
    fn round(self) -> Self;

    /// A real, comparable size for this value. Used to choose pivots in
    /// [`Matrix`](crate::tensors::Matrix) elimination, where the element type
    /// itself may have no ordering (complex numbers, for instance).
    fn magnitude(self) -> f64;
}

macro_rules! int_coefficient {
    ($($t:ty),+ $(,)?) => {$(
        impl Coefficient for $t {
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

/// A numeric type usable for computing the result of `sin`.
pub trait Sin {
    type Output;

    /// The result of sin(x).
    fn sin(self) -> <Self as Sin>::Output;
}

/// A numeric type usable for computing the result of `cos`.
pub trait Cos {
    type Output;

    /// The result of cos(x).
    fn cos(self) -> <Self as Cos>::Output;
}

/// A numeric type usable for computing the result of `tan`.
pub trait Tan {
    type Output;

    fn tan(self) -> <Self as Tan>::Output;
}

/// A numeric type usable for computing the result of `sec`.
pub trait Sec {
    type Output;

    fn sec(self) -> <Self as Sec>::Output;
}

/// A numeric type usable for computing the result of `csc`.
pub trait Csc {
    type Output;

    fn csc(self) -> <Self as Csc>::Output;
}

/// A numeric type usable for computing the result of `exp(x)`
pub trait Exp {
    type Output;

    fn exp(self) -> <Self as Exp>::Output;
}

/// A numeric type usable for computing the result of `arctan(x)`
pub trait Arctan {
    type Output;

    fn arctan(self) -> <Self as Arctan>::Output;
}

/// A numeric type usable for computing the result of `arcsin(x)`.
pub trait Arcsin {
    type Output;

    fn arcsin(self) -> <Self as Arcsin>::Output;
}

/// A numeric type usable for computing the result of `arccos(x)`.
pub trait Arccos {
    type Output;

    fn arccos(self) -> <Self as Arccos>::Output;
}

/// A numeric type usable for computing the natural logarithm `ln(x)`.
pub trait Ln {
    type Output;

    fn ln(self) -> <Self as Ln>::Output;
}

/// A numeric type usable for computing the hyperbolic sine `sinh(x)`.
pub trait Sinh {
    type Output;

    fn sinh(self) -> <Self as Sinh>::Output;
}

/// A numeric type usable for computing the hyperbolic cosine `cosh(x)`.
pub trait Cosh {
    type Output;

    fn cosh(self) -> <Self as Cosh>::Output;
}

/// A numeric type usable for computing the hyperbolic tangent `tanh(x)`.
pub trait Tanh {
    type Output;

    fn tanh(self) -> <Self as Tanh>::Output;
}

/// A numeric type usable for raising a base to a power, `x^y`.
pub trait Power<Rhs = Self> {
    type Output;

    fn power(self, exponent: Rhs) -> <Self as Power<Rhs>>::Output;
}

/// A numeric type usable for taking the reciprocal of a value
pub trait Recip {
    type Output;

    fn recip(self) -> <Self as Recip>::Output;
}

/// A dual number `a + bε`, where `ε² = 0` but `ε ≠ 0`.
///
/// Dual numbers are forward-mode automatic differentiation: evaluating a
/// function on the seed `x + 1·ε` yields `f(x) + f'(x)·ε`, because every term
/// that a second derivative would contribute carries an `ε²` factor and so
/// vanishes.
///
/// The coefficients may be any [`Coefficient`] — including [`Complex`], giving a
/// dual number with complex parts.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Dual<T> {
    /// The real part, `a`.
    pub real: T,
    /// The coefficient of the infinitesimal `ε`, `b`.
    pub dual: T,
}

impl<T> Dual<T> {
    /// Construct `real + dual·ε`.
    pub fn new(real: T, dual: T) -> Self {
        Dual { real, dual }
    }

    /// Convert both coefficients with `f` — e.g. widening a `Dual<f64>` into a
    /// `Dual<Complex<f64>>`.
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> Dual<U> {
        Dual {
            real: f(self.real),
            dual: f(self.dual),
        }
    }
}

impl<T: Neg<Output = T>> Neg for Dual<T> {
    type Output = Dual<T>;

    /// `−(a + bε) = −a − bε`.
    fn neg(self) -> Self {
        Dual::new(-self.real, -self.dual)
    }
}

impl<T: Coefficient> Dual<T> {
    /// A constant `a + 0·ε`.
    pub fn constant(real: T) -> Self {
        Dual {
            real,
            dual: T::zero(),
        }
    }

    /// A differentiation seed `a + 1·ε`; the `ε` part then accumulates the
    /// derivative with respect to this variable.
    pub fn variable(real: T) -> Self {
        Dual {
            real,
            dual: T::one(),
        }
    }
}

impl<T: Coefficient> Add for Dual<T> {
    type Output = Dual<T>;

    /// `(a + bε) + (c + dε) = (a + c) + (b + d)ε`.
    fn add(self, rhs: Self) -> Self {
        Dual::new(self.real + rhs.real, self.dual + rhs.dual)
    }
}

impl<T: Coefficient> Sub for Dual<T> {
    type Output = Dual<T>;

    /// `(a + bε) − (c + dε) = (a − c) + (b − d)ε`.
    fn sub(self, rhs: Self) -> Self {
        Dual::new(self.real - rhs.real, self.dual - rhs.dual)
    }
}

impl<T: Coefficient> Mul for Dual<T> {
    type Output = Dual<T>;

    /// `(a + bε)(c + dε) = ac + (ad + bc)ε` — the `bd·ε²` term is zero.
    fn mul(self, rhs: Self) -> Self {
        Dual::new(
            self.real * rhs.real,
            self.real * rhs.dual + self.dual * rhs.real,
        )
    }
}

impl<T: Coefficient> Div for Dual<T> {
    type Output = Dual<T>;

    /// The quotient rule: `(a + bε)/(c + dε) = a/c + ((bc − ad)/c²)ε`.
    fn div(self, rhs: Self) -> Self::Output {
        Dual::new(
            self.real / rhs.real,
            (self.dual * rhs.real - self.real * rhs.dual) / (rhs.real * rhs.real),
        )
    }
}

impl<T: Coefficient> Rem for Dual<T> {
    type Output = Dual<T>;

    /// `(a + bε) % (c + dε)`.
    ///
    /// Writing remainder as `u − q·v` with `q = trunc(a/c)` (an integer, hence
    /// locally constant, so it contributes no `ε`), the real part is `a % c` and
    /// the `ε` part is `b − q·d`.
    fn rem(self, rhs: Self) -> Self {
        let q = self.real.trunc_div(rhs.real);
        Dual::new(self.real % rhs.real, self.dual - q * rhs.dual)
    }
}

impl<T: Coefficient> Zero for Dual<T> {
    fn zero() -> Self {
        Dual::new(T::zero(), T::zero())
    }

    fn is_zero(&self) -> bool {
        self.real.is_zero() && self.dual.is_zero()
    }
}

impl<T: Coefficient> One for Dual<T> {
    fn one() -> Self {
        Dual::new(T::one(), T::zero())
    }
}

impl<T: Coefficient> Num for Dual<T> {
    type FromStrRadixErr = T::FromStrRadixErr;

    /// Parses a constant (no `ε` part); there is no literal syntax for a dual
    /// number in a given radix.
    fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        T::from_str_radix(s, radix).map(Dual::constant)
    }
}

/// Dual numbers are themselves valid coefficients, so a tensor can hold them
/// (and a dual can nest for higher-order work).
impl<T: Coefficient> Coefficient for Dual<T> {
    /// The quotient is locally constant, so it carries no `ε` part.
    fn trunc_div(self, rhs: Self) -> Self {
        Dual::constant(self.real.trunc_div(rhs.real))
    }

    fn round(self) -> Self {
        Dual::new(self.real.round(), self.dual.round())
    }

    /// The infinitesimal part does not affect magnitude.
    fn magnitude(self) -> f64 {
        self.real.magnitude()
    }
}

// Analytic operations on dual numbers apply the chain rule: for a function `f`,
// `f(a + bε) = f(a) + f'(a)·b·ε`. Each impl pairs `f` with its derivative `f'`.
// The `Float` bound gives every function/derivative we need (including `sqrt`
// for arcsin/arccos and negation), and covers exactly the meaningful cases —
// dual numbers with `f32`/`f64` coefficients.

impl<T: Float> Sin for Dual<T> {
    type Output = Dual<T>;
    /// `d/dx sin = cos`.
    fn sin(self) -> Dual<T> {
        Dual::new(self.real.sin(), self.dual * self.real.cos())
    }
}

impl<T: Float> Cos for Dual<T> {
    type Output = Dual<T>;
    /// `d/dx cos = −sin`.
    fn cos(self) -> Dual<T> {
        Dual::new(self.real.cos(), -(self.dual * self.real.sin()))
    }
}

impl<T: Float> Tan for Dual<T> {
    type Output = Dual<T>;
    /// `d/dx tan = sec² = 1/cos²`.
    fn tan(self) -> Dual<T> {
        let cos = self.real.cos();
        Dual::new(self.real.tan(), self.dual / (cos * cos))
    }
}

impl<T: Float> Csc for Dual<T> {
    type Output = Dual<T>;
    /// `csc = 1/sin`, `d/dx csc = −cos/sin²`.
    fn csc(self) -> Dual<T> {
        let sin = self.real.sin();
        Dual::new(sin.recip(), -(self.dual * self.real.cos()) / (sin * sin))
    }
}

impl<T: Float> Sec for Dual<T> {
    type Output = Dual<T>;
    /// `sec = 1/cos`, `d/dx sec = sin/cos²`.
    fn sec(self) -> Dual<T> {
        let cos = self.real.cos();
        Dual::new(cos.recip(), self.dual * self.real.sin() / (cos * cos))
    }
}

impl<T: Float> Arcsin for Dual<T> {
    type Output = Dual<T>;
    /// `d/dx arcsin = 1/√(1 − x²)`.
    fn arcsin(self) -> Dual<T> {
        let denom = (T::one() - self.real * self.real).sqrt();
        Dual::new(self.real.asin(), self.dual / denom)
    }
}

impl<T: Float> Arccos for Dual<T> {
    type Output = Dual<T>;
    /// `d/dx arccos = −1/√(1 − x²)`.
    fn arccos(self) -> Dual<T> {
        let denom = (T::one() - self.real * self.real).sqrt();
        Dual::new(self.real.acos(), -(self.dual / denom))
    }
}

impl<T: Float> Arctan for Dual<T> {
    type Output = Dual<T>;
    /// `d/dx arctan = 1/(1 + x²)`.
    fn arctan(self) -> Dual<T> {
        let denom = T::one() + self.real * self.real;
        Dual::new(self.real.atan(), self.dual / denom)
    }
}

impl<T: Float> Exp for Dual<T> {
    type Output = Dual<T>;
    /// `d/dx exp = exp`.
    fn exp(self) -> Dual<T> {
        let e = self.real.exp();
        Dual::new(e, self.dual * e)
    }
}

impl<T: Float> Ln for Dual<T> {
    type Output = Dual<T>;
    /// `d/dx ln = 1/x`.
    fn ln(self) -> Dual<T> {
        Dual::new(self.real.ln(), self.dual / self.real)
    }
}

impl<T: Float> Sinh for Dual<T> {
    type Output = Dual<T>;
    /// `d/dx sinh = cosh`.
    fn sinh(self) -> Dual<T> {
        Dual::new(self.real.sinh(), self.dual * self.real.cosh())
    }
}

impl<T: Float> Cosh for Dual<T> {
    type Output = Dual<T>;
    /// `d/dx cosh = sinh`.
    fn cosh(self) -> Dual<T> {
        Dual::new(self.real.cosh(), self.dual * self.real.sinh())
    }
}

impl<T: Float> Tanh for Dual<T> {
    type Output = Dual<T>;
    /// `d/dx tanh = 1 − tanh²`.
    fn tanh(self) -> Dual<T> {
        let t = self.real.tanh();
        Dual::new(t, self.dual * (T::one() - t * t))
    }
}

impl<T: Float> Power for Dual<T> {
    type Output = Dual<T>;
    /// `(a + bε)^(c + dε)`. Via `x^y = exp(y·ln x)`, the real part is `a^c` and
    /// the `ε` part is `c·a^(c−1)·b + a^c·ln(a)·d` — so a constant exponent gives
    /// the power rule and a constant base gives the exponential rule.
    fn power(self, exponent: Dual<T>) -> Dual<T> {
        let (a, b) = (self.real, self.dual);
        let (c, d) = (exponent.real, exponent.dual);
        let real = a.powf(c);
        let dual = c * a.powf(c - T::one()) * b + real * a.ln() * d;
        Dual::new(real, dual)
    }
}

impl<T: Float + Coefficient> Sin for Dual<Complex<T>> {
    type Output = Dual<Complex<T>>;
    /// `d/dx sin = cos`.
    fn sin(self) -> Dual<Complex<T>> {
        Dual::new(self.real.sin(), self.dual * self.real.cos())
    }
}

impl<T: Float + Coefficient> Cos for Dual<Complex<T>> {
    type Output = Dual<Complex<T>>;
    /// `d/dx cos = −sin`.
    fn cos(self) -> Dual<Complex<T>> {
        Dual::new(self.real.cos(), -(self.dual * self.real.sin()))
    }
}

impl<T: Float + Coefficient> Tan for Dual<Complex<T>> {
    type Output = Dual<Complex<T>>;
    /// `d/dx tan = sec² = 1/cos²`.
    fn tan(self) -> Dual<Complex<T>> {
        let cos = self.real.cos();
        Dual::new(self.real.tan(), self.dual / (cos * cos))
    }
}

impl<T: Float + Coefficient> Csc for Dual<Complex<T>> {
    type Output = Dual<Complex<T>>;
    /// `csc = 1/sin`, `d/dx csc = −cos/sin²`.
    fn csc(self) -> Dual<Complex<T>> {
        let sin = self.real.sin();
        Dual::new(sin.recip(), -(self.dual * self.real.cos()) / (sin * sin))
    }
}

impl<T: Float + Coefficient> Sec for Dual<Complex<T>> {
    type Output = Dual<Complex<T>>;
    /// `sec = 1/cos`, `d/dx sec = sin/cos²`.
    fn sec(self) -> Dual<Complex<T>> {
        let cos = self.real.cos();
        Dual::new(cos.recip(), self.dual * self.real.sin() / (cos * cos))
    }
}

impl<T: Float + Coefficient> Arcsin for Dual<Complex<T>> {
    type Output = Dual<Complex<T>>;
    /// `d/dx arcsin = 1/√(1 − x²)`.
    fn arcsin(self) -> Dual<Complex<T>> {
        let denom = (Complex::<T>::one() - self.real * self.real).sqrt();
        Dual::new(self.real.arcsin(), self.dual / denom)
    }
}

impl<T: Float + Coefficient> Arccos for Dual<Complex<T>> {
    type Output = Dual<Complex<T>>;
    /// `d/dx arccos = −1/√(1 − x²)`.
    fn arccos(self) -> Dual<Complex<T>> {
        let denom = (Complex::<T>::one() - self.real * self.real).sqrt();
        Dual::new(self.real.arccos(), -(self.dual / denom))
    }
}

impl<T: Float + Coefficient> Arctan for Dual<Complex<T>> {
    type Output = Dual<Complex<T>>;
    /// `d/dx arctan = 1/(1 + x²)`.
    fn arctan(self) -> Dual<Complex<T>> {
        let denom = Complex::<T>::one() + self.real * self.real;
        Dual::new(self.real.arctan(), self.dual / denom)
    }
}

impl<T: Float + Coefficient> Exp for Dual<Complex<T>> {
    type Output = Dual<Complex<T>>;
    /// `d/dx exp = exp`.
    fn exp(self) -> Dual<Complex<T>> {
        let e = self.real.exp();
        Dual::new(e, self.dual * e)
    }
}

impl<T: Float + Coefficient> Ln for Dual<Complex<T>> {
    type Output = Dual<Complex<T>>;
    /// `d/dx ln = 1/x`.
    fn ln(self) -> Dual<Complex<T>> {
        Dual::new(self.real.ln(), self.dual / self.real)
    }
}

impl<T: Float + Coefficient> Sinh for Dual<Complex<T>> {
    type Output = Dual<Complex<T>>;
    /// `d/dx sinh = cosh`.
    fn sinh(self) -> Dual<Complex<T>> {
        Dual::new(self.real.sinh(), self.dual * self.real.cosh())
    }
}

impl<T: Float + Coefficient> Cosh for Dual<Complex<T>> {
    type Output = Dual<Complex<T>>;
    /// `d/dx cosh = sinh`.
    fn cosh(self) -> Dual<Complex<T>> {
        Dual::new(self.real.cosh(), self.dual * self.real.sinh())
    }
}

impl<T: Float + Coefficient> Tanh for Dual<Complex<T>> {
    type Output = Dual<Complex<T>>;
    /// `d/dx tanh = 1 − tanh²`.
    fn tanh(self) -> Dual<Complex<T>> {
        let t = self.real.tanh();
        Dual::new(t, self.dual * (Complex::<T>::one() - t * t))
    }
}

impl<T: Float + Coefficient> Power for Dual<Complex<T>> {
    type Output = Dual<Complex<T>>;
    /// `(a + bε)^(c + dε)`. Via `x^y = exp(y·ln x)`, the real part is `a^c` and
    /// the `ε` part is `c·a^(c−1)·b + a^c·ln(a)·d` — so a constant exponent gives
    /// the power rule and a constant base gives the exponential rule.
    fn power(self, exponent: Dual<Complex<T>>) -> Dual<Complex<T>> {
        let (a, b) = (self.real, self.dual);
        let (c, d) = (exponent.real, exponent.dual);
        let real = a.power(c);
        let dual = c * a.power(c - Complex::<T>::one()) * b + real * a.ln() * d;
        Dual::new(real, dual)
    }
}

impl<T: Display> Display for Dual<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} + {}ε", self.real, self.dual)
    }
}

/// A complex number `a + bi`, where `i² = -1`.
///
/// Complex numbers are the core machinery of rotations and much of
/// wave mechanics.
///
/// The coefficients may be any dtype ([`Coefficient`]); over integer
/// coefficients the arithmetic gives Gaussian integers (with Rust's standard
/// integer division/overflow behaviour). The analytic operations additionally
/// need floating-point coefficients.
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

impl<T: Copy + Neg<Output = T>> Complex<T> {
    /// The complex conjugate `a − b·i`.
    pub fn conj(self) -> Self {
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

// ---- the analytic ops for every number type ---------------------------------
//
// Floats keep their precision (`Output = Self`); integers widen to `f64`, since
// a transcendental of an integer is not an integer.

/// Implement a unary analytic op for `f32`, `f64` and every integer type. The
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
        unary_number_impls!(@int $Trait, $method, $x, $body, i8, u8, i16, u16, i32, u32, i64, u64);
    };
    (@int $Trait:ident, $method:ident, $x:ident, $body:expr, $($t:ty),+) => {$(
        impl $Trait for $t {
            type Output = f64;
            fn $method(self) -> f64 { let $x = self as f64; $body }
        }
    )+};
}

unary_number_impls!(Sin, sin, x, x.sin());
unary_number_impls!(Cos, cos, x, x.cos());
unary_number_impls!(Tan, tan, x, x.tan());
unary_number_impls!(Csc, csc, x, x.sin().recip());
unary_number_impls!(Sec, sec, x, x.cos().recip());
unary_number_impls!(Arcsin, arcsin, x, x.asin());
unary_number_impls!(Arccos, arccos, x, x.acos());
unary_number_impls!(Arctan, arctan, x, x.atan());
unary_number_impls!(Exp, exp, x, x.exp());
unary_number_impls!(Ln, ln, x, x.ln());
unary_number_impls!(Sinh, sinh, x, x.sinh());
unary_number_impls!(Cosh, cosh, x, x.cosh());
unary_number_impls!(Tanh, tanh, x, x.tanh());
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

macro_rules! power_int_impls {
    ($($t:ty),+) => {$(
        impl Power for $t {
            type Output = f64;
            fn power(self, exponent: $t) -> f64 { (self as f64).powf(exponent as f64) }
        }
    )+};
}
power_int_impls!(i8, u8, i16, u16, i32, u32, i64, u64);
