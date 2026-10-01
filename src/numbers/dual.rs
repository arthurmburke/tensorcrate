//! [`Dual`] numbers: forward-mode automatic differentiation in one value.

use std::fmt::{self, Display};
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

use num_traits::{Float, Num, One, Zero};

use super::{
    Arccos, Arcsin, Arctan, Coefficient, Complex, Cos, Cosh, Csc, Exp, Ln, Power, Recip, Sec, Sin,
    Sinh, Sqrt, Tan, Tanh,
};

/// A dual number `a + bε`, where `ε² = 0` but `ε ≠ 0`.
///
/// Dual numbers are forward-mode automatic differentiation: evaluating a
/// function on the seed `x + 1·ε` yields `f(x) + f'(x)·ε`, because every term
/// that a second derivative would contribute carries an `ε²` factor and so
/// vanishes.
///
/// The coefficients may be any [`Coefficient`] — including [`Complex`], giving a
/// dual number with complex parts.
#[repr(C)]
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
    fn supports_fractional_division() -> bool {
        T::supports_fractional_division()
    }

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

impl<T: Float> Sqrt for Dual<T> {
    type Output = Dual<T>;
    /// `d/dx √x = 1/(2√x)`.
    fn sqrt(self) -> Dual<T> {
        let root = self.real.sqrt();
        Dual::new(root, self.dual / ((T::one() + T::one()) * root))
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

impl<T: Float + Coefficient> Sqrt for Dual<Complex<T>> {
    type Output = Dual<Complex<T>>;
    /// `d/dx √x = 1/(2√x)`, on the principal branch of the root.
    fn sqrt(self) -> Dual<Complex<T>> {
        let root = self.real.sqrt();
        let two = Complex::<T>::one() + Complex::<T>::one();
        Dual::new(root, self.dual / (two * root))
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
