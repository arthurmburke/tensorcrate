//! [`FixedPoint`] decimals: exact money, prices and quantities.
//!
//! `FixedPoint<N>` stores a value as a whole number of `10⁻ᴺ` units in an `i64`, so
//! `FixedPoint<2>` counts cents and `FixedPoint<6>` millionths. Addition, subtraction,
//! comparison and remainder are exact. Multiplication and division widen to `i128` and round
//! once, to nearest with ties away from zero; [`mul_to`](FixedPoint::mul_to),
//! [`div_to`](FixedPoint::div_to) and [`rescale`](FixedPoint::rescale) choose both the result's
//! precision and the [`Rounding`] when a rule demands one (a fee rounded up to the cent).
//!
//! Overflow panics in every build: a value that silently wrapped would be a wrong price.
//!
//! It is a [`Coefficient`], so vectors and matrices of it use the generic host kernels (the SIMD
//! paths are float-only and defer to them), and [`Storable`](crate::persist::Storable), so
//! tensors of it persist.

use std::cmp::Ordering;
use std::fmt;
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};
use std::str::FromStr;

use num_traits::{Bounded, FromPrimitive, Num, NumCast, One, Signed, ToPrimitive, Zero};

use super::{Coefficient, Conj};

/// A decimal with exactly `N` digits after the point, held as an integer count of `10⁻ᴺ`.
///
/// `N` may be 0 through 18 (`10¹⁸` is the largest power of ten an `i64` holds); larger values
/// fail to compile where the type is used.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct FixedPoint<const N: usize>(i64);

/// How a result with more digits than its type keeps is cut to fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rounding {
    /// To the nearest representable value; halves away from zero.
    Nearest,
    /// Toward negative infinity.
    Floor,
    /// Toward positive infinity.
    Ceil,
    /// Toward zero (truncation).
    TowardZero,
}

/// Why a string is not a [`FixedPoint`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseFixedPointError {
    /// No digits.
    Empty,
    /// A character other than one sign, digits and one decimal point; or a radix other than 10.
    Invalid,
    /// Nonzero digits beyond the type's precision ([`FixedPoint::parse_rounded`] accepts them).
    Inexact,
    /// Too large for the type.
    Overflow,
}

impl fmt::Display for ParseFixedPointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Empty => "no digits",
            Self::Invalid => "not a decimal number",
            Self::Inexact => "more decimal places than the type keeps",
            Self::Overflow => "out of range",
        })
    }
}

impl std::error::Error for ParseFixedPointError {}

const fn pow10(n: usize) -> i64 {
    assert!(n <= 18, "FixedPoint<N> needs N <= 18");
    let mut value = 1i64;
    let mut i = 0;
    while i < n {
        value *= 10;
        i += 1;
    }
    value
}

/// `numerator / denominator` rounded as asked; `denominator` must be nonzero.
fn divide(numerator: i128, denominator: i128, rounding: Rounding) -> i128 {
    let (numerator, denominator) = if denominator < 0 { (-numerator, -denominator) } else { (numerator, denominator) };
    let quotient = numerator / denominator; // truncated toward zero
    let remainder = numerator % denominator;
    if remainder == 0 {
        return quotient;
    }
    let away = if numerator < 0 { quotient - 1 } else { quotient + 1 };
    match rounding {
        Rounding::TowardZero => quotient,
        Rounding::Floor => quotient.min(away),
        Rounding::Ceil => quotient.max(away),
        Rounding::Nearest => {
            if 2 * remainder.abs() >= denominator {
                away
            } else {
                quotient
            }
        }
    }
}

/// `10^n` as an `i128`, for rescaling across precisions (`n` ≤ 36).
fn pow10_wide(n: usize) -> i128 {
    10i128.checked_pow(n as u32).expect("FixedPoint rescale beyond 10^38")
}

/// Widens a raw count for exact intermediate arithmetic.
fn wide(raw: i64) -> i128 {
    raw.into()
}

fn narrow(value: i128) -> i64 {
    i64::try_from(value).expect("FixedPoint overflow")
}

impl<const N: usize> FixedPoint<N> {
    /// `10ᴺ`: the raw count that makes one whole unit.
    pub const SCALE: i64 = pow10(N);
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(Self::SCALE);
    pub const MIN: Self = Self(i64::MIN);
    pub const MAX: Self = Self(i64::MAX);
    /// The smallest positive value, `10⁻ᴺ`.
    pub const EPSILON: Self = Self(1);

    /// The value `raw · 10⁻ᴺ`.
    pub const fn from_raw(raw: i64) -> Self {
        Self(raw)
    }

    /// The count of `10⁻ᴺ` units.
    pub const fn raw(self) -> i64 {
        self.0
    }

    /// A whole number.
    pub const fn from_int(value: i64) -> Self {
        match value.checked_mul(Self::SCALE) {
            Some(raw) => Self(raw),
            None => panic!("FixedPoint overflow"),
        }
    }

    /// `value` scaled by `10ᴺ` and rounded as asked; `None` for NaN, infinities and values out
    /// of range. A float written with at most `N` decimals (`4087.93`) comes back as exactly that
    /// decimal under `Nearest`, as long as it has fewer than ~15 significant digits.
    pub fn from_f64(value: f64, rounding: Rounding) -> Option<Self> {
        if !value.is_finite() {
            return None;
        }
        let scaled = value * Self::SCALE as f64;
        let rounded = match rounding {
            Rounding::Nearest => scaled.round(),
            Rounding::Floor => scaled.floor(),
            Rounding::Ceil => scaled.ceil(),
            Rounding::TowardZero => scaled.trunc(),
        };
        // i64::MAX as f64 rounds up to 2^63, which is out of range.
        (rounded >= i64::MIN as f64 && rounded < i64::MAX as f64).then_some(Self(rounded as i64))
    }

    /// The nearest `f64`.
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / Self::SCALE as f64
    }

    /// This value at precision `M`, rounded as asked when `M < N`.
    pub fn rescale<const M: usize>(self, rounding: Rounding) -> FixedPoint<M> {
        FixedPoint(narrow(rescale_raw(wide(self.0), N, M, rounding)))
    }

    /// `self · rhs` at precision `K`, from the exact product, rounded once.
    pub fn mul_to<const M: usize, const K: usize>(self, rhs: FixedPoint<M>, rounding: Rounding) -> FixedPoint<K> {
        let product = wide(self.0) * wide(rhs.0); // exact, at precision N + M
        FixedPoint(narrow(rescale_raw(product, N + M, K, rounding)))
    }

    /// `self / rhs` at precision `K`, rounded once.
    ///
    /// # Panics
    ///
    /// If `rhs` is zero.
    pub fn div_to<const M: usize, const K: usize>(self, rhs: FixedPoint<M>, rounding: Rounding) -> FixedPoint<K> {
        assert!(rhs.0 != 0, "FixedPoint division by zero");
        // self/rhs = (a·10⁻ᴺ)/(b·10⁻ᴹ); at precision K the raw result is a·10^(K+M-N)/b.
        let shift = (K + M) as isize - N as isize;
        let (numerator, denominator) = if shift >= 0 {
            (wide(self.0) * pow10_wide(shift as usize), wide(rhs.0))
        } else {
            (wide(self.0), wide(rhs.0) * pow10_wide((-shift) as usize))
        };
        FixedPoint(narrow(divide(numerator, denominator, rounding)))
    }

    /// Parses a decimal, rounding away any digits beyond `N`.
    pub fn parse_rounded(text: &str, rounding: Rounding) -> Result<Self, ParseFixedPointError> {
        let (mantissa, decimals) = parse_decimal(text)?;
        let raw = rescale_raw(mantissa, decimals, N, rounding);
        i64::try_from(raw).map(Self).map_err(|_| ParseFixedPointError::Overflow)
    }

    pub fn abs(self) -> Self {
        Self(self.0.checked_abs().expect("FixedPoint overflow"))
    }

    pub fn is_negative(self) -> bool {
        self.0 < 0
    }

    pub fn is_positive(self) -> bool {
        self.0 > 0
    }
}

/// A raw count at precision `from` moved to precision `to`.
fn rescale_raw(raw: i128, from: usize, to: usize, rounding: Rounding) -> i128 {
    match to.cmp(&from) {
        Ordering::Equal => raw,
        Ordering::Greater => raw.checked_mul(pow10_wide(to - from)).expect("FixedPoint overflow"),
        Ordering::Less => divide(raw, pow10_wide(from - to), rounding),
    }
}

/// A decimal string as `(mantissa, digits after the point)`: "-12.50" is `(-1250, 2)`.
fn parse_decimal(text: &str) -> Result<(i128, usize), ParseFixedPointError> {
    let (negative, unsigned) = match text.as_bytes().first() {
        Some(b'-') => (true, &text[1..]),
        Some(b'+') => (false, &text[1..]),
        _ => (false, text),
    };
    let (whole, fraction) = unsigned.split_once('.').unwrap_or((unsigned, ""));
    if whole.is_empty() && fraction.is_empty() {
        return Err(ParseFixedPointError::Empty);
    }
    // Trailing zeros carry no value; dropping them lets "0.50000000" parse exactly at N = 2.
    let fraction = fraction.trim_end_matches('0');
    let mut mantissa: i128 = 0;
    for byte in whole.bytes().chain(fraction.bytes()) {
        if !byte.is_ascii_digit() {
            return Err(ParseFixedPointError::Invalid);
        }
        mantissa = mantissa
            .checked_mul(10)
            .and_then(|m| m.checked_add((byte - b'0').into()))
            .ok_or(ParseFixedPointError::Overflow)?;
    }
    Ok((if negative { -mantissa } else { mantissa }, fraction.len()))
}

impl<const N: usize> FromStr for FixedPoint<N> {
    type Err = ParseFixedPointError;

    /// Parses a decimal exactly: `"0.56"`, `"-3"`, `"563.6500"`. Nonzero digits beyond `N` are
    /// an error ([`ParseFixedPointError::Inexact`]); see [`FixedPoint::parse_rounded`].
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let (mantissa, decimals) = parse_decimal(text)?;
        if decimals > N {
            return Err(ParseFixedPointError::Inexact);
        }
        let raw = mantissa.checked_mul(pow10_wide(N - decimals)).ok_or(ParseFixedPointError::Overflow)?;
        i64::try_from(raw).map(Self).map_err(|_| ParseFixedPointError::Overflow)
    }
}

impl<const N: usize> fmt::Display for FixedPoint<N> {
    /// All `N` decimals by default (`0.500000`); the alternate form (`{:#}`) drops trailing zeros
    /// (`0.5`); a precision (`{:.2}`) rounds to nearest or pads with zeros. Width, alignment, `+`
    /// and zero-padding work as they do for integers.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let decimals = match f.precision() {
            Some(precision) => precision,
            None if f.alternate() => significant_decimals(self.0, N),
            None => N,
        };
        let shown = rescale_raw(wide(self.0), N, decimals, Rounding::Nearest);
        let unit = pow10_wide(decimals);
        let (whole, fraction) = (shown.unsigned_abs() / unit.unsigned_abs(), shown.unsigned_abs() % unit.unsigned_abs());
        let digits = if decimals == 0 { whole.to_string() } else { format!("{whole}.{fraction:0decimals$}") };
        f.pad_integral(shown >= 0, "", &digits)
    }
}

/// Decimals left in `raw · 10⁻ⁿ` once trailing zeros are dropped.
fn significant_decimals(raw: i64, n: usize) -> usize {
    let mut decimals = n;
    let mut raw = raw;
    while decimals > 0 && raw % 10 == 0 {
        raw /= 10;
        decimals -= 1;
    }
    decimals
}

impl<const N: usize> fmt::Debug for FixedPoint<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

// ----------------------------------------------------------------------------- arithmetic

impl<const N: usize> Add for FixedPoint<N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0.checked_add(rhs.0).expect("FixedPoint overflow"))
    }
}

impl<const N: usize> Sub for FixedPoint<N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0.checked_sub(rhs.0).expect("FixedPoint overflow"))
    }
}

impl<const N: usize> Mul for FixedPoint<N> {
    type Output = Self;
    /// Rounded to nearest, halves away from zero.
    fn mul(self, rhs: Self) -> Self {
        self.mul_to(rhs, Rounding::Nearest)
    }
}

impl<const N: usize> Div for FixedPoint<N> {
    type Output = Self;
    /// Rounded to nearest, halves away from zero.
    fn div(self, rhs: Self) -> Self {
        self.div_to(rhs, Rounding::Nearest)
    }
}

impl<const N: usize> Rem for FixedPoint<N> {
    type Output = Self;
    /// Exact, with the sign of `self` (as for integers).
    fn rem(self, rhs: Self) -> Self {
        Self(self.0.checked_rem(rhs.0).expect("FixedPoint remainder by zero"))
    }
}

impl<const N: usize> Neg for FixedPoint<N> {
    type Output = Self;
    fn neg(self) -> Self {
        Self(self.0.checked_neg().expect("FixedPoint overflow"))
    }
}

macro_rules! assign_ops {
    ($($trait:ident $method:ident $op:tt),+) => {$(
        impl<const N: usize> $trait for FixedPoint<N> {
            fn $method(&mut self, rhs: Self) {
                *self = *self $op rhs;
            }
        }
    )+};
}
assign_ops!(AddAssign add_assign +, SubAssign sub_assign -, MulAssign mul_assign *, DivAssign div_assign /, RemAssign rem_assign %);

impl<const N: usize> Sum for FixedPoint<N> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, Add::add)
    }
}

impl<'a, const N: usize> Sum<&'a FixedPoint<N>> for FixedPoint<N> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.copied().sum()
    }
}

impl<const N: usize> Product for FixedPoint<N> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, Mul::mul)
    }
}

// ----------------------------------------------------------------------------- num-traits

impl<const N: usize> Zero for FixedPoint<N> {
    fn zero() -> Self {
        Self::ZERO
    }
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl<const N: usize> One for FixedPoint<N> {
    fn one() -> Self {
        Self::ONE
    }
}

impl<const N: usize> Num for FixedPoint<N> {
    type FromStrRadixErr = ParseFixedPointError;

    /// Decimal only: any radix but 10 is [`ParseFixedPointError::Invalid`].
    fn from_str_radix(text: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        if radix != 10 {
            return Err(ParseFixedPointError::Invalid);
        }
        text.parse()
    }
}

impl<const N: usize> Signed for FixedPoint<N> {
    fn abs(&self) -> Self {
        FixedPoint::abs(*self)
    }
    fn abs_sub(&self, other: &Self) -> Self {
        if self <= other { Self::ZERO } else { *self - *other }
    }
    fn signum(&self) -> Self {
        Self::from_int(self.0.signum())
    }
    fn is_positive(&self) -> bool {
        self.0 > 0
    }
    fn is_negative(&self) -> bool {
        self.0 < 0
    }
}

impl<const N: usize> Bounded for FixedPoint<N> {
    fn min_value() -> Self {
        Self::MIN
    }
    fn max_value() -> Self {
        Self::MAX
    }
}

impl<const N: usize> ToPrimitive for FixedPoint<N> {
    /// The whole part, truncated toward zero (as a float's `to_i64` is).
    fn to_i64(&self) -> Option<i64> {
        Some(self.0 / Self::SCALE)
    }
    fn to_u64(&self) -> Option<u64> {
        u64::try_from(self.0 / Self::SCALE).ok()
    }
    fn to_f64(&self) -> Option<f64> {
        Some(FixedPoint::to_f64(*self))
    }
}

impl<const N: usize> FromPrimitive for FixedPoint<N> {
    fn from_i64(value: i64) -> Option<Self> {
        value.checked_mul(Self::SCALE).map(Self)
    }
    fn from_u64(value: u64) -> Option<Self> {
        i64::try_from(value).ok().and_then(Self::from_i64)
    }
    /// Rounded to nearest.
    fn from_f64(value: f64) -> Option<Self> {
        FixedPoint::from_f64(value, Rounding::Nearest)
    }
}

impl<const N: usize> NumCast for FixedPoint<N> {
    /// Integers convert exactly; anything else through `f64`, rounded to nearest.
    fn from<T: ToPrimitive>(value: T) -> Option<Self> {
        match value.to_f64() {
            Some(float) if float.fract() == 0.0 => value.to_i64().and_then(Self::from_i64),
            Some(float) => FixedPoint::from_f64(float, Rounding::Nearest),
            None => None,
        }
    }
}

// ----------------------------------------------------------------------------- tensorcrate

impl<const N: usize> Coefficient for FixedPoint<N> {
    /// Division keeps `N` decimals, so elimination's fractional intermediates survive (rounded
    /// to `10⁻ᴺ`), unlike an integer's truncating division.
    fn supports_fractional_division() -> bool {
        true
    }

    fn trunc_div(self, rhs: Self) -> Self {
        Self::from_int(self.0.checked_div(rhs.0).expect("FixedPoint division by zero"))
    }

    fn round(self) -> Self {
        Self::from_int(narrow(divide(wide(self.0), wide(Self::SCALE), Rounding::Nearest)))
    }

    fn magnitude(self) -> f64 {
        self.to_f64().abs()
    }
}

impl<const N: usize> Conj for FixedPoint<N> {
    type Output = Self;
    fn conj(self) -> Self {
        self
    }
}

// ----------------------------------------------------------------------------- serde

#[cfg(feature = "serde")]
mod serde_impls {
    use std::fmt;

    use serde::de::{self, Visitor};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use super::{FixedPoint, Rounding};

    /// Written as a decimal string with all `N` places ("0.560000"), so no reader rounds it
    /// through a float.
    impl<const N: usize> Serialize for FixedPoint<N> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            serializer.collect_str(self)
        }
    }

    /// Read from a decimal string (exactly), an integer, or a float (rounded to nearest).
    impl<'de, const N: usize> Deserialize<'de> for FixedPoint<N> {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            struct FixedPointVisitor<const N: usize>;

            impl<const N: usize> Visitor<'_> for FixedPointVisitor<N> {
                type Value = FixedPoint<N>;

                fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    write!(f, "a decimal with at most {N} places")
                }

                fn visit_str<E: de::Error>(self, text: &str) -> Result<Self::Value, E> {
                    text.parse().map_err(E::custom)
                }

                fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
                    value.checked_mul(FixedPoint::<N>::SCALE).map(FixedPoint).ok_or_else(|| E::custom("out of range"))
                }

                fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
                    i64::try_from(value).map_err(|_| E::custom("out of range")).and_then(|v| self.visit_i64(v))
                }

                fn visit_f64<E: de::Error>(self, value: f64) -> Result<Self::Value, E> {
                    FixedPoint::from_f64(value, Rounding::Nearest).ok_or_else(|| E::custom("not a finite number in range"))
                }
            }

            deserializer.deserialize_any(FixedPointVisitor::<N>)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Cents = FixedPoint<2>;
    type Micros = FixedPoint<6>;

    fn m(text: &str) -> Micros {
        text.parse().unwrap()
    }

    #[test]
    fn decimal_arithmetic_is_exact() {
        assert_eq!(m("0.1") + m("0.2"), m("0.3"));
        assert_eq!(m("1") - m("0.999999"), Micros::EPSILON);
        assert_eq!(m("0.5") * m("0.5"), m("0.25"));
        assert_eq!(m("5.5") % m("2"), m("1.5"));
        assert_eq!([m("0.1"); 10].into_iter().sum::<Micros>(), Micros::ONE);
        assert_eq!(Micros::SCALE, 1_000_000);
        assert_eq!(Cents::from_int(3).raw(), 300);
    }

    #[test]
    fn products_and_quotients_round_once() {
        // 1/3 at six places.
        assert_eq!(m("1") / m("3"), m("0.333333"));
        assert_eq!(m("2") / m("3"), m("0.666667"));
        assert_eq!(m("-2") / m("3"), m("-0.666667"));
        // Halves go away from zero.
        assert_eq!(m("0.000001") * m("0.5"), m("0.000001"));
        assert_eq!(m("-0.000001") * m("0.5"), m("-0.000001"));
        // A rule's own rounding: 0.07 · 0.25 = 0.0175, up to the cent.
        let fee: Cents = Cents::from_raw(7).mul_to::<6, 2>(m("0.25"), Rounding::Ceil);
        assert_eq!(fee, Cents::from_raw(2));
        assert_eq!(m("0.0175").rescale::<2>(Rounding::Floor), Cents::from_raw(1));
        assert_eq!(m("-0.0175").rescale::<2>(Rounding::Floor), Cents::from_raw(-2));
        assert_eq!(m("-0.0175").rescale::<2>(Rounding::TowardZero), Cents::from_raw(-1));
        // Across precisions: $1.75 over 100 contracts is $0.0175 each.
        let each: Micros = "1.75".parse::<Cents>().unwrap().div_to::<2, 6>("100".parse::<Cents>().unwrap(), Rounding::Nearest);
        assert_eq!(each, m("0.0175"));
    }

    #[test]
    fn strings_round_trip() {
        assert_eq!(m("0.56").to_string(), "0.560000");
        assert_eq!(m("-3").to_string(), "-3.000000");
        assert_eq!(m("98.45863000"), m("98.45863")); // trailing zeros beyond N are not digits
        assert_eq!("563.65".parse::<Cents>().unwrap().raw(), 56_365);
        assert_eq!(format!("{:.3}", m("0.1235")), "0.124");
        assert_eq!(format!("{:>8.2}", m("-0.005")), "   -0.01");
        assert_eq!(format!("{:+.1}", m("1.25")), "+1.3");
        assert_eq!(format!("{:.8}", m("1.5")), "1.50000000");
        assert_eq!(format!("{:#}", m("99.32")), "99.32");
        assert_eq!(format!("{:#}", m("-3")), "-3");
        assert_eq!(format!("{:#}", Micros::ZERO), "0");
        assert_eq!(format!("{:>#8}", m("0.000001")), "0.000001");
        assert_eq!(format!("{:>#7}", m("4.5")), "    4.5");
        assert_eq!("0.1234567".parse::<Micros>(), Err(ParseFixedPointError::Inexact));
        assert_eq!(Micros::parse_rounded("0.1234567", Rounding::Nearest), Ok(m("0.123457")));
        assert_eq!("".parse::<Micros>(), Err(ParseFixedPointError::Empty));
        assert_eq!("1.2.3".parse::<Micros>(), Err(ParseFixedPointError::Invalid));
        assert_eq!("99999999999999".parse::<Micros>(), Err(ParseFixedPointError::Overflow));
        assert_eq!(".5".parse::<Micros>(), Ok(m("0.5")));
    }

    #[test]
    fn floats_convert_at_their_decimal_value() {
        assert_eq!(Micros::from_f64(0.1, Rounding::Nearest), Some(m("0.1")));
        assert_eq!(Micros::from_f64(4087.93, Rounding::Nearest), Some(m("4087.93")));
        assert_eq!(Micros::from_f64(f64::NAN, Rounding::Nearest), None);
        assert_eq!(Micros::from_f64(1e20, Rounding::Nearest), None);
        assert_eq!(m("0.0175").to_f64(), 0.0175);
        assert_eq!(<Micros as NumCast>::from(7_u32), Some(Micros::from_int(7)));
    }

    #[test]
    fn coefficient_rounding_works_in_whole_units() {
        assert_eq!(Coefficient::trunc_div(m("7.5"), m("2")), m("3"));
        assert_eq!(Coefficient::round(m("2.5")), m("3"));
        assert_eq!(Coefficient::round(m("-2.5")), m("-3"));
        assert_eq!(m("-1.5").magnitude(), 1.5);
        assert_eq!(<Micros as Num>::from_str_radix("10", 16), Err(ParseFixedPointError::Invalid));
    }

    #[test]
    #[should_panic(expected = "FixedPoint overflow")]
    fn overflow_panics_in_every_build() {
        let _ = Micros::MAX + Micros::EPSILON;
    }
}
