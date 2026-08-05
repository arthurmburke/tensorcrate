//! The scalar special functions the distributions are built from.
//!
//! Everything here is `f64`, including the paths that end up serving `f32`
//! tensors. A `f32` value widens exactly, so evaluating in double and rounding
//! once at the end costs one extra conversion and buys a result that is correct
//! to the last `f32` bit — whereas evaluating a rational approximation in single
//! precision loses several bits to the approximation *and* to the arithmetic.
//! The vectorized and GPU tiers make the opposite trade deliberately; see
//! [`statistics`](super) for what that means for cross-tier agreement.
//!
//! The error function is [Cody's rational Chebyshev
//! approximation](https://doi.org/10.1090/S0025-5718-1969-0247736-4), split at
//! `0.46875` and `4` into a series for the central region, a rational times
//! `exp(−x²)` for the shoulder, and an asymptotic form for the tail. The normal
//! quantile is Wichura's AS 241 (`PPND16`), accurate to about sixteen digits
//! across the whole open unit interval. Both are pure arithmetic — no
//! iteration — which is why the tiers below can vectorize them.

// The coefficient tables are transcribed at the precision they are published
// at, which is a digit or two past what a `f64` can hold. Rounding them to the
// nearest representable double is what the compiler does anyway; keeping the
// published digits is what makes the tables checkable against their sources.
#![allow(clippy::excessive_precision)]

use std::f64::consts::{FRAC_1_SQRT_2, PI};

/// `1/√π`, the coefficient of the asymptotic `erfc` expansion.
const FRAC_1_SQRT_PI: f64 = 0.564_189_583_547_756_3;

/// `√(2π)`, the normalizing constant of the normal density.
const SQRT_2PI: f64 = 2.506_628_274_631_000_5;

/// Where the central `erf` series gives way to the scaled `erfc` rational.
const ERF_CENTRAL: f64 = 0.468_75;

/// Where the scaled `erfc` rational gives way to the asymptotic expansion.
const ERFC_ASYMPTOTIC: f64 = 4.0;

/// Beyond this argument `erfc(x)` is smaller than the smallest positive
/// double, so the unscaled function is exactly zero and only [`ln_erfc`] can
/// still say anything about it.
const ERFC_UNDERFLOW: f64 = 27.25;

// ---- the error function -----------------------------------------------------

/// `erf(x)`, the integral of the normal density over `[−x, x]` scaled to reach
/// one.
pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let magnitude = x.abs();
    if magnitude <= ERF_CENTRAL {
        return erf_central(x);
    }
    // `erf = 1 − erfc` loses no accuracy here: past the split point `erfc` is
    // already below a half, so the subtraction cannot cancel catastrophically.
    let complement = 1.0 - erfc_positive(magnitude);
    if x < 0.0 { -complement } else { complement }
}

/// `erfc(x) = 1 − erf(x)`, computed without forming the difference so that the
/// right tail keeps its relative accuracy instead of vanishing into rounding.
pub fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < -ERF_CENTRAL {
        // `erfc(−x) = 2 − erfc(x)`. The left tail approaches 2, where absolute
        // and relative accuracy agree, so the subtraction is harmless.
        return 2.0 - erfc_positive(-x);
    }
    if x <= ERF_CENTRAL {
        return 1.0 - erf_central(x);
    }
    erfc_positive(x)
}

/// `ln erfc(x)`.
///
/// The reason this exists rather than `erfc(x).ln()`: `erfc` underflows to zero
/// somewhere past `x = 27`, and the inverse Gaussian distribution function
/// evaluates `exp(a) · erfc(b)` with both factors far outside the representable
/// range and a perfectly ordinary product. Working in logarithms keeps that
/// product meaningful — the scaled rational below never underflows, because the
/// `exp(−x²)` it would have been multiplied by is exactly the term that becomes
/// an addition here.
pub fn ln_erfc(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x <= ERF_CENTRAL {
        return erfc(x).ln();
    }
    erfc_scaled(x).ln() - x * x
}

/// `erf` on `[−0.46875, 0.46875]`, where the function is odd and nearly linear:
/// a rational in `x²`, multiplied by `x`.
fn erf_central(x: f64) -> f64 {
    const A: [f64; 5] = [
        3.161_123_743_870_565_6e0,
        1.138_641_541_510_501_56e2,
        3.774_852_376_853_020_21e2,
        3.209_377_589_138_469_47e3,
        1.857_777_061_846_031_53e-1,
    ];
    const B: [f64; 4] = [
        2.360_129_095_234_412_09e1,
        2.440_246_379_344_441_73e2,
        1.282_616_526_077_372_28e3,
        2.844_236_833_439_170_62e3,
    ];

    let z = x * x;
    let mut numerator = A[4] * z;
    let mut denominator = z;
    for i in 0..3 {
        numerator = (numerator + A[i]) * z;
        denominator = (denominator + B[i]) * z;
    }
    x * (numerator + A[3]) / (denominator + B[3])
}

/// `erfc(x)` for `x > 0.46875`.
///
/// The scaled value is a well-conditioned rational; the whole difficulty is
/// multiplying it by `exp(−x²)` without losing bits. Squaring `x` rounds, and
/// the rounding error is then amplified by the exponential — at `x = 20` an
/// error of one ulp in `x²` is an error of nearly `10⁻¹³` relative in the
/// result. Splitting `x` at a power of two makes the leading term's square
/// exact, so only the small correction is inexact.
fn erfc_positive(x: f64) -> f64 {
    if x > ERFC_UNDERFLOW {
        return 0.0;
    }
    let leading = (x * 16.0).trunc() / 16.0;
    let correction = (x - leading) * (x + leading);
    (-leading * leading).exp() * (-correction).exp() * erfc_scaled(x)
}

/// `exp(x²) · erfc(x)` for `x > 0.46875` — the part of `erfc` that is a
/// rational function, with the exponential factored out.
fn erfc_scaled(x: f64) -> f64 {
    const C: [f64; 9] = [
        5.641_884_969_886_700_89e-1,
        8.883_149_794_388_375_94e0,
        6.611_919_063_714_162_95e1,
        2.986_351_381_974_001_31e2,
        8.819_522_212_417_690_90e2,
        1.712_047_612_634_070_58e3,
        2.051_078_377_826_071_47e3,
        1.230_339_354_797_997_25e3,
        2.153_115_354_744_038_46e-8,
    ];
    const D: [f64; 8] = [
        1.574_492_611_070_983_47e1,
        1.176_939_508_913_124_99e2,
        5.371_811_018_620_098_58e2,
        1.621_389_574_566_690_19e3,
        3.290_799_235_733_459_63e3,
        4.362_619_090_143_247_16e3,
        3.439_367_674_143_721_64e3,
        1.230_339_354_803_749_42e3,
    ];
    const P: [f64; 6] = [
        3.053_266_349_612_323_44e-1,
        3.603_448_999_498_044_39e-1,
        1.257_817_261_112_292_46e-1,
        1.608_378_514_874_227_66e-2,
        6.587_491_615_298_378_03e-4,
        1.631_538_713_730_209_78e-2,
    ];
    const Q: [f64; 5] = [
        2.568_520_192_289_822_42e0,
        1.872_952_849_923_460_47e0,
        5.279_051_029_514_284_12e-1,
        6.051_834_131_244_131_91e-2,
        2.335_204_976_268_691_85e-3,
    ];

    if x <= ERFC_ASYMPTOTIC {
        let mut numerator = C[8] * x;
        let mut denominator = x;
        for i in 0..7 {
            numerator = (numerator + C[i]) * x;
            denominator = (denominator + D[i]) * x;
        }
        return (numerator + C[7]) / (denominator + D[7]);
    }

    // Far out, `erfc(x) ≈ exp(−x²)/(x√π)`; the rational corrects that estimate
    // and is a function of `1/x²`, which is tiny here.
    let z = 1.0 / (x * x);
    let mut numerator = P[5] * z;
    let mut denominator = z;
    for i in 0..4 {
        numerator = (numerator + P[i]) * z;
        denominator = (denominator + Q[i]) * z;
    }
    let correction = z * (numerator + P[4]) / (denominator + Q[4]);
    (FRAC_1_SQRT_PI - correction) / x
}

/// `erf⁻¹(y)` for `y` in `(−1, 1)`, as the normal quantile rescaled.
///
/// `±1` map to the infinities and anything outside the closed interval is a
/// NaN, which is what the identity `erf⁻¹(erf(x)) = x` asks for at the ends.
pub fn erf_inv(y: f64) -> f64 {
    standard_normal_ppf(0.5 * (y + 1.0)) * FRAC_1_SQRT_2
}

// ---- the normal distribution ------------------------------------------------

/// The standard normal density at `z`.
pub fn standard_normal_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / SQRT_2PI
}

/// `Φ(z)`, the standard normal distribution function.
///
/// Written as a complementary error function rather than `0.5·(1 + erf(z/√2))`
/// so that the left tail keeps relative accuracy: the `erf` form computes
/// `Φ(−10)` as the difference of two numbers that agree to fifteen digits,
/// which leaves none.
pub fn standard_normal_cdf(z: f64) -> f64 {
    0.5 * erfc(-z * FRAC_1_SQRT_2)
}

/// `ln Φ(z)`, meaningful even where `Φ(z)` itself underflows.
pub fn standard_normal_ln_cdf(z: f64) -> f64 {
    // ln(0.5) + ln erfc(−z/√2)
    ln_erfc(-z * FRAC_1_SQRT_2) - std::f64::consts::LN_2
}

/// `Φ⁻¹(p)`, the standard normal quantile, by Wichura's AS 241.
///
/// Three rational approximations meet at `|p − ½| = 0.425` and at
/// `√(−ln min(p, 1−p)) = 5`. The tails are parameterized by that square root
/// rather than by `p` itself, which is what keeps `Φ⁻¹(10⁻³⁰⁰)` as accurate as
/// `Φ⁻¹(0.4)`.
pub fn standard_normal_ppf(p: f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }

    let q = p - 0.5;
    if q.abs() <= 0.425 {
        const A: [f64; 8] = [
            3.387_132_872_796_366_6e0,
            1.331_416_678_917_843_77e2,
            1.971_590_950_306_551_44e3,
            1.373_169_376_550_946_11e4,
            4.592_195_393_154_987_15e4,
            6.726_577_092_700_870_09e4,
            3.343_057_558_358_812_81e4,
            2.509_080_928_730_122_67e3,
        ];
        const B: [f64; 7] = [
            4.231_333_070_160_091_13e1,
            6.871_870_074_920_579_08e2,
            5.394_196_021_424_751_11e3,
            2.121_379_430_158_659_59e4,
            3.930_789_580_009_271_06e4,
            2.872_908_573_572_194_27e4,
            5.226_495_278_852_854_56e3,
        ];

        let r = 0.180_625 - q * q;
        return q * polynomial(&A, r) / (polynomial(&B, r) * r + 1.0);
    }

    let tail = if q < 0.0 { p } else { 1.0 - p };
    let r = (-tail.ln()).sqrt();
    let value = if r <= 5.0 {
        const C: [f64; 8] = [
            1.423_437_110_749_683_577_34e0,
            4.630_337_846_156_545_295_90e0,
            5.769_497_221_460_691_405_50e0,
            3.647_848_324_763_204_605_04e0,
            1.270_458_252_452_368_382_58e0,
            2.417_807_251_774_506_117_70e-1,
            2.272_384_498_926_918_458_33e-2,
            7.745_450_142_783_414_076_40e-4,
        ];
        const D: [f64; 7] = [
            2.053_191_626_637_758_821_87e0,
            1.676_384_830_183_803_849_40e0,
            6.897_673_349_851_000_045_50e-1,
            1.481_039_764_274_800_745_90e-1,
            1.519_866_656_361_645_719_66e-2,
            5.475_938_084_995_344_946_00e-4,
            1.050_750_071_644_416_843_24e-9,
        ];

        let r = r - 1.6;
        polynomial(&C, r) / (polynomial(&D, r) * r + 1.0)
    } else {
        const E: [f64; 8] = [
            6.657_904_643_501_103_777_20e0,
            5.463_784_911_164_114_369_90e0,
            1.784_826_539_917_291_335_80e0,
            2.965_605_718_285_048_912_30e-1,
            2.653_218_952_657_612_309_30e-2,
            1.242_660_947_388_078_438_60e-3,
            2.711_555_568_743_487_578_15e-5,
            2.010_334_399_292_288_132_65e-7,
        ];
        const F: [f64; 7] = [
            5.998_322_065_558_879_376_90e-1,
            1.369_298_809_227_358_053_10e-1,
            1.487_536_129_085_061_485_25e-2,
            7.868_691_311_456_132_591_00e-4,
            1.846_318_317_510_054_681_80e-5,
            1.421_511_758_316_445_888_70e-7,
            2.044_263_103_389_939_785_64e-15,
        ];

        let r = r - 5.0;
        polynomial(&E, r) / (polynomial(&F, r) * r + 1.0)
    };

    if q < 0.0 { -value } else { value }
}

/// Whether a distribution parameter is a usable positive scale. A NaN answers
/// `false` here, which is what sends it on to the NaN result.
fn positive(parameter: f64) -> bool {
    parameter > 0.0
}

/// Horner's rule over coefficients given from the constant term upward.
fn polynomial(coefficients: &[f64], x: f64) -> f64 {
    let mut total = 0.0;
    for &coefficient in coefficients.iter().rev() {
        total = total * x + coefficient;
    }
    total
}

/// The density of `N(mean, stddev²)` at `x`.
pub fn normal_pdf(x: f64, mean: f64, stddev: f64) -> f64 {
    if stddev.is_nan() || stddev <= 0.0 {
        return f64::NAN;
    }
    standard_normal_pdf((x - mean) / stddev) / stddev
}

/// The distribution function of `N(mean, stddev²)` at `x`.
pub fn normal_cdf(x: f64, mean: f64, stddev: f64) -> f64 {
    if stddev.is_nan() || stddev <= 0.0 {
        return f64::NAN;
    }
    standard_normal_cdf((x - mean) / stddev)
}

/// The quantile of `N(mean, stddev²)` at probability `p`.
pub fn normal_ppf(p: f64, mean: f64, stddev: f64) -> f64 {
    if stddev.is_nan() || stddev <= 0.0 {
        return f64::NAN;
    }
    mean + stddev * standard_normal_ppf(p)
}

// ---- the inverse Gaussian (Wald) distribution -------------------------------

/// The density of the inverse Gaussian with mean `mean` and shape `shape`.
///
/// The distribution lives on the positive reals — it is the first-passage time
/// of a drifting Brownian motion — so the density is zero at and below the
/// origin rather than undefined.
pub fn inverse_gaussian_pdf(x: f64, mean: f64, shape: f64) -> f64 {
    if !positive(mean) || !positive(shape) {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    let deviation = x - mean;
    let exponent = -shape * deviation * deviation / (2.0 * mean * mean * x);
    (shape / (2.0 * PI * x * x * x)).sqrt() * exponent.exp()
}

/// The distribution function of the inverse Gaussian.
///
/// `F(x) = Φ(a) + exp(2λ/μ)·Φ(−b)`, and the second term is the whole reason
/// this is not a one-liner: `exp(2λ/μ)` overflows for a shape much larger than
/// its mean while `Φ(−b)` underflows just as fast, and their product stays
/// perfectly ordinary. Adding the logarithms and exponentiating once keeps the
/// term accurate to the last bit where a literal transcription returns `∞·0`.
pub fn inverse_gaussian_cdf(x: f64, mean: f64, shape: f64) -> f64 {
    if !positive(mean) || !positive(shape) {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    let scale = (shape / x).sqrt();
    let ratio = x / mean;
    let lower = scale * (ratio - 1.0);
    let upper = scale * (ratio + 1.0);
    let tail = 2.0 * shape / mean + standard_normal_ln_cdf(-upper);
    (standard_normal_cdf(lower) + tail.exp()).min(1.0)
}

/// The quantile of the inverse Gaussian.
///
/// The distribution function has no elementary inverse, so this is a bracketed
/// Newton iteration: Newton where the step stays inside the bracket and makes
/// progress, a bisection where it does not. The safeguard matters because the
/// density is zero at the origin and vanishingly small in the right tail, so an
/// unguarded Newton step from a bad start walks off to infinity or backwards
/// through zero.
pub fn inverse_gaussian_ppf(p: f64, mean: f64, shape: f64) -> f64 {
    if !positive(mean) || !positive(shape) || p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }

    // Bracket the answer. The lower end shrinks toward zero and the upper end
    // grows; both terminate because the distribution function is continuous and
    // spans `(0, 1)` on `(0, ∞)`.
    let mut low = 0.0;
    let mut high = mean;
    while inverse_gaussian_cdf(high, mean, shape) < p {
        low = high;
        high *= 2.0;
        if high.is_infinite() {
            return f64::INFINITY;
        }
    }
    let mut guess = 0.5 * (low + high);

    // Sixty passes is far more than the quadratic phase needs and still bounds
    // the worst case: even if every step degenerates to a bisection, the
    // bracket has been halved sixty times, which is finer than the doubles in
    // it can distinguish.
    for _ in 0..60 {
        let error = inverse_gaussian_cdf(guess, mean, shape) - p;
        if error > 0.0 {
            high = guess;
        } else {
            low = guess;
        }

        let density = inverse_gaussian_pdf(guess, mean, shape);
        let step = if density > 0.0 {
            guess - error / density
        } else {
            f64::NAN
        };
        let next = if step.is_finite() && step > low && step < high {
            step
        } else {
            0.5 * (low + high)
        };

        if next == guess {
            break;
        }
        guess = next;
    }
    guess
}

/// The moment estimate of an inverse Gaussian's shape: `n / Σ(1/xᵢ − 1/x̄)`.
///
/// This is also the maximum-likelihood estimate, which the normal's parameters
/// share — for both families the obvious estimator is the best one, so fitting
/// either is a matter of accumulating sums rather than optimizing.
pub fn inverse_gaussian_shape(values: impl Iterator<Item = f64>, mean: f64) -> f64 {
    if !positive(mean) {
        return f64::NAN;
    }
    let mut count = 0.0;
    let mut total = 0.0;
    for value in values {
        count += 1.0;
        total += 1.0 / value - 1.0 / mean;
    }
    count / total
}
