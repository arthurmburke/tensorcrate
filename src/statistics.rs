//! Descriptive statistics and distribution functions over tensors.
//!
//! Two things live here. The first is the pair of moments — [mean](Vector::mean)
//! and [standard deviation](Vector::stddev) — over a whole tensor or along one
//! axis of a matrix. The second is the distribution functions of the
//! [normal](Distribution::Normal) and [inverse Gaussian](Distribution::InverseGaussian)
//! families: density, distribution function, and quantile, applied elementwise
//! or with a separately fitted distribution per row or column.
//!
//! ```
//! use tensorcrate::statistics::{Axis, Correction, Distribution, Family};
//! use tensorcrate::tensors::Matrix;
//!
//! let observations = Matrix::<f64>::from_rows([
//!     [1.0, 2.0, 3.0, 4.0],
//!     [10.0, 12.0, 14.0, 16.0],
//! ]);
//!
//! // One value per row.
//! assert_eq!(observations.mean_axis(Axis::Rows).to_vec(), [2.5, 13.0]);
//!
//! // Or over every element at once.
//! assert_eq!(observations.mean(), 7.75);
//!
//! // A distribution is an ordinary value, and evaluates elementwise.
//! let standard = Distribution::<f64>::standard_normal();
//! assert_eq!(standard.cdf(1.96), 0.9750021048517795);
//! assert_eq!(standard.ppf(0.975), 1.9599639845400536);
//!
//! // Fitting each row and mapping it through its own distribution function
//! // puts otherwise incomparable rows on one probability scale.
//! let ranked = observations.cdf_axis(
//!     Axis::Rows,
//!     &observations.fit_axis(Axis::Rows, Family::Normal, Correction::Population),
//! );
//! assert!(ranked.row(0)[3] > 0.9 && ranked.row(1)[3] > 0.9);
//! ```
//!
//! # Which divisor
//!
//! Every variance takes a [`Correction`]: divide the sum of squared deviations
//! by `n` for the variance of the values in hand, or by `n − 1` for an unbiased
//! estimate of the variance of the population they came from. Neither is a safe
//! default — the choice depends on whether the tensor *is* the population or a
//! sample from one — so it is always spelled out at the call site.
//!
//! # Accuracy
//!
//! The variance is computed in two passes, taking the mean first and then
//! summing squared deviations from it. The one-pass `E[x²] − E[x]²` identity
//! costs half the memory traffic and is the reason so many implementations use
//! it, but it subtracts two nearly equal numbers whenever the mean is large
//! relative to the spread: for values around `10⁶` with a spread of `1`, it
//! loses every significant digit. The second pass is cheap by comparison, and
//! it is exactly the pass a SIMD or GPU kernel vectorizes best.
//!
//! The distribution functions are evaluated in `f64` on the host, whatever the
//! tensor's element type, so a `f32` tensor gets results correct to the last
//! `f32` bit. The Metal shaders evaluate in `f32` throughout, which is the one
//! place a backend change is visible in the answer: agreement between the two
//! is to about `1e-6` relative rather than exact. The moments agree far more
//! closely, differing only in summation order.

use std::fmt;

use num_traits::Float;

use crate::numbers::{Coefficient, Real};
use crate::tensors::{Analytic, Backend, BinaryOp, Host, Kernels, Matrix, Vector};

pub mod special;

pub use crate::tensors::{Axis, Family, Statistic};

// ---- summaries --------------------------------------------------------------

/// Which divisor a variance uses.
///
/// The sum of squared deviations is the same either way; this only chooses what
/// it is divided by.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Correction {
    /// Divide by `n`: the variance of the values themselves, taken as the whole
    /// population.
    Population,
    /// Divide by `n − 1`: Bessel's correction, an unbiased estimate of the
    /// variance of the population the values were sampled from.
    ///
    /// Undefined for fewer than two values, where it answers with a NaN.
    Sample,
}

impl Correction {
    /// The divisor for `count` values, or `None` where there are too few for
    /// this correction to define one.
    pub fn divisor(self, count: usize) -> Option<usize> {
        match self {
            Correction::Population => (count > 0).then_some(count),
            Correction::Sample => (count > 1).then(|| count - 1),
        }
    }
}

/// The count, mean, and summed squared deviations of a set of values.
///
/// This is the whole result of the two passes over the data, before a divisor
/// has been chosen. Asking for the mean and both variances therefore costs one
/// traversal rather than three.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Moments<T> {
    /// How many values were folded in.
    pub count: usize,
    /// Their arithmetic mean.
    pub mean: T,
    /// `Σ(xᵢ − mean)²`.
    pub sum_squared_deviations: T,
}

impl<T: Float> Moments<T> {
    /// The variance under `correction`, or a NaN where there are too few values
    /// for it to be defined.
    pub fn variance(&self, correction: Correction) -> T {
        match correction.divisor(self.count) {
            Some(divisor) => self.sum_squared_deviations / cast(divisor as f64),
            None => T::nan(),
        }
    }

    /// The standard deviation under `correction` — the square root of
    /// [`variance`](Self::variance).
    pub fn stddev(&self, correction: Correction) -> T {
        self.variance(correction).sqrt()
    }
}

/// The same summary, per row or per column of a matrix.
///
/// The two vectors stay on the backend the matrix was on, so a Metal reduction
/// leaves its results in GPU memory for whatever comes next.
pub struct AxisMoments<T, B: Backend = Host> {
    /// How many values sit behind each result — the length of one row for
    /// [`Axis::Rows`], the height of one column for [`Axis::Columns`].
    pub count: usize,
    /// One mean per row or column.
    pub means: Vector<T, B>,
    /// One `Σ(xᵢ − mean)²` per row or column.
    pub sum_squared_deviations: Vector<T, B>,
}

impl<T, B: Backend> Clone for AxisMoments<T, B>
where
    Vector<T, B>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            count: self.count,
            means: self.means.clone(),
            sum_squared_deviations: self.sum_squared_deviations.clone(),
        }
    }
}

impl<T, B: Backend> fmt::Debug for AxisMoments<T, B>
where
    Vector<T, B>: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AxisMoments")
            .field("count", &self.count)
            .field("means", &self.means)
            .field("sum_squared_deviations", &self.sum_squared_deviations)
            .finish()
    }
}

impl<T: Coefficient + Float> AxisMoments<T, Host> {
    /// One variance per row or column, under `correction`.
    pub fn variance(&self, correction: Correction) -> Vector<T, Host> {
        match correction.divisor(self.count) {
            Some(divisor) => {
                let divisor = cast::<T>(divisor as f64);
                self.sum_squared_deviations.map(|&total| total / divisor)
            }
            None => Vector::repeat(self.sum_squared_deviations.len(), T::nan()),
        }
    }

    /// One standard deviation per row or column, under `correction`.
    pub fn stddev(&self, correction: Correction) -> Vector<T, Host> {
        self.variance(correction).map(|&value| value.sqrt())
    }
}

// ---- distributions ----------------------------------------------------------

/// A distribution with its parameters fixed.
///
/// Both families are continuous and take two parameters, which is what lets one
/// [`Family`] code plus a parameter pair cross into a shader. The inverse
/// Gaussian is the first-passage time of a Brownian motion with drift; it lives
/// on the positive reals and is right-skewed, which makes it the natural
/// counterpart to the normal for durations, latencies, and other quantities
/// that cannot go negative.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Distribution<T> {
    /// `N(mean, stddev²)`.
    Normal {
        /// The distribution's mean, which is also its median and mode.
        mean: T,
        /// The distribution's standard deviation. Must be positive; anything
        /// else makes every function answer with a NaN.
        stddev: T,
    },
    /// The inverse Gaussian (Wald) distribution.
    ///
    /// As `shape` grows relative to `mean` the distribution approaches a normal
    /// with variance `mean³/shape`.
    InverseGaussian {
        /// The distribution's mean. Must be positive.
        mean: T,
        /// The shape parameter `λ`. Must be positive.
        shape: T,
    },
}

impl<T: Float> Distribution<T> {
    /// `N(0, 1)`.
    pub fn standard_normal() -> Self {
        Distribution::Normal {
            mean: T::zero(),
            stddev: T::one(),
        }
    }

    /// Rebuild a distribution from the untyped form the kernels carry.
    pub fn from_parameters(family: Family, parameters: (T, T)) -> Self {
        match family {
            Family::Normal => Distribution::Normal {
                mean: parameters.0,
                stddev: parameters.1,
            },
            Family::InverseGaussian => Distribution::InverseGaussian {
                mean: parameters.0,
                shape: parameters.1,
            },
        }
    }

    /// Which family this distribution belongs to.
    pub fn family(&self) -> Family {
        match self {
            Distribution::Normal { .. } => Family::Normal,
            Distribution::InverseGaussian { .. } => Family::InverseGaussian,
        }
    }

    /// The two parameters, in the order the family names them.
    pub fn parameters(&self) -> (T, T) {
        match *self {
            Distribution::Normal { mean, stddev } => (mean, stddev),
            Distribution::InverseGaussian { mean, shape } => (mean, shape),
        }
    }

    /// The probability density at `x`.
    pub fn pdf(&self, x: T) -> T {
        self.evaluate(Statistic::Pdf, x)
    }

    /// `P(X ≤ x)`.
    pub fn cdf(&self, x: T) -> T {
        self.evaluate(Statistic::Cdf, x)
    }

    /// The quantile at probability `p`: the value `x` for which `P(X ≤ x) = p`.
    ///
    /// Probabilities outside `[0, 1]` give a NaN, and the two ends give the
    /// infimum and supremum of the support.
    pub fn ppf(&self, p: T) -> T {
        self.evaluate(Statistic::Ppf, p)
    }

    /// One of the three, chosen at runtime — the form the tensor operations and
    /// the shaders both use.
    pub fn evaluate(&self, statistic: Statistic, x: T) -> T {
        let x = widen(x);
        let (first, second) = self.parameters();
        let (first, second) = (widen(first), widen(second));
        cast(match (self.family(), statistic) {
            (Family::Normal, Statistic::Pdf) => special::normal_pdf(x, first, second),
            (Family::Normal, Statistic::Cdf) => special::normal_cdf(x, first, second),
            (Family::Normal, Statistic::Ppf) => special::normal_ppf(x, first, second),
            (Family::InverseGaussian, Statistic::Pdf) => {
                special::inverse_gaussian_pdf(x, first, second)
            }
            (Family::InverseGaussian, Statistic::Cdf) => {
                special::inverse_gaussian_cdf(x, first, second)
            }
            (Family::InverseGaussian, Statistic::Ppf) => {
                special::inverse_gaussian_ppf(x, first, second)
            }
        })
    }
}

impl Family {
    /// Estimate this family's parameters from `values`.
    ///
    /// Both families are fitted by their maximum-likelihood estimates, which for
    /// both come out of sums rather than an optimization: the normal takes the
    /// mean and the standard deviation of the values, and the inverse Gaussian
    /// takes the same mean with shape `n / Σ(1/xᵢ − 1/x̄)`. `correction` chooses
    /// the divisor of the normal's variance; the strict maximum-likelihood fit
    /// is [`Correction::Population`].
    ///
    /// The inverse Gaussian is only defined for positive values, and fitting it
    /// to a set containing any other answers with NaN parameters — which then
    /// propagate, rather than panicking on data that may be one row of many.
    pub fn fit<T: Coefficient + Float>(
        self,
        values: &[T],
        correction: Correction,
    ) -> Distribution<T> {
        fit_with_moments(self, values, &moments_of(values), correction)
    }
}

/// The estimator proper, taking moments that have already been computed.
///
/// The backend-generic path arrives here with moments a kernel produced, so the
/// values are traversed once rather than once per parameter.
fn fit_with_moments<T: Coefficient + Float>(
    family: Family,
    values: &[T],
    moments: &Moments<T>,
    correction: Correction,
) -> Distribution<T> {
    match family {
        Family::Normal => Distribution::Normal {
            mean: moments.mean,
            stddev: moments.stddev(correction),
        },
        Family::InverseGaussian => Distribution::InverseGaussian {
            mean: moments.mean,
            shape: cast(special::inverse_gaussian_shape(
                values.iter().map(|&value| widen(value)),
                widen(moments.mean),
            )),
        },
    }
}

/// One fit per row or per column of a row-major buffer.
fn fit_axis_of<T: Coefficient + Float>(
    data: &[T],
    shape: (usize, usize),
    axis: Axis,
    family: Family,
    correction: Correction,
) -> Vec<Distribution<T>> {
    let (rows, cols) = shape;
    match axis {
        Axis::Rows => (0..rows)
            .map(|row| family.fit(&data[row * cols..(row + 1) * cols], correction))
            .collect(),
        Axis::Columns => {
            // A column is strided, so it is gathered once here rather than
            // walked repeatedly by the estimator.
            let mut column = Vec::with_capacity(rows);
            (0..cols)
                .map(|col| {
                    column.clear();
                    column.extend((0..rows).map(|row| data[row * cols + col]));
                    family.fit(&column, correction)
                })
                .collect()
        }
    }
}

// ---- host kernels -----------------------------------------------------------

/// Widen an element to the precision the special functions work in.
fn widen<T: Float>(value: T) -> f64 {
    value.to_f64().unwrap_or(f64::NAN)
}

/// Round a computed `f64` back to the tensor's element type.
fn cast<T: Float>(value: f64) -> T {
    T::from(value).unwrap_or_else(T::nan)
}

/// The sum of a slice, through the vectorized kernel where one applies.
fn sum_of<T: Coefficient + Float>(values: &[T]) -> T {
    #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
    if let Some(total) = crate::tensors::simd_dispatch::reduce(values, crate::tensors::Reduce::Sum)
    {
        return total;
    }
    values.iter().fold(T::zero(), |total, &value| total + value)
}

/// `Σ(xᵢ − mean)²`, the second pass of the variance.
fn deviation_of<T: Coefficient + Float>(values: &[T], mean: T) -> T {
    #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
    if let Some(total) = crate::tensors::simd_dispatch::sum_squared_deviations(values, mean) {
        return total;
    }
    values.iter().fold(T::zero(), |total, &value| {
        let deviation = value - mean;
        total + deviation * deviation
    })
}

/// Both passes over one contiguous run of values.
///
/// `f16` and `bf16` take both passes in `f32` — the deviations measured from
/// the unrounded mean — and round the two results once.
fn moments_of<T: Coefficient + Float>(values: &[T]) -> Moments<T> {
    if let Some(wide) = crate::compact::widen(values) {
        let moments = moments_of::<f32>(&wide);
        return Moments {
            count: moments.count,
            mean: cast(f64::from(moments.mean)),
            sum_squared_deviations: cast(f64::from(moments.sum_squared_deviations)),
        };
    }
    if values.is_empty() {
        return Moments {
            count: 0,
            mean: T::nan(),
            sum_squared_deviations: T::nan(),
        };
    }
    let mean = sum_of(values) / cast(values.len() as f64);
    Moments {
        count: values.len(),
        mean,
        sum_squared_deviations: deviation_of(values, mean),
    }
}

/// Both passes along one axis of a row-major matrix.
///
/// Along [`Axis::Rows`] each result owns a contiguous run, so this is the
/// whole-slice fold repeated. Along [`Axis::Columns`] it is not: a column is
/// strided, and walking one costs a cache line per element. Accumulating a
/// whole row of partial sums at a time instead reads the matrix once in storage
/// order and keeps the accumulator's own stride at one, which is both
/// cache-friendly and the shape a SIMD kernel wants.
fn axis_moments_of<T: Coefficient + Float>(
    data: &[T],
    shape: (usize, usize),
    axis: Axis,
) -> AxisMoments<T> {
    // As in `moments_of`: the compact types accumulate in `f32`.
    if let Some(wide) = crate::compact::widen(data) {
        let moments = axis_moments_of::<f32>(&wide, shape, axis);
        let round = |v: &Vector<f32, Host>| v.map(|&x| cast::<T>(f64::from(x)));
        return AxisMoments {
            count: moments.count,
            means: round(&moments.means),
            sum_squared_deviations: round(&moments.sum_squared_deviations),
        };
    }
    let (rows, cols) = shape;
    let count = axis.depth(shape);
    let extent = axis.extent(shape);

    if count == 0 {
        return AxisMoments {
            count,
            means: Vector::repeat(extent, T::nan()),
            sum_squared_deviations: Vector::repeat(extent, T::nan()),
        };
    }

    match axis {
        Axis::Rows => {
            let mut means = Vec::with_capacity(rows);
            let mut deviations = Vec::with_capacity(rows);
            for row in 0..rows {
                let moments = moments_of(&data[row * cols..(row + 1) * cols]);
                means.push(moments.mean);
                deviations.push(moments.sum_squared_deviations);
            }
            AxisMoments {
                count,
                means: Vector::new(means),
                sum_squared_deviations: Vector::new(deviations),
            }
        }
        Axis::Columns => {
            let mut means = vec![T::zero(); cols];
            for row in 0..rows {
                accumulate(&mut means, &data[row * cols..(row + 1) * cols]);
            }
            // Divided rather than multiplied by a reciprocal, so that a column
            // mean is bit-for-bit the mean of that column taken on its own.
            let divisor = cast::<T>(rows as f64);
            for mean in means.iter_mut() {
                *mean = *mean / divisor;
            }

            let mut deviations = vec![T::zero(); cols];
            for row in 0..rows {
                accumulate_squared_deviations(
                    &mut deviations,
                    &data[row * cols..(row + 1) * cols],
                    &means,
                );
            }
            AxisMoments {
                count,
                means: Vector::new(means),
                sum_squared_deviations: Vector::new(deviations),
            }
        }
    }
}

/// `totals += values`, elementwise.
fn accumulate<T: Coefficient + Float>(totals: &mut [T], values: &[T]) {
    #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
    if crate::tensors::simd_dispatch::accumulate(totals, values) {
        return;
    }
    for (total, &value) in totals.iter_mut().zip(values) {
        *total = *total + value;
    }
}

/// `totals += (values − means)²`, elementwise.
fn accumulate_squared_deviations<T: Coefficient + Float>(
    totals: &mut [T],
    values: &[T],
    means: &[T],
) {
    #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
    if crate::tensors::simd_dispatch::accumulate_squared_deviations(totals, values, means) {
        return;
    }
    for ((total, &value), &mean) in totals.iter_mut().zip(values).zip(means) {
        let deviation = value - mean;
        *total = *total + deviation * deviation;
    }
}

/// Panics unless there is one distribution per row or column, and all of them
/// belong to one family.
#[track_caller]
fn assert_axis_distributions<T: Float>(
    distributions: &[Distribution<T>],
    shape: (usize, usize),
    axis: Axis,
    operation: &str,
) {
    let extent = axis.extent(shape);
    assert!(
        distributions.len() == extent,
        "{operation}: {} distributions for {extent} {}",
        distributions.len(),
        match axis {
            Axis::Rows => "rows",
            Axis::Columns => "columns",
        }
    );
    // One family for the whole axis is what lets this become a single dispatch
    // with two parameter vectors rather than one call per row.
    if let Some(first) = distributions.first() {
        assert!(
            distributions
                .iter()
                .all(|distribution| distribution.family() == first.family()),
            "{operation}: the distributions belong to more than one family"
        );
    }
}

/// Evaluate `statistic` over a row-major buffer, one distribution per row or
/// column.
fn axis_distribution_of<T: Coefficient + Float>(
    data: &[T],
    shape: (usize, usize),
    axis: Axis,
    statistic: Statistic,
    distributions: &[Distribution<T>],
) -> Vec<T> {
    let (_, cols) = shape;
    data.iter()
        .enumerate()
        .map(|(index, &value)| {
            let along = match axis {
                Axis::Rows => index / cols,
                Axis::Columns => index % cols,
            };
            distributions[along].evaluate(statistic, value)
        })
        .collect()
}

// ---- host vectors -----------------------------------------------------------

impl<T: Coefficient + Float> Vector<T, Host> {
    /// The count, mean, and summed squared deviations, in two passes.
    ///
    /// An empty vector has no mean, so every field but the count is a NaN.
    pub fn moments(&self) -> Moments<T> {
        moments_of(self.data())
    }

    /// The arithmetic mean of every element.
    pub fn mean(&self) -> T {
        self.moments().mean
    }

    /// The variance of every element, under `correction`.
    pub fn variance(&self, correction: Correction) -> T {
        self.moments().variance(correction)
    }

    /// The standard deviation of every element, under `correction`.
    pub fn stddev(&self, correction: Correction) -> T {
        self.moments().stddev(correction)
    }

    /// Estimate a `family` distribution from these values.
    pub fn fit(&self, family: Family, correction: Correction) -> Distribution<T> {
        family.fit(self.data(), correction)
    }

    /// The density of `distribution` at every element.
    pub fn pdf(&self, distribution: &Distribution<T>) -> Self {
        self.distribution(Statistic::Pdf, distribution)
    }

    /// The distribution function of `distribution` at every element.
    pub fn cdf(&self, distribution: &Distribution<T>) -> Self {
        self.distribution(Statistic::Cdf, distribution)
    }

    /// The quantile of `distribution` at every element, which are read as
    /// probabilities.
    pub fn ppf(&self, distribution: &Distribution<T>) -> Self {
        self.distribution(Statistic::Ppf, distribution)
    }

    /// Any of the three, chosen at runtime.
    pub fn distribution(&self, statistic: Statistic, distribution: &Distribution<T>) -> Self {
        self.map(|&value| distribution.evaluate(statistic, value))
    }
}

// ---- host matrices ----------------------------------------------------------

impl<T: Coefficient + Float> Matrix<T, Host> {
    /// The count, mean, and summed squared deviations over every element,
    /// ignoring the shape.
    pub fn moments(&self) -> Moments<T> {
        moments_of(self.data())
    }

    /// The arithmetic mean of every element.
    pub fn mean(&self) -> T {
        self.moments().mean
    }

    /// The variance of every element, under `correction`.
    pub fn variance(&self, correction: Correction) -> T {
        self.moments().variance(correction)
    }

    /// The standard deviation of every element, under `correction`.
    pub fn stddev(&self, correction: Correction) -> T {
        self.moments().stddev(correction)
    }

    /// The count, means, and summed squared deviations along one axis.
    ///
    /// [`Axis::Rows`] folds each row and gives one result per row; see [`Axis`]
    /// for why it is named that way round.
    pub fn moments_axis(&self, axis: Axis) -> AxisMoments<T> {
        axis_moments_of(self.data(), self.shape(), axis)
    }

    /// One mean per row or column.
    pub fn mean_axis(&self, axis: Axis) -> Vector<T, Host> {
        self.moments_axis(axis).means
    }

    /// One variance per row or column, under `correction`.
    pub fn variance_axis(&self, axis: Axis, correction: Correction) -> Vector<T, Host> {
        self.moments_axis(axis).variance(correction)
    }

    /// One standard deviation per row or column, under `correction`.
    pub fn stddev_axis(&self, axis: Axis, correction: Correction) -> Vector<T, Host> {
        self.moments_axis(axis).stddev(correction)
    }

    /// Estimate a `family` distribution from every element.
    pub fn fit(&self, family: Family, correction: Correction) -> Distribution<T> {
        family.fit(self.data(), correction)
    }

    /// Estimate one `family` distribution per row or column.
    pub fn fit_axis(
        &self,
        axis: Axis,
        family: Family,
        correction: Correction,
    ) -> Vec<Distribution<T>> {
        fit_axis_of(self.data(), self.shape(), axis, family, correction)
    }

    /// The density of `distribution` at every element.
    pub fn pdf(&self, distribution: &Distribution<T>) -> Self {
        self.distribution(Statistic::Pdf, distribution)
    }

    /// The distribution function of `distribution` at every element.
    pub fn cdf(&self, distribution: &Distribution<T>) -> Self {
        self.distribution(Statistic::Cdf, distribution)
    }

    /// The quantile of `distribution` at every element, which are read as
    /// probabilities.
    pub fn ppf(&self, distribution: &Distribution<T>) -> Self {
        self.distribution(Statistic::Ppf, distribution)
    }

    /// Any of the three, chosen at runtime.
    pub fn distribution(&self, statistic: Statistic, distribution: &Distribution<T>) -> Self {
        self.map(|&value| distribution.evaluate(statistic, value))
    }

    /// The density at every element, under that element's own row or column
    /// distribution.
    ///
    /// # Panics
    ///
    /// Unless there is one distribution per row or column and they all belong
    /// to one family.
    #[track_caller]
    pub fn pdf_axis(&self, axis: Axis, distributions: &[Distribution<T>]) -> Self {
        self.distribution_axis(axis, Statistic::Pdf, distributions)
    }

    /// The distribution function at every element, under that element's own row
    /// or column distribution.
    ///
    /// Fitted per axis, this is the probability integral transform: it puts
    /// rows or columns measured on different scales onto one common `[0, 1]`
    /// scale.
    ///
    /// # Panics
    ///
    /// Unless there is one distribution per row or column and they all belong
    /// to one family.
    #[track_caller]
    pub fn cdf_axis(&self, axis: Axis, distributions: &[Distribution<T>]) -> Self {
        self.distribution_axis(axis, Statistic::Cdf, distributions)
    }

    /// The quantile at every element, under that element's own row or column
    /// distribution — the inverse of [`cdf_axis`](Self::cdf_axis), so the
    /// elements are read as probabilities.
    ///
    /// # Panics
    ///
    /// Unless there is one distribution per row or column and they all belong
    /// to one family.
    #[track_caller]
    pub fn ppf_axis(&self, axis: Axis, distributions: &[Distribution<T>]) -> Self {
        self.distribution_axis(axis, Statistic::Ppf, distributions)
    }

    /// Any of the three, chosen at runtime.
    ///
    /// # Panics
    ///
    /// Unless there is one distribution per row or column and they all belong
    /// to one family.
    #[track_caller]
    pub fn distribution_axis(
        &self,
        axis: Axis,
        statistic: Statistic,
        distributions: &[Distribution<T>],
    ) -> Self {
        assert_axis_distributions(distributions, self.shape(), axis, "distribution_axis");
        let (rows, cols) = self.shape();
        Matrix::from_flat(
            rows,
            cols,
            axis_distribution_of(self.data(), self.shape(), axis, statistic, distributions),
        )
    }
}

// ---- backend-generic surface ------------------------------------------------

/// The summaries and distribution functions, on whatever backend the tensor is
/// on.
///
/// The inherent methods above are defined for `Host` tensors of any float
/// element type, which is an impl that cannot also cover `Vector<f32, Metal>` —
/// the two would overlap at `Vector<f32, Host>`. This trait is the other half:
/// generic over the backend and over the element type, so one function can
/// summarize either. (`Metal` computes in `f32`, `f16` and `bf16`; moments
/// accumulate in `f32` on every backend.)
///
/// ```
/// use tensorcrate::numbers::Real;
/// use tensorcrate::statistics::{Correction, Distribution, Statistics};
/// use tensorcrate::tensors::{Kernels, Vector};
///
/// fn standardized<T: Real, B: Kernels<T>>(v: &Vector<T, B>) -> Vector<T, B> {
///     let fitted = Distribution::Normal {
///         mean: v.mean(),
///         stddev: v.stddev(Correction::Sample),
///     };
///     v.cdf(&fitted)
/// }
///
/// let v = Vector::new([1.0f32, 2.0, 3.0]);
/// assert!((standardized(&v).data()[1] - 0.5).abs() < 1e-6);
///
/// let wide = Vector::new([1.0f64, 2.0, 3.0]);
/// assert!((standardized(&wide).data()[1] - 0.5).abs() < 1e-15);
/// ```
///
/// On `Vector<f32, Host>` the inherent method wins method resolution and this
/// trait is never consulted; both run the same kernel, so which one resolved is
/// not observable.
pub trait Statistics: Sized {
    /// The element type the statistics are computed in.
    type Elem: Real;

    /// The count, mean, and summed squared deviations, in two passes.
    fn moments(&self) -> Moments<Self::Elem>;

    /// The arithmetic mean of every element.
    fn mean(&self) -> Self::Elem {
        self.moments().mean
    }

    /// The variance of every element, under `correction`.
    fn variance(&self, correction: Correction) -> Self::Elem {
        self.moments().variance(correction)
    }

    /// The standard deviation of every element, under `correction`.
    fn stddev(&self, correction: Correction) -> Self::Elem {
        self.moments().stddev(correction)
    }

    /// Estimate a `family` distribution from these values.
    fn fit(&self, family: Family, correction: Correction) -> Distribution<Self::Elem>;

    /// A distribution function, chosen at runtime, at every element.
    fn distribution(&self, statistic: Statistic, distribution: &Distribution<Self::Elem>) -> Self;

    /// The density of `distribution` at every element.
    fn pdf(&self, distribution: &Distribution<Self::Elem>) -> Self {
        self.distribution(Statistic::Pdf, distribution)
    }

    /// The distribution function of `distribution` at every element.
    fn cdf(&self, distribution: &Distribution<Self::Elem>) -> Self {
        self.distribution(Statistic::Cdf, distribution)
    }

    /// The quantile of `distribution` at every element, which are read as
    /// probabilities.
    fn ppf(&self, distribution: &Distribution<Self::Elem>) -> Self {
        self.distribution(Statistic::Ppf, distribution)
    }
}

impl<T: Real, B: Kernels<T>> Statistics for Vector<T, B> {
    type Elem = T;

    fn moments(&self) -> Moments<T> {
        let (mean, sum_squared_deviations) = B::vector_moments(self);
        Moments {
            count: self.len(),
            mean,
            sum_squared_deviations,
        }
    }

    fn fit(&self, family: Family, correction: Correction) -> Distribution<T> {
        // The values are read through the shared allocation rather than copied
        // back, so this costs a CPU traversal but no transfer.
        fit_with_moments(
            family,
            self.as_slice(),
            &Statistics::moments(self),
            correction,
        )
    }

    fn distribution(&self, statistic: Statistic, distribution: &Distribution<T>) -> Self {
        B::vector_distribution(
            self,
            distribution.family(),
            statistic,
            distribution.parameters(),
        )
    }
}

impl<T: Real, B: Kernels<T>> Statistics for Matrix<T, B> {
    type Elem = T;

    fn moments(&self) -> Moments<T> {
        let (mean, sum_squared_deviations) = B::matrix_moments(self);
        Moments {
            count: self.rows() * self.cols(),
            mean,
            sum_squared_deviations,
        }
    }

    fn fit(&self, family: Family, correction: Correction) -> Distribution<T> {
        fit_with_moments(
            family,
            self.as_slice(),
            &Statistics::moments(self),
            correction,
        )
    }

    fn distribution(&self, statistic: Statistic, distribution: &Distribution<T>) -> Self {
        B::matrix_distribution(
            self,
            distribution.family(),
            statistic,
            distribution.parameters(),
        )
    }
}

/// The same, along one axis of a matrix, on whatever backend it is on.
pub trait AxisStatistics: Sized {
    /// The element type the statistics are computed in.
    type Elem: Real;

    /// Where this matrix's elements live, and therefore where the reduced
    /// vectors land.
    type Backend: Kernels<Self::Elem>;

    /// The count, means, and summed squared deviations along one axis.
    fn moments_axis(&self, axis: Axis) -> AxisMoments<Self::Elem, Self::Backend>;

    /// One mean per row or column.
    fn mean_axis(&self, axis: Axis) -> Vector<Self::Elem, Self::Backend> {
        self.moments_axis(axis).means
    }

    /// One variance per row or column, under `correction`.
    fn variance_axis(
        &self,
        axis: Axis,
        correction: Correction,
    ) -> Vector<Self::Elem, Self::Backend>;

    /// One standard deviation per row or column, under `correction`.
    fn stddev_axis(&self, axis: Axis, correction: Correction) -> Vector<Self::Elem, Self::Backend> {
        Self::Backend::vector_unary(&self.variance_axis(axis, correction), Analytic::Sqrt)
    }

    /// Estimate one `family` distribution per row or column.
    fn fit_axis(
        &self,
        axis: Axis,
        family: Family,
        correction: Correction,
    ) -> Vec<Distribution<Self::Elem>>;

    /// A distribution function, chosen at runtime, at every element — under
    /// that element's own row or column distribution.
    ///
    /// # Panics
    ///
    /// Unless there is one distribution per row or column and they all belong
    /// to one family.
    fn distribution_axis(
        &self,
        axis: Axis,
        statistic: Statistic,
        distributions: &[Distribution<Self::Elem>],
    ) -> Self;

    /// The density at every element, under its own row or column distribution.
    ///
    /// # Panics
    ///
    /// Unless there is one distribution per row or column and they all belong
    /// to one family.
    fn pdf_axis(&self, axis: Axis, distributions: &[Distribution<Self::Elem>]) -> Self {
        self.distribution_axis(axis, Statistic::Pdf, distributions)
    }

    /// The distribution function at every element, under its own row or column
    /// distribution.
    ///
    /// # Panics
    ///
    /// Unless there is one distribution per row or column and they all belong
    /// to one family.
    fn cdf_axis(&self, axis: Axis, distributions: &[Distribution<Self::Elem>]) -> Self {
        self.distribution_axis(axis, Statistic::Cdf, distributions)
    }

    /// The quantile at every element, under its own row or column
    /// distribution.
    ///
    /// # Panics
    ///
    /// Unless there is one distribution per row or column and they all belong
    /// to one family.
    fn ppf_axis(&self, axis: Axis, distributions: &[Distribution<Self::Elem>]) -> Self {
        self.distribution_axis(axis, Statistic::Ppf, distributions)
    }
}

impl<T: Real, B: Kernels<T>> AxisStatistics for Matrix<T, B> {
    type Elem = T;
    type Backend = B;

    fn moments_axis(&self, axis: Axis) -> AxisMoments<T, B> {
        let (means, sum_squared_deviations) = B::matrix_axis_moments(self, axis);
        AxisMoments {
            count: axis.depth(self.shape()),
            means,
            sum_squared_deviations,
        }
    }

    fn variance_axis(&self, axis: Axis, correction: Correction) -> Vector<T, B> {
        let moments = AxisStatistics::moments_axis(self, axis);
        match correction.divisor(moments.count) {
            Some(divisor) => B::vector_broadcast(
                &moments.sum_squared_deviations,
                T::from_f64(divisor as f64),
                BinaryOp::Div,
                false,
            ),
            None => Vector::filled(axis.extent(self.shape()), T::nan()),
        }
    }

    fn fit_axis(&self, axis: Axis, family: Family, correction: Correction) -> Vec<Distribution<T>> {
        fit_axis_of(self.as_slice(), self.shape(), axis, family, correction)
    }

    fn distribution_axis(
        &self,
        axis: Axis,
        statistic: Statistic,
        distributions: &[Distribution<T>],
    ) -> Self {
        assert_axis_distributions(distributions, self.shape(), axis, "distribution_axis");
        // The parameters travel as two vectors on the tensor's own backend, so
        // a per-row fit is one dispatch rather than one per row.
        let family = distributions
            .first()
            .map_or(Family::Normal, Distribution::family);
        let first: Vec<T> = distributions
            .iter()
            .map(|distribution| distribution.parameters().0)
            .collect();
        let second: Vec<T> = distributions
            .iter()
            .map(|distribution| distribution.parameters().1)
            .collect();
        B::matrix_axis_distribution(
            self,
            axis,
            family,
            statistic,
            &Vector::build(&first),
            &Vector::build(&second),
        )
    }
}
