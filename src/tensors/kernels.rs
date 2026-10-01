//! The `f32` tensor algebra, named per backend.
//!
//! [`Vector`] and [`Matrix`] implement their products and elementwise
//! operations as inherent methods, once for each backend, and nothing ties those
//! two sets together — `Matrix<f32, Host>::matmul` and
//! `Matrix<f32, Metal>::matmul` are unrelated functions that happen to share a
//! name. [`Kernels`] is that missing link: one trait naming every operation both
//! backends provide, so code written against it compiles for either.
//!
//! Automatic differentiation is the reason it exists. The forward-mode layer in
//! [`dual`](super::dual) is written once against `Kernels` and instantiates on
//! both backends, which also means the `Host` instantiation is an exact
//! correctness oracle for the `Metal` one — same code, different memory.
//!
//! The trait is sealed, since [`Backend`] is.

use std::cmp::Ordering;

use super::fused::{self, Fresh, Program, Sink, Source};
use super::{Backend, Host, Matrix, Vector};
use crate::counters;
use crate::statistics::{Distribution, Moments};

/// An elementwise binary operation.
///
/// The representation is part of the Metal shader ABI. Keep existing
/// discriminants stable and only append new operations.
#[repr(u16)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    Rem = 4,
}

impl BinaryOp {
    /// The operator's spelling, for the shape-mismatch messages.
    pub fn name(self) -> &'static str {
        match self {
            BinaryOp::Add => "add",
            BinaryOp::Sub => "subtract",
            BinaryOp::Mul => "multiply",
            BinaryOp::Div => "divide",
            BinaryOp::Rem => "remainder",
        }
    }
}

impl From<BinaryOp> for u16 {
    fn from(op: BinaryOp) -> Self {
        op as u16
    }
}

impl TryFrom<u16> for BinaryOp {
    type Error = u16;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Add),
            1 => Ok(Self::Sub),
            2 => Ok(Self::Mul),
            3 => Ok(Self::Div),
            4 => Ok(Self::Rem),
            value => Err(value),
        }
    }
}

/// An elementwise comparison.
///
/// Like [`BinaryOp`], the representation is part of the Metal shader ABI: keep
/// existing discriminants stable and only append.
///
/// `Min` and `Max` are not differentiable where the operands tie, so a
/// convention is needed. This one splits the subgradient evenly, which is what
/// [`MaxShare`](Compare::MaxShare) computes — and it is the choice that keeps
/// `max(a, b)` and `max(b, a)` giving mirror-image gradients. Two consequences
/// worth knowing: `|x|` differentiates to `sign(x)` with `sign(0) = 0`, and
/// `relu` has slope `½` exactly at the kink rather than the `0` some frameworks
/// pick.
///
/// The last four are *predicates*: they answer with `1.0` or `0.0` rather than
/// with a value, which is how a tensor algebra with no boolean element type
/// writes a mask. A mask multiplies (to zero out entries) or adds (to bias an
/// index), so `select`, `count` and `first index where …` are all ordinary
/// arithmetic over one. They are locally constant, so their derivative is zero
/// wherever it exists and they are not meant to appear inside a differentiated
/// expression.
///
/// A predicate is `false` whenever either operand is NaN, since an unordered
/// comparison holds no way round — including `NaN ≤ NaN`.
///
/// `Min` and `Max` are IEEE `minNum`/`maxNum`, matching [`f32::min`] and
/// [`f32::max`]: a number beats a NaN, so a stray NaN operand does not
/// propagate. The one thing they do not pin down is the *sign* of a zero when
/// `−0.0` and `+0.0` tie — `fminnm` answers `−0.0` where `minps` answers `+0.0`,
/// and IEEE 754 permits both. The two zeros compare equal, so this is visible
/// only to a bit-level comparison.
#[repr(u16)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Compare {
    /// The smaller of the two operands.
    Min = 0,
    /// The larger of the two operands.
    Max = 1,
    /// `∂max(a, b)/∂a`: one where `a` is larger, zero where it is smaller, and a
    /// half where they tie. The rule for `Min` is its complement, `1 − share`.
    MaxShare = 2,
    /// `1.0` where `a < b`.
    Less = 3,
    /// `1.0` where `a ≤ b`.
    LessEqual = 4,
    /// `1.0` where `a > b`.
    Greater = 5,
    /// `1.0` where `a ≥ b`.
    GreaterEqual = 6,
}

impl Compare {
    /// Every comparison, in discriminant order.
    pub const ALL: [Compare; 7] = [
        Compare::Min,
        Compare::Max,
        Compare::MaxShare,
        Compare::Less,
        Compare::LessEqual,
        Compare::Greater,
        Compare::GreaterEqual,
    ];

    /// Apply the comparison to a pair of values — the CPU counterpart of the
    /// `compare` shader, and the definition the GPU is tested against.
    pub fn value(self, a: f32, b: f32) -> f32 {
        match self {
            Compare::Min => a.min(b),
            Compare::Max => a.max(b),
            Compare::MaxShare => match a.partial_cmp(&b) {
                Some(Ordering::Greater) => 1.0,
                Some(Ordering::Less) => 0.0,
                _ => 0.5,
            },
            Compare::Less => f32::from(a < b),
            Compare::LessEqual => f32::from(a <= b),
            Compare::Greater => f32::from(a > b),
            Compare::GreaterEqual => f32::from(a >= b),
        }
    }
}

impl From<Compare> for u16 {
    fn from(op: Compare) -> Self {
        op as u16
    }
}

impl TryFrom<u16> for Compare {
    type Error = u16;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        Self::ALL.get(value as usize).copied().ok_or(value)
    }
}

/// A whole-tensor reduction: many values in, one out.
///
/// Like the other operation enums the representation is part of the Metal shader
/// ABI, so keep the discriminants stable and only append.
///
/// Each variant is an associative fold, which is what lets the GPU evaluate it
/// as a tree and the CPU keep several accumulators in flight. Floating-point
/// addition is *not* associative, so [`Sum`](Reduce::Sum) is order-dependent by
/// a rounding error or two: the backends agree to within tolerance, not to the
/// last bit. [`Min`](Reduce::Min) and [`Max`](Reduce::Max) are exact on every
/// path.
#[repr(u16)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Reduce {
    Sum = 0,
    Min = 1,
    Max = 2,
}

impl Reduce {
    /// Every reduction, in discriminant order.
    pub const ALL: [Reduce; 3] = [Reduce::Sum, Reduce::Min, Reduce::Max];

    /// The value that leaves the fold unchanged, and therefore the answer for an
    /// empty tensor.
    pub fn identity(self) -> f32 {
        match self {
            Reduce::Sum => 0.0,
            Reduce::Min => f32::INFINITY,
            Reduce::Max => f32::NEG_INFINITY,
        }
    }

    /// Combine two partial results. `Min`/`Max` follow [`f32::min`]/[`f32::max`],
    /// so a NaN operand loses to a number rather than poisoning the fold.
    pub fn combine(self, a: f32, b: f32) -> f32 {
        match self {
            Reduce::Sum => a + b,
            Reduce::Min => a.min(b),
            Reduce::Max => a.max(b),
        }
    }

    /// Fold a slice left to right — the scalar definition the vector and GPU
    /// paths are tested against.
    pub fn fold(self, values: &[f32]) -> f32 {
        values
            .iter()
            .fold(self.identity(), |total, &value| self.combine(total, value))
    }
}

impl From<Reduce> for u16 {
    fn from(op: Reduce) -> Self {
        op as u16
    }
}

impl TryFrom<u16> for Reduce {
    type Error = u16;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        Self::ALL.get(value as usize).copied().ok_or(value)
    }
}

/// Which way a sort runs.
///
/// Both directions order by [`f32::total_cmp`] — the IEEE total order, under
/// which `−0.0` precedes `+0.0` and NaNs sit at the ends by sign rather than
/// comparing unordered. That is a stronger promise than the `<` of a comparison
/// kernel, and it is what lets the GPU sort agree with the CPU one on every
/// input rather than only on NaN-free ones.
#[repr(u16)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SortOrder {
    Ascending = 0,
    Descending = 1,
}

impl SortOrder {
    /// Every direction, in discriminant order.
    pub const ALL: [SortOrder; 2] = [SortOrder::Ascending, SortOrder::Descending];

    /// The comparator this order sorts by, for handing to [`slice::sort_by`] —
    /// or to [`Vector::sort_by`](crate::tensors::Vector::sort_by).
    pub fn comparator(self) -> impl Fn(&f32, &f32) -> Ordering + Copy {
        move |left: &f32, right: &f32| match self {
            SortOrder::Ascending => left.total_cmp(right),
            SortOrder::Descending => right.total_cmp(left),
        }
    }
}

impl From<SortOrder> for u16 {
    fn from(op: SortOrder) -> Self {
        op as u16
    }
}

impl TryFrom<u16> for SortOrder {
    type Error = u16;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        Self::ALL.get(value as usize).copied().ok_or(value)
    }
}

/// The analytic functions, paired with their derivatives.
///
/// This is the operation enum for the GPU `unary`/`unary_dual` kernels and the
/// dispatch table for the CPU path, so both sides stay in step. The variants
/// mirror the functions `math!` accepts, and each derivative is written the same
/// way as the matching [`Dual`](crate::numbers::Dual) implementation.
#[repr(u16)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Analytic {
    Sin = 0,
    Cos = 1,
    Tan = 2,
    Sec = 3,
    Csc = 4,
    Arcsin = 5,
    Arccos = 6,
    Arctan = 7,
    Exp = 8,
    Ln = 9,
    Sinh = 10,
    Cosh = 11,
    Tanh = 12,
    Sqrt = 13,
}

impl Analytic {
    /// Every function, in discriminant order.
    pub const ALL: [Analytic; 14] = [
        Analytic::Sin,
        Analytic::Cos,
        Analytic::Tan,
        Analytic::Sec,
        Analytic::Csc,
        Analytic::Arcsin,
        Analytic::Arccos,
        Analytic::Arctan,
        Analytic::Exp,
        Analytic::Ln,
        Analytic::Sinh,
        Analytic::Cosh,
        Analytic::Tanh,
        Analytic::Sqrt,
    ];

    /// `f(x)`.
    pub fn value(self, x: f32) -> f32 {
        match self {
            Analytic::Sin => x.sin(),
            Analytic::Cos => x.cos(),
            Analytic::Tan => x.tan(),
            Analytic::Sec => x.cos().recip(),
            Analytic::Csc => x.sin().recip(),
            Analytic::Arcsin => x.asin(),
            Analytic::Arccos => x.acos(),
            Analytic::Arctan => x.atan(),
            Analytic::Exp => x.exp(),
            Analytic::Ln => x.ln(),
            Analytic::Sinh => x.sinh(),
            Analytic::Cosh => x.cosh(),
            Analytic::Tanh => x.tanh(),
            Analytic::Sqrt => x.sqrt(),
        }
    }

    /// `f'(x)`.
    pub fn derivative(self, x: f32) -> f32 {
        match self {
            Analytic::Sin => x.cos(),
            Analytic::Cos => -x.sin(),
            Analytic::Tan => {
                let cos = x.cos();
                (cos * cos).recip()
            }
            Analytic::Sec => {
                let cos = x.cos();
                x.sin() / (cos * cos)
            }
            Analytic::Csc => {
                let sin = x.sin();
                -x.cos() / (sin * sin)
            }
            Analytic::Arcsin => (1.0 - x * x).sqrt().recip(),
            Analytic::Arccos => -(1.0 - x * x).sqrt().recip(),
            Analytic::Arctan => (1.0 + x * x).recip(),
            Analytic::Exp => x.exp(),
            Analytic::Ln => x.recip(),
            Analytic::Sinh => x.cosh(),
            Analytic::Cosh => x.sinh(),
            Analytic::Tanh => {
                let tanh = x.tanh();
                1.0 - tanh * tanh
            }
            Analytic::Sqrt => (2.0 * x.sqrt()).recip(),
        }
    }
}

impl From<Analytic> for u16 {
    fn from(op: Analytic) -> Self {
        op as u16
    }
}

impl TryFrom<u16> for Analytic {
    type Error = u16;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        Self::ALL.get(value as usize).copied().ok_or(value)
    }
}

/// Which way a matrix reduction folds.
///
/// The name says what is *folded*, not what survives: [`Rows`](Axis::Rows)
/// folds each row and leaves one value per row. Reducing a `3×5` matrix along
/// `Rows` therefore gives three results and along
/// [`Columns`](Axis::Columns) five — the opposite of the convention that names
/// an axis by the index it keeps, and the one that makes `matrix.mean_axis(Rows)`
/// read as "the mean of each row".
///
/// Like the operation enums above, the representation is part of the Metal
/// shader ABI.
#[repr(u16)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Axis {
    /// Fold each row, giving one result per row.
    Rows = 0,
    /// Fold each column, giving one result per column.
    Columns = 1,
}

impl Axis {
    /// How many results a fold along this axis produces for a `rows × cols`
    /// matrix.
    pub fn extent(self, shape: (usize, usize)) -> usize {
        match self {
            Axis::Rows => shape.0,
            Axis::Columns => shape.1,
        }
    }

    /// How many elements each of those results folds together.
    pub fn depth(self, shape: (usize, usize)) -> usize {
        match self {
            Axis::Rows => shape.1,
            Axis::Columns => shape.0,
        }
    }
}

impl From<Axis> for u16 {
    fn from(axis: Axis) -> Self {
        axis as u16
    }
}

impl TryFrom<u16> for Axis {
    type Error = u16;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Rows),
            1 => Ok(Self::Columns),
            value => Err(value),
        }
    }
}

/// A distribution family, as an operation code.
///
/// The parameters travel separately, as a pair of `f32`, because that is the
/// only shape a shader argument can take — see
/// [`Distribution`](crate::statistics::Distribution) for the typed form these
/// two are decomposed from.
///
/// The representation is part of the Metal shader ABI.
#[repr(u16)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Family {
    /// The normal (Gaussian) distribution, parameterized by mean and standard
    /// deviation.
    Normal = 0,
    /// The inverse Gaussian (Wald) distribution, parameterized by mean and
    /// shape.
    InverseGaussian = 1,
}

impl From<Family> for u16 {
    fn from(family: Family) -> Self {
        family as u16
    }
}

impl TryFrom<u16> for Family {
    type Error = u16;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Normal),
            1 => Ok(Self::InverseGaussian),
            value => Err(value),
        }
    }
}

/// Which function of a distribution to evaluate.
///
/// The representation is part of the Metal shader ABI.
#[repr(u16)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Statistic {
    /// The probability density function, `f(x)`.
    Pdf = 0,
    /// The cumulative distribution function, `F(x) = P(X ≤ x)`.
    Cdf = 1,
    /// The percent point function, `F⁻¹(p)` — the quantile at probability `p`,
    /// and the inverse of [`Cdf`](Statistic::Cdf).
    Ppf = 2,
}

impl From<Statistic> for u16 {
    fn from(statistic: Statistic) -> Self {
        statistic as u16
    }
}

impl TryFrom<u16> for Statistic {
    type Error = u16;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Pdf),
            1 => Ok(Self::Cdf),
            2 => Ok(Self::Ppf),
            value => Err(value),
        }
    }
}

#[cfg(test)]
mod operation_tests {
    use std::mem::size_of;

    use super::{Analytic, Axis, BinaryOp, Family, Statistic};

    #[test]
    fn operation_enums_have_a_stable_u16_representation() {
        assert_eq!(size_of::<BinaryOp>(), size_of::<u16>());
        assert_eq!(size_of::<Analytic>(), size_of::<u16>());
        assert_eq!(size_of::<Axis>(), size_of::<u16>());
        assert_eq!(size_of::<Family>(), size_of::<u16>());
        assert_eq!(size_of::<Statistic>(), size_of::<u16>());

        for op in [
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Div,
            BinaryOp::Rem,
        ] {
            assert_eq!(BinaryOp::try_from(u16::from(op)), Ok(op));
        }
        for op in Analytic::ALL {
            assert_eq!(Analytic::try_from(u16::from(op)), Ok(op));
        }
        for axis in [Axis::Rows, Axis::Columns] {
            assert_eq!(Axis::try_from(u16::from(axis)), Ok(axis));
        }
        for family in [Family::Normal, Family::InverseGaussian] {
            assert_eq!(Family::try_from(u16::from(family)), Ok(family));
        }
        for statistic in [Statistic::Pdf, Statistic::Cdf, Statistic::Ppf] {
            assert_eq!(Statistic::try_from(u16::from(statistic)), Ok(statistic));
        }
        assert_eq!(BinaryOp::try_from(u16::MAX), Err(u16::MAX));
        assert_eq!(Analytic::try_from(u16::MAX), Err(u16::MAX));
        assert_eq!(Axis::try_from(u16::MAX), Err(u16::MAX));
        assert_eq!(Family::try_from(u16::MAX), Err(u16::MAX));
        assert_eq!(Statistic::try_from(u16::MAX), Err(u16::MAX));
    }
}

/// The order-dependent vector operations, spelled as methods, on whatever
/// backend the vector is on.
///
/// [`Vector<f32, Host>`] and [`Vector<f32, Metal>`] both have `min`, `max`,
/// `clamp` and the rest as inherent methods already. Neither of those is
/// reachable when the backend is a *type parameter*, though — an inherent impl
/// has to name a concrete type, and `Vector<T, Host>` cannot be one impl with
/// `Vector<f32, B>` because the two overlap at `Vector<f32, Host>`. So generic
/// code goes through here:
///
/// ```
/// use tensorcrate::tensors::{Kernels, Ordered, Vector};
///
/// // Note the `Kernels` bound: `Backend` says where elements live, `Kernels`
/// // says what can be computed on them.
/// fn relu_then_cap<B: Kernels>(v: &Vector<f32, B>, cap: f32) -> Vector<f32, B> {
///     v.max_scalar(0.0).min_scalar(cap)
/// }
///
/// let v = Vector::new([-1.0f32, 0.5, 9.0]);
/// assert_eq!(relu_then_cap(&v, 2.0).data(), [0.0, 0.5, 2.0]);
/// ```
///
/// On a concrete backend the inherent method wins method resolution and this
/// trait is never consulted; both run the same kernel and give the same answer,
/// so which one resolved is not observable. The inherent versions also cover
/// element types this trait cannot — `Vector<f64, Host>::max` is real, while the
/// 32-bit shaders mean the backend-generic surface is `f32` only.
///
/// [`Vector<f32, Host>`]: Vector
/// [`Vector<f32, Metal>`]: Vector
pub trait Ordered: Sized {
    /// Elementwise minimum.
    ///
    /// # Panics
    ///
    /// If the two lengths differ.
    fn min(&self, other: &Self) -> Self;

    /// Elementwise maximum.
    ///
    /// # Panics
    ///
    /// If the two lengths differ.
    fn max(&self, other: &Self) -> Self;

    /// The lesser of each element and `scalar`.
    fn min_scalar(&self, scalar: f32) -> Self;

    /// The greater of each element and `scalar` — `max_scalar(0.0)` is a relu.
    fn max_scalar(&self, scalar: f32) -> Self;

    /// Confine every element to `[low, high]`, in one pass.
    ///
    /// # Panics
    ///
    /// If `low > high`.
    fn clamp(&self, low: f32, high: f32) -> Self;

    /// Elementwise comparison, including the [`Compare`] predicates.
    ///
    /// # Panics
    ///
    /// If the two lengths differ.
    fn compare(&self, other: &Self, op: Compare) -> Self;

    /// Elementwise comparison against a scalar; `scalar_left` puts the scalar on
    /// the left, which matters for every op but `Min` and `Max`.
    fn compare_scalar(&self, scalar: f32, op: Compare, scalar_left: bool) -> Self;

    /// Fold the whole vector to one value; an empty vector gives
    /// [`op.identity()`](Reduce::identity).
    fn reduce(&self, op: Reduce) -> f32;

    /// The sum of every element.
    fn sum(&self) -> f32 {
        self.reduce(Reduce::Sum)
    }

    /// The smallest element, or `None` when there are none.
    fn minimum(&self) -> Option<f32>;

    /// The largest element, or `None` when there are none.
    fn maximum(&self) -> Option<f32>;

    /// Inclusive prefix sum: `out[i] = Σ_{j ≤ i} self[j]`.
    fn prefix_sum(&self) -> Self;

    /// The elements in [`SortOrder`]'s total order.
    fn sorted(&self, order: SortOrder) -> Self;

    /// The elements under an arbitrary comparator.
    ///
    /// A closure cannot cross to the GPU, so on a device-resident vector this
    /// one sorts on the CPU and stores the result back; [`sorted`](Self::sorted)
    /// is the version that stays put.
    fn sorted_by(&self, compare: impl FnMut(&f32, &f32) -> Ordering) -> Self;
}

impl<B: Kernels> Ordered for Vector<f32, B> {
    #[track_caller]
    fn min(&self, other: &Self) -> Self {
        B::vector_compare(self, other, Compare::Min)
    }

    #[track_caller]
    fn max(&self, other: &Self) -> Self {
        B::vector_compare(self, other, Compare::Max)
    }

    fn min_scalar(&self, scalar: f32) -> Self {
        B::vector_compare_scalar(self, scalar, Compare::Min, false)
    }

    fn max_scalar(&self, scalar: f32) -> Self {
        B::vector_compare_scalar(self, scalar, Compare::Max, false)
    }

    #[track_caller]
    fn clamp(&self, low: f32, high: f32) -> Self {
        B::vector_clamp(self, low, high)
    }

    #[track_caller]
    fn compare(&self, other: &Self, op: Compare) -> Self {
        B::vector_compare(self, other, op)
    }

    fn compare_scalar(&self, scalar: f32, op: Compare, scalar_left: bool) -> Self {
        B::vector_compare_scalar(self, scalar, op, scalar_left)
    }

    fn reduce(&self, op: Reduce) -> f32 {
        B::vector_reduce(self, op)
    }

    fn minimum(&self) -> Option<f32> {
        (!self.is_empty()).then(|| self.reduce(Reduce::Min))
    }

    fn maximum(&self) -> Option<f32> {
        (!self.is_empty()).then(|| self.reduce(Reduce::Max))
    }

    fn prefix_sum(&self) -> Self {
        B::vector_prefix_sum(self)
    }

    fn sorted(&self, order: SortOrder) -> Self {
        B::vector_sort(self, order)
    }

    fn sorted_by(&self, compare: impl FnMut(&f32, &f32) -> Ordering) -> Self {
        let mut values = self.as_slice().to_vec();
        values.sort_by(compare);
        Vector::<f32, Host>::new(values).to_backend()
    }
}

/// The `f32` tensor operations a [`Backend`] provides.
///
/// Every method reads the shapes it needs from its operands, so the trait names
/// the operations without naming any dimension. The shape *rules* still hold —
/// a mismatch panics, exactly as it does on the inherent methods these forward
/// to — they are simply checked when the call happens.
///
/// This is the operation-enum-shaped surface: `B::vector_compare(&a, &b,
/// Compare::Max)`. [`Ordered`] is the same operations spelled as methods —
/// `a.max(&b)` — for the cases where that reads better.
pub trait Kernels: Backend {
    // ---- vectors ----

    fn vector_elementwise(
        a: &Vector<f32, Self>,
        b: &Vector<f32, Self>,
        op: BinaryOp,
    ) -> Vector<f32, Self>;

    fn vector_broadcast(
        a: &Vector<f32, Self>,
        scalar: f32,
        op: BinaryOp,
        scalar_left: bool,
    ) -> Vector<f32, Self>;

    /// Elementwise comparison of two vectors.
    fn vector_compare(
        a: &Vector<f32, Self>,
        b: &Vector<f32, Self>,
        op: Compare,
    ) -> Vector<f32, Self>;

    /// Elementwise comparison against a scalar.
    fn vector_compare_scalar(
        a: &Vector<f32, Self>,
        scalar: f32,
        op: Compare,
        scalar_left: bool,
    ) -> Vector<f32, Self>;

    /// Confine every element to `[low, high]`.
    ///
    /// The pair of comparisons fused into one pass: two `compare_scalar` calls
    /// would read and write the whole tensor twice, and on the GPU would be two
    /// dispatches.
    ///
    /// Panics unless `low ≤ high`.
    fn vector_clamp(a: &Vector<f32, Self>, low: f32, high: f32) -> Vector<f32, Self>;

    /// Fold the whole vector to one value; an empty vector gives
    /// [`op.identity()`](Reduce::identity).
    fn vector_reduce(a: &Vector<f32, Self>, op: Reduce) -> f32;

    /// Inclusive prefix sum: `out[i] = Σ_{j ≤ i} a[j]`.
    fn vector_prefix_sum(a: &Vector<f32, Self>) -> Vector<f32, Self>;

    /// The elements in [`SortOrder`]'s total order.
    fn vector_sort(a: &Vector<f32, Self>, order: SortOrder) -> Vector<f32, Self>;

    /// `f(a)`, elementwise.
    fn vector_unary(a: &Vector<f32, Self>, f: Analytic) -> Vector<f32, Self>;

    /// Elementwise `a^b`.
    ///
    /// A power is not a [`BinaryOp`]: that enum's variants are the operators
    /// defined for every [`Coefficient`](crate::numbers::Coefficient), and
    /// raising an integer to an integer leaves the integers. So it travels as
    /// its own kernel rather than a fifth arithmetic code.
    fn vector_power(a: &Vector<f32, Self>, b: &Vector<f32, Self>) -> Vector<f32, Self>;

    /// The same with one operand fixed: `scalar_left` selects `scalar^x` over
    /// `x^scalar`.
    fn vector_power_scalar(
        a: &Vector<f32, Self>,
        scalar: f32,
        scalar_left: bool,
    ) -> Vector<f32, Self>;

    /// `(f(value), f'(value) ⊙ tangent)` — one forward-mode step.
    fn vector_unary_dual(
        value: &Vector<f32, Self>,
        tangent: &Vector<f32, Self>,
        f: Analytic,
    ) -> (Vector<f32, Self>, Vector<f32, Self>);

    fn dot(a: &Vector<f32, Self>, b: &Vector<f32, Self>) -> f32;

    fn vecmat(v: &Vector<f32, Self>, m: &Matrix<f32, Self>) -> Vector<f32, Self>;

    fn matvec(m: &Matrix<f32, Self>, v: &Vector<f32, Self>) -> Vector<f32, Self>;

    /// `addend + m·v`, using `addend` as the accumulator when possible.
    fn matvec_add(
        m: &Matrix<f32, Self>,
        v: &Vector<f32, Self>,
        addend: Vector<f32, Self>,
    ) -> Vector<f32, Self>;

    // ---- matrices ----

    fn matrix_elementwise(
        a: &Matrix<f32, Self>,
        b: &Matrix<f32, Self>,
        op: BinaryOp,
    ) -> Matrix<f32, Self>;

    fn matrix_broadcast(
        a: &Matrix<f32, Self>,
        scalar: f32,
        op: BinaryOp,
        scalar_left: bool,
    ) -> Matrix<f32, Self>;

    /// Elementwise comparison of two matrices.
    fn matrix_compare(
        a: &Matrix<f32, Self>,
        b: &Matrix<f32, Self>,
        op: Compare,
    ) -> Matrix<f32, Self>;

    /// Elementwise comparison against a scalar.
    fn matrix_compare_scalar(
        a: &Matrix<f32, Self>,
        scalar: f32,
        op: Compare,
        scalar_left: bool,
    ) -> Matrix<f32, Self>;

    /// Confine every element to `[low, high]`; see
    /// [`vector_clamp`](Kernels::vector_clamp).
    fn matrix_clamp(a: &Matrix<f32, Self>, low: f32, high: f32) -> Matrix<f32, Self>;

    /// `f(a)`, elementwise.
    fn matrix_unary(a: &Matrix<f32, Self>, f: Analytic) -> Matrix<f32, Self>;

    /// Elementwise `a^b` over matrices; see [`vector_power`](Self::vector_power).
    fn matrix_power(a: &Matrix<f32, Self>, b: &Matrix<f32, Self>) -> Matrix<f32, Self>;

    /// The same with one operand fixed.
    fn matrix_power_scalar(
        a: &Matrix<f32, Self>,
        scalar: f32,
        scalar_left: bool,
    ) -> Matrix<f32, Self>;

    /// `(f(value), f'(value) ⊙ tangent)` — one forward-mode step.
    fn matrix_unary_dual(
        value: &Matrix<f32, Self>,
        tangent: &Matrix<f32, Self>,
        f: Analytic,
    ) -> (Matrix<f32, Self>, Matrix<f32, Self>);

    fn matmul(a: &Matrix<f32, Self>, b: &Matrix<f32, Self>) -> Matrix<f32, Self>;

    /// `addend + a·b`.
    ///
    /// Backends that can accumulate inside the product kernel do so, which is
    /// what makes a dual matmul two dispatches instead of three. `addend` is
    /// consumed because it may become the accumulator.
    fn matmul_add(
        a: &Matrix<f32, Self>,
        b: &Matrix<f32, Self>,
        addend: Matrix<f32, Self>,
    ) -> Matrix<f32, Self>;

    fn transpose(m: &Matrix<f32, Self>) -> Matrix<f32, Self>;

    /// Valid cross-correlation: output `(i, j)` is the window-shaped patch of
    /// `input` at `(i, j)` dotted with `window`, so the result is
    /// `(R−KR+1) × (C−KC+1)`.
    ///
    /// `flip` reverses the window, which is the difference between correlation
    /// (the machine-learning convention) and convolution (the signal-processing
    /// one) — and is also what the input-side gradient of either needs.
    ///
    /// Panics unless the window fits inside the input.
    fn correlate(
        input: &Matrix<f32, Self>,
        window: &Matrix<f32, Self>,
        flip: bool,
    ) -> Matrix<f32, Self>;

    /// `∂L/∂window` for a valid correlation: the input windowed by the output
    /// adjoint, which comes out exactly the window's shape —
    /// `(R − (R−KR+1) + 1) × (C − (C−KC+1) + 1)`.
    ///
    /// With the extents now ordinary numbers this is just `correlate` with the
    /// adjoint as the window. It stays a named operation because the backends
    /// reach it by different routes, and because the pairing of `input` with
    /// `adjoint` is what fixes the result shape.
    fn correlate_window_gradient(
        input: &Matrix<f32, Self>,
        adjoint: &Matrix<f32, Self>,
    ) -> Matrix<f32, Self>;

    /// `∂L/∂input` for a valid correlation: the full correlation of the adjoint
    /// with the window, which is the padded one, giving back the input's shape
    /// `(A+KR−1) × (B+KC−1)`. `forward_flip` says which convention the forward
    /// pass used; the gradient applies the window the other way round.
    fn correlate_input_gradient(
        adjoint: &Matrix<f32, Self>,
        window: &Matrix<f32, Self>,
        forward_flip: bool,
    ) -> Matrix<f32, Self>;

    /// Surround a matrix with `pad_rows` and `pad_cols` zeros on every side,
    /// giving `(R + 2·PR) × (C + 2·PC)`.
    fn pad(input: &Matrix<f32, Self>, pad_rows: usize, pad_cols: usize) -> Matrix<f32, Self>;

    /// Reverse both axes.
    fn flip(input: &Matrix<f32, Self>) -> Matrix<f32, Self>;

    /// The mean and `Σ(xᵢ − mean)²` of a vector, as `(mean, deviations)`.
    ///
    /// Both passes in one call, because on a GPU the mean has to come back to
    /// the CPU before the second pass can be encoded — returning it alongside
    /// the deviations means the caller never has to ask twice.
    fn vector_moments(a: &Vector<f32, Self>) -> (f32, f32);

    /// The same over every element of a matrix, ignoring its shape.
    fn matrix_moments(a: &Matrix<f32, Self>) -> (f32, f32);

    /// The same along one axis: one mean and one deviation sum per row or per
    /// column, as `(means, deviations)`.
    fn matrix_axis_moments(
        a: &Matrix<f32, Self>,
        axis: Axis,
    ) -> (Vector<f32, Self>, Vector<f32, Self>);

    /// A distribution function applied elementwise, with one parameter pair for
    /// the whole vector.
    fn vector_distribution(
        a: &Vector<f32, Self>,
        family: Family,
        statistic: Statistic,
        parameters: (f32, f32),
    ) -> Vector<f32, Self>;

    /// The same over a matrix.
    fn matrix_distribution(
        a: &Matrix<f32, Self>,
        family: Family,
        statistic: Statistic,
        parameters: (f32, f32),
    ) -> Matrix<f32, Self>;

    /// The same, with one parameter pair per row or per column: `first` and
    /// `second` hold one parameter each per slice along `axis`.
    fn matrix_axis_distribution(
        a: &Matrix<f32, Self>,
        axis: Axis,
        family: Family,
        statistic: Statistic,
        first: &Vector<f32, Self>,
        second: &Vector<f32, Self>,
    ) -> Matrix<f32, Self>;

    /// Run a fused elementwise program as one kernel. Reach this through
    /// [`Program::run`](super::fused::Program::run), which checks the operands
    /// against the program first.
    #[doc(hidden)]
    fn fused(
        program: &Program,
        shape: (usize, usize),
        inputs: &[Source<'_, Self>],
        updated: &mut [Sink<'_, Self>],
    ) -> Vec<Fresh<Self>>;
}

/// Every operation here already exists as an inherent method or an operator on
/// the host-backed tensors; this is pure forwarding.
impl Kernels for Host {
    fn vector_elementwise(
        a: &Vector<f32, Self>,
        b: &Vector<f32, Self>,
        op: BinaryOp,
    ) -> Vector<f32, Self> {
        counters::elementwise(a.len(), 2);
        match op {
            BinaryOp::Add => a + b,
            BinaryOp::Sub => a - b,
            BinaryOp::Mul => a * b,
            BinaryOp::Div => a / b,
            BinaryOp::Rem => a % b,
        }
    }

    fn vector_broadcast(
        a: &Vector<f32, Self>,
        scalar: f32,
        op: BinaryOp,
        scalar_left: bool,
    ) -> Vector<f32, Self> {
        counters::elementwise(a.len(), 1);
        if scalar_left {
            a.broadcast_left(scalar, op)
        } else {
            a.broadcast_right(scalar, op)
        }
    }

    fn vector_compare(
        a: &Vector<f32, Self>,
        b: &Vector<f32, Self>,
        op: Compare,
    ) -> Vector<f32, Self> {
        counters::elementwise(a.len(), 2);
        a.compare(b, op)
    }

    fn vector_compare_scalar(
        a: &Vector<f32, Self>,
        scalar: f32,
        op: Compare,
        scalar_left: bool,
    ) -> Vector<f32, Self> {
        counters::elementwise(a.len(), 1);
        a.compare_scalar(scalar, op, scalar_left)
    }

    fn vector_clamp(a: &Vector<f32, Self>, low: f32, high: f32) -> Vector<f32, Self> {
        counters::elementwise(a.len(), 1);
        a.clamp(low, high)
    }

    fn vector_reduce(a: &Vector<f32, Self>, op: Reduce) -> f32 {
        counters::kernel(a.len() * 4, 0);
        a.reduce(op)
    }

    fn vector_prefix_sum(a: &Vector<f32, Self>) -> Vector<f32, Self> {
        counters::elementwise(a.len(), 1);
        a.prefix_sum()
    }

    fn vector_sort(a: &Vector<f32, Self>, order: SortOrder) -> Vector<f32, Self> {
        counters::elementwise(a.len(), 1);
        a.sorted(order)
    }

    fn vector_unary(a: &Vector<f32, Self>, f: Analytic) -> Vector<f32, Self> {
        counters::elementwise(a.len(), 1);
        a.map(|&x| f.value(x))
    }

    fn vector_power(a: &Vector<f32, Self>, b: &Vector<f32, Self>) -> Vector<f32, Self> {
        counters::elementwise(a.len(), 2);
        a.pow_elementwise(b)
    }

    fn vector_power_scalar(
        a: &Vector<f32, Self>,
        scalar: f32,
        scalar_left: bool,
    ) -> Vector<f32, Self> {
        counters::elementwise(a.len(), 1);
        if scalar_left {
            a.map(|&x| scalar.powf(x))
        } else {
            a.pow(scalar)
        }
    }

    fn vector_unary_dual(
        value: &Vector<f32, Self>,
        tangent: &Vector<f32, Self>,
        f: Analytic,
    ) -> (Vector<f32, Self>, Vector<f32, Self>) {
        counters::kernel(4 * value.len() * 4, 2);
        assert_eq!(
            value.len(),
            tangent.len(),
            "unary_dual: value and tangent lengths differ"
        );
        let derivatives = value
            .data()
            .iter()
            .zip(tangent.data())
            .map(|(&x, &d)| f.derivative(x) * d)
            .collect::<Vec<_>>();
        (value.map(|&x| f.value(x)), Vector::new(derivatives))
    }

    fn dot(a: &Vector<f32, Self>, b: &Vector<f32, Self>) -> f32 {
        counters::kernel(2 * a.len() * 4, 0);
        a.dot(b)
    }

    fn vecmat(v: &Vector<f32, Self>, m: &Matrix<f32, Self>) -> Vector<f32, Self> {
        counters::kernel((v.len() + m.rows() * m.cols() + m.cols()) * 4, 1);
        v.vecmat(m)
    }

    fn matvec(m: &Matrix<f32, Self>, v: &Vector<f32, Self>) -> Vector<f32, Self> {
        counters::kernel((v.len() + m.rows() * m.cols() + m.rows()) * 4, 1);
        m.matvec(v)
    }

    fn matvec_add(
        m: &Matrix<f32, Self>,
        v: &Vector<f32, Self>,
        addend: Vector<f32, Self>,
    ) -> Vector<f32, Self> {
        counters::kernel((v.len() + m.rows() * m.cols() + 2 * m.rows()) * 4, 0);
        m.matvec_add(v, addend)
    }

    fn matrix_elementwise(
        a: &Matrix<f32, Self>,
        b: &Matrix<f32, Self>,
        op: BinaryOp,
    ) -> Matrix<f32, Self> {
        counters::elementwise(a.rows() * a.cols(), 2);
        match op {
            BinaryOp::Add => a + b,
            BinaryOp::Sub => a - b,
            BinaryOp::Mul => a * b,
            BinaryOp::Div => a / b,
            BinaryOp::Rem => a % b,
        }
    }

    fn matrix_broadcast(
        a: &Matrix<f32, Self>,
        scalar: f32,
        op: BinaryOp,
        scalar_left: bool,
    ) -> Matrix<f32, Self> {
        counters::elementwise(a.rows() * a.cols(), 1);
        if scalar_left {
            a.broadcast_left(scalar, op)
        } else {
            a.broadcast_right(scalar, op)
        }
    }

    fn matrix_compare(
        a: &Matrix<f32, Self>,
        b: &Matrix<f32, Self>,
        op: Compare,
    ) -> Matrix<f32, Self> {
        counters::elementwise(a.rows() * a.cols(), 2);
        a.compare(b, op)
    }

    fn matrix_compare_scalar(
        a: &Matrix<f32, Self>,
        scalar: f32,
        op: Compare,
        scalar_left: bool,
    ) -> Matrix<f32, Self> {
        counters::elementwise(a.rows() * a.cols(), 1);
        a.compare_scalar(scalar, op, scalar_left)
    }

    fn matrix_clamp(a: &Matrix<f32, Self>, low: f32, high: f32) -> Matrix<f32, Self> {
        counters::elementwise(a.rows() * a.cols(), 1);
        a.clamp(low, high)
    }

    fn matrix_unary(a: &Matrix<f32, Self>, f: Analytic) -> Matrix<f32, Self> {
        counters::elementwise(a.rows() * a.cols(), 1);
        a.map(|&x| f.value(x))
    }

    fn matrix_power(a: &Matrix<f32, Self>, b: &Matrix<f32, Self>) -> Matrix<f32, Self> {
        counters::elementwise(a.rows() * a.cols(), 2);
        a.pow_elementwise(b)
    }

    fn matrix_power_scalar(
        a: &Matrix<f32, Self>,
        scalar: f32,
        scalar_left: bool,
    ) -> Matrix<f32, Self> {
        counters::elementwise(a.rows() * a.cols(), 1);
        if scalar_left {
            a.map(|&x| scalar.powf(x))
        } else {
            a.pow(scalar)
        }
    }

    fn matrix_unary_dual(
        value: &Matrix<f32, Self>,
        tangent: &Matrix<f32, Self>,
        f: Analytic,
    ) -> (Matrix<f32, Self>, Matrix<f32, Self>) {
        counters::kernel(4 * value.rows() * value.cols() * 4, 2);
        assert_eq!(
            value.shape(),
            tangent.shape(),
            "unary_dual: value and tangent shapes differ"
        );
        let (rows, cols) = value.shape();
        let derivatives = value
            .data()
            .iter()
            .zip(tangent.data())
            .map(|(&x, &d)| f.derivative(x) * d)
            .collect::<Vec<_>>();
        (
            value.map(|&x| f.value(x)),
            Matrix::from_flat(rows, cols, derivatives),
        )
    }

    fn matmul(a: &Matrix<f32, Self>, b: &Matrix<f32, Self>) -> Matrix<f32, Self> {
        counters::kernel(
            (a.rows() * a.cols() + b.rows() * b.cols() + a.rows() * b.cols()) * 4,
            1,
        );
        a.matmul(b)
    }

    fn matmul_add(
        a: &Matrix<f32, Self>,
        b: &Matrix<f32, Self>,
        addend: Matrix<f32, Self>,
    ) -> Matrix<f32, Self> {
        counters::kernel(
            (a.rows() * a.cols() + b.rows() * b.cols() + 2 * a.rows() * b.cols()) * 4,
            0,
        );
        a.matmul_add(b, addend)
    }

    fn transpose(m: &Matrix<f32, Self>) -> Matrix<f32, Self> {
        counters::elementwise(m.rows() * m.cols(), 1);
        m.transpose()
    }

    fn correlate(
        input: &Matrix<f32, Self>,
        window: &Matrix<f32, Self>,
        flip: bool,
    ) -> Matrix<f32, Self> {
        counters::kernel(
            (input.rows() * input.cols()
                + window.rows() * window.cols()
                + correlation_shape(input.shape(), window.shape()).0
                    * correlation_shape(input.shape(), window.shape()).1)
                * 4,
            1,
        );
        let cols = input.cols();
        let (window_rows, window_cols) = window.shape();
        let (out_rows, out_cols) = correlation_shape(input.shape(), window.shape());
        let (values, taps) = (input.data(), window.data());

        let mut out = Vec::with_capacity(out_rows * out_cols);
        for row in 0..out_rows {
            for col in 0..out_cols {
                let mut sum = 0.0;
                for window_row in 0..window_rows {
                    for window_col in 0..window_cols {
                        let (tap_row, tap_col) = if flip {
                            (window_rows - 1 - window_row, window_cols - 1 - window_col)
                        } else {
                            (window_row, window_col)
                        };
                        sum += values[(row + window_row) * cols + col + window_col]
                            * taps[tap_row * window_cols + tap_col];
                    }
                }
                out.push(sum);
            }
        }
        Matrix::from_flat(out_rows, out_cols, out)
    }

    fn flip(input: &Matrix<f32, Self>) -> Matrix<f32, Self> {
        counters::elementwise(input.rows() * input.cols(), 1);
        let (rows, cols) = input.shape();
        let values = input.data();
        let mut out = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                out.push(values[(rows - 1 - row) * cols + cols - 1 - col]);
            }
        }
        Matrix::from_flat(rows, cols, out)
    }

    fn correlate_window_gradient(
        input: &Matrix<f32, Self>,
        adjoint: &Matrix<f32, Self>,
    ) -> Matrix<f32, Self> {
        // K̄[a][b] = Σᵢⱼ Ȳ[i][j]·X[i+a][j+b] — the input correlated with the
        // adjoint, which lands on exactly the window's shape.
        Self::correlate(input, adjoint, false)
    }

    fn correlate_input_gradient(
        adjoint: &Matrix<f32, Self>,
        window: &Matrix<f32, Self>,
        forward_flip: bool,
    ) -> Matrix<f32, Self> {
        counters::kernel(
            (adjoint.rows() * adjoint.cols()
                + window.rows() * window.cols()
                + (adjoint.rows() + window.rows() - 1) * (adjoint.cols() + window.cols() - 1))
                * 4,
            1,
        );
        // X̄[p][q] = Σᵤᵥ Ȳ[p−u][q−v]·K[u][v], with the taps reversed when the
        // forward pass reversed them. Out-of-range adjoint indices are the zeros
        // a full correlation pads with.
        let (out_rows, out_cols) = adjoint.shape();
        let (window_rows, window_cols) = window.shape();
        let (rows, cols) = (out_rows + window_rows - 1, out_cols + window_cols - 1);
        let (upstream, taps) = (adjoint.data(), window.data());

        let mut out = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                let mut sum = 0.0;
                for window_row in 0..window_rows {
                    for window_col in 0..window_cols {
                        if row < window_row || col < window_col {
                            continue;
                        }
                        let (source_row, source_col) = (row - window_row, col - window_col);
                        if source_row >= out_rows || source_col >= out_cols {
                            continue;
                        }
                        let (tap_row, tap_col) = if forward_flip {
                            (window_rows - 1 - window_row, window_cols - 1 - window_col)
                        } else {
                            (window_row, window_col)
                        };
                        sum += upstream[source_row * out_cols + source_col]
                            * taps[tap_row * window_cols + tap_col];
                    }
                }
                out.push(sum);
            }
        }
        Matrix::from_flat(rows, cols, out)
    }

    fn pad(input: &Matrix<f32, Self>, pad_rows: usize, pad_cols: usize) -> Matrix<f32, Self> {
        counters::kernel(
            (input.rows() * input.cols()
                + (input.rows() + 2 * pad_rows) * (input.cols() + 2 * pad_cols))
                * 4,
            1,
        );
        let (rows, cols) = input.shape();
        let (padded_rows, padded_cols) = (rows + 2 * pad_rows, cols + 2 * pad_cols);
        let values = input.data();

        let mut out = Vec::with_capacity(padded_rows * padded_cols);
        for row in 0..padded_rows {
            for col in 0..padded_cols {
                let inside = row >= pad_rows
                    && row < pad_rows + rows
                    && col >= pad_cols
                    && col < pad_cols + cols;
                out.push(if inside {
                    values[(row - pad_rows) * cols + col - pad_cols]
                } else {
                    0.0
                });
            }
        }
        Matrix::from_flat(padded_rows, padded_cols, out)
    }

    fn vector_moments(a: &Vector<f32, Self>) -> (f32, f32) {
        counters::kernel(2 * a.len() * 4, 0);
        moments_pair(&a.moments())
    }

    fn matrix_moments(a: &Matrix<f32, Self>) -> (f32, f32) {
        counters::kernel(2 * a.rows() * a.cols() * 4, 0);
        moments_pair(&a.moments())
    }

    fn matrix_axis_moments(
        a: &Matrix<f32, Self>,
        axis: Axis,
    ) -> (Vector<f32, Self>, Vector<f32, Self>) {
        counters::kernel(
            (2 * a.rows() * a.cols() + 2 * axis.extent(a.shape())) * 4,
            2,
        );
        let moments = a.moments_axis(axis);
        (moments.means, moments.sum_squared_deviations)
    }

    fn vector_distribution(
        a: &Vector<f32, Self>,
        family: Family,
        statistic: Statistic,
        parameters: (f32, f32),
    ) -> Vector<f32, Self> {
        counters::elementwise(a.len(), 1);
        a.distribution(
            statistic,
            &Distribution::from_parameters(family, parameters),
        )
    }

    fn matrix_distribution(
        a: &Matrix<f32, Self>,
        family: Family,
        statistic: Statistic,
        parameters: (f32, f32),
    ) -> Matrix<f32, Self> {
        counters::elementwise(a.rows() * a.cols(), 1);
        a.distribution(
            statistic,
            &Distribution::from_parameters(family, parameters),
        )
    }

    fn matrix_axis_distribution(
        a: &Matrix<f32, Self>,
        axis: Axis,
        family: Family,
        statistic: Statistic,
        first: &Vector<f32, Self>,
        second: &Vector<f32, Self>,
    ) -> Matrix<f32, Self> {
        counters::kernel(
            (2 * a.rows() * a.cols() + 2 * axis.extent(a.shape())) * 4,
            1,
        );
        a.distribution_axis(axis, statistic, &axis_distributions(family, first, second))
    }

    fn fused(
        program: &Program,
        shape: (usize, usize),
        inputs: &[Source<'_, Self>],
        updated: &mut [Sink<'_, Self>],
    ) -> Vec<Fresh<Self>> {
        fused::host(program, shape, inputs, updated)
    }
}

/// The pair the kernel interface passes moments around as, out of the named
/// summary the tensor API returns.
pub(crate) fn moments_pair(moments: &Moments<f32>) -> (f32, f32) {
    (moments.mean, moments.sum_squared_deviations)
}

/// Rebuild one distribution per row or column from the two parameter vectors
/// the axis kernels carry them in.
pub(crate) fn axis_distributions<B: Backend>(
    family: Family,
    first: &Vector<f32, B>,
    second: &Vector<f32, B>,
) -> Vec<Distribution<f32>> {
    first
        .as_slice()
        .iter()
        .zip(second.as_slice())
        .map(|(&first, &second)| Distribution::from_parameters(family, (first, second)))
        .collect()
}

/// The output shape of a valid correlation, which is also where the "does the
/// window fit" check lives now that both shapes are runtime values.
#[track_caller]
pub(crate) fn correlation_shape(input: (usize, usize), window: (usize, usize)) -> (usize, usize) {
    assert!(
        window.0 <= input.0 && window.1 <= input.1,
        "correlate: a {}×{} window does not fit in a {}×{} input",
        window.0,
        window.1,
        input.0,
        input.1
    );
    (input.0 - window.0 + 1, input.1 - window.1 + 1)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod gpu {
    use super::{
        Analytic, Axis, BinaryOp, Compare, Family, Fresh, Kernels, Matrix, Program, Reduce, Sink,
        SortOrder, Source, Statistic, Vector, correlation_shape, fused,
    };
    use crate::counters;
    use crate::tensors::metal_backend::{matrix_elementwise, vector_elementwise};
    use crate::tensors::{Host, Metal, MetalStorage};

    /// Forwarding again, but to the resident operations: every one of these
    /// leaves its result in GPU-shared memory.
    impl Kernels for Metal {
        fn vector_elementwise(
            a: &Vector<f32, Self>,
            b: &Vector<f32, Self>,
            op: BinaryOp,
        ) -> Vector<f32, Self> {
            counters::elementwise(a.len(), 2);
            vector_elementwise(a, b, op)
        }

        fn vector_broadcast(
            a: &Vector<f32, Self>,
            scalar: f32,
            op: BinaryOp,
            scalar_left: bool,
        ) -> Vector<f32, Self> {
            counters::elementwise(a.len(), 1);
            if scalar_left {
                a.broadcast_left(scalar, op)
            } else {
                a.broadcast_right(scalar, op)
            }
        }

        fn vector_compare(
            a: &Vector<f32, Self>,
            b: &Vector<f32, Self>,
            op: Compare,
        ) -> Vector<f32, Self> {
            counters::elementwise(a.len(), 2);
            a.compare(b, op)
        }

        fn vector_compare_scalar(
            a: &Vector<f32, Self>,
            scalar: f32,
            op: Compare,
            scalar_left: bool,
        ) -> Vector<f32, Self> {
            counters::elementwise(a.len(), 1);
            a.compare_scalar(scalar, op, scalar_left)
        }

        fn vector_clamp(a: &Vector<f32, Self>, low: f32, high: f32) -> Vector<f32, Self> {
            counters::elementwise(a.len(), 1);
            a.clamp(low, high)
        }

        fn vector_reduce(a: &Vector<f32, Self>, op: Reduce) -> f32 {
            counters::kernel(a.len() * 4, 0);
            a.reduce(op)
        }

        fn vector_prefix_sum(a: &Vector<f32, Self>) -> Vector<f32, Self> {
            counters::elementwise(a.len(), 1);
            a.prefix_sum()
        }

        fn vector_sort(a: &Vector<f32, Self>, order: SortOrder) -> Vector<f32, Self> {
            counters::elementwise(a.len(), 1);
            a.sorted(order)
        }

        fn vector_unary(a: &Vector<f32, Self>, f: Analytic) -> Vector<f32, Self> {
            counters::elementwise(a.len(), 1);
            a.analytic(f)
        }

        fn vector_power(a: &Vector<f32, Self>, b: &Vector<f32, Self>) -> Vector<f32, Self> {
            counters::elementwise(a.len(), 2);
            a.pow_elementwise(b)
        }

        fn vector_power_scalar(
            a: &Vector<f32, Self>,
            scalar: f32,
            scalar_left: bool,
        ) -> Vector<f32, Self> {
            counters::elementwise(a.len(), 1);
            a.power_scalar(scalar, scalar_left)
        }

        fn vector_unary_dual(
            value: &Vector<f32, Self>,
            tangent: &Vector<f32, Self>,
            f: Analytic,
        ) -> (Vector<f32, Self>, Vector<f32, Self>) {
            counters::kernel(4 * value.len() * 4, 2);
            match value.storage().unary_dual(tangent.storage(), f) {
                Some((v, t)) => (
                    Vector::from_storage(value.len(), v),
                    Vector::from_storage(tangent.len(), t),
                ),
                None => {
                    let (v, t) = Host::vector_unary_dual(
                        &value.to_backend::<Host>(),
                        &tangent.to_backend::<Host>(),
                        f,
                    );
                    (v.to_backend(), t.to_backend())
                }
            }
        }

        fn dot(a: &Vector<f32, Self>, b: &Vector<f32, Self>) -> f32 {
            counters::kernel(2 * a.len() * 4, 0);
            a.dot(b)
        }

        fn vecmat(v: &Vector<f32, Self>, m: &Matrix<f32, Self>) -> Vector<f32, Self> {
            counters::kernel((v.len() + m.rows() * m.cols() + m.cols()) * 4, 1);
            v.vecmat(m)
        }

        fn matvec(m: &Matrix<f32, Self>, v: &Vector<f32, Self>) -> Vector<f32, Self> {
            counters::kernel((v.len() + m.rows() * m.cols() + m.rows()) * 4, 1);
            m.matvec(v)
        }

        fn matvec_add(
            m: &Matrix<f32, Self>,
            v: &Vector<f32, Self>,
            addend: Vector<f32, Self>,
        ) -> Vector<f32, Self> {
            counters::kernel((v.len() + m.rows() * m.cols() + 2 * m.rows()) * 4, 0);
            m.matvec_add(v, addend)
        }

        fn matrix_elementwise(
            a: &Matrix<f32, Self>,
            b: &Matrix<f32, Self>,
            op: BinaryOp,
        ) -> Matrix<f32, Self> {
            counters::elementwise(a.rows() * a.cols(), 2);
            matrix_elementwise(a, b, op)
        }

        fn matrix_broadcast(
            a: &Matrix<f32, Self>,
            scalar: f32,
            op: BinaryOp,
            scalar_left: bool,
        ) -> Matrix<f32, Self> {
            counters::elementwise(a.rows() * a.cols(), 1);
            if scalar_left {
                a.broadcast_left(scalar, op)
            } else {
                a.broadcast_right(scalar, op)
            }
        }

        fn matrix_compare(
            a: &Matrix<f32, Self>,
            b: &Matrix<f32, Self>,
            op: Compare,
        ) -> Matrix<f32, Self> {
            counters::elementwise(a.rows() * a.cols(), 2);
            a.compare(b, op)
        }

        fn matrix_compare_scalar(
            a: &Matrix<f32, Self>,
            scalar: f32,
            op: Compare,
            scalar_left: bool,
        ) -> Matrix<f32, Self> {
            counters::elementwise(a.rows() * a.cols(), 1);
            a.compare_scalar(scalar, op, scalar_left)
        }

        fn matrix_clamp(a: &Matrix<f32, Self>, low: f32, high: f32) -> Matrix<f32, Self> {
            counters::elementwise(a.rows() * a.cols(), 1);
            a.clamp(low, high)
        }

        fn matrix_unary(a: &Matrix<f32, Self>, f: Analytic) -> Matrix<f32, Self> {
            counters::elementwise(a.rows() * a.cols(), 1);
            a.analytic(f)
        }

        fn matrix_power(a: &Matrix<f32, Self>, b: &Matrix<f32, Self>) -> Matrix<f32, Self> {
            counters::elementwise(a.rows() * a.cols(), 2);
            a.pow_elementwise(b)
        }

        fn matrix_power_scalar(
            a: &Matrix<f32, Self>,
            scalar: f32,
            scalar_left: bool,
        ) -> Matrix<f32, Self> {
            counters::elementwise(a.rows() * a.cols(), 1);
            a.power_scalar(scalar, scalar_left)
        }

        fn matrix_unary_dual(
            value: &Matrix<f32, Self>,
            tangent: &Matrix<f32, Self>,
            f: Analytic,
        ) -> (Matrix<f32, Self>, Matrix<f32, Self>) {
            counters::kernel(4 * value.rows() * value.cols() * 4, 2);
            let (rows, cols) = value.shape();
            match value.storage().unary_dual(tangent.storage(), f) {
                Some((v, t)) => (
                    Matrix::from_storage(rows, cols, v),
                    Matrix::from_storage(rows, cols, t),
                ),
                None => {
                    let (v, t) = Host::matrix_unary_dual(
                        &value.to_backend::<Host>(),
                        &tangent.to_backend::<Host>(),
                        f,
                    );
                    (v.to_backend(), t.to_backend())
                }
            }
        }

        fn matmul(a: &Matrix<f32, Self>, b: &Matrix<f32, Self>) -> Matrix<f32, Self> {
            counters::kernel(
                (a.rows() * a.cols() + b.rows() * b.cols() + a.rows() * b.cols()) * 4,
                1,
            );
            a.matmul(b)
        }

        fn matmul_add(
            a: &Matrix<f32, Self>,
            b: &Matrix<f32, Self>,
            addend: Matrix<f32, Self>,
        ) -> Matrix<f32, Self> {
            counters::kernel(
                (a.rows() * a.cols() + b.rows() * b.cols() + 2 * a.rows() * b.cols()) * 4,
                0,
            );
            a.matmul_add(b, addend)
        }

        fn transpose(m: &Matrix<f32, Self>) -> Matrix<f32, Self> {
            counters::elementwise(m.rows() * m.cols(), 1);
            m.transpose()
        }

        fn correlate(
            input: &Matrix<f32, Self>,
            window: &Matrix<f32, Self>,
            flip: bool,
        ) -> Matrix<f32, Self> {
            counters::kernel(
                (input.rows() * input.cols()
                    + window.rows() * window.cols()
                    + correlation_shape(input.shape(), window.shape()).0
                        * correlation_shape(input.shape(), window.shape()).1)
                    * 4,
                1,
            );
            let (rows, cols) = input.shape();
            let (window_rows, window_cols) = window.shape();
            let (out_rows, out_cols) = correlation_shape(input.shape(), window.shape());
            match input.storage().correlate(
                window.storage(),
                rows,
                cols,
                window_rows,
                window_cols,
                flip,
            ) {
                Some(data) => Matrix::from_storage(out_rows, out_cols, data),
                None => Host::correlate(
                    &input.to_backend::<Host>(),
                    &window.to_backend::<Host>(),
                    flip,
                )
                .to_backend(),
            }
        }

        fn flip(input: &Matrix<f32, Self>) -> Matrix<f32, Self> {
            counters::elementwise(input.rows() * input.cols(), 1);
            let (rows, cols) = input.shape();
            match input.storage().flip(rows, cols) {
                Some(data) => Matrix::from_storage(rows, cols, data),
                None => Host::flip(&input.to_backend::<Host>()).to_backend(),
            }
        }

        fn correlate_window_gradient(
            input: &Matrix<f32, Self>,
            adjoint: &Matrix<f32, Self>,
        ) -> Matrix<f32, Self> {
            // Correlating the input with the adjoint as the window leaves
            // exactly the window's shape, and the storage layer takes the
            // extents as numbers.
            Self::correlate(input, adjoint, false)
        }

        fn correlate_input_gradient(
            adjoint: &Matrix<f32, Self>,
            window: &Matrix<f32, Self>,
            forward_flip: bool,
        ) -> Matrix<f32, Self> {
            counters::kernel(
                (adjoint.rows() * adjoint.cols()
                    + window.rows() * window.cols()
                    + (adjoint.rows() + window.rows() - 1) * (adjoint.cols() + window.cols() - 1))
                    * 4,
                1,
            );
            let (out_rows, out_cols) = adjoint.shape();
            let (window_rows, window_cols) = window.shape();
            let (rows, cols) = (out_rows + window_rows - 1, out_cols + window_cols - 1);

            let padded =
                adjoint
                    .storage()
                    .pad(out_rows, out_cols, window_rows - 1, window_cols - 1);
            let full = padded.and_then(|padded| {
                padded.correlate(
                    window.storage(),
                    rows + window_rows - 1,
                    cols + window_cols - 1,
                    window_rows,
                    window_cols,
                    !forward_flip,
                )
            });
            match full {
                Some(data) => Matrix::from_storage(rows, cols, data),
                None => Host::correlate_input_gradient(
                    &adjoint.to_backend::<Host>(),
                    &window.to_backend::<Host>(),
                    forward_flip,
                )
                .to_backend(),
            }
        }

        fn pad(input: &Matrix<f32, Self>, pad_rows: usize, pad_cols: usize) -> Matrix<f32, Self> {
            counters::kernel(
                (input.rows() * input.cols()
                    + (input.rows() + 2 * pad_rows) * (input.cols() + 2 * pad_cols))
                    * 4,
                1,
            );
            let (rows, cols) = input.shape();
            let (padded_rows, padded_cols) = (rows + 2 * pad_rows, cols + 2 * pad_cols);
            match input.storage().pad(rows, cols, pad_rows, pad_cols) {
                Some(data) => Matrix::from_storage(padded_rows, padded_cols, data),
                None => Host::pad(&input.to_backend::<Host>(), pad_rows, pad_cols).to_backend(),
            }
        }

        fn vector_moments(a: &Vector<f32, Self>) -> (f32, f32) {
            counters::kernel(2 * a.len() * 4, 0);
            resident_moments(a.storage(), a.len())
                .unwrap_or_else(|| Host::vector_moments(&a.to_backend::<Host>()))
        }

        fn matrix_moments(a: &Matrix<f32, Self>) -> (f32, f32) {
            counters::kernel(2 * a.rows() * a.cols() * 4, 0);
            resident_moments(a.storage(), a.rows() * a.cols())
                .unwrap_or_else(|| Host::matrix_moments(&a.to_backend::<Host>()))
        }

        fn matrix_axis_moments(
            a: &Matrix<f32, Self>,
            axis: Axis,
        ) -> (Vector<f32, Self>, Vector<f32, Self>) {
            counters::kernel(
                (2 * a.rows() * a.cols() + 2 * axis.extent(a.shape())) * 4,
                2,
            );
            let (rows, cols) = a.shape();
            let extent = axis.extent((rows, cols));
            // An empty fold has no mean, and the shader has no thread to write
            // that NaN with; the host owns the convention.
            let resident = (rows != 0 && cols != 0)
                .then(|| a.storage().axis_moments(rows, cols, axis))
                .flatten();
            match resident {
                Some((means, deviations)) => (
                    Vector::from_storage(extent, means),
                    Vector::from_storage(extent, deviations),
                ),
                None => {
                    let (means, deviations) =
                        Host::matrix_axis_moments(&a.to_backend::<Host>(), axis);
                    (means.to_backend(), deviations.to_backend())
                }
            }
        }

        fn vector_distribution(
            a: &Vector<f32, Self>,
            family: Family,
            statistic: Statistic,
            parameters: (f32, f32),
        ) -> Vector<f32, Self> {
            counters::elementwise(a.len(), 1);
            match a.storage().distribution(family, statistic, parameters) {
                Some(data) => Vector::from_storage(a.len(), data),
                None => Host::vector_distribution(
                    &a.to_backend::<Host>(),
                    family,
                    statistic,
                    parameters,
                )
                .to_backend(),
            }
        }

        fn matrix_distribution(
            a: &Matrix<f32, Self>,
            family: Family,
            statistic: Statistic,
            parameters: (f32, f32),
        ) -> Matrix<f32, Self> {
            counters::elementwise(a.rows() * a.cols(), 1);
            let (rows, cols) = a.shape();
            match a.storage().distribution(family, statistic, parameters) {
                Some(data) => Matrix::from_storage(rows, cols, data),
                None => Host::matrix_distribution(
                    &a.to_backend::<Host>(),
                    family,
                    statistic,
                    parameters,
                )
                .to_backend(),
            }
        }

        fn matrix_axis_distribution(
            a: &Matrix<f32, Self>,
            axis: Axis,
            family: Family,
            statistic: Statistic,
            first: &Vector<f32, Self>,
            second: &Vector<f32, Self>,
        ) -> Matrix<f32, Self> {
            counters::kernel(
                (2 * a.rows() * a.cols() + 2 * axis.extent(a.shape())) * 4,
                1,
            );
            let (rows, cols) = a.shape();
            let resident = a.storage().axis_distribution(
                first.storage(),
                second.storage(),
                (rows, cols),
                axis,
                family,
                statistic,
            );
            match resident {
                Some(data) => Matrix::from_storage(rows, cols, data),
                None => Host::matrix_axis_distribution(
                    &a.to_backend::<Host>(),
                    axis,
                    family,
                    statistic,
                    &first.to_backend::<Host>(),
                    &second.to_backend::<Host>(),
                )
                .to_backend(),
            }
        }

        fn fused(
            program: &Program,
            shape: (usize, usize),
            inputs: &[Source<'_, Self>],
            updated: &mut [Sink<'_, Self>],
        ) -> Vec<Fresh<Self>> {
            fused::metal(program, shape, inputs, updated)
        }
    }

    /// Both moment passes over one resident allocation.
    ///
    /// The mean has to reach the CPU before the deviation pass can be encoded
    /// with it, so this is two dispatched reductions with a synchronization
    /// between them rather than one fused kernel — the same shape the two-pass
    /// variance takes everywhere else. `None` where there is no device, which
    /// sends the caller to the host.
    fn resident_moments(storage: &MetalStorage, len: usize) -> Option<(f32, f32)> {
        if len == 0 {
            return None;
        }
        let mean = storage.reduce(Reduce::Sum)? / len as f32;
        Some((mean, storage.sum_squared_deviations(mean)?))
    }
}
