//! The analytic functions, applied elementwise to tensors.
//!
//! `sin`, `exp`, `ln` and the rest are defined on single numbers by the traits
//! in [`crate::numbers`]. A tensor applies one of them to every element and
//! keeps its shape, so `exp` of a `3×4` matrix is a `3×4` matrix. This module is
//! where that lifting happens, in three overlapping spellings:
//!
//! * **Inherent methods on host tensors** — [`Vector<T, Host>`] and
//!   [`Matrix<T, Host>`] get `sin`, `exp`, … for every element type that
//!   implements the matching scalar trait. The floats — `f64`, `f32` and the
//!   compact `f16` and `bf16`, which evaluate in `f32` and round once — stay
//!   themselves and integers widen to `f64`, exactly as they do on their own; the
//!   [`Complex`](crate::numbers::Complex) and [`Dual`](crate::numbers::Dual)
//!   definitions carry over too, so the `ε` part of a dual tensor comes back
//!   holding the derivative, elementwise.
//!
//! * **Inherent methods on resident tensors** — the same names on
//!   `Vector<T, Metal>` and `Matrix<T, Metal>` for `f32`, `f16` and `bf16`,
//!   each running that type's GPU unary kernel and leaving the result in
//!   GPU-shared memory.
//!
//! * **[`Transcendental`]** — the same operations once more, for code that is
//!   generic over the backend. This is to the unary kernels what
//!   [`Ordered`](super::kernels::Ordered) is to the comparison ones.
//!
//! The scalar traits are also implemented for `&Vector` and `&Matrix`, so a
//! function written once over [`Exp`] accepts a number and a tensor alike:
//!
//! ```
//! use tensorcrate::numbers::Exp;
//! use tensorcrate::tensors::Vector;
//!
//! fn decay<T: Exp>(x: T) -> T::Output {
//!     x.exp()
//! }
//!
//! assert_eq!(decay(0.0), 1.0);
//! assert_eq!(decay(&Vector::new([0.0, 0.0])).data(), [1.0, 1.0]);
//! ```
//!
//! [`pow`](Vector::pow) is the one member of the family that takes two operands.
//! It is spelled the same three ways, and comes in an elementwise form —
//! `aᵢ^bᵢ` between two tensors of the same shape — as well as the fixed-exponent
//! one. It is deliberately not a [`BinaryOp`](super::kernels::BinaryOp): that
//! enum's variants are the operators defined for every
//! [`Coefficient`](crate::numbers::Coefficient), and raising an integer to an
//! integer leaves the integers, so `pow` widens where `+` does not.
//!
//! Reciprocal is the one unary op in [`crate::numbers`] with no place here:
//! there is no `recip` shader, and `1.0 / v` already divides elementwise.
//!
//! [`Vector<T, Host>`]: Vector
//! [`Matrix<T, Host>`]: Matrix

use crate::numbers::{
    Arccos, Arcsin, Arctan, Cos, Cosh, Csc, Exp, Ln, Power, Real, Sec, Sin, Sinh, Sqrt, Tan, Tanh,
    bf16, f16,
};
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::metal::MetalElement;
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::tensors::Metal;
use crate::tensors::kernels::{Analytic, Kernels};
use crate::tensors::{Host, Matrix, Vector, assert_same_len, assert_same_shape};

impl<T: Real> Vector<T, Host> {
    /// Apply an analytic function to every element.
    ///
    /// The named methods — [`sin`](Vector::sin), [`exp`](Vector::exp), … — are
    /// this with the function fixed, and read better when it is. This form is
    /// for when the function is a value: chosen at runtime, or forwarded from a
    /// caller. `Vector<f32, Metal>` has the same method, backed by the GPU
    /// kernel.
    ///
    /// ```
    /// use tensorcrate::tensors::{Analytic, Vector};
    ///
    /// let v = Vector::new([0.0f32, 1.0]);
    /// assert_eq!(v.analytic(Analytic::Exp).data(), v.exp().data());
    /// ```
    pub fn analytic(&self, f: Analytic) -> Self {
        let mut out = vec![T::zero(); self.len()];
        crate::vmath::unary_slice(f, self.data(), &mut out);
        Vector::new(out)
    }
}

impl<T: Real> Matrix<T, Host> {
    /// Apply an analytic function to every element; see [`Vector::analytic`].
    pub fn analytic(&self, f: Analytic) -> Self {
        let mut out = vec![T::zero(); self.rows() * self.cols()];
        crate::vmath::unary_slice(f, self.data(), &mut out);
        Matrix::from_flat(self.rows(), self.cols(), out)
    }
}

/// Generate the elementwise surface for every analytic function at once.
///
/// Each function is named several times over: as a method on
/// [`Transcendental`], as an inherent method on the host and resident tensors,
/// and as the scalar trait implemented for tensor references. The scalar trait
/// in [`crate::numbers`] and the [`Analytic`] variant share a name, which is
/// what lets one list drive all of them.
macro_rules! elementwise_analytic {
    ($($method:ident => $Trait:ident),+ $(,)?) => {
        /// The analytic functions, spelled as methods, on whatever backend the
        /// tensor is on.
        ///
        /// [`Vector<f32, Host>`] and [`Vector<f32, Metal>`] have all of these as
        /// inherent methods already, and so do the two matrix types. None of
        /// those is reachable when the backend is a *type parameter*, though —
        /// the same bind [`Ordered`](super::kernels::Ordered) exists to undo —
        /// so backend-generic code goes through here:
        ///
        /// ```
        /// use tensorcrate::numbers::Real;
        /// use tensorcrate::tensors::{Kernels, Transcendental, Vector};
        ///
        /// // Any element type the backend computes in: `f32` and `f64` on the
        /// // host, `f32` on Metal.
        /// fn squash<T: Real, B: Kernels<T>>(v: &Vector<T, B>) -> Vector<T, B> {
        ///     v.tanh()
        /// }
        ///
        /// assert_eq!(squash(&Vector::new([0.0f32])).data(), [0.0]);
        /// assert_eq!(squash(&Vector::new([0.0f64])).data(), [0.0]);
        /// ```
        ///
        /// On a concrete backend the inherent method wins method resolution and
        /// this trait is never consulted; both run the same kernel, so which one
        /// resolved is not observable. The inherent host methods also cover
        /// element types this trait cannot — `Vector<i32, Host>::exp` widens to
        /// `f64` — while `Metal` implements the trait for the types its kernels
        /// are compiled for: `f32`, `f16` and `bf16`.
        ///
        /// Every named method is [`analytic`](Transcendental::analytic) with the
        /// function fixed, so implementing that one implements them all.
        ///
        /// [`Vector<f32, Host>`]: Vector
        /// [`Vector<f32, Metal>`]: Vector
        pub trait Transcendental: Sized {
            /// The element type the functions are evaluated in.
            type Elem: Real;

            /// Apply `f` to every element.
            fn analytic(&self, f: Analytic) -> Self;

            /// Raise every element to `exponent`.
            fn pow(&self, exponent: Self::Elem) -> Self;

            /// Raise every element to the matching element of `exponents`.
            ///
            /// # Panics
            ///
            /// Unless the two tensors have the same shape.
            fn pow_elementwise(&self, exponents: &Self) -> Self;

            $(
                #[doc = concat!("Elementwise `", stringify!($method), "`.")]
                fn $method(&self) -> Self {
                    self.analytic(Analytic::$Trait)
                }
            )+
        }

        $(
            // ---- host tensors, for every element type the scalar trait covers ----

            impl<T: $Trait + Copy> Vector<T, Host> {
                #[doc = concat!("Elementwise `", stringify!($method), "`.")]
                pub fn $method(&self) -> Vector<<T as $Trait>::Output, Host> {
                    self.map(|&x| <T as $Trait>::$method(x))
                }
            }

            impl<T: $Trait + Copy> Matrix<T, Host> {
                #[doc = concat!("Elementwise `", stringify!($method), "`.")]
                pub fn $method(&self) -> Matrix<<T as $Trait>::Output, Host> {
                    self.map(|&x| <T as $Trait>::$method(x))
                }
            }

            // ---- the scalar traits, so generic code accepts tensors ----

            impl<T: $Trait + Copy> $Trait for &Vector<T, Host> {
                type Output = Vector<<T as $Trait>::Output, Host>;

                fn $method(self) -> Self::Output {
                    self.map(|&x| <T as $Trait>::$method(x))
                }
            }

            impl<T: $Trait + Copy> $Trait for &Matrix<T, Host> {
                type Output = Matrix<<T as $Trait>::Output, Host>;

                fn $method(self) -> Self::Output {
                    self.map(|&x| <T as $Trait>::$method(x))
                }
            }

            // ---- resident tensors: the GPU unary kernel ----

            #[cfg(all(feature = "metal", target_os = "macos"))]
            impl<T: MetalElement> Vector<T, Metal> {
                #[doc = concat!("Elementwise `", stringify!($method), "`, on the GPU.")]
                pub fn $method(&self) -> Self {
                    self.analytic(Analytic::$Trait)
                }
            }

            #[cfg(all(feature = "metal", target_os = "macos"))]
            impl<T: MetalElement> Matrix<T, Metal> {
                #[doc = concat!("Elementwise `", stringify!($method), "`, on the GPU.")]
                pub fn $method(&self) -> Self {
                    self.analytic(Analytic::$Trait)
                }
            }

            #[cfg(all(feature = "metal", target_os = "macos"))]
            impl<T: MetalElement> $Trait for &Vector<T, Metal> {
                type Output = Vector<T, Metal>;

                fn $method(self) -> Self::Output {
                    self.analytic(Analytic::$Trait)
                }
            }

            #[cfg(all(feature = "metal", target_os = "macos"))]
            impl<T: MetalElement> $Trait for &Matrix<T, Metal> {
                type Output = Matrix<T, Metal>;

                fn $method(self) -> Self::Output {
                    self.analytic(Analytic::$Trait)
                }
            }
        )+
    };
}

elementwise_analytic!(
    sin => Sin,
    cos => Cos,
    tan => Tan,
    sec => Sec,
    csc => Csc,
    arcsin => Arcsin,
    arccos => Arccos,
    arctan => Arctan,
    exp => Exp,
    ln => Ln,
    sinh => Sinh,
    cosh => Cosh,
    tanh => Tanh,
    sqrt => Sqrt,
);

// ---- powers ------------------------------------------------------------------

impl<T: Power + Copy + 'static> Vector<T, Host> {
    /// Raise every element to `exponent`.
    ///
    /// ```
    /// use tensorcrate::tensors::Vector;
    ///
    /// let v = Vector::new([1.0, 2.0, 3.0]);
    /// assert_eq!(v.pow(2.0).data(), [1.0, 4.0, 9.0]);
    /// ```
    ///
    /// The exponent is inspected once rather than per element, so the common
    /// powers — squaring, the square root, the reciprocal — run as the single
    /// arithmetic operation they are instead of a call into `powf`.
    ///
    /// Each of those is one IEEE operation and therefore correctly rounded,
    /// which `powf` is not: it is accurate to under an ulp, so on about one
    /// input in a thousand the two disagree in the last bit, with the fast path
    /// holding the better answer. A cube is *not* on the list, because `x·x·x`
    /// rounds twice and would be the worse one.
    pub fn pow(&self, exponent: T) -> Vector<<T as Power>::Output, Host>
    where
        <T as Power>::Output: 'static,
    {
        Vector::new(power_scalar(self.data(), exponent))
    }

    /// Raise every element to the matching element of `exponents`.
    ///
    /// # Panics
    ///
    /// Unless the two vectors have the same length.
    #[track_caller]
    pub fn pow_elementwise(
        &self,
        exponents: &Vector<T, Host>,
    ) -> Vector<<T as Power>::Output, Host> {
        assert_same_len(self.len(), exponents.len(), "pow");
        Vector::new(power_pairs(self.data(), exponents.data()))
    }
}

impl<T: Power + Copy + 'static> Matrix<T, Host> {
    /// Raise every element to `exponent`; see [`Vector::pow`].
    pub fn pow(&self, exponent: T) -> Matrix<<T as Power>::Output, Host>
    where
        <T as Power>::Output: 'static,
    {
        Matrix::from_flat(
            self.rows(),
            self.cols(),
            power_scalar(self.data(), exponent),
        )
    }

    /// Raise every element to the matching element of `exponents`.
    ///
    /// # Panics
    ///
    /// Unless the two matrices have the same shape.
    #[track_caller]
    pub fn pow_elementwise(
        &self,
        exponents: &Matrix<T, Host>,
    ) -> Matrix<<T as Power>::Output, Host> {
        assert_same_shape(self.shape(), exponents.shape(), "pow");
        Matrix::from_flat(
            self.rows(),
            self.cols(),
            power_pairs(self.data(), exponents.data()),
        )
    }
}

/// `values^exponent` over a whole buffer, through a fast path where the
/// exponent allows one.
fn power_scalar<T: Power + Copy + 'static>(values: &[T], exponent: T) -> Vec<<T as Power>::Output>
where
    <T as Power>::Output: 'static,
{
    if let Some(fast) = fast_power(values, exponent) {
        return fast;
    }
    values.iter().map(|&value| value.power(exponent)).collect()
}

/// The exponents that reduce to a single arithmetic operation.
///
/// Being *one* operation is the criterion for membership, because a single
/// IEEE operation is correctly rounded — so the fast path is never less
/// accurate than the `powf` it replaces, and is sometimes a bit more so.
/// (`powf` is accurate to under an ulp rather than correctly rounded, so the
/// two differ in the last bit on roughly one input in a thousand.) A cube is
/// excluded for the same reason the others are included: `x·x·x` rounds twice,
/// which would be a real loss rather than a rounding difference. `x⁻²` is
/// excluded likewise.
#[derive(Copy, Clone, PartialEq, Eq)]
enum FastExponent {
    /// `x⁰ = 1`, for every `x` including NaN. Exactly what `powf` answers.
    Zero,
    /// `x¹ = x`, sign of zero and NaN payload intact. Also exact.
    One,
    Square,
    /// `x^½`. `sqrt` answers `−0` where `powf` answers `+0`, which the added
    /// zero repairs, and NaN at `−∞` where `powf` answers `+∞`.
    Root,
    Reciprocal,
}

impl FastExponent {
    /// Recognize an exponent, comparing as `f64` because that holds every `f32`
    /// exactly.
    fn of(exponent: f64) -> Option<Self> {
        Some(match exponent {
            e if e == 0.0 => Self::Zero,
            e if e == 1.0 => Self::One,
            e if e == 2.0 => Self::Square,
            e if e == 0.5 => Self::Root,
            e if e == -1.0 => Self::Reciprocal,
            _ => return None,
        })
    }
}

/// Generates the specialized loop for one float type. Each arm is written as
/// its own pass so the body carries no branch and vectorizes.
macro_rules! fast_power_impl {
    ($name:ident, $t:ty) => {
        fn $name(values: &[$t], exponent: $t) -> Option<Vec<$t>> {
            let mut out = Vec::with_capacity(values.len());
            match FastExponent::of(exponent as f64)? {
                FastExponent::Zero => out.resize(values.len(), 1.0),
                FastExponent::One => out.extend_from_slice(values),
                FastExponent::Square => out.extend(values.iter().map(|&x| x * x)),
                FastExponent::Root => out.extend(values.iter().map(|&x| {
                    // `pow(−∞, ½) = +∞`, where `sqrt(−∞)` is NaN.
                    let root = x.sqrt() + 0.0;
                    if x == <$t>::NEG_INFINITY { <$t>::INFINITY } else { root }
                })),
                FastExponent::Reciprocal => out.extend(values.iter().map(|&x| 1.0 / x)),
            }
            Some(out)
        }
    };
}

fast_power_impl!(fast_power_f32, f32);
fast_power_impl!(fast_power_f64, f64);

/// Try the fast path, for the element types that have one.
///
/// The two conditions are that the element is `f32` or `f64` and that raising
/// it leaves it in the same type — which is why the integers, whose powers
/// widen to `f64`, do not come through here.
fn fast_power<T: Power + Copy + 'static>(
    values: &[T],
    exponent: T,
) -> Option<Vec<<T as Power>::Output>>
where
    <T as Power>::Output: 'static,
{
    // SAFETY for all three casts below: the `TypeId` comparisons establish that
    // `T` and the output type are exactly the concrete float named, so the
    // slice, the scalar, and the returned buffer all have identical layout.
    unsafe {
        if same::<T, f32>() && same::<<T as Power>::Output, f32>() {
            let values = std::slice::from_raw_parts(values.as_ptr().cast::<f32>(), values.len());
            let exponent = std::ptr::read((&exponent as *const T).cast::<f32>());
            return fast_power_f32(values, exponent).map(|out| retype(out));
        }
        if same::<T, f64>() && same::<<T as Power>::Output, f64>() {
            let values = std::slice::from_raw_parts(values.as_ptr().cast::<f64>(), values.len());
            let exponent = std::ptr::read((&exponent as *const T).cast::<f64>());
            return fast_power_f64(values, exponent).map(|out| retype(out));
        }
    }
    None
}

fn same<T: 'static, U: 'static>() -> bool {
    std::any::TypeId::of::<T>() == std::any::TypeId::of::<U>()
}

/// Reinterpret a `Vec<T>` as a `Vec<U>`.
///
/// # Safety
///
/// `T` and `U` must be the same type; every caller has just checked that with
/// [`same`].
unsafe fn retype<T, U>(values: Vec<T>) -> Vec<U> {
    let mut values = std::mem::ManuallyDrop::new(values);
    unsafe {
        Vec::from_raw_parts(
            values.as_mut_ptr().cast::<U>(),
            values.len(),
            values.capacity(),
        )
    }
}

/// `basesᵢ^exponentsᵢ` over two equal-length buffers.
fn power_pairs<T: Power + Copy>(bases: &[T], exponents: &[T]) -> Vec<<T as Power>::Output> {
    bases
        .iter()
        .zip(exponents)
        .map(|(&base, &exponent)| base.power(exponent))
        .collect()
}

// ---- the scalar trait, so generic code accepts tensors ----

impl<T: Power + Copy + 'static> Power<T> for &Vector<T, Host>
where
    <T as Power>::Output: 'static,
{
    type Output = Vector<<T as Power>::Output, Host>;

    fn power(self, exponent: T) -> Self::Output {
        self.pow(exponent)
    }
}

impl<T: Power + Copy + 'static> Power<T> for &Matrix<T, Host>
where
    <T as Power>::Output: 'static,
{
    type Output = Matrix<<T as Power>::Output, Host>;

    fn power(self, exponent: T) -> Self::Output {
        self.pow(exponent)
    }
}

impl<T: Power + Copy + 'static> Power<&Vector<T, Host>> for &Vector<T, Host> {
    type Output = Vector<<T as Power>::Output, Host>;

    #[track_caller]
    fn power(self, exponents: &Vector<T, Host>) -> Self::Output {
        self.pow_elementwise(exponents)
    }
}

impl<T: Power + Copy + 'static> Power<&Matrix<T, Host>> for &Matrix<T, Host> {
    type Output = Matrix<<T as Power>::Output, Host>;

    #[track_caller]
    fn power(self, exponents: &Matrix<T, Host>) -> Self::Output {
        self.pow_elementwise(exponents)
    }
}

/// A scalar base with a tensor exponent — `2^v`, elementwise.
///
/// This is the one order with no method to hang it on, since the tensor is the
/// argument rather than the receiver. The element types are named concretely
/// because a blanket impl over the scalar would collide with the two above.
macro_rules! scalar_base_power {
    ($($t:ty),+ $(,)?) => {$(
        impl Power<&Vector<$t, Host>> for $t {
            type Output = Vector<$t, Host>;

            fn power(self, exponents: &Vector<$t, Host>) -> Self::Output {
                exponents.map(|&exponent| <$t as Power>::power(self, exponent))
            }
        }

        impl Power<&Matrix<$t, Host>> for $t {
            type Output = Matrix<$t, Host>;

            fn power(self, exponents: &Matrix<$t, Host>) -> Self::Output {
                exponents.map(|&exponent| <$t as Power>::power(self, exponent))
            }
        }
    )+};
}

scalar_base_power!(f32, f64, f16, bf16);

// ---- resident tensors: the GPU power kernels ----

#[cfg(all(feature = "metal", target_os = "macos"))]
mod resident {
    use super::{Matrix, Metal, Power, Vector, assert_same_len, assert_same_shape};
    use crate::metal::MetalElement;
    use crate::tensors::Host;

    impl<T: MetalElement> Vector<T, Metal> {
        /// Raise every element to `exponent`, on the GPU.
        pub fn pow(&self, exponent: T) -> Self {
            self.power_scalar(exponent, false)
        }

        /// Raise every element to the matching element of `exponents`.
        ///
        /// # Panics
        ///
        /// Unless the two vectors have the same length.
        #[track_caller]
        pub fn pow_elementwise(&self, exponents: &Self) -> Self {
            assert_same_len(self.len(), exponents.len(), "pow");
            match self.storage().power(exponents.storage()) {
                Some(data) => Vector::from_storage(self.len(), data),
                None => self
                    .to_backend::<Host>()
                    .pow_elementwise(&exponents.to_backend::<Host>())
                    .to_backend(),
            }
        }

        /// Elementwise power with one operand fixed; `scalar_left` selects
        /// `scalar^x` over `x^scalar`.
        pub(crate) fn power_scalar(&self, scalar: T, scalar_left: bool) -> Self {
            match self.storage().power_scalar(scalar, scalar_left) {
                Some(data) => Vector::from_storage(self.len(), data),
                None => {
                    let host = self.to_backend::<Host>();
                    if scalar_left {
                        host.map(|&exponent| scalar.power(exponent)).to_backend()
                    } else {
                        host.pow(scalar).to_backend()
                    }
                }
            }
        }
    }

    impl<T: MetalElement> Matrix<T, Metal> {
        /// Raise every element to `exponent`, on the GPU.
        pub fn pow(&self, exponent: T) -> Self {
            self.power_scalar(exponent, false)
        }

        /// Raise every element to the matching element of `exponents`.
        ///
        /// # Panics
        ///
        /// Unless the two matrices have the same shape.
        #[track_caller]
        pub fn pow_elementwise(&self, exponents: &Self) -> Self {
            assert_same_shape(self.shape(), exponents.shape(), "pow");
            let (rows, cols) = self.shape();
            match self.storage().power(exponents.storage()) {
                Some(data) => Matrix::from_storage(rows, cols, data),
                None => self
                    .to_backend::<Host>()
                    .pow_elementwise(&exponents.to_backend::<Host>())
                    .to_backend(),
            }
        }

        pub(crate) fn power_scalar(&self, scalar: T, scalar_left: bool) -> Self {
            let (rows, cols) = self.shape();
            match self.storage().power_scalar(scalar, scalar_left) {
                Some(data) => Matrix::from_storage(rows, cols, data),
                None => {
                    let host = self.to_backend::<Host>();
                    if scalar_left {
                        host.map(|&exponent| scalar.power(exponent)).to_backend()
                    } else {
                        host.pow(scalar).to_backend()
                    }
                }
            }
        }
    }

    impl<T: MetalElement> Power<T> for &Vector<T, Metal> {
        type Output = Vector<T, Metal>;

        fn power(self, exponent: T) -> Self::Output {
            self.pow(exponent)
        }
    }

    impl<T: MetalElement> Power<T> for &Matrix<T, Metal> {
        type Output = Matrix<T, Metal>;

        fn power(self, exponent: T) -> Self::Output {
            self.pow(exponent)
        }
    }

    impl<T: MetalElement> Power<&Vector<T, Metal>> for &Vector<T, Metal> {
        type Output = Vector<T, Metal>;

        #[track_caller]
        fn power(self, exponents: &Vector<T, Metal>) -> Self::Output {
            self.pow_elementwise(exponents)
        }
    }

    impl<T: MetalElement> Power<&Matrix<T, Metal>> for &Matrix<T, Metal> {
        type Output = Matrix<T, Metal>;

        #[track_caller]
        fn power(self, exponents: &Matrix<T, Metal>) -> Self::Output {
            self.pow_elementwise(exponents)
        }
    }

    impl<T: MetalElement> Power<&Vector<T, Metal>> for T {
        type Output = Vector<T, Metal>;

        fn power(self, exponents: &Vector<T, Metal>) -> Self::Output {
            exponents.power_scalar(self, true)
        }
    }

    impl<T: MetalElement> Power<&Matrix<T, Metal>> for T {
        type Output = Matrix<T, Metal>;

        fn power(self, exponents: &Matrix<T, Metal>) -> Self::Output {
            exponents.power_scalar(self, true)
        }
    }
}

impl<T: Real, B: Kernels<T>> Transcendental for Vector<T, B> {
    type Elem = T;

    fn analytic(&self, f: Analytic) -> Self {
        B::vector_unary(self, f)
    }

    fn pow(&self, exponent: T) -> Self {
        B::vector_power_scalar(self, exponent, false)
    }

    fn pow_elementwise(&self, exponents: &Self) -> Self {
        B::vector_power(self, exponents)
    }
}

impl<T: Real, B: Kernels<T>> Transcendental for Matrix<T, B> {
    type Elem = T;

    fn analytic(&self, f: Analytic) -> Self {
        B::matrix_unary(self, f)
    }

    fn pow(&self, exponent: T) -> Self {
        B::matrix_power_scalar(self, exponent, false)
    }

    fn pow_elementwise(&self, exponents: &Self) -> Self {
        B::matrix_power(self, exponents)
    }
}
