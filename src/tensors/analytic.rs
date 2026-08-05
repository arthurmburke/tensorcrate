//! The analytic functions, applied elementwise to tensors.
//!
//! `sin`, `exp`, `ln` and the rest are defined on single numbers by the traits
//! in [`crate::numbers`]. A tensor applies one of them to every element and
//! keeps its shape, so `exp` of a `3×4` matrix is a `3×4` matrix. This module is
//! where that lifting happens, in three overlapping spellings:
//!
//! * **Inherent methods on host tensors** — [`Vector<T, Host>`] and
//!   [`Matrix<T, Host>`] get `sin`, `exp`, … for every element type that
//!   implements the matching scalar trait. `f64` and `f32` stay themselves and
//!   integers widen to `f64`, exactly as they do on their own; the
//!   [`Complex`](crate::numbers::Complex) and [`Dual`](crate::numbers::Dual)
//!   definitions carry over too, so the `ε` part of a dual tensor comes back
//!   holding the derivative, elementwise.
//!
//! * **Inherent methods on resident tensors** — the same names on
//!   `Vector<f32, Metal>` and `Matrix<f32, Metal>`, each running the GPU unary
//!   kernel and leaving the result in GPU-shared memory.
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
//! Reciprocal is the one unary op in [`crate::numbers`] with no place here:
//! there is no `recip` shader, and `1.0 / v` already divides elementwise.
//!
//! [`Vector<T, Host>`]: Vector
//! [`Matrix<T, Host>`]: Matrix

use crate::numbers::{
    Arccos, Arcsin, Arctan, Cos, Cosh, Csc, Exp, Ln, Sec, Sin, Sinh, Sqrt, Tan, Tanh,
};
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::tensors::Metal;
use crate::tensors::kernels::{Analytic, Kernels};
use crate::tensors::{Host, Matrix, Vector};

impl Vector<f32, Host> {
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
        self.map(|&x| f.value(x))
    }
}

impl Matrix<f32, Host> {
    /// Apply an analytic function to every element; see [`Vector::analytic`].
    pub fn analytic(&self, f: Analytic) -> Self {
        self.map(|&x| f.value(x))
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
        /// use tensorcrate::tensors::{Kernels, Transcendental, Vector};
        ///
        /// fn squash<B: Kernels>(v: &Vector<f32, B>) -> Vector<f32, B> {
        ///     v.tanh()
        /// }
        ///
        /// assert_eq!(squash(&Vector::new([0.0f32])).data(), [0.0]);
        /// ```
        ///
        /// On a concrete backend the inherent method wins method resolution and
        /// this trait is never consulted; both run the same kernel, so which one
        /// resolved is not observable. The inherent host methods also cover
        /// element types this trait cannot — `Vector<f64, Host>::exp` is real,
        /// while the 32-bit shaders mean the backend-generic surface is `f32`
        /// only.
        ///
        /// Every named method is [`analytic`](Transcendental::analytic) with the
        /// function fixed, so implementing that one implements them all.
        ///
        /// [`Vector<f32, Host>`]: Vector
        /// [`Vector<f32, Metal>`]: Vector
        pub trait Transcendental: Sized {
            /// Apply `f` to every element.
            fn analytic(&self, f: Analytic) -> Self;

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
            impl Vector<f32, Metal> {
                #[doc = concat!("Elementwise `", stringify!($method), "`, on the GPU.")]
                pub fn $method(&self) -> Self {
                    self.analytic(Analytic::$Trait)
                }
            }

            #[cfg(all(feature = "metal", target_os = "macos"))]
            impl Matrix<f32, Metal> {
                #[doc = concat!("Elementwise `", stringify!($method), "`, on the GPU.")]
                pub fn $method(&self) -> Self {
                    self.analytic(Analytic::$Trait)
                }
            }

            #[cfg(all(feature = "metal", target_os = "macos"))]
            impl $Trait for &Vector<f32, Metal> {
                type Output = Vector<f32, Metal>;

                fn $method(self) -> Self::Output {
                    self.analytic(Analytic::$Trait)
                }
            }

            #[cfg(all(feature = "metal", target_os = "macos"))]
            impl $Trait for &Matrix<f32, Metal> {
                type Output = Matrix<f32, Metal>;

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

impl<B: Kernels> Transcendental for Vector<f32, B> {
    fn analytic(&self, f: Analytic) -> Self {
        B::vector_unary(self, f)
    }
}

impl<B: Kernels> Transcendental for Matrix<f32, B> {
    fn analytic(&self, f: Analytic) -> Self {
        B::matrix_unary(self, f)
    }
}
