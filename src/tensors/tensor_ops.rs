//! Elementwise arithmetic, analytic functions and comparisons on [`Tensor`]s
//! and [`TensorView`]s, and their use as fused-program inputs.
//!
//! An operation of one tensor does not care about its shape, so it runs as the
//! [`Kernels`] vector operation over the flat row-major storage — the same
//! kernel, SIMD or GPU, a [`Vector`] of that many elements would run. A view
//! that is not its tensor's whole storage in order is first copied into order
//! with one strided copy.
//!
//! An operation of two — `+ - * /`, [`elementwise`](Tensor::elementwise),
//! [`power`](Tensor::power), [`compare`](Tensor::compare),
//! [`min`](Tensor::min) and [`max`](Tensor::max) — broadcasts its operands
//! numpy-style: the shapes align at their last axes, a missing leading axis
//! counts as `1`, and an axis of `1` repeats to match the other operand's.
//! `[D]` against `[B, T, D]` adds a bias to every position, `[T, T]` against
//! `[B, H, T, T]` masks every head, and a rank-0 tensor is a scalar against
//! anything. Two whole tensors of one shape run the vector kernel; any other
//! pair runs one strided kernel that reads both operands in place, a repeated
//! axis through a stride of zero, so neither a broadcast operand nor a
//! non-contiguous view is copied first. Shapes that do not broadcast panic,
//! naming the operation and both shapes. The result is always a new,
//! contiguous tensor.
//!
//! The operations are defined for every element type and backend that
//! implements [`Kernels`]: every float on `Host`, and `f32`, `f16` and `bf16`
//! on `Metal`, where operands and result stay in GPU memory.

use std::ops::{Add, Deref, Div, Mul, Neg, Sub};

use super::fused::{Element, Fusable, FusableMut, FusableOf, Sink, Source, View};
use super::kernels::Pairwise;
use super::layout::broadcast_shape;
use super::{Analytic, Backend, BinaryOp, Compare, Kernels, Tensor, TensorView, Vector};
use crate::numbers::Real;

/// A view's elements as a flat vector in row-major order: its tensor's own
/// storage when the view is all of it, in order, and a copy otherwise.
enum Flat<'a, T, B: Backend> {
    Borrowed(&'a Vector<T, B>),
    Copied(Vector<T, B>),
}

impl<T, B: Backend> Deref for Flat<'_, T, B> {
    type Target = Vector<T, B>;

    fn deref(&self) -> &Vector<T, B> {
        match self {
            Flat::Borrowed(vector) => vector,
            Flat::Copied(vector) => vector,
        }
    }
}

impl<'a, T: Real, B: Kernels<T>> TensorView<'a, T, B> {
    fn flat(&self) -> Flat<'a, T, B> {
        if self.is_whole() {
            Flat::Borrowed(self.data())
        } else {
            Flat::Copied(self.contiguous().into_vector())
        }
    }

    fn map(&self, f: impl FnOnce(&Vector<T, B>) -> Vector<T, B>) -> Tensor<T, B> {
        Tensor::from_parts(*self.dims(), f(&self.flat()))
    }

    /// `op(self, rhs)` with the shapes broadcast: the vector kernel `same`
    /// for two whole tensors of one shape, and otherwise one strided kernel
    /// over the views broadcast to their common shape.
    #[track_caller]
    fn zip<'b>(
        &self,
        rhs: impl Into<TensorView<'b, T, B>>,
        op: Pairwise,
        same: impl FnOnce(&Vector<T, B>, &Vector<T, B>) -> Vector<T, B>,
    ) -> Tensor<T, B> {
        let rhs = rhs.into();
        if self.shape() == rhs.shape() && self.is_whole() && rhs.is_whole() {
            return Tensor::from_parts(*self.dims(), same(self.data(), rhs.data()));
        }
        let shape = broadcast_shape(self.shape(), rhs.shape()).unwrap_or_else(|| {
            panic!(
                "{}: tensor shapes differ, {:?} and {:?}, and do not broadcast",
                op.name(),
                self.shape(),
                rhs.shape()
            )
        });
        let (a, b) = (self.broadcast_to(&shape), rhs.broadcast_to(&shape));
        Tensor::from_parts(shape, B::strided_binary(a, b, op))
    }

    /// `self op rhs`, elementwise with the shapes broadcast: the operation
    /// behind `+ - * /`, and `%` besides.
    ///
    /// # Panics
    ///
    /// If the shapes do not broadcast.
    #[track_caller]
    pub fn elementwise<'b>(
        &self,
        rhs: impl Into<TensorView<'b, T, B>>,
        op: BinaryOp,
    ) -> Tensor<T, B> {
        self.zip(rhs, Pairwise::Arithmetic(op), |a, b| {
            B::vector_elementwise(a, b, op)
        })
    }

    /// Every element combined with `scalar` by `op`: `x op scalar`, or
    /// `scalar op x` if `scalar_left`.
    pub fn with_scalar(&self, scalar: T, op: BinaryOp, scalar_left: bool) -> Tensor<T, B> {
        self.map(|a| B::vector_broadcast(a, scalar, op, scalar_left))
    }

    /// Every element multiplied by `scalar`.
    pub fn scale(&self, scalar: T) -> Tensor<T, B> {
        self.with_scalar(scalar, BinaryOp::Mul, false)
    }

    /// `f(x)` for every element.
    pub fn unary(&self, f: Analytic) -> Tensor<T, B> {
        self.map(|a| B::vector_unary(a, f))
    }

    /// Elementwise `self^rhs`, with the shapes broadcast.
    ///
    /// # Panics
    ///
    /// If the shapes do not broadcast.
    #[track_caller]
    pub fn power<'b>(&self, rhs: impl Into<TensorView<'b, T, B>>) -> Tensor<T, B> {
        self.zip(rhs, Pairwise::Power, |a, b| B::vector_power(a, b))
    }

    /// `x^scalar` for every element, or `scalar^x` if `scalar_left`.
    pub fn power_scalar(&self, scalar: T, scalar_left: bool) -> Tensor<T, B> {
        self.map(|a| B::vector_power_scalar(a, scalar, scalar_left))
    }

    /// Elementwise comparison, with the shapes broadcast: the [`Compare`]
    /// operation of each pair.
    ///
    /// # Panics
    ///
    /// If the shapes do not broadcast.
    #[track_caller]
    pub fn compare<'b>(&self, rhs: impl Into<TensorView<'b, T, B>>, op: Compare) -> Tensor<T, B> {
        self.zip(rhs, Pairwise::Compare(op), |a, b| {
            B::vector_compare(a, b, op)
        })
    }

    /// Every element compared with `scalar`, which `scalar_left` puts on the
    /// left.
    pub fn compare_scalar(&self, scalar: T, op: Compare, scalar_left: bool) -> Tensor<T, B> {
        self.map(|a| B::vector_compare_scalar(a, scalar, op, scalar_left))
    }

    /// Elementwise minimum, with the shapes broadcast.
    ///
    /// # Panics
    ///
    /// If the shapes do not broadcast.
    #[track_caller]
    pub fn min<'b>(&self, rhs: impl Into<TensorView<'b, T, B>>) -> Tensor<T, B> {
        self.compare(rhs, Compare::Min)
    }

    /// Elementwise maximum, with the shapes broadcast.
    ///
    /// # Panics
    ///
    /// If the shapes do not broadcast.
    #[track_caller]
    pub fn max<'b>(&self, rhs: impl Into<TensorView<'b, T, B>>) -> Tensor<T, B> {
        self.compare(rhs, Compare::Max)
    }

    /// The lesser of each element and `scalar`.
    pub fn min_scalar(&self, scalar: T) -> Tensor<T, B> {
        self.compare_scalar(scalar, Compare::Min, false)
    }

    /// The greater of each element and `scalar` — `max_scalar(0)` is a relu.
    pub fn max_scalar(&self, scalar: T) -> Tensor<T, B> {
        self.compare_scalar(scalar, Compare::Max, false)
    }

    /// Every element confined to `[low, high]`.
    ///
    /// # Panics
    ///
    /// If `low > high`.
    #[track_caller]
    pub fn clamp(&self, low: T, high: T) -> Tensor<T, B> {
        self.map(|a| B::vector_clamp(a, low, high))
    }
}

/// The same operations on a whole tensor; see [`TensorView`] for each.
impl<T: Real, B: Kernels<T>> Tensor<T, B> {
    /// `self op rhs`, elementwise. See [`TensorView::elementwise`].
    #[track_caller]
    pub fn elementwise<'b>(&self, rhs: impl Into<TensorView<'b, T, B>>, op: BinaryOp) -> Self {
        self.view().elementwise(rhs, op)
    }

    /// Every element combined with `scalar`. See [`TensorView::with_scalar`].
    pub fn with_scalar(&self, scalar: T, op: BinaryOp, scalar_left: bool) -> Self {
        self.view().with_scalar(scalar, op, scalar_left)
    }

    /// Every element multiplied by `scalar`.
    pub fn scale(&self, scalar: T) -> Self {
        self.view().scale(scalar)
    }

    /// `f(x)` for every element.
    pub fn unary(&self, f: Analytic) -> Self {
        self.view().unary(f)
    }

    /// Elementwise `self^rhs`.
    #[track_caller]
    pub fn power<'b>(&self, rhs: impl Into<TensorView<'b, T, B>>) -> Self {
        self.view().power(rhs)
    }

    /// `x^scalar`, or `scalar^x` if `scalar_left`.
    pub fn power_scalar(&self, scalar: T, scalar_left: bool) -> Self {
        self.view().power_scalar(scalar, scalar_left)
    }

    /// Elementwise comparison.
    #[track_caller]
    pub fn compare<'b>(&self, rhs: impl Into<TensorView<'b, T, B>>, op: Compare) -> Self {
        self.view().compare(rhs, op)
    }

    /// Every element compared with `scalar`.
    pub fn compare_scalar(&self, scalar: T, op: Compare, scalar_left: bool) -> Self {
        self.view().compare_scalar(scalar, op, scalar_left)
    }

    /// Elementwise minimum.
    #[track_caller]
    pub fn min<'b>(&self, rhs: impl Into<TensorView<'b, T, B>>) -> Self {
        self.view().min(rhs)
    }

    /// Elementwise maximum.
    #[track_caller]
    pub fn max<'b>(&self, rhs: impl Into<TensorView<'b, T, B>>) -> Self {
        self.view().max(rhs)
    }

    /// The lesser of each element and `scalar`.
    pub fn min_scalar(&self, scalar: T) -> Self {
        self.view().min_scalar(scalar)
    }

    /// The greater of each element and `scalar`.
    pub fn max_scalar(&self, scalar: T) -> Self {
        self.view().max_scalar(scalar)
    }

    /// Every element confined to `[low, high]`.
    #[track_caller]
    pub fn clamp(&self, low: T, high: T) -> Self {
        self.view().clamp(low, high)
    }
}

/// One operator over every pairing of an owned tensor, a borrowed one and a
/// view.
macro_rules! operator {
    ($Trait:ident, $method:ident, $op:expr) => {
        impl<T: Real, B: Kernels<T>> $Trait for Tensor<T, B> {
            type Output = Tensor<T, B>;
            #[track_caller]
            fn $method(self, rhs: Self) -> Tensor<T, B> {
                self.view().elementwise(&rhs, $op)
            }
        }

        impl<T: Real, B: Kernels<T>> $Trait<&Tensor<T, B>> for &Tensor<T, B> {
            type Output = Tensor<T, B>;
            #[track_caller]
            fn $method(self, rhs: &Tensor<T, B>) -> Tensor<T, B> {
                self.view().elementwise(rhs, $op)
            }
        }

        impl<'b, T: Real, B: Kernels<T>> $Trait<TensorView<'b, T, B>> for &Tensor<T, B> {
            type Output = Tensor<T, B>;
            #[track_caller]
            fn $method(self, rhs: TensorView<'b, T, B>) -> Tensor<T, B> {
                self.view().elementwise(rhs, $op)
            }
        }

        impl<'a, T: Real, B: Kernels<T>> $Trait<&Tensor<T, B>> for TensorView<'a, T, B> {
            type Output = Tensor<T, B>;
            #[track_caller]
            fn $method(self, rhs: &Tensor<T, B>) -> Tensor<T, B> {
                self.elementwise(rhs, $op)
            }
        }

        impl<'a, 'b, T: Real, B: Kernels<T>> $Trait<TensorView<'b, T, B>> for TensorView<'a, T, B> {
            type Output = Tensor<T, B>;
            #[track_caller]
            fn $method(self, rhs: TensorView<'b, T, B>) -> Tensor<T, B> {
                self.elementwise(rhs, $op)
            }
        }
    };
}

operator!(Add, add, BinaryOp::Add);
operator!(Sub, sub, BinaryOp::Sub);
operator!(Mul, mul, BinaryOp::Mul);
operator!(Div, div, BinaryOp::Div);

/// Negation is multiplication by `−1`, on every backend.
impl<T: Real, B: Kernels<T>> Neg for &Tensor<T, B> {
    type Output = Tensor<T, B>;
    fn neg(self) -> Tensor<T, B> {
        self.scale(-T::one())
    }
}

impl<T: Real, B: Kernels<T>> Neg for Tensor<T, B> {
    type Output = Tensor<T, B>;
    fn neg(self) -> Tensor<T, B> {
        self.scale(-T::one())
    }
}

impl<T: Real, B: Kernels<T>> Neg for TensorView<'_, T, B> {
    type Output = Tensor<T, B>;
    fn neg(self) -> Tensor<T, B> {
        self.scale(-T::one())
    }
}

// ---- fused programs ---------------------------------------------------------------

/// The `(rows, cols)` a fused program reads a tensor of `shape` as: every axis
/// but the last folded into the rows. `None` below two axes, where it is read
/// as a vector.
fn matrix_shape(shape: &[usize]) -> Option<(usize, usize)> {
    match shape {
        [] | [_] => None,
        [leading @ .., last] => Some((leading.iter().product(), *last)),
    }
}

/// A tensor is read by a fused program as the matrix of its rows — every axis
/// but the last folded into one — so a program over a `[B, T, D]` activation
/// runs over a `(B·T) × D` space. A tensor of one axis is a vector.
impl<T: Element, B: Backend> Fusable<B> for Tensor<T, B> {
    fn source(&self) -> Source<'_, B> {
        Source {
            data: T::source::<B>(self.as_vector().storage()),
            len: self.len(),
            view: None,
        }
    }

    fn shape(&self) -> Option<(usize, usize)> {
        matrix_shape(Tensor::shape(self))
    }
}

impl<T: Element, B: Backend> FusableOf<T, B> for Tensor<T, B> {}

/// A tensor owns its elements in row-major order, so a program can update it
/// in place as it does a [`Vector`] of the same length.
impl<T: Element, B: Backend> FusableMut<B> for Tensor<T, B> {
    fn sink(&mut self) -> Sink<'_, B> {
        let len = self.len();
        Sink {
            data: T::sink::<B>(self.vector_mut().storage_mut()),
            len,
        }
    }
}

/// A view is read by a fused program in place, through its strides, whatever
/// they are: a slice, a permutation or a broadcast of a tensor. A view of all
/// of a tensor is read as the tensor.
impl<T: Element, B: Backend> Fusable<B> for TensorView<'_, T, B> {
    fn source(&self) -> Source<'_, B> {
        let data = T::source::<B>(self.data().storage());
        let view =
            (!self.is_whole()).then(|| View::new(self.offset(), self.shape(), self.strides()));
        Source {
            data,
            len: self.len(),
            view,
        }
    }

    fn shape(&self) -> Option<(usize, usize)> {
        if self.is_whole() {
            return matrix_shape(TensorView::shape(self));
        }
        let (rows, cols) = matrix_shape(TensorView::shape(self)).unwrap_or((1, self.len()));
        Some((rows, cols))
    }
}

impl<T: Element, B: Backend> FusableOf<T, B> for TensorView<'_, T, B> {}
