//! Reductions of [`Tensor`]s and [`TensorView`]s along axes: sums, extremes,
//! means and variances over any set of axes, and the position of the extreme
//! along one.
//!
//! Each reduction names its axes with an [`Axes`] value — one axis, an array
//! or slice of them, or `..` for every axis — where a negative axis counts
//! from the last. The result has the remaining axes in order, or, with
//! `keep_dims`, every axis with the reduced ones at extent `1`, so that it
//! broadcasts back against the input.
//!
//! A reduction runs as one [`Kernels`] kernel that reads the input through
//! its strides — a permuted or sliced view is reduced where it lies — on the
//! host or, on `Metal`, the GPU. Each slice folds in `f32` (`f64` for `f64`
//! tensors) and rounds to the element type once, so an `f16` sum of ten
//! thousand ones is `10000`. The maximum and minimum pass over NaNs, as
//! [`Reduce`] does everywhere: a number beats a NaN, so a slice of only NaNs
//! folds to the identity. Sums propagate a NaN.
//!
//! ```
//! use tensorcrate::statistics::Correction;
//! use tensorcrate::tensors::Tensor;
//!
//! let x = Tensor::from_vec(&[2, 3], vec![1.0f32, 5.0, 3.0, 4.0, 2.0, 6.0]);
//! assert_eq!(x.sum_axes(-1, false).to_vec(), [9.0, 12.0]);
//! assert_eq!(x.max_axes(0, true).shape(), [1, 3]);
//! assert_eq!(x.mean_axes(.., false).to_vec(), [3.5]);
//! assert_eq!(x.var_axes(1, Correction::Sample, false).to_vec(), [4.0, 4.0]);
//! assert_eq!(x.argmax(1, false).to_vec(), [1u32, 2]);
//! ```

use std::fmt;
use std::ops::RangeFull;

use super::kernels::AxisReduction;
use super::layout::{AxisIndex, Dims, resolve_axis};
use super::{Kernels, Reduce, Tensor, TensorView};
use crate::numbers::Real;
use crate::statistics::Correction;

/// The axes a reduction folds: one axis (a `usize`, or a negative `isize` or
/// `i32` counting from the last), an array, slice or `Vec` of them, or `..`
/// for every axis. An empty collection folds none, leaving every element its
/// own slice.
pub trait Axes: fmt::Debug {
    /// The distinct axes this names among `rank` axes, in ascending order, or
    /// `None` if it names one that does not exist or one twice.
    fn resolve_axes(&self, rank: usize) -> Option<Vec<usize>>;
}

macro_rules! single_axis {
    ($($ty:ty),+) => {$(
        impl Axes for $ty {
            fn resolve_axes(&self, rank: usize) -> Option<Vec<usize>> {
                Some(vec![self.resolve(rank)?])
            }
        }
    )+};
}

single_axis!(usize, isize, i32);

impl<A: AxisIndex> Axes for [A] {
    fn resolve_axes(&self, rank: usize) -> Option<Vec<usize>> {
        let mut axes = self
            .iter()
            .map(|axis| axis.resolve(rank))
            .collect::<Option<Vec<_>>>()?;
        axes.sort_unstable();
        let distinct = axes.windows(2).all(|pair| pair[0] != pair[1]);
        distinct.then_some(axes)
    }
}

impl<A: AxisIndex, const N: usize> Axes for [A; N] {
    fn resolve_axes(&self, rank: usize) -> Option<Vec<usize>> {
        self.as_slice().resolve_axes(rank)
    }
}

impl<A: AxisIndex> Axes for Vec<A> {
    fn resolve_axes(&self, rank: usize) -> Option<Vec<usize>> {
        self.as_slice().resolve_axes(rank)
    }
}

impl<X: Axes + ?Sized> Axes for &X {
    fn resolve_axes(&self, rank: usize) -> Option<Vec<usize>> {
        (**self).resolve_axes(rank)
    }
}

impl Axes for RangeFull {
    fn resolve_axes(&self, rank: usize) -> Option<Vec<usize>> {
        Some((0..rank).collect())
    }
}

impl<T: Real, B: Kernels<T>> TensorView<'_, T, B> {
    /// `axes` resolved against this view's shape.
    #[track_caller]
    fn axes(&self, axes: impl Axes, operation: &str) -> Vec<usize> {
        axes.resolve_axes(self.rank()).unwrap_or_else(|| {
            panic!(
                "{operation}: axes {axes:?} are not distinct axes of shape {:?}",
                self.shape()
            )
        })
    }

    /// The shape of a result that folds `axes`: without them, or with them
    /// at extent `1` if `keep_dims`.
    fn reduced_shape(&self, axes: &[usize], keep_dims: bool) -> Dims {
        let shape = (0..self.rank())
            .filter_map(|axis| match axes.contains(&axis) {
                false => Some(self.shape()[axis]),
                true => keep_dims.then_some(1),
            })
            .collect::<Vec<_>>();
        Dims::new(&shape, "reduce")
    }

    /// Fold `axes` by `op`, or fill the result with `empty` — the answer for
    /// a slice of no elements, or for a correction with too few — when `op`
    /// is `None` or the slices are empty.
    #[track_caller]
    fn reduction(
        &self,
        axes: impl Axes,
        keep_dims: bool,
        operation: &str,
        op: impl FnOnce(usize) -> Option<AxisReduction>,
        empty: T,
    ) -> Tensor<T, B> {
        let axes = self.axes(axes, operation);
        let shape = self.reduced_shape(&axes, keep_dims);
        let depth = self.split_for(&axes).depth();
        match op(depth) {
            Some(op) if depth != 0 && !shape.contains(&0) => {
                Tensor::from_parts(shape, B::reduce_axes(*self, &axes, op))
            }
            _ => Tensor::filled(&shape, empty),
        }
    }

    /// Each slice along `axes` folded by `op`; an empty slice folds to
    /// [`op.identity()`](Reduce::identity). The axes are an [`Axes`]; the
    /// result has the other axes in order, and the folded ones too, at extent
    /// `1`, if `keep_dims`. Each slice folds in `f32` (`f64` for an `f64`
    /// tensor) and rounds once, reading the view in place on its backend.
    ///
    /// # Panics
    ///
    /// If `axes` names an axis that does not exist, or one twice.
    #[track_caller]
    pub fn reduce_axes(&self, op: Reduce, axes: impl Axes, keep_dims: bool) -> Tensor<T, B> {
        let name = match op {
            Reduce::Sum => "sum_axes",
            Reduce::Min => "min_axes",
            Reduce::Max => "max_axes",
        };
        self.reduction(
            axes,
            keep_dims,
            name,
            |_| Some(AxisReduction::Fold(op)),
            op.identity(),
        )
    }

    /// The sum of each slice along `axes`: `sum_axes(-1, false)` sums each
    /// row. An empty slice sums to zero.
    ///
    /// # Panics
    ///
    /// As for [`reduce_axes`](Self::reduce_axes).
    #[track_caller]
    pub fn sum_axes(&self, axes: impl Axes, keep_dims: bool) -> Tensor<T, B> {
        self.reduce_axes(Reduce::Sum, axes, keep_dims)
    }

    /// The largest element of each slice along `axes`. NaNs are passed over,
    /// as by [`Reduce::Max`]; a slice of only NaNs, or of none, gives `−∞`.
    ///
    /// # Panics
    ///
    /// As for [`reduce_axes`](Self::reduce_axes).
    #[track_caller]
    pub fn max_axes(&self, axes: impl Axes, keep_dims: bool) -> Tensor<T, B> {
        self.reduce_axes(Reduce::Max, axes, keep_dims)
    }

    /// The smallest element of each slice along `axes`. NaNs are passed over,
    /// as by [`Reduce::Min`]; a slice of only NaNs, or of none, gives `+∞`.
    ///
    /// # Panics
    ///
    /// As for [`reduce_axes`](Self::reduce_axes).
    #[track_caller]
    pub fn min_axes(&self, axes: impl Axes, keep_dims: bool) -> Tensor<T, B> {
        self.reduce_axes(Reduce::Min, axes, keep_dims)
    }

    /// The mean of each slice along `axes`: its sum, accumulated in `f32`
    /// (`f64` for `f64`), divided by its length. An empty slice has a
    /// NaN mean.
    ///
    /// # Panics
    ///
    /// As for [`reduce_axes`](Self::reduce_axes).
    #[track_caller]
    pub fn mean_axes(&self, axes: impl Axes, keep_dims: bool) -> Tensor<T, B> {
        self.reduction(
            axes,
            keep_dims,
            "mean_axes",
            |_| Some(AxisReduction::Mean),
            T::nan(),
        )
    }

    /// The variance of each slice along `axes`, divided as `correction` says:
    /// [`Correction::Population`] by the slice's length `n`, the variance of
    /// the values themselves, or [`Correction::Sample`] by `n − 1` (one delta
    /// degree of freedom), the unbiased estimate. It is computed in two
    /// passes — the mean, then the squared deviations from it — both
    /// accumulated in `f32` (`f64` for `f64`). A slice too short for the
    /// correction — empty, or a single element under `Sample` — gives NaN.
    ///
    /// # Panics
    ///
    /// As for [`reduce_axes`](Self::reduce_axes).
    #[track_caller]
    pub fn var_axes(
        &self,
        axes: impl Axes,
        correction: Correction,
        keep_dims: bool,
    ) -> Tensor<T, B> {
        self.reduction(
            axes,
            keep_dims,
            "var_axes",
            |depth| {
                let divisor = correction.divisor(depth)?;
                Some(AxisReduction::Variance { divisor })
            },
            T::nan(),
        )
    }

    /// The position along `axis` of each slice's largest element, as a `u32`
    /// index tensor: the axis removed, or kept at extent `1` if `keep_dims`.
    ///
    /// A tie goes to the first position. NaNs are passed over, as
    /// [`max_axes`](Self::max_axes) passes over them, so the answer is the
    /// position of the largest number; a slice of only NaNs answers `0`.
    ///
    /// ```
    /// use tensorcrate::tensors::Tensor;
    ///
    /// let logits = Tensor::from_vec(&[2, 3], vec![0.5f32, 2.0, 2.0, f32::NAN, -1.0, -3.0]);
    /// assert_eq!(logits.argmax(-1, false).to_vec(), [1u32, 1]);
    /// assert_eq!(logits.argmin(-1, true).shape(), [2, 1]);
    /// ```
    ///
    /// # Panics
    ///
    /// If `axis` does not exist, is empty — no element is largest — or is
    /// longer than a `u32` can index.
    #[track_caller]
    pub fn argmax(&self, axis: impl AxisIndex, keep_dims: bool) -> Tensor<u32, B> {
        self.arg_reduction(axis, keep_dims, Reduce::Max, "argmax")
    }

    /// The position along `axis` of each slice's smallest element; the
    /// counterpart of [`argmax`](Self::argmax), with the same rules for ties
    /// and NaNs.
    ///
    /// # Panics
    ///
    /// As for [`argmax`](Self::argmax).
    #[track_caller]
    pub fn argmin(&self, axis: impl AxisIndex, keep_dims: bool) -> Tensor<u32, B> {
        self.arg_reduction(axis, keep_dims, Reduce::Min, "argmin")
    }

    #[track_caller]
    fn arg_reduction(
        &self,
        axis: impl AxisIndex,
        keep_dims: bool,
        op: Reduce,
        operation: &str,
    ) -> Tensor<u32, B> {
        let axis = resolve_axis(axis, self.shape(), operation);
        let extent = self.shape()[axis];
        assert!(
            extent != 0,
            "{operation}: axis {axis} of shape {:?} is empty",
            self.shape()
        );
        assert!(
            u32::try_from(extent).is_ok(),
            "{operation}: axis {axis} of shape {:?} is longer than a u32 index reaches",
            self.shape()
        );
        let shape = self.reduced_shape(&[axis], keep_dims);
        if shape.contains(&0) {
            return Tensor::filled(&shape, 0);
        }
        Tensor::from_parts(shape, B::arg_reduce(*self, axis, op))
    }
}

/// The same reductions of a whole tensor; see [`TensorView`] for each.
impl<T: Real, B: Kernels<T>> Tensor<T, B> {
    /// Each slice along `axes` folded by `op`. See
    /// [`TensorView::reduce_axes`].
    #[track_caller]
    pub fn reduce_axes(&self, op: Reduce, axes: impl Axes, keep_dims: bool) -> Self {
        self.view().reduce_axes(op, axes, keep_dims)
    }

    /// The sum of each slice along `axes`.
    #[track_caller]
    pub fn sum_axes(&self, axes: impl Axes, keep_dims: bool) -> Self {
        self.view().sum_axes(axes, keep_dims)
    }

    /// The largest element of each slice along `axes`, passing over NaNs.
    #[track_caller]
    pub fn max_axes(&self, axes: impl Axes, keep_dims: bool) -> Self {
        self.view().max_axes(axes, keep_dims)
    }

    /// The smallest element of each slice along `axes`, passing over NaNs.
    #[track_caller]
    pub fn min_axes(&self, axes: impl Axes, keep_dims: bool) -> Self {
        self.view().min_axes(axes, keep_dims)
    }

    /// The mean of each slice along `axes`.
    #[track_caller]
    pub fn mean_axes(&self, axes: impl Axes, keep_dims: bool) -> Self {
        self.view().mean_axes(axes, keep_dims)
    }

    /// The variance of each slice along `axes` under `correction`. See
    /// [`TensorView::var_axes`].
    #[track_caller]
    pub fn var_axes(&self, axes: impl Axes, correction: Correction, keep_dims: bool) -> Self {
        self.view().var_axes(axes, correction, keep_dims)
    }

    /// The position along `axis` of each slice's largest element. See
    /// [`TensorView::argmax`].
    #[track_caller]
    pub fn argmax(&self, axis: impl AxisIndex, keep_dims: bool) -> Tensor<u32, B> {
        self.view().argmax(axis, keep_dims)
    }

    /// The position along `axis` of each slice's smallest element. See
    /// [`TensorView::argmin`].
    #[track_caller]
    pub fn argmin(&self, axis: impl AxisIndex, keep_dims: bool) -> Tensor<u32, B> {
        self.view().argmin(axis, keep_dims)
    }
}
