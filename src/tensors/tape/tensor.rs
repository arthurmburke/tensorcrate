//! Reverse-mode rules for [`TensorVar`]: N-dimensional tensors on the tape.
//!
//! Every rule is written with the tensor operations, which run on the tensor's
//! backend, so a backward pass over a `Metal` graph stays on the GPU. The
//! adjoint of an operation that broadcast an operand is summed back down to
//! that operand's shape; the adjoint of one that picked out part of a tensor
//! is written into the matching part of the parent's.

use std::ops::{Add, Div, Mul, RangeBounds, Sub};

use super::{Adjoint, Lazy, MatrixVar, Node, ScalarVar, TensorVar, VectorVar, sum_of};
use crate::numbers::Real;
use crate::statistics::Correction;
use crate::tensors::layout::{AxisIndex, bounds, resolve_axis};
use crate::tensors::{Analytic, Axes, BinaryOp, Compare, Kernels, Reduce, Tensor, TensorView};

// ---- helpers ----------------------------------------------------------------

/// `gradient` summed down to `shape`, which broadcast to `gradient`'s shape:
/// the adjoint of a broadcast. The leading axes `shape` lacks, and every axis
/// where it has `1` and the gradient more, are summed — one reduction — and
/// the result takes `shape`.
fn sum_to_shape<E: Real, B: Kernels<E>>(gradient: &Tensor<E, B>, shape: &[usize]) -> Tensor<E, B> {
    if gradient.shape() == shape {
        return gradient.contiguous();
    }
    let added = gradient.rank() - shape.len();
    let axes = (0..gradient.rank())
        .filter(|&axis| axis < added || (shape[axis - added] == 1 && gradient.shape()[axis] != 1))
        .collect::<Vec<_>>();
    let summed = if axes.is_empty() {
        gradient.contiguous()
    } else {
        gradient.sum_axes(axes, true)
    };
    summed.reshape(shape)
}

/// The axes a reduction folds, resolved against `shape` — which the forward
/// reduction has already checked — the shape its result has with them kept at
/// extent `1`, and how many elements each slice holds.
fn folding(axes: &impl Axes, shape: &[usize]) -> (Vec<usize>, Vec<usize>, usize) {
    let folded = axes
        .resolve_axes(shape.len())
        .expect("the forward reduction checked the axes");
    let mut kept = shape.to_vec();
    let mut count = 1usize;
    for &axis in &folded {
        count *= shape[axis];
        kept[axis] = 1;
    }
    (folded, kept, count)
}

impl<E: Real, B: Kernels<E>> Node<Tensor<E, B>, B> {
    /// Add `delta` into the elements of `axis` from `start` of this node's
    /// adjoint — the adjoint of a slice, written where the slice came from.
    ///
    /// With no adjoint yet, the slice is written into zeros; otherwise the
    /// slice of the adjoint is summed with `delta` and written back in place,
    /// so the rest of the adjoint is never touched. Either way it is a strided
    /// write on the tensor's backend.
    fn accumulate_slice(&self, axis: usize, start: usize, delta: TensorView<'_, E, B>) {
        let mut adjoint = self.adjoint.borrow_mut();
        match adjoint.as_mut() {
            Some(current) => {
                let len = delta.shape()[axis];
                let sum = current
                    .narrow(axis, start, len)
                    .elementwise(delta, BinaryOp::Add);
                current.write_slice(axis, start, &sum);
            }
            None => {
                let mut zeros = self.value().zeros_like();
                zeros.write_slice(axis, start, delta);
                *adjoint = Some(zeros);
            }
        }
    }
}

// ---- tensors ----------------------------------------------------------------

impl<'t, B: Kernels<E>, E: Real> TensorVar<'t, B, E> {
    /// The extent of each axis, outermost first.
    pub fn shape(&self) -> &[usize] {
        self.node.value().shape()
    }

    /// The number of axes.
    pub fn rank(&self) -> usize {
        self.node.value().rank()
    }

    /// The number of elements.
    pub fn len(&self) -> usize {
        self.node.value().len()
    }

    /// Whether this tensor holds no elements.
    pub fn is_empty(&self) -> bool {
        self.node.value().is_empty()
    }

    /// The extent of `axis`; a negative axis counts from the last.
    ///
    /// # Panics
    ///
    /// If there is no such axis.
    #[track_caller]
    pub fn dim(&self, axis: impl AxisIndex) -> usize {
        self.node.value().dim(axis)
    }

    /// A node whose value is this one's elements under another shape, in the
    /// same order, and whose adjoint takes this one's shape back.
    fn reshaped(&self, value: Tensor<E, B>) -> Self {
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            parent.accumulate(adjoint.contiguous().reshape(parent.value().shape()));
        })
    }

    // ---- elementwise ----

    /// `self op rhs` for `op` one of `+ − × ÷`.
    #[track_caller]
    fn binary(&self, rhs: &Self, op: BinaryOp) -> Self {
        self.assert_same_tape(rhs);
        let value = self.value().elementwise(rhs.value(), op);
        let (left, right) = (self.node.clone(), rhs.node.clone());
        self.record(value, move |adjoint| {
            let (a, b) = (left.value(), right.value());
            match op {
                BinaryOp::Add => {
                    left.accumulate(sum_to_shape(adjoint, a.shape()));
                    right.accumulate(sum_to_shape(adjoint, b.shape()));
                }
                BinaryOp::Sub => {
                    left.accumulate(sum_to_shape(adjoint, a.shape()));
                    right.accumulate(sum_to_shape(adjoint, b.shape()).scale(-E::one()));
                }
                // c = a ⊙ b
                BinaryOp::Mul => {
                    left.accumulate(sum_to_shape(&adjoint.elementwise(b, op), a.shape()));
                    right.accumulate(sum_to_shape(&adjoint.elementwise(a, op), b.shape()));
                }
                // c = a / b: ā += c̄/b, b̄ −= (c̄/b)·a/b
                BinaryOp::Div => {
                    let over = adjoint.elementwise(b, BinaryOp::Div);
                    let scaled = over
                        .elementwise(a, BinaryOp::Mul)
                        .elementwise(b, BinaryOp::Div);
                    left.accumulate(sum_to_shape(&over, a.shape()));
                    right.accumulate(sum_to_shape(&scaled, b.shape()).scale(-E::one()));
                }
                BinaryOp::Rem => unreachable!("no tensor remainder is recorded"),
            }
        })
    }

    /// Elementwise `self^exponent`, with the shapes broadcast:
    /// `ā += c̄·b·aᵇ⁻¹` and `b̄ += c̄·aᵇ·ln a`, each summed to its operand's
    /// shape. The exponent's gradient is that of the real power, defined for
    /// `a > 0`; elsewhere it is NaN, as `ln a` is.
    ///
    /// # Panics
    ///
    /// If the shapes do not broadcast.
    #[track_caller]
    pub fn power(&self, exponent: &Self) -> Self {
        self.assert_same_tape(exponent);
        let value = self.value().power(exponent.value());
        let (base, power) = (self.node.clone(), exponent.node.clone());
        self.record(value, move |adjoint| {
            let (a, b) = (base.value(), power.value());
            let lowered = b.with_scalar(E::one(), BinaryOp::Sub, false);
            let slope = a.power(&lowered).elementwise(b, BinaryOp::Mul);
            let base_adjoint = adjoint.elementwise(&slope, BinaryOp::Mul);
            base.accumulate(sum_to_shape(&base_adjoint, a.shape()));

            let growth = a
                .power(b)
                .elementwise(&a.unary(Analytic::Ln), BinaryOp::Mul);
            let power_adjoint = adjoint.elementwise(&growth, BinaryOp::Mul);
            power.accumulate(sum_to_shape(&power_adjoint, b.shape()));
        })
    }

    /// `x^scalar` for every element — or `scalar^x` if `scalar_left` — with
    /// `x̄ += ȳ·scalar·x^(scalar−1)`, or `x̄ += ȳ·scalar^x·ln scalar`.
    pub fn power_scalar(&self, scalar: E, scalar_left: bool) -> Self {
        let value = self.value().power_scalar(scalar, scalar_left);
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            let x = parent.value();
            let slope = if scalar_left {
                x.power_scalar(scalar, true)
                    .scale(Analytic::Ln.value(scalar))
            } else {
                x.power_scalar(scalar - E::one(), false).scale(scalar)
            };
            parent.accumulate(adjoint.elementwise(&slope, BinaryOp::Mul));
        })
    }

    /// Elementwise larger of two recorded tensors, with the shapes broadcast.
    ///
    /// The adjoint goes to whichever operand supplied the value and splits
    /// evenly where they tie — the convention [`Compare`] documents and
    /// [`VectorVar::maximum`] follows — then is summed to each operand's
    /// shape.
    ///
    /// # Panics
    ///
    /// If the shapes do not broadcast.
    #[track_caller]
    pub fn maximum(&self, other: &Self) -> Self {
        self.select_pair(other, true)
    }

    /// Elementwise smaller of two recorded tensors; see
    /// [`maximum`](Self::maximum).
    ///
    /// # Panics
    ///
    /// If the shapes do not broadcast.
    #[track_caller]
    pub fn minimum(&self, other: &Self) -> Self {
        self.select_pair(other, false)
    }

    #[track_caller]
    fn select_pair(&self, other: &Self, largest: bool) -> Self {
        self.assert_same_tape(other);
        let op = if largest { Compare::Max } else { Compare::Min };
        let value = self.value().compare(other.value(), op);
        let (left, right) = (self.node.clone(), other.node.clone());
        self.record(value, move |adjoint| {
            let (a, b) = (left.value(), right.value());
            let share = a.compare(b, Compare::MaxShare);
            let complement = share.with_scalar(E::one(), BinaryOp::Sub, true);
            let (mine, theirs) = if largest {
                (&share, &complement)
            } else {
                (&complement, &share)
            };
            let left_adjoint = adjoint.elementwise(mine, BinaryOp::Mul);
            let right_adjoint = adjoint.elementwise(theirs, BinaryOp::Mul);
            left.accumulate(sum_to_shape(&left_adjoint, a.shape()));
            right.accumulate(sum_to_shape(&right_adjoint, b.shape()));
        })
    }

    /// Elementwise maximum against a constant.
    pub fn clamp_min(&self, floor: E) -> Self {
        self.select_scalar(floor, true)
    }

    /// Elementwise minimum against a constant.
    pub fn clamp_max(&self, ceiling: E) -> Self {
        self.select_scalar(ceiling, false)
    }

    /// Confine every element to `[floor, ceiling]`; the gradient is zero
    /// wherever an element is pinned to a bound, and half where it equals one.
    pub fn clamp(&self, floor: E, ceiling: E) -> Self {
        self.clamp_min(floor).clamp_max(ceiling)
    }

    /// The rectifier `max(x, 0)`.
    pub fn relu(&self) -> Self {
        self.clamp_min(E::zero())
    }

    /// Elementwise absolute value, as `max(x, −x)`, which differentiates to
    /// `sign(x)` with `sign(0) = 0`.
    pub fn abs(&self) -> Self {
        self.maximum(&self.scale(-E::one()))
    }

    fn select_scalar(&self, scalar: E, largest: bool) -> Self {
        let op = if largest { Compare::Max } else { Compare::Min };
        let value = self.value().compare_scalar(scalar, op, false);
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            let share = parent
                .value()
                .compare_scalar(scalar, Compare::MaxShare, false);
            let weight = if largest {
                share
            } else {
                share.with_scalar(E::one(), BinaryOp::Sub, true)
            };
            parent.accumulate(adjoint.elementwise(&weight, BinaryOp::Mul));
        })
    }

    /// Apply an analytic function elementwise: `x̄ += f'(x) ⊙ ȳ`, one
    /// `unary_dual` dispatch over the flat storage.
    pub fn analytic(&self, f: Analytic) -> Self {
        let value = self.value().unary(f);
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            let x = parent.value();
            let (_, product) = B::vector_unary_dual(x.as_vector(), adjoint.as_vector(), f);
            parent.accumulate(Tensor::from_vector(x.shape(), product));
        })
    }

    /// Negate every element.
    pub fn neg(&self) -> Self {
        self.scale(-E::one())
    }

    /// Multiply by a constant.
    pub fn scale(&self, factor: E) -> Self {
        let value = self.value().scale(factor);
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            parent.accumulate(adjoint.scale(factor));
        })
    }

    /// Add a constant to every element, which leaves the derivative unchanged.
    pub fn shift(&self, offset: E) -> Self {
        let value = self.value().with_scalar(offset, BinaryOp::Add, false);
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            parent.accumulate(adjoint.contiguous());
        })
    }

    // ---- layout ----

    /// The same elements under a new shape; the adjoint takes the old shape
    /// back.
    ///
    /// # Panics
    ///
    /// If `shape` holds a different number of elements.
    #[track_caller]
    pub fn reshape(&self, shape: &[usize]) -> Self {
        self.reshaped(self.value().contiguous().reshape(shape))
    }

    /// Remove `axis`, which must have one element.
    ///
    /// # Panics
    ///
    /// If there is no such axis or its extent is not one.
    #[track_caller]
    pub fn squeeze(&self, axis: impl AxisIndex) -> Self {
        self.reshaped(self.value().contiguous().squeeze(axis))
    }

    /// Insert an axis of one element before `axis`, which may be the rank.
    ///
    /// # Panics
    ///
    /// If `axis` is past the rank, or the tensor already has
    /// [`MAX_RANK`](crate::tensors::MAX_RANK) axes.
    #[track_caller]
    pub fn unsqueeze(&self, axis: impl AxisIndex) -> Self {
        self.reshaped(self.value().contiguous().unsqueeze(axis))
    }

    /// A copy of this tensor, whose adjoint flows straight back. Every value
    /// on the tape is already contiguous, so this is only a copy; it exists
    /// so that code written against [`Tensor`] reads the same here.
    pub fn contiguous(&self) -> Self {
        let parent = self.node.clone();
        self.record(self.value().contiguous(), move |adjoint| {
            parent.accumulate(adjoint.contiguous());
        })
    }

    /// The axes reordered: axis `k` of the result is axis `axes[k]` of this
    /// tensor. The value is copied into order; the adjoint is permuted back
    /// by the inverse order.
    ///
    /// # Panics
    ///
    /// Unless `axes` names every axis exactly once.
    #[track_caller]
    pub fn permute(&self, axes: &[usize]) -> Self {
        let value = self.value().permute(axes).contiguous();
        let mut inverse = vec![0; axes.len()];
        for (position, &axis) in axes.iter().enumerate() {
            inverse[axis] = position;
        }
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            parent.accumulate(adjoint.permute(&inverse).contiguous());
        })
    }

    /// Axes `a` and `b` swapped; the adjoint swaps them back.
    ///
    /// # Panics
    ///
    /// If either axis does not exist.
    #[track_caller]
    pub fn transpose(&self, a: impl AxisIndex, b: impl AxisIndex) -> Self {
        let value = self.value().transpose(a, b).contiguous();
        let a = resolve_axis(a, self.shape(), "transpose");
        let b = resolve_axis(b, self.shape(), "transpose");
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            parent.accumulate(adjoint.transpose(a, b).contiguous());
        })
    }

    /// `len` elements of `axis` from `start`. The adjoint is written into
    /// that part of this tensor's adjoint, in place — see the
    /// [module documentation](super#tensors).
    ///
    /// # Panics
    ///
    /// If the axis does not exist or the range reaches past its extent.
    #[track_caller]
    pub fn narrow(&self, axis: impl AxisIndex, start: usize, len: usize) -> Self {
        let value = self.value().narrow(axis, start, len).contiguous();
        let axis = resolve_axis(axis, self.shape(), "narrow");
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            parent.accumulate_slice(axis, start, adjoint.view());
        })
    }

    /// The elements of `axis` in `range`; see [`narrow`](Self::narrow).
    ///
    /// # Panics
    ///
    /// If the axis does not exist or the range reaches past its extent.
    #[track_caller]
    pub fn slice(&self, axis: impl AxisIndex, range: impl RangeBounds<usize>) -> Self {
        let axis = resolve_axis(axis, self.shape(), "slice");
        let range = bounds(range, self.shape()[axis], axis, "slice");
        self.narrow(axis, range.start, range.len())
    }

    /// Element `index` of `axis`, with that axis removed; the adjoint is
    /// written into that position of this tensor's adjoint.
    ///
    /// # Panics
    ///
    /// If the axis does not exist or `index` is past its extent.
    #[track_caller]
    pub fn select(&self, axis: impl AxisIndex, index: usize) -> Self {
        let value = self.value().select(axis, index).contiguous();
        let axis = resolve_axis(axis, self.shape(), "select");
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            parent.accumulate_slice(axis, index, adjoint.view().unsqueeze(axis));
        })
    }

    /// Consecutive pieces of `axis`, of the extents in `sizes`: one
    /// [`narrow`](Self::narrow) each, so each piece's adjoint lands in its own
    /// part of this tensor's.
    ///
    /// # Panics
    ///
    /// If the axis does not exist or `sizes` does not add up to its extent.
    #[track_caller]
    pub fn split(&self, axis: impl AxisIndex, sizes: &[usize]) -> Vec<Self> {
        let _ = self.value().split(axis, sizes);
        let axis = resolve_axis(axis, self.shape(), "split");
        let mut start = 0;
        sizes
            .iter()
            .map(|&size| {
                let piece = self.narrow(axis, start, size);
                start += size;
                piece
            })
            .collect()
    }

    /// `parts` equal pieces of `axis` — `chunk(-1, 3)` splits a fused
    /// query–key–value projection. See [`split`](Self::split).
    ///
    /// # Panics
    ///
    /// If the axis does not exist, or `parts` is zero or does not divide its
    /// extent.
    #[track_caller]
    pub fn chunk(&self, axis: impl AxisIndex, parts: usize) -> Vec<Self> {
        let _ = self.value().chunk(axis, parts);
        let axis = resolve_axis(axis, self.shape(), "chunk");
        self.split(axis, &vec![self.shape()[axis] / parts; parts])
    }

    /// Recorded tensors joined along `axis`; each one's adjoint is its part
    /// of the joined adjoint, narrowed back out.
    ///
    /// # Panics
    ///
    /// If there are no pieces, they are on different tapes, their ranks
    /// differ, or two of them differ in an extent other than `axis`'s.
    #[track_caller]
    pub fn concat(pieces: &[Self], axis: impl AxisIndex) -> Self {
        let first = pieces
            .first()
            .unwrap_or_else(|| panic!("concat: there are no tensors to join"));
        for piece in pieces {
            first.assert_same_tape(piece);
        }
        let value = Tensor::concat(pieces.iter().map(|piece| piece.value()), axis);
        let axis = resolve_axis(axis, first.shape(), "concat");
        let nodes = pieces
            .iter()
            .map(|piece| piece.node.clone())
            .collect::<Vec<_>>();
        first.record(value, move |adjoint| {
            let mut start = 0;
            for node in &nodes {
                let len = node.value().shape()[axis];
                node.accumulate(adjoint.narrow(axis, start, len).contiguous());
                start += len;
            }
        })
    }

    /// Recorded tensors of one shape stacked along a new axis inserted before
    /// `axis`; each one's adjoint is its index of the stacked adjoint.
    ///
    /// # Panics
    ///
    /// If there are no pieces, they are on different tapes, their shapes
    /// differ, or they already have [`MAX_RANK`](crate::tensors::MAX_RANK)
    /// axes.
    #[track_caller]
    pub fn stack(pieces: &[Self], axis: impl AxisIndex) -> Self {
        let first = pieces
            .first()
            .unwrap_or_else(|| panic!("stack: there are no tensors to stack"));
        for piece in pieces {
            first.assert_same_tape(piece);
        }
        let value = Tensor::stack(pieces.iter().map(|piece| piece.value()), axis);
        let axis = axis
            .resolve(first.rank() + 1)
            .expect("the forward stack checked the axis");
        let nodes = pieces
            .iter()
            .map(|piece| piece.node.clone())
            .collect::<Vec<_>>();
        first.record(value, move |adjoint| {
            for (index, node) in nodes.iter().enumerate() {
                node.accumulate(adjoint.select(axis, index).contiguous());
            }
        })
    }

    /// This tensor repeated to `shape`, numpy-style; the adjoint is summed
    /// back over the repeated axes.
    ///
    /// # Panics
    ///
    /// If this tensor's shape does not broadcast to `shape`.
    #[track_caller]
    pub fn broadcast_to(&self, shape: &[usize]) -> Self {
        let value = self.value().broadcast_to(shape).contiguous();
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            parent.accumulate(sum_to_shape(adjoint, parent.value().shape()));
        })
    }

    // ---- reductions ----

    /// The sum of each slice along `axes` (see [`Tensor::sum_axes`]); every
    /// element of a slice receives the slice's adjoint.
    ///
    /// # Panics
    ///
    /// If `axes` names an axis that does not exist, or one twice.
    #[track_caller]
    pub fn sum_axes(&self, axes: impl Axes, keep_dims: bool) -> Self {
        let value = self.value().sum_axes(&axes, keep_dims);
        let (_, kept, _) = folding(&axes, self.shape());
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            let spread = adjoint.view_as(&kept).broadcast_to(parent.value().shape());
            parent.accumulate(spread.contiguous());
        })
    }

    /// The mean of each slice along `axes`; every element of a slice receives
    /// the slice's adjoint over its length.
    ///
    /// # Panics
    ///
    /// As for [`sum_axes`](Self::sum_axes).
    #[track_caller]
    pub fn mean_axes(&self, axes: impl Axes, keep_dims: bool) -> Self {
        let value = self.value().mean_axes(&axes, keep_dims);
        let (_, kept, count) = folding(&axes, self.shape());
        let share = E::from_f64(1.0 / count as f64);
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            let scaled = adjoint.scale(share);
            let spread = scaled.view_as(&kept).broadcast_to(parent.value().shape());
            parent.accumulate(spread.contiguous());
        })
    }

    /// The variance of each slice along `axes` under `correction` (see
    /// [`Tensor::var_axes`]): `x̄ += ȳ·2(x − mean)/d`, where `d` is the
    /// correction's divisor. A slice too short for the correction has a NaN
    /// variance and a NaN gradient.
    ///
    /// # Panics
    ///
    /// As for [`sum_axes`](Self::sum_axes).
    #[track_caller]
    pub fn var_axes(&self, axes: impl Axes, correction: Correction, keep_dims: bool) -> Self {
        let value = self.value().var_axes(&axes, correction, keep_dims);
        let (folded, kept, count) = folding(&axes, self.shape());
        let factor = correction
            .divisor(count)
            .map_or(E::nan(), |divisor| E::from_f64(2.0 / divisor as f64));
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            let x = parent.value();
            let deviation = x - &x.mean_axes(&folded, true);
            let weight = adjoint.scale(factor);
            parent.accumulate(deviation.elementwise(weight.view_as(&kept), BinaryOp::Mul));
        })
    }

    /// The largest element of each slice along `axes` (see
    /// [`Tensor::max_axes`]).
    ///
    /// The adjoint goes to the element that is the maximum, and where several
    /// tie for it, splits evenly among them — the even split [`Compare`]'s
    /// `MaxShare` gives two tied operands, extended to any number. NaNs, which
    /// the maximum passes over, receive none.
    ///
    /// # Panics
    ///
    /// As for [`sum_axes`](Self::sum_axes).
    #[track_caller]
    pub fn max_axes(&self, axes: impl Axes, keep_dims: bool) -> Self {
        self.extreme_axes(axes, keep_dims, Reduce::Max)
    }

    /// The smallest element of each slice along `axes`; the adjoint goes to
    /// the minimum as [`max_axes`](Self::max_axes)' goes to the maximum.
    ///
    /// # Panics
    ///
    /// As for [`sum_axes`](Self::sum_axes).
    #[track_caller]
    pub fn min_axes(&self, axes: impl Axes, keep_dims: bool) -> Self {
        self.extreme_axes(axes, keep_dims, Reduce::Min)
    }

    #[track_caller]
    fn extreme_axes(&self, axes: impl Axes, keep_dims: bool, op: Reduce) -> Self {
        let value = self.value().reduce_axes(op, &axes, keep_dims);
        let (folded, kept, _) = folding(&axes, self.shape());
        let attains = match op {
            Reduce::Max => Compare::GreaterEqual,
            _ => Compare::LessEqual,
        };
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            let x = parent.value();
            // One where an element equals its slice's extreme — `≥` the
            // maximum is equal to it, and false for a NaN — and the adjoint
            // shared among however many do.
            let extreme = x.reduce_axes(op, &folded, true);
            let mask = x.compare(&extreme, attains);
            let ties = mask.sum_axes(&folded, true).max_scalar(E::one());
            let share = adjoint.view_as(&kept).elementwise(&ties, BinaryOp::Div);
            parent.accumulate(mask.elementwise(&share, BinaryOp::Mul));
        })
    }

    /// The sum of every element: each element's gradient is the scalar's
    /// adjoint. Like [`VectorVar::sum`], the value is computed when first
    /// read, so a loss that is only differentiated never waits for the
    /// device.
    pub fn sum(&self) -> ScalarVar<'t, B, E> {
        let parent = self.node.clone();
        let total = {
            let parent = parent.clone();
            Lazy::deferred(move || sum_of(parent.value().as_slice()))
        };
        self.record_lazy(total, move |adjoint| {
            parent.accumulate(Tensor::filled(parent.value().shape(), *adjoint));
        })
    }

    /// The mean of every element; see [`sum`](Self::sum).
    pub fn mean(&self) -> ScalarVar<'t, B, E> {
        self.sum().scale(E::from_f64(1.0 / self.len() as f64))
    }

    // ---- conversions ----

    /// The elements, in row-major order, as a `rows × cols` matrix, so the
    /// matrix operations — [`matmul`](MatrixVar::matmul) and the rest — apply;
    /// the adjoint takes this tensor's shape back.
    ///
    /// # Panics
    ///
    /// If `rows × cols` is not the number of elements.
    #[track_caller]
    pub fn to_matrix(&self, rows: usize, cols: usize) -> MatrixVar<'t, B, E> {
        let len = self.len();
        assert!(
            rows.checked_mul(cols) == Some(len),
            "to_matrix: shape {:?} holds {len} elements, which {rows}×{cols} cannot",
            self.shape()
        );
        let value = self
            .value()
            .contiguous()
            .reshape(&[rows, cols])
            .into_matrix();
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            let flat = Tensor::from(adjoint.to_backend::<B>());
            parent.accumulate(flat.reshape(parent.value().shape()));
        })
    }

    /// The elements, in row-major order, as a vector; the adjoint takes this
    /// tensor's shape back.
    pub fn to_vector(&self) -> VectorVar<'t, B, E> {
        let value = self.value().contiguous().into_vector();
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            let flat = adjoint.to_backend::<B>();
            parent.accumulate(Tensor::from_vector(parent.value().shape(), flat));
        })
    }
}

impl<'t, B: Kernels<E>, E: Real> MatrixVar<'t, B, E> {
    /// The elements, in row-major order, as a tensor of `shape`; the adjoint
    /// takes this matrix's shape back. With
    /// [`TensorVar::to_matrix`], this mixes the matrix products into a tensor
    /// computation.
    ///
    /// # Panics
    ///
    /// If `shape` holds a different number of elements.
    #[track_caller]
    pub fn to_tensor(&self, shape: &[usize]) -> TensorVar<'t, B, E> {
        let value = Tensor::from(self.value().to_backend::<B>()).reshape(shape);
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            let (rows, cols) = parent.value().shape();
            parent.accumulate(adjoint.contiguous().reshape(&[rows, cols]).into_matrix());
        })
    }
}

impl<'t, B: Kernels<E>, E: Real> VectorVar<'t, B, E> {
    /// The elements as a tensor of `shape`; the adjoint is flattened back.
    ///
    /// # Panics
    ///
    /// If `shape` holds a different number of elements.
    #[track_caller]
    pub fn to_tensor(&self, shape: &[usize]) -> TensorVar<'t, B, E> {
        let value = Tensor::from(self.value().to_backend::<B>()).reshape(shape);
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            parent.accumulate(adjoint.contiguous().into_vector());
        })
    }
}

// ---- named operations and operators -----------------------------------------

/// The arithmetic operators, broadcasting, as methods and on references.
macro_rules! tensor_binary {
    ($($method:ident => $op:expr, $trait:ident :: $trait_method:ident),+ $(,)?) => {
        impl<'t, B: Kernels<E>, E: Real> TensorVar<'t, B, E> {
            $(
                #[doc = concat!(
                    "Elementwise `", stringify!($method), "` with the shapes broadcast, ",
                    "differentiated: each operand's adjoint is summed to its shape.\n\n",
                    "# Panics\n\nIf the shapes do not broadcast."
                )]
                #[track_caller]
                pub fn $method(&self, rhs: &Self) -> Self { self.binary(rhs, $op) }
            )+
        }

        $(
            impl<'t, B: Kernels<E>, E: Real> $trait for &TensorVar<'t, B, E> {
                type Output = TensorVar<'t, B, E>;
                #[track_caller]
                fn $trait_method(self, rhs: Self) -> Self::Output { self.binary(rhs, $op) }
            }
        )+
    };
}

tensor_binary!(
    add => BinaryOp::Add, Add::add,
    sub => BinaryOp::Sub, Sub::sub,
    mul => BinaryOp::Mul, Mul::mul,
    div => BinaryOp::Div, Div::div,
);
