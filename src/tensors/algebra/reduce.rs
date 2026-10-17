//! Whole-value reductions, prefix sums and sorting.
//!
//! Reductions along chosen axes of a [`Tensor`](crate::tensors::Tensor) keep
//! their axis arguments and are not part of these traits.

use std::cmp::Ordering;

use crate::tensors::{Reduce, SortOrder};

/// A fold of every element to one value.
pub trait Reduction {
    /// The scalar-like result.
    type Output;

    /// Fold every element with `op`. An empty value gives
    /// [`op.identity()`](Reduce::identity).
    fn reduce(self, op: Reduce) -> Self::Output;

    /// The sum of every element.
    fn sum(self) -> Self::Output
    where
        Self: Sized,
    {
        self.reduce(Reduce::Sum)
    }

    /// The smallest element, or `None` when there are none.
    fn minimum(self) -> Option<Self::Output>;

    /// The largest element, or `None` when there are none.
    fn maximum(self) -> Option<Self::Output>;
}

/// The inclusive prefix sum, `outᵢ = Σ_{j ≤ i} selfⱼ`.
pub trait PrefixSum {
    /// The result, with the operand's shape.
    type Output;

    /// Compute the prefix sum in storage order.
    fn prefix_sum(self) -> Self::Output;
}

/// The elements in sorted order.
pub trait Sort {
    /// The element type the comparator sees.
    type Elem;

    /// The sorted result.
    type Output;

    /// Sort under [`SortOrder`]'s total order, on the value's backend.
    fn sorted(self, order: SortOrder) -> Self::Output;

    /// Sort under an arbitrary comparator.
    ///
    /// A closure cannot run on the GPU, so a resident value sorts on the CPU
    /// and stores the result back.
    fn sorted_by(self, compare: impl FnMut(&Self::Elem, &Self::Elem) -> Ordering) -> Self::Output;
}
