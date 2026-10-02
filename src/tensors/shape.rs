//! The shape checks tensor operations make before touching any element.
//!
//! Each panics with a message naming the operation and both shapes, which is
//! how every shape-dependent operation reports a mismatch (see the
//! [`tensors`](crate::tensors) module for why that is a panic, not a `Result`).

use std::cmp::Ordering;

/// Panics unless two tensors have the same length.
#[track_caller]
pub(crate) fn assert_same_len(left: usize, right: usize, operation: &str) {
    assert!(
        left == right,
        "{operation}: vector lengths differ, {left} and {right}"
    );
}

/// Panics unless two matrices have the same shape.
#[track_caller]
pub(crate) fn assert_same_shape(left: (usize, usize), right: (usize, usize), operation: &str) {
    assert!(
        left == right,
        "{operation}: matrix shapes differ, {}×{} and {}×{}",
        left.0,
        left.1,
        right.0,
        right.1
    );
}

/// Panics unless a clamp's bounds describe a non-empty range.
///
/// Written as "not greater" rather than "less or equal" so that a NaN bound —
/// which is unordered against everything, including itself — passes rather than
/// panicking: `x.max(NaN)` is `x` on every path here, so such a bound is inert
/// rather than wrong.
#[track_caller]
pub(crate) fn assert_ordered_bounds<T: PartialOrd>(low: &T, high: &T) {
    assert!(
        !matches!(low.partial_cmp(high), Some(Ordering::Greater)),
        "clamp: the lower bound exceeds the upper one"
    );
}

/// Panics unless the inner dimensions of a product agree.
#[track_caller]
pub(crate) fn assert_inner(left: (usize, usize), right: (usize, usize), operation: &str) {
    assert!(
        left.1 == right.0,
        "{operation}: inner dimensions differ, {}×{} times {}×{}",
        left.0,
        left.1,
        right.0,
        right.1
    );
}
