//! Projections onto constraint sets.
//!
//! A projection answers "the nearest feasible point to this one". Gradient
//! descent has no idea a constraint exists, so the standard way to respect one
//! is to take the unconstrained step and then project — which is exactly what
//! [`Constrained`](crate::optim::Constrained) does with the closure it wraps:
//!
//! ```
//! use tensorcrate::projections::project_onto_capped_simplex;
//! use tensorcrate::tensors::{Host, Vector};
//!
//! let mut weights = Vector::<f32, Host>::new([0.7, 0.5, -0.2]);
//! weights = project_onto_capped_simplex(&weights, 1.0);
//!
//! // Non-negative, and the budget is spent exactly.
//! assert!(weights.data().iter().all(|&w| w >= 0.0));
//! assert!((weights.sum() - 1.0).abs() < 1e-6);
//! ```
//!
//! Every one of these is written against [`Kernels`], so the same source runs on
//! the host and on the GPU with the tensors staying where they are. Nothing here
//! loops over elements: the shapes are runtime values, and a per-element loop
//! would be a scalar CPU loop even when the vector lives in GPU memory.

use crate::tensors::{BinaryOp, Compare, Kernels, Ordered, Reduce, SortOrder, Vector};

/// The Euclidean projection of `values` onto `{r : r ≥ 0, Σr ≤ cap}` — the
/// probability simplex when `cap` is one, scaled by `cap` otherwise.
///
/// Clipping the negatives away is the whole answer whenever what is left already
/// fits inside the budget. Otherwise the constraint binds, the result has to sum
/// to exactly `cap`, and the projection is `max(r − θ, 0)` for the one θ that
/// makes it do so. Finding θ is the interesting part: sort descending, and θ is
/// determined by the longest prefix of that order whose shared offset leaves
/// every element of the prefix positive.
///
/// # How it vectorizes
///
/// The textbook statement of that search is a loop with a running total and an
/// early `break`, which is three serial dependencies in a row. Each becomes one
/// whole-vector operation instead:
///
/// - the running total is an inclusive [`prefix_sum`](Kernels::vector_prefix_sum);
/// - the candidate offsets are that prefix minus `cap`, divided elementwise by
///   `1, 2, 3, …`;
/// - the `break` is a [`Greater`](Compare::Greater) predicate against the sorted
///   values, biased by the element index and folded with a
///   [`Min`](Reduce::Min) — the smallest surviving index *is* the iteration the
///   loop would have stopped at.
///
/// So the whole projection is a sort, a scan, a handful of elementwise passes
/// and two reductions, none of which cares how long the vector is.
///
/// One element is read back to the host: the offset itself, at the index the
/// fold found. Everything else stays on whichever backend it started on.
///
/// # Numerics
///
/// The index arithmetic runs through `f32`, which represents integers exactly up
/// to 2²⁴, so vectors longer than about 16 million elements would need a wider
/// index than this. NaN inputs have no meaningful projection and are not
/// handled specially.
pub fn project_onto_capped_simplex<B: Kernels>(
    values: &Vector<f32, B>,
    cap: f32,
) -> Vector<f32, B> {
    let len = values.len();
    // Clipping alone is the projection when the budget is not binding.
    let clamped = values.max_scalar(0.0);
    if len == 0 || clamped.sum() <= cap {
        return clamped;
    }

    // Descending, so a prefix of the order is the set of entries that survive.
    let sorted = values.sorted(SortOrder::Descending);
    let running = sorted.prefix_sum();

    // candidate[i] = (running[i] − cap) / (i + 1): the offset that would make
    // the first i + 1 entries sum to exactly `cap`.
    let over_budget = B::vector_broadcast(&running, cap, BinaryOp::Sub, false);
    let divisors = Vector::<f32, B>::ramp(len, 1.0, 1.0);
    let candidates = B::vector_elementwise(&over_budget, &divisors, BinaryOp::Div);

    // The candidate stops being admissible once it exceeds the value it is
    // subtracted from — that entry would come out negative. Marking every
    // *admissible* index with `len` (past the end) leaves the inadmissible ones
    // carrying their own index, so the smallest mark is the first offending
    // index, and `len` means there was none.
    let offending = candidates.compare(&sorted, Compare::Greater);
    let admissible = B::vector_broadcast(&offending, 1.0, BinaryOp::Sub, true);
    let bias = B::vector_broadcast(&admissible, len as f32, BinaryOp::Mul, false);
    let marks = B::vector_elementwise(&Vector::<f32, B>::ramp(len, 0.0, 1.0), &bias, BinaryOp::Add);
    let first_offending = marks.reduce(Reduce::Min) as usize;

    // The offset is the last admissible candidate; if even the first entry
    // offends, no entry is dropped and nothing is subtracted.
    let offset = if first_offending == 0 {
        0.0
    } else {
        candidates.as_slice()[first_offending.min(len) - 1]
    };

    B::vector_broadcast(values, offset, BinaryOp::Sub, false).max_scalar(0.0)
}

/// The Euclidean projection onto the box `[low, high]` — clipping, which is what
/// [`clamp`](Kernels::vector_clamp) already is.
///
/// It is here for the same reason [`project_onto_capped_simplex`] is: so a
/// constrained rule can name its feasible set and the projection is one call.
///
/// # Panics
///
/// If `low > high`, which describes no feasible set at all.
#[track_caller]
pub fn project_onto_box<B: Kernels>(
    values: &Vector<f32, B>,
    low: f32,
    high: f32,
) -> Vector<f32, B> {
    B::vector_clamp(values, low, high)
}

/// The projection onto the ball `{r : ‖r‖₂ ≤ radius}`: scale down if outside,
/// leave alone if inside.
///
/// This is gradient clipping, and the norm is a [`Sum`](Reduce::Sum) reduction
/// over the squares, so it stays on the backend the vector is on.
///
/// # Panics
///
/// If `radius` is negative.
#[track_caller]
pub fn project_onto_ball<B: Kernels>(values: &Vector<f32, B>, radius: f32) -> Vector<f32, B> {
    assert!(radius >= 0.0, "project_onto_ball: the radius is negative");
    let norm = B::vector_elementwise(values, values, BinaryOp::Mul)
        .sum()
        .sqrt();
    if norm <= radius {
        return values.to_backend::<B>();
    }
    B::vector_broadcast(values, radius / norm, BinaryOp::Mul, false)
}
