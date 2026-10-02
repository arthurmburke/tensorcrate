//! The shape arithmetic behind [`Tensor`](super::Tensor) and
//! [`TensorView`](super::TensorView): extents, strides and axes, and the walk a
//! strided copy makes over them.
//!
//! A layout is a shape, one stride per axis, and an offset: element
//! `[i₀, …, iₙ₋₁]` lies at `offset + Σ iₖ·strides[k]` of flat storage. An owned
//! tensor always has the row-major layout of its shape; a view may have any
//! layout that stays inside its storage. Nothing here touches an element.

use std::fmt;
use std::ops::{Bound, Deref, Range, RangeBounds};

/// The most axes a [`Tensor`](super::Tensor) or
/// [`TensorView`](super::TensorView) may have, and the deepest layout a
/// [`Backend`](super::Backend) strided copy takes.
pub const MAX_RANK: usize = 6;

/// Up to [`MAX_RANK`] extents or strides, stored inline so that a layout
/// operation allocates nothing.
#[derive(Copy, Clone, Default, PartialEq, Eq, Hash)]
pub(crate) struct Dims {
    rank: u8,
    values: [usize; MAX_RANK],
}

impl Dims {
    /// `values` as dimensions.
    ///
    /// # Panics
    ///
    /// If there are more than [`MAX_RANK`] of them; `operation` names the
    /// caller in the message.
    #[track_caller]
    pub(crate) fn new(values: &[usize], operation: &str) -> Self {
        assert!(
            values.len() <= MAX_RANK,
            "{operation}: shape {values:?} has {} axes, more than the {MAX_RANK} a tensor may have",
            values.len()
        );
        let mut stored = [0; MAX_RANK];
        stored[..values.len()].copy_from_slice(values);
        Dims {
            rank: values.len() as u8,
            values: stored,
        }
    }

    pub(crate) fn as_slice(&self) -> &[usize] {
        &self.values[..usize::from(self.rank)]
    }

    /// These dimensions with `value` inserted at `axis`.
    #[track_caller]
    pub(crate) fn inserted(&self, axis: usize, value: usize, operation: &str) -> Self {
        let mut values = self.as_slice().to_vec();
        values.insert(axis, value);
        Dims::new(&values, operation)
    }

    /// These dimensions without the one at `axis`.
    pub(crate) fn removed(&self, axis: usize) -> Self {
        let mut values = self.as_slice().to_vec();
        values.remove(axis);
        Dims::new(&values, "remove an axis")
    }

    /// These dimensions with the one at `axis` replaced by `value`.
    pub(crate) fn with(&self, axis: usize, value: usize) -> Self {
        let mut dims = *self;
        dims.values[axis] = value;
        dims
    }
}

impl Deref for Dims {
    type Target = [usize];

    fn deref(&self) -> &[usize] {
        self.as_slice()
    }
}

impl fmt::Debug for Dims {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_slice().fmt(f)
    }
}

/// The number of elements a tensor of `shape` holds: `1` for the empty shape,
/// a scalar.
///
/// # Panics
///
/// If the count does not fit in a `usize`.
#[track_caller]
pub(crate) fn element_count(shape: &[usize], operation: &str) -> usize {
    shape
        .iter()
        .try_fold(1usize, |count, &extent| count.checked_mul(extent))
        .unwrap_or_else(|| {
            panic!("{operation}: shape {shape:?} has more elements than fit in a usize")
        })
}

/// The row-major strides of `shape`: the last axis steps by one element.
pub(crate) fn contiguous_strides(shape: &[usize]) -> Dims {
    let mut strides = [0; MAX_RANK];
    let mut step = 1usize;
    for (axis, &extent) in shape.iter().enumerate().rev() {
        strides[axis] = step;
        step = step.wrapping_mul(extent);
    }
    Dims {
        rank: shape.len() as u8,
        values: strides,
    }
}

/// Whether a layout of `shape` with `strides` is row-major order: every axis
/// that has more than one element steps by the elements of the axes after it.
/// An empty layout is trivially in order.
pub(crate) fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    if shape.contains(&0) {
        return true;
    }
    let mut step = 1;
    for (&extent, &stride) in shape.iter().zip(strides).rev() {
        if extent != 1 && stride != step {
            return false;
        }
        step *= extent;
    }
    true
}

/// One past the last storage element a layout of `shape` reads from `offset`
/// with `strides`, or `0` if it reads none. `None` if the arithmetic overflows.
pub(crate) fn reach(shape: &[usize], offset: usize, strides: &[usize]) -> Option<usize> {
    if shape.contains(&0) {
        return Some(0);
    }
    shape
        .iter()
        .zip(strides)
        .try_fold(offset, |end, (&extent, &stride)| {
            end.checked_add((extent - 1).checked_mul(stride)?)
        })?
        .checked_add(1)
}

/// The storage element at `index` of a layout, or `None` if `index` has the
/// wrong number of coordinates or any is out of range.
pub(crate) fn position(
    shape: &[usize],
    strides: &[usize],
    offset: usize,
    index: &[usize],
) -> Option<usize> {
    if index.len() != shape.len() {
        return None;
    }
    index.iter().zip(shape).zip(strides).try_fold(
        offset,
        |at, ((&coordinate, &extent), &stride)| {
            (coordinate < extent).then(|| at + coordinate * stride)
        },
    )
}

/// [`position`], panicking with the index and the shape when it is `None`.
#[track_caller]
pub(crate) fn expect_position(
    shape: &[usize],
    strides: &[usize],
    offset: usize,
    index: &[usize],
) -> usize {
    position(shape, strides, offset, index)
        .unwrap_or_else(|| panic!("index {index:?} is out of range for shape {shape:?}"))
}

/// An axis number: a `usize` counts from the first axis, and a negative
/// `isize` or `i32` from the last, so `-1` is the innermost axis.
pub trait AxisIndex: Copy + fmt::Debug {
    /// The axis this names among `rank` axes, or `None` if it names none.
    fn resolve(self, rank: usize) -> Option<usize>;
}

impl AxisIndex for usize {
    fn resolve(self, rank: usize) -> Option<usize> {
        (self < rank).then_some(self)
    }
}

impl AxisIndex for isize {
    fn resolve(self, rank: usize) -> Option<usize> {
        let rank_signed = isize::try_from(rank).ok()?;
        let axis = if self < 0 { self + rank_signed } else { self };
        (0..rank_signed).contains(&axis).then_some(axis as usize)
    }
}

impl AxisIndex for i32 {
    fn resolve(self, rank: usize) -> Option<usize> {
        (self as isize).resolve(rank)
    }
}

/// `axis` among `rank` axes.
///
/// # Panics
///
/// If it names none of them.
#[track_caller]
pub(crate) fn resolve_axis(axis: impl AxisIndex, shape: &[usize], operation: &str) -> usize {
    axis.resolve(shape.len()).unwrap_or_else(|| {
        panic!(
            "{operation}: axis {axis:?} is out of range for shape {shape:?} ({} axes)",
            shape.len()
        )
    })
}

/// `range` of `0..extent`, checked.
#[track_caller]
pub(crate) fn bounds(
    range: impl RangeBounds<usize>,
    extent: usize,
    axis: usize,
    operation: &str,
) -> Range<usize> {
    let start = match range.start_bound() {
        Bound::Included(&start) => start,
        Bound::Excluded(&start) => start.saturating_add(1),
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&end) => end.saturating_add(1),
        Bound::Excluded(&end) => end,
        Bound::Unbounded => extent,
    };
    assert!(
        start <= end && end <= extent,
        "{operation}: {start}..{end} reaches past axis {axis}'s extent of {extent}"
    );
    start..end
}

/// The strides that read `new_shape` in row-major order from a layout of
/// `shape` with `strides`, without moving an element — or `None` if no strides
/// can, because the reshape would merge axes the layout does not keep
/// adjacent. The two shapes hold the same number of elements.
pub(crate) fn reshape_strides(
    shape: &[usize],
    strides: &[usize],
    new_shape: &[usize],
) -> Option<Dims> {
    if shape.contains(&0) {
        return Some(contiguous_strides(new_shape));
    }
    // Axes of one element step nowhere, so they place no constraint.
    let old: Vec<(usize, usize)> = shape
        .iter()
        .zip(strides)
        .filter(|&(&extent, _)| extent != 1)
        .map(|(&extent, &stride)| (extent, stride))
        .collect();
    let mut new_strides = Dims::new(&vec![0; new_shape.len()], "reshape");
    let (mut next_old, mut next_new) = (0, 0);
    while next_new < new_shape.len() {
        if new_shape[next_new] == 1 {
            next_new += 1;
            continue;
        }
        // The smallest run of new axes and of old ones holding the same
        // number of elements: a group the reshape maps onto itself.
        let (old_start, new_start) = (next_old, next_new);
        let mut old_count = old[next_old].0;
        let mut new_count = new_shape[next_new];
        next_old += 1;
        next_new += 1;
        while old_count != new_count {
            if new_count < old_count {
                new_count *= new_shape[next_new];
                next_new += 1;
            } else {
                old_count *= old[next_old].0;
                next_old += 1;
            }
        }
        // The group's old axes must lie in order, one after the other.
        for axis in old_start..next_old - 1 {
            if old[axis].1 != old[axis + 1].1 * old[axis + 1].0 {
                return None;
            }
        }
        let mut step = old[next_old - 1].1;
        for axis in (new_start..next_new).rev() {
            new_strides.values[axis] = step;
            step *= new_shape[axis];
        }
    }
    // An axis of one element takes the stride a row-major layout would give
    // it, so a contiguous result still reads as contiguous.
    for axis in (0..new_shape.len()).rev() {
        if new_shape[axis] == 1 {
            new_strides.values[axis] = match axis + 1 {
                next if next < new_shape.len() => new_strides[next] * new_shape[next],
                _ => 1,
            };
        }
    }
    Some(new_strides)
}

/// The strides that read a layout of `shape` with `strides` as the larger
/// `target` it broadcasts to: an axis of `1` the target repeats, and every
/// leading axis the target adds, steps by zero. `None` unless `shape`
/// broadcasts to `target` — has no more axes, and each extent equals the
/// target's or is `1`.
pub(crate) fn broadcast_strides(
    shape: &[usize],
    strides: &[usize],
    target: &[usize],
) -> Option<Dims> {
    let added = target.len().checked_sub(shape.len())?;
    let mut out = vec![0; target.len()];
    for axis in 0..shape.len() {
        let wanted = target[added + axis];
        out[added + axis] = match shape[axis] {
            extent if extent == wanted => strides[axis],
            1 => 0,
            _ => return None,
        };
    }
    Some(Dims::new(&out, "broadcast"))
}

/// A layout split for a reduction: the axes it keeps and the ones it folds
/// together, each with its strides, from one offset. Element `[k…]` of the
/// result folds storage elements `offset + Σ kᵢ·kept_strides[i] +
/// Σ fⱼ·folded_strides[j]` over every index `[f…]` of `folded`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct Split {
    pub(crate) offset: usize,
    pub(crate) kept: Dims,
    pub(crate) kept_strides: Dims,
    pub(crate) folded: Dims,
    pub(crate) folded_strides: Dims,
}

impl Split {
    /// A layout of `shape` with `strides` from `offset`, split so that the
    /// axes in `axes` — distinct, ascending — fold and the rest are kept, in
    /// order.
    pub(crate) fn new(shape: &[usize], strides: &[usize], offset: usize, axes: &[usize]) -> Self {
        let (mut kept, mut kept_strides) = (Vec::new(), Vec::new());
        let (mut folded, mut folded_strides) = (Vec::new(), Vec::new());
        for axis in 0..shape.len() {
            if axes.contains(&axis) {
                folded.push(shape[axis]);
                folded_strides.push(strides[axis]);
            } else {
                kept.push(shape[axis]);
                kept_strides.push(strides[axis]);
            }
        }
        Split {
            offset,
            kept: Dims::new(&kept, "reduce"),
            kept_strides: Dims::new(&kept_strides, "reduce"),
            folded: Dims::new(&folded, "reduce"),
            folded_strides: Dims::new(&folded_strides, "reduce"),
        }
    }

    /// The number of results: one per index of the kept axes.
    pub(crate) fn results(&self) -> usize {
        self.kept.iter().product()
    }

    /// The number of elements each result folds together.
    pub(crate) fn depth(&self) -> usize {
        self.folded.iter().product()
    }
}

/// `shape` and the strides of `N` operands over it, simplified for a copy:
/// axes of one element dropped, and each axis merged into the next wherever
/// every operand steps across the pair as across one axis. A permutation of a
/// contiguous tensor keeps one axis per run of untouched axes, and a
/// contiguous copy becomes one axis. The shape holds at least one element.
pub(crate) fn coalesce<const N: usize>(
    shape: &[usize],
    strides: [&[usize]; N],
) -> (Dims, [Dims; N]) {
    let mut merged_shape = Dims::default();
    let mut merged = [Dims::default(); N];
    for axis in 0..shape.len() {
        let extent = shape[axis];
        if extent == 1 {
            continue;
        }
        let rank = usize::from(merged_shape.rank);
        let joins = rank > 0
            && (0..N)
                .all(|operand| merged[operand].values[rank - 1] == strides[operand][axis] * extent);
        if joins {
            merged_shape.values[rank - 1] *= extent;
            for operand in 0..N {
                merged[operand].values[rank - 1] = strides[operand][axis];
            }
        } else {
            merged_shape.values[rank] = extent;
            merged_shape.rank += 1;
            for operand in 0..N {
                merged[operand].values[rank] = strides[operand][axis];
                merged[operand].rank += 1;
            }
        }
    }
    (merged_shape, merged)
}

/// Call `run` once for each innermost run of a walk over `shape` in row-major
/// order, with `N` operands starting at `offsets` and stepping by `strides`:
/// `run(starts, len, steps)` covers `len` elements, operand `n` from
/// `starts[n]` in steps of `steps[n]`. The layout is [`coalesce`]d first, so
/// the runs are as long as the operands allow. An empty shape makes no call; a
/// scalar makes one, of one element.
pub(crate) fn for_each_run<const N: usize>(
    shape: &[usize],
    offsets: [usize; N],
    strides: [&[usize]; N],
    mut run: impl FnMut([usize; N], usize, [usize; N]),
) {
    if shape.contains(&0) {
        return;
    }
    let (shape, strides) = coalesce(shape, strides);
    let rank = shape.len();
    if rank == 0 {
        run(offsets, 1, [0; N]);
        return;
    }
    let inner = shape[rank - 1];
    let steps = std::array::from_fn(|operand| strides[operand][rank - 1]);
    let mut index = [0usize; MAX_RANK];
    let mut at = offsets;
    loop {
        run(at, inner, steps);
        // Advance the outer axes like an odometer.
        let mut axis = rank - 1;
        loop {
            if axis == 0 {
                return;
            }
            axis -= 1;
            index[axis] += 1;
            for operand in 0..N {
                at[operand] += strides[operand][axis];
            }
            if index[axis] < shape[axis] {
                break;
            }
            for operand in 0..N {
                at[operand] -= strides[operand][axis] * shape[axis];
            }
            index[axis] = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn walk(shape: &[usize], offset: usize, strides: &[usize]) -> Vec<usize> {
        let mut seen = Vec::new();
        for_each_run(shape, [offset], [strides], |[start], len, [step]| {
            seen.extend((0..len).map(|i| start + i * step));
        });
        seen
    }

    #[test]
    fn a_contiguous_layout_coalesces_to_one_axis() {
        let shape = [2, 3, 4];
        let strides = contiguous_strides(&shape);
        let (merged, [merged_strides]) = coalesce(&shape, [&strides]);
        assert_eq!(merged.as_slice(), [24]);
        assert_eq!(merged_strides.as_slice(), [1]);
        assert_eq!(walk(&shape, 5, &strides), (5..29).collect::<Vec<_>>());
    }

    #[test]
    fn a_permuted_layout_walks_in_the_permuted_order() {
        // A 2×3 matrix read transposed.
        assert_eq!(walk(&[3, 2], 0, &[1, 3]), [0, 3, 1, 4, 2, 5]);
        // Scalars and empty shapes.
        assert_eq!(walk(&[], 7, &[]), [7]);
        assert!(walk(&[3, 0, 2], 0, &[0, 0, 0]).is_empty());
    }

    #[test]
    fn reshape_strides_split_and_merge_compatible_axes() {
        // Splitting the last axis of a contiguous layout.
        let strides = reshape_strides(&[4, 6], &[6, 1], &[4, 2, 3]).unwrap();
        assert_eq!(strides.as_slice(), [6, 3, 1]);
        // Merging the leading axes of a column slice keeps its row step.
        let strides = reshape_strides(&[2, 3, 2], &[12, 4, 1], &[6, 2]).unwrap();
        assert_eq!(strides.as_slice(), [4, 1]);
        // A transpose cannot be flattened in place.
        assert!(reshape_strides(&[3, 2], &[1, 3], &[6]).is_none());
        // Axes of one element are free.
        let strides = reshape_strides(&[1, 4, 1], &[9, 2, 9], &[2, 1, 2]).unwrap();
        assert_eq!(strides.as_slice(), [4, 4, 2]);
    }

    #[test]
    fn axes_resolve_from_either_end() {
        assert_eq!(2usize.resolve(3), Some(2));
        assert_eq!(3usize.resolve(3), None);
        assert_eq!((-1isize).resolve(3), Some(2));
        assert_eq!((-3i32).resolve(3), Some(0));
        assert_eq!((-4i32).resolve(3), None);
    }

    #[test]
    fn broadcast_strides_repeat_unit_and_added_axes() {
        let strides = broadcast_strides(&[3, 1], &[1, 1], &[2, 3, 4]).unwrap();
        assert_eq!(strides.as_slice(), [0, 1, 0]);
        let strides = broadcast_strides(&[4], &[2], &[3, 4]).unwrap();
        assert_eq!(strides.as_slice(), [0, 2]);
        assert!(broadcast_strides(&[2, 4], &[4, 1], &[4]).is_none());
        assert!(broadcast_strides(&[3], &[1], &[4]).is_none());
    }

    #[test]
    fn a_split_keeps_the_unreduced_axes_in_order() {
        let split = Split::new(&[2, 3, 4, 5], &[60, 20, 5, 1], 7, &[1, 3]);
        assert_eq!(split.kept.as_slice(), [2, 4]);
        assert_eq!(split.kept_strides.as_slice(), [60, 5]);
        assert_eq!(split.folded.as_slice(), [3, 5]);
        assert_eq!(split.folded_strides.as_slice(), [20, 1]);
        assert_eq!((split.offset, split.results(), split.depth()), (7, 8, 15));
    }

    #[test]
    fn reach_is_one_past_the_last_element() {
        assert_eq!(reach(&[2, 3], 1, &[3, 1]), Some(7));
        assert_eq!(reach(&[2, 0], 100, &[3, 1]), Some(0));
        assert_eq!(reach(&[], 4, &[]), Some(5));
        assert_eq!(reach(&[2], usize::MAX, &[1]), None);
    }
}
