//! NaN-aware `min` and `max` shared by the host vector and matrix paths.

use std::cmp::Ordering;

/// The smaller of two values, ordering NaN the way [`f32::min`] does: an
/// unordered pair keeps whichever operand is not NaN.
///
/// The vector kernels use `fminnm`, which is that same rule in hardware, so the
/// two paths agree on every input rather than only on the ordered ones.
pub(crate) fn ordered_min<T: PartialOrd + Copy>(a: T, b: T) -> T {
    match a.partial_cmp(&b) {
        Some(Ordering::Greater) => b,
        Some(_) => a,
        // Unordered: a value that does not compare with itself is the NaN, so
        // the other operand wins.
        None if a.partial_cmp(&a).is_none() => b,
        None => a,
    }
}

/// The larger of two values; see [`ordered_min`].
pub(crate) fn ordered_max<T: PartialOrd + Copy>(a: T, b: T) -> T {
    match a.partial_cmp(&b) {
        Some(Ordering::Less) => b,
        Some(_) => a,
        None if a.partial_cmp(&a).is_none() => b,
        None => a,
    }
}
