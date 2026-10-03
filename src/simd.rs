//! Architecture-specific SIMD kernels for the CPU math paths.
//!
//! These are the fast path for `f32` and `f64` on the [`Host`] backend, above
//! the naive generic scalar loops in [`tensors`](crate::tensors), which run one
//! scalar operation at a time and remain the definition for every other element
//! type. The Metal GPU path is reached only by choosing the `Metal` backend.
//!
//! [`Host`]: crate::tensors::Host
//!
//! On `aarch64`, NEON is part of the architecture baseline. On `x86_64`, the
//! implementation selects AVX2+FMA at runtime and retains an SSE2 baseline for
//! older processors. The kernels operate on concrete `f32`/`f64` slices; the
//! generic-to-concrete bridge (via `TypeId`) lives in
//! `tensors::simd_dispatch`, which keeps the scalar path as the correctness
//! oracle for every non-float element type (integers, `Complex`, `Dual`).
//!
//! Every public function here is safe to call with any arguments. Each checks
//! its slice lengths and matrix extents with [`assert!`] before it reads or
//! writes through a raw pointer, and panics on a mismatch rather than touching
//! memory outside the slices — in release builds too. The unchecked bodies are
//! private `unsafe` code whose contract is exactly those checks.

/// The checks the safe entry points make before their unchecked loops.
///
/// These are the safety boundary, so they are real assertions: a debug-only
/// check would let a release build read past the end of a slice.
mod contract {
    /// Panics unless every slice an operation reads or writes has one length.
    #[track_caller]
    #[inline]
    pub(super) fn same_len(operation: &str, lengths: &[usize]) {
        assert!(
            lengths.windows(2).all(|pair| pair[0] == pair[1]),
            "simd::{operation}: slice lengths differ: {lengths:?}"
        );
    }

    /// Panics unless `a` is `m×k`, `b` is `k×n` and `out` is `m×n`, with no
    /// extent product overflowing (which would let a wrapped product match a
    /// short slice).
    #[track_caller]
    #[inline]
    pub(super) fn matmul(
        operation: &str,
        a: usize,
        b: usize,
        out: usize,
        (m, k, n): (usize, usize, usize),
    ) {
        assert!(
            m.checked_mul(k) == Some(a)
                && k.checked_mul(n) == Some(b)
                && m.checked_mul(n) == Some(out),
            "simd::{operation}: slices of {a}, {b} and {out} elements do not hold \
             a {m}×{k} by {k}×{n} product"
        );
    }

    /// Panics unless `buf` interleaves `n` complex values and `n` is a power of
    /// two (or zero), which the butterfly stages index by.
    #[track_caller]
    #[inline]
    pub(super) fn fft(operation: &str, len: usize, n: usize) {
        assert!(
            (n == 0 || n.is_power_of_two()) && n.checked_mul(2) == Some(len),
            "simd::{operation}: a buffer of {len} values does not hold {n} complex \
             points, or {n} is not a power of two"
        );
    }
}

/// Runs `$body` with `$name` bound to a constant equal to `$op`, once per
/// arithmetic operation, so that the body — usually a loop — is compiled once
/// per operation with the choice folded away, rather than deciding which
/// operation it is again for every vector. `Rem` is not here: it has no vector
/// form, and the kernels handle it before they specialize.
#[cfg(target_arch = "aarch64")]
macro_rules! specialize_binary {
    ($op:expr, $name:ident => $body:expr) => {
        match $op {
            BinaryOp::Add => {
                const $name: BinaryOp = BinaryOp::Add;
                $body
            }
            BinaryOp::Sub => {
                const $name: BinaryOp = BinaryOp::Sub;
                $body
            }
            BinaryOp::Mul => {
                const $name: BinaryOp = BinaryOp::Mul;
                $body
            }
            BinaryOp::Div => {
                const $name: BinaryOp = BinaryOp::Div;
                $body
            }
            BinaryOp::Rem => unreachable!("remainder has no vector form"),
        }
    };
}

/// The same for the comparisons.
#[cfg(target_arch = "aarch64")]
macro_rules! specialize_compare {
    ($op:expr, $name:ident => $body:expr) => {
        match $op {
            Compare::Min => {
                const $name: Compare = Compare::Min;
                $body
            }
            Compare::Max => {
                const $name: Compare = Compare::Max;
                $body
            }
            Compare::MaxShare => {
                const $name: Compare = Compare::MaxShare;
                $body
            }
            Compare::Less => {
                const $name: Compare = Compare::Less;
                $body
            }
            Compare::LessEqual => {
                const $name: Compare = Compare::LessEqual;
                $body
            }
            Compare::Greater => {
                const $name: Compare = Compare::Greater;
                $body
            }
            Compare::GreaterEqual => {
                const $name: Compare = Compare::GreaterEqual;
                $body
            }
        }
    };
}

/// Generates the elementwise / reduction / matmul kernels for one float type.
///
/// The four floating intrinsics differ only in name between `f32` (4-lane
/// `float32x4_t`) and `f64` (2-lane `float64x2_t`), so the bodies are shared and
/// the intrinsic set is passed in.
#[cfg(target_arch = "aarch64")]
macro_rules! neon_kernels {
    (
        mod $modname:ident, ty = $t:ty, vec = $v:ty, lanes = $lanes:expr,
        load = $load:ident, store = $store:ident, dup = $dup:ident,
        add = $add:ident, sub = $sub:ident, mul = $mul:ident, div = $div:ident,
        fma = $fma:ident, addv = $addv:ident, sqrt = $sqrt:ident,
        min = $min:ident, max = $max:ident, bsl = $bsl:ident,
        cgt = $cgt:ident, clt = $clt:ident, cge = $cge:ident, cle = $cle:ident
    ) => {
        pub mod $modname {
            use crate::simd::contract;
            use crate::tensors::{BinaryOp, Compare, Reduce};
            use core::arch::aarch64::*;

            /// Lanes per NEON register for this element type.
            pub const LANES: usize = $lanes;

            /// Dot product of two equal-length slices.
            ///
            /// Four independent accumulators keep several FMAs in flight so the
            /// (multi-cycle) FMA latency is hidden, then a single horizontal add
            /// combines them. The naive scalar reduction cannot do this because
            /// its `sum` carries a serial dependency across every element.
            #[inline]
            pub fn dot(a: &[$t], b: &[$t]) -> $t {
                contract::same_len("dot", &[a.len(), b.len()]);
                let n = a.len();
                let mut i = 0;
                let mut total: $t;
                // SAFETY: `a` and `b` were checked to share `n`, and every
                // vector access covers `LANES` elements that end at or before
                // `n`.
                unsafe {
                    let mut acc = [$dup(0 as $t); 4];
                    while i + 4 * LANES <= n {
                        let mut k = 0;
                        while k < 4 {
                            let off = i + k * LANES;
                            let va = $load(a.as_ptr().add(off));
                            let vb = $load(b.as_ptr().add(off));
                            acc[k] = $fma(acc[k], va, vb);
                            k += 1;
                        }
                        i += 4 * LANES;
                    }
                    let partial = $add($add(acc[0], acc[1]), $add(acc[2], acc[3]));
                    total = $addv(partial);
                    while i + LANES <= n {
                        let va = $load(a.as_ptr().add(i));
                        let vb = $load(b.as_ptr().add(i));
                        total += $addv($mul(va, vb));
                        i += LANES;
                    }
                }
                while i < n {
                    total += a[i] * b[i];
                    i += 1;
                }
                total
            }

            /// One pass of `vector` over `n` values at `x`, written to `o`, four
            /// vectors per iteration so several are in flight, and `scalar` over
            /// the tail.
            ///
            /// Every vector is loaded before the one result that replaces it is
            /// stored, so `o` may equal `x`: that is the in-place form. Always
            /// inlined, so each caller's closures — one operation, chosen before
            /// the loop — compile into a loop of their own.
            ///
            /// # Safety
            ///
            /// `x` must be valid for reads and `o` for writes of `n` elements,
            /// and the two must be equal or not overlap.
            #[inline(always)]
            unsafe fn map1(
                x: *const $t,
                o: *mut $t,
                n: usize,
                vector: impl Fn($v) -> $v,
                scalar: impl Fn($t) -> $t,
            ) {
                let mut i = 0;
                // SAFETY: every vector access covers `LANES` elements that end
                // at or before `n`.
                unsafe {
                    while i + 4 * LANES <= n {
                        let v0 = $load(x.add(i));
                        let v1 = $load(x.add(i + LANES));
                        let v2 = $load(x.add(i + 2 * LANES));
                        let v3 = $load(x.add(i + 3 * LANES));
                        $store(o.add(i), vector(v0));
                        $store(o.add(i + LANES), vector(v1));
                        $store(o.add(i + 2 * LANES), vector(v2));
                        $store(o.add(i + 3 * LANES), vector(v3));
                        i += 4 * LANES;
                    }
                    while i + LANES <= n {
                        $store(o.add(i), vector($load(x.add(i))));
                        i += LANES;
                    }
                    while i < n {
                        *o.add(i) = scalar(*x.add(i));
                        i += 1;
                    }
                }
            }

            /// [`map1`] over two operands. `o` may equal `x`, but not `y`.
            ///
            /// # Safety
            ///
            /// As for [`map1`], with `y` valid for reads of `n` elements and
            /// not overlapping `o`.
            #[inline(always)]
            unsafe fn map2(
                x: *const $t,
                y: *const $t,
                o: *mut $t,
                n: usize,
                vector: impl Fn($v, $v) -> $v,
                scalar: impl Fn($t, $t) -> $t,
            ) {
                let mut i = 0;
                // SAFETY: as in `map1`.
                unsafe {
                    while i + 4 * LANES <= n {
                        let a0 = $load(x.add(i));
                        let a1 = $load(x.add(i + LANES));
                        let a2 = $load(x.add(i + 2 * LANES));
                        let a3 = $load(x.add(i + 3 * LANES));
                        let b0 = $load(y.add(i));
                        let b1 = $load(y.add(i + LANES));
                        let b2 = $load(y.add(i + 2 * LANES));
                        let b3 = $load(y.add(i + 3 * LANES));
                        $store(o.add(i), vector(a0, b0));
                        $store(o.add(i + LANES), vector(a1, b1));
                        $store(o.add(i + 2 * LANES), vector(a2, b2));
                        $store(o.add(i + 3 * LANES), vector(a3, b3));
                        i += 4 * LANES;
                    }
                    while i + LANES <= n {
                        $store(o.add(i), vector($load(x.add(i)), $load(y.add(i))));
                        i += LANES;
                    }
                    while i < n {
                        *o.add(i) = scalar(*x.add(i), *y.add(i));
                        i += 1;
                    }
                }
            }

            /// [`map1`] of `f(scalar, x)` or `f(x, scalar)`, as `scalar_left`
            /// says.
            ///
            /// # Safety
            ///
            /// As for [`map1`].
            #[inline(always)]
            unsafe fn map_scalar(
                x: *const $t,
                o: *mut $t,
                n: usize,
                scalar: $t,
                scalar_left: bool,
                vector: impl Fn($v, $v) -> $v,
                lane: impl Fn($t, $t) -> $t,
            ) {
                // SAFETY: NEON is part of the aarch64 baseline.
                let splat = unsafe { $dup(scalar) };
                unsafe {
                    if scalar_left {
                        map1(x, o, n, |x| vector(splat, x), |x| lane(scalar, x));
                    } else {
                        map1(x, o, n, |x| vector(x, splat), |x| lane(x, scalar));
                    }
                }
            }

            /// `op` on two vectors. Called with a constant `op`, it inlines to
            /// the one instruction.
            #[inline(always)]
            fn binary_vectors(op: BinaryOp, a: $v, b: $v) -> $v {
                // SAFETY: NEON is part of the aarch64 baseline, and these
                // intrinsics only touch registers.
                unsafe {
                    match op {
                        BinaryOp::Add => $add(a, b),
                        BinaryOp::Sub => $sub(a, b),
                        BinaryOp::Mul => $mul(a, b),
                        BinaryOp::Div => $div(a, b),
                        BinaryOp::Rem => unreachable!("remainder has no vector form"),
                    }
                }
            }

            #[inline(always)]
            fn binary_values(op: BinaryOp, a: $t, b: $t) -> $t {
                match op {
                    BinaryOp::Add => a + b,
                    BinaryOp::Sub => a - b,
                    BinaryOp::Mul => a * b,
                    BinaryOp::Div => a / b,
                    BinaryOp::Rem => a % b,
                }
            }

            /// Elementwise binary operation. Remainder has no NEON
            /// floating-point instruction, so it uses the scalar loop.
            #[inline]
            pub fn elementwise(a: &[$t], b: &[$t], op: BinaryOp, out: &mut [$t]) {
                let n = a.len();
                contract::same_len("elementwise", &[n, b.len(), out.len()]);
                // SAFETY: the lengths agree, and `out` is a distinct borrow.
                unsafe { elementwise_raw(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), n, op) }
            }

            /// [`elementwise`] into the left operand: `a = a op b`.
            #[inline]
            pub fn elementwise_assign(a: &mut [$t], b: &[$t], op: BinaryOp) {
                let n = a.len();
                contract::same_len("elementwise_assign", &[n, b.len()]);
                let a = a.as_mut_ptr();
                // SAFETY: the lengths agree, `b` is a distinct borrow, and
                // reading and writing `a` at the same index is what `map2`
                // allows.
                unsafe { elementwise_raw(a, b.as_ptr(), a, n, op) }
            }

            /// # Safety
            ///
            /// As for [`map2`].
            #[inline(always)]
            unsafe fn elementwise_raw(
                a: *const $t,
                b: *const $t,
                out: *mut $t,
                n: usize,
                op: BinaryOp,
            ) {
                unsafe {
                    if op == BinaryOp::Rem {
                        for i in 0..n {
                            *out.add(i) = *a.add(i) % *b.add(i);
                        }
                        return;
                    }
                    specialize_binary!(op, OP => map2(
                        a,
                        b,
                        out,
                        n,
                        |x, y| binary_vectors(OP, x, y),
                        |x, y| binary_values(OP, x, y),
                    ))
                }
            }

            /// Tensor/scalar broadcast. `scalar_left` selects operand order for
            /// the non-commutative ops (`scalar - x` vs `x - scalar`, etc.).
            #[inline]
            pub fn broadcast(
                values: &[$t],
                scalar: $t,
                op: BinaryOp,
                scalar_left: bool,
                out: &mut [$t],
            ) {
                let n = values.len();
                contract::same_len("broadcast", &[n, out.len()]);
                // SAFETY: the lengths agree, and `out` is a distinct borrow.
                unsafe {
                    broadcast_raw(values.as_ptr(), out.as_mut_ptr(), n, scalar, op, scalar_left)
                }
            }

            /// [`broadcast`] in place.
            #[inline]
            pub fn broadcast_assign(values: &mut [$t], scalar: $t, op: BinaryOp, scalar_left: bool) {
                let (ptr, n) = (values.as_mut_ptr(), values.len());
                // SAFETY: reading and writing at the same index is what `map1`
                // allows.
                unsafe { broadcast_raw(ptr, ptr, n, scalar, op, scalar_left) }
            }

            /// # Safety
            ///
            /// As for [`map1`].
            #[inline(always)]
            unsafe fn broadcast_raw(
                x: *const $t,
                o: *mut $t,
                n: usize,
                scalar: $t,
                op: BinaryOp,
                scalar_left: bool,
            ) {
                unsafe {
                    if op == BinaryOp::Rem {
                        for i in 0..n {
                            *o.add(i) = if scalar_left {
                                scalar % *x.add(i)
                            } else {
                                *x.add(i) % scalar
                            };
                        }
                        return;
                    }
                    specialize_binary!(op, OP => map_scalar(
                        x,
                        o,
                        n,
                        scalar,
                        scalar_left,
                        |x, y| binary_vectors(OP, x, y),
                        |x, y| binary_values(OP, x, y),
                    ))
                }
            }

            /// The scalar definition of every comparison, for the ragged tail —
            /// and the oracle the vector arms below have to agree with.
            ///
            /// `Min`/`Max` are `fminnm`/`fmaxnm`, the IEEE `minNum`/`maxNum`
            /// that let a number beat a NaN, which is exactly what the
            /// `f32::min` in the scalar path does. Plain `fmin`/`fmax` (the
            /// `vminq`/`vmaxq` intrinsics) propagate the NaN instead, so they
            /// would disagree with it.
            #[inline(always)]
            fn compare_values(op: Compare, a: $t, b: $t) -> $t {
                match op {
                    Compare::Min => a.min(b),
                    Compare::Max => a.max(b),
                    Compare::MaxShare => match a.partial_cmp(&b) {
                        Some(core::cmp::Ordering::Greater) => 1.0,
                        Some(core::cmp::Ordering::Less) => 0.0,
                        _ => 0.5,
                    },
                    Compare::Less => {
                        if a < b {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    Compare::LessEqual => {
                        if a <= b {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    Compare::Greater => {
                        if a > b {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    Compare::GreaterEqual => {
                        if a >= b {
                            1.0
                        } else {
                            0.0
                        }
                    }
                }
            }

            /// One vector-wide comparison. The predicates turn a lane mask into
            /// `1.0`/`0.0` with a bitwise select, which is branchless and needs
            /// no conversion instruction.
            #[inline(always)]
            unsafe fn compare_vectors(op: Compare, a: $v, b: $v) -> $v {
                // SAFETY: NEON is part of the aarch64 baseline, and these
                // intrinsics only touch registers.
                unsafe {
                    let one = $dup(1.0);
                    let zero = $dup(0.0);
                    match op {
                        Compare::Min => $min(a, b),
                        Compare::Max => $max(a, b),
                        // Ordered greater / ordered less, so an unordered pair
                        // (either operand NaN) falls through to the tie value.
                        Compare::MaxShare => {
                            let tie = $bsl($clt(a, b), zero, $dup(0.5));
                            $bsl($cgt(a, b), one, tie)
                        }
                        Compare::Less => $bsl($clt(a, b), one, zero),
                        Compare::LessEqual => $bsl($cle(a, b), one, zero),
                        Compare::Greater => $bsl($cgt(a, b), one, zero),
                        Compare::GreaterEqual => $bsl($cge(a, b), one, zero),
                    }
                }
            }

            /// Elementwise comparison of two slices.
            #[inline]
            pub fn compare(a: &[$t], b: &[$t], op: Compare, out: &mut [$t]) {
                let n = a.len();
                contract::same_len("compare", &[n, b.len(), out.len()]);
                // SAFETY: the lengths agree, and `out` is a distinct borrow.
                unsafe { compare_raw(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), n, op) }
            }

            /// [`compare`] into the left operand.
            #[inline]
            pub fn compare_assign(a: &mut [$t], b: &[$t], op: Compare) {
                let n = a.len();
                contract::same_len("compare_assign", &[n, b.len()]);
                let a = a.as_mut_ptr();
                // SAFETY: as in `elementwise_assign`.
                unsafe { compare_raw(a, b.as_ptr(), a, n, op) }
            }

            /// # Safety
            ///
            /// As for [`map2`].
            #[inline(always)]
            unsafe fn compare_raw(a: *const $t, b: *const $t, out: *mut $t, n: usize, op: Compare) {
                unsafe {
                    specialize_compare!(op, OP => map2(
                        a,
                        b,
                        out,
                        n,
                        // SAFETY: NEON is part of the aarch64 baseline.
                        |x, y| compare_vectors(OP, x, y),
                        |x, y| compare_values(OP, x, y),
                    ))
                }
            }

            /// Comparison against a splatted scalar. `scalar_left` selects the
            /// operand order, which matters for every op but `Min` and `Max`.
            #[inline]
            pub fn compare_scalar(
                values: &[$t],
                scalar: $t,
                op: Compare,
                scalar_left: bool,
                out: &mut [$t],
            ) {
                let n = values.len();
                contract::same_len("compare_scalar", &[n, out.len()]);
                // SAFETY: the lengths agree, and `out` is a distinct borrow.
                unsafe {
                    compare_scalar_raw(values.as_ptr(), out.as_mut_ptr(), n, scalar, op, scalar_left)
                }
            }

            /// [`compare_scalar`] in place.
            #[inline]
            pub fn compare_scalar_assign(values: &mut [$t], scalar: $t, op: Compare, scalar_left: bool) {
                let (ptr, n) = (values.as_mut_ptr(), values.len());
                // SAFETY: as in `broadcast_assign`.
                unsafe { compare_scalar_raw(ptr, ptr, n, scalar, op, scalar_left) }
            }

            /// # Safety
            ///
            /// As for [`map1`].
            #[inline(always)]
            unsafe fn compare_scalar_raw(
                x: *const $t,
                o: *mut $t,
                n: usize,
                scalar: $t,
                op: Compare,
                scalar_left: bool,
            ) {
                unsafe {
                    specialize_compare!(op, OP => map_scalar(
                        x,
                        o,
                        n,
                        scalar,
                        scalar_left,
                        // SAFETY: NEON is part of the aarch64 baseline.
                        |x, y| compare_vectors(OP, x, y),
                        |x, y| compare_values(OP, x, y),
                    ))
                }
            }

            /// Elementwise square root. `fsqrt` is correctly rounded, as IEEE
            /// requires of a square root, so every lane agrees with the scalar
            /// `sqrt` bit for bit — including `−0.0`, which stays `−0.0`.
            #[inline]
            pub fn sqrt(values: &[$t], out: &mut [$t]) {
                let n = values.len();
                contract::same_len("sqrt", &[n, out.len()]);
                let mut i = 0;
                // SAFETY: the slice lengths were checked on entry, and every
                // vector access covers `LANES` elements that end at or before
                // `n`.
                unsafe {
                    while i + LANES <= n {
                        let vx = $load(values.as_ptr().add(i));
                        $store(out.as_mut_ptr().add(i), $sqrt(vx));
                        i += LANES;
                    }
                }
                while i < n {
                    out[i] = values[i].sqrt();
                    i += 1;
                }
            }

            /// Confine every element to `[low, high]`, in one pass.
            ///
            /// Two `compare_scalar` calls would stream the data twice; the pair
            /// of instructions here has no reason to.
            #[inline]
            pub fn clamp(values: &[$t], low: $t, high: $t, out: &mut [$t]) {
                let n = values.len();
                contract::same_len("clamp", &[n, out.len()]);
                // SAFETY: the lengths agree, and `out` is a distinct borrow.
                unsafe { clamp_raw(values.as_ptr(), out.as_mut_ptr(), n, low, high) }
            }

            /// [`clamp`] in place.
            #[inline]
            pub fn clamp_assign(values: &mut [$t], low: $t, high: $t) {
                let (ptr, n) = (values.as_mut_ptr(), values.len());
                // SAFETY: as in `broadcast_assign`.
                unsafe { clamp_raw(ptr, ptr, n, low, high) }
            }

            /// # Safety
            ///
            /// As for [`map1`].
            #[inline(always)]
            unsafe fn clamp_raw(x: *const $t, o: *mut $t, n: usize, low: $t, high: $t) {
                // SAFETY: every vector access covers `LANES` elements that end
                // at or before `n`.
                unsafe {
                    let (vlow, vhigh) = ($dup(low), $dup(high));
                    let mut i = 0;
                    while i + LANES <= n {
                        $store(o.add(i), $min($max($load(x.add(i)), vlow), vhigh));
                        i += LANES;
                    }
                    while i < n {
                        *o.add(i) = (*x.add(i)).max(low).min(high);
                        i += 1;
                    }
                }
            }

            #[inline]
            fn reduce_values(op: Reduce, a: $t, b: $t) -> $t {
                match op {
                    Reduce::Sum => a + b,
                    Reduce::Min => a.min(b),
                    Reduce::Max => a.max(b),
                }
            }

            fn identity(op: Reduce) -> $t {
                match op {
                    Reduce::Sum => 0.0,
                    Reduce::Min => <$t>::INFINITY,
                    Reduce::Max => <$t>::NEG_INFINITY,
                }
            }

            /// Fold a whole slice to one value.
            ///
            /// Four accumulators again, for the same reason [`dot`] keeps them:
            /// a single running total serializes the fold on the latency of one
            /// add. The lanes are combined at the end, so `Sum` associates
            /// differently from the scalar loop and may land a rounding step
            /// away from it — `Min` and `Max` are exact.
            #[inline]
            pub fn reduce(values: &[$t], op: Reduce) -> $t {
                let n = values.len();
                let mut i = 0;
                let mut total = identity(op);
                // SAFETY: every vector access covers `LANES` elements of
                // `values` that end at or before `n`.
                unsafe {
                    let mut acc = [$dup(identity(op)); 4];
                    while i + 4 * LANES <= n {
                        let mut k = 0;
                        while k < 4 {
                            let v = $load(values.as_ptr().add(i + k * LANES));
                            acc[k] = match op {
                                Reduce::Sum => $add(acc[k], v),
                                Reduce::Min => $min(acc[k], v),
                                Reduce::Max => $max(acc[k], v),
                            };
                            k += 1;
                        }
                        i += 4 * LANES;
                    }
                    while i + LANES <= n {
                        let v = $load(values.as_ptr().add(i));
                        acc[0] = match op {
                            Reduce::Sum => $add(acc[0], v),
                            Reduce::Min => $min(acc[0], v),
                            Reduce::Max => $max(acc[0], v),
                        };
                        i += LANES;
                    }
                    let mut lanes = [0 as $t; LANES];
                    for a in acc {
                        $store(lanes.as_mut_ptr(), a);
                        for lane in lanes {
                            total = reduce_values(op, total, lane);
                        }
                    }
                }
                while i < n {
                    total = reduce_values(op, total, values[i]);
                    i += 1;
                }
                total
            }

            /// `Σ(xᵢ − mean)²` — the second pass of a two-pass variance.
            ///
            /// The subtraction and the square fuse into one FMA per register,
            /// so this costs barely more than the plain sum that produced the
            /// mean. Four accumulators again, for the reason [`dot`] keeps
            /// them.
            #[inline]
            pub fn sum_squared_deviations(values: &[$t], mean: $t) -> $t {
                let n = values.len();
                let mut i = 0;
                let mut total: $t;
                // SAFETY: every vector access covers `LANES` elements of
                // `values` that end at or before `n`.
                unsafe {
                    let center = $dup(mean);
                    let mut acc = [$dup(0 as $t); 4];
                    while i + 4 * LANES <= n {
                        let mut k = 0;
                        while k < 4 {
                            let d = $sub($load(values.as_ptr().add(i + k * LANES)), center);
                            acc[k] = $fma(acc[k], d, d);
                            k += 1;
                        }
                        i += 4 * LANES;
                    }
                    let partial = $add($add(acc[0], acc[1]), $add(acc[2], acc[3]));
                    total = $addv(partial);
                    while i + LANES <= n {
                        let d = $sub($load(values.as_ptr().add(i)), center);
                        total += $addv($mul(d, d));
                        i += LANES;
                    }
                }
                while i < n {
                    let d = values[i] - mean;
                    total += d * d;
                    i += 1;
                }
                total
            }

            /// `totals += values`, elementwise and in place.
            ///
            /// This is what makes a column-wise reduction read the matrix in
            /// storage order: one row of partial sums at a time, with unit
            /// stride on both sides, instead of striding down each column in
            /// turn.
            #[inline]
            pub fn accumulate(totals: &mut [$t], values: &[$t]) {
                let n = totals.len();
                contract::same_len("accumulate", &[n, values.len()]);
                let mut i = 0;
                // SAFETY: the slice lengths were checked on entry, and every
                // vector access covers `LANES` elements that end at or before
                // `n`.
                unsafe {
                    while i + LANES <= n {
                        let t = $load(totals.as_ptr().add(i));
                        let v = $load(values.as_ptr().add(i));
                        $store(totals.as_mut_ptr().add(i), $add(t, v));
                        i += LANES;
                    }
                }
                while i < n {
                    totals[i] += values[i];
                    i += 1;
                }
            }

            /// `totals += (values − means)²`, elementwise and in place — the
            /// column-wise counterpart of [`sum_squared_deviations`].
            #[inline]
            pub fn accumulate_squared_deviations(totals: &mut [$t], values: &[$t], means: &[$t]) {
                let n = totals.len();
                contract::same_len(
                    "accumulate_squared_deviations",
                    &[n, values.len(), means.len()],
                );
                let mut i = 0;
                // SAFETY: the slice lengths were checked on entry, and every
                // vector access covers `LANES` elements that end at or before
                // `n`.
                unsafe {
                    while i + LANES <= n {
                        let d = $sub($load(values.as_ptr().add(i)), $load(means.as_ptr().add(i)));
                        let t = $load(totals.as_ptr().add(i));
                        $store(totals.as_mut_ptr().add(i), $fma(t, d, d));
                        i += LANES;
                    }
                }
                while i < n {
                    let d = values[i] - means[i];
                    totals[i] += d * d;
                    i += 1;
                }
            }

            /// Row-major matrix multiply: `a` is `m×k`, `b` is `k×n`, `out` is
            /// `m×n` (must be pre-sized; it is overwritten).
            ///
            /// The loop order is `i, p, j` with `j` innermost — the *broadcast-A,
            /// stream-B-rows* microkernel. `a[i][p]` is splat across a register
            /// and fused-multiply-added into a contiguous run of the output row
            /// using a contiguous run of `b`'s `p`-th row. Both `b` and `out` are
            /// walked with unit stride, which is what makes the vectorization (and
            /// the cache behaviour) pay off; the textbook `i, j, p` order strides
            /// down a column of `b` and defeats both.
            #[inline]
            pub fn matmul(a: &[$t], b: &[$t], m: usize, k: usize, n: usize, out: &mut [$t]) {
                contract::matmul("matmul", a.len(), b.len(), out.len(), (m, k, n));
                for v in out.iter_mut() {
                    *v = 0 as $t;
                }
                matmul_accumulate(a, b, m, k, n, out);
            }

            /// Row-major fused matrix multiply-add: `out = addend + a·b`.
            ///
            /// Initializing the accumulator from `addend` folds the addition
            /// into the same NEON FMA loop as the product.
            #[inline]
            pub fn matmul_add(
                a: &[$t],
                b: &[$t],
                addend: &[$t],
                m: usize,
                k: usize,
                n: usize,
                out: &mut [$t],
            ) {
                contract::matmul("matmul_add", a.len(), b.len(), out.len(), (m, k, n));
                contract::same_len("matmul_add", &[out.len(), addend.len()]);
                out.copy_from_slice(addend);
                matmul_accumulate(a, b, m, k, n, out);
            }

            /// Row-major `out += a·b`, the loop [`matmul`] and [`matmul_add`]
            /// share once they have initialized `out`.
            #[inline]
            pub fn matmul_accumulate(
                a: &[$t],
                b: &[$t],
                m: usize,
                k: usize,
                n: usize,
                out: &mut [$t],
            ) {
                contract::matmul("matmul_accumulate", a.len(), b.len(), out.len(), (m, k, n));
                // SAFETY: the extents were checked against the slice lengths
                // above, so every row offset plus a full or partial vector of
                // columns stays inside `a`, `b` and `out`.
                unsafe {
                    for i in 0..m {
                        let arow = i * k;
                        let orow = i * n;
                        for p in 0..k {
                            let aip = *a.get_unchecked(arow + p);
                            let va = $dup(aip);
                            let brow = p * n;
                            let mut j = 0;
                            while j + LANES <= n {
                                let vb = $load(b.as_ptr().add(brow + j));
                                let vo = $load(out.as_ptr().add(orow + j));
                                $store(out.as_mut_ptr().add(orow + j), $fma(vo, va, vb));
                                j += LANES;
                            }
                            while j < n {
                                *out.get_unchecked_mut(orow + j) +=
                                    aip * *b.get_unchecked(brow + j);
                                j += 1;
                            }
                        }
                    }
                }
            }
        }
    };
}

#[cfg(target_arch = "aarch64")]
neon_kernels! {
    mod f32k, ty = f32, vec = float32x4_t, lanes = 4,
    load = vld1q_f32, store = vst1q_f32, dup = vdupq_n_f32,
    add = vaddq_f32, sub = vsubq_f32, mul = vmulq_f32, div = vdivq_f32,
    fma = vfmaq_f32, addv = vaddvq_f32, sqrt = vsqrtq_f32,
    min = vminnmq_f32, max = vmaxnmq_f32, bsl = vbslq_f32,
    cgt = vcgtq_f32, clt = vcltq_f32, cge = vcgeq_f32, cle = vcleq_f32
}

#[cfg(target_arch = "aarch64")]
neon_kernels! {
    mod f64k, ty = f64, vec = float64x2_t, lanes = 2,
    load = vld1q_f64, store = vst1q_f64, dup = vdupq_n_f64,
    add = vaddq_f64, sub = vsubq_f64, mul = vmulq_f64, div = vdivq_f64,
    fma = vfmaq_f64, addv = vaddvq_f64, sqrt = vsqrtq_f64,
    min = vminnmq_f64, max = vmaxnmq_f64, bsl = vbslq_f64,
    cgt = vcgtq_f64, clt = vcltq_f64, cge = vcgeq_f64, cle = vcleq_f64
}

/// Radix-2 Cooley–Tukey FFT over an interleaved `[re, im, re, im, …]` buffer of
/// `n` complex numbers (`buf.len() == 2 * n`, `n` a power of two).
///
/// `f32`-only, mirroring the Metal path. The butterfly stages are vectorized
/// four complex points at a time with `vld2q`/`vst2q`, which deinterleave the
/// real and imaginary lanes in a single instruction — the layout NEON wants for
/// the complex multiply. Per-stage twiddles are precomputed once (as split
/// real/imag arrays) rather than iterated with a running complex product, so the
/// vector lanes stay independent. Stages too short to fill a vector (len 2 and 4)
/// and any ragged tail fall back to scalar butterflies.
#[cfg(target_arch = "aarch64")]
pub mod fft_f32 {
    use core::arch::aarch64::*;

    /// `direction` is `-1.0` for the forward transform, `+1.0` for the inverse
    /// (matching `radix2_fft`'s sign convention; normalization is the caller's
    /// job, exactly as in the scalar path).
    pub fn radix2(buf: &mut [f32], n: usize, direction: f32) {
        super::contract::fft("radix2", buf.len(), n);
        if n <= 1 {
            return;
        }

        // Bit-reversal permutation on complex indices (swap interleaved pairs).
        let mut j = 0usize;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if i < j {
                buf.swap(2 * i, 2 * j);
                buf.swap(2 * i + 1, 2 * j + 1);
            }
        }

        let tau = std::f32::consts::TAU;
        // Twiddle scratch for the widest stage, allocated once. Each stage uses
        // the leading `half` entries. Allocating per stage instead costs one
        // pair of allocations per stage — twenty for a 1024-point transform.
        let mut tw_re_buf = vec![0.0f32; n / 2];
        let mut tw_im_buf = vec![0.0f32; n / 2];
        let mut len = 2usize;
        while len <= n {
            let half = len / 2;
            let angle = direction * tau / len as f32;
            let step = (angle.cos(), angle.sin());

            // Precompute this stage's twiddles: tw[o] = exp(i·angle·o).
            let tw_re = &mut tw_re_buf[..half];
            let tw_im = &mut tw_im_buf[..half];
            let (mut wr, mut wi) = (1.0f32, 0.0f32);
            for o in 0..half {
                tw_re[o] = wr;
                tw_im[o] = wi;
                let nr = wr * step.0 - wi * step.1;
                let ni = wr * step.1 + wi * step.0;
                wr = nr;
                wi = ni;
            }

            let mut start = 0;
            while start < n {
                let mut o = 0;
                // SAFETY: `n` is a power of two with `buf.len() == 2 * n`
                // (checked on entry), so each block's `start + o + half + 4`
                // points stay in `buf`, and `o + 4 <= half` keeps the twiddle
                // loads in range.
                unsafe {
                    while o + 4 <= half {
                        let eptr = buf.as_ptr().add(2 * (start + o));
                        let optr = buf.as_ptr().add(2 * (start + o + half));
                        let even = vld2q_f32(eptr); // .0 = re, .1 = im
                        let odd = vld2q_f32(optr);
                        let twr = vld1q_f32(tw_re.as_ptr().add(o));
                        let twi = vld1q_f32(tw_im.as_ptr().add(o));

                        // p = odd * tw  (complex)
                        let pr = vsubq_f32(vmulq_f32(odd.0, twr), vmulq_f32(odd.1, twi));
                        let pi = vaddq_f32(vmulq_f32(odd.0, twi), vmulq_f32(odd.1, twr));

                        let sum = float32x4x2_t(vaddq_f32(even.0, pr), vaddq_f32(even.1, pi));
                        let dif = float32x4x2_t(vsubq_f32(even.0, pr), vsubq_f32(even.1, pi));
                        vst2q_f32(buf.as_mut_ptr().add(2 * (start + o)), sum);
                        vst2q_f32(buf.as_mut_ptr().add(2 * (start + o + half)), dif);
                        o += 4;
                    }
                }
                // Scalar tail (and the whole of len 2 / len 4 stages).
                while o < half {
                    let ei = 2 * (start + o);
                    let oi = 2 * (start + o + half);
                    let (er, eim) = (buf[ei], buf[ei + 1]);
                    let (or_, oim) = (buf[oi], buf[oi + 1]);
                    let pr = or_ * tw_re[o] - oim * tw_im[o];
                    let pi = or_ * tw_im[o] + oim * tw_re[o];
                    buf[ei] = er + pr;
                    buf[ei + 1] = eim + pi;
                    buf[oi] = er - pr;
                    buf[oi + 1] = eim - pi;
                    o += 1;
                }
                start += len;
            }
            len *= 2;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[path = "simd_x86.rs"]
mod x86;
#[cfg(target_arch = "x86_64")]
pub use x86::{f32k, f64k, fft_f32};
