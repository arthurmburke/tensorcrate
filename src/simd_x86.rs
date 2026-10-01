//! x86_64 SIMD kernels.
//!
//! Public entry points perform runtime selection once per operation. AVX2 with
//! FMA is used when available; SSE2 is the x86_64 baseline and keeps the SIMD
//! feature useful on older processors.
//!
//! The public functions are safe: each checks its slice lengths (and matrix
//! extents) before choosing an implementation. The `unsafe fn` bodies behind
//! them are unchecked and share one contract, stated on each as `# Safety`:
//! the CPU supports the function's `target_feature`, and the slices have the
//! lengths the safe entry point verified.

#![allow(unsafe_op_in_unsafe_fn)]

macro_rules! x86_kernels {
    (
        mod $modname:ident, ty = $t:ty,
        avx_vec = $avx_vec:ty, avx_lanes = $avx_lanes:expr,
        avx_load = $avx_load:ident, avx_store = $avx_store:ident,
        avx_set1 = $avx_set1:ident, avx_zero = $avx_zero:ident,
        avx_add = $avx_add:ident, avx_sub = $avx_sub:ident,
        avx_mul = $avx_mul:ident, avx_div = $avx_div:ident,
        avx_fma = $avx_fma:ident, avx_sqrt = $avx_sqrt:ident,
        avx_min = $avx_min:ident, avx_max = $avx_max:ident,
        avx_and = $avx_and:ident, avx_andnot = $avx_andnot:ident, avx_or = $avx_or:ident,
        avx_cmp = $avx_cmp:ident,
        sse_vec = $sse_vec:ty, sse_lanes = $sse_lanes:expr,
        sse_load = $sse_load:ident, sse_store = $sse_store:ident,
        sse_set1 = $sse_set1:ident, sse_zero = $sse_zero:ident,
        sse_add = $sse_add:ident, sse_sub = $sse_sub:ident,
        sse_mul = $sse_mul:ident, sse_div = $sse_div:ident, sse_sqrt = $sse_sqrt:ident,
        sse_min = $sse_min:ident, sse_max = $sse_max:ident,
        sse_and = $sse_and:ident, sse_andnot = $sse_andnot:ident, sse_or = $sse_or:ident,
        sse_cmplt = $sse_cmplt:ident, sse_cmple = $sse_cmple:ident,
        sse_cmpunord = $sse_cmpunord:ident
    ) => {
        pub mod $modname {
            use core::arch::x86_64::*;
            use std::arch::is_x86_feature_detected;

            use crate::simd::contract;
            use crate::tensors::{BinaryOp, Compare, Reduce};

            /// Lanes used by the preferred AVX2 implementation.
            pub const LANES: usize = $avx_lanes;

            #[inline]
            fn has_avx2_fma() -> bool {
                is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
            }

            #[inline]
            fn has_avx2() -> bool {
                is_x86_feature_detected!("avx2")
            }

            #[inline]
            pub fn dot(a: &[$t], b: &[$t]) -> $t {
                contract::same_len("dot", &[a.len(), b.len()]);
                if has_avx2_fma() {
                    unsafe { dot_avx(a, b) }
                } else {
                    unsafe { dot_sse(a, b) }
                }
            }

            /// # Safety
            ///
            /// The CPU must support `avx2` and `fma`, and the slices must have the
            /// lengths the safe entry point checks before calling this.
            #[target_feature(enable = "avx2,fma")]
            unsafe fn dot_avx(a: &[$t], b: &[$t]) -> $t {
                let mut i = 0;
                let mut acc: [$avx_vec; 4] = [$avx_zero(); 4];
                while i + 4 * $avx_lanes <= a.len() {
                    for lane in 0..4 {
                        let offset = i + lane * $avx_lanes;
                        let va = $avx_load(a.as_ptr().add(offset));
                        let vb = $avx_load(b.as_ptr().add(offset));
                        acc[lane] = $avx_fma(va, vb, acc[lane]);
                    }
                    i += 4 * $avx_lanes;
                }
                let total = $avx_add($avx_add(acc[0], acc[1]), $avx_add(acc[2], acc[3]));
                let mut lanes = [0 as $t; $avx_lanes];
                $avx_store(lanes.as_mut_ptr(), total);
                let mut sum: $t = lanes.into_iter().sum();
                while i + $avx_lanes <= a.len() {
                    let product =
                        $avx_mul($avx_load(a.as_ptr().add(i)), $avx_load(b.as_ptr().add(i)));
                    $avx_store(lanes.as_mut_ptr(), product);
                    sum += lanes.into_iter().sum::<$t>();
                    i += $avx_lanes;
                }
                while i < a.len() {
                    sum += *a.get_unchecked(i) * *b.get_unchecked(i);
                    i += 1;
                }
                sum
            }

            /// # Safety
            ///
            /// The CPU must support `sse2`, and the slices must have the
            /// lengths the safe entry point checks before calling this.
            #[target_feature(enable = "sse2")]
            unsafe fn dot_sse(a: &[$t], b: &[$t]) -> $t {
                let mut i = 0;
                let mut acc: [$sse_vec; 4] = [$sse_zero(); 4];
                while i + 4 * $sse_lanes <= a.len() {
                    for lane in 0..4 {
                        let offset = i + lane * $sse_lanes;
                        let product = $sse_mul(
                            $sse_load(a.as_ptr().add(offset)),
                            $sse_load(b.as_ptr().add(offset)),
                        );
                        acc[lane] = $sse_add(acc[lane], product);
                    }
                    i += 4 * $sse_lanes;
                }
                let total = $sse_add($sse_add(acc[0], acc[1]), $sse_add(acc[2], acc[3]));
                let mut lanes = [0 as $t; $sse_lanes];
                $sse_store(lanes.as_mut_ptr(), total);
                let mut sum: $t = lanes.into_iter().sum();
                while i + $sse_lanes <= a.len() {
                    let product =
                        $sse_mul($sse_load(a.as_ptr().add(i)), $sse_load(b.as_ptr().add(i)));
                    $sse_store(lanes.as_mut_ptr(), product);
                    sum += lanes.into_iter().sum::<$t>();
                    i += $sse_lanes;
                }
                while i < a.len() {
                    sum += *a.get_unchecked(i) * *b.get_unchecked(i);
                    i += 1;
                }
                sum
            }

            #[inline]
            pub fn elementwise(a: &[$t], b: &[$t], op: BinaryOp, out: &mut [$t]) {
                contract::same_len("elementwise", &[a.len(), b.len(), out.len()]);
                if op == BinaryOp::Rem {
                    for i in 0..a.len() {
                        out[i] = a[i] % b[i];
                    }
                } else if has_avx2() {
                    unsafe { elementwise_avx(a, b, op, out) };
                } else {
                    unsafe { elementwise_sse(a, b, op, out) };
                }
            }

            /// # Safety
            ///
            /// The CPU must support `avx2`, and the slices must have the
            /// lengths the safe entry point checks before calling this.
            #[target_feature(enable = "avx2")]
            unsafe fn elementwise_avx(a: &[$t], b: &[$t], op: BinaryOp, out: &mut [$t]) {
                let mut i = 0;
                while i + $avx_lanes <= a.len() {
                    let va = $avx_load(a.as_ptr().add(i));
                    let vb = $avx_load(b.as_ptr().add(i));
                    let result = match op {
                        BinaryOp::Add => $avx_add(va, vb),
                        BinaryOp::Sub => $avx_sub(va, vb),
                        BinaryOp::Mul => $avx_mul(va, vb),
                        BinaryOp::Div => $avx_div(va, vb),
                        BinaryOp::Rem => unreachable!(),
                    };
                    $avx_store(out.as_mut_ptr().add(i), result);
                    i += $avx_lanes;
                }
                scalar_elementwise(a, b, op, out, i);
            }

            /// # Safety
            ///
            /// The CPU must support `sse2`, and the slices must have the
            /// lengths the safe entry point checks before calling this.
            #[target_feature(enable = "sse2")]
            unsafe fn elementwise_sse(a: &[$t], b: &[$t], op: BinaryOp, out: &mut [$t]) {
                let mut i = 0;
                while i + $sse_lanes <= a.len() {
                    let va = $sse_load(a.as_ptr().add(i));
                    let vb = $sse_load(b.as_ptr().add(i));
                    let result = match op {
                        BinaryOp::Add => $sse_add(va, vb),
                        BinaryOp::Sub => $sse_sub(va, vb),
                        BinaryOp::Mul => $sse_mul(va, vb),
                        BinaryOp::Div => $sse_div(va, vb),
                        BinaryOp::Rem => unreachable!(),
                    };
                    $sse_store(out.as_mut_ptr().add(i), result);
                    i += $sse_lanes;
                }
                scalar_elementwise(a, b, op, out, i);
            }

            fn scalar_elementwise(a: &[$t], b: &[$t], op: BinaryOp, out: &mut [$t], start: usize) {
                for i in start..a.len() {
                    out[i] = match op {
                        BinaryOp::Add => a[i] + b[i],
                        BinaryOp::Sub => a[i] - b[i],
                        BinaryOp::Mul => a[i] * b[i],
                        BinaryOp::Div => a[i] / b[i],
                        BinaryOp::Rem => a[i] % b[i],
                    };
                }
            }

            #[inline]
            pub fn broadcast(
                values: &[$t],
                scalar: $t,
                op: BinaryOp,
                scalar_left: bool,
                out: &mut [$t],
            ) {
                contract::same_len("broadcast", &[values.len(), out.len()]);
                if op == BinaryOp::Rem {
                    scalar_broadcast(values, scalar, op, scalar_left, out, 0);
                } else if has_avx2() {
                    unsafe { broadcast_avx(values, scalar, op, scalar_left, out) };
                } else {
                    unsafe { broadcast_sse(values, scalar, op, scalar_left, out) };
                }
            }

            /// # Safety
            ///
            /// The CPU must support `avx2`, and the slices must have the
            /// lengths the safe entry point checks before calling this.
            #[target_feature(enable = "avx2")]
            unsafe fn broadcast_avx(
                values: &[$t],
                scalar: $t,
                op: BinaryOp,
                scalar_left: bool,
                out: &mut [$t],
            ) {
                let scalar = $avx_set1(scalar);
                let mut i = 0;
                while i + $avx_lanes <= values.len() {
                    let value = $avx_load(values.as_ptr().add(i));
                    let (lhs, rhs) = if scalar_left {
                        (scalar, value)
                    } else {
                        (value, scalar)
                    };
                    let result = match op {
                        BinaryOp::Add => $avx_add(lhs, rhs),
                        BinaryOp::Sub => $avx_sub(lhs, rhs),
                        BinaryOp::Mul => $avx_mul(lhs, rhs),
                        BinaryOp::Div => $avx_div(lhs, rhs),
                        BinaryOp::Rem => unreachable!(),
                    };
                    $avx_store(out.as_mut_ptr().add(i), result);
                    i += $avx_lanes;
                }
                scalar_broadcast(values, scalar_value(scalar), op, scalar_left, out, i);
            }

            /// # Safety
            ///
            /// The CPU must support `sse2`, and the slices must have the
            /// lengths the safe entry point checks before calling this.
            #[target_feature(enable = "sse2")]
            unsafe fn broadcast_sse(
                values: &[$t],
                scalar_value_: $t,
                op: BinaryOp,
                scalar_left: bool,
                out: &mut [$t],
            ) {
                let scalar = $sse_set1(scalar_value_);
                let mut i = 0;
                while i + $sse_lanes <= values.len() {
                    let value = $sse_load(values.as_ptr().add(i));
                    let (lhs, rhs) = if scalar_left {
                        (scalar, value)
                    } else {
                        (value, scalar)
                    };
                    let result = match op {
                        BinaryOp::Add => $sse_add(lhs, rhs),
                        BinaryOp::Sub => $sse_sub(lhs, rhs),
                        BinaryOp::Mul => $sse_mul(lhs, rhs),
                        BinaryOp::Div => $sse_div(lhs, rhs),
                        BinaryOp::Rem => unreachable!(),
                    };
                    $sse_store(out.as_mut_ptr().add(i), result);
                    i += $sse_lanes;
                }
                scalar_broadcast(values, scalar_value_, op, scalar_left, out, i);
            }

            /// # Safety
            ///
            /// The CPU must support `avx2`.
            #[target_feature(enable = "avx2")]
            unsafe fn scalar_value(value: $avx_vec) -> $t {
                let mut lanes = [0 as $t; $avx_lanes];
                $avx_store(lanes.as_mut_ptr(), value);
                lanes[0]
            }

            fn scalar_broadcast(
                values: &[$t],
                scalar: $t,
                op: BinaryOp,
                scalar_left: bool,
                out: &mut [$t],
                start: usize,
            ) {
                for i in start..values.len() {
                    let (lhs, rhs) = if scalar_left {
                        (scalar, values[i])
                    } else {
                        (values[i], scalar)
                    };
                    out[i] = match op {
                        BinaryOp::Add => lhs + rhs,
                        BinaryOp::Sub => lhs - rhs,
                        BinaryOp::Mul => lhs * rhs,
                        BinaryOp::Div => lhs / rhs,
                        BinaryOp::Rem => lhs % rhs,
                    };
                }
            }

            /// The scalar definition of every comparison, for the ragged tail.
            #[inline]
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

            // `minps`/`maxps` return their *second* operand whenever the pair is
            // unordered, so they already agree with `f32::min` when the left
            // operand is the NaN and disagree when the right one is. Selecting
            // the left operand wherever the right is NaN patches exactly that
            // case, and leaves these matching the scalar loop on every input
            // but one: a `-0.0`/`+0.0` tie, where these answer with the second
            // operand's sign. IEEE 754 leaves that sign to the implementation
            // and the two zeros compare equal, so the fixup it would take is not
            // worth putting in the inner loop.
            //
            // aarch64 needs none of the NaN handling: `fminnm`/`fmaxnm` are the
            // IEEE `minNum`/`maxNum` that `f32::min` is defined as.
            /// # Safety
            ///
            /// The CPU must support `avx2`.
            #[target_feature(enable = "avx2")]
            unsafe fn min_avx(a: $avx_vec, b: $avx_vec) -> $avx_vec {
                let unordered = $avx_cmp::<_CMP_UNORD_Q>(b, b);
                $avx_or(
                    $avx_and(unordered, a),
                    $avx_andnot(unordered, $avx_min(a, b)),
                )
            }

            /// # Safety
            ///
            /// The CPU must support `avx2`.
            #[target_feature(enable = "avx2")]
            unsafe fn max_avx(a: $avx_vec, b: $avx_vec) -> $avx_vec {
                let unordered = $avx_cmp::<_CMP_UNORD_Q>(b, b);
                $avx_or(
                    $avx_and(unordered, a),
                    $avx_andnot(unordered, $avx_max(a, b)),
                )
            }

            /// # Safety
            ///
            /// The CPU must support `sse2`.
            #[target_feature(enable = "sse2")]
            unsafe fn min_sse(a: $sse_vec, b: $sse_vec) -> $sse_vec {
                let unordered = $sse_cmpunord(b, b);
                $sse_or(
                    $sse_and(unordered, a),
                    $sse_andnot(unordered, $sse_min(a, b)),
                )
            }

            /// # Safety
            ///
            /// The CPU must support `sse2`.
            #[target_feature(enable = "sse2")]
            unsafe fn max_sse(a: $sse_vec, b: $sse_vec) -> $sse_vec {
                let unordered = $sse_cmpunord(b, b);
                $sse_or(
                    $sse_and(unordered, a),
                    $sse_andnot(unordered, $sse_max(a, b)),
                )
            }

            /// One vector-wide comparison.
            ///
            /// The predicates use the *ordered* comparisons, so a NaN operand
            /// answers `0.0` — as `a < b` does in Rust. `Greater` and
            /// `GreaterEqual` are the same instructions with the operands
            /// swapped, which is why only the two `lt`/`le` intrinsics are
            /// threaded through.
            #[target_feature(enable = "avx2")]
            unsafe fn compare_avx(op: Compare, a: $avx_vec, b: $avx_vec) -> $avx_vec {
                let one = $avx_set1(1.0);
                match op {
                    Compare::Min => min_avx(a, b),
                    Compare::Max => max_avx(a, b),
                    Compare::MaxShare => {
                        let greater = $avx_cmp::<_CMP_LT_OQ>(b, a);
                        let less = $avx_cmp::<_CMP_LT_OQ>(a, b);
                        let tie = $avx_andnot($avx_or(greater, less), $avx_set1(0.5));
                        $avx_or($avx_and(greater, one), tie)
                    }
                    Compare::Less => $avx_and($avx_cmp::<_CMP_LT_OQ>(a, b), one),
                    Compare::LessEqual => $avx_and($avx_cmp::<_CMP_LE_OQ>(a, b), one),
                    Compare::Greater => $avx_and($avx_cmp::<_CMP_LT_OQ>(b, a), one),
                    Compare::GreaterEqual => $avx_and($avx_cmp::<_CMP_LE_OQ>(b, a), one),
                }
            }

            /// # Safety
            ///
            /// The CPU must support `sse2`.
            #[target_feature(enable = "sse2")]
            unsafe fn compare_sse(op: Compare, a: $sse_vec, b: $sse_vec) -> $sse_vec {
                let one = $sse_set1(1.0);
                match op {
                    Compare::Min => min_sse(a, b),
                    Compare::Max => max_sse(a, b),
                    Compare::MaxShare => {
                        let greater = $sse_cmplt(b, a);
                        let less = $sse_cmplt(a, b);
                        let tie = $sse_andnot($sse_or(greater, less), $sse_set1(0.5));
                        $sse_or($sse_and(greater, one), tie)
                    }
                    Compare::Less => $sse_and($sse_cmplt(a, b), one),
                    Compare::LessEqual => $sse_and($sse_cmple(a, b), one),
                    Compare::Greater => $sse_and($sse_cmplt(b, a), one),
                    Compare::GreaterEqual => $sse_and($sse_cmple(b, a), one),
                }
            }

            /// Elementwise comparison of two slices.
            #[inline]
            pub fn compare(a: &[$t], b: &[$t], op: Compare, out: &mut [$t]) {
                contract::same_len("compare", &[a.len(), b.len(), out.len()]);
                if has_avx2() {
                    unsafe { compare_slices_avx(a, b, op, out) };
                } else {
                    unsafe { compare_slices_sse(a, b, op, out) };
                }
            }

            /// # Safety
            ///
            /// The CPU must support `avx2`, and the slices must have the
            /// lengths the safe entry point checks before calling this.
            #[target_feature(enable = "avx2")]
            unsafe fn compare_slices_avx(a: &[$t], b: &[$t], op: Compare, out: &mut [$t]) {
                let mut i = 0;
                while i + $avx_lanes <= a.len() {
                    let va = $avx_load(a.as_ptr().add(i));
                    let vb = $avx_load(b.as_ptr().add(i));
                    $avx_store(out.as_mut_ptr().add(i), compare_avx(op, va, vb));
                    i += $avx_lanes;
                }
                scalar_compare(a, b, op, out, i);
            }

            /// # Safety
            ///
            /// The CPU must support `sse2`, and the slices must have the
            /// lengths the safe entry point checks before calling this.
            #[target_feature(enable = "sse2")]
            unsafe fn compare_slices_sse(a: &[$t], b: &[$t], op: Compare, out: &mut [$t]) {
                let mut i = 0;
                while i + $sse_lanes <= a.len() {
                    let va = $sse_load(a.as_ptr().add(i));
                    let vb = $sse_load(b.as_ptr().add(i));
                    $sse_store(out.as_mut_ptr().add(i), compare_sse(op, va, vb));
                    i += $sse_lanes;
                }
                scalar_compare(a, b, op, out, i);
            }

            fn scalar_compare(a: &[$t], b: &[$t], op: Compare, out: &mut [$t], start: usize) {
                for i in start..a.len() {
                    out[i] = compare_values(op, a[i], b[i]);
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
                contract::same_len("compare_scalar", &[values.len(), out.len()]);
                if has_avx2() {
                    unsafe { compare_scalar_avx(values, scalar, op, scalar_left, out) };
                } else {
                    unsafe { compare_scalar_sse(values, scalar, op, scalar_left, out) };
                }
            }

            /// # Safety
            ///
            /// The CPU must support `avx2`, and the slices must have the
            /// lengths the safe entry point checks before calling this.
            #[target_feature(enable = "avx2")]
            unsafe fn compare_scalar_avx(
                values: &[$t],
                scalar: $t,
                op: Compare,
                scalar_left: bool,
                out: &mut [$t],
            ) {
                let splat = $avx_set1(scalar);
                let mut i = 0;
                while i + $avx_lanes <= values.len() {
                    let value = $avx_load(values.as_ptr().add(i));
                    let (lhs, rhs) = if scalar_left {
                        (splat, value)
                    } else {
                        (value, splat)
                    };
                    $avx_store(out.as_mut_ptr().add(i), compare_avx(op, lhs, rhs));
                    i += $avx_lanes;
                }
                scalar_compare_scalar(values, scalar, op, scalar_left, out, i);
            }

            /// # Safety
            ///
            /// The CPU must support `sse2`, and the slices must have the
            /// lengths the safe entry point checks before calling this.
            #[target_feature(enable = "sse2")]
            unsafe fn compare_scalar_sse(
                values: &[$t],
                scalar: $t,
                op: Compare,
                scalar_left: bool,
                out: &mut [$t],
            ) {
                let splat = $sse_set1(scalar);
                let mut i = 0;
                while i + $sse_lanes <= values.len() {
                    let value = $sse_load(values.as_ptr().add(i));
                    let (lhs, rhs) = if scalar_left {
                        (splat, value)
                    } else {
                        (value, splat)
                    };
                    $sse_store(out.as_mut_ptr().add(i), compare_sse(op, lhs, rhs));
                    i += $sse_lanes;
                }
                scalar_compare_scalar(values, scalar, op, scalar_left, out, i);
            }

            fn scalar_compare_scalar(
                values: &[$t],
                scalar: $t,
                op: Compare,
                scalar_left: bool,
                out: &mut [$t],
                start: usize,
            ) {
                for i in start..values.len() {
                    let (lhs, rhs) = if scalar_left {
                        (scalar, values[i])
                    } else {
                        (values[i], scalar)
                    };
                    out[i] = compare_values(op, lhs, rhs);
                }
            }

            /// Elementwise square root. `sqrtps`/`sqrtpd` are correctly
            /// rounded, so every lane agrees with the scalar `sqrt` bit for bit.
            #[inline]
            pub fn sqrt(values: &[$t], out: &mut [$t]) {
                contract::same_len("sqrt", &[values.len(), out.len()]);
                if has_avx2() {
                    unsafe { sqrt_avx(values, out) };
                } else {
                    unsafe { sqrt_sse(values, out) };
                }
            }

            /// # Safety
            ///
            /// The CPU must support `avx2`, and the slices must have the
            /// lengths the safe entry point checks before calling this.
            #[target_feature(enable = "avx2")]
            unsafe fn sqrt_avx(values: &[$t], out: &mut [$t]) {
                let mut i = 0;
                while i + $avx_lanes <= values.len() {
                    let value = $avx_load(values.as_ptr().add(i));
                    $avx_store(out.as_mut_ptr().add(i), $avx_sqrt(value));
                    i += $avx_lanes;
                }
                for i in i..values.len() {
                    out[i] = values[i].sqrt();
                }
            }

            /// # Safety
            ///
            /// The CPU must support `sse2`, and the slices must have the
            /// lengths the safe entry point checks before calling this.
            #[target_feature(enable = "sse2")]
            unsafe fn sqrt_sse(values: &[$t], out: &mut [$t]) {
                let mut i = 0;
                while i + $sse_lanes <= values.len() {
                    let value = $sse_load(values.as_ptr().add(i));
                    $sse_store(out.as_mut_ptr().add(i), $sse_sqrt(value));
                    i += $sse_lanes;
                }
                for i in i..values.len() {
                    out[i] = values[i].sqrt();
                }
            }

            /// Confine every element to `[low, high]`, in one pass.
            #[inline]
            pub fn clamp(values: &[$t], low: $t, high: $t, out: &mut [$t]) {
                contract::same_len("clamp", &[values.len(), out.len()]);
                if has_avx2() {
                    unsafe { clamp_avx(values, low, high, out) };
                } else {
                    unsafe { clamp_sse(values, low, high, out) };
                }
            }

            /// # Safety
            ///
            /// The CPU must support `avx2`, and the slices must have the
            /// lengths the safe entry point checks before calling this.
            #[target_feature(enable = "avx2")]
            unsafe fn clamp_avx(values: &[$t], low: $t, high: $t, out: &mut [$t]) {
                let (vlow, vhigh) = ($avx_set1(low), $avx_set1(high));
                let mut i = 0;
                while i + $avx_lanes <= values.len() {
                    let value = $avx_load(values.as_ptr().add(i));
                    $avx_store(
                        out.as_mut_ptr().add(i),
                        min_avx(max_avx(value, vlow), vhigh),
                    );
                    i += $avx_lanes;
                }
                scalar_clamp(values, low, high, out, i);
            }

            /// # Safety
            ///
            /// The CPU must support `sse2`, and the slices must have the
            /// lengths the safe entry point checks before calling this.
            #[target_feature(enable = "sse2")]
            unsafe fn clamp_sse(values: &[$t], low: $t, high: $t, out: &mut [$t]) {
                let (vlow, vhigh) = ($sse_set1(low), $sse_set1(high));
                let mut i = 0;
                while i + $sse_lanes <= values.len() {
                    let value = $sse_load(values.as_ptr().add(i));
                    $sse_store(
                        out.as_mut_ptr().add(i),
                        min_sse(max_sse(value, vlow), vhigh),
                    );
                    i += $sse_lanes;
                }
                scalar_clamp(values, low, high, out, i);
            }

            fn scalar_clamp(values: &[$t], low: $t, high: $t, out: &mut [$t], start: usize) {
                for i in start..values.len() {
                    out[i] = values[i].max(low).min(high);
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
            /// Four accumulators, as in [`dot`], so the fold is not serialized
            /// on one instruction's latency. `Sum` therefore associates
            /// differently from the scalar loop and can land a rounding step
            /// away from it; `Min` and `Max` are exact.
            #[inline]
            pub fn reduce(values: &[$t], op: Reduce) -> $t {
                if has_avx2() {
                    unsafe { reduce_avx(values, op) }
                } else {
                    unsafe { reduce_sse(values, op) }
                }
            }

            /// # Safety
            ///
            /// The CPU must support `avx2`, and the slices must have the
            /// lengths the safe entry point checks before calling this.
            #[target_feature(enable = "avx2")]
            unsafe fn reduce_avx(values: &[$t], op: Reduce) -> $t {
                let mut acc: [$avx_vec; 4] = [$avx_set1(identity(op)); 4];
                let mut i = 0;
                while i + 4 * $avx_lanes <= values.len() {
                    for lane in 0..4 {
                        let v = $avx_load(values.as_ptr().add(i + lane * $avx_lanes));
                        acc[lane] = match op {
                            Reduce::Sum => $avx_add(acc[lane], v),
                            Reduce::Min => min_avx(acc[lane], v),
                            Reduce::Max => max_avx(acc[lane], v),
                        };
                    }
                    i += 4 * $avx_lanes;
                }
                while i + $avx_lanes <= values.len() {
                    let v = $avx_load(values.as_ptr().add(i));
                    acc[0] = match op {
                        Reduce::Sum => $avx_add(acc[0], v),
                        Reduce::Min => min_avx(acc[0], v),
                        Reduce::Max => max_avx(acc[0], v),
                    };
                    i += $avx_lanes;
                }
                let mut lanes = [0 as $t; $avx_lanes];
                let mut total = identity(op);
                for a in acc {
                    $avx_store(lanes.as_mut_ptr(), a);
                    for lane in lanes {
                        total = reduce_values(op, total, lane);
                    }
                }
                scalar_reduce(values, op, total, i)
            }

            /// # Safety
            ///
            /// The CPU must support `sse2`, and the slices must have the
            /// lengths the safe entry point checks before calling this.
            #[target_feature(enable = "sse2")]
            unsafe fn reduce_sse(values: &[$t], op: Reduce) -> $t {
                let mut acc: [$sse_vec; 4] = [$sse_set1(identity(op)); 4];
                let mut i = 0;
                while i + 4 * $sse_lanes <= values.len() {
                    for lane in 0..4 {
                        let v = $sse_load(values.as_ptr().add(i + lane * $sse_lanes));
                        acc[lane] = match op {
                            Reduce::Sum => $sse_add(acc[lane], v),
                            Reduce::Min => min_sse(acc[lane], v),
                            Reduce::Max => max_sse(acc[lane], v),
                        };
                    }
                    i += 4 * $sse_lanes;
                }
                while i + $sse_lanes <= values.len() {
                    let v = $sse_load(values.as_ptr().add(i));
                    acc[0] = match op {
                        Reduce::Sum => $sse_add(acc[0], v),
                        Reduce::Min => min_sse(acc[0], v),
                        Reduce::Max => max_sse(acc[0], v),
                    };
                    i += $sse_lanes;
                }
                let mut lanes = [0 as $t; $sse_lanes];
                let mut total = identity(op);
                for a in acc {
                    $sse_store(lanes.as_mut_ptr(), a);
                    for lane in lanes {
                        total = reduce_values(op, total, lane);
                    }
                }
                scalar_reduce(values, op, total, i)
            }

            fn scalar_reduce(values: &[$t], op: Reduce, mut total: $t, start: usize) -> $t {
                for i in start..values.len() {
                    total = reduce_values(op, total, values[i]);
                }
                total
            }

            /// `Σ(xᵢ − mean)²` — the second pass of a two-pass variance.
            ///
            /// Four accumulators, as in [`reduce`]; the AVX2 path fuses the
            /// square and the accumulation into one FMA, and the SSE2 baseline
            /// spells the same arithmetic as a multiply and an add.
            #[inline]
            pub fn sum_squared_deviations(values: &[$t], mean: $t) -> $t {
                if has_avx2_fma() {
                    unsafe { sum_squared_deviations_avx(values, mean) }
                } else {
                    unsafe { sum_squared_deviations_sse(values, mean) }
                }
            }

            /// # Safety
            ///
            /// The CPU must support `avx2` and `fma`, and the slices must have the
            /// lengths the safe entry point checks before calling this.
            #[target_feature(enable = "avx2,fma")]
            unsafe fn sum_squared_deviations_avx(values: &[$t], mean: $t) -> $t {
                let center = $avx_set1(mean);
                let mut acc: [$avx_vec; 4] = [$avx_zero(); 4];
                let mut i = 0;
                while i + 4 * $avx_lanes <= values.len() {
                    for lane in 0..4 {
                        let offset = i + lane * $avx_lanes;
                        let d = $avx_sub($avx_load(values.as_ptr().add(offset)), center);
                        acc[lane] = $avx_fma(d, d, acc[lane]);
                    }
                    i += 4 * $avx_lanes;
                }
                while i + $avx_lanes <= values.len() {
                    let d = $avx_sub($avx_load(values.as_ptr().add(i)), center);
                    acc[0] = $avx_fma(d, d, acc[0]);
                    i += $avx_lanes;
                }
                let total = $avx_add($avx_add(acc[0], acc[1]), $avx_add(acc[2], acc[3]));
                let mut lanes = [0 as $t; $avx_lanes];
                $avx_store(lanes.as_mut_ptr(), total);
                scalar_squared_deviations(values, mean, lanes.into_iter().sum(), i)
            }

            /// # Safety
            ///
            /// The CPU must support `sse2`, and the slices must have the
            /// lengths the safe entry point checks before calling this.
            #[target_feature(enable = "sse2")]
            unsafe fn sum_squared_deviations_sse(values: &[$t], mean: $t) -> $t {
                let center = $sse_set1(mean);
                let mut acc: [$sse_vec; 4] = [$sse_zero(); 4];
                let mut i = 0;
                while i + 4 * $sse_lanes <= values.len() {
                    for lane in 0..4 {
                        let offset = i + lane * $sse_lanes;
                        let d = $sse_sub($sse_load(values.as_ptr().add(offset)), center);
                        acc[lane] = $sse_add(acc[lane], $sse_mul(d, d));
                    }
                    i += 4 * $sse_lanes;
                }
                while i + $sse_lanes <= values.len() {
                    let d = $sse_sub($sse_load(values.as_ptr().add(i)), center);
                    acc[0] = $sse_add(acc[0], $sse_mul(d, d));
                    i += $sse_lanes;
                }
                let total = $sse_add($sse_add(acc[0], acc[1]), $sse_add(acc[2], acc[3]));
                let mut lanes = [0 as $t; $sse_lanes];
                $sse_store(lanes.as_mut_ptr(), total);
                scalar_squared_deviations(values, mean, lanes.into_iter().sum(), i)
            }

            fn scalar_squared_deviations(
                values: &[$t],
                mean: $t,
                mut total: $t,
                start: usize,
            ) -> $t {
                for i in start..values.len() {
                    let d = values[i] - mean;
                    total += d * d;
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
                contract::same_len("accumulate", &[totals.len(), values.len()]);
                if has_avx2() {
                    unsafe { accumulate_avx(totals, values) };
                } else {
                    unsafe { accumulate_sse(totals, values) };
                }
            }

            /// # Safety
            ///
            /// The CPU must support `avx2`, and the slices must have the
            /// lengths the safe entry point checks before calling this.
            #[target_feature(enable = "avx2")]
            unsafe fn accumulate_avx(totals: &mut [$t], values: &[$t]) {
                let mut i = 0;
                while i + $avx_lanes <= totals.len() {
                    let sum = $avx_add(
                        $avx_load(totals.as_ptr().add(i)),
                        $avx_load(values.as_ptr().add(i)),
                    );
                    $avx_store(totals.as_mut_ptr().add(i), sum);
                    i += $avx_lanes;
                }
                scalar_accumulate(totals, values, i);
            }

            /// # Safety
            ///
            /// The CPU must support `sse2`, and the slices must have the
            /// lengths the safe entry point checks before calling this.
            #[target_feature(enable = "sse2")]
            unsafe fn accumulate_sse(totals: &mut [$t], values: &[$t]) {
                let mut i = 0;
                while i + $sse_lanes <= totals.len() {
                    let sum = $sse_add(
                        $sse_load(totals.as_ptr().add(i)),
                        $sse_load(values.as_ptr().add(i)),
                    );
                    $sse_store(totals.as_mut_ptr().add(i), sum);
                    i += $sse_lanes;
                }
                scalar_accumulate(totals, values, i);
            }

            fn scalar_accumulate(totals: &mut [$t], values: &[$t], start: usize) {
                for i in start..totals.len() {
                    totals[i] += values[i];
                }
            }

            /// `totals += (values − means)²`, elementwise and in place — the
            /// column-wise counterpart of [`sum_squared_deviations`].
            #[inline]
            pub fn accumulate_squared_deviations(totals: &mut [$t], values: &[$t], means: &[$t]) {
                contract::same_len(
                    "accumulate_squared_deviations",
                    &[totals.len(), values.len(), means.len()],
                );
                if has_avx2_fma() {
                    unsafe { accumulate_squared_deviations_avx(totals, values, means) };
                } else {
                    unsafe { accumulate_squared_deviations_sse(totals, values, means) };
                }
            }

            /// # Safety
            ///
            /// The CPU must support `avx2` and `fma`, and the slices must have the
            /// lengths the safe entry point checks before calling this.
            #[target_feature(enable = "avx2,fma")]
            unsafe fn accumulate_squared_deviations_avx(
                totals: &mut [$t],
                values: &[$t],
                means: &[$t],
            ) {
                let mut i = 0;
                while i + $avx_lanes <= totals.len() {
                    let d = $avx_sub(
                        $avx_load(values.as_ptr().add(i)),
                        $avx_load(means.as_ptr().add(i)),
                    );
                    let total = $avx_fma(d, d, $avx_load(totals.as_ptr().add(i)));
                    $avx_store(totals.as_mut_ptr().add(i), total);
                    i += $avx_lanes;
                }
                scalar_accumulate_squared_deviations(totals, values, means, i);
            }

            /// # Safety
            ///
            /// The CPU must support `sse2`, and the slices must have the
            /// lengths the safe entry point checks before calling this.
            #[target_feature(enable = "sse2")]
            unsafe fn accumulate_squared_deviations_sse(
                totals: &mut [$t],
                values: &[$t],
                means: &[$t],
            ) {
                let mut i = 0;
                while i + $sse_lanes <= totals.len() {
                    let d = $sse_sub(
                        $sse_load(values.as_ptr().add(i)),
                        $sse_load(means.as_ptr().add(i)),
                    );
                    let total = $sse_add($sse_load(totals.as_ptr().add(i)), $sse_mul(d, d));
                    $sse_store(totals.as_mut_ptr().add(i), total);
                    i += $sse_lanes;
                }
                scalar_accumulate_squared_deviations(totals, values, means, i);
            }

            fn scalar_accumulate_squared_deviations(
                totals: &mut [$t],
                values: &[$t],
                means: &[$t],
                start: usize,
            ) {
                for i in start..totals.len() {
                    let d = values[i] - means[i];
                    totals[i] += d * d;
                }
            }

            #[inline]
            pub fn matmul(a: &[$t], b: &[$t], m: usize, k: usize, n: usize, out: &mut [$t]) {
                out.fill(0 as $t);
                matmul_accumulate(a, b, m, k, n, out);
            }

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
                contract::same_len("matmul_add", &[out.len(), addend.len()]);
                out.copy_from_slice(addend);
                matmul_accumulate(a, b, m, k, n, out);
            }

            #[inline]
            pub fn matmul_accumulate(
                a: &[$t],
                b: &[$t],
                m: usize,
                k: usize,
                n: usize,
                out: &mut [$t],
            ) {
                contract::matmul("matmul", a.len(), b.len(), out.len(), (m, k, n));
                if has_avx2_fma() {
                    unsafe { matmul_avx(a, b, m, k, n, out) };
                } else {
                    unsafe { matmul_sse(a, b, m, k, n, out) };
                }
            }

            /// # Safety
            ///
            /// The CPU must support `avx2` and `fma`, and the slices must have the
            /// lengths the safe entry point checks before calling this.
            #[target_feature(enable = "avx2,fma")]
            unsafe fn matmul_avx(a: &[$t], b: &[$t], m: usize, k: usize, n: usize, out: &mut [$t]) {
                for i in 0..m {
                    let arow = i * k;
                    let orow = i * n;
                    for p in 0..k {
                        let value = *a.get_unchecked(arow + p);
                        let broadcast = $avx_set1(value);
                        let brow = p * n;
                        let mut j = 0;
                        while j + $avx_lanes <= n {
                            let rhs = $avx_load(b.as_ptr().add(brow + j));
                            let current = $avx_load(out.as_ptr().add(orow + j));
                            $avx_store(
                                out.as_mut_ptr().add(orow + j),
                                $avx_fma(broadcast, rhs, current),
                            );
                            j += $avx_lanes;
                        }
                        while j < n {
                            *out.get_unchecked_mut(orow + j) += value * *b.get_unchecked(brow + j);
                            j += 1;
                        }
                    }
                }
            }

            /// # Safety
            ///
            /// The CPU must support `sse2`, and the slices must have the
            /// lengths the safe entry point checks before calling this.
            #[target_feature(enable = "sse2")]
            unsafe fn matmul_sse(a: &[$t], b: &[$t], m: usize, k: usize, n: usize, out: &mut [$t]) {
                for i in 0..m {
                    let arow = i * k;
                    let orow = i * n;
                    for p in 0..k {
                        let value = *a.get_unchecked(arow + p);
                        let broadcast = $sse_set1(value);
                        let brow = p * n;
                        let mut j = 0;
                        while j + $sse_lanes <= n {
                            let rhs = $sse_load(b.as_ptr().add(brow + j));
                            let current = $sse_load(out.as_ptr().add(orow + j));
                            $sse_store(
                                out.as_mut_ptr().add(orow + j),
                                $sse_add(current, $sse_mul(broadcast, rhs)),
                            );
                            j += $sse_lanes;
                        }
                        while j < n {
                            *out.get_unchecked_mut(orow + j) += value * *b.get_unchecked(brow + j);
                            j += 1;
                        }
                    }
                }
            }

            #[cfg(test)]
            mod architecture_tests {
                use super::*;

                #[test]
                fn forced_instruction_paths_match_scalar() {
                    let a = (0..37)
                        .map(|i| i as $t * 0.125 as $t - 2.0 as $t)
                        .collect::<Vec<_>>();
                    let b = (0..37)
                        .map(|i| (i % 9) as $t * 0.2 as $t + 0.5 as $t)
                        .collect::<Vec<_>>();
                    let expected: $t = a.iter().zip(&b).map(|(x, y)| *x * *y).sum();
                    let tolerance = 1e-4 as $t * (1.0 as $t + expected.abs());

                    let sse = unsafe { dot_sse(&a, &b) };
                    assert!((sse - expected).abs() <= tolerance);

                    if has_avx2_fma() {
                        let avx = unsafe { dot_avx(&a, &b) };
                        assert!((avx - expected).abs() <= tolerance);
                    }
                }
            }
        }
    };
}

x86_kernels! {
    mod f32k, ty = f32,
    avx_vec = __m256, avx_lanes = 8,
    avx_load = _mm256_loadu_ps, avx_store = _mm256_storeu_ps,
    avx_set1 = _mm256_set1_ps, avx_zero = _mm256_setzero_ps,
    avx_add = _mm256_add_ps, avx_sub = _mm256_sub_ps,
    avx_mul = _mm256_mul_ps, avx_div = _mm256_div_ps,
    avx_fma = _mm256_fmadd_ps, avx_sqrt = _mm256_sqrt_ps,
    avx_min = _mm256_min_ps, avx_max = _mm256_max_ps,
    avx_and = _mm256_and_ps, avx_andnot = _mm256_andnot_ps, avx_or = _mm256_or_ps,
    avx_cmp = _mm256_cmp_ps,
    sse_vec = __m128, sse_lanes = 4,
    sse_load = _mm_loadu_ps, sse_store = _mm_storeu_ps,
    sse_set1 = _mm_set1_ps, sse_zero = _mm_setzero_ps,
    sse_add = _mm_add_ps, sse_sub = _mm_sub_ps,
    sse_mul = _mm_mul_ps, sse_div = _mm_div_ps, sse_sqrt = _mm_sqrt_ps,
    sse_min = _mm_min_ps, sse_max = _mm_max_ps,
    sse_and = _mm_and_ps, sse_andnot = _mm_andnot_ps, sse_or = _mm_or_ps,
    sse_cmplt = _mm_cmplt_ps, sse_cmple = _mm_cmple_ps,
    sse_cmpunord = _mm_cmpunord_ps
}

x86_kernels! {
    mod f64k, ty = f64,
    avx_vec = __m256d, avx_lanes = 4,
    avx_load = _mm256_loadu_pd, avx_store = _mm256_storeu_pd,
    avx_set1 = _mm256_set1_pd, avx_zero = _mm256_setzero_pd,
    avx_add = _mm256_add_pd, avx_sub = _mm256_sub_pd,
    avx_mul = _mm256_mul_pd, avx_div = _mm256_div_pd,
    avx_fma = _mm256_fmadd_pd, avx_sqrt = _mm256_sqrt_pd,
    avx_min = _mm256_min_pd, avx_max = _mm256_max_pd,
    avx_and = _mm256_and_pd, avx_andnot = _mm256_andnot_pd, avx_or = _mm256_or_pd,
    avx_cmp = _mm256_cmp_pd,
    sse_vec = __m128d, sse_lanes = 2,
    sse_load = _mm_loadu_pd, sse_store = _mm_storeu_pd,
    sse_set1 = _mm_set1_pd, sse_zero = _mm_setzero_pd,
    sse_add = _mm_add_pd, sse_sub = _mm_sub_pd,
    sse_mul = _mm_mul_pd, sse_div = _mm_div_pd, sse_sqrt = _mm_sqrt_pd,
    sse_min = _mm_min_pd, sse_max = _mm_max_pd,
    sse_and = _mm_and_pd, sse_andnot = _mm_andnot_pd, sse_or = _mm_or_pd,
    sse_cmplt = _mm_cmplt_pd, sse_cmple = _mm_cmple_pd,
    sse_cmpunord = _mm_cmpunord_pd
}

pub mod fft_f32 {
    use core::arch::x86_64::*;
    use std::arch::is_x86_feature_detected;

    /// Radix-2 FFT over an interleaved complex buffer.
    pub fn radix2(buf: &mut [f32], n: usize, direction: f32) {
        crate::simd::contract::fft("radix2", buf.len(), n);
        if n <= 1 {
            return;
        }

        let mut reversed = 0usize;
        for i in 1..n {
            let mut bit = n >> 1;
            while reversed & bit != 0 {
                reversed ^= bit;
                bit >>= 1;
            }
            reversed ^= bit;
            if i < reversed {
                buf.swap(2 * i, 2 * reversed);
                buf.swap(2 * i + 1, 2 * reversed + 1);
            }
        }

        let use_avx = is_x86_feature_detected!("avx");
        // Twiddle scratch for the widest stage, allocated once rather than once
        // per stage; see the aarch64 path for the same change.
        let mut twiddle_buf = vec![0.0f32; n];
        let mut len = 2usize;
        while len <= n {
            let half = len / 2;
            let angle = direction * std::f32::consts::TAU / len as f32;
            let step = (angle.cos(), angle.sin());
            let twiddles = &mut twiddle_buf[..2 * half];
            let (mut real, mut imaginary) = (1.0f32, 0.0f32);
            for offset in 0..half {
                twiddles[2 * offset] = real;
                twiddles[2 * offset + 1] = imaginary;
                (real, imaginary) = (
                    real * step.0 - imaginary * step.1,
                    real * step.1 + imaginary * step.0,
                );
            }

            for start in (0..n).step_by(len) {
                let mut offset = if use_avx {
                    unsafe { butterflies_avx(buf, &twiddles, start, half) }
                } else {
                    0
                };
                while offset < half {
                    let even = 2 * (start + offset);
                    let odd = 2 * (start + offset + half);
                    let (odd_real, odd_imaginary) = (buf[odd], buf[odd + 1]);
                    let twiddle_real = twiddles[2 * offset];
                    let twiddle_imaginary = twiddles[2 * offset + 1];
                    let product_real = odd_real * twiddle_real - odd_imaginary * twiddle_imaginary;
                    let product_imaginary =
                        odd_real * twiddle_imaginary + odd_imaginary * twiddle_real;
                    let (even_real, even_imaginary) = (buf[even], buf[even + 1]);
                    buf[even] = even_real + product_real;
                    buf[even + 1] = even_imaginary + product_imaginary;
                    buf[odd] = even_real - product_real;
                    buf[odd + 1] = even_imaginary - product_imaginary;
                    offset += 1;
                }
            }
            len *= 2;
        }
    }

    /// One stage's butterflies for the block at `start`, four complex points
    /// at a time; returns how many it did, leaving the tail to the scalar loop.
    ///
    /// # Safety
    ///
    /// The CPU must support `avx`; `buf` must hold `2 * (start + 2 * half)`
    /// values and `twiddles` `2 * half`, which [`radix2`] guarantees by
    /// checking that `n` is a power of two and `buf.len() == 2 * n`.
    #[target_feature(enable = "avx")]
    unsafe fn butterflies_avx(
        buf: &mut [f32],
        twiddles: &[f32],
        start: usize,
        half: usize,
    ) -> usize {
        let mut offset = 0;
        while offset + 4 <= half {
            let even_ptr = buf.as_ptr().add(2 * (start + offset));
            let odd_ptr = buf.as_ptr().add(2 * (start + offset + half));
            let even = _mm256_loadu_ps(even_ptr);
            let odd = _mm256_loadu_ps(odd_ptr);
            let twiddle = _mm256_loadu_ps(twiddles.as_ptr().add(2 * offset));

            let odd_real = _mm256_moveldup_ps(odd);
            let odd_imaginary = _mm256_movehdup_ps(odd);
            let twiddle_swapped = _mm256_permute_ps::<0b1011_0001>(twiddle);
            let product = _mm256_addsub_ps(
                _mm256_mul_ps(odd_real, twiddle),
                _mm256_mul_ps(odd_imaginary, twiddle_swapped),
            );

            _mm256_storeu_ps(
                buf.as_mut_ptr().add(2 * (start + offset)),
                _mm256_add_ps(even, product),
            );
            _mm256_storeu_ps(
                buf.as_mut_ptr().add(2 * (start + offset + half)),
                _mm256_sub_ps(even, product),
            );
            offset += 4;
        }
        offset
    }
}
