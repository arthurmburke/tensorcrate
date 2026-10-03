//! The consuming Host elementwise operations: an owned left operand holds the
//! result, and the answer is the borrowing operation's, bit for bit.
//!
//! Lengths straddle the SIMD gate (16), the vector widths, and the point where
//! the kernels split across threads (twice 32 Ki elements).

use half::{bf16, f16};
use tensorcrate::numbers::Real;
use tensorcrate::tensors::{BinaryOp, Compare, Matrix, Vector};

const LENGTHS: [usize; 9] = [0, 1, 15, 16, 17, 100, 4_099, 65_537, 70_001];

const OPS: [BinaryOp; 5] = [
    BinaryOp::Add,
    BinaryOp::Sub,
    BinaryOp::Mul,
    BinaryOp::Div,
    BinaryOp::Rem,
];

/// Deterministic values with zeros and NaNs among them, so the operations meet
/// the awkward inputs: division by zero, `0 % 0`, an unordered comparison.
fn values<T: Real>(len: usize, seed: usize) -> Vec<T> {
    (0..len)
        .map(|i| {
            let k = i + seed;
            let value = match k % 29 {
                0 => 0.0,
                1 => f64::NAN,
                _ => ((k * 37 % 101) as f64 - 50.0) / 8.0,
            };
            T::from_f64(value)
        })
        .collect()
}

/// Equal element for element, NaN with NaN and a zero's sign included.
fn same<T: Real>(what: &str, got: &[T], want: &[T]) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let (g, w) = (g.into_f64(), w.into_f64());
        assert!(
            (g.is_nan() && w.is_nan()) || g.to_bits() == w.to_bits(),
            "{what}: element {i} is {g}, expected {w}"
        );
    }
}

fn binary_scalar<T: Real>(op: BinaryOp, a: T, b: T) -> T {
    match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => a - b,
        BinaryOp::Mul => a * b,
        BinaryOp::Div => a / b,
        BinaryOp::Rem => a % b,
    }
}

macro_rules! consuming_tests {
    ($($module:ident: $T:ty),+ $(,)?) => {$(
        mod $module {
            use super::*;

            type T = $T;

            fn t(value: f64) -> T {
                T::from_f64(value)
            }

            #[test]
            fn owned_operators_match_borrowed() {
                for len in LENGTHS {
                    let a = Vector::new(values::<T>(len, 0));
                    let b = Vector::new(values::<T>(len, 5));
                    let owned = [
                        (a.clone() + b.clone(), &a + &b),
                        (a.clone() - b.clone(), &a - &b),
                        (a.clone() * b.clone(), &a * &b),
                        (a.clone() / b.clone(), &a / &b),
                        (a.clone() % b.clone(), &a % &b),
                    ];
                    for (op, (got, want)) in OPS.iter().zip(owned) {
                        same(&format!("{op:?} len={len}"), got.data(), want.data());
                    }
                }
            }

            #[test]
            fn owned_left_with_borrowed_right_leaves_the_right_alone() {
                for len in [3, 16, 4_099] {
                    let a = Vector::new(values::<T>(len, 0));
                    let b = Vector::new(values::<T>(len, 5));
                    let kept = b.clone();
                    let got = a.clone() - &b;
                    same("rhs untouched", b.data(), kept.data());
                    same("a - &b", got.data(), (&a - &b).data());
                }
            }

            #[test]
            fn matrices_match_borrowed() {
                for (rows, cols) in [(1, 1), (3, 5), (4, 4), (64, 65), (300, 250)] {
                    let n = rows * cols;
                    let a = Matrix::from_flat(rows, cols, values::<T>(n, 1));
                    let b = Matrix::from_flat(rows, cols, values::<T>(n, 9));
                    let results = [
                        (a.clone() + b.clone(), &a + &b),
                        (a.clone() - &b, &a - &b),
                        (a.clone() * b.clone(), &a * &b),
                        (a.clone() / &b, &a / &b),
                        (a.clone() % b.clone(), &a % &b),
                    ];
                    for (op, (got, want)) in OPS.iter().zip(results) {
                        assert_eq!(got.shape(), (rows, cols));
                        same(&format!("{op:?} {rows}x{cols}"), got.as_slice(), want.as_slice());
                    }
                    same("min", a.clone().into_min(&b).as_slice(), a.min(&b).as_slice());
                    same("max", a.clone().into_max(&b).as_slice(), a.max(&b).as_slice());
                    same(
                        "relu",
                        a.clone().into_max_scalar(t(0.0)).as_slice(),
                        a.max_scalar(t(0.0)).as_slice(),
                    );
                    same(
                        "clamp",
                        a.clone().into_clamp(t(-2.0), t(3.0)).as_slice(),
                        a.clamp(t(-2.0), t(3.0)).as_slice(),
                    );
                    same(
                        "scale",
                        a.clone().into_scale(t(1.5)).as_slice(),
                        a.scale(t(1.5)).as_slice(),
                    );
                    same(
                        "compare",
                        a.clone().into_compare(&b, Compare::Less).as_slice(),
                        a.compare(&b, Compare::Less).as_slice(),
                    );
                    same(
                        "neg",
                        (-a.clone()).as_slice(),
                        (-&a).as_slice(),
                    );
                }
            }

            #[test]
            fn scalar_broadcasts_respect_operand_order() {
                for len in LENGTHS {
                    let x = Vector::new(values::<T>(len, 3));
                    let scalar = t(2.5);
                    for op in OPS {
                        same(
                            &format!("{op:?} right len={len}"),
                            x.clone().into_broadcast_right(scalar, op).data(),
                            x.broadcast_right(scalar, op).data(),
                        );
                        same(
                            &format!("{op:?} left len={len}"),
                            x.clone().into_broadcast_left(scalar, op).data(),
                            x.broadcast_left(scalar, op).data(),
                        );
                    }
                    same("scale", x.clone().into_scale(scalar).data(), x.scale(scalar).data());
                }
            }

            #[test]
            fn comparisons_and_clamp_match_borrowed() {
                for len in LENGTHS {
                    let a = Vector::new(values::<T>(len, 2));
                    let b = Vector::new(values::<T>(len, 7));
                    same("min", a.clone().into_min(&b).data(), a.min(&b).data());
                    same("max", a.clone().into_max(&b).data(), a.max(&b).data());
                    same(
                        "min_scalar",
                        a.clone().into_min_scalar(t(1.0)).data(),
                        a.min_scalar(t(1.0)).data(),
                    );
                    same(
                        "relu",
                        a.clone().into_max_scalar(t(0.0)).data(),
                        a.max_scalar(t(0.0)).data(),
                    );
                    same(
                        "clamp",
                        a.clone().into_clamp(t(-1.0), t(1.0)).data(),
                        a.clamp(t(-1.0), t(1.0)).data(),
                    );
                    for op in Compare::ALL {
                        same(
                            &format!("{op:?} len={len}"),
                            a.clone().into_compare(&b, op).data(),
                            a.compare(&b, op).data(),
                        );
                        for scalar_left in [false, true] {
                            same(
                                &format!("{op:?} scalar left={scalar_left} len={len}"),
                                a.clone().into_compare_scalar(t(0.5), op, scalar_left).data(),
                                a.compare_scalar(t(0.5), op, scalar_left).data(),
                            );
                        }
                    }
                }
            }

            #[test]
            fn into_map_and_negation_match_scalar_definitions() {
                for len in LENGTHS {
                    let x = Vector::new(values::<T>(len, 4));
                    let want = x.data().iter().map(|&v| v * v).collect::<Vec<_>>();
                    same("into_map", x.clone().into_map(|v| v * v).data(), &want);
                    let want = x.data().iter().map(|&v| -v).collect::<Vec<_>>();
                    same("neg", (-x.clone()).data(), &want);
                    same("neg ref", (-&x).data(), &want);
                }
            }
        }
    )+};
}

consuming_tests!(single: f32, double: f64, half_f16: f16, brain: bf16);

/// Where the SIMD tier has an in-place kernel, the result lives in the
/// allocation that came in. `f16` and `bf16` keep their out-of-place kernels,
/// so they are not asked.
macro_rules! reuse_tests {
    ($($module:ident: $T:ty),+ $(,)?) => {$(
        mod $module {
            use super::*;

            type T = $T;

            #[test]
            fn consuming_operations_keep_the_allocation() {
                for len in [5, 16, 4_099, 70_001] {
                    let a = Vector::new(values::<T>(len, 0));
                    let b = Vector::new(values::<T>(len, 5));
                    let at = a.data().as_ptr();
                    let sum = a + &b;
                    assert_eq!(sum.data().as_ptr(), at, "add, len={len}");
                    let at = sum.data().as_ptr();
                    let relu = sum.into_max_scalar(T::from_f64(0.0));
                    assert_eq!(relu.data().as_ptr(), at, "relu, len={len}");
                    let at = relu.data().as_ptr();
                    let clamped = relu.into_clamp(T::from_f64(0.0), T::from_f64(1.0));
                    assert_eq!(clamped.data().as_ptr(), at, "clamp, len={len}");
                    let at = clamped.data().as_ptr();
                    let scaled = (-clamped).into_scale(T::from_f64(2.0));
                    assert_eq!(scaled.data().as_ptr(), at, "neg and scale, len={len}");
                }
            }
        }
    )+};
}

reuse_tests!(reuse_f32: f32, reuse_f64: f64);

#[test]
fn integers_run_in_place_without_a_kernel() {
    for len in [0, 1, 16, 4_099] {
        let a = Vector::new((0..len as i32).map(|i| i * 3 - 7).collect::<Vec<_>>());
        let b = Vector::new((0..len as i32).map(|i| i % 5 + 1).collect::<Vec<_>>());
        for (op, owned, borrowed) in [
            (BinaryOp::Add, a.clone() + b.clone(), &a + &b),
            (BinaryOp::Sub, a.clone() - &b, &a - &b),
            (BinaryOp::Mul, a.clone() * b.clone(), &a * &b),
            (BinaryOp::Div, a.clone() / b.clone(), &a / &b),
            (BinaryOp::Rem, a.clone() % &b, &a % &b),
        ] {
            assert_eq!(owned, borrowed, "{op:?}, len={len}");
        }
        let at = a.data().as_ptr();
        let negated = -a.clone().into_scale(2);
        assert_eq!(negated.len(), len);
        let reused = a.into_map(|x| x + 1);
        assert_eq!(reused.data().as_ptr(), at, "allocation, len={len}");
    }
}

#[test]
fn broadcast_scalar_left_of_integers() {
    let v = Vector::new(vec![1, 2, 3]).into_broadcast_left(10, BinaryOp::Sub);
    assert_eq!(v.data(), [9, 8, 7]);
    let v = Vector::new(vec![1, 2, 3]).into_broadcast_right(10, BinaryOp::Sub);
    assert_eq!(v.data(), [-9, -8, -7]);
}

#[test]
#[should_panic(expected = "vector lengths differ")]
fn owned_operator_checks_lengths() {
    let _ = Vector::new([1.0f32, 2.0]) + Vector::new([1.0f32]);
}

#[test]
#[should_panic(expected = "matrix shapes differ")]
fn owned_operator_checks_shapes() {
    let _ = Matrix::from_rows([[1.0f32, 2.0]]) + &Matrix::from_rows([[1.0f32], [2.0]]);
}

#[test]
#[should_panic(expected = "low")]
fn into_clamp_rejects_inverted_bounds() {
    let _ = Vector::new([1.0f32]).into_clamp(2.0, 1.0);
}

#[test]
fn binary_scalar_reference_agrees_with_operators() {
    // The scalar definition every test above leans on is the language's own.
    let a = Vector::new(values::<f64>(40, 0));
    let b = Vector::new(values::<f64>(40, 3));
    for op in OPS {
        let want = a
            .data()
            .iter()
            .zip(b.data())
            .map(|(&x, &y)| binary_scalar(op, x, y))
            .collect::<Vec<_>>();
        let got = match op {
            BinaryOp::Add => a.clone() + &b,
            BinaryOp::Sub => a.clone() - &b,
            BinaryOp::Mul => a.clone() * &b,
            BinaryOp::Div => a.clone() / &b,
            BinaryOp::Rem => a.clone() % &b,
        };
        same(&format!("{op:?}"), got.data(), &want);
    }
}
