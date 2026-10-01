//! Kernel fusion inside `math!`.
//!
//! Every block here is evaluated twice — fused, and with fusion switched off
//! through `fused::Mode::Unfused`, which runs the block exactly as the macro
//! wrote it before fusion existed. On the host the two must agree bit for bit;
//! on Metal, within the tolerance the unfused kernels already have.

use tensorcrate::math;
use tensorcrate::tensors::fused::{self, Mode};
use tensorcrate::tensors::{Matrix, Vector};

/// Evaluate `block` fused and unfused.
fn both<R>(block: impl Fn() -> R) -> (R, R) {
    let fused = fused::with_mode(Mode::Fused, &block);
    let unfused = fused::with_mode(Mode::Unfused, &block);
    (fused, unfused)
}

fn assert_bits<T: Copy + Into<f64> + 'static>(fused: &[T], unfused: &[T], what: &str) {
    assert_eq!(fused.len(), unfused.len(), "{what}");
    for (i, (&f, &u)) in fused.iter().zip(unfused).enumerate() {
        let (f, u): (f64, f64) = (f.into(), u.into());
        assert!(
            f.to_bits() == u.to_bits() || (f.is_nan() && u.is_nan()),
            "{what}: element {i} is {f} fused, {u} unfused"
        );
    }
}

fn same_vector<T: Copy + Into<f64> + 'static>(
    (fused, unfused): (Vector<T>, Vector<T>),
    what: &str,
) {
    assert_bits(fused.as_slice(), unfused.as_slice(), what);
}

fn same_matrix<T: Copy + Into<f64> + 'static>(
    (fused, unfused): (Matrix<T>, Matrix<T>),
    what: &str,
) {
    assert_eq!(fused.shape(), unfused.shape(), "{what}");
    assert_bits(fused.as_slice(), unfused.as_slice(), what);
}

#[test]
fn arithmetic_chains_fuse_exactly() {
    let t = 0.3;
    same_vector(
        both(|| {
            math! {
                let a = [1.5, -2.25, 3.0, 0.125, -7.5];
                let b = [0.5, 4.0, -1.0, 9.0, 2.0];
                (a + b * 2 - t) / (b .* b + 1) - -a % 2
            }
        }),
        "arithmetic",
    );
    same_matrix(
        both(|| {
            math! {
                let m = [[1, 2, 3], [4, 5, 6]];
                let n = [[0.5, -1, 2], [3, -4, 0.25]];
                2 - m .* n / 3 + n
            }
        }),
        "matrix arithmetic",
    );
}

#[test]
fn every_analytic_function_fuses_exactly() {
    same_vector(
        both(|| {
            math! {
                let x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
                sin(x) + cos(x) * tan(x) - sec(x) + csc(x) / 7
                    + arcsin(x) - arccos(x) + arctan(x)
                    + exp(x) * ln(x) + sinh(x) - cosh(x) + tanh(x) * sqrt(x)
            }
        }),
        "analytic",
    );
}

#[test]
fn ordering_functions_fuse_exactly() {
    same_vector(
        both(|| {
            math! {
                let x = [-3, -1.5, 0, 0.5, 2, 4];
                let y = [1, -2, 0.25, 0.5, 3, -4];
                clamp(max(x, y) * 2 - min(x, 0.5), -1, 3) + max(1, y)
            }
        }),
        "ordering",
    );
}

#[test]
fn single_precision_blocks_fuse_exactly() {
    same_vector(
        both(|| {
            math! {
                dtype = f32;
                let x = [0.1, 0.7, 1.3, 2.9];
                exp(x) * 0.5 + sqrt(x) - x .* x
            }
        }),
        "f32",
    );
}

#[test]
fn bindings_are_inlined_recomputed_or_materialized() {
    same_vector(
        both(|| {
            math! {
                let x = [1, 2, 3, 4];
                let cheap = x * 3 + 1;          // two uses, recomputed in each
                let costly = exp(x) - 1;        // two uses, computed once
                let once = cheap .* costly;     // one use, never materialized
                once + cheap - costly
            }
        }),
        "bindings",
    );
}

#[test]
fn transposes_inside_a_group_become_transposed_reads() {
    same_matrix(
        both(|| {
            math! {
                let a = [[1, 2, 3], [4, 5, 6]];
                let b = [[1, 0], [0, 1], [2, 2]];
                transpose(a * 2 + 1) - b + transpose(transpose(b))
            }
        }),
        "transpose",
    );
}

#[test]
fn groups_feed_and_follow_products() {
    same_matrix(
        both(|| {
            math! {
                let x = [[1, 2], [3, 4]];
                let w = [[0.5, -1], [2, 0.25]];
                let h = max(x @ w + 1, 0);       // a relu after a product
                let a = h * 2 - 1;
                let b = exp(h) / 10;             // independent of `a`: fused with it
                sin(a @ b) + 1
            }
        }),
        "products",
    );
}

#[test]
fn a_fused_clamp_still_rejects_crossed_bounds() {
    let crossed = std::panic::catch_unwind(|| {
        math! {
            let x = [1, 2];
            clamp(x * 2, 3, 1)
        }
    });
    assert!(crossed.is_err());
}

#[test]
fn the_result_type_is_unchanged() {
    let v: Vector<f64> = math! { let x = [1, 2]; x * 2 + 1 };
    assert_eq!(v.as_slice(), [3.0, 5.0]);
    let m: Matrix<f32> = math! { dtype = f32; let x = [[1, 2]]; sqrt(x * 8) - 1 };
    assert_eq!(m.as_slice(), [1.828_427_1, 3.0]);
}

/// Kernel counts, measured as the Metal operations actually queued: every
/// resident operation is one dispatch, so this counts what the GPU runs.
#[cfg(all(feature = "counters", feature = "metal", target_os = "macos"))]
mod counts {
    use tensorcrate::counters;
    use tensorcrate::math;
    use tensorcrate::tensors::fused::{self, Mode};

    fn dispatches<R>(block: impl Fn() -> R) -> (u64, u64) {
        let ((), fused) = counters::measure(|| drop(fused::with_mode(Mode::Fused, &block)));
        let ((), unfused) = counters::measure(|| drop(fused::with_mode(Mode::Unfused, &block)));
        (fused.dispatches, unfused.dispatches)
    }

    #[test]
    fn a_chain_is_one_kernel() {
        let counts = dispatches(|| {
            math! {
                backend = Metal;
                let x = [1, 2, 3];
                let y = [4, 5, 6];
                sin(x * 2 + y) - x .* y
            }
        });
        assert_eq!(counts, (1, 5));
    }

    #[test]
    fn independent_bindings_share_a_kernel() {
        // `a` and `b` each feed the product, so both are materialized — by one
        // kernel, not two; the product carries the final `* 3 + 1` as its
        // epilogue, which is one more.
        let counts = dispatches(|| {
            math! {
                backend = Metal;
                let x = [[1, 2], [3, 4]];
                let a = x * 2 + 1;
                let b = exp(x) - 1;
                (a @ b) * 3 + 1
            }
        });
        assert_eq!(counts, (2, 7));
    }

    #[test]
    fn a_dense_layer_is_one_kernel() {
        let counts = dispatches(|| {
            math! {
                backend = Metal;
                let x = [[1, 2], [3, 4], [5, 6]];
                let w = [[0.5, -1], [0.25, 2]];
                let b = [[1, -1], [1, -1], [1, -1]];
                tanh(x @ w + b)
            }
        });
        // Three literals uploaded is no kernel; the product, the bias and the
        // activation are one. Unfused: the product, the add and the tanh.
        assert_eq!(counts, (1, 3));
    }

    #[test]
    fn switching_fusion_off_in_the_block_restores_every_kernel() {
        let ((), counts) = counters::measure(|| {
            let _ = math! { backend = Metal; fuse = false; let x = [1, 2]; sin(x * 2 + 1) - x };
        });
        assert_eq!(counts.dispatches, 4);
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    use tensorcrate::math;
    use tensorcrate::tensors::fused::{self, Mode};
    use tensorcrate::tensors::{Host, Matrix, Metal, Vector};

    fn close(actual: &[f32], expected: &[f32], what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what}");
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() <= 1e-5 * (1.0 + e.abs()),
                "{what}: element {i} is {a}, expected {e}"
            );
        }
    }

    #[test]
    fn metal_blocks_fuse_and_agree_with_the_host() {
        let block = || {
            math! {
                backend = Metal;
                let a = [[1, 2, 3], [4, 5, 6]];
                let b = [[0.5, -1, 2], [3, -4, 0.25]];
                let c = transpose(a * 2 - b) @ (b .* b + 1);
                clamp(tanh(c / 50) - -sqrt(c .* c + 1) / 9, -2, 2)
            }
        };
        let fused: Matrix<f32, Metal> = fused::with_mode(Mode::Fused, block);
        let unfused: Matrix<f32, Metal> = fused::with_mode(Mode::Unfused, block);
        let host: Matrix<f32> = math! {
            dtype = f32;
            let a = [[1, 2, 3], [4, 5, 6]];
            let b = [[0.5, -1, 2], [3, -4, 0.25]];
            let c = transpose(a * 2 - b) @ (b .* b + 1);
            clamp(tanh(c / 50) - -sqrt(c .* c + 1) / 9, -2, 2)
        };
        let fused = fused.to_backend::<Host>();
        close(
            fused.as_slice(),
            unfused.to_backend::<Host>().as_slice(),
            "against unfused",
        );
        close(fused.as_slice(), host.as_slice(), "against the host");
    }

    #[test]
    fn products_become_epilogues_only_where_they_can() {
        // Read straight: the epilogue. Read straight and transposed: the
        // product is materialized and both reads load it.
        let metal: (Matrix<f32, Metal>, Matrix<f32, Metal>) = (
            math! {
                backend = Metal;
                let x = [[1, 2, 3], [4, 5, 6]];
                let w = [[0.5, -1], [0.25, 2], [1, 0]];
                sqrt(x @ w .* (x @ w) + 1) - 2
            },
            math! {
                backend = Metal;
                let x = [[1, 2], [3, 4]];
                let w = [[0.5, -1], [0.25, 2]];
                transpose(x @ w) * 2 + (x @ w)
            },
        );
        let host: (Matrix<f32>, Matrix<f32>) = (
            math! {
                dtype = f32;
                let x = [[1, 2, 3], [4, 5, 6]];
                let w = [[0.5, -1], [0.25, 2], [1, 0]];
                sqrt(x @ w .* (x @ w) + 1) - 2
            },
            math! {
                dtype = f32;
                let x = [[1, 2], [3, 4]];
                let w = [[0.5, -1], [0.25, 2]];
                transpose(x @ w) * 2 + (x @ w)
            },
        );
        close(metal.0.to_backend::<Host>().as_slice(), host.0.as_slice(), "epilogue");
        close(metal.1.to_backend::<Host>().as_slice(), host.1.as_slice(), "transposed too");
    }

    #[test]
    fn a_metal_group_over_the_input_limit_is_split_and_still_right() {
        let v: Vector<f32, Metal> = math! {
            backend = Metal;
            [1] + [2] + [3] + [4] + [5] + [6] + [7] + [8] + [9] + [10]
                + [11] + [12] + [13] + [14] + [15] + [16] + [17] + [18] + [19] + [20]
        };
        assert_eq!(v.to_backend::<Host>().as_slice(), [210.0]);
    }

    #[test]
    fn horizontally_fused_metal_bindings_agree_with_the_host() {
        let metal: Matrix<f32, Metal> = math! {
            backend = Metal;
            let x = [[1, 2], [3, 4]];
            let a = x * 2 + 1;
            let b = exp(x / 4) - 1;
            a @ b
        };
        let host: Matrix<f32> = math! {
            dtype = f32;
            let x = [[1, 2], [3, 4]];
            let a = x * 2 + 1;
            let b = exp(x / 4) - 1;
            a @ b
        };
        close(
            metal.to_backend::<Host>().as_slice(),
            host.as_slice(),
            "horizontal",
        );
    }
}
