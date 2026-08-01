//! Comparisons, clamping, reductions, scans and sorting over vectors — the
//! order-dependent half of the tensor algebra.
//!
//! The host results are checked against plain scalar loops, and the `Metal`
//! ones against the host: same answers, different memory. Lengths deliberately
//! straddle the SIMD threshold (16 elements) and the GPU's power-of-two
//! boundaries, so the scalar fallbacks, the vector bodies and the bitonic
//! padding all run.

use tensorcrate::projections::{project_onto_ball, project_onto_box, project_onto_capped_simplex};
use tensorcrate::tensors::{Compare, Host, Reduce, SortOrder, Vector};

/// A deterministic spread of signs and magnitudes.
fn values(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * 37 % 23) as f32) * 0.5 - 5.0)
        .collect()
}

fn vector(len: usize) -> Vector<f32, Host> {
    Vector::new(values(len))
}

const LENGTHS: [usize; 8] = [0, 1, 5, 15, 16, 17, 64, 100];

#[test]
fn elementwise_minimum_and_maximum_match_the_scalar_loop() {
    for len in LENGTHS {
        let a = vector(len);
        let b = Vector::new(values(len).into_iter().rev().collect::<Vec<_>>());

        let smallest = a.min(&b);
        let largest = a.max(&b);
        for i in 0..len {
            assert_eq!(smallest[i], a[i].min(b[i]), "min len={len} i={i}");
            assert_eq!(largest[i], a[i].max(b[i]), "max len={len} i={i}");
        }

        // Against a scalar, `max_scalar(0.0)` being the relu.
        let relu = a.max_scalar(0.0);
        let capped = a.min_scalar(1.0);
        for i in 0..len {
            assert_eq!(relu[i], a[i].max(0.0));
            assert_eq!(capped[i], a[i].min(1.0));
        }
    }
}

// `max` then `min` is the definition under test, not a hand-rolled
// `f32::clamp`: that one panics on a NaN bound and propagates a NaN input.
#[allow(clippy::manual_clamp)]
#[test]
fn clamp_confines_every_element_to_the_bounds() {
    for len in LENGTHS {
        let clamped = vector(len).clamp(-1.0, 2.0);
        for (i, &value) in values(len).iter().enumerate() {
            assert_eq!(clamped[i], value.max(-1.0).min(2.0), "len={len} i={i}");
        }
    }

    // The degenerate bound is allowed; the inverted one is not.
    assert_eq!(vector(20).clamp(1.0, 1.0).data(), vec![1.0; 20]);
    let inverted = std::panic::catch_unwind(|| vector(20).clamp(2.0, 1.0));
    assert!(inverted.is_err(), "an inverted range has no projection");
}

#[test]
fn reductions_fold_the_whole_vector() {
    for len in LENGTHS {
        let v = vector(len);
        let raw = values(len);

        let sum: f32 = raw.iter().sum();
        assert!(
            (v.sum() - sum).abs() <= 1e-4 * (1.0 + sum.abs()),
            "len={len}"
        );
        assert_eq!(
            v.minimum(),
            raw.iter().copied().reduce(f32::min),
            "min len={len}"
        );
        assert_eq!(
            v.maximum(),
            raw.iter().copied().reduce(f32::max),
            "max len={len}"
        );
    }

    // An empty vector has no extreme, but its sum is still the identity.
    let empty = Vector::<f32, Host>::new([]);
    assert_eq!(empty.sum(), 0.0);
    assert_eq!(empty.minimum(), None);
    assert_eq!(empty.maximum(), None);
    assert_eq!(empty.reduce(Reduce::Min), f32::INFINITY);
}

#[test]
fn prefix_sum_is_the_running_total() {
    for len in LENGTHS {
        let scanned = vector(len).prefix_sum();
        let mut running = 0.0;
        for (i, &value) in values(len).iter().enumerate() {
            running += value;
            assert!((scanned[i] - running).abs() <= 1e-4, "len={len} i={i}");
        }
    }
}

#[test]
fn sorting_follows_the_total_order() {
    for len in LENGTHS {
        for order in SortOrder::ALL {
            let sorted = vector(len).sorted(order);
            let mut want = values(len);
            want.sort_by(order.comparator());
            assert_eq!(sorted.data(), want, "{order:?} len={len}");
        }
    }

    // The total order puts NaNs at the ends by sign and separates the zeros,
    // which is what `<` cannot do.
    let awkward = Vector::new([f32::NAN, 0.0, -0.0, -f32::NAN, 1.0, f32::NEG_INFINITY]);
    let ascending = awkward.sorted(SortOrder::Ascending);
    let bits = ascending
        .data()
        .iter()
        .map(|v| v.to_bits())
        .collect::<Vec<_>>();
    assert_eq!(bits[0], (-f32::NAN).to_bits(), "-NaN sorts first");
    assert_eq!(bits[1], f32::NEG_INFINITY.to_bits());
    assert_eq!(bits[2], (-0.0f32).to_bits(), "-0.0 precedes +0.0");
    assert_eq!(bits[3], 0.0f32.to_bits());
    assert_eq!(bits[5], f32::NAN.to_bits(), "+NaN sorts last");
}

#[test]
fn sort_by_takes_an_arbitrary_comparator() {
    // A comparator with no relation to the numeric order: by distance from 2.
    let mut v = Vector::new([0.0f32, 5.0, 1.5, 3.0]);
    v.sort_by(|left, right| (left - 2.0).abs().total_cmp(&(right - 2.0).abs()));
    assert_eq!(v.data(), [1.5, 3.0, 0.0, 5.0]);

    // It is not limited to floats: any element type, any ordering.
    let words = Vector::new(["pear", "fig", "banana"]);
    assert_eq!(
        words.sorted_by(|a, b| a.len().cmp(&b.len())).data(),
        ["fig", "pear", "banana"]
    );
}

// ---- the projection ---------------------------------------------------------

/// The scalar statement of the projection: sort, walk the prefix, stop at the
/// first offset that would make a retained entry negative. This is the loop the
/// vectorized version has to reproduce.
fn reference_projection(values: &[f32], cap: f32) -> Vec<f32> {
    let clamped = values
        .iter()
        .map(|value| value.max(0.0))
        .collect::<Vec<_>>();
    if clamped.iter().sum::<f32>() <= cap {
        return clamped;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| right.total_cmp(left));
    let mut running = 0.0;
    let mut offset = 0.0;
    for (index, value) in sorted.iter().enumerate() {
        running += value;
        let candidate = (running - cap) / (index + 1) as f32;
        if candidate > *value {
            break;
        }
        offset = candidate;
    }
    values
        .iter()
        .map(|value| (value - offset).max(0.0))
        .collect()
}

/// Inputs that exercise both branches: budgets that bind, budgets that do not,
/// all-negative vectors, and exact ties in the sorted order.
fn projection_cases() -> Vec<(Vec<f32>, f32)> {
    let mut cases = vec![
        (vec![], 1.0),
        (vec![0.5], 1.0),
        (vec![5.0], 1.0),
        (vec![-1.0, -2.0, -3.0], 1.0),
        (vec![0.1, 0.2, 0.3], 1.0),
        (vec![0.7, 0.5, -0.2], 1.0),
        (vec![2.0, 2.0, 2.0, 2.0], 1.0),
        (vec![1.0, 1.0, 1.0], 3.0),
        (vec![10.0, -4.0, 3.5, 3.5, 0.0], 2.5),
        (vec![0.25; 40], 1.0),
    ];
    for len in [17usize, 33, 64, 129] {
        for cap in [0.5f32, 3.0, 25.0] {
            cases.push((values(len), cap));
        }
    }
    cases
}

fn assert_projection_close(got: &[f32], want: &[f32], label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-4 * (1.0 + w.abs()),
            "{label} at {i}: {g} vs {w}"
        );
    }
}

#[test]
fn capped_simplex_projection_matches_the_scalar_loop() {
    for (input, cap) in projection_cases() {
        let want = reference_projection(&input, cap);
        let got = project_onto_capped_simplex(&Vector::<f32, Host>::new(input.clone()), cap);
        assert_projection_close(
            got.as_slice(),
            &want,
            &format!("cap={cap} len={}", input.len()),
        );
    }
}

#[test]
fn capped_simplex_projection_lands_on_the_constraint_set() {
    for (input, cap) in projection_cases() {
        let projected = project_onto_capped_simplex(&Vector::<f32, Host>::new(input.clone()), cap);
        let total = projected.sum();
        assert!(
            projected.data().iter().all(|&value| value >= 0.0),
            "every entry is non-negative"
        );
        assert!(
            total <= cap + 1e-4,
            "the budget holds: {total} vs {cap} for {input:?}"
        );

        // When the unconstrained clip overshoots, the budget must be spent
        // exactly — that is what makes this a projection onto the face rather
        // than just a feasible point.
        let clipped: f32 = input.iter().map(|v| v.max(0.0)).sum();
        if clipped > cap && !input.is_empty() {
            assert!(
                (total - cap).abs() <= 1e-3 * (1.0 + cap.abs()),
                "the binding case saturates: {total} vs {cap}"
            );
        }
    }
}

#[test]
fn projection_is_idempotent_and_fixes_feasible_points() {
    for (input, cap) in projection_cases() {
        let once = project_onto_capped_simplex(&Vector::<f32, Host>::new(input), cap);
        let twice = project_onto_capped_simplex(&once, cap);
        assert_projection_close(twice.as_slice(), once.as_slice(), "idempotent");
    }
}

#[test]
fn box_and_ball_projections() {
    let v = Vector::<f32, Host>::new([3.0, -4.0, 0.5]);
    assert_eq!(project_onto_box(&v, -1.0, 1.0).data(), [1.0, -1.0, 0.5]);

    // 3-4-5: the norm is 5.0 before the 0.5 is added, so scaling is visible.
    let ray = Vector::<f32, Host>::new([3.0, 4.0]);
    assert_eq!(project_onto_ball(&ray, 10.0).data(), [3.0, 4.0]);
    let shrunk = project_onto_ball(&ray, 1.0);
    assert!((shrunk[0] - 0.6).abs() < 1e-6 && (shrunk[1] - 0.8).abs() < 1e-6);
}

// ---- the same operations, resident on the GPU -------------------------------

#[cfg(all(feature = "metal", target_os = "macos"))]
mod resident {
    use super::*;
    use tensorcrate::tensors::{Kernels, Metal};

    fn assert_close(got: &[f32], want: &[f32], label: &str) {
        assert_eq!(got.len(), want.len(), "{label}: length");
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-4 * (1.0 + w.abs()),
                "{label} at {i}: {g} vs {w}"
            );
        }
    }

    #[test]
    fn comparisons_and_clamping_agree_with_the_host() {
        for len in LENGTHS {
            let host = vector(len);
            let other = Vector::new(values(len).into_iter().rev().collect::<Vec<_>>());
            let resident = host.to_backend::<Metal>();
            let resident_other = other.to_backend::<Metal>();

            for op in Compare::ALL {
                assert_close(
                    resident.compare(&resident_other, op).as_slice(),
                    host.compare(&other, op).as_slice(),
                    &format!("{op:?} len={len}"),
                );
                assert_close(
                    resident.compare_scalar(0.5, op, false).as_slice(),
                    host.compare_scalar(0.5, op, false).as_slice(),
                    &format!("{op:?} scalar len={len}"),
                );
            }

            assert_close(
                resident.clamp(-1.0, 2.0).as_slice(),
                host.clamp(-1.0, 2.0).as_slice(),
                &format!("clamp len={len}"),
            );
            assert_close(
                resident.min(&resident_other).as_slice(),
                host.min(&other).as_slice(),
                &format!("min len={len}"),
            );
        }
    }

    #[test]
    fn reductions_agree_with_the_host() {
        // Lengths past one and two threadgroups, so the tree reduction really
        // does run more than one round.
        for len in [0usize, 1, 17, 256, 257, 1024, 5000] {
            let host = vector(len);
            let resident = host.to_backend::<Metal>();
            for op in Reduce::ALL {
                let want = host.reduce(op);
                let got = resident.reduce(op);
                assert!(
                    (got - want).abs() <= 1e-3 * (1.0 + want.abs()) || (got == want),
                    "{op:?} len={len}: {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn scans_and_sorts_agree_with_the_host() {
        for len in [0usize, 1, 5, 16, 17, 64, 100, 1000] {
            let host = vector(len);
            let resident = host.to_backend::<Metal>();

            assert_close(
                resident.prefix_sum().as_slice(),
                host.prefix_sum().as_slice(),
                &format!("prefix_sum len={len}"),
            );

            for order in SortOrder::ALL {
                // Sorting is a permutation, so this one is exact rather than
                // approximate — every bit has to match.
                assert_eq!(
                    resident.sorted(order).to_vec(),
                    host.sorted(order).to_vec(),
                    "{order:?} len={len}"
                );
            }
        }
    }

    #[test]
    fn the_gpu_sort_orders_nans_like_the_host_one() {
        let awkward = Vector::new([f32::NAN, 0.0, -0.0, -f32::NAN, 1.0, f32::NEG_INFINITY, 2.5]);
        for order in SortOrder::ALL {
            let host = awkward.sorted(order);
            let resident = awkward.to_backend::<Metal>().sorted(order);
            let host_bits = host.data().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            let gpu_bits = resident
                .to_vec()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>();
            assert_eq!(gpu_bits, host_bits, "{order:?}");
        }
    }

    #[test]
    fn results_stay_in_shared_memory() {
        let resident = vector(64).to_backend::<Metal>();
        if !resident.is_device_resident() {
            // No Metal device on this machine: the operations above still ran,
            // on the CPU fallback, and matched.
            return;
        }
        assert!(resident.clamp(0.0, 1.0).is_device_resident());
        assert!(resident.max_scalar(0.0).is_device_resident());
        assert!(resident.prefix_sum().is_device_resident());
        assert!(resident.sorted(SortOrder::Descending).is_device_resident());
        assert!(
            project_onto_capped_simplex(&resident, 1.0).is_device_resident(),
            "the whole projection stays on the device"
        );
    }

    #[test]
    fn the_projection_agrees_across_backends() {
        for (input, cap) in projection_cases() {
            let host = Vector::<f32, Host>::new(input.clone());
            let want = project_onto_capped_simplex(&host, cap);
            let got = project_onto_capped_simplex(&host.to_backend::<Metal>(), cap);
            assert_close(
                got.as_slice(),
                want.as_slice(),
                &format!("cap={cap} len={}", input.len()),
            );
        }
    }

    #[test]
    fn the_kernels_trait_reaches_both_backends() {
        // What generic code sees: one source, either backend.
        fn normalize<B: Kernels>(v: &Vector<f32, B>) -> f32 {
            B::vector_reduce(&B::vector_clamp(v, 0.0, 1.0), Reduce::Sum)
        }
        let host = vector(50);
        let want = normalize(&host);
        let got = normalize(&host.to_backend::<Metal>());
        assert!((got - want).abs() <= 1e-4 * (1.0 + want.abs()));
    }
}
