//! The `Metal` tensor backend: same answers as `Host`, different memory.
//!
//! These tests run everywhere the backend compiles, with or without a Metal
//! device — a machine without one falls back to CPU storage and CPU kernels, and
//! the results are supposed to be identical either way. The claim that
//! intermediates *stay* in shared memory is checked with `is_device_resident`,
//! guarded on the inputs actually having landed there.

#![cfg(all(feature = "metal", target_os = "macos"))]

use rinterp::tensors::{Backend, Host, Matrix, Metal, Vector};

/// Deterministic filler with a mix of signs and magnitudes, all integral so
/// GPU and CPU accumulation orders agree exactly.
fn value(index: usize) -> f32 {
    (index % 7) as f32 - 3.0
}

fn vector<const N: usize>() -> Vector<f32, N, Host> {
    Vector::new(std::array::from_fn(value))
}

fn matrix<const R: usize, const C: usize>() -> Matrix<f32, R, C, Host> {
    Matrix::from_rows(std::array::from_fn(|row| {
        std::array::from_fn(|col| value(row * C + col + row))
    }))
}

#[test]
fn moving_between_backends_preserves_the_elements() {
    let v = vector::<64>();
    let resident = v.to_backend::<Metal>();
    assert_eq!(resident.to_backend::<Host>(), v);
    assert_eq!(resident.to_array(), *v.data());
    assert_eq!(resident.as_slice(), v.as_slice());
    assert_eq!(resident.len(), 64);

    let m = matrix::<9, 5>();
    let resident = m.to_backend::<Metal>();
    assert_eq!(resident.to_backend::<Host>(), m);
    assert_eq!(resident.to_rows(), *m.data());
    assert_eq!(resident.shape(), (9, 5));

    // A vector and a matrix of the same element count share a flat layout.
    assert_eq!(
        matrix::<3, 3>().to_backend::<Metal>().as_slice(),
        matrix::<3, 3>().as_slice()
    );

    // Empty shapes are still shapes.
    let empty = Vector::<f32, 0>::new([]).to_backend::<Metal>();
    assert!(empty.is_empty());
    assert_eq!(empty.to_backend::<Host>(), Vector::<f32, 0>::new([]));
}

#[test]
fn filled_allocates_directly_on_the_backend() {
    let zeros = Vector::<f32, 33, Metal>::filled(0.0);
    assert_eq!(zeros.to_backend::<Host>(), Vector::<f32, 33>::zeros());

    let sevens = Matrix::<f32, 4, 6, Metal>::filled(7.0);
    assert_eq!(sevens.to_rows(), [[7.0f32; 6]; 4]);
}

#[test]
fn stacking_vectors_uses_row_major_matrix_layout() {
    let rows = [
        Metal::store_vector::<3>(&[-3.0, -2.0, -1.0]),
        Metal::store_vector::<3>(&[4.0, 5.0, 6.0]),
    ];
    let vertical = Metal::vstack::<2, 3>(rows);
    assert_eq!(
        Metal::matrix_slice::<2, 3>(&vertical),
        &[-3.0, -2.0, -1.0, 4.0, 5.0, 6.0]
    );

    let columns = [
        Metal::store_vector::<2>(&[1.0, 2.0]),
        Metal::store_vector::<2>(&[3.0, 4.0]),
        Metal::store_vector::<2>(&[5.0, 6.0]),
    ];
    let horizontal = Metal::hstack::<2, 3>(columns);
    assert_eq!(
        Metal::matrix_slice::<2, 3>(&horizontal),
        &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
    );

    let empty = Metal::vstack::<0, 3>([]);
    assert!(Metal::matrix_slice::<0, 3>(&empty).is_empty());
    let empty = Metal::hstack::<2, 0>([]);
    assert!(Metal::matrix_slice::<2, 0>(&empty).is_empty());
}

#[test]
fn concatenating_and_stacking_matrices_preserves_row_major_layout() {
    let host_concat = Host::concat::<2, 2, 1>([[1.0, 2.0], [3.0, 4.0]], [[5.0], [6.0]]);
    assert_eq!(host_concat, [[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]]);

    let host_stack = Host::stack::<1, 2, 2>([[1.0, 2.0]], [[3.0, 4.0], [5.0, 6.0]]);
    assert_eq!(host_stack, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);

    let left = Metal::store_matrix::<2, 2>(&[1.0, 2.0, 3.0, 4.0]);
    let right = Metal::store_matrix::<2, 1>(&[5.0, 6.0]);
    let concat = Metal::concat::<2, 2, 1>(left, right);
    assert_eq!(
        Metal::matrix_slice::<2, 3>(&concat),
        &[1.0, 2.0, 5.0, 3.0, 4.0, 6.0]
    );

    let top = Metal::store_matrix::<1, 2>(&[1.0, 2.0]);
    let bottom = Metal::store_matrix::<2, 2>(&[3.0, 4.0, 5.0, 6.0]);
    let stack = Metal::stack::<1, 2, 2>(top, bottom);
    assert_eq!(
        Metal::matrix_slice::<3, 2>(&stack),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );

    let empty_left = Metal::store_matrix::<2, 0>(&[]);
    let right = Metal::store_matrix::<2, 1>(&[7.0, 8.0]);
    let concat = Metal::concat::<2, 0, 1>(empty_left, right);
    assert_eq!(Metal::matrix_slice::<2, 1>(&concat), &[7.0, 8.0]);

    let empty_top = Metal::store_matrix::<0, 2>(&[]);
    let bottom = Metal::store_matrix::<1, 2>(&[9.0, 10.0]);
    let stack = Metal::stack::<0, 1, 2>(empty_top, bottom);
    assert_eq!(Metal::matrix_slice::<1, 2>(&stack), &[9.0, 10.0]);
}

#[test]
fn merging_matrix_collections_preserves_input_order() {
    let matrices = [
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
        [[9.0, 10.0], [11.0, 12.0]],
    ];
    assert_eq!(
        Host::hmerge::<2, 2, 3>(matrices),
        [
            [1.0, 2.0, 5.0, 6.0, 9.0, 10.0],
            [3.0, 4.0, 7.0, 8.0, 11.0, 12.0],
        ]
    );
    assert_eq!(
        Host::vmerge::<2, 2, 3>(matrices),
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
        ]
    );

    let matrices = [
        Metal::store_matrix::<2, 2>(&[1.0, 2.0, 3.0, 4.0]),
        Metal::store_matrix::<2, 2>(&[5.0, 6.0, 7.0, 8.0]),
        Metal::store_matrix::<2, 2>(&[9.0, 10.0, 11.0, 12.0]),
    ];
    let horizontal = Metal::hmerge::<2, 2, 3>(matrices);
    assert_eq!(
        Metal::matrix_slice::<2, 6>(&horizontal),
        &[
            1.0, 2.0, 5.0, 6.0, 9.0, 10.0, 3.0, 4.0, 7.0, 8.0, 11.0, 12.0
        ]
    );

    let matrices = [
        Metal::store_matrix::<2, 2>(&[1.0, 2.0, 3.0, 4.0]),
        Metal::store_matrix::<2, 2>(&[5.0, 6.0, 7.0, 8.0]),
        Metal::store_matrix::<2, 2>(&[9.0, 10.0, 11.0, 12.0]),
    ];
    let vertical = Metal::vmerge::<2, 2, 3>(matrices);
    assert_eq!(
        Metal::matrix_slice::<6, 2>(&vertical),
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0
        ]
    );

    assert!(Metal::matrix_slice::<2, 0>(&Metal::hmerge::<2, 3, 0>([])).is_empty());
    assert!(Metal::matrix_slice::<0, 3>(&Metal::vmerge::<2, 3, 0>([])).is_empty());
}

#[test]
fn vectors_convert_to_row_and_column_matrices() {
    let row = Vector::new([1.0f32, 2.0, 3.0]).into_row_matrix();
    assert_eq!(row.to_rows(), [[1.0, 2.0, 3.0]]);

    let column = Vector::new([1.0f32, 2.0, 3.0]).into_column_matrix();
    assert_eq!(column.to_rows(), [[1.0], [2.0], [3.0]]);

    let resident = Vector::new([4.0f32, 5.0, 6.0]).to_backend::<Metal>();
    let was_resident = resident.is_device_resident();
    let row = resident.into_row_matrix();
    assert_eq!(row.is_device_resident(), was_resident);
    assert_eq!(row.to_rows(), [[4.0, 5.0, 6.0]]);

    let resident = Vector::new([7.0f32, 8.0, 9.0]).to_backend::<Metal>();
    let was_resident = resident.is_device_resident();
    let column = resident.into_column_matrix();
    assert_eq!(column.is_device_resident(), was_resident);
    assert_eq!(column.to_rows(), [[7.0], [8.0], [9.0]]);
}

#[test]
fn products_match_the_host_backend() {
    let a = matrix::<12, 20>();
    let b = matrix::<20, 7>();
    let v = vector::<20>();
    let row = vector::<12>();

    let (ga, gb) = (a.to_backend::<Metal>(), b.to_backend::<Metal>());
    assert_eq!(ga.matmul(&gb).to_backend::<Host>(), a.matmul(&b));
    assert_eq!(
        ga.matvec(&v.to_backend()).to_backend::<Host>(),
        a.matvec(&v)
    );
    assert_eq!(
        row.to_backend::<Metal>().vecmat(&ga).to_backend::<Host>(),
        row.vecmat(&a)
    );
    assert_eq!(ga.transpose().to_backend::<Host>(), a.transpose());

    let u = vector::<64>();
    assert_eq!(
        u.to_backend::<Metal>().dot(&u.to_backend::<Metal>()),
        u.dot(&u)
    );

    // A degenerate inner dimension is an empty sum, not a failure.
    let thin = Matrix::<f32, 3, 0>::from_rows([[], [], []]).to_backend::<Metal>();
    let wide = Matrix::<f32, 0, 3>::from_rows([]).to_backend::<Metal>();
    assert_eq!(thin.matmul(&wide).to_rows(), [[0.0f32; 3]; 3]);
}

#[test]
fn large_products_agree_with_the_host_within_float_tolerance() {
    // Big enough that the host backend offloads to Metal itself, and big enough
    // that the tiled GPU kernel and the CPU loops sum in different orders.
    const N: usize = 96;
    let a = Matrix::<f32, N, N>::from_rows(std::array::from_fn(|row| {
        std::array::from_fn(|col| ((row * N + col) % 23) as f32 * 0.25 - 2.0)
    }));
    let b = a.transpose();

    let host = a.matmul(&b);
    let resident = a
        .to_backend::<Metal>()
        .matmul(&b.to_backend::<Metal>())
        .to_backend::<Host>();
    for (row, expected_row) in resident.data().iter().zip(host.data()) {
        for (actual, expected) in row.iter().zip(expected_row) {
            assert!(
                (actual - expected).abs() < 1e-2,
                "resident={actual} host={expected}"
            );
        }
    }
}

#[test]
fn elementwise_operators_and_broadcasts_match_the_host_backend() {
    let a = vector::<48>();
    let b = vector::<48>().broadcast_right(9.0, 0); // no zeros, so `/` and `%` are safe
    let (ga, gb) = (a.to_backend::<Metal>(), b.to_backend::<Metal>());

    assert_eq!((&ga + &gb).to_backend::<Host>(), a + b);
    assert_eq!((&ga - &gb).to_backend::<Host>(), a - b);
    assert_eq!((&ga * &gb).to_backend::<Host>(), a * b);
    assert_eq!((&ga / &gb).to_backend::<Host>(), a / b);
    // The shaders have no remainder kernel, so this one falls back to the host
    // and comes back; the answer still has to match.
    assert_eq!((&ga % &gb).to_backend::<Host>(), a % b);
    assert_eq!((-&ga).to_backend::<Host>(), -a);
    assert_eq!(ga.scale(2.5).to_backend::<Host>(), a.scale(2.5));
    assert_eq!(
        ga.broadcast_left(1.0, 1).to_backend::<Host>(),
        a.broadcast_left(1.0, 1)
    );

    let m = matrix::<8, 8>();
    let n = matrix::<8, 8>().broadcast_right(9.0, 0);
    let (gm, gn) = (m.to_backend::<Metal>(), n.to_backend::<Metal>());
    assert_eq!((&gm + &gn).to_backend::<Host>(), m + n);
    assert_eq!((&gm * &gn).to_backend::<Host>(), m * n);
    assert_eq!((&gm % &gn).to_backend::<Host>(), m % n);
    assert_eq!(gm.scale(-1.0).to_backend::<Host>(), m.scale(-1.0));

    // The by-value operators consume their operands, like the host ones.
    assert_eq!((gm + gn).to_backend::<Host>(), m + n);
}

#[test]
fn a_chain_of_operations_stays_in_shared_memory() {
    let a = matrix::<32, 32>().to_backend::<Metal>();
    if !a.is_device_resident() {
        eprintln!("no Metal device; skipping the residency check");
        return;
    }

    let b = matrix::<32, 32>().to_backend::<Metal>();
    let v = vector::<32>().to_backend::<Metal>();

    // Nothing in here should touch the host: every intermediate is a shared
    // allocation produced by a kernel that read shared allocations.
    let product = a.matmul(&b);
    assert!(product.is_device_resident());
    let scaled = product.scale(0.5);
    assert!(scaled.is_device_resident());
    let summed = &scaled + &product;
    assert!(summed.is_device_resident());
    assert!(summed.matvec(&v).is_device_resident());
    assert!((&v * &v).is_device_resident());

    // `%` has no kernel: it round-trips, and the result comes back resident.
    assert!((&v % &v.broadcast_right(9.0, 0)).is_device_resident());
}

#[test]
fn resident_tensors_clone_compare_and_print_like_host_ones() {
    let v = vector::<5>();
    let resident = v.to_backend::<Metal>();

    let copy = resident.clone();
    assert_eq!(copy, resident);
    assert_eq!(copy.to_backend::<Host>(), v);
    assert_ne!(resident, resident.scale(2.0));

    assert_eq!(resident.to_string(), v.to_string());
    let m = matrix::<3, 4>();
    assert_eq!(m.to_backend::<Metal>().to_string(), m.to_string());
    assert!(format!("{resident:?}").starts_with("Vector"));
}

#[test]
fn host_tensors_are_unchanged_by_the_backend_parameter() {
    // The default backend is still a stack array: `Copy`, same size as `[f32; N]`.
    let v = vector::<4>();
    let copied = v;
    assert_eq!(copied, v);
    assert_eq!(size_of_val(&v), size_of::<[f32; 4]>());
    assert_eq!(size_of::<Matrix<f32, 3, 3>>(), size_of::<[[f32; 3]; 3]>());
}
