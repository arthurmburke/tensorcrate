//! The `Metal` tensor backend: same answers as `Host`, different memory.
//!
//! These tests run everywhere the backend compiles, with or without a Metal
//! device — a machine without one falls back to CPU storage and CPU kernels, and
//! the results are supposed to be identical either way. The claim that
//! intermediates *stay* in shared memory is checked with `is_device_resident`,
//! guarded on the inputs actually having landed there.

#![cfg(all(feature = "metal", target_os = "macos"))]

use tensorcrate::numbers::{bf16, f16};
use tensorcrate::tensors::{BinaryOp, Host, Matrix, Metal, Vector};

/// Deterministic filler with a mix of signs and magnitudes, all integral so
/// GPU and CPU accumulation orders agree exactly.
fn value(index: usize) -> f32 {
    (index % 7) as f32 - 3.0
}

fn vector(len: usize) -> Vector<f32, Host> {
    Vector::new((0..len).map(value).collect::<Vec<_>>())
}

fn matrix(rows: usize, cols: usize) -> Matrix<f32, Host> {
    Matrix::from_rows((0..rows).map(|row| {
        (0..cols)
            .map(|col| value(row * cols + col + row))
            .collect::<Vec<_>>()
    }))
}

#[test]
fn moving_between_backends_preserves_the_elements() {
    let v = vector(64);
    let resident = v.to_backend::<Metal>();
    assert_eq!(resident.to_backend::<Host>(), v);
    assert_eq!(resident.to_vec(), v.data());
    assert_eq!(resident.as_slice(), v.as_slice());
    assert_eq!(resident.len(), 64);

    let m = matrix(9, 5);
    let resident = m.to_backend::<Metal>();
    assert_eq!(resident.to_backend::<Host>(), m);
    assert_eq!(resident.to_backend::<Host>().to_rows(), m.to_rows());
    assert_eq!(resident.shape(), (9, 5));

    // A vector and a matrix of the same element count share a flat layout.
    assert_eq!(
        matrix(3, 3).to_backend::<Metal>().as_slice(),
        matrix(3, 3).as_slice()
    );

    // Empty shapes are still shapes.
    let empty = Vector::<f32>::new([]).to_backend::<Metal>();
    assert!(empty.is_empty());
    assert_eq!(empty.to_backend::<Host>(), Vector::<f32>::new([]));
}

#[test]
fn filled_allocates_directly_on_the_backend() {
    let zeros = Vector::<f32, Metal>::filled(33, 0.0);
    assert_eq!(zeros.to_backend::<Host>(), Vector::<f32>::zeros(33));

    let sevens = Matrix::<f32, Metal>::filled(4, 6, 7.0);
    assert_eq!(sevens.to_backend::<Host>().to_rows(), [[7.0f32; 6]; 4]);
}

/// `Matrix` and `Vector` results on both backends, as host rows.
fn both<T: tensorcrate::numbers::Real>(
    on_host: impl Fn() -> Matrix<T, Host>,
    on_metal: impl Fn() -> Matrix<T, Metal>,
) -> (Vec<Vec<T>>, Vec<Vec<T>>) {
    (
        on_host().to_rows(),
        on_metal().to_backend::<Host>().to_rows(),
    )
}

#[test]
fn stacking_vectors_uses_row_major_matrix_layout() {
    let vectors = [
        Vector::new([-3.0f32, -2.0, -1.0]),
        Vector::new([4.0, 5.0, 6.0]),
    ];
    let resident = vectors.each_ref().map(|v| v.to_backend::<Metal>());

    let (host, metal) = both(|| Vector::vstack(&vectors), || Vector::vstack(&resident));
    assert_eq!(host, [[-3.0, -2.0, -1.0], [4.0, 5.0, 6.0]]);
    assert_eq!(metal, host);

    let columns = [
        Vector::new([1.0f32, 2.0]),
        Vector::new([3.0, 4.0]),
        Vector::new([5.0, 6.0]),
    ];
    let resident = columns.each_ref().map(|v| v.to_backend::<Metal>());
    let (host, metal) = both(|| Vector::hstack(&columns), || Vector::hstack(&resident));
    assert_eq!(host, [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]);
    assert_eq!(metal, host);

    // Empty vectors still stack: to a matrix with a zero extent.
    let empty = [Vector::<f32, Metal>::filled(0, 0.0), Vector::filled(0, 0.0)];
    assert_eq!(Vector::vstack(&empty).shape(), (2, 0));
    assert_eq!(Vector::hstack(&empty).shape(), (0, 2));
}

#[test]
#[should_panic(expected = "differ in length")]
fn stacking_vectors_of_different_lengths_panics() {
    Vector::vstack([Vector::new([1.0f32]), Vector::new([1.0, 2.0])]);
}

#[test]
#[should_panic(expected = "no vectors to stack")]
fn stacking_no_vectors_panics() {
    Vector::<f32, Host>::hstack(Vec::<Vector<f32>>::new());
}

#[test]
fn hstack_and_vstack_place_matrices_in_row_major_order() {
    let a = Matrix::from_rows([[1.0f32, 2.0], [3.0, 4.0]]);
    let b = Matrix::from_rows([[5.0f32], [6.0]]);
    let (ra, rb) = (a.to_backend::<Metal>(), b.to_backend::<Metal>());
    let (host, metal) = both(|| Matrix::hstack([&a, &b]), || Matrix::hstack([&ra, &rb]));
    assert_eq!(host, [[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]]);
    assert_eq!(metal, host);

    let top = Matrix::from_rows([[1.0f32, 2.0]]);
    let bottom = Matrix::from_rows([[3.0f32, 4.0], [5.0, 6.0]]);
    let (rt, rb) = (top.to_backend::<Metal>(), bottom.to_backend::<Metal>());
    let (host, metal) = both(
        || Matrix::vstack([&top, &bottom]),
        || Matrix::vstack([&rt, &rb]),
    );
    assert_eq!(host, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    assert_eq!(metal, host);

    // A matrix with a zero extent contributes nothing.
    let no_columns = Matrix::<f32, Metal>::filled(2, 0, 0.0);
    let right = Matrix::from_rows([[7.0f32], [8.0]]).to_backend::<Metal>();
    let joined = Matrix::hstack([&no_columns, &right]);
    assert_eq!(joined.to_backend::<Host>().to_rows(), [[7.0], [8.0]]);

    let no_rows = Matrix::<f32, Metal>::filled(0, 2, 0.0);
    let below = Matrix::from_rows([[9.0f32, 10.0]]).to_backend::<Metal>();
    let joined = Matrix::vstack([&no_rows, &below]);
    assert_eq!(joined.to_backend::<Host>().to_rows(), [[9.0, 10.0]]);
}

#[test]
fn stacking_many_matrices_preserves_input_order() {
    let matrices = (0..3)
        .map(|k| {
            Matrix::from_flat(
                2,
                2,
                (1..=4).map(|i| (4 * k + i) as f32).collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    let resident = matrices
        .iter()
        .map(|m| m.to_backend::<Metal>())
        .collect::<Vec<_>>();

    let (host, metal) = both(|| Matrix::hstack(&matrices), || Matrix::hstack(&resident));
    assert_eq!(
        host,
        [
            [1.0, 2.0, 5.0, 6.0, 9.0, 10.0],
            [3.0, 4.0, 7.0, 8.0, 11.0, 12.0]
        ]
    );
    assert_eq!(metal, host);

    let (host, metal) = both(|| Matrix::vstack(&matrices), || Matrix::vstack(&resident));
    assert_eq!(
        host.concat(),
        (1..=12).map(|i| i as f32).collect::<Vec<_>>()
    );
    assert_eq!(metal, host);
}

#[test]
#[should_panic(expected = "differ in their columns")]
fn vstack_of_matrices_with_different_columns_panics() {
    let a = Matrix::<f32>::from_rows([[1.0, 2.0]]);
    let b = Matrix::<f32>::from_rows([[1.0, 2.0, 3.0]]);
    Matrix::vstack([&a, &b]);
}

#[test]
#[should_panic(expected = "differ in their rows")]
fn hstack_of_matrices_with_different_rows_panics() {
    let a = Matrix::<f32>::from_rows([[1.0], [2.0]]);
    let b = Matrix::<f32>::from_rows([[1.0]]);
    Matrix::hstack([&a, &b]);
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
    assert_eq!(row.shape(), (1, 3));
    assert_eq!(row.to_backend::<Host>().to_rows(), [[4.0, 5.0, 6.0]]);

    let resident = Vector::new([7.0f32, 8.0, 9.0]).to_backend::<Metal>();
    let was_resident = resident.is_device_resident();
    let column = resident.into_column_matrix();
    assert_eq!(column.is_device_resident(), was_resident);
    assert_eq!(column.shape(), (3, 1));
    assert_eq!(column.to_backend::<Host>().to_rows(), [[7.0], [8.0], [9.0]]);
}

#[test]
fn products_match_the_host_backend() {
    let a = matrix(12, 20);
    let b = matrix(20, 7);
    let v = vector(20);
    let vector_addend = vector(12);
    let matrix_addend = matrix(12, 7);
    let row = vector(12);

    let (ga, gb) = (a.to_backend::<Metal>(), b.to_backend::<Metal>());
    assert_eq!(ga.matmul(&gb).to_backend::<Host>(), a.matmul(&b));
    assert_eq!(
        ga.matvec(&v.to_backend()).to_backend::<Host>(),
        a.matvec(&v)
    );
    assert_eq!(
        ga.matvec_add(&v.to_backend(), vector_addend.to_backend())
            .to_backend::<Host>(),
        a.matvec_add(&v, vector_addend)
    );
    assert_eq!(
        ga.matmul_add(&gb, matrix_addend.to_backend())
            .to_backend::<Host>(),
        a.matmul_add(&b, matrix_addend)
    );
    assert_eq!(
        row.to_backend::<Metal>().vecmat(&ga).to_backend::<Host>(),
        row.vecmat(&a)
    );
    assert_eq!(ga.transpose().to_backend::<Host>(), a.transpose());

    let u = vector(64);
    assert_eq!(
        u.to_backend::<Metal>().dot(&u.to_backend::<Metal>()),
        u.dot(&u)
    );

    // A degenerate inner dimension is an empty sum, not a failure.
    // `from_rows` cannot infer a row type from no rows at all, so a zero extent
    // is spelled with the flat constructor.
    let thin = Matrix::<f32>::from_flat(3, 0, []).to_backend::<Metal>();
    let wide = Matrix::<f32>::from_flat(0, 3, []).to_backend::<Metal>();
    assert_eq!(
        thin.matmul(&wide).to_backend::<Host>().to_rows(),
        [[0.0f32; 3]; 3]
    );
}

#[test]
fn low_precision_storage_supports_compact_and_f32_products() {
    assert_eq!(size_of::<f16>(), 2);
    assert_eq!(size_of::<bf16>(), 2);

    let left = Matrix::from_rows([
        [1.0f32, 2.0, -1.0, 0.5],
        [0.0, 3.0, 2.0, -2.0],
        [4.0, -1.0, 0.5, 2.0],
    ]);
    let right = Matrix::from_rows([[2.0f32, -1.0], [0.5, 3.0], [-2.0, 1.0], [4.0, 0.25]]);
    let expected = left.matmul(&right);

    let left_f16 = Matrix::<f16, Metal>::from_f32(&left);
    let right_f16 = Matrix::<f16, Metal>::from_f32(&right);
    let compact_f16 = left_f16.matmul(&right_f16).to_f32::<Host>();
    let wide_f16 = left_f16.matmul_f32(&right_f16).to_backend::<Host>();
    for ((compact, wide), expected) in compact_f16
        .as_slice()
        .iter()
        .zip(wide_f16.as_slice())
        .zip(expected.as_slice())
    {
        assert!((compact - expected).abs() < 0.02);
        assert!((wide - expected).abs() < 1e-4);
    }

    let left_bf16 = Matrix::<bf16, Metal>::from_f32(&left);
    let right_bf16 = Matrix::<bf16, Metal>::from_f32(&right);
    let compact_bf16 = left_bf16.matmul(&right_bf16).to_f32::<Host>();
    let wide_bf16 = left_bf16.matmul_f32(&right_bf16).to_backend::<Host>();
    for ((compact, wide), expected) in compact_bf16
        .as_slice()
        .iter()
        .zip(wide_bf16.as_slice())
        .zip(expected.as_slice())
    {
        assert!((compact - expected).abs() < 0.1);
        assert!((wide - expected).abs() < 1e-4);
    }
}

#[test]
fn large_products_agree_with_the_host_within_float_tolerance() {
    // Big enough that the tiled GPU kernel and the CPU loops sum in different
    // orders.
    const N: usize = 96;
    let a = Matrix::<f32>::from_rows((0..N).map(|row| {
        (0..N)
            .map(|col| ((row * N + col) % 23) as f32 * 0.25 - 2.0)
            .collect::<Vec<_>>()
    }));
    let b = a.transpose();

    let host = a.matmul(&b);
    let resident = a
        .to_backend::<Metal>()
        .matmul(&b.to_backend::<Metal>())
        .to_backend::<Host>();
    for (actual, expected) in resident.data().iter().zip(host.data()) {
        assert!(
            (actual - expected).abs() < 1e-2,
            "resident={actual} host={expected}"
        );
    }
}

#[test]
fn elementwise_operators_and_broadcasts_match_the_host_backend() {
    let a = vector(48);
    let b = vector(48).broadcast_right(9.0, BinaryOp::Add); // no zeros
    let (ga, gb) = (a.to_backend::<Metal>(), b.to_backend::<Metal>());

    assert_eq!((&ga + &gb).to_backend::<Host>(), a.clone() + b.clone());
    assert_eq!((&ga - &gb).to_backend::<Host>(), a.clone() - b.clone());
    assert_eq!((&ga * &gb).to_backend::<Host>(), a.clone() * b.clone());
    assert_eq!((&ga / &gb).to_backend::<Host>(), a.clone() / b.clone());
    // The shaders have no remainder kernel, so this one falls back to the host
    // and comes back; the answer still has to match.
    assert_eq!((&ga % &gb).to_backend::<Host>(), a.clone() % b);
    assert_eq!((-&ga).to_backend::<Host>(), -a.clone());
    assert_eq!(ga.scale(2.5).to_backend::<Host>(), a.scale(2.5));
    assert_eq!(
        ga.broadcast_left(1.0, BinaryOp::Sub).to_backend::<Host>(),
        a.broadcast_left(1.0, BinaryOp::Sub)
    );

    let m = matrix(8, 8);
    let n = matrix(8, 8).broadcast_right(9.0, BinaryOp::Add);
    let (gm, gn) = (m.to_backend::<Metal>(), n.to_backend::<Metal>());
    assert_eq!((&gm + &gn).to_backend::<Host>(), m.clone() + n.clone());
    assert_eq!((&gm * &gn).to_backend::<Host>(), m.clone() * n.clone());
    assert_eq!((&gm % &gn).to_backend::<Host>(), m.clone() % n.clone());
    assert_eq!(gm.scale(-1.0).to_backend::<Host>(), m.scale(-1.0));

    // The by-value operators consume their operands, like the host ones.
    assert_eq!((gm + gn).to_backend::<Host>(), m + n);
}

#[test]
fn a_chain_of_operations_stays_in_shared_memory() {
    let a = matrix(32, 32).to_backend::<Metal>();
    if !a.is_device_resident() {
        eprintln!("no Metal device; skipping the residency check");
        return;
    }

    let b = matrix(32, 32).to_backend::<Metal>();
    let v = vector(32).to_backend::<Metal>();
    let vector_addend = vector(32).to_backend::<Metal>();
    let matrix_addend = matrix(32, 32).to_backend::<Metal>();

    // Nothing in here should touch the host: every intermediate is a shared
    // allocation produced by a kernel that read shared allocations.
    let product = a.matmul(&b);
    assert!(product.is_device_resident());
    let scaled = product.scale(0.5);
    assert!(scaled.is_device_resident());
    let summed = &scaled + &product;
    assert!(summed.is_device_resident());
    assert!(summed.matvec(&v).is_device_resident());
    assert!(a.matvec_add(&v, vector_addend).is_device_resident());
    assert!(a.matmul_add(&b, matrix_addend).is_device_resident());
    assert!((&v * &v).is_device_resident());

    // `%` has no kernel: it round-trips, and the result comes back resident.
    assert!((&v % &v.broadcast_right(9.0, BinaryOp::Add)).is_device_resident());
}

#[test]
fn resident_tensors_clone_compare_and_print_like_host_ones() {
    let v = vector(5);
    let resident = v.to_backend::<Metal>();

    let copy = resident.clone();
    assert_eq!(copy, resident);
    assert_eq!(copy.to_backend::<Host>(), v);
    assert_ne!(resident, resident.scale(2.0));

    assert_eq!(resident.to_string(), v.to_string());
    let m = matrix(3, 4);
    assert_eq!(m.to_backend::<Metal>().to_string(), m.to_string());
    assert!(format!("{resident:?}").starts_with("Vector"));
}

#[test]
fn host_tensors_are_unchanged_by_the_backend_parameter() {
    // The default backend owns a heap allocation now, so a host tensor clones
    // rather than copies — but the values and the shape are what they were.
    let v = vector(4);
    let copied = v.clone();
    assert_eq!(copied, v);
    assert_eq!(copied.len(), 4);

    // The shape travels with the value, so it is the same size whatever the
    // extents — which is the whole point of holding the dimensions at runtime.
    assert_eq!(size_of_val(&matrix(3, 3)), size_of_val(&matrix(64, 64)));
}
