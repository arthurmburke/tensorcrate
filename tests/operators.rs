use tensorcrate::math;
use tensorcrate::numbers::{Complex, Conj, Field};
use tensorcrate::tensors::{
    Adjoint, BlockOperator, CooBuilder, CsrMatrix, DiagonalOperator, Dotc, ElementOperator, Host,
    LinearOperator, Matrix, Tensor, Vector,
};

fn close<T: Field>(actual: &[T], expected: &[T], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).magnitude() <= tolerance * (1.0 + expected.magnitude()),
            "operator result differs"
        );
    }
}

#[test]
fn assembly_compresses_shared_entries_and_preserves_reusable_pattern() {
    let mut builder = CooBuilder::new(3, 3);
    builder.add_local(&[0, 1], &[0, 1], &[2.0, -1.0, -1.0, 2.0]);
    builder.add_local(&[1, 2], &[1, 2], &[2.0, -1.0, -1.0, 2.0]);
    builder.push(0, 2, 1.0);
    builder.push(0, 2, -1.0);
    let mut a = builder.finish();
    assert_eq!(a.row_offsets(), [0, 2, 5, 7]);
    assert_eq!(a.column_indices(), [0, 1, 0, 1, 2, 1, 2]);
    assert_eq!(a.diagonal(), [2.0, 4.0, 2.0]);
    a.values_mut()[0] = 0.0;
    assert_eq!(a.nnz(), 7);
    let x = Vector::new([1.0, 2.0, 3.0]);
    let mut y = Vector::zeros(3);
    a.apply(&x, &mut y);
    close(y.as_slice(), a.to_dense().matvec(&x).as_slice(), 1e-12);
}

#[test]
fn csr_rectangular_empty_and_adjoint_products_match_dense() {
    type C = Complex<f64>;
    let a = CsrMatrix::from_triplets(
        4,
        3,
        [
            (0, 2, C::new(2.0, 1.0)),
            (2, 0, C::new(-1.0, 3.0)),
            (2, 2, C::new(1.0, -2.0)),
        ],
    );
    let x = Vector::new([C::new(1.0, 2.0); 3]);
    let mut y = Vector::filled(4, C::new(f64::NAN, f64::NAN));
    a.apply(&x, &mut y);
    close(y.as_slice(), a.to_dense().matvec(&x).as_slice(), 1e-12);
    let mut z = Vector::zeros(3);
    a.apply_adjoint(&y, &mut z);
    close(
        z.as_slice(),
        a.to_dense().adjoint().matvec(&y).as_slice(),
        1e-12,
    );
    close(
        a.adjoint().to_dense().as_slice(),
        a.to_dense().adjoint().as_slice(),
        1e-12,
    );
    a.values();
    let empty = CsrMatrix::<f64>::from_parts(0, 3, vec![0], vec![], vec![]);
    let mut out = Vector::filled(3, f64::NAN);
    empty.apply_adjoint(&Vector::zeros(0), &mut out);
    assert_eq!(out.as_slice(), [0.0; 3]);
    assert_eq!(empty.transpose().shape(), (3, 0));
}

#[test]
fn malformed_csr_is_rejected() {
    for (offsets, columns) in [
        (vec![1, 2, 2], vec![0, 1]),
        (vec![0, 2, 1], vec![0]),
        (vec![0, 2, 2], vec![1, 0]),
        (vec![0, 2, 2], vec![0, 0]),
        (vec![0, 1, 1], vec![2]),
        (vec![0, 3, 2], vec![0, 1]),
    ] {
        let values = vec![1.0; columns.len()];
        assert!(
            std::panic::catch_unwind(|| CsrMatrix::from_parts(2, 2, offsets, columns, values))
                .is_err()
        );
    }
}

fn block<T: Field, B: tensorcrate::tensors::OperatorBackend<T>>() -> BlockOperator<T, B> {
    let mut a = BlockOperator::new(&[2, 1], &[1, 2]).unwrap();
    a.set_block(
        0,
        0,
        Matrix::from_flat(2, 1, vec![T::from_real(2.0), T::from_real(3.0)]).to_backend::<B>(),
    )
    .unwrap();
    a.set_block(
        0,
        1,
        CsrMatrix::from_triplets(2, 2, [(0, 1, T::from_real(4.0)), (1, 0, T::from_real(5.0))])
            .to_backend::<B>(),
    )
    .unwrap();
    a.set_block(
        1,
        1,
        Matrix::from_flat(1, 2, vec![T::from_real(6.0), T::from_real(7.0)]).to_backend::<B>(),
    )
    .unwrap();
    a
}

#[test]
fn mixed_blocks_and_nested_partitions_match_dense() {
    let a = block::<f64, Host>();
    let dense = Matrix::from_rows([[2.0, 0.0, 4.0], [3.0, 5.0, 0.0], [0.0, 6.0, 7.0]]);
    let mut outer = BlockOperator::new(&[1, 3], &[3, 1]).unwrap();
    outer.set_block(1, 0, a).unwrap();
    outer
        .set_block(0, 1, DiagonalOperator::new(Vector::new([9.0])))
        .unwrap();
    let x = Vector::new([1.0, 2.0, 3.0, 4.0]);
    let mut y = Vector::filled(4, f64::NAN);
    outer.apply(&x, &mut y);
    let expected = dense.matvec(&Vector::new([1.0, 2.0, 3.0]));
    close(
        y.as_slice(),
        &[36.0, expected[0], expected[1], expected[2]],
        1e-12,
    );
    outer.apply_add(2.0, &x, -1.0, &mut y);
    close(
        y.as_slice(),
        &[36.0, expected[0], expected[1], expected[2]],
        1e-12,
    );
    let mut z = Vector::zeros(4);
    outer.apply_adjoint(&x, &mut z);
    let transposed = dense.transpose().matvec(&Vector::new([2.0, 3.0, 4.0]));
    close(
        z.as_slice(),
        &[transposed[0], transposed[1], transposed[2], 9.0],
        1e-12,
    );
    assert!(outer.set_block(1, 1, Matrix::<f64>::zeros(1, 1)).is_err());
    assert!(
        outer
            .set_block(usize::MAX, 0, Matrix::<f64>::zeros(0, 0))
            .is_err()
    );
    assert!(BlockOperator::<f64>::new(&[usize::MAX, 1], &[]).is_err());
}

fn elements<T: Field>() -> ElementOperator<T> {
    ElementOperator::new(
        4,
        2,
        vec![0, 1, 1, 2],
        [2.0, 1.0, -3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            .into_iter()
            .map(T::from_real)
            .collect(),
    )
    .unwrap()
}

#[test]
fn element_operator_sums_overlapping_dofs_and_preserves_isolated_dofs() {
    let a = elements::<f64>();
    let dense = Matrix::from_rows([
        [2.0, 1.0, 0.0, 0.0],
        [-3.0, 9.0, 6.0, 0.0],
        [0.0, 7.0, 8.0, 0.0],
        [0.0; 4],
    ]);
    let x = Vector::new([1.0, 2.0, 3.0, 4.0]);
    let mut y = Vector::filled(4, f64::NAN);
    a.apply(&x, &mut y);
    close(y.as_slice(), dense.matvec(&x).as_slice(), 1e-12);
    a.apply_adjoint(&x, &mut y);
    close(y.as_slice(), dense.transpose().matvec(&x).as_slice(), 1e-12);
    assert!(ElementOperator::<f64>::new(2, 0, vec![], vec![]).is_err());
    assert!(ElementOperator::<f64>::new(2, 2, vec![0, 2], vec![1.0; 4]).is_err());
    assert!(ElementOperator::<f64>::new(2, 2, vec![0], vec![1.0; 2]).is_err());
}

#[test]
fn complex_adjoint_identity_and_dotc_distinguish_bilinear_dot() {
    type C = Complex<f64>;
    let x = Vector::new([C::new(1.0, 2.0), C::new(3.0, -4.0)]);
    let y = Vector::new([C::new(-2.0, 1.0), C::new(1.0, 3.0)]);
    assert_eq!(x.dotc(&x), C::new(30.0, 0.0));
    assert_ne!(x.dotc(&y), x.dot(&y));
    let a = Matrix::from_rows([
        [C::new(1.0, 2.0), C::new(-3.0, 1.0)],
        [C::new(4.0, -2.0), C::new(5.0, 3.0)],
    ]);
    close(
        &[a.matvec(&x).dotc(&y)],
        &[x.dotc(&a.adjoint().matvec(&y))],
        1e-12,
    );
    let tensor = Tensor::from_vec(&[1, 2, 2], a.as_slice().to_vec());
    assert_eq!(tensor.adjoint().as_slice(), a.adjoint().as_slice());
    assert_eq!(tensor.dotc(&tensor), a.dotc(&a));
    assert_eq!(tensor.conj().as_slice(), a.conj().as_slice());
}

#[test]
fn hermitian_macro_operations_match_fused_and_unfused_expansion() {
    type C = Complex<f64>;
    let fused: C = math! { let a = [1 + 2i, 3 - 4i]; dotc(a, a) };
    let unfused: C = math! { fuse = false; let a = [1 + 2i, 3 - 4i]; dotc(a, a) };
    assert_eq!(fused, C::new(30.0, 0.0));
    assert_eq!(fused, unfused);
    let a: Matrix<C> = math! { adjoint([[1 + 2i, 3 + 4i]]) };
    assert_eq!(a.shape(), (2, 1));
    assert_eq!(a[(0, 0)], C::new(1.0, -2.0));
    let real: Matrix<f64> = math! { adjoint([[1, 2]]) };
    assert_eq!(real.to_rows(), [[1.0], [2.0]]);
    let scalar = C::new(2.0, 3.0);
    assert_eq!(scalar.dotc(&scalar), C::new(13.0, 0.0));
    assert_eq!(scalar.adjoint(), C::new(2.0, -3.0));
    let dual: tensorcrate::numbers::Dual<f64> = math! { let a = [1 + 1d, 2]; dotc(a, a) };
    assert_eq!(dual.real, 5.0);
    assert_eq!(dual.dual, 2.0);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    use super::*;
    use tensorcrate::metal::OperatorElement;
    use tensorcrate::tensors::Metal;

    fn check_operator<
        T: Field + OperatorElement,
        H: LinearOperator<T>,
        G: LinearOperator<T, Metal>,
    >(
        host: H,
        gpu: G,
    ) {
        let (rows, cols) = host.shape();
        let x = Vector::new(
            (0..cols)
                .map(|i| T::from_real((i + 1) as f64))
                .collect::<Vec<_>>(),
        );
        let device_x = x.to_backend::<Metal>();
        let mut want = Vector::filled(rows, T::from_real(f64::NAN));
        let mut got = want.to_backend::<Metal>();
        host.apply(&x, &mut want);
        gpu.apply(&device_x, &mut got);
        close(got.as_slice(), want.as_slice(), 2e-5);
        host.apply_add(T::from_real(2.0), &x, T::from_real(-1.0), &mut want);
        gpu.apply_add(T::from_real(2.0), &device_x, T::from_real(-1.0), &mut got);
        close(got.as_slice(), want.as_slice(), 2e-5);
        let input = Vector::new(
            (0..rows)
                .map(|i| T::from_real((i + 1) as f64))
                .collect::<Vec<_>>(),
        );
        let mut expected = Vector::zeros(cols);
        let mut result = expected.to_backend::<Metal>();
        host.apply_adjoint(&input, &mut expected);
        gpu.apply_adjoint(&input.to_backend::<Metal>(), &mut result);
        close(result.as_slice(), expected.as_slice(), 2e-5);
    }

    #[test]
    fn resident_real_and_complex_operators_match_host() {
        check_operator(block::<f32, Host>(), block::<f32, Metal>());
        check_operator(
            block::<Complex<f32>, Host>(),
            block::<Complex<f32>, Metal>(),
        );
        let a = elements::<f32>();
        let gpu = a.to_backend::<Metal>();
        check_operator(a, gpu);
        let a = ElementOperator::new(
            4,
            2,
            vec![0, 1, 1, 2],
            (0..8)
                .map(|i| Complex::new(i as f32, (i + 1) as f32))
                .collect(),
        )
        .unwrap();
        let gpu = a.to_backend::<Metal>();
        check_operator(a, gpu);
        let a = CsrMatrix::from_triplets(
            4,
            3,
            [
                (0, 2, Complex::new(1.0f32, 2.0)),
                (2, 0, Complex::new(-3.0, 4.0)),
            ],
        );
        let gpu = a.to_backend::<Metal>();
        check_operator(a, gpu);
    }

    #[test]
    fn gpu_hermitian_products_and_dense_adjoint_match_host() {
        let a = Vector::new(
            (0..1031)
                .map(|i| Complex::new((i % 13) as f32, (i % 7) as f32))
                .collect::<Vec<_>>(),
        );
        let gpu = a.to_backend::<Metal>();
        close(&[gpu.dotc(&gpu)], &[a.dotc(&a)], 1e-5);
        let a = Matrix::from_flat(2, 3, a.as_slice()[..6].to_vec());
        close(
            a.to_backend::<Metal>().adjoint().as_slice(),
            a.adjoint().as_slice(),
            1e-6,
        );
        let result: f32 = math! { backend = Metal; dtype = f32; dotc([1, 2, 3], [1, 2, 3]) };
        assert_eq!(result, 14.0);
        let result: Matrix<f32, Metal> =
            math! { backend = Metal; dtype = f32; adjoint([[1, 2, 3]]) };
        assert_eq!(result.as_slice(), [1.0, 2.0, 3.0]);
        let tensor = Tensor::from_vec(
            &[2, 2, 3],
            (0..12)
                .map(|i| Complex::new(i as f32, (i + 1) as f32))
                .collect::<Vec<_>>(),
        );
        let gpu = tensor.to_backend::<Metal>();
        close(gpu.adjoint().as_slice(), tensor.adjoint().as_slice(), 1e-6);
        close(&[gpu.dotc(&gpu)], &[tensor.dotc(&tensor)], 1e-5);
        let half: tensorcrate::numbers::f16 =
            math! { backend = Metal; dtype = f16; dotc([1, 2], [1, 2]) };
        assert_eq!(half.to_f32(), 5.0);
        let half: Matrix<tensorcrate::numbers::bf16, Metal> =
            math! { backend = Metal; dtype = bf16; adjoint([[1, 2]]) };
        assert_eq!(half.shape(), (2, 1));
    }
}
