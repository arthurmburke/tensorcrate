use super::*;
use crate::{
    numbers::{Coefficient, Complex},
    tensors::{Backend, Host, Matrix, OperatorBackend},
};

#[test]
fn triangular_levels_stay_queued_until_the_final_read() {
    use crate::tensors::{CsrMatrix, LinearOperator, Triangle, TriangularOperator, Vector};
    let a = TriangularOperator::new(
        CsrMatrix::from_triplets(
            5,
            5,
            [
                (0, 0, 2.0f32),
                (1, 1, 2.0),
                (2, 0, -1.0),
                (2, 2, 2.0),
                (3, 1, -1.0),
                (3, 3, 2.0),
                (4, 2, -1.0),
                (4, 4, 2.0),
            ],
        ),
        Triangle::Lower,
        false,
    )
    .unwrap();
    assert_eq!(a.level_count(), 3);
    let a = a.to_backend::<Metal>();
    let x = Vector::new([1.0f32; 5]).to_backend::<Metal>();
    let mut y = Vector::<f32>::zeros(5).to_backend::<Metal>();
    crate::metal::synchronize();
    let before =
        with_gpu(|gpu| Some((gpu.waits.get(), gpu.committed.get(), gpu.operations.get()))).unwrap();
    a.apply(&x, &mut y);
    let after =
        with_gpu(|gpu| Some((gpu.waits.get(), gpu.committed.get(), gpu.operations.get()))).unwrap();
    assert_eq!(after.0, before.0);
    assert_eq!(after.1, before.1);
    assert_eq!(after.2 - before.2, 4);
    assert_eq!(y.as_slice(), [0.5, 0.5, 0.75, 0.75, 0.875]);
}
#[test]
fn gmres_uses_fewer_syncs_than_modified_gram_schmidt() {
    use crate::solvers::{
        GmresOptions, Orthogonalization, StopReason, gmres_with_orthogonalization,
    };
    use crate::tensors::{CsrMatrix, Vector};
    let a = CsrMatrix::from_triplets(24, 24, (0..24).map(|i| (i, i, 2.0f32 + i as f32 * 0.1)))
        .to_backend::<Metal>();
    let b = Vector::new((0..24).map(|i| 1.0f32 + (i % 3) as f32).collect::<Vec<_>>())
        .to_backend::<Metal>();
    let options = GmresOptions {
        relative_tolerance: 1e-5,
        restart: 24,
        ..Default::default()
    };
    let solve = |method| {
        let mut x = Vector::<f32>::zeros(24).to_backend::<Metal>();
        crate::metal::synchronize();
        let before = with_gpu(|gpu| Some(gpu.waits.get())).unwrap();
        let report = gmres_with_orthogonalization(&a, &b, &mut x, options, method).unwrap();
        let syncs = with_gpu(|gpu| Some(gpu.waits.get())).unwrap() - before;
        assert_eq!(report.reason, StopReason::Converged);
        (syncs, x.to_vec())
    };
    let (reference, want) = solve(Orthogonalization::ModifiedGramSchmidt);
    let (batched, got) = solve(Orthogonalization::Backend);
    assert!(
        batched * 2 < reference,
        "{batched} batched syncs vs {reference} reference syncs"
    );
    for (a, b) in want.into_iter().zip(got) {
        assert!((a - b).abs() < 1e-4);
    }
}

#[test]
fn batched_complex_orthogonalization_reads_once_and_preserves_orthogonality() {
    type C = Complex<f32>;
    const N: usize = 513;
    const K: usize = 12;
    let mut q = vec![C::new(0.0, 0.0); N * K];
    for k in 0..K {
        q[k * N + k] = C::new(0.0, 1.0);
    }
    let q = Matrix::from_flat(K, N, q);
    let mut work = vec![C::new(0.0, 0.0); N];
    for (i, w) in work[..K].iter_mut().enumerate() {
        *w = C::new(i as f32 + 1.0, 2.0);
    }
    work[N - 1] = C::new(3.0, 4.0);
    let mut reference = work.clone();
    let expected = Host::orthogonalize(q.storage(), N, K, &mut reference);
    let q = q.to_backend::<Metal>();
    let mut work = Metal::from_vec(work);
    crate::metal::synchronize();
    let before = with_gpu(|gpu| Some((gpu.waits.get(), gpu.committed.get()))).unwrap();
    let (actual, norm_before, norm_after) = orthogonalize(q.storage(), N, K, &mut work);
    let after = with_gpu(|gpu| Some((gpu.waits.get(), gpu.committed.get()))).unwrap();
    assert_eq!(after.0 - before.0, 1);
    assert_eq!(after.1 - before.1, 1);
    for (a, b) in actual.into_iter().zip(expected.0) {
        assert!((a - b).magnitude() < 1e-5);
    }
    assert!((norm_before - expected.1).abs() < 1e-4);
    assert!((norm_after - 5.0).abs() < 1e-5);
    for value in &work.as_slice()[..K] {
        assert!(value.magnitude() < 1e-6);
    }
}
