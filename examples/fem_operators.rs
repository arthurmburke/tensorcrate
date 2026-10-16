//! A small complex block system with sparse field and dense coupling blocks.
//! This demonstrates algebra and execution, not a Maxwell discretization.

use tensorcrate::errors::Error;
use tensorcrate::numbers::Complex;
use tensorcrate::solvers::{GmresOptions, StopReason, gmres};
use tensorcrate::tensors::{
    BlockOperator, CsrMatrix, Host, LinearOperator, Matrix, OperatorBackend, Vector,
};

type C = Complex<f32>;

fn system<B: OperatorBackend<C>>() -> Result<BlockOperator<C, B>, Error> {
    let mut a = BlockOperator::new(&[3, 1], &[3, 1])?;
    let field = CsrMatrix::from_triplets(
        3,
        3,
        [
            (0, 0, C::new(4.0, 0.2)),
            (0, 1, C::new(-1.0, 0.0)),
            (1, 0, C::new(-1.0, 0.0)),
            (1, 1, C::new(4.0, 0.2)),
            (1, 2, C::new(-1.0, 0.0)),
            (2, 1, C::new(-1.0, 0.0)),
            (2, 2, C::new(4.0, 0.2)),
        ],
    );
    a.set_block(0, 0, field.to_backend::<B>())?;
    a.set_block(
        0,
        1,
        Matrix::from_rows([[C::new(1.0, 0.0)], [C::new(0.0, 0.0)], [C::new(1.0, 0.0)]])
            .to_backend::<B>(),
    )?;
    a.set_block(
        1,
        0,
        Matrix::from_rows([[C::new(1.0, 0.0), C::new(0.0, 0.0), C::new(1.0, 0.0)]])
            .to_backend::<B>(),
    )?;
    a.set_block(
        1,
        1,
        Matrix::from_rows([[C::new(3.0, 0.5)]]).to_backend::<B>(),
    )?;
    Ok(a)
}

fn main() -> Result<(), Error> {
    let a = system::<Host>()?;
    let expected = Vector::new([
        C::new(1.0, 0.5),
        C::new(-2.0, 1.0),
        C::new(3.0, -0.5),
        C::new(0.5, 1.0),
    ]);
    let mut b = Vector::zeros(4);
    a.apply(&expected, &mut b);
    let options = GmresOptions {
        relative_tolerance: 1e-5,
        ..Default::default()
    };
    let mut x = Vector::zeros(4);
    let report = gmres(&a, &b, &mut x, options)?;
    assert_eq!(report.reason, StopReason::Converged);
    println!(
        "Host: {} iterations, residual {:.3e}",
        report.iterations, report.residual_norm
    );

    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        use tensorcrate::tensors::Metal;
        let a = system::<Metal>()?;
        let mut x = Vector::<C>::filled(4, C::new(0.0, 0.0)).to_backend::<Metal>();
        let report = gmres(&a, &b.to_backend::<Metal>(), &mut x, options)?;
        assert_eq!(report.reason, StopReason::Converged);
        assert!(x.is_device_resident());
        println!(
            "Metal: {} iterations, residual {:.3e}",
            report.iterations, report.residual_norm
        );
    }
    Ok(())
}
