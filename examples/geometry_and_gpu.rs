use tensorcrate::errors::Error;
use tensorcrate::geometry::{AffineTetrahedron, Mat3, QuadratureRule, Vec3};

fn main() -> Result<(), Error> {
    let cell = AffineTetrahedron::from_vertices([
        Vec3::new([0.0, 0.0, 0.0]),
        Vec3::new([2.0, 0.0, 0.0]),
        Vec3::new([0.0, 3.0, 0.0]),
        Vec3::new([0.0, 0.0, 4.0]),
    ])?;
    let rule = QuadratureRule::<f64, 3>::tetrahedron(2)?;
    let integral = rule.integrate_mapped(&cell, |x| x[0] + x[1] + x[2]);
    assert!((integral - 9.0).abs() < 1e-12);
    println!("Cell volume: {}, integral: {integral}", cell.measure());
    let a = Mat3::new([[0.0, 2.0, 1.0], [1.0, -2.0, 0.0], [3.0, 1.0, 2.0]]);
    let x = a.solve(&Vec3::new([1.0, 2.0, 3.0]))?;
    println!("Pivoted small solve: {:?}", x.0);
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        use tensorcrate::{
            solvers::{GmresOptions, Ilu0, StopReason, gmres_preconditioned_operator},
            tensors::{CsrMatrix, Metal, Vector},
        };
        let a = CsrMatrix::from_triplets(
            3,
            3,
            [
                (0, 0, 4.0f32),
                (0, 1, -1.0),
                (1, 0, -1.0),
                (1, 1, 4.0),
                (1, 2, -1.0),
                (2, 1, -1.0),
                (2, 2, 4.0),
            ],
        );
        let inverse = Ilu0::new(&a)?.operator::<Metal>()?;
        let mut x = Vector::<f32, Metal>::filled(3, 0.0);
        let b = Vector::new([1.0f32, 2.0, 3.0]).to_backend::<Metal>();
        let report = gmres_preconditioned_operator(
            &a.to_backend::<Metal>(),
            &b,
            &mut x,
            &inverse,
            GmresOptions {
                relative_tolerance: 1e-5,
                ..Default::default()
            },
        )?;
        assert_eq!(report.reason, StopReason::Converged);
        assert!(x.is_device_resident());
        println!(
            "Metal ILU-GMRES: {} iterations, residual {:.3e}",
            report.iterations, report.residual_norm
        );
    }
    Ok(())
}
