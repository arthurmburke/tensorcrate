//! Forward-mode automatic differentiation over tensors.
//!
//! Three independent checks run against every rule:
//!
//! 1. the existing scalar `Dual<f32>` path over `Vector<Dual<f32>, N>` — the
//!    oracle, since it is derived from the algebra in `numbers.rs`;
//! 2. central finite differences, which catch a rule that is self-consistently
//!    wrong in both implementations;
//! 3. the `Metal` backend against the `Host` backend, which must agree because
//!    they run the same code over different memory.

use rinterp::numbers::Dual;
use rinterp::tensors::dual::{
    DualMatrix, DualVector, gradient, gradient_wrt_matrix, jacobian, jacobian_wrt_matrix,
};
use rinterp::tensors::{Analytic, BinaryOp, Host, Kernels, Matrix, Vector};

#[cfg(all(feature = "metal", target_os = "macos"))]
use rinterp::tensors::Metal;

/// Loose enough for `f32` GPU transcendentals (the shaders compile with Metal's
/// default fast-math), tight enough to catch a wrong derivative.
const TOLERANCE: f32 = 2e-3;

fn close(actual: f32, expected: f32) -> bool {
    (actual - expected).abs() <= TOLERANCE * (1.0 + expected.abs())
}

fn assert_slice_close(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(close(a, e), "{what}: at {index}, {a} vs {e}");
    }
}

fn value(index: usize) -> f32 {
    // Inside (0, 1): away from zero, because the tests divide by these values and
    // by their squares, and below one, because `arcsin`/`arccos` are only defined
    // there. Also clear of the poles of `tan` and `csc`.
    0.15 + (index % 5) as f32 * 0.17
}

fn vector<const N: usize>() -> Vector<f32, N> {
    Vector::new(std::array::from_fn(value))
}

fn direction<const N: usize>() -> Vector<f32, N> {
    Vector::new(std::array::from_fn(|i| 0.5 - (i % 3) as f32 * 0.4))
}

fn matrix<const R: usize, const C: usize>() -> Matrix<f32, R, C> {
    Matrix::from_rows(std::array::from_fn(|row| {
        std::array::from_fn(|col| value(row * C + col + row))
    }))
}

fn matrix_direction<const R: usize, const C: usize>() -> Matrix<f32, R, C> {
    Matrix::from_rows(std::array::from_fn(|row| {
        std::array::from_fn(|col| 0.3 - ((row + col) % 4) as f32 * 0.2)
    }))
}

// ---- against the scalar `Dual` oracle ---------------------------------------

// The by-reference operators are the ones that matter on a non-`Copy` backend, so
// the tests use them even where the host tensors would copy happily.
#[allow(clippy::op_ref)]
#[test]
fn elementwise_rules_match_the_scalar_dual_path() {
    let (a, b) = (vector::<12>(), direction::<12>().shift_up());
    let (da, db) = (direction::<12>(), vector::<12>());

    let dual_a = DualVector::<12>::new(a, da);
    let dual_b = DualVector::<12>::new(b, db);

    // The oracle: the same inputs as a `Vector<Dual<f32>, N>`, where every
    // operation is the scalar dual arithmetic from `numbers.rs`.
    let oracle_a = dual_a.to_dual_vector();
    let oracle_b = dual_b.to_dual_vector();

    for (op, label) in [
        (BinaryOp::Add, "add"),
        (BinaryOp::Sub, "sub"),
        (BinaryOp::Mul, "mul"),
        (BinaryOp::Div, "div"),
    ] {
        let actual = match op {
            BinaryOp::Add => &dual_a + &dual_b,
            BinaryOp::Sub => &dual_a - &dual_b,
            BinaryOp::Mul => &dual_a * &dual_b,
            BinaryOp::Div => &dual_a / &dual_b,
            BinaryOp::Rem => unreachable!(),
        };
        let expected = match op {
            BinaryOp::Add => oracle_a + oracle_b,
            BinaryOp::Sub => oracle_a - oracle_b,
            BinaryOp::Mul => oracle_a * oracle_b,
            BinaryOp::Div => oracle_a / oracle_b,
            BinaryOp::Rem => unreachable!(),
        };
        for (index, dual) in expected.data().iter().enumerate() {
            assert!(
                close(actual.value().as_slice()[index], dual.real),
                "{label} value at {index}"
            );
            assert!(
                close(actual.tangent().as_slice()[index], dual.dual),
                "{label} tangent at {index}: {} vs {}",
                actual.tangent().as_slice()[index],
                dual.dual
            );
        }
    }
}

#[test]
fn every_analytic_function_matches_the_scalar_dual_path() {
    let a = vector::<8>();
    let da = direction::<8>();
    let dual = DualVector::<8>::new(a, da);
    let oracle = dual.to_dual_vector();

    use rinterp::numbers::{
        Arccos, Arcsin, Arctan, Cos, Cosh, Csc, Exp, Ln, Sec, Sin, Sinh, Tan, Tanh,
    };

    // `arcsin`/`arccos` need |x| < 1, which `value()` satisfies.
    let cases: [(&str, DualVector<8>, Vector<Dual<f32>, 8>); 13] = [
        ("sin", dual.sin(), oracle.map(|&x| Sin::sin(x))),
        ("cos", dual.cos(), oracle.map(|&x| Cos::cos(x))),
        ("tan", dual.tan(), oracle.map(|&x| Tan::tan(x))),
        ("sec", dual.sec(), oracle.map(|&x| Sec::sec(x))),
        ("csc", dual.csc(), oracle.map(|&x| Csc::csc(x))),
        ("arcsin", dual.arcsin(), oracle.map(|&x| Arcsin::arcsin(x))),
        ("arccos", dual.arccos(), oracle.map(|&x| Arccos::arccos(x))),
        ("arctan", dual.arctan(), oracle.map(|&x| Arctan::arctan(x))),
        ("exp", dual.exp(), oracle.map(|&x| Exp::exp(x))),
        ("ln", dual.ln(), oracle.map(|&x| Ln::ln(x))),
        ("sinh", dual.sinh(), oracle.map(|&x| Sinh::sinh(x))),
        ("cosh", dual.cosh(), oracle.map(|&x| Cosh::cosh(x))),
        ("tanh", dual.tanh(), oracle.map(|&x| Tanh::tanh(x))),
    ];

    for (label, actual, expected) in cases {
        for (index, dual) in expected.data().iter().enumerate() {
            assert!(
                close(actual.value().as_slice()[index], dual.real),
                "{label} value at {index}"
            );
            assert!(
                close(actual.tangent().as_slice()[index], dual.dual),
                "{label} derivative at {index}: {} vs {}",
                actual.tangent().as_slice()[index],
                dual.dual
            );
        }
    }
}

#[test]
fn products_match_the_dual_coefficient_path() {
    // `Matrix<Dual<f32>, R, C>` already supports matmul through `Coefficient`,
    // so the whole product rule can be checked against it at once.
    let (a, da) = (matrix::<4, 5>(), matrix_direction::<4, 5>());
    let (b, db) = (matrix::<5, 3>(), matrix_direction::<5, 3>());

    let dual_a = DualMatrix::<4, 5>::new(a, da);
    let dual_b = DualMatrix::<5, 3>::new(b, db);
    let product = dual_a.matmul(&dual_b);

    let oracle = dual_a.to_dual_matrix().matmul(&dual_b.to_dual_matrix());
    for (row, expected_row) in oracle.data().iter().enumerate() {
        for (col, expected) in expected_row.iter().enumerate() {
            assert!(
                close(product.value().to_rows()[row][col], expected.real),
                "value at {row},{col}"
            );
            assert!(
                close(product.tangent().to_rows()[row][col], expected.dual),
                "tangent at {row},{col}: {} vs {}",
                product.tangent().to_rows()[row][col],
                expected.dual
            );
        }
    }

    // matvec and dot follow the same rule.
    let (v, dv) = (vector::<5>(), direction::<5>());
    let dual_v = DualVector::<5>::new(v, dv);
    let mapped = dual_a.matvec(&dual_v);
    let oracle_mapped = dual_a.to_dual_matrix().matvec(&dual_v.to_dual_vector());
    for (index, expected) in oracle_mapped.data().iter().enumerate() {
        assert!(close(mapped.value().as_slice()[index], expected.real));
        assert!(close(mapped.tangent().as_slice()[index], expected.dual));
    }

    let self_dot = dual_v.dot(&dual_v);
    let oracle_dot = dual_v.to_dual_vector().dot(&dual_v.to_dual_vector());
    assert!(close(self_dot.real, oracle_dot.real));
    assert!(close(self_dot.dual, oracle_dot.dual));

    // vecmat, and transposition of both parts.
    let row_vector = DualVector::<4>::new(vector::<4>(), direction::<4>());
    let through_matrix = row_vector.vecmat(&dual_a);
    let oracle_row = row_vector.to_dual_vector().vecmat(&dual_a.to_dual_matrix());
    for (index, expected) in oracle_row.data().iter().enumerate() {
        assert!(close(
            through_matrix.value().as_slice()[index],
            expected.real
        ));
        assert!(close(
            through_matrix.tangent().as_slice()[index],
            expected.dual
        ));
    }
    assert_eq!(
        dual_a.transpose().tangent().to_rows(),
        da.transpose().to_rows()
    );
}

// ---- against finite differences ---------------------------------------------

/// A test function exercising products, an analytic function and a reduction:
/// `f(x) = Σ tanh(Ax) ⊙ (x ⊙ x)`.
fn probe<const N: usize, B: Kernels>(a: &Matrix<f32, N, N, B>, x: &DualVector<N, B>) -> Dual<f32> {
    let mapped = DualMatrix::constant(a.to_backend::<B>()).matvec(x).tanh();
    (&mapped * &(x * x)).sum()
}

#[test]
fn gradients_match_central_finite_differences() {
    let a = matrix::<4, 4>();
    let at = vector::<4>();

    let analytic = gradient(&at, |x| probe(&a, x));

    let step = 1e-3f32;
    for input in 0..4 {
        let mut forward = *at.data();
        let mut backward = *at.data();
        forward[input] += step;
        backward[input] -= step;
        let numeric = (probe(&a, &DualVector::constant(Vector::new(forward))).real
            - probe(&a, &DualVector::constant(Vector::new(backward))).real)
            / (2.0 * step);
        assert!(
            (analytic.as_slice()[input] - numeric).abs() < 5e-3,
            "gradient component {input}: forward mode {} vs finite difference {numeric}",
            analytic.as_slice()[input]
        );
    }
}

#[test]
fn jacobian_columns_are_the_seeded_tangents() {
    // f(x) = A·(x ⊙ x), so J = 2·A·diag(x).
    let a = matrix::<3, 4>();
    let at = vector::<4>();
    let computed = jacobian::<4, 3, Host>(&at, |x| DualMatrix::constant(a).matvec(&(x * x)));

    for row in 0..3 {
        for col in 0..4 {
            let expected = 2.0 * a.data()[row][col] * at.data()[col];
            assert!(
                close(computed.data()[row][col], expected),
                "J[{row},{col}]: {} vs {expected}",
                computed.data()[row][col]
            );
        }
    }
}

#[test]
fn fused_multiply_adds_propagate_every_tangent() {
    let a = matrix::<3, 4>();
    let da = matrix_direction::<3, 4>();
    let x = vector::<4>();
    let dx = direction::<4>();
    let bias = vector::<3>();
    let dbias = direction::<3>();

    let fused =
        DualMatrix::new(a, da).matvec_add(&DualVector::new(x, dx), &DualVector::new(bias, dbias));
    assert_slice_close(
        fused.value().as_slice(),
        a.matvec_add(&x, bias).as_slice(),
        "fused matvec value",
    );
    assert_slice_close(
        fused.tangent().as_slice(),
        (da.matvec(&x) + a.matvec(&dx) + dbias).as_slice(),
        "fused matvec tangent",
    );

    let b = matrix::<4, 2>();
    let db = matrix_direction::<4, 2>();
    let addend = matrix::<3, 2>();
    let daddend = matrix_direction::<3, 2>();
    let fused = DualMatrix::new(a, da)
        .matmul_add(&DualMatrix::new(b, db), &DualMatrix::new(addend, daddend));
    assert_slice_close(
        fused.value().as_slice(),
        a.matmul_add(&b, addend).as_slice(),
        "fused matmul value",
    );
    assert_slice_close(
        fused.tangent().as_slice(),
        (da.matmul(&b) + a.matmul(&db) + daddend).as_slice(),
        "fused matmul tangent",
    );
}

#[test]
fn constants_have_no_tangent_and_seeds_are_one_hot() {
    let constant = DualVector::<5>::constant(vector::<5>());
    assert_eq!(constant.tangent().to_array(), [0.0; 5]);
    assert_eq!(constant.sin().tangent().to_array(), [0.0; 5]);

    let seeded = DualVector::<5>::seed(vector::<5>(), 2);
    assert_eq!(seeded.tangent().to_array(), [0.0, 0.0, 1.0, 0.0, 0.0]);
    // An out-of-range seed differentiates with respect to nothing.
    assert_eq!(
        DualVector::<5>::seed(vector::<5>(), 9).tangent().to_array(),
        [0.0; 5]
    );

    let matrix_seed = DualMatrix::<2, 3>::seed(matrix::<2, 3>(), 1, 2);
    assert_eq!(
        matrix_seed.tangent().to_rows(),
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    );
}

#[test]
fn scalar_helpers_follow_the_product_and_quotient_rules() {
    let x = DualVector::<4>::new(vector::<4>(), direction::<4>());

    // A constant shift leaves the tangent alone; a constant scale multiplies it.
    assert_eq!(x.shift(2.5).tangent().to_array(), x.tangent().to_array());
    assert_slice_close(
        x.scale(3.0).tangent().as_slice(),
        &x.tangent().scale(3.0).to_array(),
        "scale",
    );

    // A *dual* scalar brings its own tangent: d(a·s) = ȧs + aṡ.
    let s = Dual::new(2.0f32, 5.0);
    let scaled = x.broadcast(s, BinaryOp::Mul, false);
    for i in 0..4 {
        let expected = x.tangent().as_slice()[i] * s.real + x.value().as_slice()[i] * s.dual;
        assert!(close(scaled.tangent().as_slice()[i], expected));
    }

    // d(s/a) = ṡ/a − sȧ/a², checked against the scalar path.
    let divided = x.broadcast(s, BinaryOp::Div, true);
    let oracle = x
        .to_dual_vector()
        .map(|&element| Dual::new(s.real, s.dual) / element);
    for (i, expected) in oracle.data().iter().enumerate() {
        assert!(close(divided.value().as_slice()[i], expected.real));
        assert!(
            close(divided.tangent().as_slice()[i], expected.dual),
            "s/a tangent at {i}: {} vs {}",
            divided.tangent().as_slice()[i],
            expected.dual
        );
    }

    // Reciprocal: d(1/a) = −ȧ/a².
    let reciprocal = x.recip();
    for i in 0..4 {
        let (v, d) = (x.value().as_slice()[i], x.tangent().as_slice()[i]);
        assert!(close(reciprocal.value().as_slice()[i], 1.0 / v));
        assert!(close(reciprocal.tangent().as_slice()[i], -d / (v * v)));
    }
}

// ---- the two backends against each other ------------------------------------

#[cfg(all(feature = "metal", target_os = "macos"))]
#[allow(clippy::op_ref)]
#[test]
fn the_metal_backend_agrees_with_the_host_backend() {
    let (a, da) = (matrix::<8, 8>(), matrix_direction::<8, 8>());
    let (b, db) = (matrix::<8, 8>(), matrix_direction::<8, 8>());
    let (v, dv) = (vector::<8>(), direction::<8>());

    let host_a = DualMatrix::<8, 8, Host>::new(a, da);
    let host_b = DualMatrix::<8, 8, Host>::new(b, db);
    let host_v = DualVector::<8, Host>::new(v, dv);

    let gpu_a = DualMatrix::<8, 8, Metal>::new(a.to_backend(), da.to_backend());
    let gpu_b = DualMatrix::<8, 8, Metal>::new(b.to_backend(), db.to_backend());
    let gpu_v = DualVector::<8, Metal>::new(v.to_backend(), dv.to_backend());

    // A chain with a product, an elementwise product, an analytic function and a
    // scalar — every kind of rule at once.
    let host = (&host_a.matmul(&host_b).tanh() * &host_a).scale(0.5);
    let gpu = (&gpu_a.matmul(&gpu_b).tanh() * &gpu_a).scale(0.5);
    assert_slice_close(
        gpu.value().as_slice(),
        host.value().as_slice(),
        "chain value",
    );
    assert_slice_close(
        gpu.tangent().as_slice(),
        host.tangent().as_slice(),
        "chain tangent",
    );

    // The whole chain stays in shared memory.
    if gpu_a.value().is_device_resident() {
        assert!(gpu.value().is_device_resident());
        assert!(gpu.tangent().is_device_resident());
    }

    let host_mapped = host_a.matvec(&host_v).exp();
    let gpu_mapped = gpu_a.matvec(&gpu_v).exp();
    assert_slice_close(
        gpu_mapped.tangent().as_slice(),
        host_mapped.tangent().as_slice(),
        "matvec tangent",
    );

    let (host_dot, gpu_dot) = (host_v.dot(&host_v), gpu_v.dot(&gpu_v));
    assert!(close(gpu_dot.real, host_dot.real));
    assert!(close(gpu_dot.dual, host_dot.dual));

    // Gradients agree, which exercises seeding on both backends.
    let host_gradient = gradient(&v, |x| probe(&a, x));
    let gpu_gradient = gradient(&v.to_backend::<Metal>(), |x| probe(&a.to_backend(), x));
    assert_slice_close(
        gpu_gradient.as_slice(),
        host_gradient.as_slice(),
        "gradient",
    );

    // And the primal-only analytic path the `unary` kernel serves.
    assert_slice_close(
        v.to_backend::<Metal>().analytic(Analytic::Tanh).as_slice(),
        &v.map(|&x| x.tanh()).to_array(),
        "primal tanh",
    );
}

/// Every arm of the shader's `analytic_value`/`analytic_derivative` switch against
/// the CPU table — a mismatched discriminant or formula in one of the thirteen would
/// otherwise only show up for whichever function the other tests happen to use.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn every_analytic_variant_agrees_between_the_shader_and_the_cpu_table() {
    let (v, dv) = (vector::<16>(), direction::<16>());
    let host = DualVector::<16, Host>::new(v, dv);
    let gpu = DualVector::<16, Metal>::new(v.to_backend(), dv.to_backend());

    let gpu_v = v.to_backend::<Metal>();
    for f in Analytic::ALL {
        let (host_result, gpu_result) = (host.analytic(f), gpu.analytic(f));
        assert_slice_close(
            gpu_result.value().as_slice(),
            host_result.value().as_slice(),
            &format!("{f:?} value"),
        );
        assert_slice_close(
            gpu_result.tangent().as_slice(),
            host_result.tangent().as_slice(),
            &format!("{f:?} derivative"),
        );
        // The primal-only kernel shares the same switch, and every function is
        // defined on the inputs, so nothing here should be NaN.
        assert_slice_close(
            gpu_v.analytic(f).as_slice(),
            v.analytic_on_host(f).as_slice(),
            &format!("{f:?} primal"),
        );
    }
}

/// The host backend reaches analytic functions through `math!` or `map`; this is
/// just the shape the test above needs to compare against.
#[cfg(all(feature = "metal", target_os = "macos"))]
trait AnalyticOnHost {
    fn analytic_on_host(&self, f: Analytic) -> Self;
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl<const N: usize> AnalyticOnHost for Vector<f32, N, Host> {
    fn analytic_on_host(&self, f: Analytic) -> Self {
        Host::vector_unary(self, f)
    }
}

/// `Vector` has no "shift every element" helper of its own; the tests want one to
/// keep divisors away from zero.
trait ShiftUp {
    fn shift_up(self) -> Self;
}

impl<const N: usize> ShiftUp for Vector<f32, N> {
    fn shift_up(self) -> Self {
        self.broadcast_right(2.0, BinaryOp::Add)
    }
}

// ---- differentiating with respect to a matrix --------------------------------

#[test]
fn flattening_round_trips_and_reshapes_preserve_row_major_order() {
    let (m, dm) = (matrix::<2, 3>(), matrix_direction::<2, 3>());
    let dual = DualMatrix::<2, 3>::new(m, dm);

    let flat = dual.flattened();
    assert_eq!(flat.value().as_slice(), m.as_slice());
    assert_eq!(flat.tangent().as_slice(), dm.as_slice());

    let restored = DualMatrix::<2, 3>::from_flattened(flat);
    assert_eq!(restored.value().to_rows(), m.to_rows());
    assert_eq!(restored.tangent().to_rows(), dm.to_rows());

    // A vector viewed as a one-row or one-column matrix keeps its elements.
    let v = DualVector::<4>::new(vector::<4>(), direction::<4>());
    let row = v.into_row();
    assert_eq!(row.value().to_rows()[0], vector::<4>().to_array());
    let column = v.into_column();
    assert_eq!(column.value().to_rows()[2][0], vector::<4>().to_array()[2]);
    assert_eq!(column.value().shape(), (4, 1));
}

/// `f(A) = ‖A·x‖²`, whose gradient with respect to `A` is `2(Ax)xᵀ`.
fn projection_energy<const R: usize, const C: usize, B: Kernels>(
    x: &Vector<f32, C, B>,
    a: &DualMatrix<R, C, B>,
) -> Dual<f32> {
    let mapped = a.matvec(&DualVector::constant(duplicate(x)));
    mapped.dot(&mapped)
}

fn duplicate<const N: usize, B: Kernels>(v: &Vector<f32, N, B>) -> Vector<f32, N, B> {
    v.to_backend::<B>()
}

#[test]
fn gradient_wrt_matrix_matches_the_closed_form_and_finite_differences() {
    let a = matrix::<3, 4>();
    let x = vector::<4>();

    let computed = gradient_wrt_matrix(&a, |m| projection_energy(&x, m));

    // Closed form: 2(Ax)xᵀ.
    let projected = a.matvec(&x);
    for row in 0..3 {
        for col in 0..4 {
            let expected = 2.0 * projected.data()[row] * x.data()[col];
            assert!(
                close(computed.data()[row][col], expected),
                "∂f/∂A[{row},{col}]: {} vs {expected}",
                computed.data()[row][col]
            );
        }
    }

    // And central differences on every entry, which needs no closed form.
    let step = 1e-3f32;
    for row in 0..3 {
        for col in 0..4 {
            let mut forward = *a.data();
            let mut backward = *a.data();
            forward[row][col] += step;
            backward[row][col] -= step;
            let numeric =
                (projection_energy(&x, &DualMatrix::constant(Matrix::from_rows(forward))).real
                    - projection_energy(&x, &DualMatrix::constant(Matrix::from_rows(backward)))
                        .real)
                    / (2.0 * step);
            assert!(
                (computed.data()[row][col] - numeric).abs() < 5e-3,
                "∂f/∂A[{row},{col}]: forward mode {} vs finite difference {numeric}",
                computed.data()[row][col]
            );
        }
    }
}

#[test]
fn jacobian_wrt_matrix_has_one_column_per_input_element() {
    // f(A) = A·x, so ∂(Ax)ᵢ/∂Aⱼₖ = δᵢⱼ·xₖ.
    let a = matrix::<3, 4>();
    let x = vector::<4>();
    let computed = jacobian_wrt_matrix(&a, |m| m.matvec(&DualVector::constant(x)));

    for output in 0..3 {
        for row in 0..3 {
            for col in 0..4 {
                let expected = if output == row { x.data()[col] } else { 0.0 };
                assert!(
                    close(computed.data()[output][row * 4 + col], expected),
                    "J[{output}, ({row},{col})]: {} vs {expected}",
                    computed.data()[output][row * 4 + col]
                );
            }
        }
    }
}

#[test]
fn a_matrix_valued_function_flattens_into_a_square_jacobian() {
    // f(A) = A·B, so ∂(AB)ᵢⱼ/∂Aₖₗ = δᵢₖ·Bₗⱼ.
    let a = matrix::<2, 2>();
    let b = matrix_direction::<2, 2>();
    let computed = jacobian_wrt_matrix(&a, |m| m.matmul(&DualMatrix::constant(b)).into_flattened());

    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for l in 0..2 {
                    let expected = if i == k { b.data()[l][j] } else { 0.0 };
                    let actual = computed.data()[i * 2 + j][k * 2 + l];
                    assert!(
                        close(actual, expected),
                        "J[({i},{j}),({k},{l})]: {actual} vs {expected}"
                    );
                }
            }
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn matrix_derivatives_agree_between_the_backends() {
    let a = matrix::<4, 5>();
    let x = vector::<5>();

    let host = gradient_wrt_matrix(&a, |m| projection_energy(&x, m));
    let resident = gradient_wrt_matrix(&a.to_backend::<Metal>(), |m| {
        projection_energy(&x.to_backend::<Metal>(), m)
    });
    assert_slice_close(
        resident.as_slice(),
        host.as_slice(),
        "gradient with respect to a matrix",
    );
    if a.to_backend::<Metal>().is_device_resident() {
        assert!(resident.is_device_resident(), "the gradient stays resident");
    }

    let host = jacobian_wrt_matrix(&a, |m| m.matvec(&DualVector::constant(x)));
    let resident = jacobian_wrt_matrix(&a.to_backend::<Metal>(), |m| {
        m.matvec(&DualVector::constant(x.to_backend::<Metal>()))
    });
    assert_slice_close(
        resident.as_slice(),
        host.as_slice(),
        "Jacobian with respect to a matrix",
    );

    // Flattening is a move on this backend, so it must not disturb the values.
    let dual = DualMatrix::<4, 5, Metal>::new(a.to_backend(), a.to_backend());
    assert_slice_close(
        dual.flattened().value().as_slice(),
        a.as_slice(),
        "resident flattening",
    );
}
