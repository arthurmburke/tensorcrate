//! Graphs of fused kernels around matrix and dot products.
//!
//! As for single programs, the oracle is the unfused computation: on the host
//! a graph built under the exact algebra must agree with the same operations
//! done one kernel at a time bit for bit, and on Metal within tolerance.

use tensorcrate::tensors::fused::{
    self, Algebra, Builder, CostModel, DType, Decl, Fusable, Mode, ProgramError, RowStatistic,
};
use tensorcrate::tensors::{Analytic, Compare, Host, Kernels, Matrix, Vector};

#[cfg(all(feature = "metal", target_os = "macos"))]
use tensorcrate::tensors::Metal;

/// A small deterministic generator, so a failure reproduces.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    fn vector(&mut self, len: usize) -> Vec<f32> {
        (0..len)
            .map(|_| self.next() as f32 / (1u64 << 30) as f32 - 1.0)
            .collect()
    }

    fn matrix(&mut self, rows: usize, cols: usize) -> Matrix<f32> {
        Matrix::from_flat(rows, cols, self.vector(rows * cols))
    }
}

fn assert_bits_eq(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: lengths differ");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "{what}: element {i} is {a:?}, expected {e:?}"
        );
    }
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: lengths differ");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tolerance * (1.0 + e.abs()),
            "{what}: element {i} is {a}, expected {e}"
        );
    }
}

/// Every element of `m` combined with element `c` of `row`, by `f`.
fn with_row(m: &Matrix<f32>, row: &[f32], f: impl Fn(f32, f32) -> f32) -> Matrix<f32> {
    let cols = m.cols();
    let values: Vec<f32> = m
        .as_slice()
        .iter()
        .enumerate()
        .map(|(i, &x)| f(x, row[i % cols]))
        .collect();
    Matrix::from_flat(m.rows(), cols, values)
}

/// Exact, so that each kernel's arithmetic is the operations as written.
fn exact<T: tensorcrate::numbers::Real>(b: Builder<T>) -> fused::Graph<T> {
    b.build_graph_with(&CostModel::BALANCED, Algebra::Exact)
        .unwrap()
}

// ---- matrix products ------------------------------------------------------------

/// `relu(x·w₁ + b₁)·w₂ + b₂` over a `batch × inputs` x.
fn two_layers(batch: usize, inputs: usize, hidden: usize, outputs: usize) -> fused::Graph {
    let f = DType::F32;
    let mut b = Builder::<f32>::new();
    let x = b.input(Decl::matrix(f, (batch, inputs)));
    let w1 = b.input(Decl::matrix(f, (inputs, hidden)));
    let b1 = b.input(Decl::vector(f, hidden));
    let w2 = b.input(Decl::matrix(f, (hidden, outputs)));
    let b2 = b.input(Decl::vector(f, outputs));
    let first = b.matmul(x, w1);
    let shifted = b.add(first, b1);
    let zero = b.constant(0.0);
    let hidden_layer = b.compare(Compare::Max, shifted, zero);
    let second = b.matmul(hidden_layer, w2);
    let y = b.add(second, b2);
    b.output(y, f);
    exact(b)
}

type Layers = (Matrix<f32>, Matrix<f32>, Vec<f32>, Matrix<f32>, Vec<f32>);

fn layer_operands(
    rng: &mut Lcg,
    (batch, inputs, hidden, outputs): (usize, usize, usize, usize),
) -> Layers {
    (
        rng.matrix(batch, inputs),
        rng.matrix(inputs, hidden),
        rng.vector(hidden),
        rng.matrix(hidden, outputs),
        rng.vector(outputs),
    )
}

/// The two layers, one operation at a time.
fn two_layers_by_hand((x, w1, b1, w2, b2): &Layers) -> Vec<f32> {
    let hidden = with_row(&x.matmul(w1), b1, |p, b| (p + b).max(0.0));
    with_row(&hidden.matmul(w2), b2, |p, b| p + b)
        .as_slice()
        .to_vec()
}

const LAYERS: [(usize, usize, usize, usize); 4] = [
    (1, 1, 1, 1),
    (4, 3, 5, 2),
    (17, 33, 15, 9),
    (64, 128, 96, 10),
];

#[test]
fn each_layer_is_its_product_with_the_rest_as_its_epilogue() {
    let mut rng = Lcg(1);
    for shape in LAYERS {
        let graph = two_layers(shape.0, shape.1, shape.2, shape.3);
        // The hidden layer is written by the first product's epilogue, and
        // the output by the second's.
        assert_eq!(graph.kernels(), 2, "{graph}");
        assert_eq!(graph.space(), [shape.0, shape.3]);
        let operands = layer_operands(&mut rng, shape);
        let (x, w1, b1, w2, b2) = &operands;
        let (b1, b2) = (Vector::new(b1.clone()), Vector::new(b2.clone()));
        let inputs: [&dyn Fusable<Host>; 5] = [x, w1, &b1, w2, &b2];
        let want = two_layers_by_hand(&operands);
        for mode in [Mode::Fused, Mode::Unfused] {
            let got = fused::with_mode(mode, || graph.run(&inputs, &mut []))
                .remove(0)
                .into_matrix::<f32>();
            assert_eq!(got.shape(), (shape.0, shape.3));
            assert_bits_eq(got.as_slice(), &want, &format!("{shape:?}, {mode:?}"));
        }
    }
}

#[test]
fn computed_and_transposed_operands_are_computed_first() {
    // exp(x)ᵀ · w: the transpose and the exponential are one kernel, which
    // writes the operand; the product is stored as it is.
    let f = DType::F32;
    let mut b = Builder::<f32>::new();
    let x = b.input(Decl::matrix(f, (3, 4)));
    let w = b.input(Decl::matrix(f, (3, 2)));
    let e = b.unary(Analytic::Exp, x);
    let et = b.transpose(e, 0, 1);
    let y = b.matmul(et, w);
    b.output(y, f);
    let graph = exact(b);
    assert_eq!(graph.kernels(), 2, "{graph}");

    let mut rng = Lcg(2);
    let (x, w) = (rng.matrix(3, 4), rng.matrix(3, 2));
    let got = graph
        .run::<Host>(&[&x, &w], &mut [])
        .remove(0)
        .into_matrix::<f32>();
    let want = Host::matrix_unary(&x, Analytic::Exp).transpose().matmul(&w);
    assert_bits_eq(got.as_slice(), want.as_slice(), "exp(x)ᵀ·w");
}

#[test]
fn an_operand_of_another_type_or_layout_is_copied_into_a_matrix() {
    // A view and a half-precision matrix, multiplied in f32.
    let mut b = Builder::<f32>::new();
    let x = b.input(Decl::matrix(DType::F32, (2, 3)));
    let w = b.input(Decl::matrix(DType::F16, (3, 2)));
    let y = b.matmul(x, w);
    b.output(y, DType::F32);
    let graph = exact(b);

    let big = Matrix::from_flat(4, 5, (0..20).map(|v| v as f32).collect::<Vec<_>>());
    let x = big.view(1..3, 1..4);
    let halves: Vec<half::f16> = [1.0, -1.0, 0.5, 2.0, -0.25, 0.0]
        .into_iter()
        .map(half::f16::from_f32)
        .collect();
    let w = Matrix::from_flat(3, 2, halves);
    let got = graph
        .run::<Host>(&[&x, &w], &mut [])
        .remove(0)
        .into_matrix::<f32>();
    let wide = Matrix::from_flat(3, 2, vec![1.0f32, -1.0, 0.5, 2.0, -0.25, 0.0]);
    let want = x.to_matrix().matmul(&wide);
    assert_bits_eq(got.as_slice(), want.as_slice(), "view · f16");
}

#[test]
fn a_product_another_kernel_reads_is_stored_by_its_epilogue_too() {
    // h = x·w₁; y = relu(h)·w₂ + h. The first epilogue writes relu(h) for the
    // second product and keeps h for the second epilogue.
    let f = DType::F32;
    let mut b = Builder::<f32>::new();
    let x = b.input(Decl::matrix(f, (5, 4)));
    let w1 = b.input(Decl::matrix(f, (4, 4)));
    let w2 = b.input(Decl::matrix(f, (4, 4)));
    let h = b.matmul(x, w1);
    let zero = b.constant(0.0);
    let relu = b.compare(Compare::Max, h, zero);
    let second = b.matmul(relu, w2);
    let y = b.add(second, h);
    b.output(y, f);
    let graph = exact(b);
    assert_eq!(graph.kernels(), 2, "{graph}");

    let mut rng = Lcg(3);
    let (x, w1, w2) = (rng.matrix(5, 4), rng.matrix(4, 4), rng.matrix(4, 4));
    let got = graph
        .run::<Host>(&[&x, &w1, &w2], &mut [])
        .remove(0)
        .into_matrix::<f32>();
    let h = x.matmul(&w1);
    let relu = Host::matrix_compare_scalar(&h, 0.0, Compare::Max, false);
    let want = Host::matrix_elementwise(&relu.matmul(&w2), &h, tensorcrate::tensors::BinaryOp::Add);
    assert_bits_eq(got.as_slice(), want.as_slice(), "relu(h)·w₂ + h");
}

#[test]
fn a_training_step_updates_in_place_after_every_product() {
    // p ← p − lr·xᵀ(x·p − t): the residual is the first product's epilogue,
    // the step a last kernel that overwrites p, and lr a uniform.
    let f = DType::F32;
    let mut b = Builder::<f32>::new();
    let x = b.input(Decl::matrix(f, (6, 3)));
    let t = b.input(Decl::matrix(f, (6, 2)));
    let p = b.update(Decl::matrix(f, (3, 2)));
    let lr = b.uniform(0.1);
    let prediction = b.matmul(x, p);
    let residual = b.sub(prediction, t);
    let xt = b.transpose(x, 0, 1);
    let gradient = b.matmul(xt, residual);
    let step = b.mul(lr, gradient);
    let next = b.sub(p, step);
    b.set(0, next);
    let mut graph = exact(b);
    // p copied for the product, the product and residual, xᵀ copied, the
    // gradient, and the step.
    assert_eq!(graph.kernels(), 5, "{graph}");
    graph.set_uniform(0, 0.05);
    assert_eq!(graph.uniforms(), [0.05]);

    let mut rng = Lcg(4);
    let (x, t, start) = (rng.matrix(6, 3), rng.matrix(6, 2), rng.matrix(3, 2));
    let mut p = start.clone();
    let fresh = graph.run::<Host>(&[&x, &t], &mut [&mut p]);
    assert!(fresh.is_empty());

    let residual =
        Host::matrix_elementwise(&x.matmul(&start), &t, tensorcrate::tensors::BinaryOp::Sub);
    let gradient = x.transpose().matmul(&residual);
    let step = Host::matrix_broadcast(&gradient, 0.05, tensorcrate::tensors::BinaryOp::Mul, true);
    let want = Host::matrix_elementwise(&start, &step, tensorcrate::tensors::BinaryOp::Sub);
    assert_bits_eq(p.as_slice(), want.as_slice(), "the step");
}

// ---- dot products ---------------------------------------------------------------

#[test]
fn dot_products_sum_their_fused_operands_along_the_last_axis() {
    // (2x + 1)·v for each row of x: the products and what they are computed
    // from are the summed program.
    let f = DType::F32;
    for (rows, cols) in [(1, 1), (3, 4), (17, 1000), (300, 7)] {
        let mut b = Builder::<f32>::new();
        let x = b.input(Decl::matrix(f, (rows, cols)));
        let v = b.input(Decl::vector(f, cols));
        let doubled = b.scale(x, 2.0);
        let shifted = b.shift(doubled, 1.0);
        let dots = b.dot(shifted, v);
        b.output(dots, f);
        let graph = exact(b);
        // The sums are the output: one kernel computes and sums the products.
        assert_eq!(graph.kernels(), 1, "{graph}");
        assert_eq!(graph.space(), [rows]);

        let mut rng = Lcg(5);
        let (x, v) = (rng.matrix(rows, cols), Vector::new(rng.vector(cols)));
        let got = graph
            .run::<Host>(&[&x, &v], &mut [])
            .remove(0)
            .into_vector::<f32>();
        // As `run_sum` defines a sum: the products, then a product with ones.
        let products = with_row(&x, v.as_slice(), |x, v| (x * 2.0 + 1.0) * v);
        let want = Host::matvec(&products, &Vector::filled(cols, 1.0));
        assert_bits_eq(got.as_slice(), want.as_slice(), &format!("{rows}×{cols}"));
    }
}

#[test]
fn a_dot_product_of_vectors_is_a_scalar() {
    let f = DType::F32;
    let mut b = Builder::<f32>::new();
    let x = b.input(Decl::vector(f, 5));
    let y = b.input(Decl::vector(f, 5));
    let dot = b.dot(x, y);
    assert_eq!(dot.shape(), Vec::<usize>::new());
    let root = b.unary(Analytic::Sqrt, dot);
    b.output(root, f);
    let graph = exact(b);
    assert_eq!(graph.space(), Vec::<usize>::new().as_slice());

    let x = Vector::new([1.0f32, 2.0, 3.0, 4.0, 5.0]);
    let y = Vector::new([1.0f32, 1.0, 1.0, 1.0, 1.0]);
    let got = graph
        .run::<Host>(&[&x, &y], &mut [])
        .remove(0)
        .into_vector::<f32>();
    assert_eq!(got.to_vec(), [15.0f32.sqrt()]);
}

#[test]
fn a_dot_product_of_row_statistics_sums_what_it_computes() {
    // (x − mean)·v, each row: the mean is computed for the program, which
    // then runs, and its output is summed.
    let f = DType::F32;
    let mut b = Builder::<f32>::new();
    let x = b.input(Decl::matrix(f, (4, 6)));
    let v = b.input(Decl::vector(f, 6));
    let mean = b.row_statistic(x, RowStatistic::Mean);
    let centered = b.sub(x, mean);
    let dots = b.dot(centered, v);
    b.output(dots, f);
    let graph = exact(b);

    let mut rng = Lcg(6);
    let (x, v) = (rng.matrix(4, 6), Vector::new(rng.vector(6)));
    let got = graph
        .run::<Host>(&[&x, &v], &mut [])
        .remove(0)
        .into_vector::<f32>();
    for (r, &got) in got.as_slice().iter().enumerate() {
        let row = &x.as_slice()[r * 6..(r + 1) * 6];
        let mean = row.iter().sum::<f32>() / 6.0;
        let want: f32 = row
            .iter()
            .zip(v.as_slice())
            .map(|(x, v)| (x - mean) * v)
            .sum();
        assert!((got - want).abs() < 1e-5, "row {r}: {got} against {want}");
    }
}

// ---- building ---------------------------------------------------------------------

#[test]
fn a_program_with_a_product_needs_a_graph() {
    let f = DType::F32;
    let mut b = Builder::<f32>::new();
    let x = b.input(Decl::matrix(f, (2, 2)));
    let y = b.matmul(x, x);
    b.output(y, f);
    assert_eq!(b.build().unwrap_err(), ProgramError::NotElementwise);

    // Without one, a graph is the one program.
    let mut b = Builder::<f32>::new();
    let x = b.input(Decl::matrix(f, (2, 2)));
    let y = b.shift(x, 1.0);
    b.output(y, f);
    assert_eq!(b.build_graph().unwrap().kernels(), 1);
}

#[test]
#[should_panic(expected = "cannot multiply a [2, 3] value by a [2, 3] one")]
fn a_product_needs_matching_inner_dimensions() {
    let mut b = Builder::<f32>::new();
    let x = b.input(Decl::matrix(DType::F32, (2, 3)));
    b.matmul(x, x);
}

#[test]
fn a_graph_disassembles_each_kernel() {
    let graph = two_layers(2, 3, 4, 5);
    let listing = graph.to_string();
    assert!(
        listing.starts_with("graph over [2, 5], 2 kernels"),
        "{listing}"
    );
    assert!(listing.contains("kernel 0: in0·in1"), "{listing}");
    assert!(listing.contains("kernel 1:"), "{listing}");
}

// ---- Metal against the host -----------------------------------------------------------

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_graphs_match_the_host() {
    let mut rng = Lcg(7);
    for shape in LAYERS {
        let graph = two_layers(shape.0, shape.1, shape.2, shape.3);
        let operands = layer_operands(&mut rng, shape);
        let (x, w1, b1, w2, b2) = &operands;
        let (b1, b2) = (Vector::new(b1.clone()), Vector::new(b2.clone()));
        let want = two_layers_by_hand(&operands);
        let (x, w1, b1, w2, b2) = (
            x.to_backend::<Metal>(),
            w1.to_backend::<Metal>(),
            b1.to_backend::<Metal>(),
            w2.to_backend::<Metal>(),
            b2.to_backend::<Metal>(),
        );
        let inputs: [&dyn Fusable<Metal>; 5] = [&x, &w1, &b1, &w2, &b2];
        // Interpreted, then compiled once the programs have been seen.
        for round in 0..3 {
            let got = graph
                .run(&inputs, &mut [])
                .remove(0)
                .into_vector::<f32>()
                .to_backend::<Host>();
            let tolerance = 1e-5 * (shape.1.max(shape.2) as f32).sqrt() * 4.0;
            assert_close(
                got.as_slice(),
                &want,
                tolerance,
                &format!("{shape:?}, round {round}"),
            );
        }
    }

    let f = DType::F32;
    let mut b = Builder::<f32>::new();
    let x = b.input(Decl::matrix(f, (300, 70)));
    let v = b.input(Decl::vector(f, 70));
    let e = b.unary(Analytic::Tanh, x);
    let dots = b.dot(e, v);
    b.output(dots, f);
    let graph = exact(b);
    let (x, v) = (rng.matrix(300, 70), Vector::new(rng.vector(70)));
    let want = graph
        .run::<Host>(&[&x, &v], &mut [])
        .remove(0)
        .into_vector::<f32>();
    let (gx, gv) = (x.to_backend::<Metal>(), v.to_backend::<Metal>());
    for round in 0..3 {
        let got = graph
            .run::<Metal>(&[&gx, &gv], &mut [])
            .remove(0)
            .into_vector::<f32>()
            .to_backend::<Host>();
        assert_close(
            got.as_slice(),
            want.as_slice(),
            1e-4,
            &format!("dots, round {round}"),
        );
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn a_metal_training_step_matches_the_host() {
    let f = DType::F32;
    let mut b = Builder::<f32>::new();
    let x = b.input(Decl::matrix(f, (40, 8)));
    let t = b.input(Decl::matrix(f, (40, 3)));
    let p = b.update(Decl::matrix(f, (8, 3)));
    let lr = b.uniform(0.01);
    let prediction = b.matmul(x, p);
    let residual = b.sub(prediction, t);
    let xt = b.transpose(x, 0, 1);
    let gradient = b.matmul(xt, residual);
    let step = b.mul(lr, gradient);
    let next = b.sub(p, step);
    b.set(0, next);
    let graph = exact(b);

    let mut rng = Lcg(8);
    let (x, t, start) = (rng.matrix(40, 8), rng.matrix(40, 3), rng.matrix(8, 3));
    let mut host = start.clone();
    let mut metal = start.to_backend::<Metal>();
    let (gx, gt) = (x.to_backend::<Metal>(), t.to_backend::<Metal>());
    for _ in 0..5 {
        graph.run::<Host>(&[&x, &t], &mut [&mut host]);
        graph.run::<Metal>(&[&gx, &gt], &mut [&mut metal]);
    }
    assert_close(
        metal.to_backend::<Host>().as_slice(),
        host.as_slice(),
        1e-4,
        "five steps",
    );
}
