//! Shapes that are not known until the program runs.
//!
//! Everything else in the suite happens to use sizes that are visible in the
//! source, which a const-generic implementation would also accept. These tests
//! are the ones that could not have been written before: each shape here comes
//! from a value the compiler cannot see through, so nothing can be specialized
//! or folded away at compile time.
//!
//! `black_box` is what guarantees that. Without it an optimizer is free to
//! notice that `parse("7")` is `7` and the test would prove nothing about
//! dynamic shapes.

use std::hint::black_box;

use tensorcrate::errors::Error;
use tensorcrate::optim::{Adam, minimize};
use tensorcrate::tensors::{Matrix, Tape, Vector, gradient, jacobian};

/// A dimension the compiler cannot fold: it comes out of a string at runtime.
fn dimension(text: &str) -> usize {
    black_box(text).parse().expect("a number")
}

#[test]
fn vectors_and_matrices_take_their_shape_from_runtime_values() {
    let n = dimension("7");
    let (rows, cols) = (dimension("3"), dimension("5"));

    let v = Vector::<f64>::zeros(n);
    assert_eq!(v.len(), n);

    let m = Matrix::<f64>::from_rows((0..rows).map(|row| {
        (0..cols)
            .map(|col| (row * cols + col) as f64)
            .collect::<Vec<_>>()
    }));
    assert_eq!(m.shape(), (rows, cols));
    assert_eq!(m[(2, 4)], 14.0);
    assert_eq!(m.row(1), [5.0, 6.0, 7.0, 8.0, 9.0]);

    // The same binding can hold different shapes on different iterations, which
    // is the thing a shape-in-the-type design cannot express.
    let mut widths = Vec::new();
    for size in 1..=4 {
        let square = Matrix::<f64>::identity(black_box(size));
        widths.push(square.cols());
        assert_eq!(square.determinant(), 1.0);
    }
    assert_eq!(widths, [1, 2, 3, 4]);
}

#[test]
fn products_compose_across_runtime_shapes() {
    let (r, k, c) = (dimension("4"), dimension("6"), dimension("3"));

    let a = Matrix::<f64>::from_rows(
        (0..r).map(|i| (0..k).map(|j| (i + j) as f64).collect::<Vec<_>>()),
    );
    let b = Matrix::<f64>::from_rows(
        (0..k).map(|i| (0..c).map(|j| (i as f64) - (j as f64)).collect::<Vec<_>>()),
    );

    let product = a.matmul(&b);
    assert_eq!(product.shape(), (r, c));

    // Checked against the definition, since neither shape is a literal here.
    for i in 0..r {
        for j in 0..c {
            let expected: f64 = (0..k).map(|p| a[(i, p)] * b[(p, j)]).sum();
            assert_eq!(product[(i, j)], expected);
        }
    }

    let x = Vector::<f64>::new((0..c).map(|j| j as f64).collect::<Vec<_>>());
    assert_eq!(product.matvec(&x).len(), r);
    assert_eq!(a.transpose().shape(), (k, r));
}

#[test]
fn a_mismatched_shape_panics_and_says_which_shapes() {
    let a = Matrix::<f64>::zeros(dimension("2"), dimension("3"));
    let b = Matrix::<f64>::zeros(dimension("4"), dimension("2"));

    let panic = std::panic::catch_unwind(|| a.matmul(&b)).expect_err("should panic");
    let message = panic
        .downcast_ref::<String>()
        .expect("a formatted panic message");
    assert!(
        message.contains("2×3") && message.contains("4×2"),
        "the message should name both shapes: {message}"
    );

    // Elementwise operations check the whole shape, not just the inner one.
    let u = Vector::<f64>::zeros(dimension("3"));
    let v = Vector::<f64>::zeros(dimension("5"));
    let panic = std::panic::catch_unwind(|| &u + &v).expect_err("should panic");
    let message = panic.downcast_ref::<String>().expect("a message");
    assert!(message.contains('3') && message.contains('5'), "{message}");
}

#[test]
fn a_non_square_matrix_reports_a_shape_error_rather_than_panicking() {
    // Inversion already returned a `Result`, so squareness joins singularity
    // there instead of becoming a panic.
    let m = Matrix::<f64>::zeros(dimension("2"), dimension("3"));
    assert!(matches!(m.inverse(), Err(Error::Shape(_))));
}

#[test]
fn a_tensor_round_trips_without_its_shape_being_known_in_advance() {
    // The reader has no idea what is in the file; the header tells it.
    let original = Matrix::<f64>::from_rows((0..dimension("6")).map(|row| {
        (0..dimension("2"))
            .map(|c| (row + c) as f64)
            .collect::<Vec<_>>()
    }));
    let mut bytes = Vec::new();
    original.write_to(&mut bytes).expect("write");

    let loaded = Matrix::<f64>::read_from(&bytes[..]).expect("read");
    assert_eq!(loaded.shape(), (6, 2));
    assert_eq!(loaded, original);
}

#[test]
fn autodiff_works_at_a_shape_chosen_at_runtime() {
    let n = dimension("9");
    let at = Vector::<f32>::new((0..n).map(|i| i as f32 * 0.25 - 1.0).collect::<Vec<_>>());

    // ∇‖x‖² = 2x, whatever the length.
    let forward = gradient(&at, |x| x.dot(x));
    let reverse = tensorcrate::tensors::tape::gradient(&at, |x| x.dot(x));
    for i in 0..n {
        assert!((forward[i] - 2.0 * at[i]).abs() < 1e-5);
        assert!((reverse[i] - 2.0 * at[i]).abs() < 1e-5);
    }

    // A Jacobian whose shape falls out of the function rather than the type.
    let squares = jacobian(&at, |x| x * x);
    assert_eq!(squares.shape(), (n, n));
    for i in 0..n {
        assert!((squares[(i, i)] - 2.0 * at[i]).abs() < 1e-5);
    }
}

#[test]
fn a_tape_records_shapes_that_differ_between_runs() {
    for size in [2usize, 5, 11] {
        let n = black_box(size);
        let a = Matrix::<f32>::identity(n).scale(2.0);
        let x = Vector::<f32>::new(vec![1.0; n]);

        let tape = Tape::new();
        let recorded_a = tape.matrix(a);
        let recorded_x = tape.vector(x);
        let mapped = recorded_a.matvec(&recorded_x);
        mapped.sum().backward();

        // d(Σ 2x)/dx = 2 in every coordinate, and ∂/∂Aᵢⱼ = xⱼ = 1.
        assert_eq!(recorded_x.grad().to_vec(), vec![2.0; n]);
        assert_eq!(recorded_a.grad().shape(), (n, n));
        assert!(recorded_a.grad().as_slice().iter().all(|&g| g == 1.0));
    }
}

#[test]
fn an_optimizer_fits_a_parameter_whose_length_is_not_a_literal() {
    let features = dimension("4");
    let samples = dimension("16");

    let truth = Vector::<f32>::new(
        (0..features)
            .map(|i| 0.5 - i as f32 * 0.25)
            .collect::<Vec<_>>(),
    );
    let design = Matrix::<f32>::from_rows((0..samples).map(|row| {
        (0..features)
            .map(|col| ((row * 3 + col * 5) % 7) as f32 * 0.25 - 0.75)
            .collect::<Vec<_>>()
    }));
    let targets = design.matvec(&truth);

    let mut parameters = Vector::<f32>::zeros(features);
    let mut rule = Adam::new(0.1);
    minimize(&mut parameters, &mut rule, 1500, |x, _step| {
        let tape = x.tape();
        let residual = &tape.matrix(design.clone()).matvec(x) - &tape.vector(targets.clone());
        residual.dot(&residual)
    });

    assert_eq!(parameters.len(), features);
    for i in 0..features {
        assert!(
            (parameters[i] - truth[i]).abs() < 1e-2,
            "coefficient {i}: {} vs {}",
            parameters[i],
            truth[i]
        );
    }
}

#[test]
fn convolution_shapes_are_derived_from_the_operands() {
    let (rows, cols) = (dimension("9"), dimension("7"));
    let (window_rows, window_cols) = (dimension("3"), dimension("2"));

    let input = Matrix::<f32>::from_rows((0..rows).map(|row| {
        (0..cols)
            .map(|col| ((row + col) % 5) as f32)
            .collect::<Vec<_>>()
    }));
    let window = Matrix::<f32>::from_rows((0..window_rows).map(|_| vec![0.5f32; window_cols]));

    let tape = Tape::new();
    let recorded_input = tape.matrix(input);
    let recorded_window = tape.matrix(window);
    let output = recorded_input.correlate(&recorded_window);

    // Valid correlation: `(R − KR + 1) × (C − KC + 1)`, computed rather than named.
    assert_eq!(
        output.shape(),
        (rows - window_rows + 1, cols - window_cols + 1)
    );

    output.sum().backward();
    assert_eq!(recorded_input.grad().shape(), (rows, cols));
    assert_eq!(recorded_window.grad().shape(), (window_rows, window_cols));

    // Padding widens both axes by twice the margin.
    let padded = recorded_input.pad(dimension("2"), dimension("1"));
    assert_eq!(padded.shape(), (rows + 4, cols + 2));
}
