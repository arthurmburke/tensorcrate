//! Convolutions: the primal operation and both of its gradients.
//!
//! The forward pass is checked against convolutions worked out by hand, because
//! an implementation that is self-consistently off by a flip or a shift would
//! pass a comparison against itself. The gradients are then checked three ways:
//! against forward mode, against finite differences, and `Metal` against `Host`.

use tensorcrate::optim::{Adam, minimize};
use tensorcrate::tensors::dual::{DualMatrix, gradient_wrt_matrix};
use tensorcrate::tensors::{Host, Kernels, Matrix, Tape};

#[cfg(all(feature = "metal", target_os = "macos"))]
use tensorcrate::tensors::Metal;

fn close(actual: f32, expected: f32) -> bool {
    (actual - expected).abs() <= 2e-3 * (1.0 + expected.abs())
}

fn assert_slice_close(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(close(a, e), "{what}: at {index}, {a} vs {e}");
    }
}

/// A deterministic pseudo-random image. Smooth or periodic patterns look fine
/// but leave a filter under-determined: several windows then produce identical
/// outputs, and a fit recovers one of them rather than the one used.
fn image(rows: usize, cols: usize) -> Matrix<f32> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / 8_388_608.0 - 1.0
    };
    Matrix::from_rows((0..rows).map(|_| (0..cols).map(|_| next()).collect::<Vec<_>>()))
}

fn window(rows: usize, cols: usize) -> Matrix<f32> {
    Matrix::from_rows((0..rows).map(|row| {
        (0..cols)
            .map(|col| 0.4 - ((row + 2 * col) % 5) as f32 * 0.25)
            .collect::<Vec<_>>()
    }))
}

// ---- the forward pass, against convolutions done by hand ----------------------

#[test]
fn correlation_slides_the_window_without_reversing_it() {
    // A 3×3 input and a 2×2 window give a 2×2 output; every entry is a sum of
    // four products that can be written out.
    let input = Matrix::<f32>::from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let taps = Matrix::<f32>::from_rows([[1.0, 10.0], [100.0, 1000.0]]);

    let correlated = Host::correlate(&input, &taps, false);
    assert_eq!(
        correlated.to_rows(),
        [
            // 1·1 + 2·10 + 4·100 + 5·1000, then the window slides right and down
            [5421.0, 6532.0],
            [8754.0, 9865.0],
        ]
    );

    // Convolution reverses the window, so the same input gives the reversed sum:
    // 1·1000 + 2·100 + 4·10 + 5·1.
    let convolved = Host::correlate(&input, &taps, true);
    assert_eq!(convolved.to_rows(), [[1245.0, 2356.0], [4578.0, 5689.0]]);

    // A 1×1 window is a plain scale, whichever convention is used.
    let scale = Matrix::<f32>::from_rows([[3.0]]);
    assert_eq!(
        Host::correlate(&input, &scale, false).to_rows(),
        input.scale(3.0).to_rows()
    );
    assert_eq!(
        Host::correlate(&input, &scale, true).to_rows(),
        input.scale(3.0).to_rows()
    );

    // A window the size of the input leaves a single number: the inner product.
    let whole = Host::correlate(&input, &image(3, 3), false);
    let expected: f32 = input
        .as_slice()
        .iter()
        .zip(image(3, 3).as_slice())
        .map(|(a, b)| a * b)
        .sum();
    assert!(close(whole.to_rows()[0][0], expected));
}

#[test]
fn padding_and_flipping_do_what_they_say() {
    let input = Matrix::<f32>::from_rows([[1.0, 2.0], [3.0, 4.0]]);

    assert_eq!(
        Host::pad(&input, 1, 1).to_rows(),
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 0.0],
            [0.0, 3.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    );
    // Asymmetric padding, and none at all.
    assert_eq!(
        Host::pad(&input, 0, 2).to_rows(),
        [
            [0.0, 0.0, 1.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 4.0, 0.0, 0.0]
        ]
    );
    assert_eq!(Host::pad(&input, 0, 0).to_rows(), input.to_rows());

    assert_eq!(Host::flip(&input).to_rows(), [[4.0, 3.0], [2.0, 1.0]]);
    assert_eq!(Host::flip(&Host::flip(&input)).to_rows(), input.to_rows());

    // Convolution is correlation with the window flipped — the identity the two
    // conventions differ by.
    let taps = window(2, 3);
    let image = image(4, 5);
    assert_slice_close(
        Host::correlate(&image, &taps, true).as_slice(),
        Host::correlate(&image, &Host::flip(&taps), false).as_slice(),
        "convolve = correlate with a flipped window",
    );
}

// ---- the gradients -----------------------------------------------------------

#[test]
fn both_gradients_agree_with_forward_mode() {
    let input = image(6, 7);
    let taps = window(3, 2);

    // ∂/∂input of Σ correlate(X, K).
    let tape = Tape::new();
    let recorded_input = tape.matrix(input.clone());
    let recorded_taps = tape.matrix(taps.clone());
    recorded_input.correlate(&recorded_taps).sum().backward();

    let forward_input = gradient_wrt_matrix(&input, |x| {
        x.correlate(&DualMatrix::constant(taps.clone())).sum()
    });
    assert_slice_close(
        recorded_input.grad().as_slice(),
        forward_input.as_slice(),
        "∂/∂input",
    );

    let forward_taps = gradient_wrt_matrix(&taps, |k| {
        DualMatrix::constant(input.clone()).correlate(k).sum()
    });
    assert_slice_close(
        recorded_taps.grad().as_slice(),
        forward_taps.as_slice(),
        "∂/∂window",
    );

    // The same for the flipped convention.
    let tape = Tape::new();
    let recorded_input = tape.matrix(input.clone());
    let recorded_taps = tape.matrix(taps.clone());
    recorded_input.convolve(&recorded_taps).sum().backward();

    assert_slice_close(
        recorded_input.grad().as_slice(),
        gradient_wrt_matrix(&input, |x| {
            x.convolve(&DualMatrix::constant(taps.clone())).sum()
        })
        .as_slice(),
        "∂/∂input, convolution",
    );
    assert_slice_close(
        recorded_taps.grad().as_slice(),
        gradient_wrt_matrix(&taps, |k| {
            DualMatrix::constant(input.clone()).convolve(k).sum()
        })
        .as_slice(),
        "∂/∂window, convolution",
    );
}

#[test]
fn the_window_gradient_is_the_sum_of_the_patches_it_saw() {
    // With a loss of Σ Y, every output element contributes 1, so ∂L/∂K[a][b] is
    // just the sum of the input entries that tap ever multiplied.
    let input = image(5, 4);
    let tape = Tape::new();
    let taps = tape.matrix(window(2, 2));
    tape.matrix(input.clone()).correlate(&taps).sum().backward();

    for tap_row in 0..2 {
        for tap_col in 0..2 {
            let expected: f32 = (0..4)
                .flat_map(|row| (0..3).map(move |col| (row, col)))
                .map(|(row, col)| input[(row + tap_row, col + tap_col)])
                .sum();
            assert!(
                close(taps.grad()[(tap_row, tap_col)], expected),
                "∂/∂K[{tap_row},{tap_col}]: {} vs {expected}",
                taps.grad()[(tap_row, tap_col)]
            );
        }
    }
}

#[test]
fn the_input_gradient_counts_how_often_each_pixel_was_used() {
    // Σ correlate(X, K) differentiates to a map of window sums: interior pixels
    // are seen by every tap, edges by fewer. With a window of all ones, that is
    // literally a count of the windows covering each pixel.
    let ones = Matrix::<f32>::filled(2, 2, 1.0);
    let tape = Tape::new();
    let input = tape.matrix(image(4, 4));
    input.correlate(&tape.matrix(ones)).sum().backward();

    // A 4×4 input with a 2×2 window: corners covered once, edges twice, the
    // 2×2 interior four times.
    assert_eq!(
        input.grad().to_rows(),
        [
            [1.0, 2.0, 2.0, 1.0],
            [2.0, 4.0, 4.0, 2.0],
            [2.0, 4.0, 4.0, 2.0],
            [1.0, 2.0, 2.0, 1.0],
        ]
    );
}

#[test]
fn convolution_gradients_match_central_differences() {
    let input = image(5, 5);
    let taps = window(3, 3);

    // A loss with some structure, so the gradient is not a constant map.
    let loss_at = |image: Matrix<f32>, window: Matrix<f32>| -> f32 {
        let output = Host::correlate(&image, &window, false);
        output.as_slice().iter().map(|value| value.tanh()).sum()
    };

    let tape = Tape::new();
    let recorded_input = tape.matrix(input.clone());
    let recorded_taps = tape.matrix(taps.clone());
    recorded_input
        .correlate(&recorded_taps)
        .tanh()
        .sum()
        .backward();

    let step = 1e-3;
    for row in 0..5 {
        for col in 0..5 {
            let mut forward = input.to_rows();
            let mut backward = input.to_rows();
            forward[row][col] += step;
            backward[row][col] -= step;
            let numeric = (loss_at(Matrix::from_rows(forward.clone()), taps.clone())
                - loss_at(Matrix::from_rows(backward.clone()), taps.clone()))
                / (2.0 * step);
            assert!(
                (recorded_input.grad()[(row, col)] - numeric).abs() < 5e-3,
                "∂/∂X[{row},{col}]: {} vs {numeric}",
                recorded_input.grad()[(row, col)]
            );
        }
    }
    for row in 0..3 {
        for col in 0..3 {
            let mut forward = taps.to_rows();
            let mut backward = taps.to_rows();
            forward[row][col] += step;
            backward[row][col] -= step;
            let numeric = (loss_at(input.clone(), Matrix::from_rows(forward.clone()))
                - loss_at(input.clone(), Matrix::from_rows(backward.clone())))
                / (2.0 * step);
            assert!(
                (recorded_taps.grad()[(row, col)] - numeric).abs() < 5e-3,
                "∂/∂K[{row},{col}]: {} vs {numeric}",
                recorded_taps.grad()[(row, col)]
            );
        }
    }
}

#[test]
fn padding_and_flipping_differentiate() {
    let input = image(3, 4);

    // Padding routes the adjoint back from the interior and drops the border.
    let tape = Tape::new();
    let recorded = tape.matrix(input.clone());
    let padded = recorded.pad(2, 1);
    assert_eq!(padded.value().shape(), (7, 6));
    padded.sum().backward();
    assert_eq!(recorded.grad().to_rows(), [[1.0; 4]; 3]);

    // Seeding one padded cell credits only the pixel underneath it.
    let tape = Tape::new();
    let recorded = tape.matrix(input.clone());
    let mut seed = [[0.0f32; 6]; 7];
    seed[3][2] = 1.0; // interior: input row 1, col 1
    recorded.pad(2, 1).backward_with(Matrix::from_rows(seed));
    let mut expected = [[0.0f32; 4]; 3];
    expected[1][1] = 1.0;
    assert_eq!(recorded.grad().to_rows(), expected);

    // A seed in the border credits nothing.
    let tape = Tape::new();
    let recorded = tape.matrix(input.clone());
    let mut seed = [[0.0f32; 6]; 7];
    seed[0][0] = 1.0;
    recorded.pad(2, 1).backward_with(Matrix::from_rows(seed));
    assert_eq!(recorded.grad().to_rows(), [[0.0; 4]; 3]);

    // Flipping is its own inverse, gradient included.
    let tape = Tape::new();
    let recorded = tape.matrix(input.clone());
    let flipped = recorded.flipped();
    assert_eq!(flipped.value().to_rows(), Host::flip(&input).to_rows());
    let mut seed = [[0.0f32; 4]; 3];
    seed[0][0] = 1.0;
    flipped.backward_with(Matrix::from_rows(seed));
    let mut expected = [[0.0f32; 4]; 3];
    expected[2][3] = 1.0;
    assert_eq!(recorded.grad().to_rows(), expected);
}

// ---- it learns ----------------------------------------------------------------

#[test]
fn a_convolutional_filter_can_be_learned_from_its_output() {
    // The classic sanity check: generate data with a known filter, then recover
    // the filter from the images alone.
    let input = image(8, 8);
    let truth =
        Matrix::<f32>::from_rows([[0.25, -0.5, 0.25], [-0.5, 1.0, -0.5], [0.25, -0.5, 0.25]]);
    let targets = Host::correlate(&input, &truth, false);

    let mut taps = Matrix::<f32>::zeros(3, 3);
    let final_loss = minimize(&mut taps, &mut Adam::new(0.05), 3000, |k, _| {
        let tape = k.tape();
        let predicted = tape.matrix(input.clone()).correlate(k);
        let residual = &predicted - &tape.matrix(targets.clone());
        residual.frobenius_dot(&residual)
    });

    assert!(final_loss < 1e-6, "loss {final_loss}");
    assert_slice_close(taps.as_slice(), truth.as_slice(), "recovered filter");
}

// ---- the two backends ---------------------------------------------------------

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn convolutions_agree_between_the_backends() {
    let input = image(9, 7);
    let taps = window(3, 4);

    for flip in [false, true] {
        let host = Host::correlate(&input, &taps, flip);
        let resident = Metal::correlate(&input.to_backend(), &taps.to_backend(), flip);
        assert_slice_close(
            resident.as_slice(),
            host.as_slice(),
            &format!("correlate, flip = {flip}"),
        );
    }

    assert_slice_close(
        Metal::pad(&input.to_backend(), 2, 3).as_slice(),
        Host::pad(&input, 2, 3).as_slice(),
        "pad",
    );
    assert_slice_close(
        Metal::flip(&input.to_backend()).as_slice(),
        Host::flip(&input).as_slice(),
        "flip",
    );

    // Both gradients, through a nonlinearity, on both backends.
    let host_tape = Tape::<Host>::new();
    let host_input = host_tape.matrix(input.clone());
    let host_taps = host_tape.matrix(taps.clone());
    host_input.correlate(&host_taps).tanh().sum().backward();

    let gpu_tape = Tape::<Metal>::new();
    let gpu_input = gpu_tape.matrix(input.to_backend::<Metal>());
    let gpu_taps = gpu_tape.matrix(taps.to_backend::<Metal>());
    gpu_input.correlate(&gpu_taps).tanh().sum().backward();

    assert_slice_close(
        gpu_input.grad().as_slice(),
        host_input.grad().as_slice(),
        "∂/∂input",
    );
    assert_slice_close(
        gpu_taps.grad().as_slice(),
        host_taps.grad().as_slice(),
        "∂/∂window",
    );
    if input.to_backend::<Metal>().is_device_resident() {
        assert!(
            gpu_input.grad().is_device_resident(),
            "the input gradient stays resident"
        );
        assert!(
            gpu_taps.grad().is_device_resident(),
            "the window gradient stays resident"
        );
    }
}
