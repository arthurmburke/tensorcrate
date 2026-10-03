//! Dynamically-shaped vector and matrix operations.

use tensorcrate::errors::Error;
use tensorcrate::numbers::{Complex, Dual};
use tensorcrate::tensors::{BinaryOp, Matrix, Vector, chained_matmul_cost};

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

#[test]
fn vectors_carry_their_length() {
    let u: Vector<i64> = Vector::new([1, 2, 3]);
    let v: Vector<i64> = Vector::new([4, 5, 6]);
    assert_eq!(u.len(), 3);
    assert_eq!(&u + &v, Vector::new([5, 7, 9]));
    assert_eq!(u.dot(&v), 32);
    assert_eq!(u.scale(2), Vector::new([2, 4, 6]));
}

#[test]
fn fft_handles_power_of_two_and_general_lengths() {
    let impulse = Vector::new([1.0_f64, 0.0, 0.0, 0.0]).fft();
    for value in impulse.data() {
        assert!(close(value.real, 1.0));
        assert!(close(value.im, 0.0));
    }

    let shifted = Vector::new([0.0_f64, 1.0, 0.0, 0.0]).fft();
    let expected = [
        Complex::new(1.0, 0.0),
        Complex::new(0.0, -1.0),
        Complex::new(-1.0, 0.0),
        Complex::new(0.0, 1.0),
    ];
    for (actual, expected) in shifted.data().iter().zip(expected) {
        assert!(close(actual.real, expected.real));
        assert!(close(actual.im, expected.im));
    }

    // Three is a prime length; the direct leaf must use the same convention and
    // output ordering as the fast paths.
    let general = Vector::new([1.0_f64, 2.0, 3.0]).fft();
    let root = 3.0_f64.sqrt() / 2.0;
    let expected = [
        Complex::new(6.0, 0.0),
        Complex::new(-1.5, root),
        Complex::new(-1.5, -root),
    ];
    for (actual, expected) in general.data().iter().zip(expected) {
        assert!(close(actual.real, expected.real));
        assert!(close(actual.im, expected.im));
    }
}

#[test]
fn fft_supports_edge_lengths_and_f32() {
    let empty = Vector::<f64>::new([]).fft();
    assert!(empty.is_empty());

    let singleton = Vector::new([7.0_f64]).fft();
    assert_eq!(singleton, Vector::new([Complex::new(7.0, 0.0)]));

    let values: Vector<Complex<f32>> = Vector::new([1.0_f32, -1.0]).fft();
    assert!((values.get(0).unwrap().real - 0.0).abs() < 1e-6);
    assert!((values.get(1).unwrap().real - 2.0).abs() < 1e-6);
}

fn assert_fft_matches_dft<const N: usize>(input: [f64; N]) {
    let actual = Vector::new(input).fft();
    let tau = std::f64::consts::TAU;
    for frequency in 0..N {
        let mut expected = Complex::new(0.0, 0.0);
        for (index, value) in input.iter().enumerate() {
            let angle = -tau * frequency as f64 * index as f64 / N as f64;
            expected = expected + Complex::new(value * angle.cos(), value * angle.sin());
        }
        assert!(
            close(actual[frequency].real, expected.real)
                && close(actual[frequency].im, expected.im),
            "N={N}, bin={frequency}: actual={:?}, expected={expected:?}",
            actual[frequency]
        );
    }
}

#[test]
fn fft_uses_mixed_radices_for_composite_lengths() {
    // Exercise repeated and mixed factors: 3², 2×3, 2×5, 2²×3, and 3×5.
    assert_fft_matches_dft([1.0, -2.0, 3.5, 4.0, -1.0, 0.5]);
    assert_fft_matches_dft([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_fft_matches_dft([1.0, 0.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 5.0]);
    assert_fft_matches_dft([
        0.25, 1.0, -1.5, 2.0, 3.25, -0.5, 4.0, -2.0, 1.25, 0.0, 2.5, -3.0,
    ]);
    assert_fft_matches_dft([
        1.0, 1.5, -2.0, 0.0, 3.0, -1.0, 2.25, 4.0, -3.5, 0.5, 1.25, -0.75, 2.0, -1.5, 3.25,
    ]);
}

/// Deterministic non-degenerate samples, so a length can be exercised without
/// spelling out an array literal of that size.
fn ramp<const N: usize>() -> [f64; N] {
    std::array::from_fn(|i| ((i as f64) * 0.7).sin() * 3.0 + (i as f64) * 0.01)
}

#[test]
fn fft_matches_dft_at_deep_recursion_depths() {
    // The mixed-radix decomposition shares one root table across the whole
    // recursion, indexed by a stride that grows with each split. These lengths
    // exercise several distinct descent shapes: a three-distinct-factor chain,
    // a repeated-factor chain four deep, a squared prime, and a length whose
    // smallest factor exceeds the radix cutoff so it hits the direct path both
    // at the root and beneath a split.
    assert_fft_matches_dft(ramp::<105>()); // 3·5·7
    assert_fft_matches_dft(ramp::<240>()); // 2⁴·3·5
    assert_fft_matches_dft(ramp::<121>()); // 11²
    assert_fft_matches_dft(ramp::<17>()); // prime > 15, direct at the root
    assert_fft_matches_dft(ramp::<34>()); // 2·17, direct beneath a split
    assert_fft_matches_dft(ramp::<98>()); // 2·7²
}

#[test]
fn fft_ifft_round_trips_at_deep_recursion_depths() {
    assert_fft_ifft_round_trip(ramp::<105>());
    assert_fft_ifft_round_trip(ramp::<240>());
    assert_fft_ifft_round_trip(ramp::<121>());
    assert_fft_ifft_round_trip(ramp::<34>());
    assert_fft_ifft_round_trip(ramp::<1000>()); // 2³·5³, the deepest chain here
}

fn assert_fft_ifft_round_trip<const N: usize>(input: [f64; N]) {
    let reconstructed = Vector::new(input).fft().ifft();
    for (actual, expected) in reconstructed.data().iter().zip(input) {
        assert!(
            close(actual.real, expected) && close(actual.im, 0.0),
            "N={N}: actual={actual:?}, expected={expected}"
        );
    }
}

fn assert_ifft_matches_idft<const N: usize>(input: [Complex<f64>; N]) {
    let actual = Vector::new(input).ifft();
    let tau = std::f64::consts::TAU;
    for index in 0..N {
        let mut expected = Complex::new(0.0, 0.0);
        for (frequency, value) in input.iter().enumerate() {
            let angle = tau * frequency as f64 * index as f64 / N as f64;
            expected = expected + *value * Complex::new(angle.cos(), angle.sin());
        }
        expected = expected / N as f64;
        assert!(
            close(actual[index].real, expected.real) && close(actual[index].im, expected.im),
            "N={N}, sample={index}: actual={:?}, expected={expected:?}",
            actual[index]
        );
    }
}

#[test]
fn ifft_inverts_power_of_two_mixed_radix_and_prime_transforms() {
    assert_fft_ifft_round_trip([1.0, -2.0, 3.5, 4.0]);
    assert_fft_ifft_round_trip([1.0, -2.0, 3.5, 4.0, -1.0, 0.5]);
    assert_fft_ifft_round_trip([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_fft_ifft_round_trip([1.0, 0.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 5.0]);
    assert_fft_ifft_round_trip([
        0.25, 1.0, -1.5, 2.0, 3.25, -0.5, 4.0, -2.0, 1.25, 0.0, 2.5, -3.0,
    ]);
    assert_fft_ifft_round_trip([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

    assert_ifft_matches_idft([
        Complex::new(1.0, -0.5),
        Complex::new(-2.0, 1.0),
        Complex::new(3.5, 2.0),
        Complex::new(0.0, -1.5),
        Complex::new(2.25, 0.75),
        Complex::new(-3.0, 4.0),
    ]);
}

#[test]
fn ifft_supports_edge_lengths_and_f32() {
    let empty = Vector::<Complex<f64>>::new([]).ifft();
    assert!(empty.is_empty());

    let singleton = Vector::new([Complex::new(7.0_f64, -2.0)]).ifft();
    assert_eq!(singleton, Vector::new([Complex::new(7.0, -2.0)]));

    let values = Vector::new([Complex::new(0.0_f32, 0.0), Complex::new(2.0, 0.0)]).ifft();
    assert!((values[0].real - 1.0).abs() < 1e-6);
    assert!((values[1].real + 1.0).abs() < 1e-6);
    assert!(values.data().iter().all(|value| value.im.abs() < 1e-6));
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn tensor_operations_cover_matrix_vector_shapes() {
    let a = Matrix::<f32>::from_rows((0..32).map(|row| {
        (0..32)
            .map(|col| (row + col) as f32 * 0.125)
            .collect::<Vec<_>>()
    }));
    let identity = Matrix::<f32>::identity(32);
    assert_eq!(a.matmul(&identity), a);

    let wide = Matrix::<f32>::from_rows([[1.0f32; 256]; 128]);
    let row = Vector::<f32>::new([1.0; 128]);
    assert!(row.vecmat(&wide).data().iter().all(|&value| value == 128.0));
    let tall = wide.transpose();
    let column = Vector::<f32>::new([1.0; 128]);
    assert!(
        tall.matvec(&column)
            .data()
            .iter()
            .all(|&value| value == 128.0)
    );

    let dot_left = Vector::<f32>::new([1.0; 32768]);
    let dot_right = Vector::<f32>::new([2.0; 32768]);
    assert_eq!(dot_left.dot(&dot_right), 65536.0);

    let values = Vector::<f32>::new(
        (0..4096)
            .map(|index| index as f32 * 0.25)
            .collect::<Vec<_>>(),
    );
    let factors = Vector::<f32>::new(
        (0..4096)
            .map(|index| (index % 7) as f32 + 1.0)
            .collect::<Vec<_>>(),
    );
    let multiplied = &values * &factors;
    let scaled = values.scale(2.0);
    let shifted = values.broadcast_right(3.0, BinaryOp::Add);
    let reversed = values.broadcast_left(10.0, BinaryOp::Sub);
    for index in 0..4096 {
        assert_eq!(multiplied[index], values[index] * factors[index]);
        assert_eq!(scaled[index], values[index] * 2.0);
        assert_eq!(shifted[index], values[index] + 3.0);
        assert_eq!(reversed[index], 10.0 - values[index]);
    }

    let signal = Vector::<f32>::new(
        (0..4096)
            .map(|index| (index % 19) as f32 - 4.0)
            .collect::<Vec<_>>(),
    );
    let reconstructed = signal.fft().ifft();
    for (actual, expected) in reconstructed.data().iter().zip(signal.data()) {
        assert!((actual.real - expected).abs() < 2e-3);
        assert!(actual.im.abs() < 2e-3);
    }
}

#[test]
fn matrices_are_statically_shaped() {
    let a: Matrix<i32> = Matrix::from_rows([[1, 2, 3], [4, 5, 6]]);
    let b: Matrix<i32> = Matrix::from_rows([[7, 8], [9, 10], [11, 12]]);
    let product: Matrix<i32> = a.matmul(&b);
    assert_eq!(product, Matrix::from_rows([[58, 64], [139, 154]]));
    assert_eq!(a.transpose(), Matrix::from_rows([[1, 4], [2, 5], [3, 6]]));

    let column: Vector<i32> = Vector::new([1, 2, 3]);
    assert_eq!(a.matvec(&column), Vector::new([14, 32]));
    let row: Vector<i32> = Vector::new([1, 2]);
    assert_eq!(row.vecmat(&a), Vector::new([9, 12, 15]));
}

#[test]
fn chained_matmul_restores_optimal_order_and_const_result_shape() {
    let a: Matrix<i64> = Matrix::from_rows([[1, 2, 3], [4, 5, 6]]);
    let b: Matrix<i64> = Matrix::from_rows([[1, 0], [0, 1], [1, 1]]);
    let c: Matrix<i64> = Matrix::from_rows([[1, 2, 3, 4], [5, 6, 7, 8]]);
    let chain = [&a, &b, &c];

    let product: Matrix<i64> = Matrix::chained_matmul(&chain).unwrap();
    assert_eq!(product, a.matmul(&b).matmul(&c));
    assert_eq!(Matrix::chained_matmul_cost(&chain), Ok(28));
}

#[test]
fn hu_shing_cost_finds_the_classic_optimum() {
    let a = Matrix::<i64>::zeros(10, 20);
    let b = Matrix::<i64>::zeros(20, 5);
    let c = Matrix::<i64>::zeros(5, 30);
    let chain = [&a, &b, &c];
    assert_eq!(Matrix::chained_matmul_cost(&chain), Ok(2500));
    assert_eq!(chained_matmul_cost(&[(10, 20), (20, 5), (5, 30)]), Ok(2500));
}

#[test]
fn determinant_and_inverse() {
    let a: Matrix<f64> = Matrix::from_rows([[4.0, 7.0], [2.0, 6.0]]);
    assert!(close(a.determinant(), 10.0));
    let identity = a.matmul(&a.inverse().unwrap());
    assert!(close(*identity.get(0, 0).unwrap(), 1.0));
    assert!(close(*identity.get(0, 1).unwrap(), 0.0));
    assert!(close(*identity.get(1, 1).unwrap(), 1.0));

    let singular = Matrix::from_rows([[1.0, 2.0], [2.0, 4.0]]);
    assert_eq!(singular.inverse(), Err(Error::Singular));

    // Integer Gauss–Jordan division would truncate the first 1/2 to zero and
    // silently return the wrong answer, even though this matrix is unimodular.
    let integer = Matrix::<i32>::from_rows([[2, 1], [1, 1]]);
    assert!(matches!(
        integer.inverse(),
        Err(Error::InvalidArgument(message))
            if message.contains("fractional division")
    ));
}

#[test]
fn complex_and_dual_elements_work() {
    let i = Complex::new(0.0, 1.0);
    let z = Complex::new(0.0, 0.0);
    let m: Matrix<Complex<f64>> = Matrix::from_rows([[i, z], [z, i]]);
    assert_eq!(m.matmul(&m).get(0, 0), Some(&Complex::new(-1.0, 0.0)));

    let x = Dual::variable(2.0);
    let one = Dual::constant(1.0);
    let zero = Dual::constant(0.0);
    let a: Matrix<Dual<f64>> = Matrix::from_rows([[x, one], [zero, one]]);
    let squared = a.matmul(&a);
    assert_eq!(
        (
            squared.get(0, 0).unwrap().real,
            squared.get(0, 0).unwrap().dual
        ),
        (4.0, 4.0)
    );
}

#[test]
fn all_vector_and_matrix_products_support_complex_coefficients() {
    let gaussian: Vector<Complex<i32>> = Vector::new([Complex::new(1, 2)]);
    assert_eq!(gaussian.dot(&gaussian), Complex::new(-3, 4));

    let u = Vector::new([Complex::new(1.0, 1.0), Complex::new(3.0, 0.0)]);
    let v = Vector::new([Complex::new(2.0, -1.0), Complex::new(0.0, 1.0)]);
    assert_eq!(u.dot(&v), Complex::new(3.0, 4.0));

    let matrix = Matrix::from_rows([
        [Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)],
        [Complex::new(2.0, 0.0), Complex::new(1.0, 0.0)],
    ]);
    assert_eq!(
        matrix.matvec(&v),
        Vector::new([Complex::new(1.0, -1.0), Complex::new(4.0, -1.0)])
    );
    assert_eq!(
        u.vecmat(&matrix),
        Vector::new([Complex::new(7.0, 1.0), Complex::new(2.0, 1.0)])
    );

    let identity = Matrix::<Complex<f64>>::identity(2);
    assert_eq!(matrix.matmul(&identity), matrix);
}

#[test]
fn all_vector_and_matrix_products_support_dual_coefficients() {
    let integral: Vector<Dual<i64>> = Vector::new([Dual::new(2, 1)]);
    assert_eq!(integral.dot(&integral), Dual::new(4, 4));

    let x = Dual::variable(2.0_f64);
    let one = Dual::constant(1.0);
    let two = Dual::constant(2.0);
    let three = Dual::constant(3.0);
    let zero = Dual::constant(0.0);

    let u = Vector::new([x, one]);
    let v = Vector::new([two, three]);
    assert_eq!(u.dot(&v), Dual::new(7.0, 2.0));

    let matrix = Matrix::from_rows([[x, one], [zero, one]]);
    assert_eq!(
        matrix.matvec(&v),
        Vector::new([Dual::new(7.0, 2.0), Dual::constant(3.0)])
    );
    assert_eq!(
        u.vecmat(&matrix),
        Vector::new([Dual::new(4.0, 4.0), Dual::new(3.0, 1.0)])
    );

    let squared = matrix.matmul(&matrix);
    assert_eq!(squared.get(0, 0), Some(&Dual::new(4.0, 4.0)));
    assert_eq!(squared.get(0, 1), Some(&Dual::new(3.0, 1.0)));
}

#[test]
fn constructors_and_map_preserve_shape() {
    let zeros: Matrix<i32> = Matrix::zeros(2, 3);
    assert_eq!(zeros.shape(), (2, 3));
    assert_eq!(zeros.to_rows(), [[0, 0, 0], [0, 0, 0]]);
    let identity: Matrix<f64> = Matrix::identity(3);
    assert_eq!(
        identity.to_rows(),
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    );

    let real: Vector<f64> = Vector::new([1.0, 2.0]);
    let lifted: Vector<Complex<f64>> = real.map(|&x| Complex::constant(x));
    assert_eq!(lifted.get(1), Some(&Complex::new(2.0, 0.0)));
}

/// Small integers, so any summation order gives the same float.
fn integral<T: From<i16>>(rows: usize, cols: usize, seed: usize) -> Vec<T> {
    (0..rows * cols)
        .map(|i| T::from(((i * 7 + seed * 13) % 9) as i16 - 4))
        .collect()
}

#[test]
fn transposes_of_every_shape_move_each_element_once() {
    // Shapes on either side of the blocking and of the size at which the
    // transpose splits across threads, including long thin ones.
    for (rows, cols) in [
        (0, 5),
        (1, 1),
        (3, 70),
        (33, 31),
        (1001, 333),
        (2, 70_001),
        (70_001, 3),
    ] {
        let data: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
        let t = Matrix::from_flat(rows, cols, data.clone()).transpose();
        assert_eq!(t.shape(), (cols, rows));
        for i in 0..rows {
            for j in 0..cols {
                assert_eq!(
                    t.as_slice()[j * rows + i],
                    data[i * cols + j],
                    "{rows}×{cols} at ({i}, {j})"
                );
            }
        }
        // A type the threads do not split, through the same blocking.
        let ints: Vec<i64> = (0..rows * cols).map(|i| i as i64).collect();
        let t = Matrix::from_flat(rows, cols, ints.clone()).transpose();
        assert!((0..rows * cols).all(|k| t.as_slice()[(k % cols) * rows + k / cols] == ints[k]));
    }
}

#[test]
fn products_reading_an_operand_transposed_match_transposing_it_first() {
    use tensorcrate::tensors::{Host, Kernels, Transposed};
    fn check<T: tensorcrate::numbers::Real + From<i16>>()
    where
        Host: Kernels<T>,
    {
        // Big enough for Accelerate, and too small for it.
        for (m, k, n) in [(37, 53, 29), (2, 3, 4)] {
            let addend = Matrix::<T>::from_flat(m, n, integral(m, n, 1));
            // aᵀ·b, with a stored k × m.
            let a = Matrix::<T>::from_flat(k, m, integral(k, m, 2));
            let b = Matrix::<T>::from_flat(k, n, integral(k, n, 3));
            let read = Host::matmul_transposed_add(&a, &b, Transposed::Left, addend.clone());
            let copied = Host::matmul_add(&a.transpose(), &b, addend.clone());
            assert_eq!(read, copied, "aᵀ·b, {m}×{k}×{n}");
            // a·bᵀ, with b stored n × k.
            let a = Matrix::<T>::from_flat(m, k, integral(m, k, 4));
            let b = Matrix::<T>::from_flat(n, k, integral(n, k, 5));
            let read = Host::matmul_transposed_add(&a, &b, Transposed::Right, addend.clone());
            let copied = Host::matmul_add(&a, &b.transpose(), addend);
            assert_eq!(read, copied, "a·bᵀ, {m}×{k}×{n}");
        }
    }
    check::<f32>();
    check::<f64>();
}

#[test]
#[should_panic(expected = "inner dimensions")]
fn a_transposed_product_checks_its_inner_dimensions() {
    use tensorcrate::tensors::{Host, Kernels, Transposed};
    let a = Matrix::<f32>::from_flat(40, 30, vec![0.0; 1200]);
    let b = Matrix::<f32>::from_flat(40, 31, vec![0.0; 1240]);
    // aᵀ·b needs a's rows to match b's rows: fine. a·bᵀ needs columns to match.
    let _ = Host::matmul_transposed_add(
        &a,
        &b,
        Transposed::Right,
        Matrix::from_flat(40, 40, vec![0.0; 1600]),
    );
}
