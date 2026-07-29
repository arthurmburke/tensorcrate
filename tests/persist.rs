//! Saving and loading tensors.
//!
//! Almost everything here round-trips through a `Vec<u8>` rather than the
//! filesystem: the property under test is that bytes written are bytes read,
//! and a temporary file adds nothing to that but flakiness. One test at the end
//! covers the `save`/`load` path helpers, since those are the part a `Vec<u8>`
//! round trip cannot exercise.
//!
//! The two things worth pinning down are that *every* element type survives —
//! including the nested `Complex`/`Dual` combinations, and including float bit
//! patterns that compare unequal to themselves — and that each way of getting
//! it wrong produces its own error rather than silently reinterpreted data.

use tensorcrate::errors::Error;
use tensorcrate::numbers::{Complex, Dual};
use tensorcrate::persist::Storable;
use tensorcrate::tensors::{Matrix, Vector};

/// Writes a vector and reads it straight back at the same type.
#[track_caller]
fn revive_vector<T: Storable + PartialEq + std::fmt::Debug>(vector: &Vector<T>) -> Vector<T> {
    let mut bytes = Vec::new();
    vector.write_to(&mut bytes).expect("write");
    Vector::<T>::read_from(&bytes[..]).expect("read")
}

#[track_caller]
fn revive_matrix<T: Storable + PartialEq + std::fmt::Debug>(matrix: &Matrix<T>) -> Matrix<T> {
    let mut bytes = Vec::new();
    matrix.write_to(&mut bytes).expect("write");
    Matrix::<T>::read_from(&bytes[..]).expect("read")
}

/// The bytes of a valid 2-element `f64` vector, for corrupting.
fn sample_vector_bytes() -> Vec<u8> {
    let mut bytes = Vec::new();
    Vector::new([1.5f64, -2.5])
        .write_to(&mut bytes)
        .expect("write");
    bytes
}

#[test]
fn every_integer_width_round_trips() {
    assert_eq!(
        revive_vector(&Vector::new([i8::MIN, 0, i8::MAX])).data(),
        &[i8::MIN, 0, i8::MAX]
    );
    assert_eq!(
        revive_vector(&Vector::new([u8::MIN, 7, u8::MAX])).data(),
        &[u8::MIN, 7, u8::MAX]
    );
    assert_eq!(
        revive_vector(&Vector::new([i16::MIN, i16::MAX])).data(),
        &[i16::MIN, i16::MAX]
    );
    assert_eq!(
        revive_vector(&Vector::new([u16::MIN, u16::MAX])).data(),
        &[u16::MIN, u16::MAX]
    );
    assert_eq!(
        revive_vector(&Vector::new([i32::MIN, i32::MAX])).data(),
        &[i32::MIN, i32::MAX]
    );
    assert_eq!(
        revive_vector(&Vector::new([u32::MIN, u32::MAX])).data(),
        &[u32::MIN, u32::MAX]
    );
    assert_eq!(
        revive_vector(&Vector::new([i64::MIN, i64::MAX])).data(),
        &[i64::MIN, i64::MAX]
    );
    assert_eq!(
        revive_vector(&Vector::new([u64::MIN, u64::MAX])).data(),
        &[u64::MIN, u64::MAX]
    );
}

#[test]
fn floats_round_trip_bit_for_bit() {
    let original = Vector::new([f64::MIN, -0.0, 0.0, f64::MAX, f64::EPSILON]);
    assert_eq!(revive_vector(&original), original);

    // `-0.0 == 0.0`, so equality alone would not catch a sign flip.
    let signed_zero = revive_vector(&Vector::new([-0.0f64, 0.0]));
    assert!(signed_zero[0].is_sign_negative());
    assert!(signed_zero[1].is_sign_positive());

    // NaN compares unequal to itself, so check the bits directly. The payload
    // has to survive too, not just NaN-ness.
    let payload = f64::from_bits(0x7ff8_0000_dead_beef);
    let exotic = revive_vector(&Vector::new([
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        payload,
    ]));
    assert!(exotic[0].is_nan());
    assert_eq!(exotic[1], f64::INFINITY);
    assert_eq!(exotic[2], f64::NEG_INFINITY);
    assert_eq!(exotic[3].to_bits(), payload.to_bits());

    let small = revive_vector(&Vector::new([f32::MIN, f32::MAX, -0.0f32]));
    assert_eq!(small[0], f32::MIN);
    assert!(small[2].is_sign_negative());
}

#[test]
fn complex_and_dual_elements_round_trip_at_every_nesting() {
    let complex = Vector::new([Complex::new(1.5f64, -2.5), Complex::new(0.0, 4.25)]);
    assert_eq!(revive_vector(&complex), complex);

    let dual = Vector::new([Dual::new(1.0f32, 2.0), Dual::new(-3.0, 0.5)]);
    assert_eq!(revive_vector(&dual), dual);

    // Gaussian integers: the wrapper is generic over the leaf, not just floats.
    let gaussian = Vector::new([Complex::new(3i32, -4), Complex::new(0, 1)]);
    assert_eq!(revive_vector(&gaussian), gaussian);

    let complex_of_dual = Vector::new([Complex::new(Dual::new(1.0f64, 2.0), Dual::new(3.0, 4.0))]);
    assert_eq!(revive_vector(&complex_of_dual), complex_of_dual);

    let dual_of_complex =
        Vector::new([Dual::new(Complex::new(1.0f64, 2.0), Complex::new(3.0, 4.0))]);
    assert_eq!(revive_vector(&dual_of_complex), dual_of_complex);

    let nested_matrix = Matrix::from_rows([
        [Complex::new(Dual::new(1.0f64, 2.0), Dual::new(3.0, 4.0))],
        [Complex::new(Dual::new(5.0, 6.0), Dual::new(7.0, 8.0))],
    ]);
    assert_eq!(revive_matrix(&nested_matrix), nested_matrix);
}

#[test]
fn matrices_round_trip_in_row_major_order() {
    let m = Matrix::from_rows([[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let revived = revive_matrix(&m);
    assert_eq!(revived, m);
    assert_eq!(revived.shape(), (2, 3));
    // Order matters: a transposed read would still be 6 elements.
    assert_eq!(revived.row(0), [1.0, 2.0, 3.0]);
    assert_eq!(revived.row(1), [4.0, 5.0, 6.0]);
}

#[test]
fn degenerate_shapes_round_trip() {
    let empty = Vector::<f64>::new([]);
    assert_eq!(revive_vector(&empty), empty);

    let single = Matrix::<f64>::from_rows([[42.0]]);
    assert_eq!(revive_matrix(&single), single);

    let column = Matrix::<i32>::from_rows([[1], [2], [3]]);
    assert_eq!(revive_matrix(&column), column);
}

#[test]
fn the_stored_shape_is_the_shape_that_is_loaded() {
    // With dimensions held at runtime there is no shape to disagree with: the
    // header's extents *are* the loaded tensor's extents. This is the one place
    // the move to dynamic dimensions changed behaviour rather than spelling —
    // reading used to require knowing the size in advance, and a mismatch was
    // an `Error::Shape`.
    let bytes = sample_vector_bytes();
    let loaded = Vector::<f64>::read_from(&bytes[..]).expect("read");
    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded, Vector::new([1.5f64, -2.5]));

    let mut matrix_bytes = Vec::new();
    Matrix::from_rows([[1.0f64, 2.0], [3.0, 4.0]])
        .write_to(&mut matrix_bytes)
        .expect("write");
    let loaded = Matrix::<f64>::read_from(&matrix_bytes[..]).expect("read");
    assert_eq!(loaded.shape(), (2, 2));

    // A 1×4 and a 2×2 hold the same four values in the same order, so the
    // header is the only thing that tells them apart — and it does.
    let mut wide_bytes = Vec::new();
    Matrix::from_rows([[1.0f64, 2.0, 3.0, 4.0]])
        .write_to(&mut wide_bytes)
        .expect("write");
    assert_eq!(
        Matrix::<f64>::read_from(&wide_bytes[..])
            .expect("read")
            .shape(),
        (1, 4)
    );
}

#[test]
fn a_wrong_element_type_is_a_format_error() {
    let bytes = sample_vector_bytes();

    // Same byte count as the stored `f64`s, so nothing but the tag can catch it.
    match Vector::<i64>::read_from(&bytes[..]) {
        Err(Error::Format(msg)) => {
            assert!(
                msg.contains("f64") && msg.contains("i64"),
                "names both types: {msg}"
            );
        }
        other => panic!("expected a format error, got {other:?}"),
    }

    // Complex and Dual have identical layouts and identical sizes; only the tag
    // separates them.
    let mut complex_bytes = Vec::new();
    Vector::new([Complex::new(1.0f64, 2.0)])
        .write_to(&mut complex_bytes)
        .expect("write");
    match Vector::<Dual<f64>>::read_from(&complex_bytes[..]) {
        Err(Error::Format(msg)) => {
            assert!(
                msg.contains("Complex<f64>") && msg.contains("Dual<f64>"),
                "{msg}"
            );
        }
        other => panic!("expected a format error, got {other:?}"),
    }
}

#[test]
fn a_vector_and_a_matrix_are_not_interchangeable() {
    // An N-element vector and an N×1 matrix hold the same values in the same
    // order; the kind byte is what keeps them apart.
    let mut vector_bytes = Vec::new();
    Vector::new([1.0f64, 2.0])
        .write_to(&mut vector_bytes)
        .expect("write");
    match Matrix::<f64>::read_from(&vector_bytes[..]) {
        Err(Error::Format(msg)) => {
            assert!(msg.contains("matrix") && msg.contains("vector"), "{msg}")
        }
        other => panic!("expected a format error, got {other:?}"),
    }

    let mut matrix_bytes = Vec::new();
    Matrix::<f64>::from_rows([[1.0], [2.0]])
        .write_to(&mut matrix_bytes)
        .expect("write");
    assert!(matches!(
        Vector::<f64>::read_from(&matrix_bytes[..]),
        Err(Error::Format(_))
    ));
}

#[test]
fn foreign_and_future_data_is_rejected() {
    let mut wrong_magic = sample_vector_bytes();
    wrong_magic[0] = b'X';
    match Vector::<f64>::read_from(&wrong_magic[..]) {
        Err(Error::Format(msg)) => assert!(msg.contains("not a tensor file"), "{msg}"),
        other => panic!("expected a format error, got {other:?}"),
    }

    let mut future_version = sample_vector_bytes();
    future_version[4] = 99;
    match Vector::<f64>::read_from(&future_version[..]) {
        Err(Error::Format(msg)) => assert!(msg.contains("99"), "should name the version: {msg}"),
        other => panic!("expected a format error, got {other:?}"),
    }
}

#[test]
fn truncated_input_is_an_io_error() {
    let bytes = sample_vector_bytes();

    // Cut inside the header.
    assert!(matches!(
        Vector::<f64>::read_from(&bytes[..5]),
        Err(Error::Io(_))
    ));

    // Cut inside the element payload: the header validates, then the data runs
    // out mid-way.
    assert!(matches!(
        Vector::<f64>::read_from(&bytes[..bytes.len() - 4]),
        Err(Error::Io(_))
    ));

    assert!(matches!(
        Vector::<f64>::read_from(&[][..]),
        Err(Error::Io(_))
    ));
}

#[test]
fn trailing_bytes_are_ignored() {
    // A tensor may be one record inside a larger stream, so reading stops at the
    // end of the elements rather than demanding EOF.
    let mut bytes = sample_vector_bytes();
    bytes.extend_from_slice(b"and then some");
    assert_eq!(
        Vector::<f64>::read_from(&bytes[..]).expect("read"),
        Vector::new([1.5, -2.5])
    );
}

#[test]
fn tensors_can_be_concatenated_in_one_stream() {
    // Consequence of the format being self-delimiting: successive reads from the
    // same cursor pick up successive tensors.
    let mut bytes = Vec::new();
    Vector::new([1.0f64, 2.0])
        .write_to(&mut bytes)
        .expect("write");
    Matrix::from_rows([[3.0f64, 4.0]])
        .write_to(&mut bytes)
        .expect("write");

    let mut cursor = std::io::Cursor::new(bytes);
    assert_eq!(
        Vector::<f64>::read_from(&mut cursor).expect("first"),
        Vector::new([1.0, 2.0])
    );
    assert_eq!(
        Matrix::<f64>::read_from(&mut cursor).expect("second"),
        Matrix::from_rows([[3.0, 4.0]])
    );
}

#[test]
fn the_byte_layout_is_pinned() {
    // A round-trip test passes just as happily if both halves change together,
    // which would silently orphan every file already written. This spells the
    // format out. Changing it means bumping the version byte and deciding what
    // to do about existing data — not editing this expectation.
    let mut actual = Vec::new();
    Vector::new([1.5f64, -2.5])
        .write_to(&mut actual)
        .expect("write");

    let mut expected = Vec::new();
    expected.extend_from_slice(b"TCR1"); // magic
    expected.push(1); // format version
    expected.push(0); // kind: vector
    expected.push(1); // element tag length
    expected.push(0x11); // element tag: f64
    expected.extend_from_slice(&2u64.to_le_bytes()); // rows
    expected.extend_from_slice(&1u64.to_le_bytes()); // columns
    expected.extend_from_slice(&1.5f64.to_le_bytes());
    expected.extend_from_slice(&(-2.5f64).to_le_bytes());

    assert_eq!(actual, expected);
    assert_eq!(actual.len(), 24 + 2 * 8, "header is 23 + tag length bytes");

    // A matrix differs only in the kind byte and the extents.
    let mut matrix = Vec::new();
    Matrix::from_rows([[1.5f64, -2.5]])
        .write_to(&mut matrix)
        .expect("write");
    assert_eq!(matrix[5], 1, "kind: matrix");
    assert_eq!(matrix[8..16], 1u64.to_le_bytes(), "one row");
    assert_eq!(matrix[16..24], 2u64.to_le_bytes(), "two columns");
    assert_eq!(matrix[24..], actual[24..], "same payload as the vector");
}

#[test]
fn save_and_load_use_the_filesystem() {
    let path = std::env::temp_dir().join(format!(
        "tensorcrate-persist-{}-{:?}.tcr",
        std::process::id(),
        std::thread::current().id()
    ));

    let original = Matrix::from_rows([[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    original.save(&path).expect("save");
    let loaded = Matrix::<f64>::load(&path).expect("load");
    assert_eq!(loaded, original);

    // The shape comes back from the file over the path API too, so a second
    // load with no shape named up front still reconstructs the original.
    assert_eq!(Matrix::<f64>::load(&path).expect("load").shape(), (2, 3));

    std::fs::remove_file(&path).expect("cleanup");

    assert!(matches!(Matrix::<f64>::load(&path), Err(Error::Io(_))));
}
