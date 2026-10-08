//! N-dimensional tensors and views: layout operations against a naive
//! index-arithmetic reference, elementwise operations over non-contiguous
//! views, conversions and persistence — on `Host` for every float type, and on
//! `Metal` against `Host`.

use tensorcrate::numbers::{bf16, f16};
use tensorcrate::tensors::fused::{Builder, DType, Decl};
use tensorcrate::tensors::{
    Analytic, BinaryOp, Compare, Host, Kernels, MAX_RANK, Matrix, Tensor, TensorView, Vector,
};

/// Deterministic, integral, mixed-sign filler, so every backend's arithmetic
/// on it is exact.
fn value(index: usize) -> f32 {
    ((index * 7 + 3) % 23) as f32 - 11.0
}

fn tensor(shape: &[usize]) -> Tensor<f32> {
    let len = shape.iter().product::<usize>();
    Tensor::from_vec(shape, (0..len).map(value).collect::<Vec<_>>())
}

/// Every index of `shape`, in row-major order.
fn indices(shape: &[usize]) -> Vec<Vec<usize>> {
    let len = shape.iter().product::<usize>();
    (0..len)
        .map(|mut flat| {
            let mut index = vec![0; shape.len()];
            for axis in (0..shape.len()).rev() {
                index[axis] = flat % shape[axis];
                flat /= shape[axis];
            }
            index
        })
        .collect()
}

/// The elements of `view`, read one index at a time — the reference every
/// copy is checked against.
fn naive<T: Copy + 'static>(view: TensorView<'_, T, Host>) -> Vec<T> {
    indices(view.shape())
        .iter()
        .map(|index| view.get(index).unwrap())
        .collect()
}

#[test]
fn a_tensor_knows_its_shape() {
    let t = tensor(&[2, 3, 4]);
    assert_eq!(t.shape(), [2, 3, 4]);
    assert_eq!(t.rank(), 3);
    assert_eq!(t.len(), 24);
    assert_eq!(t.dim(-1), 4);
    assert_eq!(t.dim(0usize), 2);
    assert_eq!(t.strides(), [12, 4, 1]);
    assert_eq!(t[[1, 2, 3]], value(23));
    assert_eq!(t.get(&[1, 2, 3]), Some(value(23)));
    assert_eq!(t.get(&[2, 0, 0]), None);
    assert_eq!(t.get(&[0, 0]), None);

    let scalar = Tensor::from_vec(&[], vec![5.0f32]);
    assert_eq!(scalar.rank(), 0);
    assert_eq!(scalar.len(), 1);
    assert_eq!(scalar[[]], 5.0);

    let empty = Tensor::<f32>::zeros(&[3, 0, 2]);
    assert!(empty.is_empty());
    assert_eq!(empty.len(), 0);

    let mut ones = Tensor::<f64>::ones(&[2, 2]);
    ones[[0, 1]] = 4.0;
    assert_eq!(ones.data(), [1.0, 4.0, 1.0, 1.0]);
}

#[test]
#[should_panic(expected = "from_vec: 5 elements cannot fill shape [2, 3]")]
fn a_shape_must_fit_its_elements() {
    Tensor::from_vec(&[2, 3], vec![0.0f32; 5]);
}

#[test]
#[should_panic(expected = "more than the 6 a tensor may have")]
fn rank_is_bounded() {
    Tensor::<f32>::zeros(&[1; MAX_RANK + 1]);
}

#[test]
fn reshape_and_conversions_move_the_storage() {
    let t = tensor(&[2, 3, 4]);
    let address = t.data().as_ptr();
    let reshaped = t.reshape(&[6, 4]);
    assert_eq!(reshaped.data().as_ptr(), address);
    let matrix: Matrix<f32> = reshaped.into_matrix();
    assert_eq!(matrix.shape(), (6, 4));
    assert_eq!(matrix.data().as_ptr(), address);
    let back = Tensor::from(matrix).unsqueeze(0).squeeze(0);
    assert_eq!(back.shape(), [6, 4]);
    let vector: Vector<f32> = back.into_vector();
    assert_eq!(vector.data().as_ptr(), address);
    let again = Tensor::from(vector);
    assert_eq!(again.shape(), [24]);
    assert_eq!(again.data().as_ptr(), address);
    assert_eq!(again.unsqueeze(1).shape(), [24, 1]);
}

#[test]
#[should_panic(expected = "into_matrix: shape [2, 3, 4] has 3 axes, not 2")]
fn only_two_axes_make_a_matrix() {
    tensor(&[2, 3, 4]).into_matrix();
}

#[test]
fn permutations_read_through_strides_and_copy_exactly() {
    let t = tensor(&[2, 3, 4, 5]);
    let p = t.permute(&[2, 0, 3, 1]);
    assert_eq!(p.shape(), [4, 2, 5, 3]);
    assert_eq!(p.strides(), [5, 60, 1, 20]);
    assert!(!p.is_contiguous());
    for index in indices(p.shape()) {
        let source = [index[1], index[3], index[0], index[2]];
        assert_eq!(p.get(&index), t.get(&source));
        assert_eq!(p[[index[0], index[1], index[2], index[3]]], t[source]);
    }
    let copied = p.contiguous();
    assert_eq!(copied.shape(), p.shape());
    assert_eq!(copied.to_vec(), naive(p));
    assert_eq!(p.to_vec(), naive(p));

    let t2 = t.transpose(1, -1);
    assert_eq!(t2.shape(), [2, 5, 4, 3]);
    assert_eq!(t2.contiguous().to_vec(), naive(t2));
    // Transposing back is the identity.
    assert_eq!(t2.transpose(1, 3).contiguous(), t);
}

#[test]
#[should_panic(expected = "permute: [0, 0, 1] is not an order of the 3 axes")]
fn a_permutation_names_every_axis_once() {
    tensor(&[2, 3, 4]).permute(&[0, 0, 1]);
}

#[test]
fn slices_selections_and_splits_are_views() {
    let t = tensor(&[4, 5, 6]);
    let narrow = t.narrow(1, 1, 3);
    assert_eq!(narrow.shape(), [4, 3, 6]);
    assert_eq!(narrow.offset(), 6);
    assert_eq!(narrow.contiguous().to_vec(), naive(narrow));
    assert_eq!(
        t.slice(-1, 2..).contiguous().to_vec(),
        naive(t.slice(2, 2..6))
    );
    assert_eq!(t.slice(0, ..=1).shape(), [2, 5, 6]);

    let row = t.select(0, 2).select(1, 4);
    assert_eq!(row.shape(), [5]);
    for i in 0..5 {
        assert_eq!(row.get(&[i]), t.get(&[2, i, 4]));
    }

    let pieces = t.split(2, &[1, 2, 3]);
    assert_eq!(
        pieces.iter().map(|p| p.shape()[2]).collect::<Vec<_>>(),
        [1, 2, 3]
    );
    assert_eq!(Tensor::concat(pieces.iter().copied(), 2), t);
    let thirds = t.chunk(-1, 3);
    assert_eq!(thirds.len(), 3);
    assert_eq!(thirds[1].get(&[0, 0, 0]), t.get(&[0, 0, 2]));

    // An empty slice is still a view, of nothing.
    let empty = t.narrow(1, 5, 0);
    assert!(empty.is_empty());
    assert!(empty.contiguous().is_empty());
    assert_eq!(empty.contiguous().shape(), [4, 0, 6]);
}

#[test]
#[should_panic(expected = "narrow: 3..6 reaches past axis 1's extent of 5")]
fn narrowing_checks_the_extent() {
    tensor(&[4, 5]).narrow(1, 3, 3);
}

#[test]
#[should_panic(expected = "chunk: axis 1 of shape [2, 5] does not divide into 3 equal parts")]
fn chunks_are_equal() {
    tensor(&[2, 5]).chunk(1, 3);
}

#[test]
fn reshaping_a_view_works_in_place_when_the_layout_allows() {
    let t = tensor(&[2, 3, 12]);
    let third = t.chunk(2, 3)[1];
    // Splitting the last axis of a column slice needs no copy.
    let heads = third.try_reshape(&[2, 3, 2, 2]).unwrap();
    assert_eq!(heads.strides(), [36, 12, 2, 1]);
    assert_eq!(heads.contiguous().reshape(&[2, 3, 4]), third.contiguous());
    // Merging axes a permutation separated does.
    assert!(t.permute(&[1, 0, 2]).try_reshape(&[6, 12]).is_none());
    // A contiguous view always reshapes.
    assert_eq!(
        t.view().try_reshape(&[6, 12]).unwrap().contiguous(),
        t.clone().reshape(&[6, 12])
    );
}

#[test]
fn squeeze_and_unsqueeze_add_and_remove_unit_axes() {
    let t = tensor(&[3, 1, 4]);
    let squeezed = t.view().squeeze(1);
    assert_eq!(squeezed.shape(), [3, 4]);
    assert!(squeezed.is_contiguous());
    let unsqueezed = squeezed.unsqueeze(2).unsqueeze(0);
    assert_eq!(unsqueezed.shape(), [1, 3, 4, 1]);
    assert!(unsqueezed.is_contiguous());
    assert_eq!(unsqueezed.contiguous().reshape(&[3, 1, 4]), t);
    let transposed = t.transpose(0, 2).squeeze(1);
    assert_eq!(transposed.shape(), [4, 3]);
    assert_eq!(transposed.contiguous().to_vec(), naive(transposed));
}

#[test]
#[should_panic(expected = "squeeze: axis 0 of shape [3, 1, 4] has 3 elements, not 1")]
fn only_unit_axes_squeeze() {
    tensor(&[3, 1, 4]).squeeze(0);
}

#[test]
fn concat_and_stack_along_every_axis() {
    let shape = [2, 3, 4];
    for axis in 0..3 {
        let mut other_shape = shape;
        other_shape[axis] = 2;
        let a = tensor(&shape);
        let b = tensor(&other_shape).scale(10.0);
        // A non-contiguous piece: `b` stored transposed and read back.
        let stored = b.transpose(0, 2).contiguous();
        let b_view = stored.transpose(0, 2);
        let joined = Tensor::concat([a.view(), b_view], axis);
        let mut joined_shape = shape;
        joined_shape[axis] += 2;
        assert_eq!(joined.shape(), joined_shape);
        for index in indices(&joined_shape) {
            let expected = if index[axis] < shape[axis] {
                a.get(&index)
            } else {
                let mut inner = index.clone();
                inner[axis] -= shape[axis];
                b.get(&inner)
            };
            assert_eq!(joined.get(&index), expected, "axis {axis}, index {index:?}");
        }
    }
    for axis in 0..=3 {
        let (a, b) = (tensor(&shape), tensor(&shape).scale(-1.0));
        let stacked = Tensor::stack([&a, &b], axis);
        assert_eq!(stacked.rank(), 4);
        assert_eq!(stacked.dim(axis), 2);
        assert_eq!(stacked.select(axis, 0).contiguous(), a);
        assert_eq!(stacked.select(axis, 1).contiguous(), b);
    }
    // Empty pieces contribute nothing.
    let a = tensor(&[2, 3]);
    let none = Tensor::<f32>::zeros(&[0, 3]);
    assert_eq!(Tensor::concat([&none, &a, &none], 0), a);
}

#[test]
#[should_panic(expected = "concat: shapes [2, 3] and [2, 4] differ outside axis 0")]
fn concat_checks_the_other_extents() {
    Tensor::concat([&tensor(&[2, 3]), &tensor(&[2, 4])], 0);
}

#[test]
fn writing_a_slice_overwrites_only_that_slice() {
    let mut cache = Tensor::<f32>::zeros(&[2, 2, 5, 3]);
    let step = tensor(&[2, 3, 2]); // [batch, features, heads], stored oddly
    let source = step.permute(&[0, 2, 1]).unsqueeze(2); // [2, 2, 1, 3]
    cache.write_slice(2, 3, source);
    for index in indices(cache.shape()) {
        let expected = if index[2] == 3 {
            step.get(&[index[0], index[3], index[1]]).unwrap()
        } else {
            0.0
        };
        assert_eq!(cache.get(&index), Some(expected), "{index:?}");
    }
}

#[test]
#[should_panic(
    expected = "write_slice: a [2, 2] source does not fit shape [2, 5] from 4 along axis 1"
)]
fn a_written_slice_must_fit() {
    let mut target = Tensor::<f32>::zeros(&[2, 5]);
    target.write_slice(1, 4, &tensor(&[2, 2]));
}

/// The attention layout change: `[B, T, 3·D]` split into query, key and value,
/// each `[B, H, T, Dh]`, and merged back — exactly, with no arithmetic.
fn split_heads<T: Copy + 'static, B: tensorcrate::tensors::Backend>(
    qkv: &Tensor<T, B>,
    heads: usize,
) -> Vec<Tensor<T, B>> {
    let [batch, time, width] = qkv.shape() else {
        panic!("not [B, T, 3D]")
    };
    let (batch, time, model) = (*batch, *time, width / 3);
    qkv.chunk(-1, 3)
        .into_iter()
        .map(|part| {
            part.try_reshape(&[batch, time, heads, model / heads])
                .expect("splitting the last axis needs no copy")
                .permute(&[0, 2, 1, 3])
                .contiguous()
        })
        .collect()
}

fn merge_heads<T: Copy + 'static, B: tensorcrate::tensors::Backend>(
    parts: &[Tensor<T, B>],
) -> Tensor<T, B> {
    let merged = parts
        .iter()
        .map(|part| {
            let [batch, heads, time, depth] = part.shape() else {
                panic!("not [B, H, T, Dh]")
            };
            let shape = [*batch, *time, heads * depth];
            part.transpose(1, 2).contiguous().reshape(&shape)
        })
        .collect::<Vec<_>>();
    Tensor::concat(&merged, -1)
}

#[test]
fn attention_heads_round_trip_exactly() {
    let (batch, time, model, heads) = (2, 5, 12, 3);
    let qkv = tensor(&[batch, time, 3 * model]);
    let parts = split_heads(&qkv, heads);
    for (k, part) in parts.iter().enumerate() {
        assert_eq!(part.shape(), [batch, heads, time, model / heads]);
        for index in indices(part.shape()) {
            let [b, h, t, d] = index[..] else {
                unreachable!()
            };
            assert_eq!(
                part.get(&index),
                qkv.get(&[b, t, k * model + h * (model / heads) + d])
            );
        }
    }
    assert_eq!(merge_heads(&parts), qkv);
}

#[test]
fn elementwise_operations_read_non_contiguous_views() {
    let a = tensor(&[3, 4, 5]);
    // No zeros, so `%` and `/` stay finite.
    let b = tensor(&[5, 4, 3]).with_scalar(0.25, BinaryOp::Add, false);
    let a_t = a.transpose(0, 2); // [5, 4, 3], strided
    let b_slice = Tensor::concat([&b, &b], 1);
    let b_view = b_slice.narrow(1, 4, 4); // [5, 4, 3], offset
    let reference = |f: fn(f32, f32) -> f32| {
        naive(a_t)
            .into_iter()
            .zip(naive(b_view))
            .map(|(x, y)| f(x, y))
            .collect::<Vec<_>>()
    };
    assert_eq!((a_t + b_view).to_vec(), reference(|x, y| x + y));
    assert_eq!((a_t - b_view).to_vec(), reference(|x, y| x - y));
    assert_eq!((a_t * b_view).to_vec(), reference(|x, y| x * y));
    assert_eq!((a_t / b_view).to_vec(), reference(|x, y| x / y));
    assert_eq!(
        a_t.elementwise(b_view, BinaryOp::Rem).to_vec(),
        reference(|x, y| x % y)
    );
    assert_eq!(a_t.max(b_view).to_vec(), reference(f32::max));
    assert_eq!(a_t.min(b_view).to_vec(), reference(f32::min));
    assert_eq!(
        a_t.compare(b_view, Compare::Less).to_vec(),
        reference(|x, y| if x < y { 1.0 } else { 0.0 })
    );
    let sum = &a_t.contiguous() + b_view;
    assert_eq!(sum.shape(), [5, 4, 3]);
    assert_eq!(sum.to_vec(), reference(|x, y| x + y));

    let unary = a_t.unary(Analytic::Exp);
    for (got, x) in unary.to_vec().into_iter().zip(naive(a_t)) {
        assert!((got - x.exp()).abs() <= 1e-6 * x.exp().max(1.0));
    }
    assert_eq!(
        a_t.clamp(-2.0, 3.0).to_vec(),
        naive(a_t)
            .iter()
            .map(|x| x.clamp(-2.0, 3.0))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        a_t.with_scalar(1.0, BinaryOp::Sub, true).to_vec(),
        naive(a_t).iter().map(|x| 1.0 - x).collect::<Vec<_>>()
    );
    assert_eq!(
        (-a_t).to_vec(),
        naive(a_t).iter().map(|x| -x).collect::<Vec<_>>()
    );
    assert_eq!(
        a_t.max_scalar(0.0).to_vec(),
        naive(a_t).iter().map(|x| x.max(0.0)).collect::<Vec<_>>()
    );
    assert_eq!(
        a_t.power_scalar(2.0, false).to_vec(),
        naive(a_t).iter().map(|x| x * x).collect::<Vec<_>>()
    );
}

#[test]
#[should_panic(expected = "add: tensor shapes differ, [3, 4] and [4, 3]")]
fn elementwise_shapes_must_match() {
    let _ = &tensor(&[3, 4]) + &tensor(&[4, 3]);
}

/// Elementwise operations on views are generic over the element type.
fn check_generic<T: tensorcrate::numbers::Real>()
where
    Host: Kernels<T>,
{
    let a = Tensor::<T>::from_vec(
        &[2, 3, 2],
        (0..12)
            .map(|i| T::from_f64(value(i) as f64))
            .collect::<Vec<_>>(),
    );
    let p = a.permute(&[2, 0, 1]);
    let doubled = &p.contiguous() + p;
    let expected = naive(p).into_iter().map(|x| x + x).collect::<Vec<_>>();
    assert_eq!(doubled.to_vec(), expected);
}

#[test]
fn every_float_type_runs_the_same_operations() {
    check_generic::<f32>();
    check_generic::<f64>();
    check_generic::<f16>();
    check_generic::<bf16>();
}

#[test]
fn integer_tensors_have_the_layout_operations() {
    let ids = Tensor::from_vec(&[2, 3], vec![1u32, 2, 3, 4, 5, 6]);
    assert_eq!(
        ids.transpose(0, 1).contiguous().into_vec(),
        [1, 4, 2, 5, 3, 6]
    );
    let joined = Tensor::concat([ids.view(), ids.slice(1, 1..)], 1);
    assert_eq!(joined.into_vec(), [1, 2, 3, 2, 3, 4, 5, 6, 5, 6]);
}

#[test]
fn fused_programs_read_tensors_and_strided_views() {
    let mut builder = Builder::<f32>::new();
    let x = builder.input(Decl::tensor(DType::F32, &[2, 3, 4]));
    let y = builder.input(Decl::tensor(DType::F32, &[2, 3, 4]));
    let product = builder.mul(x, y);
    builder.output(product, DType::F32);
    let program = builder.build().unwrap();

    let a = tensor(&[2, 3, 4]);
    let wide = tensor(&[2, 3, 8]);
    let half = wide.narrow(2, 4, 4); // rows step 8, columns step 1
    let out = program
        .run(&[&a, &half], &mut [])
        .remove(0)
        .into_matrix::<f32>();
    let expected = naive(a.view())
        .into_iter()
        .zip(naive(half))
        .map(|(x, y)| x * y)
        .collect::<Vec<_>>();
    assert_eq!(out.data(), expected);

    // A column broadcast from a strided view of one column, of its own shape.
    let mut builder = Builder::<f32>::new();
    let x = builder.input(Decl::tensor(DType::F32, &[2, 3, 4]));
    let column = builder.input(Decl::tensor(DType::F32, &[2, 3, 1]));
    let sum = builder.add(x, column);
    builder.output(sum, DType::F32);
    let program = builder.build().unwrap();
    let firsts = wide.narrow(-1, 0, 1);
    let out = program
        .run(&[&a, &firsts], &mut [])
        .remove(0)
        .into_matrix::<f32>();
    for (row, values) in out.row_iter().enumerate() {
        for (col, &got) in values.iter().enumerate() {
            let (b, t) = (row / 3, row % 3);
            let expected = a.get(&[b, t, col]).unwrap() + wide.get(&[b, t, 0]).unwrap();
            assert_eq!(got, expected);
        }
    }
}

#[test]
fn a_view_that_moves_the_last_axis_is_read_in_place() {
    let mut builder = Builder::<f32>::new();
    let x = builder.input(Decl::tensor(DType::F32, &[4, 2, 3]));
    let y = builder.shift(x, 1.0);
    builder.output(y, DType::F32);
    let program = builder.build().unwrap();
    let a = tensor(&[2, 3, 4]);
    let moved = a.permute(&[2, 0, 1]);
    let out = program
        .run(&[&moved], &mut [])
        .remove(0)
        .into_tensor::<f32>();
    let expected: Vec<f32> = naive(moved).into_iter().map(|v| v + 1.0).collect();
    assert_eq!(out.shape(), [4, 2, 3]);
    assert_eq!(out.into_vec(), expected);
}

#[test]
#[should_panic(expected = "cannot be read as [8, 3]")]
fn a_view_that_cannot_be_reshaped_in_place_is_refused() {
    // Its axes are out of order in storage, so no strides read it as 8 × 3.
    let mut builder = Builder::<f32>::new();
    let x = builder.input(Decl::matrix(DType::F32, (8, 3)));
    builder.output(x, DType::F32);
    let program = builder.build().unwrap();
    let a = tensor(&[2, 3, 4]);
    let moved = a.permute(&[2, 0, 1]);
    program.run(&[&moved], &mut []);
}

#[test]
fn tensors_save_and_load_with_their_shape() {
    let t = tensor(&[2, 1, 3, 4]);
    let mut bytes = Vec::new();
    t.write_to(&mut bytes).unwrap();
    let loaded = Tensor::<f32>::read_from(&bytes[..]).unwrap();
    assert_eq!(loaded, t);
    assert!(Tensor::<f64>::read_from(&bytes[..]).is_err());
    // A tensor file is not a matrix file.
    assert!(Matrix::<f32>::read_from(&bytes[..]).is_err());

    // But a matrix or vector file reads as a tensor.
    let m = Matrix::from_rows([[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    let mut bytes = Vec::new();
    m.write_to(&mut bytes).unwrap();
    assert_eq!(
        Tensor::<f32>::read_from(&bytes[..]).unwrap(),
        Tensor::from(m)
    );
    let v = Vector::new([1.0f64, 2.0]);
    let mut bytes = Vec::new();
    v.write_to(&mut bytes).unwrap();
    assert_eq!(
        Tensor::<f64>::read_from(&bytes[..]).unwrap(),
        Tensor::from(v)
    );

    // Scalars and empty tensors survive too.
    for t in [
        Tensor::from_vec(&[], vec![bf16::from_f32(1.5)]),
        Tensor::<bf16>::zeros(&[3, 0]),
    ] {
        let mut bytes = Vec::new();
        t.write_to(&mut bytes).unwrap();
        assert_eq!(Tensor::<bf16>::read_from(&bytes[..]).unwrap(), t);
    }

    let path = std::env::temp_dir().join(format!("nd_tensor_{}.tensor", std::process::id()));
    t.save(&path).unwrap();
    assert_eq!(Tensor::<f32>::load(&path).unwrap(), t);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn display_nests_one_bracket_per_axis() {
    let t = Tensor::from_vec(&[2, 2, 2], (1..=8).collect::<Vec<i32>>());
    assert_eq!(
        t.to_string(),
        "[[[ 1 2 ]\n  [ 3 4 ]]\n [[ 5 6 ]\n  [ 7 8 ]]]"
    );
    assert_eq!(Tensor::from_vec(&[], vec![3]).to_string(), "3");
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    use super::*;
    use tensorcrate::metal::MetalElement;
    use tensorcrate::tensors::Metal;

    fn on_both<T: MetalElement>(shape: &[usize]) -> (Tensor<T>, Tensor<T, Metal>) {
        let len = shape.iter().product::<usize>();
        let host = Tensor::from_vec(
            shape,
            (0..len)
                .map(|i| T::from_f64(value(i) as f64))
                .collect::<Vec<_>>(),
        );
        let device = host.to_backend::<Metal>();
        (host, device)
    }

    fn layouts_agree<T: MetalElement>() {
        let (host, device) = on_both::<T>(&[3, 4, 5, 2]);
        let resident = device.as_vector().is_device_resident();
        for axes in [[0, 1, 2, 3], [3, 2, 1, 0], [1, 3, 0, 2], [0, 2, 1, 3]] {
            let h = host.permute(&axes).contiguous();
            let d = device.permute(&axes).contiguous();
            assert_eq!(d.to_backend::<Host>(), h, "{} permute {axes:?}", T::SUFFIX);
            assert_eq!(d.as_vector().is_device_resident(), resident);
        }
        let h = host.narrow(1, 1, 2).select(2, 3).contiguous();
        let d = device.narrow(1, 1, 2).select(2, 3).contiguous();
        assert_eq!(d.to_backend::<Host>(), h);
        for axis in 0..4 {
            let h = Tensor::concat([host.view(), host.narrow(axis, 1, 1)], axis);
            let d = Tensor::concat([device.view(), device.narrow(axis, 1, 1)], axis);
            assert_eq!(d.to_backend::<Host>(), h, "{} concat {axis}", T::SUFFIX);
            let h = Tensor::stack([host.view(), host.view()], axis);
            let d = Tensor::stack([device.view(), device.view()], axis);
            assert_eq!(d.to_backend::<Host>(), h, "{} stack {axis}", T::SUFFIX);
        }
        let (mut h, mut d) = on_both::<T>(&[3, 6, 2]);
        h.write_slice(1, 2, host.select(2, 1).narrow(1, 0, 3));
        d.write_slice(1, 2, device.select(2, 1).narrow(1, 0, 3));
        assert_eq!(d.to_backend::<Host>(), h);
    }

    #[test]
    fn layout_operations_agree_with_host_for_every_metal_type() {
        layouts_agree::<f32>();
        layouts_agree::<f16>();
        layouts_agree::<bf16>();
    }

    fn arithmetic_agrees<T: MetalElement>()
    where
        Metal: Kernels<T>,
    {
        let (a_host, a_device) = on_both::<T>(&[4, 3, 5]);
        let (b_host, b_device) = on_both::<T>(&[5, 3, 4]);
        let (ah, ad) = (a_host.transpose(0, 2), a_device.transpose(0, 2));
        let (bh, bd) = (b_host.view(), b_device.view());
        for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul] {
            let h = ah.elementwise(bh, op);
            let d = ad.elementwise(bd, op);
            assert!(
                d.as_vector().is_device_resident() || !b_device.as_vector().is_device_resident()
            );
            assert_eq!(d.to_backend::<Host>(), h, "{} {op:?}", T::SUFFIX);
        }
        for op in Compare::ALL {
            assert_eq!(
                ad.compare(bd, op).to_backend::<Host>(),
                ah.compare(bh, op),
                "{} {op:?}",
                T::SUFFIX
            );
        }
        assert_eq!(
            ad.clamp(T::from_f64(-2.0), T::from_f64(3.0))
                .to_backend::<Host>(),
            ah.clamp(T::from_f64(-2.0), T::from_f64(3.0))
        );
        assert_eq!((-ad).to_backend::<Host>(), -ah);
        let unary_host = ah.unary(Analytic::Tanh).to_vec();
        let unary_device = ad.unary(Analytic::Tanh).to_backend::<Host>().to_vec();
        for (d, h) in unary_device.iter().zip(&unary_host) {
            let (d, h) = (d.into_f64(), h.into_f64());
            assert!(
                (d - h).abs() <= 1e-2 * h.abs().max(1e-2),
                "{} tanh {d} vs {h}",
                T::SUFFIX
            );
        }
    }

    #[test]
    fn elementwise_operations_agree_with_host_for_every_metal_type() {
        arithmetic_agrees::<f32>();
        arithmetic_agrees::<f16>();
        arithmetic_agrees::<bf16>();
    }

    fn heads_round_trip<T: MetalElement>() {
        let (host, device) = on_both::<T>(&[2, 7, 3 * 16]);
        let parts_host = split_heads(&host, 4);
        let parts_device = split_heads(&device, 4);
        for (d, h) in parts_device.iter().zip(&parts_host) {
            assert_eq!(d.shape(), [2, 4, 7, 4]);
            assert_eq!(&d.to_backend::<Host>(), h);
        }
        let merged = merge_heads(&parts_device);
        assert_eq!(merged.to_backend::<Host>(), host);
        assert_eq!(
            merged.as_vector().is_device_resident(),
            device.as_vector().is_device_resident()
        );
    }

    #[test]
    fn attention_heads_round_trip_exactly_on_the_device() {
        heads_round_trip::<f32>();
        heads_round_trip::<f16>();
        heads_round_trip::<bf16>();
    }

    #[test]
    fn every_element_width_copies_on_the_device() {
        fn check<T: Copy + PartialEq + std::fmt::Debug + 'static>(make: impl Fn(usize) -> T) {
            let host = Tensor::from_vec(&[3, 4, 2], (0..24).map(&make).collect::<Vec<_>>());
            let device = host.to_backend::<Metal>();
            let axes = [2, 0, 1];
            assert_eq!(
                device.permute(&axes).contiguous().to_backend::<Host>(),
                host.permute(&axes).contiguous()
            );
            assert_eq!(
                Tensor::concat([device.view(), device.slice(1, 1..3)], 1).to_backend::<Host>(),
                Tensor::concat([host.view(), host.slice(1, 1..3)], 1)
            );
        }
        check(|i| i as u8);
        check(|i| i as u16);
        check(|i| i as u32);
        check(|i| i as f64 * 0.5);
        check(|i| (i as u64, i as u64 * 3)); // 16 bytes
        let host = Tensor::from_vec(
            &[3, 4, 2],
            (0..24).map(|i| [i as u8; 3]).collect::<Vec<_>>(),
        );
        let device = host.to_backend::<Metal>();
        let unsupported = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            device.permute(&[2, 0, 1]).contiguous()
        }));
        assert!(
            unsupported.is_err(),
            "an unsupported element width must not fall back to Host"
        );
    }
}
