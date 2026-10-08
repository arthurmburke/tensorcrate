//! Broadcasting binary operations and axis reductions on N-dimensional
//! tensors: against naive index-arithmetic references on `Host` for every
//! float type, over non-contiguous views, and on `Metal` against `Host`.

use tensorcrate::numbers::{Real, bf16, f16};
use tensorcrate::statistics::Correction;
use tensorcrate::tensors::fused::{Builder, DType, Decl};
use tensorcrate::tensors::{BinaryOp, Compare, Host, Kernels, Tensor, TensorView, Vector};

/// Deterministic, integral, mixed-sign filler, so every backend's sums of it
/// are exact.
fn value(index: usize) -> f64 {
    ((index * 7 + 3) % 23) as f64 - 11.0
}

fn tensor<T: Real>(shape: &[usize]) -> Tensor<T> {
    let len = shape.iter().product::<usize>();
    Tensor::from_vec(
        shape,
        (0..len).map(|i| T::from_f64(value(i))).collect::<Vec<_>>(),
    )
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

/// The element of `view` that broadcast index `index` of a larger shape reads:
/// aligned at the last axis, an axis of one read at zero.
fn broadcast_get<T: Copy + 'static>(view: TensorView<'_, T, Host>, index: &[usize]) -> T {
    let skip = index.len() - view.rank();
    let own = view
        .shape()
        .iter()
        .zip(&index[skip..])
        .map(|(&extent, &i)| if extent == 1 { 0 } else { i })
        .collect::<Vec<_>>();
    view.get(&own).unwrap()
}

/// `f(a, b)` over the broadcast shape, element by element.
fn naive_broadcast<T: Real>(
    a: TensorView<'_, T, Host>,
    b: TensorView<'_, T, Host>,
    shape: &[usize],
    f: impl Fn(T, T) -> T,
) -> Vec<T> {
    indices(shape)
        .iter()
        .map(|index| f(broadcast_get(a, index), broadcast_get(b, index)))
        .collect()
}

/// Equal element for element, a NaN matching a NaN.
fn assert_same<T: Real>(got: &[T], want: &[T], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            g == w || (g.is_nan() && w.is_nan()),
            "{what}[{k}]: {g} vs {w}"
        );
    }
}

// ---- broadcasting ------------------------------------------------------------

/// A scalar operation the reference applies.
type Reference = fn(f32, f32) -> f32;

/// Every broadcasting binary operation of `a` and `b`, against the reference.
fn check_broadcast(a: TensorView<'_, f32, Host>, b: TensorView<'_, f32, Host>, shape: &[usize]) {
    let cases: [(BinaryOp, Reference); 5] = [
        (BinaryOp::Add, |x, y| x + y),
        (BinaryOp::Sub, |x, y| x - y),
        (BinaryOp::Mul, |x, y| x * y),
        (BinaryOp::Div, |x, y| x / y),
        (BinaryOp::Rem, |x, y| x % y),
    ];
    for (op, f) in cases {
        let got = a.elementwise(b, op);
        assert_eq!(got.shape(), shape, "{op:?}");
        assert_same(
            &got.to_vec(),
            &naive_broadcast(a, b, shape, f),
            &format!("{op:?} {shape:?}"),
        );
    }
    for op in Compare::ALL {
        assert_eq!(
            a.compare(b, op).to_vec(),
            naive_broadcast(a, b, shape, |x, y| op.value(x, y)),
            "{op:?}"
        );
    }
    assert_eq!(a.min(b).to_vec(), naive_broadcast(a, b, shape, f32::min));
    assert_eq!(a.max(b).to_vec(), naive_broadcast(a, b, shape, f32::max));
    let base = a
        .with_scalar(0.0, BinaryOp::Add, false)
        .unary(tensorcrate::tensors::Analytic::Exp);
    assert_eq!(
        base.power(b).to_vec(),
        naive_broadcast(base.view(), b, shape, |x, y| x.powf(y))
    );
    assert_eq!((a + b).to_vec(), naive_broadcast(a, b, shape, |x, y| x + y));
    assert_eq!((a * b).to_vec(), naive_broadcast(a, b, shape, |x, y| x * y));
}

#[test]
fn shapes_broadcast_from_the_last_axis() {
    let x = tensor::<f32>(&[2, 3, 4]);
    // A bias over the features, on either side.
    let bias = tensor::<f32>(&[4]).with_scalar(0.5, BinaryOp::Add, false);
    check_broadcast(x.view(), bias.view(), &[2, 3, 4]);
    check_broadcast(bias.view(), x.view(), &[2, 3, 4]);
    // Size-one axes on both sides at once.
    let column = tensor::<f32>(&[2, 3, 1]).with_scalar(0.25, BinaryOp::Add, false);
    let row = tensor::<f32>(&[1, 4]).with_scalar(0.25, BinaryOp::Add, false);
    check_broadcast(column.view(), row.view(), &[2, 3, 4]);
    // A rank-0 tensor is a scalar against anything.
    let scalar = Tensor::from_vec(&[], vec![2.5f32]);
    check_broadcast(x.view(), scalar.view(), &[2, 3, 4]);
    check_broadcast(scalar.view(), scalar.view(), &[]);
}

#[test]
fn attention_shaped_operands_broadcast() {
    let (batch, heads, time, dim) = (2, 3, 5, 4);
    // A [T, T] causal mask over [B, H, T, T] scores.
    let scores = tensor::<f32>(&[batch, heads, time, time]);
    let mask = Tensor::from_vec(
        &[time, time],
        indices(&[time, time])
            .iter()
            .map(|i| if i[1] <= i[0] { 0.0f32 } else { -1e9 })
            .collect::<Vec<_>>(),
    );
    let masked = &scores + &mask;
    assert_eq!(masked.shape(), [batch, heads, time, time]);
    for index in indices(masked.shape()) {
        let want = scores.get(&index).unwrap() + mask.get(&index[2..]).unwrap();
        assert_eq!(masked.get(&index), Some(want));
    }
    check_broadcast(scores.view(), mask.view(), &[batch, heads, time, time]);
    // A [D] scale over [B, T, D] activations, and [B, 1, 1, T] padding.
    let x = tensor::<f32>(&[batch, time, dim]);
    let gain = tensor::<f32>(&[dim]).with_scalar(0.5, BinaryOp::Add, false);
    check_broadcast(x.view(), gain.view(), &[batch, time, dim]);
    let padding = tensor::<f32>(&[batch, 1, 1, time]).with_scalar(0.5, BinaryOp::Add, false);
    check_broadcast(scores.view(), padding.view(), &[batch, heads, time, time]);
}

#[test]
fn broadcast_operands_may_be_non_contiguous_views() {
    let x = tensor::<f32>(&[4, 3, 2]);
    let xt = x.permute(&[2, 1, 0]); // [2, 3, 4], strided
    let wide = tensor::<f32>(&[3, 8]).with_scalar(0.5, BinaryOp::Add, false);
    let narrow = wide.narrow(1, 2, 4); // [3, 4], offset and strided rows
    check_broadcast(xt, narrow, &[2, 3, 4]);
    // A column picked out of a matrix, broadcast along the rows.
    let picked = wide.select(1, 5).unsqueeze(1); // [3, 1], stride 8
    check_broadcast(xt, picked, &[2, 3, 4]);
    // Same shapes, both strided.
    check_broadcast(xt, xt, &[2, 3, 4]);
}

#[test]
fn same_shape_tensors_take_the_vector_kernel() {
    // Two whole tensors of one shape give exactly what the flat vectors do.
    let a = tensor::<f32>(&[3, 4]);
    let b = tensor::<f32>(&[3, 4]).with_scalar(0.5, BinaryOp::Add, false);
    let flat = |t: &Tensor<f32>| Vector::new(t.to_vec());
    assert_eq!(
        (&a / &b).to_vec(),
        Host::vector_elementwise(&flat(&a), &flat(&b), BinaryOp::Div).to_vec()
    );
    assert_eq!(
        a.compare(&b, Compare::Max).to_vec(),
        Host::vector_compare(&flat(&a), &flat(&b), Compare::Max).to_vec()
    );
}

#[test]
#[should_panic(expected = "add: tensor shapes differ, [2, 3] and [4], and do not broadcast")]
fn incompatible_shapes_name_the_operation() {
    let _ = &tensor::<f32>(&[2, 3]) + &tensor::<f32>(&[4]);
}

#[test]
#[should_panic(expected = "power: tensor shapes differ, [2, 3, 4] and [3, 3]")]
fn incompatible_shapes_name_the_power() {
    let _ = tensor::<f32>(&[2, 3, 4]).power(&tensor::<f32>(&[3, 3]));
}

#[test]
fn a_broadcast_view_repeats_with_zero_strides() {
    let bias = tensor::<f32>(&[4]);
    let repeated = bias.broadcast_to(&[2, 3, 4]);
    assert_eq!(repeated.strides(), [0, 0, 1]);
    assert!(!repeated.is_contiguous());
    let copied = repeated.contiguous();
    assert_eq!(copied.shape(), [2, 3, 4]);
    for index in indices(&[2, 3, 4]) {
        assert_eq!(copied.get(&index), bias.get(&index[2..]));
    }
    let column = tensor::<f32>(&[3, 1]);
    assert_eq!(column.broadcast_to(&[2, 3, 5]).strides(), [0, 1, 0]);
    // A bias broadcast over leading axes is a fused-program input: its rows
    // fold into one stride-zero axis.
    let x = tensor::<f32>(&[2, 3, 4]);
    let mut builder = Builder::<f32>::new();
    let tensor = Decl::tensor(DType::F32, &[2, 3, 4]);
    let (a, b) = (builder.input(tensor.clone()), builder.input(tensor));
    let sum = builder.add(a, b);
    builder.output(sum, DType::F32);
    let program = builder.build().unwrap();
    let fused = program
        .run(&[&x, &repeated], &mut [])
        .remove(0)
        .into_matrix::<f32>();
    assert_eq!(fused.data(), (&x + &bias).to_vec());
}

#[test]
#[should_panic(expected = "broadcast_to: shape [3, 2] does not broadcast to [3, 4]")]
fn only_unit_axes_broadcast() {
    let _ = tensor::<f32>(&[3, 2]).broadcast_to(&[3, 4]);
}

fn broadcast_generic<T: Real>()
where
    Host: Kernels<T>,
{
    let a = tensor::<T>(&[2, 3, 4]);
    let b = tensor::<T>(&[3, 1]);
    let sum = &a + &b;
    let want = naive_broadcast(a.view(), b.view(), &[2, 3, 4], |x, y| x + y);
    assert_eq!(sum.to_vec(), want);
    let product = a
        .transpose(0, 1)
        .elementwise(b.view().unsqueeze(1), BinaryOp::Mul);
    let want = naive_broadcast(
        a.transpose(0, 1),
        b.view().unsqueeze(1),
        &[3, 2, 4],
        |x, y| x * y,
    );
    assert_eq!(product.to_vec(), want);
}

#[test]
fn every_float_type_broadcasts() {
    broadcast_generic::<f32>();
    broadcast_generic::<f64>();
    broadcast_generic::<f16>();
    broadcast_generic::<bf16>();
}

// ---- reductions ----------------------------------------------------------------

/// Every subset of `0..rank`, as ascending axis lists.
fn subsets(rank: usize) -> Vec<Vec<usize>> {
    (0..1usize << rank)
        .map(|bits| (0..rank).filter(|axis| bits & (1 << axis) != 0).collect())
        .collect()
}

/// The slices of `view` along `axes`, one per index of the other axes in
/// row-major order, each in its own row-major order, widened to `f64`.
fn naive_slices<T: Real>(view: TensorView<'_, T, Host>, axes: &[usize]) -> Vec<Vec<f64>> {
    let shape = view.shape();
    let kept = (0..shape.len())
        .filter(|axis| !axes.contains(axis))
        .collect::<Vec<_>>();
    let results = kept.iter().map(|&axis| shape[axis]).product::<usize>();
    let mut slices = vec![Vec::new(); results];
    for index in indices(shape) {
        let slot = kept
            .iter()
            .fold(0, |slot, &axis| slot * shape[axis] + index[axis]);
        slices[slot].push(view.get(&index).unwrap().into_f64());
    }
    slices
}

fn mean(slice: &[f64]) -> f64 {
    slice.iter().sum::<f64>() / slice.len() as f64
}

fn variance(slice: &[f64], ddof: usize) -> f64 {
    let m = mean(slice);
    slice.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (slice.len() - ddof) as f64
}

fn kept_shape(shape: &[usize], axes: &[usize], keep_dims: bool) -> Vec<usize> {
    (0..shape.len())
        .filter_map(|axis| match axes.contains(&axis) {
            false => Some(shape[axis]),
            true => keep_dims.then_some(1),
        })
        .collect()
}

fn assert_close(got: &[f64], want: &[f64], tolerance: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tolerance * w.abs().max(1.0) || (g.is_nan() && w.is_nan()),
            "{what}[{k}]: {g} vs {w}"
        );
    }
}

fn widened<T: Real>(t: &Tensor<T>) -> Vec<f64> {
    t.to_vec().into_iter().map(Real::into_f64).collect()
}

/// Every reduction of `view` over every subset of its axes, with and without
/// `keep_dims`, against the naive slices.
fn check_reductions<T: Real>(view: TensorView<'_, T, Host>, tolerance: f64)
where
    Host: Kernels<T>,
{
    let round = |x: f64| T::from_f64(x).into_f64();
    for axes in subsets(view.rank()) {
        let slices = naive_slices(view, &axes);
        for keep_dims in [false, true] {
            let shape = kept_shape(view.shape(), &axes, keep_dims);
            let what = format!("axes {axes:?} keep {keep_dims}");
            let sum = view.sum_axes(axes.as_slice(), keep_dims);
            assert_eq!(sum.shape(), shape, "{what}");
            let sums = slices
                .iter()
                .map(|s| round(s.iter().sum()))
                .collect::<Vec<_>>();
            assert_eq!(widened(&sum), sums, "sum {what}");
            let max = slices
                .iter()
                .map(|s| s.iter().copied().fold(f64::NEG_INFINITY, f64::max))
                .collect::<Vec<_>>();
            assert_eq!(widened(&view.max_axes(&axes, keep_dims)), max, "max {what}");
            let min = slices
                .iter()
                .map(|s| s.iter().copied().fold(f64::INFINITY, f64::min))
                .collect::<Vec<_>>();
            assert_eq!(
                widened(&view.min_axes(axes.clone(), keep_dims)),
                min,
                "min {what}"
            );
            let means = slices.iter().map(|s| mean(s)).collect::<Vec<_>>();
            let got = view.mean_axes(axes.as_slice(), keep_dims);
            assert_eq!(got.shape(), shape);
            assert_close(&widened(&got), &means, tolerance, &format!("mean {what}"));
            let population = slices.iter().map(|s| variance(s, 0)).collect::<Vec<_>>();
            let got = view.var_axes(axes.as_slice(), Correction::Population, keep_dims);
            assert_close(
                &widened(&got),
                &population,
                tolerance,
                &format!("var {what}"),
            );
            if slices.iter().all(|s| s.len() > 1) {
                let sample = slices.iter().map(|s| variance(s, 1)).collect::<Vec<_>>();
                let got = view.var_axes(axes.as_slice(), Correction::Sample, keep_dims);
                assert_close(
                    &widened(&got),
                    &sample,
                    tolerance,
                    &format!("sample {what}"),
                );
            }
        }
    }
}

#[test]
fn reductions_over_every_subset_of_axes_match_the_reference() {
    let x = tensor::<f32>(&[2, 3, 4, 5]);
    check_reductions(x.view(), 1e-6);
    let x = tensor::<f64>(&[2, 3, 4, 5]);
    check_reductions(x.view(), 1e-12);
    // Compact types: the sums of these integers are exact in the `f32`
    // accumulator, rounded once.
    let x = tensor::<f16>(&[2, 3, 4, 5]);
    check_reductions(x.view(), 1e-3);
    let x = tensor::<bf16>(&[2, 3, 4, 5]);
    check_reductions(x.view(), 1e-2);
}

#[test]
fn reductions_read_non_contiguous_views_in_place() {
    let x = tensor::<f32>(&[3, 4, 5, 6]);
    // A permutation, a slice and a selection of one.
    check_reductions(x.permute(&[2, 0, 3, 1]), 1e-6);
    check_reductions(x.narrow(3, 1, 4).transpose(0, 2), 1e-6);
    check_reductions(x.select(1, 2).slice(2, 1..5), 1e-6);
    // A broadcast view: every repeat counts.
    let row = tensor::<f32>(&[5]);
    let repeated = row.broadcast_to(&[3, 5]);
    assert_eq!(
        repeated.sum_axes(0, false).to_vec(),
        row.to_vec().iter().map(|x| 3.0 * x).collect::<Vec<_>>()
    );
}

#[test]
fn axes_may_count_from_the_end() {
    let x = tensor::<f32>(&[2, 3, 4, 5]);
    assert_eq!(x.sum_axes(-1, false), x.sum_axes(3usize, false));
    assert_eq!(x.sum_axes([-1isize, 0], true), x.sum_axes([0, 3], true));
    assert_eq!(
        x.max_axes(vec![-3i32, -2], false),
        x.max_axes([1, 2], false)
    );
    assert_eq!(x.mean_axes(.., false).shape(), [] as [usize; 0]);
    assert_eq!(x.mean_axes(.., true).shape(), [1, 1, 1, 1]);
    assert_eq!(x.sum_axes(.., false), x.sum_axes([0, 1, 2, 3], false));
    // No axes at all: every element is its own slice.
    let none: [usize; 0] = [];
    assert_eq!(x.sum_axes(none, false), x);
    assert_eq!(
        x.var_axes(none, Correction::Population, false),
        Tensor::zeros(&[2, 3, 4, 5])
    );
}

#[test]
#[should_panic(expected = "sum_axes: axes [1, -2] are not distinct axes of shape [2, 3, 4]")]
fn an_axis_may_be_reduced_once() {
    tensor::<f32>(&[2, 3, 4]).sum_axes([1isize, -3 + 1], false);
}

#[test]
#[should_panic(expected = "mean_axes: axes 3 are not distinct axes of shape [2, 3, 4]")]
fn a_reduced_axis_must_exist() {
    tensor::<f32>(&[2, 3, 4]).mean_axes(3usize, false);
}

#[test]
fn empty_reductions_give_the_identity() {
    let empty = Tensor::<f32>::zeros(&[2, 0, 3]);
    assert_eq!(empty.sum_axes(1, false), Tensor::zeros(&[2, 3]));
    assert_eq!(
        empty.max_axes(1, true),
        Tensor::filled(&[2, 1, 3], f32::NEG_INFINITY)
    );
    assert_eq!(
        empty.min_axes(1, false),
        Tensor::filled(&[2, 3], f32::INFINITY)
    );
    assert!(
        empty
            .mean_axes(1, false)
            .to_vec()
            .iter()
            .all(|x| x.is_nan())
    );
    assert!(
        empty
            .var_axes(1, Correction::Population, false)
            .to_vec()
            .iter()
            .all(|x| x.is_nan())
    );
    // Reducing a non-empty axis of an empty tensor gives an empty result.
    assert_eq!(empty.sum_axes(2, false).shape(), [2, 0]);
    assert_eq!(empty.argmax(2, false).shape(), [2, 0]);
    // One element has no sample variance.
    let single = tensor::<f32>(&[3, 1]);
    assert!(
        single
            .var_axes(1, Correction::Sample, false)
            .to_vec()
            .iter()
            .all(|x| x.is_nan())
    );
    assert_eq!(
        single.var_axes(1, Correction::Population, false),
        Tensor::zeros(&[3])
    );
}

#[test]
fn compact_sums_accumulate_in_f32() {
    let ones = Tensor::<f16>::ones(&[2, 10_000]);
    assert_eq!(
        ones.sum_axes(1, false).to_vec(),
        [f16::from_f32(10_000.0); 2]
    );
    let ones = Tensor::<bf16>::ones(&[3, 1000]);
    assert_eq!(ones.mean_axes(-1, false).to_vec(), [bf16::from_f32(1.0); 3]);
}

#[test]
fn extremes_pass_over_nans() {
    let nan = f32::NAN;
    let x = Tensor::from_vec(&[3, 3], vec![1.0, nan, 3.0, nan, nan, nan, -2.0, 5.0, nan]);
    assert_eq!(x.max_axes(1, false).to_vec(), [3.0, f32::NEG_INFINITY, 5.0]);
    assert_eq!(x.min_axes(1, false).to_vec(), [1.0, f32::INFINITY, -2.0]);
    assert_eq!(x.max_axes(0, false).to_vec(), [1.0, 5.0, 3.0]);
    assert_eq!(x.argmax(1, false).to_vec(), [2u32, 0, 1]);
    assert_eq!(x.argmin(1, false).to_vec(), [0u32, 0, 0]);
    assert_eq!(x.argmax(0, false).to_vec(), [0u32, 2, 0]);
    // A sum propagates the NaN.
    assert!(x.sum_axes(1, false).to_vec()[0].is_nan());
}

/// The first position of the extreme of each slice along `axis`, skipping
/// NaNs, or `0` for a slice of NaNs.
fn naive_arg<T: Real>(view: TensorView<'_, T, Host>, axis: usize, max: bool) -> Vec<u32> {
    naive_slices(view, &[axis])
        .iter()
        .map(|slice| {
            let mut best: Option<(usize, f64)> = None;
            for (k, &x) in slice.iter().enumerate() {
                let better =
                    !x.is_nan() && best.is_none_or(|(_, b)| if max { x > b } else { x < b });
                if better {
                    best = Some((k, x));
                }
            }
            best.map_or(0, |(k, _)| k as u32)
        })
        .collect()
}

fn check_args<T: Real>(view: TensorView<'_, T, Host>)
where
    Host: Kernels<T>,
{
    for axis in 0..view.rank() {
        for keep_dims in [false, true] {
            let shape = kept_shape(view.shape(), &[axis], keep_dims);
            let max = view.argmax(axis, keep_dims);
            assert_eq!(max.shape(), shape);
            assert_eq!(max.to_vec(), naive_arg(view, axis, true), "argmax {axis}");
            let min = view.argmin(axis as isize - view.rank() as isize, keep_dims);
            assert_eq!(min.to_vec(), naive_arg(view, axis, false), "argmin {axis}");
        }
    }
}

#[test]
fn arg_extremes_along_every_axis_match_the_reference() {
    // The filler repeats every 23 elements, so ties are common.
    let x = tensor::<f32>(&[2, 3, 4, 5]);
    check_args(x.view());
    check_args(x.permute(&[3, 1, 0, 2]));
    check_args(tensor::<f64>(&[3, 4, 2]).view());
    check_args(tensor::<f16>(&[3, 4, 2]).view());
    check_args(tensor::<bf16>(&[3, 4, 2]).view());
    // Ties go to the first position.
    let tied = Tensor::from_vec(&[2, 4], vec![1.0f32, 7.0, 7.0, 2.0, -3.0, -3.0, 0.0, -3.0]);
    assert_eq!(tied.argmax(1, false), Tensor::from_vec(&[2], vec![1u32, 2]));
    assert_eq!(
        tied.argmin(1, true),
        Tensor::from_vec(&[2, 1], vec![0u32, 0])
    );
}

#[test]
#[should_panic(expected = "argmax: axis 1 of shape [2, 0] is empty")]
fn an_empty_axis_has_no_argmax() {
    Tensor::<f32>::zeros(&[2, 0]).argmax(1, false);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    use super::*;
    use tensorcrate::metal::MetalElement;
    use tensorcrate::tensors::Metal;

    /// Two operands on the host, and the same two on the device.
    type Pair<'a, T> = (
        TensorView<'a, T>,
        TensorView<'a, T>,
        TensorView<'a, T, Metal>,
        TensorView<'a, T, Metal>,
    );

    fn on_both<T: MetalElement>(shape: &[usize]) -> (Tensor<T>, Tensor<T, Metal>) {
        let host = tensor::<T>(shape);
        let device = host.to_backend::<Metal>();
        (host, device)
    }

    fn close<T: Real>(got: &Tensor<T, Metal>, want: &Tensor<T>, tolerance: f64, what: &str) {
        assert_eq!(got.shape(), want.shape(), "{what}");
        let got = got.to_backend::<Host>();
        assert_close(&widened(&got), &widened(want), tolerance, what);
    }

    fn broadcasts_agree<T: MetalElement>(tolerance: f64)
    where
        Metal: Kernels<T>,
    {
        let (scores_h, scores_d) = on_both::<T>(&[2, 3, 5, 5]);
        let (mask_h, mask_d) = on_both::<T>(&[5, 5]);
        let (x_h, x_d) = on_both::<T>(&[2, 5, 4]);
        let (bias_h, bias_d) = on_both::<T>(&[4]);
        let (col_h, col_d) = on_both::<T>(&[2, 1]);
        let scalar_h = Tensor::<T>::from_vec(&[], vec![T::from_f64(1.5)]);
        let scalar_d = scalar_h.to_backend::<Metal>();
        let resident = x_d.as_vector().is_device_resident();
        let pairs: Vec<Pair<'_, T>> = vec![
            (
                scores_h.view(),
                mask_h.view(),
                scores_d.view(),
                mask_d.view(),
            ),
            (
                mask_h.view(),
                scores_h.view(),
                mask_d.view(),
                scores_d.view(),
            ),
            (x_h.view(), bias_h.view(), x_d.view(), bias_d.view()),
            (
                x_h.transpose(0, 1),
                col_h.view(),
                x_d.transpose(0, 1),
                col_d.view(),
            ),
            (x_h.view(), scalar_h.view(), x_d.view(), scalar_d.view()),
            (
                x_h.permute(&[2, 1, 0]),
                x_h.permute(&[2, 1, 0]),
                x_d.permute(&[2, 1, 0]),
                x_d.permute(&[2, 1, 0]),
            ),
        ];
        for (ah, bh, ad, bd) in pairs {
            for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul] {
                let d = ad.elementwise(bd, op);
                assert_eq!(d.as_vector().is_device_resident(), resident);
                assert_eq!(
                    d.to_backend::<Host>(),
                    ah.elementwise(bh, op),
                    "{} {op:?}",
                    T::SUFFIX
                );
            }
            for op in Compare::ALL {
                let d = ad.compare(bd, op);
                assert_eq!(
                    d.to_backend::<Host>(),
                    ah.compare(bh, op),
                    "{} {op:?}",
                    T::SUFFIX
                );
            }
            let what = format!("{} divide {:?} by {:?}", T::SUFFIX, ah.shape(), bh.shape());
            let shifted_h = bh.with_scalar(T::from_f64(0.5), BinaryOp::Add, false);
            let shifted_d = bd.with_scalar(T::from_f64(0.5), BinaryOp::Add, false);
            close(
                &ad.elementwise(&shifted_d, BinaryOp::Div),
                &ah.elementwise(&shifted_h, BinaryOp::Div),
                tolerance,
                &what,
            );
            let base_h = ah.with_scalar(T::from_f64(12.0), BinaryOp::Add, false);
            let base_d = ad.with_scalar(T::from_f64(12.0), BinaryOp::Add, false);
            let exponent_h = bh.with_scalar(T::from_f64(0.1), BinaryOp::Mul, false);
            let exponent_d = bd.with_scalar(T::from_f64(0.1), BinaryOp::Mul, false);
            close(
                &base_d.power(&exponent_d),
                &base_h.power(&exponent_h),
                10.0 * tolerance,
                &what,
            );
            let remainder = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                ad.elementwise(&shifted_d, BinaryOp::Rem)
            }));
            assert!(
                remainder.is_err(),
                "Metal remainder must not fall back to Host"
            );
        }
    }

    #[test]
    fn broadcasts_agree_with_host_for_every_metal_type() {
        broadcasts_agree::<f32>(1e-6);
        broadcasts_agree::<f16>(1e-3);
        broadcasts_agree::<bf16>(1e-2);
    }

    fn reductions_agree<T: MetalElement>(shape: &[usize], tolerance: f64)
    where
        Metal: Kernels<T>,
    {
        let (host, device) = on_both::<T>(shape);
        let resident = device.as_vector().is_device_resident();
        let views = [
            (host.view(), device.view()),
            (host.permute(&[3, 1, 0, 2]), device.permute(&[3, 1, 0, 2])),
            (
                host.narrow(1, 1, shape[1] - 1),
                device.narrow(1, 1, shape[1] - 1),
            ),
        ];
        for (h, d) in views {
            for axes in subsets(h.rank()) {
                for keep_dims in [false, true] {
                    let what = format!(
                        "{} {:?} axes {axes:?} keep {keep_dims}",
                        T::SUFFIX,
                        h.shape()
                    );
                    let sum = d.sum_axes(axes.as_slice(), keep_dims);
                    assert_eq!(sum.as_vector().is_device_resident(), resident);
                    // Sums of these integers are exact in `f32`, whatever the order.
                    assert_eq!(
                        sum.to_backend::<Host>(),
                        h.sum_axes(axes.as_slice(), keep_dims),
                        "sum {what}"
                    );
                    assert_eq!(
                        d.max_axes(axes.as_slice(), keep_dims).to_backend::<Host>(),
                        h.max_axes(axes.as_slice(), keep_dims),
                        "max {what}"
                    );
                    assert_eq!(
                        d.min_axes(axes.as_slice(), keep_dims).to_backend::<Host>(),
                        h.min_axes(axes.as_slice(), keep_dims),
                        "min {what}"
                    );
                    close(
                        &d.mean_axes(axes.as_slice(), keep_dims),
                        &h.mean_axes(axes.as_slice(), keep_dims),
                        tolerance,
                        &format!("mean {what}"),
                    );
                    for correction in [Correction::Population, Correction::Sample] {
                        close(
                            &d.var_axes(axes.as_slice(), correction, keep_dims),
                            &h.var_axes(axes.as_slice(), correction, keep_dims),
                            tolerance,
                            &format!("var {correction:?} {what}"),
                        );
                    }
                }
            }
            for axis in 0..h.rank() {
                for keep_dims in [false, true] {
                    let max = d.argmax(axis, keep_dims);
                    assert_eq!(max.as_vector().is_device_resident(), resident);
                    assert_eq!(
                        max.to_backend::<Host>(),
                        h.argmax(axis, keep_dims),
                        "{} argmax {axis}",
                        T::SUFFIX
                    );
                    assert_eq!(
                        d.argmin(axis, keep_dims).to_backend::<Host>(),
                        h.argmin(axis, keep_dims),
                        "{} argmin {axis}",
                        T::SUFFIX
                    );
                }
            }
        }
    }

    #[test]
    fn reductions_agree_with_host_for_every_metal_type() {
        // Short slices fold one per thread; slices of 64 or more, one per
        // SIMD group — along the last axis, and down a leading one.
        for shape in [[2, 3, 4, 5], [3, 2, 5, 70], [80, 3, 2, 4]] {
            reductions_agree::<f32>(&shape, 1e-5);
            reductions_agree::<f16>(&shape, 2e-3);
            reductions_agree::<bf16>(&shape, 1e-2);
        }
    }

    #[test]
    fn many_long_slices_reduce_on_the_device() {
        // Over a thousand results of 100-element slices down a leading axis:
        // one thread per result.
        let (host, device) = on_both::<f32>(&[100, 1100]);
        assert_eq!(
            device.sum_axes(0, false).to_backend::<Host>(),
            host.sum_axes(0, false)
        );
        assert_eq!(
            device.argmax(0, false).to_backend::<Host>(),
            host.argmax(0, false)
        );
        let (host, device) = on_both::<f32>(&[1100, 100]);
        assert_eq!(
            device.max_axes(1, false).to_backend::<Host>(),
            host.max_axes(1, false)
        );
        close(
            &device.var_axes(1, Correction::Sample, true),
            &host.var_axes(1, Correction::Sample, true),
            1e-5,
            "var",
        );
    }

    #[test]
    fn the_device_passes_over_nans_like_the_host() {
        let nan = f32::NAN;
        let mut values = vec![nan; 3 * 70];
        values[5] = 2.0;
        values[69] = 2.0;
        values[70 + 3] = -1.0;
        values[140 + 1] = 4.0;
        values[140 + 2] = 4.0;
        for len in [70, 7] {
            let slice = |row: usize| &values[row * 70..row * 70 + len];
            let rows = (0..3).flat_map(slice).copied().collect::<Vec<_>>();
            let host = Tensor::from_vec(&[3, len], rows);
            let device = host.to_backend::<Metal>();
            let bits = |t: Tensor<f32>| t.to_vec().iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(device.max_axes(1, false).to_backend()),
                bits(host.max_axes(1, false))
            );
            assert_eq!(
                bits(device.min_axes(1, false).to_backend()),
                bits(host.min_axes(1, false))
            );
            assert_eq!(
                device.argmax(1, false).to_backend::<Host>(),
                host.argmax(1, false)
            );
            assert_eq!(
                device.argmin(1, false).to_backend::<Host>(),
                host.argmin(1, false)
            );
        }
        let host = Tensor::from_vec(&[2, 70], [vec![nan; 70], vec![1.0; 70]].concat());
        assert_eq!(host.argmax(1, false).to_vec(), [0, 0]);
        assert_eq!(host.to_backend::<Metal>().argmax(1, false).to_vec(), [0, 0]);
    }

    #[test]
    fn compact_sums_accumulate_in_f32_on_the_device() {
        let ones = Tensor::<f16>::ones(&[2, 10_000]).to_backend::<Metal>();
        assert_eq!(
            ones.sum_axes(1, false).to_vec(),
            [f16::from_f32(10_000.0); 2]
        );
        let ones = Tensor::<bf16>::ones(&[1000, 3]).to_backend::<Metal>();
        assert_eq!(
            ones.sum_axes(0, false).to_vec(),
            [bf16::from_f32(1000.0); 3]
        );
    }

    #[test]
    fn index_tensors_compare_across_backends() {
        let host = tensor::<f32>(&[4, 6]);
        let device = host.to_backend::<Metal>();
        let ids: Tensor<u32, Metal> = device.argmax(1, false);
        assert_eq!(ids, host.argmax(1, false).to_backend::<Metal>());
        assert_eq!(ids.get(&[2]), host.argmax(1, false).get(&[2]));
    }
}
