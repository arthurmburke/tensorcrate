//! Reverse mode over N-dimensional tensors.
//!
//! Every differentiable tensor operation is one [`Op`] below. Each is checked
//! three ways: against central finite differences in `f64` on `Host`, which
//! is the oracle; in `f32` on `Host` against the `f64` gradient; and, with
//! the `metal` feature, on `Metal` against `Host` in `f32`, `f16` and `bf16`.
//! The inputs are rank 3 and 4, broadcast against one another, and read
//! through permutations and slices. Then the pieces a per-operation check
//! cannot show: the tie conventions, adjoints accumulating into a slice of an
//! existing one, empty tensors, optimizers updating a tensor parameter, and a
//! small attention-shaped network that trains.

use tensorcrate::numbers::{Real, bf16, f16};
use tensorcrate::optim::{Adam, Momentum, Rule, Sgd, minimize};
use tensorcrate::statistics::Correction;
use tensorcrate::tensors::{
    Analytic, Host, Kernels, Matrix, MatrixVar, ScalarVar, Tape, Tensor, TensorVar,
};

// ---- inputs ------------------------------------------------------------------

/// A tensor of `shape` with distinct values spread over `[low, high)`: a
/// golden-ratio sequence, so no two elements tie and none sits on a kink.
fn filled(shape: &[usize], seed: usize, low: f64, high: f64) -> Tensor<f64> {
    let len = shape.iter().product::<usize>();
    let values = (0..len)
        .map(|i| {
            let t = ((i + 1) as f64 * 0.618_033_988_75 + seed as f64 * 0.137).fract();
            low + (high - low) * t
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(shape, values)
}

fn convert<E: Real>(tensor: &Tensor<f64>) -> Tensor<E> {
    Tensor::from_vec(
        tensor.shape(),
        tensor
            .as_slice()
            .iter()
            .map(|&x| E::from_f64(x))
            .collect::<Vec<_>>(),
    )
}

fn to_f64<E: Real>(values: &[E]) -> Vec<f64> {
    values.iter().map(|&x| x.into_f64()).collect()
}

// ---- the operations ----------------------------------------------------------

/// One differentiable tensor operation, applied to recorded inputs.
#[derive(Copy, Clone, Debug)]
enum Op {
    AddSame,
    AddBroadcast,
    SubBroadcast,
    MulBroadcast,
    DivBroadcast,
    DivLeftBroadcast,
    ScalarOperand,
    PermutedOperand,
    NarrowedOperand,
    Maximum,
    Minimum,
    Power,
    PowerScalar,
    PowerScalarLeft,
    ScaleShiftNeg,
    Abs,
    Relu,
    Clamp,
    Analytic(Analytic),
    Reshape,
    Permute,
    Transpose,
    Narrow,
    Slice,
    Select,
    Split,
    Chunk,
    Concat,
    Stack,
    SqueezeUnsqueeze,
    BroadcastTo,
    Contiguous,
    SumAxes,
    SumAxesKeep,
    SumAll,
    MeanAxes,
    VarAxes(Correction),
    MaxAxes,
    MinAxes,
    MatrixRoundTrip,
    VectorRoundTrip,
    SliceIntoExistingAdjoint,
    SliceBeforeOtherUse,
}

const X: [usize; 4] = [2, 3, 4, 5];

impl Op {
    fn all() -> Vec<Op> {
        let mut ops = vec![
            Op::AddSame,
            Op::AddBroadcast,
            Op::SubBroadcast,
            Op::MulBroadcast,
            Op::DivBroadcast,
            Op::DivLeftBroadcast,
            Op::ScalarOperand,
            Op::PermutedOperand,
            Op::NarrowedOperand,
            Op::Maximum,
            Op::Minimum,
            Op::Power,
            Op::PowerScalar,
            Op::PowerScalarLeft,
            Op::ScaleShiftNeg,
            Op::Abs,
            Op::Relu,
            Op::Clamp,
        ];
        ops.extend(Analytic::ALL.map(Op::Analytic));
        ops.extend([
            Op::Reshape,
            Op::Permute,
            Op::Transpose,
            Op::Narrow,
            Op::Slice,
            Op::Select,
            Op::Split,
            Op::Chunk,
            Op::Concat,
            Op::Stack,
            Op::SqueezeUnsqueeze,
            Op::BroadcastTo,
            Op::Contiguous,
            Op::SumAxes,
            Op::SumAxesKeep,
            Op::SumAll,
            Op::MeanAxes,
            Op::VarAxes(Correction::Population),
            Op::VarAxes(Correction::Sample),
            Op::MaxAxes,
            Op::MinAxes,
            Op::MatrixRoundTrip,
            Op::VectorRoundTrip,
            Op::SliceIntoExistingAdjoint,
            Op::SliceBeforeOtherUse,
        ]);
        ops
    }

    /// The operation's inputs, in `f64`.
    fn inputs(self) -> Vec<Tensor<f64>> {
        let x = || filled(&X, 1, -1.0, 1.0);
        let positive = |shape: &[usize], seed| filled(shape, seed, 0.5, 1.5);
        match self {
            Op::AddSame => vec![x(), filled(&X, 2, -1.0, 1.0)],
            Op::AddBroadcast => vec![x(), filled(&[4, 1], 2, -1.0, 1.0)],
            Op::SubBroadcast => vec![filled(&[3, 1, 5], 2, -1.0, 1.0), x()],
            Op::MulBroadcast => vec![
                filled(&[2, 1, 4, 1], 1, -1.0, 1.0),
                filled(&[1, 3, 1, 5], 2, -1.0, 1.0),
            ],
            Op::DivBroadcast => vec![x(), positive(&[5], 2)],
            Op::DivLeftBroadcast => vec![filled(&[3, 1, 1], 2, -1.0, 1.0), positive(&X, 3)],
            Op::ScalarOperand => vec![x(), filled(&[], 2, 0.5, 1.5)],
            Op::PermutedOperand => vec![x(), filled(&[3, 1], 2, -1.0, 1.0)],
            Op::NarrowedOperand => vec![x(), filled(&[2, 1, 4, 5], 2, -1.0, 1.0)],
            Op::Maximum => vec![x(), filled(&[4, 5], 2, -1.0, 1.0)],
            Op::Minimum => vec![filled(&[3, 1, 5], 2, -1.0, 1.0), x()],
            Op::Power => vec![positive(&[2, 3, 4], 1), filled(&[4], 2, -1.0, 2.0)],
            Op::PowerScalar => vec![positive(&X, 1)],
            Op::Analytic(_) => vec![filled(&[2, 3, 4], 1, 0.15, 0.85)],
            Op::Split | Op::Chunk => vec![filled(&[2, 3, 4, 6], 1, -1.0, 1.0)],
            Op::Concat => vec![
                filled(&[2, 3, 4, 2], 1, -1.0, 1.0),
                filled(&[2, 3, 4, 3], 2, -1.0, 1.0),
            ],
            Op::Stack => vec![
                filled(&[2, 3, 4], 1, -1.0, 1.0),
                filled(&[2, 3, 4], 2, -1.0, 1.0),
            ],
            Op::SqueezeUnsqueeze => vec![filled(&[2, 1, 4], 1, -1.0, 1.0)],
            Op::BroadcastTo => vec![filled(&[3, 1, 5], 1, -1.0, 1.0)],
            Op::MatrixRoundTrip => vec![
                filled(&[2, 3, 4], 1, -1.0, 1.0),
                filled(&[4, 5], 2, -1.0, 1.0),
            ],
            Op::VectorRoundTrip => vec![filled(&[2, 3, 4], 1, -1.0, 1.0)],
            _ => vec![x()],
        }
    }

    fn apply<'t, B: Kernels<E>, E: Real>(self, v: &[TensorVar<'t, B, E>]) -> TensorVar<'t, B, E> {
        let c = |value: f64| E::from_f64(value);
        match self {
            Op::AddSame | Op::AddBroadcast => &v[0] + &v[1],
            Op::SubBroadcast => &v[0] - &v[1],
            Op::MulBroadcast => &v[0] * &v[1],
            Op::DivBroadcast | Op::DivLeftBroadcast => &v[0] / &v[1],
            Op::ScalarOperand => &(&v[0] * &v[1]) + &v[1],
            // [2, 4, 3, 5] against [3, 1]: a permuted operand, broadcast.
            Op::PermutedOperand => &v[0].permute(&[0, 2, 1, 3]) * &v[1],
            Op::NarrowedOperand => &v[0].narrow(1, 1, 1) * &v[1],
            Op::Maximum => v[0].maximum(&v[1]),
            Op::Minimum => v[0].minimum(&v[1]),
            Op::Power => v[0].power(&v[1]),
            Op::PowerScalar => v[0].power_scalar(c(2.5), false),
            Op::PowerScalarLeft => v[0].power_scalar(c(1.7), true),
            Op::ScaleShiftNeg => v[0].scale(c(-1.5)).shift(c(0.25)).neg(),
            Op::Abs => v[0].abs(),
            Op::Relu => v[0].relu(),
            Op::Clamp => v[0].clamp(c(-0.5), c(0.4)),
            Op::Analytic(f) => v[0].analytic(f),
            Op::Reshape => v[0].reshape(&[6, 20]),
            Op::Permute => v[0].permute(&[0, 2, 3, 1]),
            Op::Transpose => v[0].transpose(-1, 1),
            Op::Narrow => v[0].narrow(2, 1, 2),
            Op::Slice => v[0].slice(-1, 1..4),
            Op::Select => v[0].select(1, 2),
            Op::Split => {
                let pieces = v[0].split(-1, &[2, 4]);
                // [2, 3, 4, 1] against [2, 3, 4, 4]: the pieces interact.
                &pieces[0].sum_axes(-1, true) * &pieces[1].tanh()
            }
            Op::Chunk => {
                let [q, k, w] = <[_; 3]>::try_from(v[0].chunk(-1, 3)).ok().unwrap();
                &(&q * &k) + &w.sin()
            }
            Op::Concat => TensorVar::concat(&[v[1].clone(), v[0].clone(), v[1].scale(c(2.0))], -1),
            Op::Stack => TensorVar::stack(&[v[0].clone(), v[1].clone(), v[0].clone()], 1),
            Op::SqueezeUnsqueeze => v[0].squeeze(1).unsqueeze(0).unsqueeze(-1),
            Op::BroadcastTo => v[0].broadcast_to(&X),
            Op::Contiguous => v[0].contiguous(),
            Op::SumAxes => v[0].sum_axes([1, 3], false),
            Op::SumAxesKeep => v[0].sum_axes(-2, true),
            Op::SumAll => v[0].sum_axes(.., false),
            Op::MeanAxes => v[0].mean_axes([0, 2], true),
            Op::VarAxes(Correction::Population) => {
                v[0].var_axes([1, 2], Correction::Population, false)
            }
            Op::VarAxes(correction) => v[0].var_axes(-1, correction, true),
            Op::MaxAxes => v[0].max_axes([1, 3], false),
            Op::MinAxes => v[0].min_axes(-1, true),
            Op::MatrixRoundTrip => {
                let product = v[0].to_matrix(6, 4).matmul(&v[1].to_matrix(4, 5));
                product.tanh().to_tensor(&[2, 3, 5])
            }
            Op::VectorRoundTrip => {
                let flat = v[0].to_vector();
                (&flat * &flat).to_tensor(&[4, 6])
            }
            // The narrow's rule runs after the product's, so it adds into an
            // adjoint that is already there.
            Op::SliceIntoExistingAdjoint => {
                let head = v[0].narrow(1, 0, 2).sum_axes(1, true);
                &head + &(&v[0] * &v[0])
            }
            // Here the product is recorded first, so the narrow's rule writes
            // into zeros and the product's adds to that.
            Op::SliceBeforeOtherUse => {
                let square = &v[0] * &v[0];
                &v[0].narrow(-1, 2, 3).exp().sum_axes(-1, true) + &square
            }
        }
    }
}

// ---- the harness ---------------------------------------------------------------

/// The gradients of `⟨op(inputs), seed⟩` by one backward pass, with the seed
/// a fixed tensor of the output's shape, and the output itself.
fn reverse<B: Kernels<E>, E: Real>(
    op: Op,
    inputs: &[Tensor<E, B>],
    seed: impl Fn(&[usize]) -> Tensor<E, B>,
) -> (Vec<Tensor<E, B>>, Tensor<E, B>) {
    let tape = Tape::<B>::new();
    let vars = inputs
        .iter()
        .map(|x| tape.tensor(x.contiguous()))
        .collect::<Vec<_>>();
    let out = op.apply(&vars);
    out.backward_with(seed(out.shape()));
    (
        vars.iter().map(|v| v.grad()).collect(),
        out.value().contiguous(),
    )
}

/// The seed every check projects an output onto.
fn seed_for(shape: &[usize]) -> Tensor<f64> {
    filled(shape, 7, -1.0, 1.0)
}

/// `⟨op(inputs), seed⟩` evaluated directly, for the finite differences.
fn projected(op: Op, inputs: &[Tensor<f64>], seed: &Tensor<f64>) -> f64 {
    let tape = Tape::<Host>::new();
    let vars = inputs
        .iter()
        .map(|x| tape.tensor(x.clone()))
        .collect::<Vec<_>>();
    let out = op.apply(&vars);
    out.value()
        .as_slice()
        .iter()
        .zip(seed.as_slice())
        .map(|(a, b)| a * b)
        .sum()
}

fn assert_close(got: &[f64], want: &[f64], tolerance: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    let scale = want.iter().fold(1.0f64, |m, w| m.max(w.abs()));
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tolerance * scale,
            "{what}[{k}]: {g} vs {w} (tolerance {tolerance} × {scale})"
        );
    }
}

#[test]
fn every_operation_matches_finite_differences_in_f64() {
    for op in Op::all() {
        let inputs = op.inputs();
        let (grads, out) = reverse::<Host, f64>(op, &inputs, seed_for);
        let seed = seed_for(out.shape());
        for (which, input) in inputs.iter().enumerate() {
            let mut numeric = Vec::with_capacity(input.len());
            for k in 0..input.len() {
                let x = input.as_slice()[k];
                let h = 1e-6 * x.abs().max(1.0);
                let nudged = |delta: f64| {
                    let mut moved = inputs.clone();
                    moved[which].data_mut()[k] = x + delta;
                    projected(op, &moved, &seed)
                };
                numeric.push((nudged(h) - nudged(-h)) / (2.0 * h));
            }
            assert_eq!(
                grads[which].shape(),
                input.shape(),
                "{op:?}: gradient shape"
            );
            assert_close(
                grads[which].as_slice(),
                &numeric,
                1e-6,
                &format!("{op:?} ∂/∂input{which}"),
            );
        }
    }
}

/// The gradients in `E` against the `f64` ones at the same inputs — the
/// inputs rounded to `E` first, so the two see the same ties.
fn agrees_with_f64<E: Real>(tolerance: f64)
where
    Host: Kernels<E>,
{
    for op in Op::all() {
        let narrow = op.inputs().iter().map(convert::<E>).collect::<Vec<_>>();
        let rounded = narrow
            .iter()
            .map(|x| Tensor::from_vec(x.shape(), to_f64(x.as_slice())))
            .collect::<Vec<_>>();
        let seed = |shape: &[usize]| convert::<E>(&seed_for(shape));
        let (want, _) = reverse::<Host, f64>(op, &rounded, |shape| {
            let seed = seed(shape);
            Tensor::from_vec(shape, to_f64(seed.as_slice()))
        });
        let (got, _) = reverse::<Host, E>(op, &narrow, seed);
        for (which, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_close(
                &to_f64(g.as_slice()),
                w.as_slice(),
                tolerance,
                &format!("{op:?} {} ∂/∂input{which}", std::any::type_name::<E>()),
            );
        }
    }
}

#[test]
fn every_operation_agrees_in_f32_with_f64() {
    agrees_with_f64::<f32>(2e-4);
}

#[test]
fn every_operation_agrees_in_f16_and_bf16_with_f64() {
    agrees_with_f64::<f16>(1e-2);
    agrees_with_f64::<bf16>(5e-2);
}

// ---- conventions ---------------------------------------------------------------

#[test]
fn axis_extremes_share_the_adjoint_among_ties_and_skip_nans() {
    let tape = Tape::<Host>::new();
    let x = tape.tensor(Tensor::from_vec(
        &[2, 4],
        vec![1.0f32, 3.0, 3.0, 2.0, f32::NAN, -1.0, 5.0, 5.0],
    ));
    let top = x.max_axes(-1, false);
    assert_eq!(top.value().to_vec(), [3.0, 5.0]);
    top.sum().backward();
    assert_eq!(x.grad().to_vec(), [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.5]);

    let bottom = x.min_axes([0, 1], true);
    assert_eq!(bottom.value().to_vec(), [-1.0]);
    bottom.sum().backward();
    assert_eq!(x.grad().to_vec(), [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);

    // A slice of only NaNs has no maximum to credit.
    let tape = Tape::<Host>::new();
    let nans = tape.tensor(Tensor::from_vec(&[1, 2], vec![f32::NAN, f32::NAN]));
    nans.max_axes(1, false).sum().backward();
    assert_eq!(nans.grad().to_vec(), [0.0, 0.0]);
}

#[test]
fn elementwise_extremes_split_ties_and_sum_over_the_broadcast() {
    let tape = Tape::<Host>::new();
    let x = tape.tensor(Tensor::from_vec(&[2, 2], vec![1.0f32, 2.0, 3.0, 0.0]));
    let floor = tape.tensor(Tensor::from_vec(&[2], vec![1.0f32, 1.0]));
    x.maximum(&floor).sum().backward();
    // Column 0 ties at 1 then x wins; column 1 the floor wins then x does.
    assert_eq!(x.grad().to_vec(), [0.5, 1.0, 1.0, 0.0]);
    assert_eq!(floor.grad().to_vec(), [0.5, 1.0]);
}

#[test]
fn whole_tensor_sums_and_means_are_lazy_scalars() {
    let tape = Tape::<Host>::new();
    let x = tape.tensor(filled(&[2, 3, 4], 1, -1.0, 1.0));
    let total = x.sum();
    let mean = x.mean();
    total.add(&mean).backward();
    let expected = 1.0 + 1.0 / 24.0;
    for g in x.grad().as_slice() {
        assert!((g - expected).abs() < 1e-12);
    }
    let sum = x.value().as_slice().iter().sum::<f64>();
    assert!((total.value() - sum).abs() < 1e-12);
    assert!((mean.value() - sum / 24.0).abs() < 1e-12);
}

fn empty_and_unit_extents<B: Kernels<f32>>() {
    let tape = Tape::<B>::new();
    let empty = tape.tensor(Tensor::<f32, B>::zeros(&[0, 3]));
    let unit = tape.tensor(Tensor::from_slice(&[1, 3, 1], &[1.0f32, 2.0, 3.0]));
    let folded = empty.sum_axes(0, false); // [3] of zeros
    assert_eq!(folded.value().to_vec(), [0.0; 3]);
    let joined = TensorVar::concat(&[empty.clone(), unit.reshape(&[1, 3])], 0);
    assert_eq!(joined.shape(), [1, 3]);
    let loss = (&(&joined + &folded) * &unit.squeeze(0).squeeze(-1)).sum();
    loss.backward();
    assert_eq!(empty.grad().shape(), [0, 3]);
    assert_eq!(unit.grad().to_vec(), [2.0, 4.0, 6.0]);

    // Narrowing to nothing sends nothing back, into an adjoint or not.
    let tape = Tape::<B>::new();
    let x = tape.tensor(Tensor::from_slice(&[2, 2], &[1.0f32, 2.0, 3.0, 4.0]));
    let none = x.narrow(1, 1, 0);
    assert!(none.is_empty());
    (&none.sum_axes(1, true) + &x).sum().backward();
    assert_eq!(x.grad().to_vec(), [1.0; 4]);
    let after = x.scale(2.0);
    (&x.narrow(0, 2, 0).sum_axes(0, false) + &after.sum_axes(0, false))
        .sum()
        .backward();
    assert_eq!(x.grad().to_vec(), [2.0; 4]);
}

#[test]
fn empty_and_unit_extents_differentiate() {
    empty_and_unit_extents::<Host>();
}

#[test]
fn a_second_backward_pass_gives_the_same_tensor_gradients() {
    let tape = Tape::<Host>::new();
    let x = tape.tensor(filled(&[2, 3, 4], 1, -1.0, 1.0));
    let pieces = x.chunk(1, 3);
    let loss = (&(&pieces[0] * &pieces[2]) + &pieces[1].exp()).sum();
    loss.backward();
    let first = x.grad();
    loss.backward();
    assert_eq!(x.grad(), first);
    tape.zero_grad();
    assert!(!x.has_grad());
}

#[test]
#[should_panic(expected = "add: tensor shapes differ, [2, 3] and [4]")]
fn shapes_that_do_not_broadcast_are_rejected() {
    let tape = Tape::<Host>::new();
    let a = tape.tensor(Tensor::<f32>::zeros(&[2, 3]));
    let b = tape.tensor(Tensor::<f32>::zeros(&[4]));
    let _ = &a + &b;
}

#[test]
#[should_panic(expected = "different tapes")]
fn tensors_from_different_tapes_are_rejected() {
    let (first, second) = (Tape::<Host>::new(), Tape::<Host>::new());
    let a = first.tensor(Tensor::<f32>::zeros(&[2]));
    let b = second.tensor(Tensor::<f32>::zeros(&[2]));
    let _ = TensorVar::concat(&[a, b], 0);
}

#[test]
#[should_panic(expected = "to_matrix: shape [2, 3] holds 6 elements, which 4×2 cannot")]
fn a_matrix_of_the_wrong_size_is_rejected() {
    let tape = Tape::<Host>::new();
    let a = tape.tensor(Tensor::<f32>::zeros(&[2, 3]));
    let _ = a.to_matrix(4, 2);
}

// ---- optimizers ----------------------------------------------------------------

/// `Σ w ⊙ (p − t)²` for a tensor parameter.
fn tensor_loss<'t, B: Kernels<f32>>(
    p: &TensorVar<'t, B>,
    target: &Tensor<f32, B>,
    weights: &Tensor<f32, B>,
) -> ScalarVar<'t, B> {
    let tape = p.tape();
    let d = p - &tape.tensor(target.contiguous());
    (&(&d * &d) * &tape.tensor(weights.contiguous())).sum()
}

/// The same loss for the same elements held as a matrix.
fn matrix_loss<'t, B: Kernels<f32>>(
    p: &MatrixVar<'t, B>,
    target: &Matrix<f32, B>,
    weights: &Matrix<f32, B>,
) -> ScalarVar<'t, B> {
    let tape = p.tape();
    let d = p - &tape.matrix(target.to_backend::<B>());
    (&(&d * &d) * &tape.matrix(weights.to_backend::<B>())).sum()
}

/// Run `rule_t` on a `[2, 3, 4]` tensor and `rule_m` on the `6 × 4` matrix of
/// the same elements, asserting the trajectories agree at every step.
fn trajectories_agree<B: Kernels<f32>, RT, RM>(mut rule_t: RT, mut rule_m: RM, steps: usize)
where
    RT: Rule<Tensor<f32, B>>,
    RM: Rule<Matrix<f32, B>>,
{
    let start = convert::<f32>(&filled(&[2, 3, 4], 1, -1.0, 1.0));
    let target = convert::<f32>(&filled(&[2, 3, 4], 2, -1.0, 1.0)).to_backend::<B>();
    let weights = convert::<f32>(&filled(&[2, 3, 4], 3, 0.5, 1.5)).to_backend::<B>();
    let as_matrix = |t: &Tensor<f32, B>| t.contiguous().reshape(&[6, 4]).into_matrix();

    let mut tensor = start.to_backend::<B>();
    let mut matrix = as_matrix(&tensor);
    let (target_m, weights_m) = (as_matrix(&target), as_matrix(&weights));
    let mut first_loss = None;
    let mut last_loss = 0.0;
    for step in 0..steps {
        let tape = Tape::<B>::new();
        let p = tape.tensor(tensor.contiguous());
        let loss = tensor_loss(&p, &target, &weights);
        loss.backward();
        rule_t.update(&mut tensor, &p.grad());

        let tape = Tape::<B>::new();
        let q = tape.matrix(matrix.to_backend::<B>());
        let reference = matrix_loss(&q, &target_m, &weights_m);
        reference.backward();
        rule_m.update(&mut matrix, &q.grad());

        assert_eq!(tensor.shape(), [2, 3, 4]);
        let (got, want) = (tensor.to_vec(), matrix.as_slice().to_vec());
        for (k, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-6,
                "step {step}, element {k}: {g} vs {w}"
            );
        }
        last_loss = *loss.value();
        first_loss.get_or_insert(last_loss);
    }
    assert!(
        last_loss < 0.1 * first_loss.unwrap(),
        "the loss fell from {first_loss:?} to {last_loss}"
    );
}

#[test]
fn adam_trains_a_tensor_along_the_matrix_trajectory() {
    trajectories_agree::<Host, _, _>(Adam::new(0.05), Adam::new(0.05), 200);
}

#[test]
fn momentum_and_sgd_train_a_tensor_along_the_matrix_trajectory() {
    trajectories_agree::<Host, _, _>(Momentum::new(0.05, 0.9), Momentum::new(0.05, 0.9), 100);
    trajectories_agree::<Host, _, _>(Sgd::new(0.2), Sgd::new(0.2), 100);
}

#[test]
fn minimize_drives_a_tensor_parameter() {
    let target = convert::<f32>(&filled(&[2, 2, 3], 4, -1.0, 1.0));
    let mut parameters = Tensor::<f32>::zeros(&[2, 2, 3]);
    let mut rule = Adam::new(0.1);
    let loss = minimize(&mut parameters, &mut rule, 300, |p, _| {
        let d = p - &p.tape().tensor(target.clone());
        (&d * &d).mean_axes(.., false).sum()
    });
    assert!(loss < 1e-4, "final loss {loss}");
    for (p, t) in parameters.as_slice().iter().zip(target.as_slice()) {
        assert!((p - t).abs() < 1e-2);
    }
}

#[test]
fn a_tensor_parameter_reads_its_gradient_from_a_view() {
    // The gradient is the first half of the last axis of a packed one.
    let mut weights = Tensor::<f32>::ones(&[2, 3, 4]);
    let packed = Tensor::from_vec(
        &[2, 3, 8],
        (0..48)
            .map(|i| if i % 8 < 4 { 1.0f32 } else { 9.0 })
            .collect::<Vec<_>>(),
    );
    Sgd::new(0.5).update(&mut weights, &packed.narrow(-1, 0, 4));
    assert_eq!(weights.to_vec(), [0.5; 24]);
}

// ---- a composed network --------------------------------------------------------

/// A one-layer, two-head attention-shaped block with no batched product:
/// `x·W` split into query, key and value, the heads separated by a reshape
/// and a permutation, a gated mix of them, the heads merged back and a bias
/// broadcast over every position.
fn block<'t, B: Kernels<f32>>(
    x: &TensorVar<'t, B>,
    w: &TensorVar<'t, B>,
    bias: &TensorVar<'t, B>,
) -> TensorVar<'t, B> {
    let (batch, steps, width, heads) = (2, 4, 8, 2);
    let projected = x
        .to_matrix(batch * steps, width)
        .matmul(&w.to_matrix(width, 3 * width))
        .to_tensor(&[batch, steps, 3 * width]);
    let split = |t: &TensorVar<'t, B>| {
        t.reshape(&[batch, steps, heads, width / heads])
            .permute(&[0, 2, 1, 3])
    };
    let [q, k, v] = <[_; 3]>::try_from(projected.chunk(-1, 3)).ok().unwrap();
    let (q, k, v) = (split(&q), split(&k), split(&v));
    // A per-head score over the feature axis, normalized over the steps.
    let score = (&q * &k).sum_axes(-1, true).tanh();
    let centred = &score - &score.max_axes(2, true);
    let mixed = &centred.exp() * &v;
    let merged = mixed.permute(&[0, 2, 1, 3]).reshape(&[batch, steps, width]);
    &merged + bias
}

fn train_block<B: Kernels<f32>>() -> Vec<f32> {
    let x = convert::<f32>(&filled(&[2, 4, 8], 1, -1.0, 1.0)).to_backend::<B>();
    let target = convert::<f32>(&filled(&[2, 4, 8], 2, -0.5, 0.5)).to_backend::<B>();
    let mut w = convert::<f32>(&filled(&[8, 24], 3, -0.3, 0.3)).to_backend::<B>();
    let mut bias = Tensor::<f32, B>::zeros(&[8]);
    let (mut rule_w, mut rule_b) = (Adam::new(0.02), Adam::new(0.02));
    let mut losses = Vec::new();
    for _ in 0..150 {
        let tape = Tape::<B>::new();
        let (wv, bv) = (tape.tensor(w.contiguous()), tape.tensor(bias.contiguous()));
        let out = block(&tape.tensor(x.contiguous()), &wv, &bv);
        let error = &out - &tape.tensor(target.contiguous());
        let loss = (&error * &error).mean();
        loss.backward();
        rule_w.update(&mut w, &wv.grad());
        rule_b.update(&mut bias, &bv.grad());
        losses.push(*loss.value());
    }
    losses
}

#[test]
fn an_attention_shaped_block_trains() {
    let losses = train_block::<Host>();
    let (first, last) = (losses[0], *losses.last().unwrap());
    assert!(last < 0.25 * first, "loss {first} → {last}");
}

// ---- the two backends ----------------------------------------------------------

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    use super::*;
    use tensorcrate::metal::MetalElement;
    use tensorcrate::tensors::Metal;

    fn agree<E: MetalElement>(tolerance: f64)
    where
        Metal: Kernels<E>,
        Host: Kernels<E>,
    {
        for op in Op::all() {
            let inputs = op.inputs().iter().map(convert::<E>).collect::<Vec<_>>();
            let seed = |shape: &[usize]| convert::<E>(&seed_for(shape));
            let (want, _) = reverse::<Host, E>(op, &inputs, seed);
            let device = inputs
                .iter()
                .map(|x| x.to_backend::<Metal>())
                .collect::<Vec<_>>();
            let (got, _) =
                reverse::<Metal, E>(op, &device, |shape| seed(shape).to_backend::<Metal>());
            for (which, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g.shape(), w.shape(), "{op:?}: gradient shape");
                if device[0].as_vector().is_device_resident() && !g.is_empty() {
                    assert!(
                        g.as_vector().is_device_resident(),
                        "{op:?}: the gradient stays on the device"
                    );
                }
                assert_close(
                    &to_f64(g.as_slice()),
                    &to_f64(w.as_slice()),
                    tolerance,
                    &format!("{op:?} {} ∂/∂input{which}", std::any::type_name::<E>()),
                );
            }
        }
    }

    #[test]
    fn every_operation_agrees_between_the_backends_in_f32() {
        agree::<f32>(1e-4);
    }

    #[test]
    fn every_operation_agrees_between_the_backends_in_f16() {
        agree::<f16>(1e-2);
    }

    #[test]
    fn every_operation_agrees_between_the_backends_in_bf16() {
        agree::<bf16>(5e-2);
    }

    #[test]
    fn empty_and_unit_extents_differentiate_on_metal() {
        empty_and_unit_extents::<Metal>();
    }

    #[test]
    fn optimizers_train_a_resident_tensor_along_the_matrix_trajectory() {
        trajectories_agree::<Metal, _, _>(Adam::new(0.05), Adam::new(0.05), 100);
        trajectories_agree::<Metal, _, _>(Momentum::new(0.05, 0.9), Momentum::new(0.05, 0.9), 50);
    }

    #[test]
    fn the_attention_shaped_block_trains_alike_on_metal() {
        let (host, device) = (train_block::<Host>(), train_block::<Metal>());
        for (step, (h, d)) in host.iter().zip(&device).enumerate() {
            assert!(
                (h - d).abs() <= 1e-3 * h.abs().max(1.0),
                "step {step}: {h} vs {d}"
            );
        }
        assert!(*device.last().unwrap() < 0.25 * device[0]);
    }
}
