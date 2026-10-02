//! Fused elementwise programs.
//!
//! The contract is that fusion changes where intermediates live and nothing
//! else, so the oracle throughout is the unfused computation: on the host a
//! fused program must agree with it bit for bit, and on Metal — which builds
//! with fast math — within the tolerance the unfused kernels already have.

use half::{bf16, f16};
use tensorcrate::optim::{AdaGrad, Adam, Momentum, Parameter, RmsProp, Rule, Sgd};
use tensorcrate::tensors::fused::{
    self, Algebra, Builder, DType, Fusable, Instr, Mode, Program, ProgramError, Remap, Value,
};
use tensorcrate::tensors::{Analytic, BinaryOp, Compare, Host, Kernels, Matrix, Vector};

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

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn uniform(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * (self.next() as f32 / (1u64 << 31) as f32)
    }

    fn vector(&mut self, len: usize, low: f32, high: f32) -> Vec<f32> {
        (0..len).map(|_| self.uniform(low, high)).collect()
    }
}

fn assert_bits_eq(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: lengths differ");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.to_bits() == e.to_bits(),
            "{what}: element {i} is {a:?} ({:#010x}), expected {e:?} ({:#010x})",
            a.to_bits(),
            e.to_bits()
        );
    }
}

/// Lengths that straddle the SIMD lane width, the SIMD dispatch threshold and
/// the interpreter's 1024-element tiles.
const LENGTHS: [usize; 9] = [0, 1, 3, 15, 16, 17, 1023, 1025, 3001];

// ---- the optimizers, against their unfused definitions ----------------------

/// The update rules exactly as they were written before fusion, one kernel per
/// operation. The fused rules must reproduce these bit for bit on the host.
mod reference {
    use tensorcrate::optim::Parameter;

    pub fn sgd<P: Parameter<Elem = f32>>(rate: f32, p: &mut P, g: &P) {
        *p = p.subtract(&g.scale(rate));
    }

    pub fn momentum<P: Parameter<Elem = f32>>(
        rate: f32,
        mu: f32,
        nesterov: bool,
        velocity: &mut Option<P>,
        p: &mut P,
        g: &P,
    ) {
        let v = match velocity.take() {
            Some(previous) => previous.scale(mu).add(g),
            None => g.duplicate(),
        };
        let step = if nesterov {
            g.add(&v.scale(mu))
        } else {
            v.duplicate()
        };
        *p = p.subtract(&step.scale(rate));
        *velocity = Some(v);
    }

    pub fn adagrad<P: Parameter<Elem = f32>>(
        rate: f32,
        eps: f32,
        total: &mut Option<P>,
        p: &mut P,
        g: &P,
    ) {
        let squared = g.multiply(g);
        let t = match total.take() {
            Some(previous) => previous.add(&squared),
            None => squared,
        };
        let step = g.divide(&t.sqrt().shift(eps));
        *p = p.subtract(&step.scale(rate));
        *total = Some(t);
    }

    pub fn rmsprop<P: Parameter<Elem = f32>>(
        rate: f32,
        decay: f32,
        eps: f32,
        mean_square: &mut Option<P>,
        p: &mut P,
        g: &P,
    ) {
        let squared = g.multiply(g).scale(1.0 - decay);
        let ms = match mean_square.take() {
            Some(previous) => previous.scale(decay).add(&squared),
            None => squared,
        };
        let step = g.divide(&ms.sqrt().shift(eps));
        *p = p.subtract(&step.scale(rate));
        *mean_square = Some(ms);
    }

    pub struct Adam<P> {
        pub first: Option<P>,
        pub second: Option<P>,
        pub steps: u32,
    }

    pub fn adam<P: Parameter<Elem = f32>>(
        (rate, b1, b2, eps): (f32, f32, f32, f32),
        state: &mut Adam<P>,
        p: &mut P,
        g: &P,
    ) {
        state.steps += 1;
        let first = match state.first.take() {
            Some(previous) => previous.scale(b1).add(&g.scale(1.0 - b1)),
            None => g.scale(1.0 - b1),
        };
        let squared = g.multiply(g);
        let second = match state.second.take() {
            Some(previous) => previous.scale(b2).add(&squared.scale(1.0 - b2)),
            None => squared.scale(1.0 - b2),
        };
        let c1 = 1.0 - b1.powi(state.steps as i32);
        let c2 = 1.0 - b2.powi(state.steps as i32);
        let step = first
            .scale(c1.recip())
            .divide(&second.scale(c2.recip()).sqrt().shift(eps));
        *p = p.subtract(&step.scale(rate));
        state.first = Some(first);
        state.second = Some(second);
    }
}

/// Drive every rule and its reference side by side for several steps, with a
/// fresh gradient each step: under [`Algebra::Exact`] they must agree bit for
/// bit after each one, and under [`Algebra::Reassociate`] — which may regroup
/// the update's arithmetic — to within rounding.
fn rules_match_their_references<P: Parameter<Elem = f32>>(
    start: &P,
    gradients: &[P],
    bits: impl Fn(&P) -> Vec<f32>,
) {
    fused::with_algebra(Algebra::Exact, || {
        rules_agree(start, gradients, &|a: &P, b: &P, what: &str| {
            assert_bits_eq(&bits(a), &bits(b), what)
        })
    });
    fused::with_algebra(Algebra::Reassociate, || {
        rules_agree(start, gradients, &|a: &P, b: &P, what: &str| {
            for (i, (x, y)) in bits(a).iter().zip(bits(b)).enumerate() {
                assert!(
                    (x - y).abs() <= 1e-5 * (1.0 + y.abs()),
                    "{what}, reassociated: element {i} is {x}, expected {y}"
                );
            }
        })
    });
}

fn rules_agree<P: Parameter<Elem = f32>>(start: &P, gradients: &[P], same: &dyn Fn(&P, &P, &str)) {
    let (mut fused, mut unfused) = (start.duplicate(), start.duplicate());
    let mut rule = Sgd::new(0.05);
    for g in gradients {
        rule.update(&mut fused, g);
        reference::sgd(0.05, &mut unfused, g);
        same(&fused, &unfused, "sgd");
    }

    for nesterov in [false, true] {
        let (mut fused, mut unfused) = (start.duplicate(), start.duplicate());
        let mut rule = if nesterov {
            Momentum::nesterov(0.05, 0.9)
        } else {
            Momentum::new(0.05, 0.9)
        };
        let mut velocity = None;
        for g in gradients {
            rule.update(&mut fused, g);
            reference::momentum(0.05, 0.9, nesterov, &mut velocity, &mut unfused, g);
            same(&fused, &unfused, "momentum");
        }
    }

    let (mut fused, mut unfused) = (start.duplicate(), start.duplicate());
    let mut rule = AdaGrad::new(0.1);
    let mut total = None;
    for g in gradients {
        rule.update(&mut fused, g);
        reference::adagrad(0.1, 1e-8, &mut total, &mut unfused, g);
        same(&fused, &unfused, "adagrad");
    }

    let (mut fused, mut unfused) = (start.duplicate(), start.duplicate());
    let mut rule = RmsProp::new(0.01);
    let mut mean_square = None;
    for g in gradients {
        rule.update(&mut fused, g);
        reference::rmsprop(0.01, 0.9, 1e-8, &mut mean_square, &mut unfused, g);
        same(&fused, &unfused, "rmsprop");
    }

    let (mut fused, mut unfused) = (start.duplicate(), start.duplicate());
    let mut rule = Adam::new(0.001);
    let mut state = reference::Adam {
        first: None,
        second: None,
        steps: 0,
    };
    for g in gradients {
        rule.update(&mut fused, g);
        reference::adam((0.001, 0.9, 0.999, 1e-8), &mut state, &mut unfused, g);
        same(&fused, &unfused, "adam");
    }

    // `reset` discards the moments, and the next step is a first step again.
    rule.reset();
    let mut fresh = start.duplicate();
    let mut state = reference::Adam {
        first: None,
        second: None,
        steps: 0,
    };
    let mut expected = start.duplicate();
    rule.update(&mut fresh, &gradients[0]);
    reference::adam(
        (0.001, 0.9, 0.999, 1e-8),
        &mut state,
        &mut expected,
        &gradients[0],
    );
    same(&fresh, &expected, "adam after reset");
}

#[test]
fn fused_rules_reproduce_the_unfused_updates_bit_for_bit_on_vectors() {
    let mut rng = Lcg(7);
    for len in LENGTHS {
        let start = Vector::new(rng.vector(len, -2.0, 2.0));
        let gradients: Vec<_> = (0..4)
            .map(|_| Vector::new(rng.vector(len, -5.0, 5.0)))
            .collect();
        rules_match_their_references(&start, &gradients, |v: &Vector<f32>| v.to_vec());
    }
}

#[test]
fn fused_rules_reproduce_the_unfused_updates_bit_for_bit_on_matrices() {
    let mut rng = Lcg(11);
    for (rows, cols) in [(1, 1), (3, 5), (37, 29), (64, 64)] {
        let start = Matrix::from_flat(rows, cols, rng.vector(rows * cols, -1.0, 1.0));
        let gradients: Vec<_> = (0..3)
            .map(|_| Matrix::from_flat(rows, cols, rng.vector(rows * cols, -3.0, 3.0)))
            .collect();
        rules_match_their_references(&start, &gradients, |m: &Matrix<f32>| m.as_slice().to_vec());
    }
}

#[test]
fn fused_rules_reproduce_the_unfused_updates_bit_for_bit_on_scalars() {
    let gradients = [0.5f32, -1.25, 3.0, 1e-3];
    rules_match_their_references(&0.75f32, &gradients, |x: &f32| vec![*x]);
}

// ---- arbitrary programs, fused against unfused --------------------------------

const UNARIES: [Analytic; 6] = [
    Analytic::Sqrt,
    Analytic::Exp,
    Analytic::Sin,
    Analytic::Tanh,
    Analytic::Ln,
    Analytic::Arctan,
];

const BINARIES: [BinaryOp; 5] = [
    BinaryOp::Add,
    BinaryOp::Sub,
    BinaryOp::Mul,
    BinaryOp::Div,
    BinaryOp::Rem,
];

/// A random program over `inputs` `f32` inputs with `outputs` outputs.
fn random_program(rng: &mut Lcg, inputs: usize, outputs: usize, metal: bool) -> Program {
    loop {
        let mut b = Builder::new();
        let mut values: Vec<Value> = (0..inputs).map(|_| b.input(DType::F32)).collect();
        for _ in 0..rng.below(24) + 2 {
            let pick = |rng: &mut Lcg, values: &[Value]| values[rng.below(values.len())];
            let value = match rng.below(10) {
                0 => b.constant(rng.uniform(-2.0, 2.0)),
                1 | 2 => {
                    let op = UNARIES[rng.below(UNARIES.len())];
                    let a = pick(rng, &values);
                    b.unary(op, a)
                }
                3 => {
                    let op = Compare::ALL[rng.below(Compare::ALL.len())];
                    let (x, y) = (pick(rng, &values), pick(rng, &values));
                    b.compare(op, x, y)
                }
                _ => {
                    let ops = if metal { &BINARIES[..4] } else { &BINARIES[..] };
                    let op = ops[rng.below(ops.len())];
                    let (x, y) = (pick(rng, &values), pick(rng, &values));
                    b.binary(op, x, y)
                }
            };
            values.push(value);
        }
        // Store the most recent values, which usually depend on the most.
        for k in 0..outputs {
            let value = values[values.len() - 1 - k.min(values.len() - 1)];
            b.output(value, DType::F32);
        }
        if let Ok(program) = b.build() {
            return program;
        }
    }
}

fn run_on<B: Kernels>(
    program: &Program,
    shape: (usize, usize),
    inputs: &[Vector<f32>],
) -> Vec<Vec<f32>> {
    let inputs: Vec<Vector<f32, B>> = inputs.iter().map(|v| v.to_backend::<B>()).collect();
    let refs: Vec<&dyn Fusable<B>> = inputs.iter().map(|v| v as &dyn Fusable<B>).collect();
    program
        .run(shape, &refs, &mut [])
        .into_iter()
        .map(|output| output.into_vector::<f32>().to_vec())
        .collect()
}

#[test]
fn random_programs_agree_with_their_unfused_evaluation_bit_for_bit() {
    let mut rng = Lcg(2026);
    for case in 0..120 {
        let inputs = rng.below(4) + 1;
        let outputs = rng.below(3) + 1;
        let program = random_program(&mut rng, inputs, outputs, false);
        let len = LENGTHS[case % LENGTHS.len()];
        let values: Vec<Vector<f32>> = (0..inputs)
            .map(|_| Vector::new(rng.vector(len, -3.0, 3.0)))
            .collect();

        let fused = run_on::<Host>(&program, (1, len), &values);
        let unfused = fused::with_mode(Mode::Unfused, || {
            run_on::<Host>(&program, (1, len), &values)
        });
        for (k, (f, u)) in fused.iter().zip(&unfused).enumerate() {
            assert_bits_eq(
                f,
                u,
                &format!("case {case}, output {k}, len {len}\n{program}"),
            );
        }
    }
}

#[test]
fn remapped_loads_read_transposed_and_broadcast_operands() {
    // out[r][c] = a[r][c] + bᵀ[r][c] · row[c] − col[r], over a 3×4 space.
    let mut b = Builder::<f32>::new();
    let a = b.input(DType::F32);
    let t = b.input_remapped(DType::F32, Remap::Transpose);
    let row = b.input_remapped(DType::F32, Remap::Row);
    let col = b.input_remapped(DType::F32, Remap::Column);
    let product = b.mul(t, row);
    let sum = b.add(a, product);
    let out = b.sub(sum, col);
    b.output(out, DType::F32);
    let program = b.build().unwrap();

    let a = Matrix::from_flat(3, 4, (0..12).map(|i| i as f32).collect::<Vec<_>>());
    let bt = Matrix::from_flat(4, 3, (0..12).map(|i| 0.5 * i as f32).collect::<Vec<_>>());
    let row = Vector::new([1.0f32, 2.0, 3.0, 4.0]);
    let col = Vector::new([10.0f32, 20.0, 30.0]);

    let expected: Vec<f32> = (0..3)
        .flat_map(|r| {
            let (a, bt, row, col) = (&a, &bt, &row, &col);
            (0..4)
                .map(move |c| a.as_slice()[r * 4 + c] + bt.as_slice()[c * 3 + r] * row[c] - col[r])
        })
        .collect();

    let inputs: [&dyn Fusable<Host>; 4] = [&a, &bt, &row, &col];
    let fused = program
        .run((3, 4), &inputs, &mut [])
        .remove(0)
        .into_matrix::<f32>();
    assert_eq!(fused.shape(), (3, 4));
    assert_bits_eq(fused.as_slice(), &expected, "remapped");

    let unfused = fused::with_mode(Mode::Unfused, || {
        program
            .run((3, 4), &inputs, &mut [])
            .remove(0)
            .into_matrix::<f32>()
    });
    assert_bits_eq(unfused.as_slice(), &expected, "remapped, unfused");
}

#[test]
fn remaps_survive_tile_boundaries() {
    // A tall space, so tiles start mid-row, with every remap in one program.
    let mut rng = Lcg(5);
    let (rows, cols) = (300, 7);
    let mut b = Builder::<f32>::new();
    let t = b.input_remapped(DType::F32, Remap::Transpose);
    let row = b.input_remapped(DType::F32, Remap::Row);
    let col = b.input_remapped(DType::F32, Remap::Column);
    let x = b.mul(t, row);
    let y = b.add(x, col);
    b.output(y, DType::F32);
    let program = b.build().unwrap();

    let t = Matrix::from_flat(cols, rows, rng.vector(rows * cols, -1.0, 1.0));
    let row = Vector::new(rng.vector(cols, -1.0, 1.0));
    let col = Vector::new(rng.vector(rows, -1.0, 1.0));
    let inputs: [&dyn Fusable<Host>; 3] = [&t, &row, &col];
    let fused = program
        .run((rows, cols), &inputs, &mut [])
        .remove(0)
        .into_vector::<f32>();
    let unfused = fused::with_mode(Mode::Unfused, || {
        program
            .run((rows, cols), &inputs, &mut [])
            .remove(0)
            .into_vector::<f32>()
    });
    assert_bits_eq(fused.as_slice(), unfused.as_slice(), "tall remap");
}

#[test]
fn in_place_tensors_are_read_then_overwritten() {
    // x ← x·2 + y and y ← x − y, both from the *old* x.
    let mut b = Builder::new();
    let x = b.update(DType::F32);
    let y = b.update(DType::F32);
    let doubled = b.scale(x, 2.0);
    let new_x = b.add(doubled, y);
    let new_y = b.sub(x, y);
    b.set(0, new_x);
    b.set(1, new_y);
    let program = b.build().unwrap();
    assert_eq!(program.updated(), 2);
    assert_eq!(program.fresh_outputs(), 0);

    let mut x = Vector::new((0..2000).map(|i| i as f32).collect::<Vec<_>>());
    let mut y = Vector::new(vec![1.0f32; 2000]);
    let (old_x, old_y) = (x.to_vec(), y.to_vec());
    let outputs = program.run((1, 2000), &[], &mut [&mut x, &mut y]);
    assert!(outputs.is_empty());
    for i in 0..2000 {
        assert_eq!(x[i], old_x[i] * 2.0 + old_y[i]);
        assert_eq!(y[i], old_x[i] - old_y[i]);
    }
}

#[test]
fn compact_types_widen_on_load_and_narrow_on_store() {
    let mut b = Builder::<f32>::new();
    let h = b.input(DType::F16);
    let bf = b.input(DType::Bf16);
    let sum = b.add(h, bf);
    b.output(sum, DType::F32);
    b.output(sum, DType::F16);
    b.output(sum, DType::Bf16);
    let program = b.build().unwrap();

    let mut rng = Lcg(3);
    let a: Vec<f16> = rng
        .vector(100, -50.0, 50.0)
        .into_iter()
        .map(f16::from_f32)
        .collect();
    let c: Vec<bf16> = rng
        .vector(100, -50.0, 50.0)
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    let av = Vector::new(a.clone());
    let cv = Vector::new(c.clone());
    let inputs: [&dyn Fusable<Host>; 2] = [&av, &cv];

    for mode in [Mode::Fused, Mode::Unfused] {
        let mut outputs =
            fused::with_mode(mode, || program.run((1, 100), &inputs, &mut [])).into_iter();
        let wide = outputs.next().unwrap().into_vector::<f32>();
        let half = outputs.next().unwrap().into_vector::<f16>();
        let brain = outputs.next().unwrap().into_vector::<bf16>();
        for i in 0..100 {
            let exact = a[i].to_f32() + c[i].to_f32();
            assert_eq!(wide[i].to_bits(), exact.to_bits());
            assert_eq!(half[i].to_bits(), f16::from_f32(exact).to_bits());
            assert_eq!(brain[i].to_bits(), bf16::from_f32(exact).to_bits());
        }
    }
}

// ---- debugging ------------------------------------------------------------------

#[test]
fn a_program_disassembles() {
    let mut b = Builder::new();
    let x = b.input(DType::F32);
    let y = b.unary(Analytic::Sqrt, x);
    let z = b.shift(y, 1.0);
    b.output(z, DType::Bf16);
    let program = b.build().unwrap();
    assert_eq!(
        program.to_string(),
        "program(f32) -> (bf16), 0 in place, 2 registers\n\
         \x20   0: r0 = in0\n\
         \x20   1: r0 = Sqrt r0\n\
         \x20   2: r1 = 1.0\n\
         \x20   3: r0 = add r0, r1\n\
         \x20   4: out0 = r0\n"
    );
}

#[test]
fn a_trace_exposes_every_intermediate() {
    let mut b = Builder::new();
    let x = b.input(DType::F32);
    let y = b.mul(x, x);
    let z = b.shift(y, 1.0);
    b.output(z, DType::F32);
    let program = b.build().unwrap();

    let v = Vector::new([1.0f32, 2.0, 3.0]);
    let trace = program.trace::<Host>((1, 3), &[&v]);
    assert_eq!(trace.len(), program.code().len());
    let values: Vec<Option<Vec<f32>>> = trace
        .into_iter()
        .map(|step| step.map(|m| m.as_slice().to_vec()))
        .collect();
    assert_eq!(
        values,
        [
            Some(vec![1.0, 2.0, 3.0]),
            Some(vec![1.0, 4.0, 9.0]),
            Some(vec![1.0, 1.0, 1.0]),
            Some(vec![2.0, 5.0, 10.0]),
            None,
        ]
    );
}

#[test]
fn turning_fusion_off_is_scoped_and_restored() {
    assert_eq!(fused::mode(), Mode::Fused);
    fused::with_mode(Mode::Unfused, || {
        assert_eq!(fused::mode(), Mode::Unfused);
        fused::with_mode(Mode::Fused, || assert_eq!(fused::mode(), Mode::Fused));
        assert_eq!(fused::mode(), Mode::Unfused);
    });
    assert_eq!(fused::mode(), Mode::Fused);

    // Restored on unwind, too.
    let _ = std::panic::catch_unwind(|| fused::with_mode(Mode::Unfused, || panic!("inside")));
    assert_eq!(fused::mode(), Mode::Fused);
}

// ---- validation -----------------------------------------------------------------

#[test]
fn malformed_programs_are_rejected() {
    use Instr::*;
    let f = DType::F32;
    let check = |code: Vec<Instr>, inputs: Vec<DType>, outputs: Vec<DType>, updated, err| {
        assert_eq!(Program::new(code, inputs, outputs, updated), Err(err));
    };

    check(
        vec![Store { src: 0, output: 0 }],
        vec![],
        vec![f],
        0,
        ProgramError::Undefined { at: 0, reg: 0 },
    );
    check(
        vec![
            Load {
                dst: 0,
                input: 1,
                remap: Remap::Identity,
            },
            Store { src: 0, output: 0 },
        ],
        vec![f],
        vec![f],
        0,
        ProgramError::BadInput { at: 0 },
    );
    check(
        vec![Const {
            dst: 16,
            value: 1.0,
        }],
        vec![],
        vec![f],
        0,
        ProgramError::BadRegister { at: 0 },
    );
    check(
        vec![Const { dst: 0, value: 1.0 }],
        vec![],
        vec![f],
        0,
        ProgramError::OutputNotStoredOnce { output: 0 },
    );
    check(
        vec![
            Const { dst: 0, value: 1.0 },
            Store { src: 0, output: 0 },
            Store { src: 0, output: 0 },
        ],
        vec![],
        vec![f],
        0,
        ProgramError::OutputNotStoredOnce { output: 0 },
    );
    // An in-place tensor read through a remap would race other threads.
    check(
        vec![
            Load {
                dst: 0,
                input: 0,
                remap: Remap::Transpose,
            },
            Store { src: 0, output: 0 },
        ],
        vec![f],
        vec![f],
        1,
        ProgramError::RemappedUpdate { at: 0 },
    );
    // ... and one read after its store would see the new value.
    check(
        vec![
            Const { dst: 0, value: 1.0 },
            Store { src: 0, output: 0 },
            Load {
                dst: 1,
                input: 0,
                remap: Remap::Identity,
            },
        ],
        vec![f],
        vec![f],
        1,
        ProgramError::LoadAfterStore { at: 2 },
    );
    check(vec![], vec![], vec![], 0, ProgramError::NoOutputs);
    check(
        vec![Const { dst: 0, value: 1.0 }, Store { src: 0, output: 0 }],
        vec![DType::F16],
        vec![f],
        1,
        ProgramError::UpdateTypeMismatch { slot: 0 },
    );
}

#[test]
fn the_builder_reuses_registers_once_values_die() {
    // A forty-step chain only ever has two values live.
    let mut b = Builder::new();
    let mut x = b.input(DType::F32);
    for _ in 0..40 {
        x = b.shift(x, 1.0);
    }
    b.output(x, DType::F32);
    let program = b.build().unwrap();
    assert!(program.registers() <= 2, "{program}");

    // Seventeen shared values, summed in one order and multiplied in the
    // other: whichever chain runs first, every value is still owed to the
    // other one, so all seventeen are live at once and cannot fit.
    let build = |algebra: Algebra| {
        let mut b = Builder::<f32>::new();
        let inputs: Vec<Value> = (0..16).map(|_| b.input(DType::F32)).collect();
        let mut values: Vec<Value> = inputs.iter().map(|&x| b.unary(Analytic::Sqrt, x)).collect();
        let both = b.mul(inputs[0], inputs[1]);
        values.push(b.unary(Analytic::Sqrt, both));
        let mut sum = values[0];
        for &value in &values[1..] {
            sum = b.add(sum, value);
        }
        let mut product = values[16];
        for &value in values[..16].iter().rev() {
            product = b.mul(product, value);
        }
        b.output(sum, DType::F32);
        b.output(product, DType::F32);
        b.build_with(&fused::CostModel::BALANCED, algebra)
    };
    assert_eq!(build(Algebra::Exact), Err(ProgramError::TooManyRegisters));
    // Reassociating puts both chains in one canonical order, so each value can
    // be added and multiplied in as soon as it is computed.
    let program = build(Algebra::Reassociate).expect("reassociated, it fits");
    assert!(program.registers() <= fused::REGISTERS, "{program}");
}

#[test]
#[should_panic(expected = "needs 4")]
fn a_short_input_is_caught_before_running() {
    let mut b = Builder::<f32>::new();
    let x = b.input(DType::F32);
    b.output(x, DType::F32);
    let program = b.build().unwrap();
    let short = Vector::new([1.0f32; 3]);
    program.run::<Host>((2, 2), &[&short], &mut []);
}

// ---- work counts ------------------------------------------------------------------

#[cfg(feature = "counters")]
#[test]
fn an_adam_step_is_one_kernel_with_no_allocations() {
    use tensorcrate::counters;

    let mut rng = Lcg(1);
    let n = 4096;
    let gradient = Vector::new(rng.vector(n, -1.0, 1.0));
    let mut parameters = Vector::new(rng.vector(n, -1.0, 1.0));
    let mut rule = Adam::new(0.01);

    // The first step allocates the two moments.
    let ((), first) = counters::measure(|| rule.update(&mut parameters, &gradient));
    assert_eq!((first.kernels, first.allocations), (1, 2));

    // Every later one reads p, g, m, v and writes p, m, v in place: 7n floats.
    let ((), steady) = counters::measure(|| rule.update(&mut parameters, &gradient));
    assert_eq!(steady.kernels, 1);
    assert_eq!(steady.allocations, 0);
    assert_eq!(steady.bytes, 7 * n as u64 * 4);

    // Unfused, the step as written is fourteen kernels moving 33n floats. The
    // rule builds its programs when it is created, so that is when the algebra
    // is chosen.
    let mut exact = fused::with_algebra(Algebra::Exact, || Adam::new(0.01));
    exact.update(&mut parameters, &gradient);
    let ((), unfused) = counters::measure(|| {
        fused::with_mode(Mode::Unfused, || exact.update(&mut parameters, &gradient))
    });
    assert_eq!(unfused.kernels, 14);
    assert_eq!(unfused.bytes, 33 * n as u64 * 4);

    // Reassociated, the bias correction and the rate fold into one constant,
    // which saves a whole pass even before fusion.
    let ((), regrouped) = counters::measure(|| {
        fused::with_mode(Mode::Unfused, || rule.update(&mut parameters, &gradient))
    });
    assert!(regrouped.kernels < 14, "{regrouped:?}");
    assert!(regrouped.bytes < 33 * n as u64 * 4, "{regrouped:?}");
}

// ---- Metal against the host -----------------------------------------------------------

#[cfg(all(feature = "metal", target_os = "macos"))]
fn close(a: f32, e: f32) -> bool {
    if a.is_nan() || e.is_nan() {
        return a.is_nan() && e.is_nan();
    }
    if a.is_infinite() || e.is_infinite() {
        return a == e;
    }
    (a - e).abs() <= 1e-5 * (1.0 + e.abs())
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn assert_close(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(close(a, e), "{what}: element {i} is {a}, expected {e}");
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_programs_agree_with_the_host_within_tolerance() {
    let mut rng = Lcg(99);
    for case in 0..60 {
        let inputs = rng.below(4) + 1;
        let outputs = rng.below(3) + 1;
        let program = random_program(&mut rng, inputs, outputs, true);
        // Keep the analytic functions on their well-conditioned domains so the
        // comparison measures fusion, not the GPU's transcendental accuracy.
        let len = LENGTHS[case % LENGTHS.len()];
        let values: Vec<Vector<f32>> = (0..inputs)
            .map(|_| Vector::new(rng.vector(len, 0.5, 1.5)))
            .collect();
        let host = run_on::<Host>(&program, (1, len), &values);
        let metal = run_on::<Metal>(&program, (1, len), &values);
        let unfused = fused::with_mode(Mode::Unfused, || {
            run_on::<Metal>(&program, (1, len), &values)
        });
        for (k, ((m, u), h)) in metal.iter().zip(&unfused).zip(&host).enumerate() {
            for (i, ((&m, &u), &h)) in m.iter().zip(u).zip(h).enumerate() {
                // Fast-math shaders already differ from the host in places —
                // `tanh` of a large argument is NaN on the GPU, for one. Where
                // the unfused kernels do, fusion is held to *them*.
                let oracle = if close(u, h) { h } else { u };
                assert!(
                    close(m, oracle),
                    "case {case}, output {k}, element {i}: fused Metal {m}, unfused Metal {u}, \
                     host {h}\n{program}"
                );
            }
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_adam_tracks_the_host_oracle() {
    let mut rng = Lcg(42);
    let n = 5000;
    let start = rng.vector(n, -1.0, 1.0);
    let gradients: Vec<Vec<f32>> = (0..5).map(|_| rng.vector(n, -2.0, 2.0)).collect();

    let mut host = Vector::new(start.clone());
    let mut metal = Vector::new(start).to_backend::<Metal>();
    let (mut host_rule, mut metal_rule) = (Adam::new(0.01), Adam::new(0.01));
    for g in &gradients {
        host_rule.update(&mut host, &Vector::new(g.clone()));
        metal_rule.update(&mut metal, &Vector::new(g.clone()).to_backend::<Metal>());
    }
    assert_close(&metal.to_vec(), &host.to_vec(), "adam");
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_remaps_in_place_updates_and_compact_types() {
    let mut rng = Lcg(8);
    let (rows, cols) = (33, 70);

    let mut b = Builder::<f32>::new();
    let t = b.input_remapped(DType::F32, Remap::Transpose);
    let row = b.input_remapped(DType::Bf16, Remap::Row);
    let col = b.input_remapped(DType::F16, Remap::Column);
    let acc = b.update(DType::F32);
    let x = b.mul(t, row);
    let y = b.add(x, col);
    let z = b.add(acc, y);
    b.set(0, z);
    b.output(y, DType::F16);
    b.output(y, DType::Bf16);
    let program = b.build().unwrap();

    let t = Matrix::from_flat(cols, rows, rng.vector(rows * cols, -1.0, 1.0));
    let row: Vec<bf16> = rng
        .vector(cols, -1.0, 1.0)
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    let col: Vec<f16> = rng
        .vector(rows, -1.0, 1.0)
        .into_iter()
        .map(f16::from_f32)
        .collect();
    let acc = Matrix::from_flat(rows, cols, rng.vector(rows * cols, -1.0, 1.0));

    // Host.
    let (row_h, col_h) = (Vector::new(row.clone()), Vector::new(col.clone()));
    let mut acc_h = acc.clone();
    let mut host = program
        .run::<Host>((rows, cols), &[&t, &row_h, &col_h], &mut [&mut acc_h])
        .into_iter();
    let host_half = host.next().unwrap().into_vector::<f16>();
    let host_brain = host.next().unwrap().into_vector::<bf16>();

    // Metal.
    let t_m = t.to_backend::<Metal>();
    let (row_m, col_m) = (row_h.to_backend::<Metal>(), col_h.to_backend::<Metal>());
    let mut acc_m = acc.to_backend::<Metal>();
    let mut metal = program
        .run::<Metal>((rows, cols), &[&t_m, &row_m, &col_m], &mut [&mut acc_m])
        .into_iter();
    let metal_half = metal.next().unwrap().into_vector::<f16>();
    let metal_brain = metal.next().unwrap().into_vector::<bf16>();

    assert_close(acc_m.as_slice(), acc_h.as_slice(), "in place");
    let widen16 = |v: &[f16]| v.iter().map(|x| x.to_f32()).collect::<Vec<_>>();
    let widenb = |v: &[bf16]| v.iter().map(|x| x.to_f32()).collect::<Vec<_>>();
    // Narrowing can turn a last-bit difference into a one-step difference in
    // the compact type, so compare at the compact type's own precision.
    for (m, h) in widen16(metal_half.as_slice())
        .iter()
        .zip(widen16(host_half.as_slice()))
    {
        assert!(
            (m - h).abs() <= 1e-3 * (1.0 + h.abs()),
            "f16: {m} against {h}"
        );
    }
    for (m, h) in widenb(metal_brain.as_slice())
        .iter()
        .zip(widenb(host_brain.as_slice()))
    {
        assert!(
            (m - h).abs() <= 8e-3 * (1.0 + h.abs()),
            "bf16: {m} against {h}"
        );
    }
}

// ---- matrix products with an epilogue ---------------------------------------------

/// Three epilogues: a dense layer (`relu(xw + bias)`), a gated one that keeps
/// the product too (`[p, tanh(0.5·p + c)·g]`), and one that reads a whole
/// second matrix (`exp(−p²)·m`).
fn epilogues() -> Vec<(Program, Vec<Remap>)> {
    let mut layer = Builder::new();
    let p = layer.input(DType::F32);
    let bias = layer.input_remapped(DType::F32, Remap::Row);
    let shifted = layer.add(p, bias);
    let zero = layer.constant(0.0);
    let relu = layer.compare(Compare::Max, shifted, zero);
    layer.output(relu, DType::F32);

    let mut gated = Builder::new();
    let p = gated.input(DType::F32);
    let c = gated.input_remapped(DType::F32, Remap::Column);
    let g = gated.input_remapped(DType::F32, Remap::Transpose);
    let half = gated.scale(p, 0.5);
    let shifted = gated.add(half, c);
    let squashed = gated.unary(Analytic::Tanh, shifted);
    let out = gated.mul(squashed, g);
    gated.output(p, DType::F32);
    gated.output(out, DType::F32);

    let mut bump = Builder::new();
    let p = bump.input(DType::F32);
    let m = bump.input(DType::F32);
    let square = bump.mul(p, p);
    let negated = bump.scale(square, -1.0);
    let e = bump.unary(Analytic::Exp, negated);
    let out = bump.mul(e, m);
    bump.output(out, DType::F32);

    vec![
        (layer.build().unwrap(), vec![Remap::Row]),
        (
            gated.build().unwrap(),
            vec![Remap::Column, Remap::Transpose],
        ),
        (bump.build().unwrap(), vec![Remap::Identity]),
    ]
}

/// Shapes `(m, k, n)` on and off the 16- and 64-wide tile edges.
const PRODUCTS: [(usize, usize, usize); 7] = [
    (1, 1, 1),
    (3, 5, 2),
    (16, 16, 16),
    (17, 33, 15),
    (64, 64, 64),
    (65, 130, 67),
    (128, 7, 200),
];

fn epilogue_operands(
    rng: &mut Lcg,
    (m, k, n): (usize, usize, usize),
    remaps: &[Remap],
) -> (Matrix<f32>, Matrix<f32>, Vec<Vector<f32>>) {
    let a = Matrix::from_flat(m, k, rng.vector(m * k, -1.0, 1.0));
    let b = Matrix::from_flat(k, n, rng.vector(k * n, -1.0, 1.0));
    let inputs = remaps
        .iter()
        .map(|remap| Vector::new(rng.vector(remap.input_len((m, n)), -1.0, 1.0)))
        .collect();
    (a, b, inputs)
}

fn run_matmul_on<B: Kernels>(
    program: &Program,
    a: &Matrix<f32>,
    b: &Matrix<f32>,
    inputs: &[Vector<f32>],
) -> Vec<Vec<f32>> {
    let inputs: Vec<Vector<f32, B>> = inputs.iter().map(|v| v.to_backend::<B>()).collect();
    let refs: Vec<&dyn Fusable<B>> = inputs.iter().map(|v| v as &dyn Fusable<B>).collect();
    program
        .run_matmul(&a.to_backend::<B>(), &b.to_backend::<B>(), &refs)
        .into_iter()
        .map(|output| output.into_vector::<f32>().to_vec())
        .collect()
}

#[test]
fn a_matmul_epilogue_is_the_product_then_the_program_bit_for_bit() {
    let mut rng = Lcg(31);
    for (program, remaps) in epilogues() {
        for shape in PRODUCTS {
            let (a, b, inputs) = epilogue_operands(&mut rng, shape, &remaps);
            let fused = run_matmul_on::<Host>(&program, &a, &b, &inputs);
            let unfused = fused::with_mode(Mode::Unfused, || {
                run_matmul_on::<Host>(&program, &a, &b, &inputs)
            });
            // And by hand: the product materialized, then the program over it.
            let product = Vector::new(a.matmul(&b).data().to_vec());
            let mut operands = vec![product];
            operands.extend(inputs.iter().cloned());
            let by_hand = run_on::<Host>(&program, (shape.0, shape.2), &operands);
            for (k, ((f, u), h)) in fused.iter().zip(&unfused).zip(&by_hand).enumerate() {
                assert_bits_eq(f, u, &format!("{shape:?} output {k} against unfused"));
                assert_bits_eq(f, h, &format!("{shape:?} output {k} against by hand"));
            }
        }
    }
}

#[cfg(feature = "counters")]
#[test]
fn a_matmul_epilogue_is_one_kernel_that_never_writes_the_product() {
    use tensorcrate::counters;
    let (program, remaps) = epilogues().remove(0);
    let (m, k, n) = (32, 48, 64);
    let (a, b, inputs) = epilogue_operands(&mut Lcg(5), (m, k, n), &remaps);
    counters::reset();
    run_matmul_on::<Host>(&program, &a, &b, &inputs);
    let counts = counters::snapshot();
    assert_eq!(counts.kernels, 1);
    assert_eq!(counts.allocations, 1);
    // Both operands and the bias row read, the result written.
    assert_eq!(counts.bytes as usize, (m * k + k * n + n * m + m * n) * 4);
}

#[test]
#[should_panic(expected = "the product is read through")]
fn a_matmul_epilogue_reads_the_product_unremapped() {
    let mut b = Builder::new();
    let p = b.input_remapped(DType::F32, Remap::Transpose);
    b.output(p, DType::F32);
    let program = b.build().unwrap();
    let square = Matrix::from_flat(2, 2, vec![1.0f32; 4]);
    program.run_matmul(&square, &square, &[]);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_matmul_epilogues_agree_with_the_host_on_both_product_kernels() {
    let mut rng = Lcg(77);
    for tensorops in [true, false] {
        tensorcrate::metal::set_tensorops(tensorops);
        for (program, remaps) in epilogues() {
            for (m, k, n) in PRODUCTS {
                let (a, b, inputs) = epilogue_operands(&mut rng, (m, k, n), &remaps);
                let host = run_matmul_on::<Host>(&program, &a, &b, &inputs);
                let metal = run_matmul_on::<Metal>(&program, &a, &b, &inputs);
                let unfused = fused::with_mode(Mode::Unfused, || {
                    run_matmul_on::<Metal>(&program, &a, &b, &inputs)
                });
                // A sum of k products in a different order: scale the tolerance
                // with k.
                let tolerance = 1e-6 * (k as f32).sqrt() * 4.0;
                for (out, ((m_out, u_out), h_out)) in
                    metal.iter().zip(&unfused).zip(&host).enumerate()
                {
                    for (i, ((&x, &u), &h)) in m_out.iter().zip(u_out).zip(h_out).enumerate() {
                        assert!(
                            (x - u).abs() <= tolerance * (1.0 + u.abs())
                                && (x - h).abs() <= tolerance * (1.0 + h.abs()),
                            "tensorops {tensorops}, {m}×{k}×{n}, output {out}, element {i}: fused {x}, \
                             unfused {u}, host {h}\n{program}"
                        );
                    }
                }
            }
        }
    }
    tensorcrate::metal::set_tensorops(true);
}

// ---- uniforms ---------------------------------------------------------------------

/// `exp(x · (u − 1)) · u + c`: one uniform read directly, and folded with
/// constants into another constant.
fn uniform_program(u: f32) -> (Program, Program) {
    let build = |uniform: bool| {
        let mut b = Builder::<f32>::new();
        let x = b.input(DType::F32);
        let u = if uniform { b.uniform(u) } else { b.constant(u) };
        let one = b.constant(1.0);
        let less = b.sub(u, one);
        let scaled = b.mul(x, less);
        let grown = b.unary(Analytic::Exp, scaled);
        let weighted = b.mul(grown, u);
        let shifted = b.shift(weighted, 0.25);
        b.output(shifted, DType::F32);
        fused::with_algebra(Algebra::Exact, || b.build().unwrap())
    };
    (build(true), build(false))
}

#[test]
fn a_uniform_set_after_building_matches_a_program_built_with_it() {
    let x = Vector::new(Lcg(3).vector(100, -1.0, 1.0));
    let (mut program, _) = uniform_program(0.5);
    assert_eq!(program.uniforms(), [0.5]);
    for value in [0.5f32, 2.0, -3.25, 1.0, 0.0] {
        program.set_uniform(0, value);
        assert_eq!(program.uniforms(), [value]);
        let (_, constant) = uniform_program(value);
        let got = program.run_vectors(&[&x]).remove(0);
        let want = constant.run_vectors(&[&x]).remove(0);
        assert_bits_eq(got.as_slice(), want.as_slice(), &format!("uniform {value}"));
    }
}

#[test]
#[should_panic(expected = "uniform 1 out of 1")]
fn setting_a_missing_uniform_panics() {
    let (mut program, _) = uniform_program(0.5);
    program.set_uniform(1, 2.0);
}

#[test]
fn adam_coefficients_set_after_creation_take_effect() {
    let mut rng = Lcg(5);
    let gradients: Vec<_> = (0..3)
        .map(|_| Vector::new(rng.vector(64, -1.0, 1.0)))
        .collect();
    let start = Vector::new(rng.vector(64, -1.0, 1.0));

    let mut direct = Adam::new(0.01);
    direct.first_decay = 0.8;
    let mut changed = Adam::new(0.5);
    changed.rate = 0.01;
    changed.first_decay = 0.8;
    let (mut a, mut b) = (start.clone(), start.clone());
    for g in &gradients {
        direct.update(&mut a, g);
        changed.update(&mut b, g);
        assert_bits_eq(
            a.as_slice(),
            b.as_slice(),
            "coefficients changed after creation",
        );
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn a_uniform_set_after_building_reaches_the_gpu() {
    let host = Vector::new(Lcg(9).vector(4096, -1.0, 1.0));
    let x = host.to_backend::<Metal>();
    if !x.is_device_resident() {
        return;
    }
    let (mut program, _) = uniform_program(0.5);
    // Several runs per value, so the specialized kernel is compiled and reused
    // across values.
    for value in [0.5f32, 2.0, -3.25] {
        program.set_uniform(0, value);
        let want = program.run_vectors(&[&host]).remove(0);
        for _ in 0..3 {
            let got = program.run_vectors(&[&x]).remove(0).to_backend::<Host>();
            for (i, (&a, &e)) in got.as_slice().iter().zip(want.as_slice()).enumerate() {
                assert!(close(a, e), "uniform {value}, element {i}: {a} vs {e}");
            }
        }
    }
}
