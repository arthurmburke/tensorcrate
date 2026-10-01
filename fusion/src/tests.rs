//! The optimizer's guarantees, checked against an `f64` evaluator: rewrites
//! without reassociation are exact, with it close; every plan fits; the chosen
//! plan never costs more than the program as written.

use crate::*;

/// Inputs per slot and named constants, for evaluation.
struct Env {
    inputs: Vec<f64>,
    named: Vec<f64>,
}

fn scalar(s: &Scalar, env: &Env) -> f64 {
    match s {
        Scalar::Literal(bits) => f64::from_bits(*bits),
        Scalar::Named(id) => env.named[*id as usize],
        Scalar::Binary(op, a, b) => binary(*op, scalar(a, env), scalar(b, env)),
    }
}

fn binary(op: Bin, a: f64, b: f64) -> f64 {
    match op {
        Bin::Add => a + b,
        Bin::Sub => a - b,
        Bin::Mul => a * b,
        Bin::Div => a / b,
        Bin::Rem => a % b,
    }
}

fn compare(op: Cmp, a: f64, b: f64) -> f64 {
    let flag = |c: bool| if c { 1.0 } else { 0.0 };
    match op {
        Cmp::Min => a.min(b),
        Cmp::Max => a.max(b),
        Cmp::MaxShare => {
            if a > b {
                1.0
            } else if a < b {
                0.0
            } else {
                0.5
            }
        }
        Cmp::Less => flag(a < b),
        Cmp::LessEqual => flag(a <= b),
        Cmp::Greater => flag(a > b),
        Cmp::GreaterEqual => flag(a >= b),
    }
}

fn unary(f: Function, x: f64) -> f64 {
    match f.0 {
        0 => x.sin(),
        8 => x.exp(),
        12 => x.tanh(),
        13 => x.sqrt(),
        _ => x.cos(),
    }
}

/// Each output's value when `graph` runs.
fn run_graph(graph: &Graph, env: &Env) -> Vec<f64> {
    let mut values = Vec::with_capacity(graph.nodes.len());
    for node in &graph.nodes {
        values.push(match *node {
            Node::Load { slot, .. } => env.inputs[usize::from(slot)],
            Node::Const(ref s) => scalar(s, env),
            Node::Binary(op, a, b) => binary(op, values[a], values[b]),
            Node::Unary(f, a) => unary(f, values[a]),
            Node::Cmp(op, a, b) => compare(op, values[a], values[b]),
        });
    }
    let mut out = vec![f64::NAN; graph.output_bytes.len()];
    for store in &graph.stores {
        out[usize::from(store.output)] = values[store.value];
    }
    out
}

/// The same for allocated code, through its registers.
fn run_code(code: &[Instr], outputs: usize, env: &Env) -> Vec<f64> {
    let mut r = [f64::NAN; 64];
    let mut out = vec![f64::NAN; outputs];
    for instr in code {
        match *instr {
            Instr::Load { dst, slot, .. } => r[usize::from(dst)] = env.inputs[usize::from(slot)],
            Instr::Const { dst, ref value } => r[usize::from(dst)] = scalar(value, env),
            Instr::Binary { dst, op, a, b } => {
                r[usize::from(dst)] = binary(op, r[usize::from(a)], r[usize::from(b)])
            }
            Instr::Unary { dst, function, a } => {
                r[usize::from(dst)] = unary(function, r[usize::from(a)])
            }
            Instr::Cmp { dst, op, a, b } => {
                r[usize::from(dst)] = compare(op, r[usize::from(a)], r[usize::from(b)])
            }
            Instr::Store { src, output } => out[usize::from(output)] = r[usize::from(src)],
        }
    }
    out
}

/// A small deterministic generator.
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
    fn unit(&mut self) -> f64 {
        self.next() as f64 / (1u64 << 31) as f64
    }
}

/// A random program over `inputs` slots, biased towards the associative
/// chains and shared values the rewrites look for.
fn random_graph(rng: &mut Lcg, inputs: u8, size: usize, outputs: u8) -> Graph {
    let mut nodes = Vec::new();
    for slot in 0..inputs {
        nodes.push(Node::Load { slot, remap: 0 });
    }
    for _ in 0..size {
        let n = nodes.len();
        let pick = |rng: &mut Lcg| n - 1 - rng.below(n.min(4));
        let node = match rng.below(10) {
            0 => Node::Const(match rng.below(4) {
                0 => Scalar::literal(-1.0),
                1 => Scalar::literal(1.0),
                2 => Scalar::Named(rng.below(3) as u32),
                _ => Scalar::literal(0.5 + rng.below(4) as f64),
            }),
            1 => Node::Unary(Function([0, 8, 12, 13][rng.below(4)]), pick(rng)),
            2 => Node::Cmp(
                [Cmp::Min, Cmp::Max, Cmp::Less][rng.below(3)],
                pick(rng),
                pick(rng),
            ),
            3 => Node::Binary(Bin::Div, pick(rng), pick(rng)),
            _ => Node::Binary(
                [Bin::Add, Bin::Sub, Bin::Mul][rng.below(3)],
                pick(rng),
                rng.below(n),
            ),
        };
        nodes.push(node);
    }
    let stores = (0..outputs)
        .map(|output| Store {
            value: nodes.len() - 1 - usize::from(output) * 2,
            output,
        })
        .collect();
    Graph {
        nodes,
        stores,
        input_bytes: vec![4; usize::from(inputs)],
        output_bytes: vec![4; usize::from(outputs)],
    }
}

fn env(rng: &mut Lcg, inputs: usize) -> Env {
    Env {
        // Away from zero, so divisions and square roots stay tame.
        inputs: (0..inputs).map(|_| 1.0 + rng.unit()).collect(),
        named: vec![1.5, 0.25, 3.0],
    }
}

fn agree(a: &[f64], b: &[f64], tolerance: f64) -> bool {
    a.iter().zip(b).all(|(x, y)| {
        (x.is_nan() && y.is_nan())
            || x == y
            || (x - y).abs() <= tolerance * (1.0 + x.abs().max(y.abs()))
    })
}

#[test]
fn every_candidate_computes_what_the_program_does() {
    let mut rng = Lcg(7);
    for case in 0..150 {
        let graph = random_graph(&mut rng, 3, 6 + case % 14, 1 + (case % 3) as u8);
        let env = env(&mut rng, 3);
        let want = run_graph(&graph, &env);
        let exact = Options {
            reassociate: false,
            ..Options::default()
        };
        for plan in candidates(&graph, &exact) {
            assert_eq!(
                run_code(&plan.code, graph.output_bytes.len(), &env)
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "case {case}: {:?} is not exact\n{graph:#?}\n{:#?}",
                plan.variant,
                plan.code
            );
            let bits = |v: Vec<f64>| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(run_graph(&plan.graph, &env)),
                bits(run_code(&plan.code, graph.output_bytes.len(), &env))
            );
        }
        for plan in candidates(&graph, &Options::default()) {
            let got = run_code(&plan.code, graph.output_bytes.len(), &env);
            assert!(
                agree(&got, &want, 1e-9),
                "case {case}: {:?} gives {got:?}, the program {want:?}\n{graph:#?}\n{:#?}",
                plan.variant,
                plan.code
            );
        }
    }
}

#[test]
fn the_chosen_program_fits_and_never_costs_more() {
    let mut rng = Lcg(11);
    for case in 0..100 {
        let graph = random_graph(&mut rng, 4, 10 + case % 20, 2);
        let options = Options::default();
        let written = cost(&graph, &options.model);
        let plan = optimize(&graph, &options).expect("small programs fit");
        assert!(plan.registers <= options.max_registers);
        assert!(
            plan.cost.total <= written.total + 1e-9,
            "case {case}: {} > {written}",
            plan.cost
        );
    }
}

fn load(slot: u8) -> Node {
    Node::Load { slot, remap: 0 }
}

fn graph(nodes: Vec<Node>, store: Value, inputs: usize) -> Graph {
    Graph {
        nodes,
        stores: vec![Store {
            value: store,
            output: 0,
        }],
        input_bytes: vec![4; inputs],
        output_bytes: vec![4],
    }
}

fn count(plan: &Plan, pred: impl Fn(&Node) -> bool) -> usize {
    plan.graph.nodes.iter().filter(|n| pred(n)).count()
}

#[test]
fn common_subexpressions_are_computed_once() {
    // exp(a + b) · exp(b + a)
    let g = graph(
        vec![
            load(0),
            load(1),
            Node::Binary(Bin::Add, 0, 1),
            Node::Unary(Function(8), 2),
            Node::Binary(Bin::Add, 1, 0),
            Node::Unary(Function(8), 4),
            Node::Binary(Bin::Mul, 3, 5),
        ],
        6,
        2,
    );
    let options = Options {
        reassociate: false,
        ..Options::default()
    };
    let plan = optimize(&g, &options).unwrap();
    assert_eq!(
        count(&plan, |n| matches!(n, Node::Unary(..))),
        1,
        "{:#?}",
        plan.graph
    );
    assert_eq!(plan.cost.special_ops, 1);
}

#[test]
fn exact_identities_remove_instructions() {
    // ((a · 1) − (b · −1)) / 1  =  a + b
    let g = graph(
        vec![
            load(0),
            load(1),
            Node::Const(Scalar::literal(1.0)),
            Node::Const(Scalar::literal(-1.0)),
            Node::Binary(Bin::Mul, 0, 2),
            Node::Binary(Bin::Mul, 1, 3),
            Node::Binary(Bin::Sub, 4, 5),
            Node::Binary(Bin::Div, 6, 2),
        ],
        7,
        2,
    );
    let options = Options {
        reassociate: false,
        ..Options::default()
    };
    let plan = optimize(&g, &options).unwrap();
    assert_eq!(plan.graph.nodes.len(), 3, "{:#?}", plan.graph);
    assert!(matches!(plan.graph.nodes[2], Node::Binary(Bin::Add, ..)));
}

#[test]
fn reassociation_folds_constants_and_saves_divisions() {
    // ((x · 2) · 3) / y / z  =  x · (2 · 3) / (y · z)
    let g = graph(
        vec![
            load(0),
            load(1),
            load(2),
            Node::Const(Scalar::literal(2.0)),
            Node::Const(Scalar::literal(3.0)),
            Node::Binary(Bin::Mul, 0, 3),
            Node::Binary(Bin::Mul, 5, 4),
            Node::Binary(Bin::Div, 6, 1),
            Node::Binary(Bin::Div, 7, 2),
        ],
        8,
        3,
    );
    let plan = optimize(&g, &Options::default()).unwrap();
    assert_eq!(
        count(&plan, |n| matches!(n, Node::Binary(Bin::Div, ..))),
        1,
        "{:#?}",
        plan.graph
    );
    assert_eq!(
        count(&plan, |n| matches!(n, Node::Const(_))),
        1,
        "{:#?}",
        plan.graph
    );
    let exact = optimize(
        &g,
        &Options {
            reassociate: false,
            ..Options::default()
        },
    )
    .unwrap();
    assert!(plan.cost.total < exact.cost.total);
}

#[test]
fn a_balanced_sum_shortens_the_critical_path() {
    // exp(a) + exp(b) + exp(c) + exp(d), left-deep: the latency model makes the
    // chain's depth matter once γ is large.
    let mut nodes: Vec<Node> = (0..4).map(load).collect();
    for i in 0..4 {
        nodes.push(Node::Unary(Function(8), i));
    }
    nodes.push(Node::Binary(Bin::Add, 4, 5));
    nodes.push(Node::Binary(Bin::Add, 8, 6));
    nodes.push(Node::Binary(Bin::Add, 9, 7));
    let g = graph(nodes, 10, 4);
    let options = Options {
        model: CostModel {
            gamma: 10.0,
            ..CostModel::METAL
        },
        ..Options::default()
    };
    let plan = optimize(&g, &options).unwrap();
    assert_eq!(
        plan.variant.association,
        Association::Balanced,
        "{:#?}",
        plan.graph
    );
    let written = cost(&g, &options.model);
    assert!(plan.cost.critical_path < written.critical_path);
}

#[test]
fn register_pressure_is_minimized_and_limits_hold() {
    // Sixteen loads summed pairwise in a bad order: as written, every load is
    // live before the first addition.
    let mut nodes: Vec<Node> = (0..16).map(load).collect();
    let mut acc = 0;
    for i in 1..16 {
        nodes.push(Node::Binary(Bin::Add, acc, i));
        acc = nodes.len() - 1;
    }
    let g = graph(nodes, acc, 16);
    assert!(cost(&g, &CostModel::HOST).peak_registers >= 16);
    let plan = optimize(
        &g,
        &Options {
            reassociate: false,
            ..Options::default()
        },
    )
    .unwrap();
    assert!(
        plan.registers <= 2,
        "{} registers\n{:#?}",
        plan.registers,
        plan.code
    );
}
