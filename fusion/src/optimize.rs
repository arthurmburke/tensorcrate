//! Choosing the cheapest equivalent program.

use crate::cost::{Cost, CostModel};
use crate::graph::{Bin, Graph, Node, Store, Value};
use crate::rewrite::{Association, Rematerialize, eliminate_dead, rematerialize, rewrite};
use crate::schedule::{self, Instr, Schedule};

/// What the optimizer may do, and what it must fit.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Options {
    pub model: CostModel,
    /// Allow rebuilding associative chains in another grouping. Not exact:
    /// results may change by rounding.
    pub reassociate: bool,
    /// The registers a program may use.
    pub max_registers: usize,
    /// The instructions a program may hold, stores included.
    pub max_instructions: usize,
    /// Search every order of graphs with at most this many nodes.
    pub exhaustive_limit: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            model: CostModel::default(),
            reassociate: true,
            max_registers: 16,
            max_instructions: 256,
            exhaustive_limit: 10,
        }
    }
}

/// How a plan was derived from the original program.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Variant {
    pub association: Association,
    pub rematerialize: Rematerialize,
    pub schedule: Schedule,
}

impl Variant {
    /// The program as written: no rewrites, built order.
    pub const AS_WRITTEN: Variant = Variant {
        association: Association::Keep,
        rematerialize: Rematerialize::Nothing,
        schedule: Schedule::AsBuilt,
    };
}

/// An allocated program and what it costs.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    /// The rewritten graph, its nodes in execution order.
    pub graph: Graph,
    /// The graph with registers assigned, stores last.
    pub code: Vec<Instr>,
    pub registers: usize,
    pub cost: Cost,
    pub variant: Variant,
}

/// No equivalent program fits the register and instruction limits.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DoesNotFit;

fn latency(model: &CostModel) -> impl Fn(&Node) -> f64 + '_ {
    move |node| {
        let l = &model.latency;
        match node {
            Node::Load { .. } => l.load,
            Node::Const(_) => l.constant,
            Node::Binary(Bin::Div | Bin::Rem, ..) => l.divide,
            Node::Binary(..) | Node::Cmp(..) => l.arithmetic,
            Node::Unary(f, _) if f.is_transcendental() => l.transcendental,
            Node::Unary(..) => l.sqrt,
        }
    }
}

/// The cost of `graph` run in `order` with `registers` registers.
fn cost_of(graph: &Graph, registers: usize, model: &CostModel) -> Cost {
    let finish = schedule::finish_times(graph, latency(model));
    let critical = graph
        .stores
        .iter()
        .map(|store| finish[store.value])
        .fold(0.0, f64::max);
    let mut memory = 0usize;
    let mut special = 0usize;
    for node in &graph.nodes {
        match node {
            Node::Load { slot, .. } => memory += usize::from(graph.input_bytes[usize::from(*slot)]),
            Node::Binary(Bin::Div | Bin::Rem, ..) | Node::Unary(..) => special += 1,
            _ => {}
        }
    }
    for store in &graph.stores {
        memory += usize::from(graph.output_bytes[usize::from(store.output)]);
    }
    Cost::weigh(
        model,
        graph.nodes.len(),
        registers,
        critical,
        memory,
        special,
    )
}

/// `graph` with its nodes permuted into `order`.
fn reorder(graph: &Graph, order: &[Value]) -> Graph {
    let mut position = vec![0; graph.nodes.len()];
    for (at, &value) in order.iter().enumerate() {
        position[value] = at;
    }
    Graph {
        nodes: order
            .iter()
            .map(|&value| graph.nodes[value].remapped(|operand| position[operand]))
            .collect(),
        stores: graph
            .stores
            .iter()
            .map(|store| Store {
                value: position[store.value],
                output: store.output,
            })
            .collect(),
        input_bytes: graph.input_bytes.clone(),
        output_bytes: graph.output_bytes.clone(),
    }
}

/// Every equivalent program the optimizer considers, cheapest first,
/// whether or not it fits the limits.
pub fn candidates(graph: &Graph, options: &Options) -> Vec<Plan> {
    debug_assert!(graph.validate().is_ok(), "{:?}", graph.validate());
    let model = &options.model;
    let mut associations = vec![Association::Keep];
    if options.reassociate {
        associations.extend([Association::LeftDeep, Association::Balanced]);
    }
    let mut plans: Vec<Plan> = Vec::new();
    let mut seen: Vec<Graph> = Vec::new();
    // The program exactly as written, so nothing is ever chosen over it that
    // costs more.
    let written = eliminate_dead(graph.clone());
    let order: Vec<Value> = (0..written.nodes.len()).collect();
    let (code, registers) = schedule::allocate(&written, &order);
    plans.push(Plan {
        cost: cost_of(&written, registers, model),
        graph: written,
        code,
        registers,
        variant: Variant::AS_WRITTEN,
    });

    for association in associations {
        let rewritten = rewrite(graph, association);
        for what in [
            Rematerialize::Nothing,
            Rematerialize::Constants,
            Rematerialize::ConstantsAndLoads,
        ] {
            let variant_graph = rematerialize(&rewritten, what);
            if seen.contains(&variant_graph) {
                continue;
            }
            seen.push(variant_graph.clone());
            for (schedule, order) in
                schedule::orders(&variant_graph, latency(model), options.exhaustive_limit)
            {
                let ordered = reorder(&variant_graph, &order);
                let identity: Vec<Value> = (0..ordered.nodes.len()).collect();
                let (code, registers) = schedule::allocate(&ordered, &identity);
                debug_assert_eq!(registers, schedule::peak(&ordered, &identity));
                plans.push(Plan {
                    cost: cost_of(&ordered, registers, model),
                    graph: ordered,
                    code,
                    registers,
                    variant: Variant {
                        association,
                        rematerialize: what,
                        schedule,
                    },
                });
            }
        }
    }
    plans.sort_by(|a, b| a.cost.total.total_cmp(&b.cost.total));
    plans
}

/// The cheapest equivalent program that fits the limits.
pub fn optimize(graph: &Graph, options: &Options) -> Result<Plan, DoesNotFit> {
    candidates(graph, options)
        .into_iter()
        .find(|plan| fits(plan, options))
        .ok_or(DoesNotFit)
}

fn fits(plan: &Plan, options: &Options) -> bool {
    plan.registers <= options.max_registers && plan.code.len() <= options.max_instructions
}

/// The cost of a program as written, scheduled and allocated as built.
pub fn cost(graph: &Graph, model: &CostModel) -> Cost {
    let graph = eliminate_dead(graph.clone());
    let order: Vec<Value> = (0..graph.nodes.len()).collect();
    let (_, registers) = schedule::allocate(&graph, &order);
    cost_of(&graph, registers, model)
}
