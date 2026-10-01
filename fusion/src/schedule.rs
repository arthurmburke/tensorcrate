//! Ordering a graph's nodes and assigning them registers.
//!
//! Every topological order computes the same values; they differ in how many
//! are live at once, which is the register count a program needs. Finding the
//! order with the fewest is NP-hard for graphs with sharing, so several
//! heuristics each propose one, small graphs are searched exhaustively, and the
//! cost model keeps the best.

use crate::graph::{Graph, Node, Scalar, Value};

/// One instruction of an allocated program.
#[derive(Clone, Debug, PartialEq)]
pub enum Instr {
    Load {
        dst: u8,
        slot: u8,
        remap: u8,
    },
    Const {
        dst: u8,
        value: Scalar,
    },
    Binary {
        dst: u8,
        op: crate::Bin,
        a: u8,
        b: u8,
    },
    Unary {
        dst: u8,
        function: crate::Function,
        a: u8,
    },
    Cmp {
        dst: u8,
        op: crate::Cmp,
        a: u8,
        b: u8,
    },
    Store {
        src: u8,
        output: u8,
    },
}

/// The strategies that propose orders.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Schedule {
    /// The order the graph was built in.
    AsBuilt,
    /// Depth first from the stores, the operand needing more registers first
    /// — Sethi–Ullman order, optimal for trees.
    DepthFirst,
    /// Greedily, the ready node that leaves the fewest values live.
    LeastLive,
    /// Greedily, the ready node with the longest chain still after it.
    CriticalPath,
    /// Every order, pruned by the best found so far.
    Exhaustive,
}

/// Each value's readers counted with multiplicity removed, and stores.
struct Liveness {
    users: Vec<usize>,
    stored: Vec<bool>,
}

impl Liveness {
    fn new(graph: &Graph) -> Self {
        Liveness {
            users: graph.users().iter().map(Vec::len).collect(),
            stored: graph.stored(),
        }
    }
}

/// Distinct operands, in order.
fn distinct(node: &Node) -> Vec<Value> {
    let mut operands = node.operands();
    operands.dedup();
    operands
}

/// The most values live at once when `graph` runs in `order`.
pub(crate) fn peak(graph: &Graph, order: &[Value]) -> usize {
    let liveness = Liveness::new(graph);
    let mut remaining = liveness.users.clone();
    let (mut live, mut peak) = (0usize, 0usize);
    for &value in order {
        let mut dying = 0;
        for operand in distinct(&graph.nodes[value]) {
            remaining[operand] -= 1;
            if remaining[operand] == 0 && !liveness.stored[operand] {
                dying += 1;
            }
        }
        // The result may take a dying operand's register.
        live = live + 1 - dying;
        peak = peak.max(live);
    }
    peak
}

/// The longest weighted chain ending at each node.
pub(crate) fn finish_times(graph: &Graph, latency: impl Fn(&Node) -> f64) -> Vec<f64> {
    let mut finish = vec![0.0f64; graph.nodes.len()];
    for (value, node) in graph.nodes.iter().enumerate() {
        let start = node
            .operands()
            .iter()
            .map(|&operand| finish[operand])
            .fold(0.0, f64::max);
        finish[value] = start + latency(node);
    }
    finish
}

/// The longest weighted chain from each node to the end of the program.
fn heights(graph: &Graph, latency: &impl Fn(&Node) -> f64) -> Vec<f64> {
    let users = graph.users();
    let mut height = vec![0.0f64; graph.nodes.len()];
    for value in (0..graph.nodes.len()).rev() {
        let after = users[value]
            .iter()
            .map(|&user| height[user])
            .fold(0.0, f64::max);
        height[value] = after + latency(&graph.nodes[value]);
    }
    height
}

/// The orders the strategies propose, each once, the as-built order first.
pub(crate) fn orders(
    graph: &Graph,
    latency: impl Fn(&Node) -> f64,
    exhaustive_limit: usize,
) -> Vec<(Schedule, Vec<Value>)> {
    let mut found: Vec<(Schedule, Vec<Value>)> = Vec::new();
    let mut offer = |schedule: Schedule, order: Vec<Value>| {
        if !found.iter().any(|(_, existing)| *existing == order) {
            found.push((schedule, order));
        }
    };
    offer(Schedule::AsBuilt, (0..graph.nodes.len()).collect());
    offer(Schedule::DepthFirst, depth_first(graph));
    offer(Schedule::LeastLive, list_schedule(graph, &latency, false));
    offer(Schedule::CriticalPath, list_schedule(graph, &latency, true));
    if graph.nodes.len() <= exhaustive_limit
        && let Some(order) = exhaustive(graph)
    {
        offer(Schedule::Exhaustive, order);
    }
    found
}

/// Registers each node needs to evaluate, counting a shared operand in full at
/// each use — exact for trees, an estimate with sharing.
fn needs(graph: &Graph) -> Vec<usize> {
    let mut need = vec![1usize; graph.nodes.len()];
    for (value, node) in graph.nodes.iter().enumerate() {
        need[value] = match *node {
            Node::Unary(_, a) => need[a],
            Node::Binary(_, a, b) | Node::Cmp(_, a, b) => {
                if a == b {
                    need[a]
                } else if need[a] == need[b] {
                    need[a] + 1
                } else {
                    need[a].max(need[b])
                }
            }
            _ => 1,
        };
    }
    need
}

fn depth_first(graph: &Graph) -> Vec<Value> {
    let need = needs(graph);
    let mut emitted = vec![false; graph.nodes.len()];
    let mut order = Vec::with_capacity(graph.nodes.len());
    fn visit(
        graph: &Graph,
        need: &[usize],
        value: Value,
        emitted: &mut [bool],
        order: &mut Vec<Value>,
    ) {
        if emitted[value] {
            return;
        }
        let mut operands = distinct(&graph.nodes[value]);
        operands.sort_by_key(|&operand| std::cmp::Reverse(need[operand]));
        for operand in operands {
            visit(graph, need, operand, emitted, order);
        }
        emitted[value] = true;
        order.push(value);
    }
    for store in &graph.stores {
        visit(graph, &need, store.value, &mut emitted, &mut order);
    }
    order
}

/// List scheduling: repeatedly emit the best ready node. By liveness, the one
/// that frees the most operands for the one value it adds — so leaves wait
/// until something needs them — breaking ties by the longer chain after it; by
/// critical path, the other way round.
fn list_schedule(
    graph: &Graph,
    latency: &impl Fn(&Node) -> f64,
    critical_first: bool,
) -> Vec<Value> {
    let liveness = Liveness::new(graph);
    let height = heights(graph, latency);
    let mut remaining = liveness.users.clone();
    let mut waiting: Vec<usize> = graph
        .nodes
        .iter()
        .map(|node| distinct(node).len())
        .collect();
    let users = graph.users();
    let mut ready: Vec<Value> = (0..graph.nodes.len())
        .filter(|&v| waiting[v] == 0)
        .collect();
    let mut order = Vec::with_capacity(graph.nodes.len());
    while !ready.is_empty() {
        let growth = |value: Value, remaining: &[usize]| -> i64 {
            let dying = distinct(&graph.nodes[value])
                .iter()
                .filter(|&&operand| remaining[operand] == 1 && !liveness.stored[operand])
                .count();
            1 - dying as i64
        };
        let best = (0..ready.len())
            .min_by(|&i, &j| {
                let (a, b) = (ready[i], ready[j]);
                let by_growth = growth(a, &remaining).cmp(&growth(b, &remaining));
                let by_height = height[b].total_cmp(&height[a]);
                let primary = if critical_first {
                    by_height.then(by_growth)
                } else {
                    by_growth.then(by_height)
                };
                primary.then(a.cmp(&b))
            })
            .unwrap();
        let value = ready.swap_remove(best);
        for operand in distinct(&graph.nodes[value]) {
            remaining[operand] -= 1;
        }
        order.push(value);
        for &user in &users[value] {
            waiting[user] -= 1;
            if waiting[user] == 0 {
                ready.push(user);
            }
        }
    }
    order
}

/// The order with the fewest values live at peak, by branch and bound.
fn exhaustive(graph: &Graph) -> Option<Vec<Value>> {
    struct Search<'g> {
        graph: &'g Graph,
        stored: Vec<bool>,
        users: Vec<Vec<Value>>,
        best_peak: usize,
        best: Option<Vec<Value>>,
        nodes_visited: usize,
    }
    impl Search<'_> {
        #[allow(clippy::too_many_arguments)]
        fn step(
            &mut self,
            order: &mut Vec<Value>,
            waiting: &mut Vec<usize>,
            remaining: &mut Vec<usize>,
            live: usize,
            peak: usize,
        ) {
            self.nodes_visited += 1;
            if peak >= self.best_peak || self.nodes_visited > 200_000 {
                return;
            }
            if order.len() == self.graph.nodes.len() {
                self.best_peak = peak;
                self.best = Some(order.clone());
                return;
            }
            let ready: Vec<Value> = (0..self.graph.nodes.len())
                .filter(|&v| waiting[v] == 0 && !order.contains(&v))
                .collect();
            for value in ready {
                let operands = distinct(&self.graph.nodes[value]);
                let mut dying = 0;
                for &operand in &operands {
                    remaining[operand] -= 1;
                    if remaining[operand] == 0 && !self.stored[operand] {
                        dying += 1;
                    }
                }
                let now = live + 1 - dying;
                for &user in &self.users[value] {
                    waiting[user] -= 1;
                }
                order.push(value);
                self.step(order, waiting, remaining, now, peak.max(now));
                order.pop();
                for &user in &self.users[value] {
                    waiting[user] += 1;
                }
                for &operand in &operands {
                    remaining[operand] += 1;
                }
            }
        }
    }
    let users = graph.users();
    let mut search = Search {
        graph,
        stored: graph.stored(),
        users: users.clone(),
        best_peak: usize::MAX,
        best: None,
        nodes_visited: 0,
    };
    let mut waiting: Vec<usize> = graph
        .nodes
        .iter()
        .map(|node| distinct(node).len())
        .collect();
    let mut remaining: Vec<usize> = users.iter().map(Vec::len).collect();
    search.step(&mut Vec::new(), &mut waiting, &mut remaining, 0, 0);
    search.best
}

/// Assign registers in `order`: a value's register is freed after its last
/// reader, and a result takes the register of an operand that dies with it —
/// every backend reads an instruction's operands before writing its result.
/// Stores follow every node. Returns the code and the registers used.
pub(crate) fn allocate(graph: &Graph, order: &[Value]) -> (Vec<Instr>, usize) {
    let liveness = Liveness::new(graph);
    let mut remaining = liveness.users.clone();
    let mut free: Vec<u8> = Vec::new();
    let mut next = 0u8;
    let mut assigned = vec![u8::MAX; graph.nodes.len()];
    let mut code = Vec::with_capacity(order.len() + graph.stores.len());
    for &value in order {
        let node = &graph.nodes[value];
        let mut released = Vec::new();
        for operand in distinct(node) {
            remaining[operand] -= 1;
            if remaining[operand] == 0 && !liveness.stored[operand] {
                released.push(assigned[operand]);
            }
        }
        let dst = if let Some(&reg) = released.first() {
            reg
        } else if let Some(position) = free
            .iter()
            .enumerate()
            .min_by_key(|(_, r)| **r)
            .map(|(i, _)| i)
        {
            free.swap_remove(position)
        } else {
            next += 1;
            next - 1
        };
        free.extend(released.iter().skip(1));
        let reg = |operand: Value| assigned[operand];
        code.push(match *node {
            Node::Load { slot, remap } => Instr::Load { dst, slot, remap },
            Node::Const(ref value) => Instr::Const {
                dst,
                value: value.clone(),
            },
            Node::Binary(op, a, b) => Instr::Binary {
                dst,
                op,
                a: reg(a),
                b: reg(b),
            },
            Node::Unary(function, a) => Instr::Unary {
                dst,
                function,
                a: reg(a),
            },
            Node::Cmp(op, a, b) => Instr::Cmp {
                dst,
                op,
                a: reg(a),
                b: reg(b),
            },
        });
        assigned[value] = dst;
    }
    for store in &graph.stores {
        code.push(Instr::Store {
            src: assigned[store.value],
            output: store.output,
        });
    }
    (code, usize::from(next))
}
