//! Rewriting a graph into equivalent ones.
//!
//! Every graph is rebuilt through [`Rebuilder`], which hash-conses: a node
//! identical to one already built is that node, so common subexpressions are
//! computed once. Commutative operands are put in a canonical order first, so
//! `a + b` and `b + a` are found to be the same, and a few identities that hold
//! exactly in IEEE arithmetic are applied as nodes are built:
//!
//! | rewrite              | why it is exact                                   |
//! |----------------------|---------------------------------------------------|
//! | `x · 1 → x`, `x / 1 → x` | multiplying by one is the identity             |
//! | `x − (+0) → x`       | `−0 − +0 = −0`, so the zero's sign survives       |
//! | `x + (−0) → x`       | `+0 + −0 = +0` and `−0 + −0 = −0`                 |
//! | `(−x)·(−1) → x`      | negation is exact                                 |
//! | `a − (−b) → a + b`, `a + (−b) → a − b` | the same, by negation's exactness |
//! | `c₁ ∘ c₂ → c`        | folded in the program's own type, as the unfused kernels would |
//!
//! where `−x` is the `x · −1` that negation compiles to.
//!
//! **Reassociation** is not exact — `(a + b) + c` and `a + (b + c)` can round
//! differently — and runs only when the caller allows it. A chain of one
//! associative family is flattened through every link that nothing else reads:
//!
//! - sums: `+`, `−` and negation, as signed terms;
//! - products: `·` and `÷`, as a numerator and a denominator, so `a / b / c`
//!   becomes `a / (b · c)` and spends one division instead of two;
//! - `min` and `max`.
//!
//! Its constants are gathered into one folded constant, and the chain is rebuilt
//! in a canonical order — the same terms always give the same tree, which lets
//! common subexpressions be found across chains — as a left-deep chain or a
//! balanced tree. The balanced tree has the shortest critical path; the
//! left-deep chain needs the fewest registers. The cost model chooses.

use std::collections::HashMap;

use crate::graph::{Bin, Cmp, Graph, Node, Scalar, Store, Value};

/// How associative chains are rebuilt.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Association {
    /// As written.
    Keep,
    /// Flattened and rebuilt as a left-deep chain.
    LeftDeep,
    /// Flattened and rebuilt as a balanced tree.
    Balanced,
}

/// Builds a graph node by node, hash-consing and simplifying as it goes.
pub(crate) struct Rebuilder {
    pub(crate) nodes: Vec<Node>,
    index: HashMap<Node, Value>,
    /// Whether rewrites that are not bit-exact are allowed.
    relaxed: bool,
}

impl Rebuilder {
    pub(crate) fn new(relaxed: bool) -> Self {
        Rebuilder {
            nodes: Vec::new(),
            index: HashMap::new(),
            relaxed,
        }
    }

    pub(crate) fn push(&mut self, node: Node) -> Value {
        if let Some(&value) = self.index.get(&node) {
            return value;
        }
        self.nodes.push(node.clone());
        let value = self.nodes.len() - 1;
        self.index.insert(node, value);
        value
    }

    pub(crate) fn constant(&mut self, scalar: Scalar) -> Value {
        self.push(Node::Const(scalar))
    }

    fn scalar(&self, value: Value) -> Option<&Scalar> {
        match &self.nodes[value] {
            Node::Const(scalar) => Some(scalar),
            _ => None,
        }
    }

    fn is_constant(&self, value: Value, of: f64) -> bool {
        self.scalar(value).is_some_and(|scalar| scalar.is(of))
    }

    /// `x` when `value` is `x · −1`.
    fn negation_of(&self, value: Value) -> Option<Value> {
        match self.nodes[value] {
            Node::Binary(Bin::Mul, a, b) if self.is_constant(b, -1.0) => Some(a),
            Node::Binary(Bin::Mul, a, b) if self.is_constant(a, -1.0) => Some(b),
            _ => None,
        }
    }

    pub(crate) fn negate(&mut self, value: Value) -> Value {
        let minus_one = self.constant(Scalar::literal(-1.0));
        self.binary(Bin::Mul, value, minus_one)
    }

    pub(crate) fn binary(&mut self, op: Bin, a: Value, b: Value) -> Value {
        if let (Some(x), Some(y)) = (self.scalar(a), self.scalar(b)) {
            let folded = Scalar::Binary(op, Box::new(x.clone()), Box::new(y.clone()));
            return self.constant(folded);
        }
        match op {
            Bin::Mul => {
                if self.is_constant(b, 1.0) {
                    return a;
                }
                if self.is_constant(a, 1.0) {
                    return b;
                }
                if self.is_constant(b, -1.0)
                    && let Some(x) = self.negation_of(a)
                {
                    return x;
                }
                if self.is_constant(a, -1.0)
                    && let Some(x) = self.negation_of(b)
                {
                    return x;
                }
            }
            Bin::Div if self.is_constant(b, 1.0) => return a,
            Bin::Sub => {
                if self.is_constant(b, 0.0) {
                    return a;
                }
                if let Some(y) = self.negation_of(b) {
                    return self.binary(Bin::Add, a, y);
                }
            }
            Bin::Add => {
                if self.is_constant(b, -0.0) {
                    return a;
                }
                if self.is_constant(a, -0.0) {
                    return b;
                }
                if let Some(y) = self.negation_of(b) {
                    return self.binary(Bin::Sub, a, y);
                }
                if let Some(x) = self.negation_of(a) {
                    return self.binary(Bin::Sub, b, x);
                }
            }
            _ => {}
        }
        let (a, b) = if op.commutes() && b < a {
            (b, a)
        } else {
            (a, b)
        };
        self.push(Node::Binary(op, a, b))
    }

    pub(crate) fn compare(&mut self, op: Cmp, a: Value, b: Value) -> Value {
        // `Less(b, a)` is `Greater(a, b)` exactly. `Min` and `Max` commute too,
        // except that `−0` against `+0` may come out either way, which only a
        // relaxed rebuild may change.
        let mirror = match op {
            Cmp::Min | Cmp::Max if !self.relaxed => None,
            _ => op.mirrored(),
        };
        match mirror {
            Some(mirrored) if b < a => self.push(Node::Cmp(mirrored, b, a)),
            _ => self.push(Node::Cmp(op, a, b)),
        }
    }

    /// Rebuild one node of the source graph, its operands already mapped.
    fn copy(&mut self, node: &Node, map: &[Value]) -> Value {
        match *node {
            Node::Binary(op, a, b) => self.binary(op, map[a], map[b]),
            Node::Cmp(op, a, b) => self.compare(op, map[a], map[b]),
            Node::Unary(f, a) => self.push(Node::Unary(f, map[a])),
            ref leaf => self.push(leaf.clone()),
        }
    }

    /// Combine `terms` with `op`, as `association` says.
    fn reduce(&mut self, op: Bin, terms: &[Value], association: Association) -> Value {
        self.reduce_with(terms, association, |builder, a, b| builder.binary(op, a, b))
    }

    fn reduce_with(
        &mut self,
        terms: &[Value],
        association: Association,
        mut combine: impl FnMut(&mut Self, Value, Value) -> Value,
    ) -> Value {
        assert!(!terms.is_empty(), "a chain has at least one term");
        let mut level = terms.to_vec();
        if association == Association::Balanced {
            while level.len() > 1 {
                let mut next = Vec::with_capacity(level.len().div_ceil(2));
                for pair in level.chunks(2) {
                    next.push(match *pair {
                        [a, b] => combine(self, a, b),
                        [a] => a,
                        _ => unreachable!(),
                    });
                }
                level = next;
            }
            level[0]
        } else {
            let mut acc = level[0];
            for &term in &level[1..] {
                acc = combine(self, acc, term);
            }
            acc
        }
    }

    /// Split mapped terms into the non-constant ones, in canonical order, and
    /// the constants.
    fn split_constants(&self, terms: Vec<Term>) -> (Vec<Term>, Vec<(bool, Scalar)>) {
        let mut values = Vec::new();
        let mut constants = Vec::new();
        for (inverse, value) in terms {
            match self.scalar(value) {
                Some(scalar) => constants.push((inverse, scalar.clone())),
                None => values.push((inverse, value)),
            }
        }
        values.sort_by_key(|&(inverse, value)| (value, inverse));
        (values, constants)
    }
}

/// A chain's term: a value, and whether it is subtracted (in a sum) or divided
/// by (in a product).
type Term = (bool, Value);

/// The families of associative operations.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Family {
    Sum,
    Product,
    Min,
    Max,
}

fn family(graph: &Graph, value: Value) -> Option<Family> {
    match graph.nodes[value] {
        Node::Binary(Bin::Add | Bin::Sub, ..) => Some(Family::Sum),
        Node::Binary(Bin::Mul | Bin::Div, ..) => Some(Family::Product),
        Node::Cmp(Cmp::Min, ..) => Some(Family::Min),
        Node::Cmp(Cmp::Max, ..) => Some(Family::Max),
        _ => None,
    }
}

/// Whether `value` is `x · −1` in the source graph, and if so `x`.
fn source_negation(graph: &Graph, value: Value) -> Option<Value> {
    let is_minus_one = |v: Value| matches!(&graph.nodes[v], Node::Const(s) if s.is(-1.0));
    match graph.nodes[value] {
        Node::Binary(Bin::Mul, a, b) if is_minus_one(b) => Some(a),
        Node::Binary(Bin::Mul, a, b) if is_minus_one(a) => Some(b),
        _ => None,
    }
}

/// Collect the terms of the chain rooted at `value`: source values, each
/// flagged when it is subtracted (a sum) or divided by (a product). A link is
/// flattened through only when it is the root or nothing else reads it.
fn flatten(
    graph: &Graph,
    absorbable: &[bool],
    value: Value,
    root: bool,
    inverse: bool,
    family: Family,
    out: &mut Vec<(bool, Value)>,
) {
    let through = root || absorbable[value];
    match (family, &graph.nodes[value]) {
        (Family::Sum, &Node::Binary(op @ (Bin::Add | Bin::Sub), a, b)) if through => {
            flatten(graph, absorbable, a, false, inverse, family, out);
            flatten(
                graph,
                absorbable,
                b,
                false,
                inverse ^ (op == Bin::Sub),
                family,
                out,
            );
        }
        (Family::Sum, Node::Binary(Bin::Mul, ..))
            if through && source_negation(graph, value).is_some() =>
        {
            let x = source_negation(graph, value).unwrap();
            flatten(graph, absorbable, x, false, !inverse, family, out);
        }
        (Family::Product, &Node::Binary(op @ (Bin::Mul | Bin::Div), a, b)) if through => {
            flatten(graph, absorbable, a, false, inverse, family, out);
            flatten(
                graph,
                absorbable,
                b,
                false,
                inverse ^ (op == Bin::Div),
                family,
                out,
            );
        }
        (Family::Min, &Node::Cmp(Cmp::Min, a, b)) | (Family::Max, &Node::Cmp(Cmp::Max, a, b))
            if through =>
        {
            flatten(graph, absorbable, a, false, false, family, out);
            flatten(graph, absorbable, b, false, false, family, out);
        }
        _ => out.push((inverse, value)),
    }
}

/// Rebuild the chain rooted at source `value` from its flattened terms.
fn reassociate(
    builder: &mut Rebuilder,
    graph: &Graph,
    absorbable: &[bool],
    map: &[Value],
    value: Value,
    family: Family,
    association: Association,
) -> Value {
    let mut terms = Vec::new();
    flatten(graph, absorbable, value, true, false, family, &mut terms);
    let mapped: Vec<(bool, Value)> = terms
        .iter()
        .map(|&(inverse, v)| (inverse, map[v]))
        .collect();

    match family {
        Family::Min | Family::Max => {
            let op = if family == Family::Min {
                Cmp::Min
            } else {
                Cmp::Max
            };
            let mut values: Vec<Value> = mapped.iter().map(|&(_, v)| v).collect();
            values.sort_unstable();
            values.dedup(); // min(a, a) is a
            builder.reduce_with(&values, association, |b, x, y| b.compare(op, x, y))
        }
        Family::Sum | Family::Product => {
            let (values, constants) = builder.split_constants(mapped);
            let (combine, invert) = if family == Family::Sum {
                (Bin::Add, Bin::Sub)
            } else {
                (Bin::Mul, Bin::Div)
            };
            // One constant for all of them, folded by the caller.
            let mut folded: Option<Scalar> = None;
            for (inverse, scalar) in constants {
                folded = Some(match (folded, inverse) {
                    (Some(acc), false) => Scalar::Binary(combine, Box::new(acc), Box::new(scalar)),
                    (Some(acc), true) => Scalar::Binary(invert, Box::new(acc), Box::new(scalar)),
                    (None, false) => scalar,
                    (None, true) if family == Family::Sum => {
                        Scalar::Binary(Bin::Mul, Box::new(scalar), Box::new(Scalar::literal(-1.0)))
                    }
                    (None, true) => {
                        Scalar::Binary(Bin::Div, Box::new(Scalar::literal(1.0)), Box::new(scalar))
                    }
                });
            }
            let mut direct: Vec<Value> = values.iter().filter(|t| !t.0).map(|t| t.1).collect();
            let inverted: Vec<Value> = values.iter().filter(|t| t.0).map(|t| t.1).collect();
            if let Some(scalar) = folded {
                let constant = builder.constant(scalar);
                direct.push(constant);
            }
            match (direct.is_empty(), inverted.is_empty()) {
                (false, true) => builder.reduce(combine, &direct, association),
                (false, false) => {
                    let a = builder.reduce(combine, &direct, association);
                    let b = builder.reduce(combine, &inverted, association);
                    builder.binary(invert, a, b)
                }
                (true, false) => {
                    let b = builder.reduce(combine, &inverted, association);
                    if family == Family::Sum {
                        builder.negate(b)
                    } else {
                        let one = builder.constant(Scalar::literal(1.0));
                        builder.binary(Bin::Div, one, b)
                    }
                }
                (true, true) => unreachable!("a chain has at least one term"),
            }
        }
    }
}

/// `graph` rebuilt through a [`Rebuilder`]: common subexpressions merged,
/// exact identities applied, chains reassociated if `association` says so, and
/// dead code removed. `relaxed` permits rewrites that are not bit-exact; any
/// `association` but [`Keep`](Association::Keep) implies it.
pub(crate) fn rewrite(graph: &Graph, association: Association) -> Graph {
    let relaxed = association != Association::Keep;
    let users = graph.users();
    let stored = graph.stored();
    let absorbable: Vec<bool> = (0..graph.nodes.len())
        .map(|v| users[v].len() == 1 && !stored[v])
        .collect();

    let mut builder = Rebuilder::new(relaxed);
    let mut map = Vec::with_capacity(graph.nodes.len());
    for (value, node) in graph.nodes.iter().enumerate() {
        let rebuilt = match family(graph, value) {
            Some(family) if relaxed => reassociate(
                &mut builder,
                graph,
                &absorbable,
                &map,
                value,
                family,
                association,
            ),
            _ => builder.copy(node, &map),
        };
        map.push(rebuilt);
    }
    let stores = graph
        .stores
        .iter()
        .map(|store| Store {
            value: map[store.value],
            output: store.output,
        })
        .collect();
    eliminate_dead(Graph {
        nodes: builder.nodes,
        stores,
        input_bytes: graph.input_bytes.clone(),
        output_bytes: graph.output_bytes.clone(),
    })
}

/// Drop every node no store depends on, keeping the rest in order.
pub(crate) fn eliminate_dead(graph: Graph) -> Graph {
    let mut live = vec![false; graph.nodes.len()];
    for store in &graph.stores {
        live[store.value] = true;
    }
    for value in (0..graph.nodes.len()).rev() {
        if live[value] {
            for operand in graph.nodes[value].operands() {
                live[operand] = true;
            }
        }
    }
    let mut renumber = vec![usize::MAX; graph.nodes.len()];
    let mut nodes = Vec::new();
    for (value, node) in graph.nodes.iter().enumerate() {
        if live[value] {
            renumber[value] = nodes.len();
            nodes.push(node.remapped(|operand| renumber[operand]));
        }
    }
    let stores = graph
        .stores
        .iter()
        .map(|store| Store {
            value: renumber[store.value],
            output: store.output,
        })
        .collect();
    Graph {
        nodes,
        stores,
        ..graph
    }
}

/// Which leaves to recompute at each use rather than keep live in between.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Rematerialize {
    Nothing,
    Constants,
    /// Constants and loads. A load recomputed is a load repeated, so this
    /// trades memory traffic for registers.
    ConstantsAndLoads,
}

/// `graph` with each chosen leaf emitted afresh just before every node that
/// reads it, so it is live only for that instruction.
pub(crate) fn rematerialize(graph: &Graph, what: Rematerialize) -> Graph {
    let chosen = |node: &Node| match (what, node) {
        (Rematerialize::Nothing, _) => false,
        (_, Node::Const(_)) => true,
        (Rematerialize::ConstantsAndLoads, Node::Load { .. }) => true,
        _ => false,
    };
    if what == Rematerialize::Nothing || !graph.nodes.iter().any(chosen) {
        return graph.clone();
    }
    let stored = graph.stored();
    let mut nodes = Vec::new();
    let mut map = vec![usize::MAX; graph.nodes.len()];
    for (value, node) in graph.nodes.iter().enumerate() {
        if chosen(node) {
            // Emitted at each use instead; a stored leaf keeps one copy here.
            if stored[value] {
                map[value] = nodes.len();
                nodes.push(node.clone());
            }
            continue;
        }
        let mut copies: Vec<(Value, Value)> = Vec::new();
        for operand in node.operands() {
            if chosen(&graph.nodes[operand]) && !copies.iter().any(|&(from, _)| from == operand) {
                copies.push((operand, nodes.len()));
                nodes.push(graph.nodes[operand].clone());
            }
        }
        let rebuilt = node.remapped(|operand| {
            copies
                .iter()
                .find(|&&(from, _)| from == operand)
                .map_or(map[operand], |&(_, to)| to)
        });
        map[value] = nodes.len();
        nodes.push(rebuilt);
    }
    let stores = graph
        .stores
        .iter()
        .map(|store| Store {
            value: map[store.value],
            output: store.output,
        })
        .collect();
    Graph {
        nodes,
        stores,
        input_bytes: graph.input_bytes.clone(),
        output_bytes: graph.output_bytes.clone(),
    }
}
