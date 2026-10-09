//! Graphs of fused kernels around matrix and dot products.
//!
//! A [`Builder`] whose values include a [`matmul`](Builder::matmul) or a
//! [`dot`](Builder::dot) cannot run as one elementwise kernel: a product reads
//! a whole row and column for each element it computes. It builds a [`Graph`]
//! instead, a short sequence of kernels with the elementwise work around each
//! product fused into it.
//!
//! - A dot product sums the elementwise products of its operands along their
//!   last axis. The products, and the elementwise work they are computed from,
//!   run as a program whose output is summed by rows ([`Program::run_sum`]):
//!   one kernel, and on Metal the products are never stored.
//! - A matrix product runs on the backend's product kernels. Elementwise work
//!   that reads it — at its own shape, without a remap — is its epilogue
//!   ([`Program::run_matmul`]): the same dispatch on Metal. An operand that is
//!   a given matrix is read where it lies; one computed elementwise is
//!   computed first, by a kernel of its own or by the epilogue of the product
//!   it is computed from. So `relu(x·w₁ + b₁)·w₂ + b₂` is two kernels.
//! - Whatever is left — the graph's outputs and the tensors it updates in
//!   place — is a last elementwise kernel, or the epilogue of the last product
//!   it reads.
//!
//! The intermediates between kernels are tensors of the program's element type
//! on the backend that runs the graph, so a graph on Metal stays on the
//! device.

use std::collections::{HashMap, HashSet};
use std::fmt;

use super::super::{Axis, Kernels, Matrix, Vector};
use super::{
    Algebra, BinaryOp, Builder, CostModel, DType, Decl, Element, Fusable, FusableMut, MAX_INPUTS,
    Meta, Node, Output, Place, Program, ProgramError, Remap, Var, algebra, axis_sum, dtype_of,
    load_unfused, narrow,
};
use crate::numbers::Real;
use crate::tensors::layout::{Dims, broadcast_shape};

/// What a kernel of a [`Graph`] reads or writes besides its program's own
/// constants.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum Value {
    /// Given input `k`.
    Input(usize),
    /// In-place tensor `k`, as it was before the graph ran.
    Update(usize),
    /// The value of the builder's node `id`, computed by an earlier kernel.
    Temp(usize),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Input(k) => write!(f, "in{k}"),
            Value::Update(k) => write!(f, "upd{k}"),
            Value::Temp(id) => write!(f, "t{id}"),
        }
    }
}

/// Where one of a kernel's fresh outputs goes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Write {
    Temp(usize),
    Output(usize),
}

impl fmt::Display for Write {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Write::Temp(id) => write!(f, "t{id}"),
            Write::Output(k) => write!(f, "out{k}"),
        }
    }
}

#[derive(Clone, Debug)]
enum Kind {
    /// The program alone; the last kernel also updates the graph's in-place
    /// tensors.
    Elementwise { last: bool },
    /// `a·b`, then the program, if any, as its epilogue: the product is its
    /// input 0, and its reads fill the slots after it. Without a program the
    /// product itself is the kernel's one write.
    Product { a: Value, b: Value },
    /// The sums of the program's one output along its last axis: dot
    /// products of this shape.
    Sum { shape: Dims },
    /// No kernel: the outputs are temporaries already computed, each handed
    /// over as it is.
    Store,
}

/// One kernel of a [`Graph`].
#[derive(Clone, Debug)]
struct Stage<T> {
    kind: Kind,
    program: Option<Program<T>>,
    /// What each of the program's given input slots reads, in order — after
    /// the product, for an epilogue.
    reads: Vec<Value>,
    /// Where each of the program's fresh outputs goes, or for a product or a
    /// sum without one, the result.
    writes: Vec<Write>,
    /// Each of the graph's uniforms the program reads, and its index there.
    uniforms: Vec<(usize, usize)>,
}

/// Fused kernels around matrix and dot products, built by
/// [`Builder::build_graph`].
///
/// It runs like a [`Program`]: given inputs read, in-place tensors overwritten,
/// fresh outputs returned, all of the shape of the graph's
/// [`space`](Self::space). Inside, it runs [`kernels`](Self::kernels)
/// kernels in order, keeping what one passes the next on the backend:
///
/// - A dot product's elementwise products, and the work they are computed
///   from, run as a program whose output is summed by rows
///   ([`Program::run_sum`]): one kernel, and on Metal the products are never
///   stored.
/// - A matrix product runs on the backend's product kernels, with the
///   elementwise work that reads it — at its own shape, without a remap — as
///   its epilogue ([`Program::run_matmul`]). An operand that is a given matrix
///   is read where it lies; one computed elementwise is computed first, by a
///   kernel of its own or by the epilogue of the product it is computed from.
///   So `relu(x·w₁ + b₁)·w₂ + b₂` is two kernels.
/// - The outputs and the tensors updated in place are a last elementwise
///   kernel, or the epilogue of the last product they read.
///
/// ```
/// use tensorcrate::tensors::fused::{Builder, DType, Decl};
/// use tensorcrate::tensors::{Analytic, Vector};
///
/// // The cosine of the angle between two vectors.
/// let mut b = Builder::<f32>::new();
/// let x = b.input(Decl::vector(DType::F32, 3));
/// let y = b.input(Decl::vector(DType::F32, 3));
/// let xy = b.dot(x, y);
/// let xx = b.dot(x, x);
/// let yy = b.dot(y, y);
/// let norms = b.mul(xx, yy);
/// let norm = b.unary(Analytic::Sqrt, norms);
/// let cosine = b.div(xy, norm);
/// b.output(cosine, DType::F32);
/// let graph = b.build_graph().unwrap();
/// assert_eq!(graph.kernels(), 4);
///
/// let x = Vector::new([1.0f32, 0.0, 0.0]);
/// let y = Vector::new([1.0f32, 1.0, 0.0]);
/// let cosine = graph.run(&[&x, &y], &mut []).remove(0).into_vector::<f32>();
/// assert!((cosine[0] - 0.5f32.sqrt()).abs() < 1e-6);
/// ```
#[derive(Clone, Debug)]
pub struct Graph<T = f32> {
    stages: Vec<Stage<T>>,
    /// Each given input's storage type and shape.
    inputs: Vec<(DType, Dims)>,
    /// Each in-place tensor's storage type and shape.
    updates: Vec<(DType, Dims)>,
    /// Each output's storage type.
    outputs: Vec<DType>,
    /// The shape of the outputs.
    space: Dims,
    /// Each uniform's current value.
    uniforms: Vec<T>,
}

/// Work the planner schedules, in order, before its programs are built.
enum Work {
    /// Compute node `root`'s value into a temporary.
    Materialize { root: usize, done: HashSet<usize> },
    /// `a·b`: node `node`'s value.
    Product { node: usize, a: Value, b: Value },
    /// The row sums of node `root`, the elementwise products of dot product
    /// `node`.
    Sum {
        node: usize,
        root: usize,
        done: HashSet<usize>,
    },
    /// The graph's outputs and in-place tensors.
    Final { done: HashSet<usize> },
}

/// Where a region's root goes.
#[derive(Copy, Clone, Debug)]
enum Dest {
    Temp(usize),
    Output(usize),
    Update(usize),
}

/// What a region reads besides inputs: each temporary, and whether only as
/// itself rather than through a remap; and whether it reads row statistics.
struct Survey {
    temps: HashMap<usize, bool>,
    statistics: bool,
}

/// A region's program, and what it reads and writes.
struct Built<T> {
    program: Program<T>,
    reads: Vec<Value>,
    writes: Vec<Write>,
    uniforms: Vec<(usize, usize)>,
}

impl<T: Real> Builder<T> {
    /// Build a [`Graph`] of kernels, the elementwise work fused around each
    /// matrix and dot product, with this thread's [`Algebra`] and the
    /// [`CostModel::BALANCED`] cost model. A builder without products builds
    /// a graph of one kernel, the program [`build`](Self::build) would.
    pub fn build_graph(self) -> Result<Graph<T>, ProgramError> {
        self.build_graph_with(&CostModel::BALANCED, algebra())
    }

    /// [`build_graph`](Self::build_graph) under a cost model and algebra of
    /// your choosing, which every kernel's program is optimized under (see
    /// [`build_with`](Self::build_with)).
    pub fn build_graph_with(
        mut self,
        model: &CostModel,
        algebra: Algebra,
    ) -> Result<Graph<T>, ProgramError> {
        // The graph's stores: the in-place tensors first, then the outputs.
        let mut roots: Vec<(usize, Dest)> = Vec::new();
        for (k, &(_, _, value)) in self.updates.iter().enumerate() {
            let value = value.ok_or(ProgramError::OutputNotStoredOnce { output: k as u8 })?;
            roots.push((value.id(), Dest::Update(k)));
        }
        for (k, &(_, value)) in self.outputs.iter().enumerate() {
            roots.push((value.id(), Dest::Output(k)));
        }
        if roots.is_empty() {
            return Err(ProgramError::NoOutputs);
        }

        // Every product the stores read, in order: operands come first.
        let mut products = Vec::new();
        let mut seen = HashSet::new();
        let mut pending: Vec<usize> = roots.iter().map(|&(id, _)| id).collect();
        while let Some(id) = pending.pop() {
            if !seen.insert(id) {
                continue;
            }
            match self.nodes[id] {
                Node::Binary(_, a, b) | Node::Cmp(_, a, b) => pending.extend([a, b]),
                Node::Unary(_, a) | Node::View(a, _) => pending.push(a),
                Node::MatMul(a, b) | Node::Dot(a, b) => {
                    products.push(id);
                    pending.extend([a, b]);
                }
                Node::Load(..) | Node::Const(_) | Node::Uniform(_) => {}
            }
        }
        products.sort_unstable();

        // Each product, after what it needs computed first. `done` is every
        // node a later kernel reads as a temporary rather than computing.
        let mut done = HashSet::new();
        let mut work = Vec::new();
        for &node in &products {
            match self.nodes[node] {
                Node::MatMul(a, b) => {
                    let a = self.product_operand(a, &mut done, &mut work);
                    let b = self.product_operand(b, &mut done, &mut work);
                    work.push(Work::Product { node, a, b });
                }
                Node::Dot(a, b) => {
                    let meta = self.join(self.var(a), self.var(b));
                    let root = self.node(Node::Binary(BinaryOp::Mul, a, b), meta);
                    work.push(Work::Sum {
                        node,
                        root,
                        done: done.clone(),
                    });
                }
                _ => unreachable!("only products are listed"),
            }
            done.insert(node);
        }
        work.push(Work::Final { done });

        // A region that reads a product as itself, at its shape, and after
        // everything else it reads, is that product's epilogue — unless it
        // updates tensors in place or reads row statistics, which an epilogue
        // cannot.
        let producer: HashMap<usize, usize> = work
            .iter()
            .enumerate()
            .filter_map(|(at, item)| match *item {
                Work::Materialize { root, .. } => Some((root, at)),
                Work::Product { node, .. } | Work::Sum { node, .. } => Some((node, at)),
                Work::Final { .. } => None,
            })
            .collect();
        let surveys: Vec<Option<Survey>> = work
            .iter()
            .map(|item| match item {
                Work::Materialize { root, done } => Some(self.survey(&[*root], done)),
                Work::Sum { root, done, .. } => Some(self.survey(&[*root], done)),
                Work::Final { done } => {
                    let ids: Vec<usize> = roots.iter().map(|&(id, _)| id).collect();
                    Some(self.survey(&ids, done))
                }
                Work::Product { .. } => None,
            })
            .collect();
        let mut epilogue: Vec<Option<usize>> = vec![None; work.len()];
        let mut fused = vec![false; work.len()];
        for (at, item) in work.iter().enumerate() {
            let space = match item {
                Work::Materialize { root, .. } => Some(self.meta[*root].dims),
                Work::Final { .. } if self.updates.is_empty() => self.space_of(&roots),
                _ => continue,
            };
            let survey = surveys[at].as_ref().expect("surveyed above");
            if survey.statistics {
                continue;
            }
            let Some((&temp, &direct)) = survey.temps.iter().max_by_key(|(temp, _)| producer[temp])
            else {
                continue;
            };
            let from = producer[&temp];
            let product = matches!(work[from], Work::Product { .. });
            if product && direct && epilogue[from].is_none() && space == Some(self.meta[temp].dims)
            {
                epilogue[from] = Some(at);
                fused[at] = true;
            }
        }

        // A product with an epilogue stores itself too only if something
        // else reads it.
        let read_elsewhere = |node: usize, epilogue: usize| {
            work.iter().enumerate().any(|(at, item)| {
                at != epilogue
                    && match item {
                        Work::Product { a, b, .. } => {
                            *a == Value::Temp(node) || *b == Value::Temp(node)
                        }
                        _ => surveys[at]
                            .as_ref()
                            .is_some_and(|survey| survey.temps.contains_key(&node)),
                    }
            })
        };

        let final_roots: Vec<(usize, Dest)> = roots.clone();
        let region_of = |item: &Work| -> (Vec<(usize, Dest)>, HashSet<usize>) {
            match item {
                Work::Materialize { root, done } => {
                    (vec![(*root, Dest::Temp(*root))], done.clone())
                }
                Work::Final { done } => (final_roots.clone(), done.clone()),
                _ => unreachable!("only regions are epilogues"),
            }
        };
        let mut stages = Vec::new();
        for (at, item) in work.iter().enumerate() {
            if fused[at] {
                continue;
            }
            let stage = match *item {
                Work::Product { node, a, b } => match epilogue[at] {
                    None => Stage {
                        kind: Kind::Product { a, b },
                        program: None,
                        reads: Vec::new(),
                        writes: vec![Write::Temp(node)],
                        uniforms: Vec::new(),
                    },
                    Some(region) => {
                        let (region_roots, region_done) = region_of(&work[region]);
                        let keep = read_elsewhere(node, region);
                        let built = self.region(
                            &region_roots,
                            &region_done,
                            Some(node),
                            keep,
                            model,
                            algebra,
                        )?;
                        Stage {
                            kind: Kind::Product { a, b },
                            program: Some(built.program),
                            reads: built.reads[1..].to_vec(),
                            writes: built.writes,
                            uniforms: built.uniforms,
                        }
                    }
                },
                Work::Sum {
                    node,
                    root,
                    ref done,
                } => {
                    let built = self.region(
                        &[(root, Dest::Temp(node))],
                        done,
                        None,
                        false,
                        model,
                        algebra,
                    )?;
                    Stage {
                        kind: Kind::Sum {
                            shape: self.meta[node].dims,
                        },
                        program: Some(built.program),
                        reads: built.reads,
                        writes: built.writes,
                        uniforms: built.uniforms,
                    }
                }
                // Outputs that are each a temporary already of the graph's
                // shape need no kernel to write them.
                Work::Final { ref done } if self.stored_as_computed(&roots, done) => Stage {
                    kind: Kind::Store,
                    program: None,
                    reads: roots.iter().map(|&(id, _)| Value::Temp(id)).collect(),
                    writes: (0..self.outputs.len()).map(Write::Output).collect(),
                    uniforms: Vec::new(),
                },
                Work::Materialize { .. } | Work::Final { .. } => {
                    let (region_roots, region_done) = region_of(item);
                    let built =
                        self.region(&region_roots, &region_done, None, false, model, algebra)?;
                    Stage {
                        kind: Kind::Elementwise {
                            last: matches!(item, Work::Final { .. }),
                        },
                        program: Some(built.program),
                        reads: built.reads,
                        writes: built.writes,
                        uniforms: built.uniforms,
                    }
                }
            };
            stages.push(stage);
        }

        let space = self
            .space_of(&roots)
            .ok_or(ProgramError::OutputShape { output: 0 })?;
        Ok(Graph {
            stages,
            inputs: self.inputs.clone(),
            updates: self
                .updates
                .iter()
                .map(|&(dtype, dims, _)| (dtype, dims))
                .collect(),
            outputs: self.outputs.iter().map(|&(dtype, _)| dtype).collect(),
            space,
            uniforms: self.uniforms.clone(),
        })
    }

    /// What a matrix product reads as operand `x`: a given matrix where it
    /// lies, a temporary already computed, or — scheduled now — `x` computed
    /// into a temporary of its own.
    fn product_operand(&self, x: usize, done: &mut HashSet<usize>, work: &mut Vec<Work>) -> Value {
        if done.contains(&x) {
            return Value::Temp(x);
        }
        if let Node::Load(slot, remap) = self.nodes[x]
            && usize::from(slot) < MAX_INPUTS
            && remap.is_identity()
            && self.inputs[usize::from(slot)].1 == self.meta[x].dims
        {
            return Value::Input(usize::from(slot));
        }
        work.push(Work::Materialize {
            root: x,
            done: done.clone(),
        });
        done.insert(x);
        Value::Temp(x)
    }

    /// Whether `roots` are only outputs, each a different temporary in `done`
    /// already of the shape they all have.
    fn stored_as_computed(&self, roots: &[(usize, Dest)], done: &HashSet<usize>) -> bool {
        let Some(space) = self.space_of(roots) else {
            return false;
        };
        let distinct: HashSet<usize> = roots.iter().map(|&(id, _)| id).collect();
        distinct.len() == roots.len()
            && roots.iter().all(|&(id, dest)| {
                matches!(dest, Dest::Output(_)) && done.contains(&id) && self.meta[id].dims == space
            })
    }

    /// The shape `roots` and the in-place tensors broadcast to, if they do.
    fn space_of(&self, roots: &[(usize, Dest)]) -> Option<Dims> {
        let mut space = Dims::default();
        for &(_, dims, _) in &self.updates {
            space = broadcast_shape(&space, &dims)?;
        }
        for &(id, _) in roots {
            space = broadcast_shape(&space, &self.meta[id].dims)?;
        }
        Some(space)
    }

    /// The temporaries the region of `roots` reads, stopping at those in
    /// `done`, and whether it reads row statistics.
    fn survey(&self, roots: &[usize], done: &HashSet<usize>) -> Survey {
        let mut survey = Survey {
            temps: HashMap::new(),
            statistics: false,
        };
        let mut seen = HashSet::new();
        let mut pending = roots.to_vec();
        while let Some(id) = pending.pop() {
            if !seen.insert(id) {
                continue;
            }
            if done.contains(&id) {
                survey.temps.entry(id).or_insert(true);
                continue;
            }
            match self.nodes[id] {
                Node::Load(slot, _) => survey.statistics |= usize::from(slot) >= 2 * MAX_INPUTS,
                Node::View(of, _) => {
                    survey.temps.insert(of, false);
                }
                Node::Binary(_, a, b) | Node::Cmp(_, a, b) => pending.extend([a, b]),
                Node::Unary(_, a) => pending.push(a),
                Node::Const(_) | Node::Uniform(_) => {}
                Node::MatMul(..) | Node::Dot(..) => unreachable!("products are done first"),
            }
        }
        survey
    }

    /// The program computing `roots`, reading the nodes in `done` as
    /// temporaries. With `product`, that temporary is the program's input 0 —
    /// an epilogue's — and with `keep` it is also stored as is, last.
    fn region(
        &self,
        roots: &[(usize, Dest)],
        done: &HashSet<usize>,
        product: Option<usize>,
        keep: bool,
        model: &CostModel,
        algebra: Algebra,
    ) -> Result<Built<T>, ProgramError> {
        let mut copier = Copier::new(self, done);
        let first = product.map(|node| {
            let meta = Meta {
                dtype: dtype_of::<T>(),
                ..self.meta[node]
            };
            copier.read(Value::Temp(node), meta)
        });
        if roots
            .iter()
            .any(|&(_, dest)| matches!(dest, Dest::Update(_)))
        {
            for &(dtype, dims, _) in &self.updates {
                let current = copier.to.update(Decl::of(dtype, &dims));
                copier.updates.push(current);
            }
        }
        let mut writes = Vec::new();
        for &(id, dest) in roots {
            let value = copier.copy(id);
            match dest {
                Dest::Temp(node) => {
                    copier.to.output(value, dtype_of::<T>());
                    writes.push(Write::Temp(node));
                }
                Dest::Output(k) => {
                    copier.to.output(value, self.outputs[k].0);
                    writes.push(Write::Output(k));
                }
                Dest::Update(k) => copier.to.set(k, value),
            }
        }
        if let (true, Some(node), Some(first)) = (keep, product, first) {
            copier.to.output(first, dtype_of::<T>());
            writes.push(Write::Temp(node));
        }
        let Copier {
            to,
            reads,
            uniforms,
            ..
        } = copier;
        Ok(Built {
            program: to.build_with(model, algebra)?,
            reads,
            writes,
            uniforms,
        })
    }
}

/// Copies one region of a builder's values into a builder of its own, which
/// reads the region's temporaries and inputs as inputs.
struct Copier<'b, T> {
    from: &'b Builder<T>,
    to: Builder<T>,
    done: &'b HashSet<usize>,
    copied: HashMap<usize, Var>,
    /// What each of `to`'s input slots reads, and the value loading it.
    reads: Vec<Value>,
    loads: HashMap<Value, Var>,
    /// Each in-place tensor's current value, when `to` updates them.
    updates: Vec<Var>,
    /// Each uniform of `from` that `to` reads, and its index in `to`.
    uniforms: Vec<(usize, usize)>,
}

impl<'b, T: Real> Copier<'b, T> {
    fn new(from: &'b Builder<T>, done: &'b HashSet<usize>) -> Self {
        Copier {
            from,
            to: Builder::new(),
            done,
            copied: HashMap::new(),
            reads: Vec::new(),
            loads: HashMap::new(),
            updates: Vec::new(),
            uniforms: Vec::new(),
        }
    }

    /// `value`, of `meta`'s storage type and shape, as an input of `to`.
    fn read(&mut self, value: Value, meta: Meta) -> Var {
        if let Some(&var) = self.loads.get(&value) {
            return var;
        }
        let var = self.to.input(Decl::of(meta.dtype, &meta.dims));
        self.reads.push(value);
        self.loads.insert(value, var);
        var
    }

    /// Node `id` of `from`, copied into `to`.
    fn copy(&mut self, id: usize) -> Var {
        if let Some(&var) = self.copied.get(&id) {
            return var;
        }
        let meta = self.from.meta[id];
        let var = if self.done.contains(&id) {
            let meta = Meta {
                dtype: dtype_of::<T>(),
                ..meta
            };
            self.read(Value::Temp(id), meta)
        } else {
            match self.from.nodes[id] {
                Node::Load(slot, remap) => {
                    let slot = usize::from(slot);
                    let input = |(dtype, dims): (DType, Dims)| Meta {
                        dims,
                        dtype,
                        tensor: false,
                    };
                    let whole = if slot < MAX_INPUTS {
                        self.read(Value::Input(slot), input(self.from.inputs[slot]))
                    } else if slot < 2 * MAX_INPUTS {
                        let k = slot - MAX_INPUTS;
                        match self.updates.get(k) {
                            Some(&current) => current,
                            None => {
                                let (dtype, dims, _) = self.from.updates[k];
                                self.read(Value::Update(k), input((dtype, dims)))
                            }
                        }
                    } else {
                        let (of, statistic) = self.from.derived[slot - 2 * MAX_INPUTS];
                        let of = usize::from(of);
                        let x = self.read(Value::Input(of), input(self.from.inputs[of]));
                        self.to.row_statistic(x, statistic)
                    };
                    self.remapped(whole, remap, meta.dims)
                }
                Node::Const(value) => self.to.constant(value),
                Node::Uniform(k) => {
                    let var = self.to.uniform(self.from.uniforms[k]);
                    self.uniforms.push((k, self.to.uniforms.len() - 1));
                    var
                }
                Node::Binary(op, a, b) => {
                    let (a, b) = (self.copy(a), self.copy(b));
                    self.to.binary(op, a, b)
                }
                Node::Unary(op, a) => {
                    let a = self.copy(a);
                    self.to.unary(op, a)
                }
                Node::Cmp(op, a, b) => {
                    let (a, b) = (self.copy(a), self.copy(b));
                    self.to.compare(op, a, b)
                }
                Node::View(of, remap) => {
                    let whole = self.copy(of);
                    self.remapped(whole, remap, meta.dims)
                }
                Node::MatMul(..) | Node::Dot(..) => unreachable!("products are done first"),
            }
        };
        self.copied.insert(id, var);
        var
    }

    /// `whole` read through `remap` as a value of shape `dims`.
    fn remapped(&mut self, whole: Var, remap: Remap, dims: Dims) -> Var {
        if remap.is_identity() && self.to.meta[whole.id()].dims == dims {
            return whole;
        }
        let id = self.to.reindex(whole.id(), remap, dims);
        self.to.var(id)
    }
}

/// A product's operand: a given matrix where it lies, or one of the graph's
/// own.
enum Operand<'a, T, B: crate::tensors::Backend> {
    Borrowed(&'a Matrix<T, B>),
    Owned(Matrix<T, B>),
}

impl<T, B: crate::tensors::Backend> std::ops::Deref for Operand<'_, T, B> {
    type Target = Matrix<T, B>;

    fn deref(&self) -> &Matrix<T, B> {
        match self {
            Operand::Borrowed(matrix) => matrix,
            Operand::Owned(matrix) => matrix,
        }
    }
}

impl<T: Real> Graph<T> {
    /// How many kernels a run launches.
    pub fn kernels(&self) -> usize {
        self.stages
            .iter()
            .filter(|stage| !matches!(stage.kind, Kind::Store))
            .count()
    }

    /// The shape of every output, and of every tensor updated in place.
    pub fn space(&self) -> &[usize] {
        &self.space
    }

    /// The current value of every uniform, in declaration order (see
    /// [`Builder::uniform`]).
    pub fn uniforms(&self) -> &[T] {
        &self.uniforms
    }

    /// Set uniform `index` to `value` in every kernel that reads it.
    ///
    /// # Panics
    ///
    /// If the graph has no uniform `index`.
    #[track_caller]
    pub fn set_uniform(&mut self, index: usize, value: T) {
        assert!(
            index < self.uniforms.len(),
            "fused graph: uniform {index} out of {}",
            self.uniforms.len()
        );
        self.uniforms[index] = value;
        for stage in &mut self.stages {
            for &(graph, local) in &stage.uniforms {
                if graph == index {
                    let program = stage.program.as_mut().expect("a uniform is a program's");
                    program.set_uniform(local, value);
                }
            }
        }
    }
}

impl<T: Element> Graph<T> {
    /// Run every kernel in order.
    ///
    /// `inputs` are the given inputs, in declaration order, and `updated` the
    /// tensors updated in place, which are overwritten only by the last kernel,
    /// after every other has read them. The fresh outputs come back in
    /// declaration order, each of the graph's [`space`](Self::space).
    ///
    /// # Panics
    ///
    /// If the counts, storage types or shapes disagree with the graph.
    #[track_caller]
    pub fn run<B: Kernels<T>>(
        &self,
        inputs: &[&dyn Fusable<B>],
        updated: &mut [&mut dyn FusableMut<B>],
    ) -> Vec<Output<B>> {
        assert_eq!(
            inputs.len(),
            self.inputs.len(),
            "fused graph: expected {} inputs, got {}",
            self.inputs.len(),
            inputs.len()
        );
        assert_eq!(
            updated.len(),
            self.updates.len(),
            "fused graph: expected {} in-place tensors, got {}",
            self.updates.len(),
            updated.len()
        );
        let mut temps: HashMap<usize, Matrix<T, B>> = HashMap::new();
        let mut outputs: Vec<Option<Output<B>>> = self.outputs.iter().map(|_| None).collect();
        for stage in &self.stages {
            let fresh = match (&stage.kind, &stage.program) {
                (Kind::Elementwise { last: true }, Some(program)) => {
                    let reads = Self::reads(&stage.reads, inputs, None, &temps);
                    program.run(&reads, updated)
                }
                (Kind::Elementwise { last: false }, Some(program)) => {
                    let reads = Self::reads(&stage.reads, inputs, Some(updated), &temps);
                    program.run(&reads, &mut [])
                }
                (Kind::Product { a, b }, program) => {
                    let a = self.operand(*a, inputs, &temps);
                    let b = self.operand(*b, inputs, &temps);
                    match program {
                        Some(program) => {
                            let reads = Self::reads(&stage.reads, inputs, Some(updated), &temps);
                            program.run_matmul(&a, &b, &reads)
                        }
                        None => {
                            let product = B::matmul(&a, &b);
                            drop((a, b));
                            temps.insert(Self::temp(stage.writes[0]), product);
                            continue;
                        }
                    }
                }
                (Kind::Sum { shape }, Some(program)) => {
                    let sums = {
                        let reads = Self::reads(&stage.reads, inputs, Some(updated), &temps);
                        if program.row_statistics().is_empty() {
                            program.run_sum(&reads, Axis::Rows)
                        } else {
                            let products = program.run(&reads, &mut []).remove(0);
                            axis_sum(products.into_matrix::<T>(), Axis::Rows)
                        }
                    };
                    temps.insert(Self::temp(stage.writes[0]), matrix_of(sums, shape));
                    continue;
                }
                (Kind::Store, _) => {
                    for (&value, &write) in stage.reads.iter().zip(&stage.writes) {
                        let (Value::Temp(node), Write::Output(k)) = (value, write) else {
                            unreachable!("a store hands temporaries to outputs");
                        };
                        let matrix = temps.remove(&node).expect("computed by an earlier kernel");
                        outputs[k] =
                            Some(Output::new(&self.space, narrow(matrix, self.outputs[k])));
                    }
                    continue;
                }
                (_, None) => unreachable!("only a product runs without a program"),
            };
            for (output, &write) in fresh.into_iter().zip(&stage.writes) {
                match write {
                    Write::Temp(node) => {
                        temps.insert(node, output.into_matrix::<T>());
                    }
                    Write::Output(k) => outputs[k] = Some(output),
                }
            }
        }
        outputs
            .into_iter()
            .map(|output| output.expect("the last kernel writes every output"))
            .collect()
    }

    fn temp(write: Write) -> usize {
        match write {
            Write::Temp(node) => node,
            Write::Output(_) => unreachable!("a product or sum alone writes a temporary"),
        }
    }

    /// The operands a program reads. The last kernel, which overwrites the
    /// in-place tensors, never reads them as plain inputs.
    fn reads<'a, B: Kernels<T>>(
        values: &[Value],
        inputs: &[&'a dyn Fusable<B>],
        updated: Option<&'a [&mut dyn FusableMut<B>]>,
        temps: &'a HashMap<usize, Matrix<T, B>>,
    ) -> Vec<&'a dyn Fusable<B>> {
        values
            .iter()
            .map(|&value| match value {
                Value::Input(k) => inputs[k],
                Value::Update(k) => {
                    let updated = updated.expect("only the last kernel updates in place");
                    &*updated[k] as &dyn Fusable<B>
                }
                Value::Temp(node) => &temps[&node] as &dyn Fusable<B>,
            })
            .collect()
    }

    /// A product's operand as a matrix of `T`: a given matrix of `T` where it
    /// lies, and anything else — a view, a vector or a tensor of its shape,
    /// another storage type — copied into one.
    #[track_caller]
    fn operand<'a, B: Kernels<T>>(
        &self,
        value: Value,
        inputs: &[&'a dyn Fusable<B>],
        temps: &'a HashMap<usize, Matrix<T, B>>,
    ) -> Operand<'a, T, B> {
        let k = match value {
            Value::Temp(node) => return Operand::Borrowed(&temps[&node]),
            Value::Input(k) => k,
            Value::Update(_) => unreachable!("an in-place tensor is copied before a product"),
        };
        let (dtype, dims) = self.inputs[k];
        let (rows, cols) = (dims[0], dims[1]);
        let source = inputs[k].source();
        assert_eq!(
            source.dtype(),
            dtype,
            "fused graph: input {k} is {} but the graph reads {}",
            source.dtype().name(),
            dtype.name()
        );
        let matrix = inputs[k]
            .as_any()
            .and_then(|any| any.downcast_ref::<Matrix<T, B>>());
        if let Some(matrix) = matrix {
            assert_eq!(
                matrix.shape(),
                (rows, cols),
                "fused graph: input {k} is a {}×{} matrix, but the graph multiplies a {rows}×{cols} one",
                matrix.rows(),
                matrix.cols()
            );
            return Operand::Borrowed(matrix);
        }
        assert_eq!(
            source.len,
            rows * cols,
            "fused graph: input {k} holds {} elements, but its shape {:?} needs {}",
            source.len,
            dims.as_slice(),
            rows * cols
        );
        let place = match source.view {
            None => Place::grid(0, cols, 1, rows),
            Some(view) => {
                let strides = view.read_as(&dims).unwrap_or_else(|| {
                    panic!(
                        "fused graph: input {k} is a view of shape {:?}, which cannot be read as \
                         {:?}",
                        view.dims.as_slice(),
                        dims.as_slice()
                    )
                });
                Place::grid(view.offset, strides[0], strides[1], rows)
            }
        };
        Operand::Owned(load_unfused(&source, (rows, cols), place))
    }
}

/// `values` as a matrix of a tensor of `shape`: its last axis by the others
/// folded together.
fn matrix_of<T: Copy + 'static, B: crate::tensors::Backend>(
    values: Vector<T, B>,
    shape: &[usize],
) -> Matrix<T, B> {
    let (rows, cols) = match *shape {
        [] => (1, 1),
        [ref leading @ .., last] => (leading.iter().product(), last),
    };
    Matrix::from_storage(rows, cols, values.into_storage())
}

impl<T: Real + fmt::Debug> fmt::Display for Graph<T> {
    /// Each kernel in order, with its program's disassembly.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "graph over {:?}, {} kernels", self.space, self.kernels())?;
        for (at, stage) in self.stages.iter().enumerate() {
            let reads: Vec<String> = stage.reads.iter().map(Value::to_string).collect();
            let writes: Vec<String> = stage.writes.iter().map(Write::to_string).collect();
            match &stage.kind {
                Kind::Elementwise { .. } => writeln!(
                    f,
                    "kernel {at}: ({}) -> ({})",
                    reads.join(", "),
                    writes.join(", ")
                )?,
                Kind::Product { a, b } => writeln!(
                    f,
                    "kernel {at}: {a}·{b}, then ({}) -> ({})",
                    reads.join(", "),
                    writes.join(", ")
                )?,
                Kind::Sum { shape } => writeln!(
                    f,
                    "kernel {at}: row sums of ({}) -> {} {:?}",
                    reads.join(", "),
                    writes.join(", "),
                    shape
                )?,
                Kind::Store => {
                    writeln!(f, "then: ({}) -> ({})", reads.join(", "), writes.join(", "))?
                }
            }
            if let Some(program) = &stage.program {
                for line in program.to_string().lines() {
                    writeln!(f, "    {line}")?;
                }
            }
        }
        Ok(())
    }
}
