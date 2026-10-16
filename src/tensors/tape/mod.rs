//! Reverse-mode automatic differentiation.
//!
//! Forward mode ([`dual`](super::dual)) carries a tangent alongside every value
//! and costs one pass per *input*. Reverse mode records the computation and then
//! walks it backwards, costing one pass per *output* — so a scalar loss over a
//! whole weight matrix takes a single backward pass instead of one per element.
//!
//! # Using it
//!
//! ```
//! use tensorcrate::tensors::{Matrix, Tape, Vector};
//!
//! let tape = Tape::new();
//! let a = tape.matrix(Matrix::<f32>::from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
//! let x = tape.vector(Vector::new([1.0f32, 0.5, -1.0]));
//!
//! let mapped = a.matvec(&x);        // y = A·x
//! let loss = mapped.dot(&mapped);   // f = ‖A·x‖²
//! loss.backward();
//!
//! // ∂f/∂A = 2(Ax)xᵀ and ∂f/∂x = 2Aᵀ(Ax), both from that one pass.
//! let projected = a.value().matvec(x.value());
//! for row in 0..2 {
//!     for col in 0..3 {
//!         let expected = 2.0 * projected[row] * x.value()[col];
//!         assert!((a.grad()[(row, col)] - expected).abs() < 1e-5);
//!     }
//! }
//! ```
//!
//! # How it is built
//!
//! [`Tape`] owns the nodes in creation order, which is already a topological
//! order because an operation is recorded after its operands. [`Var`] is a cheap
//! handle onto one node — an `Rc` and a tape borrow — so passing values around
//! copies nothing.
//!
//! Each node stores a closure that pushes adjoints into its operands. Those
//! closures capture their operands' nodes, so a rule reads whatever shapes it
//! needs from the values recorded there rather than re-deriving them. Adjoints
//! start as `None` rather than zeros, which keeps untouched nodes from
//! allocating (and, on `Metal`, from dispatching) — and is why the zero an
//! adjoint falls back to is built *from* a recorded value, by
//! [`Adjoint::zeros_like`], instead of from a type.
//!
//! Every rule is expressed with the [`Kernels`] trait, so reverse mode runs on
//! either backend and a `Metal` graph stays resident from the forward pass right
//! through the backward one. Two pieces of earlier machinery do most of the
//! work: the accumulating matmul (`Ā += C̄·Bᵀ` in one dispatch) and the
//! `unary_dual` kernel, whose tangent output `f'(a) ⊙ d` is exactly the
//! vector-Jacobian product an analytic node needs — reverse mode adds no shaders
//! at all.
//!
//! # What is differentiable
//!
//! Scalars, vectors, matrices and N-dimensional tensors of any [`Real`]
//! element type (`f32` by default), on every backend that implements
//! [`Kernels`] for it — `Host` for all of them, `Metal` for `f32`, `f16` and
//! `bf16`. For scalars, vectors and matrices:
//!
//! - every [`Analytic`] function, elementwise;
//! - the binary operations `+ - * / %`, elementwise, and as operators on `&Var`;
//! - the comparisons [`maximum`](VectorVar::maximum) and
//!   [`minimum`](VectorVar::minimum), against another tensor or a constant, and
//!   the [`abs`](VectorVar::abs), [`relu`](VectorVar::relu) and
//!   [`clamp`](VectorVar::clamp) built on them — the nonsmooth family behind L1,
//!   Huber and hinge losses, with ties splitting the subgradient (see
//!   [`Compare`]);
//! - negation, and scaling or shifting by a constant;
//! - products: [`matmul`](MatrixVar::matmul), [`matvec`](MatrixVar::matvec),
//!   their fused multiply-add forms, [`vecmat`](VectorVar::vecmat), and
//!   [`dot`](VectorVar::dot);
//! - reductions to a scalar: [`sum`](VectorVar::sum), [`dot`](VectorVar::dot);
//! - reductions along one axis: [`row_sums`](MatrixVar::row_sums) and
//!   [`column_sums`](MatrixVar::column_sums), which together with
//!   [`outer`](VectorVar::outer) express a row-wise softmax and cross-entropy;
//! - reshaping: [`transpose`](MatrixVar::transpose),
//!   [`flattened`](MatrixVar::flattened), [`into_row`](VectorVar::into_row),
//!   [`into_column`](VectorVar::into_column);
//! - broadcasting a differentiable scalar across a tensor with
//!   [`expand`](ScalarVar::expand), whose backward is the sum of the adjoint.
//!
//! Everything placed on the tape is a leaf whose gradient can be read with
//! [`grad`](Var::grad) after a backward pass.
//!
//! # Tensors
//!
//! [`Tape::tensor`] records a [`Tensor`] of any rank as a [`TensorVar`], and
//! the tensor operations differentiate:
//!
//! - `+ - * /` (as methods and as operators on `&TensorVar`),
//!   [`power`](TensorVar::power), [`maximum`](TensorVar::maximum) and
//!   [`minimum`](TensorVar::minimum), all broadcasting numpy-style as
//!   [`Tensor`]'s do. An operand that was broadcast receives its adjoint
//!   summed back down to its own shape — one reduction over the axes it was
//!   repeated along — so a `[D]` bias added to `[B, T, D]` activations gets
//!   the sum over the batch and the positions;
//! - every [`Analytic`] function, [`power_scalar`](TensorVar::power_scalar),
//!   negation, scaling and shifting by a constant, and
//!   [`abs`](TensorVar::abs), [`relu`](TensorVar::relu) and
//!   [`clamp`](TensorVar::clamp), with ties splitting as for vectors;
//! - layout: [`reshape`](TensorVar::reshape),
//!   [`permute`](TensorVar::permute), [`transpose`](TensorVar::transpose),
//!   [`narrow`](TensorVar::narrow), [`slice`](TensorVar::slice),
//!   [`select`](TensorVar::select), [`split`](TensorVar::split),
//!   [`chunk`](TensorVar::chunk), [`concat`](TensorVar::concat),
//!   [`stack`](TensorVar::stack), [`squeeze`](TensorVar::squeeze),
//!   [`unsqueeze`](TensorVar::unsqueeze),
//!   [`broadcast_to`](TensorVar::broadcast_to) and
//!   [`contiguous`](TensorVar::contiguous);
//! - reductions: [`sum_axes`](TensorVar::sum_axes),
//!   [`mean_axes`](TensorVar::mean_axes), [`var_axes`](TensorVar::var_axes)
//!   under either correction, [`max_axes`](TensorVar::max_axes) and
//!   [`min_axes`](TensorVar::min_axes) — whose adjoint goes to the extreme
//!   element, split evenly among elements that tie for it, the many-way form
//!   of [`Compare::MaxShare`]'s half each — and the whole-tensor
//!   [`sum`](TensorVar::sum) and [`mean`](TensorVar::mean) to a scalar;
//! - conversions to and from the matrix and vector variables —
//!   [`to_matrix`](TensorVar::to_matrix), [`to_vector`](TensorVar::to_vector),
//!   [`MatrixVar::to_tensor`], [`VectorVar::to_tensor`] — so a tensor
//!   computation can use [`matmul`](MatrixVar::matmul) and the other matrix
//!   products.
//!
//! A value on the tape is a whole, contiguous tensor, so a layout operation
//! records a copy of the elements it reads — one strided copy on the backend
//! — rather than a view. Its adjoint goes back the same way: a permutation's
//! by the inverse permutation, a concatenation's by narrowing each piece's
//! part back out, and a slice's — from `narrow`, `slice`, `select`, `split`
//! or `chunk` — by writing it into the matching part of the parent's
//! adjoint in place, a strided write that adds to whatever is already there
//! and leaves the rest untouched. Every rule is a tensor operation on the
//! backend, so on `Metal` the backward pass stays on the GPU.
//!
//! ```
//! use tensorcrate::tensors::{Tape, Tensor};
//!
//! let tape = Tape::new();
//! // A batch of 2 sequences of 3 positions, each 4 features wide, and a bias.
//! let features = (0..24).map(|i| i as f32 * 0.1).collect::<Vec<_>>();
//! let x = tape.tensor(Tensor::from_vec(&[2, 3, 4], features));
//! let bias = tape.tensor(Tensor::from_vec(&[4], vec![0.5f32, -0.5, 0.25, 0.0]));
//!
//! // Two heads of two features, brought forward; the first position dropped.
//! let heads = (&x + &bias).reshape(&[2, 3, 2, 2]).permute(&[0, 2, 1, 3]);
//! let tail = heads.narrow(2, 1, 2);
//! let loss = tail.tanh().max_axes(-1, false).sum();
//! loss.backward();
//!
//! assert_eq!(x.grad().shape(), [2, 3, 4]);
//! assert_eq!(bias.grad().shape(), [4]); // summed over the batch and positions
//! // The dropped position receives nothing.
//! assert!(x.grad().as_slice()[..4].iter().all(|&g| g == 0.0));
//! ```

use std::cell::{Cell, OnceCell, RefCell};
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Rem, Sub};
use std::rc::Rc;

use super::{
    Analytic, Backend, BinaryOp, Compare, Host, Kernels, Matrix, Tensor, Transposed, Vector,
};
use crate::numbers::Real;

mod tensor;

// ---- what a node can hold ---------------------------------------------------

/// A value a tape node can carry: a [`Real`] scalar, or a tensor of one on
/// backend `B`.
///
/// The methods are what adjoint bookkeeping needs — a zero to start from, a copy,
/// and a sum — expressed without a `Clone` bound, which on a generic backend
/// would infect every signature that touches a node.
pub trait Adjoint<B: Backend>: Sized + 'static {
    /// The additive identity of *this value's* shape.
    ///
    /// A zero adjoint has to match the value it belongs to, and the shape is a
    /// runtime value the type does not carry — so the zero is built from an
    /// existing value rather than conjured from nothing.
    fn zeros_like(&self) -> Self;

    /// A second copy of this value on the same backend.
    fn duplicate(&self) -> Self;

    /// Elementwise sum, used to accumulate contributions from several consumers.
    fn add(&self, other: &Self) -> Self;
}

impl<T: Real, B: Kernels<T>> Adjoint<B> for T {
    fn zeros_like(&self) -> Self {
        T::zero()
    }

    fn duplicate(&self) -> Self {
        *self
    }

    fn add(&self, other: &Self) -> Self {
        *self + *other
    }
}

impl<T: Real, B: Kernels<T>> Adjoint<B> for Vector<T, B> {
    fn zeros_like(&self) -> Self {
        Vector::filled(self.len(), T::zero())
    }

    fn duplicate(&self) -> Self {
        self.to_backend::<B>()
    }

    fn add(&self, other: &Self) -> Self {
        B::vector_elementwise(self, other, BinaryOp::Add)
    }
}

impl<T: Real, B: Kernels<T>> Adjoint<B> for Matrix<T, B> {
    fn zeros_like(&self) -> Self {
        let (rows, cols) = self.shape();
        Matrix::filled(rows, cols, T::zero())
    }

    fn duplicate(&self) -> Self {
        self.to_backend::<B>()
    }

    fn add(&self, other: &Self) -> Self {
        B::matrix_elementwise(self, other, BinaryOp::Add)
    }
}

impl<T: Real, B: Kernels<T>> Adjoint<B> for Tensor<T, B> {
    fn zeros_like(&self) -> Self {
        Tensor::filled(self.shape(), T::zero())
    }

    fn duplicate(&self) -> Self {
        self.contiguous()
    }

    fn add(&self, other: &Self) -> Self {
        self.elementwise(other, BinaryOp::Add)
    }
}

// ---- nodes and the tape -----------------------------------------------------

/// What a node does with its adjoint: push contributions into its operands.
type Rule<T> = Box<dyn Fn(&T)>;

/// The type-erased face of a node: run its rule, or forget its adjoint.
trait Backprop {
    fn propagate(&self);
    fn clear(&self);
}

/// A node's value: computed when the node is recorded, or — for a scalar
/// reduced from a tensor — when it is first needed.
///
/// Reading a reduction of a device tensor waits for the device, and nothing
/// may need the value: a loss is usually only differentiated, and its backward
/// pass needs its adjoint, not its value. So a reduction is recorded unread,
/// scalar operations on an unread value are themselves unread, and the wait
/// happens when something reads one — [`Var::value`], or a rule whose
/// derivative depends on it.
struct Lazy<T> {
    value: OnceCell<T>,
    pending: Cell<Option<Box<dyn FnOnce() -> T>>>,
}

impl<T> Lazy<T> {
    fn ready(value: T) -> Self {
        Lazy {
            value: OnceCell::from(value),
            pending: Cell::new(None),
        }
    }

    fn deferred(compute: impl FnOnce() -> T + 'static) -> Self {
        Lazy {
            value: OnceCell::new(),
            pending: Cell::new(Some(Box::new(compute))),
        }
    }

    fn get(&self) -> &T {
        self.value.get_or_init(|| {
            let compute = self.pending.take().expect("a lazy value is computed once");
            compute()
        })
    }

    fn is_ready(&self) -> bool {
        self.value.get().is_some()
    }
}

struct Node<T, B: Backend> {
    /// Position in the tape, which is also this node's place in topological
    /// order.
    index: usize,
    value: Lazy<T>,
    /// `None` is a zero adjoint that has not been allocated.
    adjoint: RefCell<Option<T>>,
    /// Pushes this node's adjoint into its operands. `None` for a leaf.
    rule: Option<Rule<T>>,
    marker: PhantomData<B>,
}

impl<T: Adjoint<B>, B: Backend> Node<T, B> {
    fn value(&self) -> &T {
        self.value.get()
    }

    fn accumulate(&self, delta: T) {
        let mut adjoint = self.adjoint.borrow_mut();
        *adjoint = Some(match adjoint.as_ref() {
            Some(current) => current.add(&delta),
            None => delta,
        });
    }

    /// The adjoint accumulated so far, or a fresh zero of this node's shape —
    /// the starting point for an accumulating matmul.
    fn current_or_zeros(&self) -> T {
        match self.adjoint.borrow().as_ref() {
            Some(current) => current.duplicate(),
            None => self.value().zeros_like(),
        }
    }
}

impl<T: Adjoint<B>, B: Backend> Backprop for Node<T, B> {
    fn propagate(&self) {
        // A node with no adjoint contributed nothing, and a leaf has no rule.
        let adjoint = self.adjoint.borrow();
        if let (Some(adjoint), Some(rule)) = (adjoint.as_ref(), self.rule.as_ref()) {
            rule(adjoint);
        }
    }

    fn clear(&self) {
        *self.adjoint.borrow_mut() = None;
    }
}

/// A recording of a computation, in the order it happened.
pub struct Tape<B: Backend = Host> {
    nodes: RefCell<Vec<Rc<dyn Backprop>>>,
    marker: PhantomData<B>,
}

impl<B: Backend> Default for Tape<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Tape<B> {
    /// An empty tape.
    ///
    /// The backend is normally inferred from the first tensor recorded on it. A
    /// tape holding only scalars has nothing to infer from and needs naming:
    /// `Tape::<Host>::new()`.
    pub fn new() -> Self {
        Tape {
            nodes: RefCell::new(Vec::new()),
            marker: PhantomData,
        }
    }

    /// Record a scalar input.
    pub fn scalar<T: Real>(&self, value: T) -> ScalarVar<'_, B, T>
    where
        B: Kernels<T>,
    {
        self.push(value, None)
    }

    /// Record a vector input.
    pub fn vector<T: Real>(&self, value: Vector<T, B>) -> VectorVar<'_, B, T>
    where
        B: Kernels<T>,
    {
        self.push(value, None)
    }

    /// Record a matrix input.
    pub fn matrix<T: Real>(&self, value: Matrix<T, B>) -> MatrixVar<'_, B, T>
    where
        B: Kernels<T>,
    {
        self.push(value, None)
    }

    /// Record an N-dimensional tensor input.
    pub fn tensor<T: Real>(&self, value: Tensor<T, B>) -> TensorVar<'_, B, T>
    where
        B: Kernels<T>,
    {
        self.push(value, None)
    }

    /// How many nodes have been recorded.
    pub fn len(&self) -> usize {
        self.nodes.borrow().len()
    }

    /// Whether nothing has been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Forget every adjoint, keeping the graph. Without this, a second
    /// [`backward`](ScalarVar::backward) adds to the gradients already there.
    pub fn zero_grad(&self) {
        for node in self.nodes.borrow().iter() {
            node.clear();
        }
    }

    fn push<T: Adjoint<B>>(&self, value: T, rule: Option<Rule<T>>) -> Var<'_, T, B> {
        self.push_lazy(Lazy::ready(value), rule)
    }

    fn push_lazy<T: Adjoint<B>>(&self, value: Lazy<T>, rule: Option<Rule<T>>) -> Var<'_, T, B> {
        let mut nodes = self.nodes.borrow_mut();
        let node = Rc::new(Node {
            index: nodes.len(),
            value,
            adjoint: RefCell::new(None),
            rule,
            marker: PhantomData,
        });
        nodes.push(node.clone() as Rc<dyn Backprop>);
        Var { tape: self, node }
    }
}

/// A handle on one recorded value.
///
/// Cloning a `Var` shares the node rather than copying the value, so operands can
/// be reused freely — `a.mul(&a)` records a single node with two edges to `a`,
/// and its gradient accumulates twice, as it should.
pub struct Var<'t, T, B: Backend> {
    tape: &'t Tape<B>,
    node: Rc<Node<T, B>>,
}

impl<T, B: Backend> Clone for Var<'_, T, B> {
    fn clone(&self) -> Self {
        Var {
            tape: self.tape,
            node: self.node.clone(),
        }
    }
}

/// A recorded scalar.
pub type ScalarVar<'t, B = Host, T = f32> = Var<'t, T, B>;
/// A recorded vector.
pub type VectorVar<'t, B = Host, T = f32> = Var<'t, Vector<T, B>, B>;
/// A recorded matrix.
pub type MatrixVar<'t, B = Host, T = f32> = Var<'t, Matrix<T, B>, B>;
/// A recorded N-dimensional tensor; its operations are listed under
/// [`Var`]'s tensor methods and in the [module documentation](self#tensors).
pub type TensorVar<'t, B = Host, T = f32> = Var<'t, Tensor<T, B>, B>;

impl<'t, T: Adjoint<B>, B: Backend> Var<'t, T, B> {
    /// The value computed in the forward pass.
    pub fn value(&self) -> &T {
        self.node.value()
    }

    /// The gradient accumulated by the last backward pass — zeros of this
    /// value's shape if it did not reach the output that was seeded.
    pub fn grad(&self) -> T {
        match self.node.adjoint.borrow().as_ref() {
            Some(adjoint) => adjoint.duplicate(),
            None => self.node.value().zeros_like(),
        }
    }

    /// Whether any gradient has reached this value.
    pub fn has_grad(&self) -> bool {
        self.node.adjoint.borrow().is_some()
    }

    /// The tape this value lives on.
    pub fn tape(&self) -> &'t Tape<B> {
        self.tape
    }

    /// Propagate `seed` back from this value: `seed` is `∂output/∂self`, so a
    /// scalar output uses `1.0` (see [`backward`](ScalarVar::backward)) and a
    /// tensor output needs the direction you want projected.
    ///
    /// Everything recorded *after* this value is skipped, which is what makes it
    /// safe to keep one tape for several outputs.
    ///
    /// Each call is a fresh pass: the adjoints it is about to write are cleared
    /// first, so running it twice gives the same gradients rather than double
    /// them. Gradients from two different outputs therefore do not pile up — add
    /// the outputs into one scalar and differentiate that instead, which is also
    /// one pass rather than two.
    pub fn backward_with(&self, seed: T) {
        let nodes = self.tape.nodes.borrow();
        // A fresh pass invalidates every gradient on the tape, including nodes
        // recorded after this output. Those later nodes are not propagated, but
        // retaining their previous adjoints would make `grad()` report stale
        // results from an earlier pass.
        for node in nodes.iter() {
            node.clear();
        }
        let reachable = &nodes[..=self.node.index];
        self.node.accumulate(seed);
        for node in reachable.iter().rev() {
            node.propagate();
        }
    }

    fn assert_same_tape<U>(&self, other: &Var<'t, U, B>) {
        assert!(
            std::ptr::eq(self.tape, other.tape),
            "reverse-mode operands belong to different tapes"
        );
    }

    /// Record a node whose rule pushes into this graph.
    fn record<U: Adjoint<B>>(&self, value: U, rule: impl Fn(&U) + 'static) -> Var<'t, U, B> {
        self.tape.push(value, Some(Box::new(rule)))
    }

    /// [`record`](Self::record) a value that may not be computed yet.
    fn record_lazy<U: Adjoint<B>>(
        &self,
        value: Lazy<U>,
        rule: impl Fn(&U) + 'static,
    ) -> Var<'t, U, B> {
        self.tape.push_lazy(value, Some(Box::new(rule)))
    }
}

impl<B: Kernels<E>, E: Real> ScalarVar<'_, B, E> {
    /// Propagate from this scalar, seeding `∂self/∂self = 1`.
    pub fn backward(&self) {
        self.backward_with(E::one());
    }
}

// ---- helpers ----------------------------------------------------------------

fn vector_op<B: Kernels<E>, E: Real>(
    a: &Vector<E, B>,
    b: &Vector<E, B>,
    op: BinaryOp,
) -> Vector<E, B> {
    B::vector_elementwise(a, b, op)
}

fn matrix_op<B: Kernels<E>, E: Real>(
    a: &Matrix<E, B>,
    b: &Matrix<E, B>,
    op: BinaryOp,
) -> Matrix<E, B> {
    B::matrix_elementwise(a, b, op)
}

fn negated_vector<B: Kernels<E>, E: Real>(v: &Vector<E, B>) -> Vector<E, B> {
    B::vector_broadcast(v, -E::one(), BinaryOp::Mul, false)
}

fn negated_matrix<B: Kernels<E>, E: Real>(m: &Matrix<E, B>) -> Matrix<E, B> {
    B::matrix_broadcast(m, -E::one(), BinaryOp::Mul, false)
}

/// The sum of a slice, read in place.
fn sum_of<E: Real>(values: &[E]) -> E {
    if let Some(total) = crate::compact::reduce(values, super::Reduce::Sum) {
        return total;
    }
    values.iter().fold(E::zero(), |total, &value| total + value)
}

/// Elementwise `trunc(a / b)`, the locally constant quotient behind `%`.
///
/// There is no truncation kernel, so this reads both operands in place — free on
/// a resident tensor — and stores the result back.
fn truncated_quotient<E: Real>(a: &[E], b: &[E]) -> Vec<E> {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x / y).trunc())
        .collect::<Vec<_>>()
}

// ---- scalars ----------------------------------------------------------------

impl<'t, B: Kernels<E>, E: Real> ScalarVar<'t, B, E> {
    /// `f` of this value, computed now if the value is known and otherwise
    /// when it is first needed.
    fn map(&self, f: impl FnOnce(E) -> E + 'static) -> Lazy<E> {
        if self.node.value.is_ready() {
            return Lazy::ready(f(*self.node.value()));
        }
        let node = self.node.clone();
        Lazy::deferred(move || f(*node.value()))
    }

    /// `f` of this value and `other`'s, likewise.
    fn map2(&self, other: &Self, f: impl FnOnce(E, E) -> E + 'static) -> Lazy<E> {
        if self.node.value.is_ready() && other.node.value.is_ready() {
            return Lazy::ready(f(*self.node.value(), *other.node.value()));
        }
        let (left, right) = (self.node.clone(), other.node.clone());
        Lazy::deferred(move || f(*left.value(), *right.value()))
    }

    fn binary(&self, rhs: &Self, op: BinaryOp) -> Self {
        self.assert_same_tape(rhs);
        let value = self.map2(rhs, move |x, y| match op {
            BinaryOp::Add => x + y,
            BinaryOp::Sub => x - y,
            BinaryOp::Mul => x * y,
            BinaryOp::Div => x / y,
            BinaryOp::Rem => x % y,
        });
        let (left, right) = (self.node.clone(), rhs.node.clone());
        self.record_lazy(value, move |adjoint| {
            let adjoint = *adjoint;
            // Only the derivatives that depend on the operands read them.
            let operands = || (*left.value(), *right.value());
            match op {
                BinaryOp::Add => {
                    left.accumulate(adjoint);
                    right.accumulate(adjoint);
                }
                BinaryOp::Sub => {
                    left.accumulate(adjoint);
                    right.accumulate(-adjoint);
                }
                BinaryOp::Mul => {
                    let (x, y) = operands();
                    left.accumulate(adjoint * y);
                    right.accumulate(adjoint * x);
                }
                BinaryOp::Div => {
                    let (x, y) = operands();
                    left.accumulate(adjoint / y);
                    right.accumulate(-adjoint * x / (y * y));
                }
                // a % b = a − trunc(a/b)·b, and the quotient is locally constant.
                BinaryOp::Rem => {
                    let (x, y) = operands();
                    left.accumulate(adjoint);
                    right.accumulate(-adjoint * (x / y).trunc());
                }
            }
        })
    }

    /// Apply an analytic function: `dy/dx = f'(x)`.
    pub fn analytic(&self, f: Analytic) -> Self {
        let parent = self.node.clone();
        self.record_lazy(self.map(move |x| f.value(x)), move |adjoint| {
            parent.accumulate(*adjoint * f.derivative(*parent.value()));
        })
    }

    /// The larger of two recorded scalars, with the adjoint split at a tie.
    pub fn maximum(&self, other: &Self) -> Self {
        self.select(other, true)
    }

    /// The smaller of two recorded scalars.
    pub fn minimum(&self, other: &Self) -> Self {
        self.select(other, false)
    }

    fn select(&self, other: &Self, largest: bool) -> Self {
        let op = if largest { Compare::Max } else { Compare::Min };
        let (left, right) = (self.node.clone(), other.node.clone());
        self.record_lazy(
            self.map2(other, move |x, y| op.value(x, y)),
            move |adjoint| {
                let share = Compare::MaxShare.value(*left.value(), *right.value());
                let (mine, theirs) = if largest {
                    (share, E::one() - share)
                } else {
                    (E::one() - share, share)
                };
                left.accumulate(*adjoint * mine);
                right.accumulate(*adjoint * theirs);
            },
        )
    }

    /// Absolute value, differentiating to `sign(x)` with `sign(0) = 0`.
    pub fn abs(&self) -> Self {
        self.maximum(&self.scale(-E::one()))
    }

    /// `max(x, 0)`.
    pub fn relu(&self) -> Self {
        let zero = self.tape.scalar(E::zero());
        self.maximum(&zero)
    }

    /// Negate.
    pub fn neg(&self) -> Self {
        self.scale(-E::one())
    }

    /// Multiply by a constant.
    pub fn scale(&self, factor: E) -> Self {
        let parent = self.node.clone();
        self.record_lazy(self.map(move |x| x * factor), move |adjoint| {
            parent.accumulate(*adjoint * factor);
        })
    }

    /// Add a constant, which leaves the derivative unchanged.
    pub fn shift(&self, offset: E) -> Self {
        let parent = self.node.clone();
        self.record_lazy(self.map(move |x| x + offset), move |adjoint| {
            parent.accumulate(*adjoint);
        })
    }

    /// Broadcast this scalar across a length-`len` vector. Its gradient is the
    /// sum of the vector's adjoint, since it reaches every element.
    pub fn expand(&self, len: usize) -> VectorVar<'t, B, E> {
        let parent = self.node.clone();
        self.record(Vector::filled(len, *self.node.value()), move |adjoint| {
            parent.accumulate(sum_of(adjoint.as_slice()));
        })
    }

    /// Broadcast this scalar across a `rows × cols` matrix.
    pub fn expand_matrix(&self, rows: usize, cols: usize) -> MatrixVar<'t, B, E> {
        let parent = self.node.clone();
        self.record(
            Matrix::filled(rows, cols, *self.node.value()),
            move |adjoint| {
                parent.accumulate(sum_of(adjoint.as_slice()));
            },
        )
    }
}

// ---- vectors ----------------------------------------------------------------

impl<'t, B: Kernels<E>, E: Real> VectorVar<'t, B, E> {
    /// The number of elements.
    pub fn len(&self) -> usize {
        self.node.value().len()
    }

    /// Whether this vector holds no elements.
    pub fn is_empty(&self) -> bool {
        self.node.value().is_empty()
    }

    fn binary(&self, rhs: &Self, op: BinaryOp) -> Self {
        self.assert_same_tape(rhs);
        let value = vector_op(self.value(), rhs.value(), op);
        let (left, right) = (self.node.clone(), rhs.node.clone());
        self.record(value, move |adjoint| match op {
            BinaryOp::Add => {
                left.accumulate(adjoint.duplicate());
                right.accumulate(adjoint.duplicate());
            }
            BinaryOp::Sub => {
                left.accumulate(adjoint.duplicate());
                right.accumulate(negated_vector(adjoint));
            }
            // c = a ⊙ b
            BinaryOp::Mul => {
                left.accumulate(vector_op(adjoint, right.value(), BinaryOp::Mul));
                right.accumulate(vector_op(adjoint, left.value(), BinaryOp::Mul));
            }
            // c = a / b: ā += c̄/b, b̄ −= c̄·a/b²
            BinaryOp::Div => {
                left.accumulate(vector_op(adjoint, right.value(), BinaryOp::Div));
                let squared = vector_op(right.value(), right.value(), BinaryOp::Mul);
                let scaled = vector_op(
                    &vector_op(adjoint, left.value(), BinaryOp::Mul),
                    &squared,
                    BinaryOp::Div,
                );
                right.accumulate(negated_vector(&scaled));
            }
            // c = a % b: the quotient is locally constant, so b̄ −= c̄·trunc(a/b)
            BinaryOp::Rem => {
                left.accumulate(adjoint.duplicate());
                let quotient = Vector::<E, B>::build(&truncated_quotient(
                    left.value().as_slice(),
                    right.value().as_slice(),
                ));
                let scaled = vector_op(adjoint, &quotient, BinaryOp::Mul);
                right.accumulate(negated_vector(&scaled));
            }
        })
    }

    /// Elementwise larger of two recorded vectors.
    ///
    /// The adjoint goes to whichever operand supplied the value, and splits
    /// evenly where they tie — the convention [`Compare`] documents. The mask is
    /// one `MaxShare` dispatch, so the rule stays on the backend.
    pub fn maximum(&self, other: &Self) -> Self {
        self.select(other, true)
    }

    /// Elementwise smaller of two recorded vectors.
    pub fn minimum(&self, other: &Self) -> Self {
        self.select(other, false)
    }

    fn select(&self, other: &Self, largest: bool) -> Self {
        let op = if largest { Compare::Max } else { Compare::Min };
        let value = B::vector_compare(self.value(), other.value(), op);
        let (left, right) = (self.node.clone(), other.node.clone());
        self.record(value, move |adjoint| {
            let share = B::vector_compare(left.value(), right.value(), Compare::MaxShare);
            let complement = B::vector_broadcast(&share, E::one(), BinaryOp::Sub, true);
            let (mine, theirs) = if largest {
                (&share, &complement)
            } else {
                (&complement, &share)
            };
            left.accumulate(B::vector_elementwise(adjoint, mine, BinaryOp::Mul));
            right.accumulate(B::vector_elementwise(adjoint, theirs, BinaryOp::Mul));
        })
    }

    /// Elementwise maximum against a constant.
    pub fn clamp_min(&self, floor: E) -> Self {
        self.select_scalar(floor, true)
    }

    /// Elementwise minimum against a constant.
    pub fn clamp_max(&self, ceiling: E) -> Self {
        self.select_scalar(ceiling, false)
    }

    /// Confine every element to `[floor, ceiling]`; the gradient is zero wherever
    /// an element is pinned to a bound.
    pub fn clamp(&self, floor: E, ceiling: E) -> Self {
        self.clamp_min(floor).clamp_max(ceiling)
    }

    /// The rectifier `max(a, 0)`.
    pub fn relu(&self) -> Self {
        self.clamp_min(E::zero())
    }

    /// Elementwise absolute value, as `max(a, −a)`, which differentiates to
    /// `sign(a)` with `sign(0) = 0`.
    pub fn abs(&self) -> Self {
        self.maximum(&self.scale(-E::one()))
    }

    fn select_scalar(&self, scalar: E, largest: bool) -> Self {
        let op = if largest { Compare::Max } else { Compare::Min };
        let value = B::vector_compare_scalar(self.value(), scalar, op, false);
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            let share = B::vector_compare_scalar(parent.value(), scalar, Compare::MaxShare, false);
            let weight = if largest {
                share
            } else {
                B::vector_broadcast(&share, E::one(), BinaryOp::Sub, true)
            };
            parent.accumulate(B::vector_elementwise(adjoint, &weight, BinaryOp::Mul));
        })
    }

    /// Apply an analytic function elementwise: `ā += f'(a) ⊙ ȳ`.
    ///
    /// The backward pass is one `unary_dual` dispatch — the kernel forward mode
    /// uses, whose tangent output is exactly this product.
    pub fn analytic(&self, f: Analytic) -> Self {
        let value = B::vector_unary(self.value(), f);
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            let (_, product) = B::vector_unary_dual(parent.value(), adjoint, f);
            parent.accumulate(product);
        })
    }

    /// Negate every element.
    pub fn neg(&self) -> Self {
        self.scale(-E::one())
    }

    /// Multiply by a constant.
    pub fn scale(&self, factor: E) -> Self {
        let value = B::vector_broadcast(self.value(), factor, BinaryOp::Mul, false);
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            parent.accumulate(B::vector_broadcast(adjoint, factor, BinaryOp::Mul, false));
        })
    }

    /// Add a constant to every element, which leaves the derivative unchanged.
    pub fn shift(&self, offset: E) -> Self {
        let value = B::vector_broadcast(self.value(), offset, BinaryOp::Add, false);
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            parent.accumulate(adjoint.duplicate());
        })
    }

    /// Sum of the elements: every element's gradient is the scalar's adjoint.
    pub fn sum(&self) -> ScalarVar<'t, B, E> {
        let parent = self.node.clone();
        let total = {
            let parent = parent.clone();
            Lazy::deferred(move || sum_of(parent.value().as_slice()))
        };
        self.record_lazy(total, move |adjoint| {
            parent.accumulate(Vector::filled(parent.value().len(), *adjoint));
        })
    }

    /// Dot product: `ū += s̄·v` and `v̄ += s̄·u`.
    pub fn dot(&self, other: &Self) -> ScalarVar<'t, B, E> {
        self.assert_same_tape(other);
        let (left, right) = (self.node.clone(), other.node.clone());
        let value = {
            let (left, right) = (left.clone(), right.clone());
            Lazy::deferred(move || B::dot(left.value(), right.value()))
        };
        self.record_lazy(value, move |adjoint| {
            let scale = *adjoint;
            left.accumulate(B::vector_broadcast(
                right.value(),
                scale,
                BinaryOp::Mul,
                false,
            ));
            right.accumulate(B::vector_broadcast(
                left.value(),
                scale,
                BinaryOp::Mul,
                false,
            ));
        })
    }

    /// Row vector times matrix, `(1×N)·(N×C)`: `x̄ += A·ȳ` and `Ā += x ⊗ ȳ`.
    pub fn vecmat(&self, m: &MatrixVar<'t, B, E>) -> VectorVar<'t, B, E> {
        self.assert_same_tape(m);
        let value = B::vecmat(self.value(), m.value());
        let (vector, matrix) = (self.node.clone(), m.node.clone());
        self.record(value, move |adjoint| {
            vector.accumulate(B::matvec(matrix.value(), adjoint));
            matrix.accumulate(outer(vector.value(), adjoint));
        })
    }

    /// View as a `1 × N` matrix; the adjoint flows straight back.
    pub fn into_row(&self) -> MatrixVar<'t, B, E> {
        let len = self.len();
        let value = Matrix::from_storage(1, len, self.value().duplicate().into_storage());
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            parent.accumulate(Vector::build(adjoint.as_slice()));
        })
    }

    /// Outer product `u ⊗ v`: the `N × M` matrix with entries `uᵢvⱼ`.
    ///
    /// The rank-one product that shows up wherever a gradient meets an input —
    /// it is what `matvec` accumulates into its matrix operand, and what a
    /// weight update looks like. Its own rules are the two contractions of the
    /// adjoint: `ū = Ȳ·v` and `v̄ = uᵀ·Ȳ`.
    pub fn outer(&self, other: &VectorVar<'t, B, E>) -> MatrixVar<'t, B, E> {
        self.assert_same_tape(other);
        let value = outer(self.value(), other.value());
        let (left, right) = (self.node.clone(), other.node.clone());
        self.record(value, move |adjoint| {
            left.accumulate(B::matvec(adjoint, right.value()));
            right.accumulate(B::vecmat(left.value(), adjoint));
        })
    }

    /// View as an `N × 1` matrix; the adjoint flows straight back.
    pub fn into_column(&self) -> MatrixVar<'t, B, E> {
        let len = self.len();
        let value = Matrix::from_storage(len, 1, self.value().duplicate().into_storage());
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            parent.accumulate(Vector::build(adjoint.as_slice()));
        })
    }
}

/// The outer product `u ⊗ v` as a `u.len() × v.len()` matrix, built from the
/// column/row views the backend already provides.
fn outer<B: Kernels<E>, E: Real>(u: &Vector<E, B>, v: &Vector<E, B>) -> Matrix<E, B> {
    let column = Matrix::from_storage(u.len(), 1, u.to_backend::<B>().into_storage());
    let row = Matrix::from_storage(1, v.len(), v.to_backend::<B>().into_storage());
    B::matmul(&column, &row)
}

// ---- matrices ---------------------------------------------------------------

impl<'t, B: Kernels<E>, E: Real> MatrixVar<'t, B, E> {
    /// The `(rows, columns)` extents.
    pub fn shape(&self) -> (usize, usize) {
        self.node.value().shape()
    }

    /// The number of rows.
    pub fn rows(&self) -> usize {
        self.node.value().rows()
    }

    /// The number of columns.
    pub fn cols(&self) -> usize {
        self.node.value().cols()
    }

    fn binary(&self, rhs: &Self, op: BinaryOp) -> Self {
        self.assert_same_tape(rhs);
        let value = matrix_op(self.value(), rhs.value(), op);
        let (left, right) = (self.node.clone(), rhs.node.clone());
        self.record(value, move |adjoint| match op {
            BinaryOp::Add => {
                left.accumulate(adjoint.duplicate());
                right.accumulate(adjoint.duplicate());
            }
            BinaryOp::Sub => {
                left.accumulate(adjoint.duplicate());
                right.accumulate(negated_matrix(adjoint));
            }
            BinaryOp::Mul => {
                left.accumulate(matrix_op(adjoint, right.value(), BinaryOp::Mul));
                right.accumulate(matrix_op(adjoint, left.value(), BinaryOp::Mul));
            }
            BinaryOp::Div => {
                left.accumulate(matrix_op(adjoint, right.value(), BinaryOp::Div));
                let squared = matrix_op(right.value(), right.value(), BinaryOp::Mul);
                let scaled = matrix_op(
                    &matrix_op(adjoint, left.value(), BinaryOp::Mul),
                    &squared,
                    BinaryOp::Div,
                );
                right.accumulate(negated_matrix(&scaled));
            }
            BinaryOp::Rem => {
                left.accumulate(adjoint.duplicate());
                let (rows, cols) = left.value().shape();
                let quotient = Matrix::<E, B>::build(
                    rows,
                    cols,
                    &truncated_quotient(left.value().as_slice(), right.value().as_slice()),
                );
                let scaled = matrix_op(adjoint, &quotient, BinaryOp::Mul);
                right.accumulate(negated_matrix(&scaled));
            }
        })
    }

    /// Elementwise larger of two recorded matrices; see [`VectorVar::maximum`].
    pub fn maximum(&self, other: &Self) -> Self {
        self.select(other, true)
    }

    /// Elementwise smaller of two recorded matrices.
    pub fn minimum(&self, other: &Self) -> Self {
        self.select(other, false)
    }

    fn select(&self, other: &Self, largest: bool) -> Self {
        let op = if largest { Compare::Max } else { Compare::Min };
        let value = B::matrix_compare(self.value(), other.value(), op);
        let (left, right) = (self.node.clone(), other.node.clone());
        self.record(value, move |adjoint| {
            let share = B::matrix_compare(left.value(), right.value(), Compare::MaxShare);
            let complement = B::matrix_broadcast(&share, E::one(), BinaryOp::Sub, true);
            let (mine, theirs) = if largest {
                (&share, &complement)
            } else {
                (&complement, &share)
            };
            left.accumulate(B::matrix_elementwise(adjoint, mine, BinaryOp::Mul));
            right.accumulate(B::matrix_elementwise(adjoint, theirs, BinaryOp::Mul));
        })
    }

    /// Elementwise maximum against a constant.
    pub fn clamp_min(&self, floor: E) -> Self {
        self.select_scalar(floor, true)
    }

    /// Elementwise minimum against a constant.
    pub fn clamp_max(&self, ceiling: E) -> Self {
        self.select_scalar(ceiling, false)
    }

    /// Confine every element to `[floor, ceiling]`.
    pub fn clamp(&self, floor: E, ceiling: E) -> Self {
        self.clamp_min(floor).clamp_max(ceiling)
    }

    /// The rectifier `max(A, 0)`.
    pub fn relu(&self) -> Self {
        self.clamp_min(E::zero())
    }

    /// Elementwise absolute value; see [`VectorVar::abs`].
    pub fn abs(&self) -> Self {
        self.maximum(&self.scale(-E::one()))
    }

    fn select_scalar(&self, scalar: E, largest: bool) -> Self {
        let op = if largest { Compare::Max } else { Compare::Min };
        let value = B::matrix_compare_scalar(self.value(), scalar, op, false);
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            let share = B::matrix_compare_scalar(parent.value(), scalar, Compare::MaxShare, false);
            let weight = if largest {
                share
            } else {
                B::matrix_broadcast(&share, E::one(), BinaryOp::Sub, true)
            };
            parent.accumulate(B::matrix_elementwise(adjoint, &weight, BinaryOp::Mul));
        })
    }

    /// Sum along each row, one entry per row.
    ///
    /// This is `A·1`, so it reuses the product kernels rather than needing a
    /// reduction of its own, and the adjoint spreads straight back along each
    /// row: `Ā += ȳ ⊗ 1`.
    pub fn row_sums(&self) -> VectorVar<'t, B, E> {
        let cols = self.cols();
        let value = B::matvec(self.value(), &Vector::<E, B>::filled(cols, E::one()));
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            parent.accumulate(outer(adjoint, &Vector::<E, B>::filled(cols, E::one())));
        })
    }

    /// Sum along each column, one entry per column: `1ᵀ·A`, with `Ā += 1 ⊗ ȳ`.
    pub fn column_sums(&self) -> VectorVar<'t, B, E> {
        let rows = self.rows();
        let value = B::vecmat(&Vector::<E, B>::filled(rows, E::one()), self.value());
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            parent.accumulate(outer(&Vector::<E, B>::filled(rows, E::one()), adjoint));
        })
    }

    /// Apply an analytic function elementwise; see [`VectorVar::analytic`].
    pub fn analytic(&self, f: Analytic) -> Self {
        let value = B::matrix_unary(self.value(), f);
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            let (_, product) = B::matrix_unary_dual(parent.value(), adjoint, f);
            parent.accumulate(product);
        })
    }

    /// Negate every element.
    pub fn neg(&self) -> Self {
        self.scale(-E::one())
    }

    /// Multiply by a constant.
    pub fn scale(&self, factor: E) -> Self {
        let value = B::matrix_broadcast(self.value(), factor, BinaryOp::Mul, false);
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            parent.accumulate(B::matrix_broadcast(adjoint, factor, BinaryOp::Mul, false));
        })
    }

    /// Add a constant to every element.
    pub fn shift(&self, offset: E) -> Self {
        let value = B::matrix_broadcast(self.value(), offset, BinaryOp::Add, false);
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            parent.accumulate(adjoint.duplicate());
        })
    }

    /// Matrix product: `Ā += C̄·Bᵀ` and `B̄ += Aᵀ·C̄`.
    ///
    /// Both use the accumulating matmul, so each contribution is one dispatch
    /// rather than a product followed by a sum.
    pub fn matmul(&self, other: &MatrixVar<'t, B, E>) -> MatrixVar<'t, B, E> {
        self.assert_same_tape(other);
        let value = B::matmul(self.value(), other.value());
        let (left, right) = (self.node.clone(), other.node.clone());
        self.record(value, move |adjoint| {
            let left_adjoint = B::matmul_transposed_add(
                adjoint,
                right.value(),
                Transposed::Right,
                left.current_or_zeros(),
            );
            *left.adjoint.borrow_mut() = Some(left_adjoint);

            let right_adjoint = B::matmul_transposed_add(
                left.value(),
                adjoint,
                Transposed::Left,
                right.current_or_zeros(),
            );
            *right.adjoint.borrow_mut() = Some(right_adjoint);
        })
    }

    /// Fused matrix multiply-add: `self·other + addend`.
    pub fn matmul_add(
        &self,
        other: &MatrixVar<'t, B, E>,
        addend: &MatrixVar<'t, B, E>,
    ) -> MatrixVar<'t, B, E> {
        self.assert_same_tape(other);
        self.assert_same_tape(addend);
        let value = B::matmul_add(self.value(), other.value(), addend.value().duplicate());
        let (left, right, bias) = (self.node.clone(), other.node.clone(), addend.node.clone());
        self.record(value, move |adjoint| {
            let left_adjoint = B::matmul_transposed_add(
                adjoint,
                right.value(),
                Transposed::Right,
                left.current_or_zeros(),
            );
            *left.adjoint.borrow_mut() = Some(left_adjoint);

            let right_adjoint = B::matmul_transposed_add(
                left.value(),
                adjoint,
                Transposed::Left,
                right.current_or_zeros(),
            );
            *right.adjoint.borrow_mut() = Some(right_adjoint);
            bias.accumulate(adjoint.duplicate());
        })
    }

    /// Matrix times vector: `Ā += ȳ ⊗ x` and `x̄ += Aᵀ·ȳ`.
    pub fn matvec(&self, v: &VectorVar<'t, B, E>) -> VectorVar<'t, B, E> {
        self.assert_same_tape(v);
        let value = B::matvec(self.value(), v.value());
        let (matrix, vector) = (self.node.clone(), v.node.clone());
        self.record(value, move |adjoint| {
            matrix.accumulate(outer(adjoint, vector.value()));
            vector.accumulate(B::vecmat(adjoint, matrix.value()));
        })
    }

    /// Fused matrix-vector multiply-add: `self·v + addend`.
    pub fn matvec_add(
        &self,
        v: &VectorVar<'t, B, E>,
        addend: &VectorVar<'t, B, E>,
    ) -> VectorVar<'t, B, E> {
        self.assert_same_tape(v);
        self.assert_same_tape(addend);
        let value = B::matvec_add(self.value(), v.value(), addend.value().duplicate());
        let (matrix, vector, bias) = (self.node.clone(), v.node.clone(), addend.node.clone());
        self.record(value, move |adjoint| {
            matrix.accumulate(outer(adjoint, vector.value()));
            vector.accumulate(B::vecmat(adjoint, matrix.value()));
            bias.accumulate(adjoint.duplicate());
        })
    }

    /// Transpose; the adjoint transposes back.
    pub fn transpose(&self) -> MatrixVar<'t, B, E> {
        let value = B::transpose(self.value());
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            parent.accumulate(B::transpose(adjoint));
        })
    }

    /// Sum of every element.
    pub fn sum(&self) -> ScalarVar<'t, B, E> {
        let parent = self.node.clone();
        let total = {
            let parent = parent.clone();
            Lazy::deferred(move || sum_of(parent.value().as_slice()))
        };
        self.record_lazy(total, move |adjoint| {
            let (rows, cols) = parent.value().shape();
            parent.accumulate(Matrix::filled(rows, cols, *adjoint));
        })
    }

    /// The Frobenius inner product `Σᵢⱼ aᵢⱼbᵢⱼ`, the usual scalar loss.
    pub fn frobenius_dot(&self, other: &Self) -> ScalarVar<'t, B, E> {
        self.binary(other, BinaryOp::Mul).sum()
    }

    /// Valid cross-correlation with a recorded window — the "convolution" of a
    /// convolutional layer, where the window is usually the parameter.
    ///
    /// Both gradients are correlations of their own, so the backward pass needs
    /// no new kernel:
    ///
    /// - `K̄ += correlate(X, Ȳ)`, the input windowed by the output adjoint;
    /// - `X̄ += correlate(pad(Ȳ, KR−1, KC−1), K, flipped)`, a *full*
    ///   correlation, which is what the zero padding spells.
    pub fn correlate(&self, window: &MatrixVar<'t, B, E>) -> MatrixVar<'t, B, E> {
        self.correlate_with(window, false)
    }

    /// Convolution proper: the window is reversed before it is applied.
    ///
    /// This is the signal-processing convention. Machine learning calls
    /// [`correlate`](Self::correlate) "convolution"; the two differ only by that
    /// reversal, and both differentiate here.
    pub fn convolve(&self, window: &MatrixVar<'t, B, E>) -> MatrixVar<'t, B, E> {
        self.correlate_with(window, true)
    }

    fn correlate_with(&self, window: &MatrixVar<'t, B, E>, flip: bool) -> MatrixVar<'t, B, E> {
        let value = B::correlate(self.value(), window.value(), flip);
        let (input, taps) = (self.node.clone(), window.node.clone());
        self.record(value, move |adjoint| {
            // The window's gradient is the input correlated with the adjoint,
            // reversed when the forward pass reversed the window.
            let window_gradient = B::correlate_window_gradient(input.value(), adjoint);
            taps.accumulate(if flip {
                B::flip(&window_gradient)
            } else {
                window_gradient
            });

            // The input's gradient is the full correlation of the adjoint with
            // the window, applied the other way round.
            input.accumulate(B::correlate_input_gradient(adjoint, taps.value(), flip));
        })
    }

    /// Surround with zeros; the adjoint of the padding is discarded, and the
    /// interior flows straight back.
    pub fn pad(&self, pad_rows: usize, pad_cols: usize) -> MatrixVar<'t, B, E> {
        let value = B::pad(self.value(), pad_rows, pad_cols);
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            let (rows, cols) = parent.value().shape();
            let padded_cols = cols + 2 * pad_cols;
            let interior = adjoint.as_slice();
            let mut inner = Vec::with_capacity(rows * cols);
            for row in 0..rows {
                let start = (row + pad_rows) * padded_cols + pad_cols;
                inner.extend_from_slice(&interior[start..start + cols]);
            }
            parent.accumulate(Matrix::build(rows, cols, &inner));
        })
    }

    /// Reverse both axes; the adjoint reverses back.
    pub fn flipped(&self) -> Self {
        let value = B::flip(self.value());
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            parent.accumulate(B::flip(adjoint));
        })
    }

    /// Row-major flattening, and its exact inverse on the way back.
    pub fn flattened(&self) -> VectorVar<'t, B, E> {
        let (rows, cols) = self.shape();
        let value = Vector::from_storage(rows * cols, self.value().duplicate().into_storage());
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            parent.accumulate(Matrix::build(rows, cols, adjoint.as_slice()));
        })
    }
}

// ---- named operations and operators -----------------------------------------

/// The binary operations, as inherent methods and as operators on references.
macro_rules! binary_methods {
    ($($method:ident => $op:expr, $trait:ident :: $trait_method:ident),+ $(,)?) => {
        impl<'t, B: Kernels<E>, E: Real> ScalarVar<'t, B, E> {
            $(
                #[doc = concat!("Elementwise `", stringify!($method), "`, differentiated.")]
                pub fn $method(&self, rhs: &Self) -> Self { self.binary(rhs, $op) }
            )+
        }

        impl<'t, B: Kernels<E>, E: Real> VectorVar<'t, B, E> {
            $(
                #[doc = concat!("Elementwise `", stringify!($method), "`, differentiated.")]
                pub fn $method(&self, rhs: &Self) -> Self { self.binary(rhs, $op) }
            )+
        }

        impl<'t, B: Kernels<E>, E: Real> MatrixVar<'t, B, E> {
            $(
                #[doc = concat!("Elementwise `", stringify!($method), "`, differentiated.")]
                pub fn $method(&self, rhs: &Self) -> Self { self.binary(rhs, $op) }
            )+
        }

        $(
            impl<'t, B: Kernels<E>, E: Real> $trait for &ScalarVar<'t, B, E> {
                type Output = ScalarVar<'t, B, E>;
                fn $trait_method(self, rhs: Self) -> Self::Output { self.binary(rhs, $op) }
            }

            impl<'t, B: Kernels<E>, E: Real> $trait for &VectorVar<'t, B, E> {
                type Output = VectorVar<'t, B, E>;
                fn $trait_method(self, rhs: Self) -> Self::Output { self.binary(rhs, $op) }
            }

            impl<'t, B: Kernels<E>, E: Real> $trait for &MatrixVar<'t, B, E> {
                type Output = MatrixVar<'t, B, E>;
                fn $trait_method(self, rhs: Self) -> Self::Output { self.binary(rhs, $op) }
            }
        )+
    };
}

binary_methods!(
    add => BinaryOp::Add, Add::add,
    sub => BinaryOp::Sub, Sub::sub,
    mul => BinaryOp::Mul, Mul::mul,
    div => BinaryOp::Div, Div::div,
    rem => BinaryOp::Rem, Rem::rem,
);

/// The analytic functions, as methods on every shape.
macro_rules! analytic_methods {
    ($($method:ident => $variant:ident),+ $(,)?) => {
        impl<'t, B: Kernels<E>, E: Real> ScalarVar<'t, B, E> {
            $(
                #[doc = concat!("`", stringify!($method), "`, differentiated.")]
                pub fn $method(&self) -> Self { self.analytic(Analytic::$variant) }
            )+
        }

        impl<'t, B: Kernels<E>, E: Real> VectorVar<'t, B, E> {
            $(
                #[doc = concat!("Elementwise `", stringify!($method), "`, differentiated.")]
                pub fn $method(&self) -> Self { self.analytic(Analytic::$variant) }
            )+
        }

        impl<'t, B: Kernels<E>, E: Real> MatrixVar<'t, B, E> {
            $(
                #[doc = concat!("Elementwise `", stringify!($method), "`, differentiated.")]
                pub fn $method(&self) -> Self { self.analytic(Analytic::$variant) }
            )+
        }

        impl<'t, B: Kernels<E>, E: Real> TensorVar<'t, B, E> {
            $(
                #[doc = concat!("Elementwise `", stringify!($method), "`, differentiated.")]
                pub fn $method(&self) -> Self { self.analytic(Analytic::$variant) }
            )+
        }
    };
}

analytic_methods!(
    sin => Sin,
    cos => Cos,
    tan => Tan,
    sec => Sec,
    csc => Csc,
    arcsin => Arcsin,
    arccos => Arccos,
    arctan => Arctan,
    exp => Exp,
    ln => Ln,
    sinh => Sinh,
    cosh => Cosh,
    tanh => Tanh,
    sqrt => Sqrt,
);

// ---- whole derivatives ------------------------------------------------------

/// The gradient of a scalar-valued `f` at `at`, in one backward pass.
///
/// The counterpart to [`dual::gradient`](super::dual::gradient), which needs one
/// forward pass per input element. This one runs the function once whatever the
/// input size.
pub fn gradient<B: Kernels<E>, E: Real>(
    at: &Vector<E, B>,
    f: impl for<'t> FnOnce(&VectorVar<'t, B, E>) -> ScalarVar<'t, B, E>,
) -> Vector<E, B> {
    let tape = Tape::<B>::new();
    let input = tape.vector(at.to_backend::<B>());
    f(&input).backward();
    input.grad()
}

/// The gradient of a scalar-valued `f` with respect to a matrix input, in one
/// backward pass — where
/// [`dual::gradient_wrt_matrix`](super::dual::gradient_wrt_matrix) needs one
/// forward pass per element.
pub fn gradient_wrt_matrix<B: Kernels<E>, E: Real>(
    at: &Matrix<E, B>,
    f: impl for<'t> FnOnce(&MatrixVar<'t, B, E>) -> ScalarVar<'t, B, E>,
) -> Matrix<E, B> {
    let tape = Tape::<B>::new();
    let input = tape.matrix(at.to_backend::<B>());
    f(&input).backward();
    input.grad()
}

/// The full Jacobian of a vector-valued `f` at `at`, as `OUT × IN`.
///
/// Row `i` is `∂fᵢ/∂x`, recovered by seeding the output with `eᵢ`. The graph is
/// built once and each row is a separate backward pass over it, so the forward
/// computation is shared — that is the difference from
/// [`dual::jacobian`](super::dual::jacobian), which re-runs `f` for every input.
///
/// Which mode is cheaper is a matter of shape: forward costs one pass per input,
/// reverse one per output. For a tall Jacobian prefer forward, for a wide one
/// prefer reverse, and they agree to within floating-point error either way.
///
/// ```
/// use tensorcrate::tensors::Vector;
/// use tensorcrate::tensors::tape::jacobian;
///
/// // An elementwise map has a diagonal Jacobian: d sin(x)ᵢ/dxⱼ = δᵢⱼ cos(xᵢ).
/// let at = Vector::new([0.5f32, 2.0]);
/// let computed = jacobian(&at, |x| x.sin());
///
/// assert!((computed[(0, 0)] - 0.5f32.cos()).abs() < 1e-6);
/// assert!((computed[(1, 1)] - 2.0f32.cos()).abs() < 1e-6);
/// assert_eq!(computed[(0, 1)], 0.0);
/// assert_eq!(computed[(1, 0)], 0.0);
/// ```
pub fn jacobian<B: Kernels<E>, E: Real>(
    at: &Vector<E, B>,
    f: impl for<'t> FnOnce(&VectorVar<'t, B, E>) -> VectorVar<'t, B, E>,
) -> Matrix<E, B> {
    let tape = Tape::<B>::new();
    let input = tape.vector(at.to_backend::<B>());
    let output = f(&input);
    let (inputs, outputs) = (input.len(), output.len());
    let rows = (0..outputs)
        .map(|row| {
            output.backward_with(basis_vector::<B, E>(outputs, row));
            input.grad()
        })
        .collect::<Vec<_>>();
    Vector::stack_rows(inputs, &rows)
}

/// The full Jacobian of a vector-valued `f` with respect to a matrix input, as
/// `OUT × (rows·cols)` with columns in the row-major order of the input.
///
/// A matrix-valued `f` needs no separate driver: flatten its output with
/// [`MatrixVar::flattened`] and the result is the standard
/// `(OR·OC) × (rows·cols)` Jacobian.
pub fn jacobian_wrt_matrix<B: Kernels<E>, E: Real>(
    at: &Matrix<E, B>,
    f: impl for<'t> FnOnce(&MatrixVar<'t, B, E>) -> VectorVar<'t, B, E>,
) -> Matrix<E, B> {
    let tape = Tape::<B>::new();
    let input = tape.matrix(at.to_backend::<B>());
    let output = f(&input);
    let inputs = at.rows() * at.cols();
    let outputs = output.len();
    let rows = (0..outputs)
        .map(|row| {
            output.backward_with(basis_vector::<B, E>(outputs, row));
            Vector::from_storage(inputs, input.grad().into_storage())
        })
        .collect::<Vec<_>>();
    Vector::stack_rows(inputs, &rows)
}

/// The `index`th standard basis vector of length `len`, the seed that extracts
/// one Jacobian row.
fn basis_vector<B: Kernels<E>, E: Real>(len: usize, index: usize) -> Vector<E, B> {
    let mut seed = vec![E::zero(); len];
    if let Some(slot) = seed.get_mut(index) {
        *slot = E::one();
    }
    Vector::build(&seed)
}
