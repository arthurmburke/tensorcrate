//! Reverse-mode automatic differentiation.
//!
//! Forward mode ([`dual`](super::dual)) carries a tangent alongside every value
//! and costs one pass per *input*. Reverse mode records the computation and then
//! walks it backwards, costing one pass per *output* — so a scalar loss over a
//! whole weight matrix takes a single backward pass instead of `R * C` forward
//! ones.
//!
//! # Using it
//!
//! ```
//! use rinterp::tensors::{Matrix, Tape, Vector};
//!
//! let tape = Tape::new();
//! let a = tape.matrix(Matrix::<f32, 2, 3>::from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
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
//!         let expected = 2.0 * projected.data()[row] * x.value().data()[col];
//!         assert!((a.grad().data()[row][col] - expected).abs() < 1e-5);
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
//! closures are built where the operation is written, so the shapes are static
//! there and the rules need no runtime shape checks; the tape itself only sees
//! `dyn Backprop`. Adjoints start as `None` rather than zeros, which keeps
//! untouched nodes from allocating (and, on `Metal`, from dispatching).
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
//! `f32` scalars, vectors and matrices, on both backends:
//!
//! - all thirteen [`Analytic`] functions, elementwise;
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
//! Everything placed on the tape is a leaf whose gradient can be read

use std::cell::RefCell;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Rem, Sub};
use std::rc::Rc;

use super::{Analytic, BinaryOp, Compare, Host, Kernels, Matrix, Vector};

// ---- what a node can hold ---------------------------------------------------

/// A value a tape node can carry: `f32`, or an `f32` tensor on backend `B`.
///
/// The methods are what adjoint bookkeeping needs — a zero to start from, a copy,
/// and a sum — expressed without a `Clone` bound, which on a generic backend
/// would infect every signature that touches a node.
pub trait Adjoint<B: Kernels>: Sized + 'static {
    /// The additive identity of this shape.
    fn zeros() -> Self;

    /// A second copy of this value on the same backend.
    fn duplicate(&self) -> Self;

    /// Elementwise sum, used to accumulate contributions from several consumers.
    fn add(&self, other: &Self) -> Self;
}

impl<B: Kernels> Adjoint<B> for f32 {
    fn zeros() -> Self {
        0.0
    }

    fn duplicate(&self) -> Self {
        *self
    }

    fn add(&self, other: &Self) -> Self {
        self + other
    }
}

impl<const N: usize, B: Kernels> Adjoint<B> for Vector<f32, N, B> {
    fn zeros() -> Self {
        Vector::filled(0.0)
    }

    fn duplicate(&self) -> Self {
        self.to_backend::<B>()
    }

    fn add(&self, other: &Self) -> Self {
        B::vector_elementwise(self, other, BinaryOp::Add)
    }
}

impl<const R: usize, const C: usize, B: Kernels> Adjoint<B> for Matrix<f32, R, C, B> {
    fn zeros() -> Self {
        Matrix::filled(0.0)
    }

    fn duplicate(&self) -> Self {
        self.to_backend::<B>()
    }

    fn add(&self, other: &Self) -> Self {
        B::matrix_elementwise(self, other, BinaryOp::Add)
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

struct Node<T, B: Kernels> {
    /// Position in the tape, which is also this node's place in topological
    /// order.
    index: usize,
    value: T,
    /// `None` is a zero adjoint that has not been allocated.
    adjoint: RefCell<Option<T>>,
    /// Pushes this node's adjoint into its operands. `None` for a leaf.
    rule: Option<Rule<T>>,
    marker: PhantomData<B>,
}

impl<T: Adjoint<B>, B: Kernels> Node<T, B> {
    fn accumulate(&self, delta: T) {
        let mut adjoint = self.adjoint.borrow_mut();
        *adjoint = Some(match adjoint.as_ref() {
            Some(current) => current.add(&delta),
            None => delta,
        });
    }
}

impl<T: Adjoint<B>, B: Kernels> Backprop for Node<T, B> {
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
pub struct Tape<B: Kernels = Host> {
    nodes: RefCell<Vec<Rc<dyn Backprop>>>,
    marker: PhantomData<B>,
}

impl<B: Kernels> Default for Tape<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Kernels> Tape<B> {
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
    pub fn scalar(&self, value: f32) -> ScalarVar<'_, B> {
        self.push(value, None)
    }

    /// Record a vector input.
    pub fn vector<const N: usize>(&self, value: Vector<f32, N, B>) -> VectorVar<'_, N, B> {
        self.push(value, None)
    }

    /// Record a matrix input.
    pub fn matrix<const R: usize, const C: usize>(
        &self,
        value: Matrix<f32, R, C, B>,
    ) -> MatrixVar<'_, R, C, B> {
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
pub struct Var<'t, T, B: Kernels> {
    tape: &'t Tape<B>,
    node: Rc<Node<T, B>>,
}

impl<T, B: Kernels> Clone for Var<'_, T, B> {
    fn clone(&self) -> Self {
        Var {
            tape: self.tape,
            node: self.node.clone(),
        }
    }
}

/// A recorded scalar.
pub type ScalarVar<'t, B = Host> = Var<'t, f32, B>;
/// A recorded length-`N` vector.
pub type VectorVar<'t, const N: usize, B = Host> = Var<'t, Vector<f32, N, B>, B>;
/// A recorded `R × C` matrix.
pub type MatrixVar<'t, const R: usize, const C: usize, B = Host> = Var<'t, Matrix<f32, R, C, B>, B>;

impl<'t, T: Adjoint<B>, B: Kernels> Var<'t, T, B> {
    /// The value computed in the forward pass.
    pub fn value(&self) -> &T {
        &self.node.value
    }

    /// The gradient accumulated by the last backward pass — zeros if this value
    /// did not reach the output that was seeded.
    pub fn grad(&self) -> T {
        match self.node.adjoint.borrow().as_ref() {
            Some(adjoint) => adjoint.duplicate(),
            None => T::zeros(),
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
}

impl<B: Kernels> ScalarVar<'_, B> {
    /// Propagate from this scalar, seeding `∂self/∂self = 1`.
    pub fn backward(&self) {
        self.backward_with(1.0);
    }
}

// ---- helpers ----------------------------------------------------------------

fn vector_op<const N: usize, B: Kernels>(
    a: &Vector<f32, N, B>,
    b: &Vector<f32, N, B>,
    op: BinaryOp,
) -> Vector<f32, N, B> {
    B::vector_elementwise(a, b, op)
}

fn matrix_op<const R: usize, const C: usize, B: Kernels>(
    a: &Matrix<f32, R, C, B>,
    b: &Matrix<f32, R, C, B>,
    op: BinaryOp,
) -> Matrix<f32, R, C, B> {
    B::matrix_elementwise(a, b, op)
}

fn negated_vector<const N: usize, B: Kernels>(v: &Vector<f32, N, B>) -> Vector<f32, N, B> {
    B::vector_broadcast(v, -1.0, BinaryOp::Mul, false)
}

fn negated_matrix<const R: usize, const C: usize, B: Kernels>(
    m: &Matrix<f32, R, C, B>,
) -> Matrix<f32, R, C, B> {
    B::matrix_broadcast(m, -1.0, BinaryOp::Mul, false)
}

/// Elementwise `trunc(a / b)`, the locally constant quotient behind `%`.
///
/// There is no truncation kernel, so this reads both operands in place — free on
/// a resident tensor — and stores the result back.
fn truncated_quotient(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x / y).trunc())
        .collect::<Vec<_>>()
}

// ---- scalars ----------------------------------------------------------------

impl<'t, B: Kernels> ScalarVar<'t, B> {
    fn binary(&self, rhs: &Self, op: BinaryOp) -> Self {
        self.assert_same_tape(rhs);
        let (x, y) = (self.node.value, rhs.node.value);
        let value = match op {
            BinaryOp::Add => x + y,
            BinaryOp::Sub => x - y,
            BinaryOp::Mul => x * y,
            BinaryOp::Div => x / y,
            BinaryOp::Rem => x % y,
        };
        let (left, right) = (self.node.clone(), rhs.node.clone());
        self.record(value, move |adjoint| {
            let adjoint = *adjoint;
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
                    left.accumulate(adjoint * y);
                    right.accumulate(adjoint * x);
                }
                BinaryOp::Div => {
                    left.accumulate(adjoint / y);
                    right.accumulate(-adjoint * x / (y * y));
                }
                // a % b = a − trunc(a/b)·b, and the quotient is locally constant.
                BinaryOp::Rem => {
                    left.accumulate(adjoint);
                    right.accumulate(-adjoint * (x / y).trunc());
                }
            }
        })
    }

    /// Apply an analytic function: `dy/dx = f'(x)`.
    pub fn analytic(&self, f: Analytic) -> Self {
        let x = self.node.value;
        let parent = self.node.clone();
        self.record(f.value(x), move |adjoint| {
            parent.accumulate(adjoint * f.derivative(x));
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
        let (x, y) = (self.node.value, other.node.value);
        let op = if largest { Compare::Max } else { Compare::Min };
        let (left, right) = (self.node.clone(), other.node.clone());
        self.record(op.value(x, y), move |adjoint| {
            let share = Compare::MaxShare.value(x, y);
            let (mine, theirs) = if largest {
                (share, 1.0 - share)
            } else {
                (1.0 - share, share)
            };
            left.accumulate(adjoint * mine);
            right.accumulate(adjoint * theirs);
        })
    }

    /// Absolute value, differentiating to `sign(x)` with `sign(0) = 0`.
    pub fn abs(&self) -> Self {
        self.maximum(&self.scale(-1.0))
    }

    /// `max(x, 0)`.
    pub fn relu(&self) -> Self {
        let zero = self.tape.scalar(0.0);
        self.maximum(&zero)
    }

    /// Negate.
    pub fn neg(&self) -> Self {
        self.scale(-1.0)
    }

    /// Multiply by a constant.
    pub fn scale(&self, factor: f32) -> Self {
        let parent = self.node.clone();
        self.record(self.node.value * factor, move |adjoint| {
            parent.accumulate(adjoint * factor);
        })
    }

    /// Add a constant, which leaves the derivative unchanged.
    pub fn shift(&self, offset: f32) -> Self {
        let parent = self.node.clone();
        self.record(self.node.value + offset, move |adjoint| {
            parent.accumulate(*adjoint);
        })
    }

    /// Broadcast this scalar across a length-`N` vector. Its gradient is the sum
    /// of the vector's adjoint, since it reaches every element.
    pub fn expand<const N: usize>(&self) -> VectorVar<'t, N, B> {
        let parent = self.node.clone();
        self.record(Vector::filled(self.node.value), move |adjoint| {
            parent.accumulate(adjoint.as_slice().iter().sum());
        })
    }

    /// Broadcast this scalar across an `R × C` matrix.
    pub fn expand_matrix<const R: usize, const C: usize>(&self) -> MatrixVar<'t, R, C, B> {
        let parent = self.node.clone();
        self.record(Matrix::filled(self.node.value), move |adjoint| {
            parent.accumulate(adjoint.as_slice().iter().sum());
        })
    }
}

// ---- vectors ----------------------------------------------------------------

impl<'t, const N: usize, B: Kernels> VectorVar<'t, N, B> {
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
                left.accumulate(vector_op(adjoint, &right.value, BinaryOp::Mul));
                right.accumulate(vector_op(adjoint, &left.value, BinaryOp::Mul));
            }
            // c = a / b: ā += c̄/b, b̄ −= c̄·a/b²
            BinaryOp::Div => {
                left.accumulate(vector_op(adjoint, &right.value, BinaryOp::Div));
                let squared = vector_op(&right.value, &right.value, BinaryOp::Mul);
                let scaled = vector_op(
                    &vector_op(adjoint, &left.value, BinaryOp::Mul),
                    &squared,
                    BinaryOp::Div,
                );
                right.accumulate(negated_vector(&scaled));
            }
            // c = a % b: the quotient is locally constant, so b̄ −= c̄·trunc(a/b)
            BinaryOp::Rem => {
                left.accumulate(adjoint.duplicate());
                let quotient = B::store_vector::<N>(&truncated_quotient(
                    left.value.as_slice(),
                    right.value.as_slice(),
                ));
                let scaled = vector_op(adjoint, &Vector { data: quotient }, BinaryOp::Mul);
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
            let share = B::vector_compare(&left.value, &right.value, Compare::MaxShare);
            let complement = B::vector_broadcast(&share, 1.0, BinaryOp::Sub, true);
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
    pub fn clamp_min(&self, floor: f32) -> Self {
        self.select_scalar(floor, true)
    }

    /// Elementwise minimum against a constant.
    pub fn clamp_max(&self, ceiling: f32) -> Self {
        self.select_scalar(ceiling, false)
    }

    /// Confine every element to `[floor, ceiling]`; the gradient is zero wherever
    /// an element is pinned to a bound.
    pub fn clamp(&self, floor: f32, ceiling: f32) -> Self {
        self.clamp_min(floor).clamp_max(ceiling)
    }

    /// The rectifier `max(a, 0)`.
    pub fn relu(&self) -> Self {
        self.clamp_min(0.0)
    }

    /// Elementwise absolute value, as `max(a, −a)`, which differentiates to
    /// `sign(a)` with `sign(0) = 0`.
    pub fn abs(&self) -> Self {
        self.maximum(&self.scale(-1.0))
    }

    fn select_scalar(&self, scalar: f32, largest: bool) -> Self {
        let op = if largest { Compare::Max } else { Compare::Min };
        let value = B::vector_compare_scalar(self.value(), scalar, op, false);
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            let share = B::vector_compare_scalar(&parent.value, scalar, Compare::MaxShare, false);
            let weight = if largest {
                share
            } else {
                B::vector_broadcast(&share, 1.0, BinaryOp::Sub, true)
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
            let (_, product) = B::vector_unary_dual(&parent.value, adjoint, f);
            parent.accumulate(product);
        })
    }

    /// Negate every element.
    pub fn neg(&self) -> Self {
        self.scale(-1.0)
    }

    /// Multiply by a constant.
    pub fn scale(&self, factor: f32) -> Self {
        let value = B::vector_broadcast(self.value(), factor, BinaryOp::Mul, false);
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            parent.accumulate(B::vector_broadcast(adjoint, factor, BinaryOp::Mul, false));
        })
    }

    /// Add a constant to every element, which leaves the derivative unchanged.
    pub fn shift(&self, offset: f32) -> Self {
        let value = B::vector_broadcast(self.value(), offset, BinaryOp::Add, false);
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            parent.accumulate(adjoint.duplicate());
        })
    }

    /// Sum of the elements: every element's gradient is the scalar's adjoint.
    pub fn sum(&self) -> ScalarVar<'t, B> {
        let total: f32 = self.value().as_slice().iter().sum();
        let parent = self.node.clone();
        self.record(total, move |adjoint| {
            parent.accumulate(Vector::filled(*adjoint));
        })
    }

    /// Dot product: `ū += s̄·v` and `v̄ += s̄·u`.
    pub fn dot(&self, other: &Self) -> ScalarVar<'t, B> {
        self.assert_same_tape(other);
        let value = B::dot(self.value(), other.value());
        let (left, right) = (self.node.clone(), other.node.clone());
        self.record(value, move |adjoint| {
            let scale = *adjoint;
            left.accumulate(B::vector_broadcast(
                &right.value,
                scale,
                BinaryOp::Mul,
                false,
            ));
            right.accumulate(B::vector_broadcast(
                &left.value,
                scale,
                BinaryOp::Mul,
                false,
            ));
        })
    }

    /// Row vector times matrix, `(1×N)·(N×C)`: `x̄ += A·ȳ` and `Ā += x ⊗ ȳ`.
    pub fn vecmat<const C: usize>(&self, m: &MatrixVar<'t, N, C, B>) -> VectorVar<'t, C, B> {
        self.assert_same_tape(m);
        let value = B::vecmat(self.value(), m.value());
        let (vector, matrix) = (self.node.clone(), m.node.clone());
        self.record(value, move |adjoint| {
            vector.accumulate(B::matvec(&matrix.value, adjoint));
            matrix.accumulate(outer(&vector.value, adjoint));
        })
    }

    /// View as a `1 × N` matrix; the adjoint flows straight back.
    pub fn into_row(&self) -> MatrixVar<'t, 1, N, B> {
        let value = Matrix {
            data: B::vector_into_row::<N>(self.value().duplicate().data),
        };
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            parent.accumulate(Vector {
                data: B::store_vector::<N>(adjoint.as_slice()),
            });
        })
    }

    /// Outer product `u ⊗ v`: the `N × M` matrix with entries `uᵢvⱼ`.
    ///
    /// The rank-one product that shows up wherever a gradient meets an input —
    /// it is what `matvec` accumulates into its matrix operand, and what a
    /// weight update looks like. Its own rules are the two contractions of the
    /// adjoint: `ū = Ȳ·v` and `v̄ = uᵀ·Ȳ`.
    pub fn outer<const M: usize>(&self, other: &VectorVar<'t, M, B>) -> MatrixVar<'t, N, M, B> {
        self.assert_same_tape(other);
        let value = outer(self.value(), other.value());
        let (left, right) = (self.node.clone(), other.node.clone());
        self.record(value, move |adjoint| {
            left.accumulate(B::matvec(adjoint, &right.value));
            right.accumulate(B::vecmat(&left.value, adjoint));
        })
    }

    /// View as an `N × 1` matrix; the adjoint flows straight back.
    pub fn into_column(&self) -> MatrixVar<'t, N, 1, B> {
        let value = Matrix {
            data: B::vector_into_column::<N>(self.value().duplicate().data),
        };
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            parent.accumulate(Vector {
                data: B::store_vector::<N>(adjoint.as_slice()),
            });
        })
    }
}

/// The outer product `u ⊗ v` as an `R × C` matrix, built from the column/row
/// views the backend already provides.
fn outer<const R: usize, const C: usize, B: Kernels>(
    u: &Vector<f32, R, B>,
    v: &Vector<f32, C, B>,
) -> Matrix<f32, R, C, B> {
    let column: Matrix<f32, R, 1, B> = Matrix {
        data: B::vector_into_column::<R>(u.to_backend::<B>().data),
    };
    let row: Matrix<f32, 1, C, B> = Matrix {
        data: B::vector_into_row::<C>(v.to_backend::<B>().data),
    };
    B::matmul(&column, &row)
}

// ---- matrices ---------------------------------------------------------------

impl<'t, const R: usize, const C: usize, B: Kernels> MatrixVar<'t, R, C, B> {
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
                left.accumulate(matrix_op(adjoint, &right.value, BinaryOp::Mul));
                right.accumulate(matrix_op(adjoint, &left.value, BinaryOp::Mul));
            }
            BinaryOp::Div => {
                left.accumulate(matrix_op(adjoint, &right.value, BinaryOp::Div));
                let squared = matrix_op(&right.value, &right.value, BinaryOp::Mul);
                let scaled = matrix_op(
                    &matrix_op(adjoint, &left.value, BinaryOp::Mul),
                    &squared,
                    BinaryOp::Div,
                );
                right.accumulate(negated_matrix(&scaled));
            }
            BinaryOp::Rem => {
                left.accumulate(adjoint.duplicate());
                let quotient = B::store_matrix::<R, C>(&truncated_quotient(
                    left.value.as_slice(),
                    right.value.as_slice(),
                ));
                let scaled = matrix_op(adjoint, &Matrix { data: quotient }, BinaryOp::Mul);
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
            let share = B::matrix_compare(&left.value, &right.value, Compare::MaxShare);
            let complement = B::matrix_broadcast(&share, 1.0, BinaryOp::Sub, true);
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
    pub fn clamp_min(&self, floor: f32) -> Self {
        self.select_scalar(floor, true)
    }

    /// Elementwise minimum against a constant.
    pub fn clamp_max(&self, ceiling: f32) -> Self {
        self.select_scalar(ceiling, false)
    }

    /// Confine every element to `[floor, ceiling]`.
    pub fn clamp(&self, floor: f32, ceiling: f32) -> Self {
        self.clamp_min(floor).clamp_max(ceiling)
    }

    /// The rectifier `max(A, 0)`.
    pub fn relu(&self) -> Self {
        self.clamp_min(0.0)
    }

    /// Elementwise absolute value; see [`VectorVar::abs`].
    pub fn abs(&self) -> Self {
        self.maximum(&self.scale(-1.0))
    }

    fn select_scalar(&self, scalar: f32, largest: bool) -> Self {
        let op = if largest { Compare::Max } else { Compare::Min };
        let value = B::matrix_compare_scalar(self.value(), scalar, op, false);
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            let share = B::matrix_compare_scalar(&parent.value, scalar, Compare::MaxShare, false);
            let weight = if largest {
                share
            } else {
                B::matrix_broadcast(&share, 1.0, BinaryOp::Sub, true)
            };
            parent.accumulate(B::matrix_elementwise(adjoint, &weight, BinaryOp::Mul));
        })
    }

    /// Sum along each row, one entry per row.
    ///
    /// This is `A·1`, so it reuses the product kernels rather than needing a
    /// reduction of its own, and the adjoint spreads straight back along each
    /// row: `Ā += ȳ ⊗ 1`.
    pub fn row_sums(&self) -> VectorVar<'t, R, B> {
        let value = B::matvec(self.value(), &Vector::<f32, C, B>::filled(1.0));
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            parent.accumulate(outer(adjoint, &Vector::<f32, C, B>::filled(1.0)));
        })
    }

    /// Sum along each column, one entry per column: `1ᵀ·A`, with `Ā += 1 ⊗ ȳ`.
    pub fn column_sums(&self) -> VectorVar<'t, C, B> {
        let value = B::vecmat(&Vector::<f32, R, B>::filled(1.0), self.value());
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            parent.accumulate(outer(&Vector::<f32, R, B>::filled(1.0), adjoint));
        })
    }

    /// Apply an analytic function elementwise; see [`VectorVar::analytic`].
    pub fn analytic(&self, f: Analytic) -> Self {
        let value = B::matrix_unary(self.value(), f);
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            let (_, product) = B::matrix_unary_dual(&parent.value, adjoint, f);
            parent.accumulate(product);
        })
    }

    /// Negate every element.
    pub fn neg(&self) -> Self {
        self.scale(-1.0)
    }

    /// Multiply by a constant.
    pub fn scale(&self, factor: f32) -> Self {
        let value = B::matrix_broadcast(self.value(), factor, BinaryOp::Mul, false);
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            parent.accumulate(B::matrix_broadcast(adjoint, factor, BinaryOp::Mul, false));
        })
    }

    /// Add a constant to every element.
    pub fn shift(&self, offset: f32) -> Self {
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
    pub fn matmul<const C2: usize>(
        &self,
        other: &MatrixVar<'t, C, C2, B>,
    ) -> MatrixVar<'t, R, C2, B> {
        self.assert_same_tape(other);
        let value = B::matmul(self.value(), other.value());
        let (left, right) = (self.node.clone(), other.node.clone());
        self.record(value, move |adjoint| {
            let left_adjoint = B::matmul_add(
                adjoint,
                &B::transpose(&right.value),
                current_or_zeros(&left.adjoint),
            );
            *left.adjoint.borrow_mut() = Some(left_adjoint);

            let right_adjoint = B::matmul_add(
                &B::transpose(&left.value),
                adjoint,
                current_or_zeros(&right.adjoint),
            );
            *right.adjoint.borrow_mut() = Some(right_adjoint);
        })
    }

    /// Fused matrix multiply-add: `self·other + addend`.
    pub fn matmul_add<const C2: usize>(
        &self,
        other: &MatrixVar<'t, C, C2, B>,
        addend: &MatrixVar<'t, R, C2, B>,
    ) -> MatrixVar<'t, R, C2, B> {
        self.assert_same_tape(other);
        self.assert_same_tape(addend);
        let value = B::matmul_add(self.value(), other.value(), addend.value().duplicate());
        let (left, right, bias) = (self.node.clone(), other.node.clone(), addend.node.clone());
        self.record(value, move |adjoint| {
            let left_adjoint = B::matmul_add(
                adjoint,
                &B::transpose(&right.value),
                current_or_zeros(&left.adjoint),
            );
            *left.adjoint.borrow_mut() = Some(left_adjoint);

            let right_adjoint = B::matmul_add(
                &B::transpose(&left.value),
                adjoint,
                current_or_zeros(&right.adjoint),
            );
            *right.adjoint.borrow_mut() = Some(right_adjoint);
            bias.accumulate(adjoint.duplicate());
        })
    }

    /// Matrix times vector: `Ā += ȳ ⊗ x` and `x̄ += Aᵀ·ȳ`.
    pub fn matvec(&self, v: &VectorVar<'t, C, B>) -> VectorVar<'t, R, B> {
        self.assert_same_tape(v);
        let value = B::matvec(self.value(), v.value());
        let (matrix, vector) = (self.node.clone(), v.node.clone());
        self.record(value, move |adjoint| {
            matrix.accumulate(outer(adjoint, &vector.value));
            vector.accumulate(B::vecmat(adjoint, &matrix.value));
        })
    }

    /// Fused matrix-vector multiply-add: `self·v + addend`.
    pub fn matvec_add(
        &self,
        v: &VectorVar<'t, C, B>,
        addend: &VectorVar<'t, R, B>,
    ) -> VectorVar<'t, R, B> {
        self.assert_same_tape(v);
        self.assert_same_tape(addend);
        let value = B::matvec_add(self.value(), v.value(), addend.value().duplicate());
        let (matrix, vector, bias) = (self.node.clone(), v.node.clone(), addend.node.clone());
        self.record(value, move |adjoint| {
            matrix.accumulate(outer(adjoint, &vector.value));
            vector.accumulate(B::vecmat(adjoint, &matrix.value));
            bias.accumulate(adjoint.duplicate());
        })
    }

    /// Transpose; the adjoint transposes back.
    pub fn transpose(&self) -> MatrixVar<'t, C, R, B> {
        let value = B::transpose(self.value());
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            parent.accumulate(B::transpose(adjoint));
        })
    }

    /// Sum of every element.
    pub fn sum(&self) -> ScalarVar<'t, B> {
        let total: f32 = self.value().as_slice().iter().sum();
        let parent = self.node.clone();
        self.record(total, move |adjoint| {
            parent.accumulate(Matrix::filled(*adjoint));
        })
    }

    /// The Frobenius inner product `Σᵢⱼ aᵢⱼbᵢⱼ`, the usual scalar loss.
    pub fn frobenius_dot(&self, other: &Self) -> ScalarVar<'t, B> {
        self.binary(other, BinaryOp::Mul).sum()
    }

    /// Row-major flattening, and its exact inverse on the way back.
    pub fn flattened(&self) -> VectorVar<'t, { R * C }, B> {
        let value = Vector {
            data: B::matrix_into_flattened::<R, C>(self.value().duplicate().data),
        };
        let parent = self.node.clone();
        self.record(value, move |adjoint| {
            parent.accumulate(Matrix {
                data: B::store_matrix::<R, C>(adjoint.as_slice()),
            });
        })
    }
}

/// The adjoint accumulated so far, or a fresh zero — the starting point for an
/// accumulating matmul.
fn current_or_zeros<T: Adjoint<B>, B: Kernels>(slot: &RefCell<Option<T>>) -> T {
    match slot.borrow().as_ref() {
        Some(current) => current.duplicate(),
        None => T::zeros(),
    }
}

// ---- named operations and operators -----------------------------------------

/// The binary operations, as inherent methods and as operators on references.
macro_rules! binary_methods {
    ($($method:ident => $op:expr, $trait:ident :: $trait_method:ident),+ $(,)?) => {
        impl<'t, B: Kernels> ScalarVar<'t, B> {
            $(
                #[doc = concat!("Elementwise `", stringify!($method), "`, differentiated.")]
                pub fn $method(&self, rhs: &Self) -> Self { self.binary(rhs, $op) }
            )+
        }

        impl<'t, const N: usize, B: Kernels> VectorVar<'t, N, B> {
            $(
                #[doc = concat!("Elementwise `", stringify!($method), "`, differentiated.")]
                pub fn $method(&self, rhs: &Self) -> Self { self.binary(rhs, $op) }
            )+
        }

        impl<'t, const R: usize, const C: usize, B: Kernels> MatrixVar<'t, R, C, B> {
            $(
                #[doc = concat!("Elementwise `", stringify!($method), "`, differentiated.")]
                pub fn $method(&self, rhs: &Self) -> Self { self.binary(rhs, $op) }
            )+
        }

        $(
            impl<'t, B: Kernels> $trait for &ScalarVar<'t, B> {
                type Output = ScalarVar<'t, B>;
                fn $trait_method(self, rhs: Self) -> Self::Output { self.binary(rhs, $op) }
            }

            impl<'t, const N: usize, B: Kernels> $trait for &VectorVar<'t, N, B> {
                type Output = VectorVar<'t, N, B>;
                fn $trait_method(self, rhs: Self) -> Self::Output { self.binary(rhs, $op) }
            }

            impl<'t, const R: usize, const C: usize, B: Kernels> $trait
                for &MatrixVar<'t, R, C, B>
            {
                type Output = MatrixVar<'t, R, C, B>;
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

/// The analytic functions, as methods on all three shapes.
macro_rules! analytic_methods {
    ($($method:ident => $variant:ident),+ $(,)?) => {
        impl<'t, B: Kernels> ScalarVar<'t, B> {
            $(
                #[doc = concat!("`", stringify!($method), "`, differentiated.")]
                pub fn $method(&self) -> Self { self.analytic(Analytic::$variant) }
            )+
        }

        impl<'t, const N: usize, B: Kernels> VectorVar<'t, N, B> {
            $(
                #[doc = concat!("Elementwise `", stringify!($method), "`, differentiated.")]
                pub fn $method(&self) -> Self { self.analytic(Analytic::$variant) }
            )+
        }

        impl<'t, const R: usize, const C: usize, B: Kernels> MatrixVar<'t, R, C, B> {
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
);

// ---- whole derivatives ------------------------------------------------------

/// The gradient of a scalar-valued `f` at `at`, in one backward pass.
///
/// The counterpart to [`dual::gradient`](super::dual::gradient), which needs one
/// forward pass per input element. This one runs the function once whatever the
/// input size.
pub fn gradient<const IN: usize, B: Kernels>(
    at: &Vector<f32, IN, B>,
    f: impl for<'t> FnOnce(&VectorVar<'t, IN, B>) -> ScalarVar<'t, B>,
) -> Vector<f32, IN, B> {
    let tape = Tape::<B>::new();
    let input = tape.vector(at.to_backend::<B>());
    f(&input).backward();
    input.grad()
}

/// The gradient of a scalar-valued `f` with respect to a matrix input, in one
/// backward pass — where
/// [`dual::gradient_wrt_matrix`](super::dual::gradient_wrt_matrix) needs `R * C`
/// forward ones.
pub fn gradient_wrt_matrix<const R: usize, const C: usize, B: Kernels>(
    at: &Matrix<f32, R, C, B>,
    f: impl for<'t> FnOnce(&MatrixVar<'t, R, C, B>) -> ScalarVar<'t, B>,
) -> Matrix<f32, R, C, B> {
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
/// Which mode is cheaper is a matter of shape: forward costs `IN` passes,
/// reverse costs `OUT`. For a tall Jacobian prefer forward, for a wide one
/// prefer reverse, and they agree to within floating-point error either way.
///
/// ```
/// use rinterp::tensors::Vector;
/// use rinterp::tensors::tape::jacobian;
///
/// // An elementwise map has a diagonal Jacobian: d sin(x)ᵢ/dxⱼ = δᵢⱼ cos(xᵢ).
/// let at = Vector::new([0.5f32, 2.0]);
/// let computed = jacobian(&at, |x| x.sin());
///
/// assert!((computed.data()[0][0] - 0.5f32.cos()).abs() < 1e-6);
/// assert!((computed.data()[1][1] - 2.0f32.cos()).abs() < 1e-6);
/// assert_eq!(computed.data()[0][1], 0.0);
/// assert_eq!(computed.data()[1][0], 0.0);
/// ```
pub fn jacobian<const IN: usize, const OUT: usize, B: Kernels>(
    at: &Vector<f32, IN, B>,
    f: impl for<'t> FnOnce(&VectorVar<'t, IN, B>) -> VectorVar<'t, OUT, B>,
) -> Matrix<f32, OUT, IN, B> {
    let tape = Tape::<B>::new();
    let input = tape.vector(at.to_backend::<B>());
    let output = f(&input);
    let rows = std::array::from_fn(|row| {
        output.backward_with(basis_vector::<OUT, B>(row));
        input.grad().data
    });
    Matrix {
        data: B::vstack::<OUT, IN>(rows),
    }
}

/// The full Jacobian of a vector-valued `f` with respect to a matrix input, as
/// `OUT × (R·C)` with columns in the row-major order of the input.
///
/// A matrix-valued `f` needs no separate driver: flatten its output with
/// [`MatrixVar::flattened`] and the result is the standard
/// `(OR·OC) × (R·C)` Jacobian.
pub fn jacobian_wrt_matrix<const R: usize, const C: usize, const OUT: usize, B: Kernels>(
    at: &Matrix<f32, R, C, B>,
    f: impl for<'t> FnOnce(&MatrixVar<'t, R, C, B>) -> VectorVar<'t, OUT, B>,
) -> Matrix<f32, OUT, { R * C }, B> {
    let tape = Tape::<B>::new();
    let input = tape.matrix(at.to_backend::<B>());
    let output = f(&input);
    let rows = std::array::from_fn(|row| {
        output.backward_with(basis_vector::<OUT, B>(row));
        B::matrix_into_flattened::<R, C>(input.grad().data)
    });
    Matrix {
        data: B::vstack::<OUT, { R * C }>(rows),
    }
}

/// The `index`th standard basis vector, the seed that extracts one Jacobian row.
fn basis_vector<const N: usize, B: Kernels>(index: usize) -> Vector<f32, N, B> {
    let mut seed = vec![0.0f32; N];
    if let Some(slot) = seed.get_mut(index) {
        *slot = 1.0;
    }
    Vector {
        data: B::store_vector::<N>(&seed),
    }
}
