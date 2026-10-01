//! The fused-kernel optimizer shared by `tensorcrate` and its `math!` macro.
//!
//! A fused elementwise program is a small dataflow graph run once per element.
//! Many programs compute the same values — the graph can be rewritten,
//! scheduled in any topological order and given registers in many ways — and
//! they differ in what they cost. This crate enumerates those equivalent
//! programs and picks the cheapest under a [`CostModel`]:
//!
//! ```text
//! C = α · instructions + β · peak registers + γ · critical path
//!   + δ · memory traffic + ε · special operations
//! ```
//!
//! The candidates are the product of
//!
//! - **rewrites** ([`Association`]): common subexpressions merged and exact
//!   identities applied always; associative chains reassociated, left-deep or
//!   balanced, when [`Options::reassociate`] allows;
//! - **rematerialization** ([`Rematerialize`]): constants, and loads, emitted
//!   again at each use instead of held in a register;
//! - **schedules** ([`Schedule`]): several heuristic orders, and every order for
//!   small graphs.
//!
//! The program as written is always a candidate, so the result never costs
//! more than the input.
//!
//! It is a separate crate because both of its users need it: the macro runs
//! it at compile time on the kernels it fuses statically, and the library at
//! run time on programs built with `fused::Builder`. One optimizer means a
//! kernel written either way comes out the same.

mod cost;
mod graph;
mod optimize;
mod rewrite;
mod schedule;

pub use cost::{Cost, CostModel, Latency};
pub use graph::{Bin, Cmp, Function, Graph, Node, Scalar, Store, Value};
pub use optimize::{DoesNotFit, Options, Plan, Variant, candidates, cost, optimize};
pub use rewrite::{Association, Rematerialize};
pub use schedule::{Instr, Schedule};

#[cfg(test)]
mod tests;
