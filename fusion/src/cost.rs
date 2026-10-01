//! What a program costs.
//!
//! ```text
//! C = α · instructions + β · peak registers + γ · critical path
//!   + δ · memory traffic + ε · special operations
//! ```
//!
//! - **instructions**: every instruction but the stores.
//! - **peak registers**: the most values live at once, which is also the number
//!   of registers the program needs.
//! - **critical path**: the longest chain of dependent operations, weighted by
//!   each operation's [`Latency`] — what bounds a single element's time when
//!   there is nothing else to overlap it with.
//! - **memory traffic**: bytes moved per element, every load and store charged
//!   its storage size; a value loaded twice is charged twice.
//! - **special operations**: divisions, remainders, square roots and
//!   transcendental functions, which cost a library call or a long-latency unit
//!   rather than one arithmetic instruction.
//!
//! The coefficients say what the program will run on. The host tile
//! interpreter runs each instruction as a pass over a tile, so instructions
//! dominate; a compiled GPU kernel has instructions to spare and is bound by
//! memory and by the registers that limit how many threads it can keep in
//! flight.

/// The coefficients of the cost function, and the latencies its critical path
/// is measured in.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CostModel {
    /// Per instruction.
    pub alpha: f64,
    /// Per register at the peak.
    pub beta: f64,
    /// Per unit of critical-path latency.
    pub gamma: f64,
    /// Per byte moved per element.
    pub delta: f64,
    /// Per special operation.
    pub epsilon: f64,
    pub latency: Latency,
}

/// Relative latencies, in units of one arithmetic operation.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Latency {
    pub load: f64,
    pub constant: f64,
    pub arithmetic: f64,
    pub divide: f64,
    pub sqrt: f64,
    pub transcendental: f64,
}

impl Latency {
    pub const DEFAULT: Latency = Latency {
        load: 4.0,
        constant: 0.0,
        arithmetic: 1.0,
        divide: 4.0,
        sqrt: 4.0,
        transcendental: 16.0,
    };
}

impl CostModel {
    /// The host tile interpreter: each instruction is a pass over a
    /// cache-resident tile, so instruction count dominates; registers are
    /// tiles in L1, cheap until they run out.
    pub const HOST: CostModel = CostModel {
        alpha: 1.0,
        beta: 0.1,
        gamma: 0.05,
        delta: 0.25,
        epsilon: 4.0,
        latency: Latency::DEFAULT,
    };

    /// A compiled GPU kernel: memory-bound, with registers limiting how many
    /// threads stay resident and latency hidden only by those threads.
    pub const METAL: CostModel = CostModel {
        alpha: 0.25,
        beta: 0.5,
        gamma: 0.25,
        delta: 1.0,
        epsilon: 2.0,
        latency: Latency::DEFAULT,
    };

    /// For a program whose backend is not known when it is built.
    pub const BALANCED: CostModel = CostModel {
        alpha: 0.6,
        beta: 0.3,
        gamma: 0.15,
        delta: 0.6,
        epsilon: 3.0,
        latency: Latency::DEFAULT,
    };
}

impl Default for CostModel {
    fn default() -> Self {
        CostModel::BALANCED
    }
}

/// A program's cost, term by term.
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct Cost {
    pub instructions: usize,
    pub peak_registers: usize,
    pub critical_path: f64,
    pub memory_bytes: usize,
    pub special_ops: usize,
    /// The weighted sum.
    pub total: f64,
}

impl Cost {
    pub(crate) fn weigh(
        model: &CostModel,
        instructions: usize,
        peak_registers: usize,
        critical_path: f64,
        memory_bytes: usize,
        special_ops: usize,
    ) -> Cost {
        let total = model.alpha * instructions as f64
            + model.beta * peak_registers as f64
            + model.gamma * critical_path
            + model.delta * memory_bytes as f64
            + model.epsilon * special_ops as f64;
        Cost {
            instructions,
            peak_registers,
            critical_path,
            memory_bytes,
            special_ops,
            total,
        }
    }
}

impl std::fmt::Display for Cost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "cost {:.2}: {} instructions, {} registers, critical path {:.1}, {} bytes, {} special",
            self.total,
            self.instructions,
            self.peak_registers,
            self.critical_path,
            self.memory_bytes,
            self.special_ops
        )
    }
}
