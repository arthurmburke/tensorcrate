//! The value graph a fused program is optimized as.
//!
//! A [`Graph`] is a program before registers: single-assignment nodes, each
//! naming its operands by index, in an order where every operand precedes its
//! uses, plus the stores that make values the program's outputs. Nothing here
//! knows the element type; constants are symbolic ([`Scalar`]) so that folding
//! them stays the caller's business, done in the program's own type.

/// A node's index in its graph.
pub type Value = usize;

/// The arithmetic operations, numbered as `tensorcrate::tensors::BinaryOp`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Bin {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    Rem = 4,
}

impl Bin {
    /// Every operation, in code order.
    pub const ALL: [Bin; 5] = [Bin::Add, Bin::Sub, Bin::Mul, Bin::Div, Bin::Rem];

    pub fn from_code(code: u16) -> Option<Bin> {
        Self::ALL.get(usize::from(code)).copied()
    }

    /// Whether `a op b` is `b op a`, bit for bit.
    pub fn commutes(self) -> bool {
        matches!(self, Bin::Add | Bin::Mul)
    }
}

/// The comparisons, numbered as `tensorcrate::tensors::Compare`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Cmp {
    Min = 0,
    Max = 1,
    MaxShare = 2,
    Less = 3,
    LessEqual = 4,
    Greater = 5,
    GreaterEqual = 6,
}

impl Cmp {
    pub const ALL: [Cmp; 7] = [
        Cmp::Min,
        Cmp::Max,
        Cmp::MaxShare,
        Cmp::Less,
        Cmp::LessEqual,
        Cmp::Greater,
        Cmp::GreaterEqual,
    ];

    pub fn from_code(code: u16) -> Option<Cmp> {
        Self::ALL.get(usize::from(code)).copied()
    }

    /// The same comparison with its operands swapped: `a < b` is `b > a`.
    /// `None` for `MaxShare`, which has no mirror image among the others.
    pub fn mirrored(self) -> Option<Cmp> {
        Some(match self {
            Cmp::Min => Cmp::Min,
            Cmp::Max => Cmp::Max,
            Cmp::Less => Cmp::Greater,
            Cmp::LessEqual => Cmp::GreaterEqual,
            Cmp::Greater => Cmp::Less,
            Cmp::GreaterEqual => Cmp::LessEqual,
            Cmp::MaxShare => return None,
        })
    }
}

/// An analytic function, by its `tensorcrate::tensors::Analytic` code.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Function(pub u16);

impl Function {
    /// `Analytic::Sqrt`, the one function that is a single hardware operation
    /// rather than a library routine.
    pub const SQRT: Function = Function(13);

    pub fn is_transcendental(self) -> bool {
        self != Self::SQRT
    }
}

/// A constant: a scalar computed once, before the kernel, and the same for
/// every element.
///
/// Leaves are the caller's: a value it knows ([`Literal`](Scalar::Literal)) or
/// one only it can name ([`Named`](Scalar::Named) — a `math!` scalar
/// subexpression, whose value exists only at run time). Folding two constants
/// builds a [`Binary`](Scalar::Binary), which the caller evaluates in the
/// program's element type, exactly as the unfused kernels would have combined
/// them.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Scalar {
    /// The bits of an `f64` holding the value exactly.
    Literal(u64),
    /// The caller's constant number `id`, whose value is not known here.
    Named(u32),
    Binary(Bin, Box<Scalar>, Box<Scalar>),
}

impl Scalar {
    pub fn literal(value: f64) -> Scalar {
        Scalar::Literal(value.to_bits())
    }

    /// The value, when it is a literal.
    pub fn known(&self) -> Option<f64> {
        match *self {
            Scalar::Literal(bits) => Some(f64::from_bits(bits)),
            _ => None,
        }
    }

    /// Whether this is exactly `value`, sign of zero included.
    pub fn is(&self, value: f64) -> bool {
        self.known()
            .is_some_and(|known| known.to_bits() == value.to_bits())
    }
}

/// One single-assignment operation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Node {
    /// Read input `slot` through remap code `remap` (`fused::Remap`).
    Load {
        slot: u8,
        remap: u8,
    },
    Const(Scalar),
    Binary(Bin, Value, Value),
    Unary(Function, Value),
    Cmp(Cmp, Value, Value),
}

impl Node {
    /// The values this node reads, in operand order.
    pub fn operands(&self) -> Vec<Value> {
        match *self {
            Node::Binary(_, a, b) | Node::Cmp(_, a, b) => vec![a, b],
            Node::Unary(_, a) => vec![a],
            Node::Load { .. } | Node::Const(_) => vec![],
        }
    }

    /// The same node reading `map(operand)` in place of each operand.
    pub fn remapped(&self, map: impl Fn(Value) -> Value) -> Node {
        match *self {
            Node::Binary(op, a, b) => Node::Binary(op, map(a), map(b)),
            Node::Cmp(op, a, b) => Node::Cmp(op, map(a), map(b)),
            Node::Unary(f, a) => Node::Unary(f, map(a)),
            ref other => other.clone(),
        }
    }

    /// An operation an extra copy of costs nothing but its instruction: it
    /// reads no registers.
    pub fn is_leaf(&self) -> bool {
        matches!(self, Node::Load { .. } | Node::Const(_))
    }
}

/// Make `value` output number `output`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Store {
    pub value: Value,
    pub output: u8,
}

/// A program as values.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Graph {
    /// Every operand precedes its uses.
    pub nodes: Vec<Node>,
    /// Run after every node, in this order.
    pub stores: Vec<Store>,
    /// Bytes per element of each input slot's storage, for the memory traffic.
    pub input_bytes: Vec<u8>,
    /// The same for each output slot.
    pub output_bytes: Vec<u8>,
}

impl Graph {
    /// The values that are some store's.
    pub fn stored(&self) -> Vec<bool> {
        let mut stored = vec![false; self.nodes.len()];
        for store in &self.stores {
            stored[store.value] = true;
        }
        stored
    }

    /// Each value's readers, in order.
    pub fn users(&self) -> Vec<Vec<Value>> {
        let mut users = vec![Vec::new(); self.nodes.len()];
        for (at, node) in self.nodes.iter().enumerate() {
            for operand in node.operands() {
                if !users[operand].contains(&at) {
                    users[operand].push(at);
                }
            }
        }
        users
    }

    /// Check that every operand precedes its use and every store names a value.
    pub fn validate(&self) -> Result<(), String> {
        for (at, node) in self.nodes.iter().enumerate() {
            if node.operands().iter().any(|&operand| operand >= at) {
                return Err(format!("node {at} reads a value defined after it"));
            }
            if let Node::Load { slot, .. } = node
                && usize::from(*slot) >= self.input_bytes.len()
            {
                return Err(format!("node {at} loads undeclared input {slot}"));
            }
        }
        for store in &self.stores {
            if store.value >= self.nodes.len() {
                return Err(format!("a store reads undefined value {}", store.value));
            }
            if usize::from(store.output) >= self.output_bytes.len() {
                return Err(format!("a store writes undeclared output {}", store.output));
            }
        }
        Ok(())
    }
}
