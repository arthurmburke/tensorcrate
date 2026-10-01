//! The `math!` language as a tree.
//!
//! [`parse`](crate::parse) builds an [`Expr<()>`] from the macro input, checking
//! everything that does not depend on types: the syntax, the literals and the
//! built-in functions with their arities. [`types`](crate::types) then
//! annotates every node with its [`Ty`](crate::types::Ty), giving the
//! `Expr<Ty>` that [`emit`](crate::emit) and [`fusion`](crate::fusion)
//! generate code from. One tree serves both stages, so the language is
//! described once.

use proc_macro2::{Ident, Span};

/// An expression, with an annotation `T` on every node: nothing after parsing,
/// its type after checking.
#[derive(Clone, Debug)]
pub(crate) struct Expr<T = ()> {
    pub(crate) kind: ExprKind<T>,
    pub(crate) span: Span,
    pub(crate) ty: T,
}

#[derive(Clone, Debug)]
pub(crate) enum ExprKind<T> {
    Literal(Literal),
    /// `[a, b]` is a vector, `[[a, b], [c, d]]` a matrix. Matrix rows all have
    /// the same length; a vector is one row.
    Tensor {
        rows: Vec<Vec<Expr<T>>>,
        matrix: bool,
    },
    /// A name: a binding of the block, or else an `f64` from the surrounding
    /// Rust code.
    Var(Ident),
    /// Parentheses, or an invisible group left by a `macro_rules!` expansion.
    Group {
        inner: Box<Expr<T>>,
        parenthesized: bool,
    },
    Neg(Box<Expr<T>>),
    /// `+ - * / %`. Between two vectors, `*` is their dot product.
    Binary {
        op: Arith,
        left: Box<Expr<T>>,
        right: Box<Expr<T>>,
    },
    /// `a @ b`: a matrix/matrix, matrix/vector or vector/matrix product.
    MatMul(Box<Expr<T>>, Box<Expr<T>>),
    /// `a .* b`: always elementwise.
    ElementwiseMul(Box<Expr<T>>, Box<Expr<T>>),
    Call {
        builtin: Builtin,
        args: Vec<Expr<T>>,
        /// Where the function's name was written.
        name_span: Span,
    },
}

impl<T> Expr<T> {
    /// The expressions directly inside this one, in source order.
    pub(crate) fn children(&self) -> Vec<&Expr<T>> {
        match &self.kind {
            ExprKind::Literal(_) | ExprKind::Var(_) => vec![],
            ExprKind::Tensor { rows, .. } => rows.iter().flatten().collect(),
            ExprKind::Group { inner, .. } | ExprKind::Neg(inner) => vec![inner],
            ExprKind::Binary { left, right, .. }
            | ExprKind::MatMul(left, right)
            | ExprKind::ElementwiseMul(left, right) => vec![left, right],
            ExprKind::Call { args, .. } => args.iter().collect(),
        }
    }

    /// Whether this expression, or any inside it, satisfies `test`.
    pub(crate) fn any(&self, test: &impl Fn(&Expr<T>) -> bool) -> bool {
        test(self) || self.children().into_iter().any(|child| child.any(test))
    }
}

/// The arithmetic operators.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Arith {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}

impl Arith {
    /// The variant of `tensorcrate::tensors::BinaryOp` with the same meaning.
    pub(crate) fn variant(self) -> &'static str {
        match self {
            Arith::Add => "Add",
            Arith::Sub => "Sub",
            Arith::Mul => "Mul",
            Arith::Div => "Div",
            Arith::Rem => "Rem",
        }
    }
}

/// A numeric literal: its value and the axis its suffix introduces.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Literal {
    pub(crate) kind: LitKind,
    pub(crate) value: f64,
}

/// Which axis a literal's suffix introduces: none, `i` or `d` (for ε).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum LitKind {
    Real,
    Imaginary,
    Epsilon,
}

/// The functions callable in `math!`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Builtin {
    Min,
    Max,
    Clamp,
    Sum,
    Minimum,
    Maximum,
    PrefixSum,
    Sorted { descending: bool },
    Pow,
    Dot,
    MatMul,
    Det,
    Inv,
    Transpose,
    Conj,
    Analytic(Function),
}

impl Builtin {
    /// The built-in called `name` with `arity` arguments, if there is one.
    /// `sorted` takes its order as an optional second argument, which the
    /// parser reads; it is resolved here as ascending.
    pub(crate) fn resolve(name: &str, arity: usize) -> Option<Builtin> {
        Some(match (name, arity) {
            ("min", 2) => Builtin::Min,
            ("max", 2) => Builtin::Max,
            ("clamp", 3) => Builtin::Clamp,
            ("sum", 1) => Builtin::Sum,
            ("minimum", 1) => Builtin::Minimum,
            ("maximum", 1) => Builtin::Maximum,
            ("prefix_sum", 1) => Builtin::PrefixSum,
            ("sorted", 1 | 2) => Builtin::Sorted { descending: false },
            ("pow", 2) => Builtin::Pow,
            ("dot", 2) => Builtin::Dot,
            ("matmul", 2) => Builtin::MatMul,
            ("det", 1) => Builtin::Det,
            ("inv", 1) => Builtin::Inv,
            ("transpose", 1) => Builtin::Transpose,
            ("conj", 1) => Builtin::Conj,
            (_, 1) => Builtin::Analytic(Function::named(name)?),
            _ => return None,
        })
    }

    /// The name it is called by.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Builtin::Min => "min",
            Builtin::Max => "max",
            Builtin::Clamp => "clamp",
            Builtin::Sum => "sum",
            Builtin::Minimum => "minimum",
            Builtin::Maximum => "maximum",
            Builtin::PrefixSum => "prefix_sum",
            Builtin::Sorted { .. } => "sorted",
            Builtin::Pow => "pow",
            Builtin::Dot => "dot",
            Builtin::MatMul => "matmul",
            Builtin::Det => "det",
            Builtin::Inv => "inv",
            Builtin::Transpose => "transpose",
            Builtin::Conj => "conj",
            Builtin::Analytic(function) => function.name(),
        }
    }
}

/// The analytic functions, each defined for every numeric type and mapped
/// elementwise over tensors.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Function {
    Sin,
    Cos,
    Tan,
    Sec,
    Csc,
    Arcsin,
    Arccos,
    Arctan,
    Exp,
    Ln,
    Sinh,
    Cosh,
    Tanh,
    Sqrt,
}

impl Function {
    pub(crate) const ALL: [Function; 14] = [
        Function::Sin,
        Function::Cos,
        Function::Tan,
        Function::Sec,
        Function::Csc,
        Function::Arcsin,
        Function::Arccos,
        Function::Arctan,
        Function::Exp,
        Function::Ln,
        Function::Sinh,
        Function::Cosh,
        Function::Tanh,
        Function::Sqrt,
    ];

    fn named(name: &str) -> Option<Function> {
        Self::ALL
            .into_iter()
            .find(|function| function.name() == name)
    }

    /// Its `tensorcrate::tensors::Analytic` code, which is its place in
    /// [`ALL`](Self::ALL).
    pub(crate) fn code(self) -> u16 {
        Self::ALL.iter().position(|&f| f == self).unwrap() as u16
    }

    /// The name it is called by, which is also the method implementing it.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Function::Sin => "sin",
            Function::Cos => "cos",
            Function::Tan => "tan",
            Function::Sec => "sec",
            Function::Csc => "csc",
            Function::Arcsin => "arcsin",
            Function::Arccos => "arccos",
            Function::Arctan => "arctan",
            Function::Exp => "exp",
            Function::Ln => "ln",
            Function::Sinh => "sinh",
            Function::Cosh => "cosh",
            Function::Tanh => "tanh",
            Function::Sqrt => "sqrt",
        }
    }

    /// The `tensorcrate::numbers` trait implementing it for every numeric
    /// type, which is also its `tensorcrate::tensors::Analytic` variant.
    pub(crate) fn trait_name(self) -> &'static str {
        match self {
            Function::Sin => "Sin",
            Function::Cos => "Cos",
            Function::Tan => "Tan",
            Function::Sec => "Sec",
            Function::Csc => "Csc",
            Function::Arcsin => "Arcsin",
            Function::Arccos => "Arccos",
            Function::Arctan => "Arctan",
            Function::Exp => "Exp",
            Function::Ln => "Ln",
            Function::Sinh => "Sinh",
            Function::Cosh => "Cosh",
            Function::Tanh => "Tanh",
            Function::Sqrt => "Sqrt",
        }
    }
}

/// A whole block: its `let` bindings in order, then the result.
pub(crate) struct Block<T = ()> {
    pub(crate) bindings: Vec<Binding<T>>,
    pub(crate) result: Expr<T>,
}

pub(crate) struct Binding<T = ()> {
    pub(crate) name: Ident,
    pub(crate) value: Expr<T>,
}
