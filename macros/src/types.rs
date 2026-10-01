//! The static types of `math!` expressions, and the pass that checks a block
//! and annotates every node with its type.
//!
//! Every shape and type error is reported here, so generation can rely on
//! what it is given.

use std::collections::HashMap;

use proc_macro2::{Span, TokenStream};
use quote::quote;

use crate::ast::{Arith, Binding, Block, Builtin, Expr, ExprKind, LitKind};

/// The shape of a value. Tensor dimensions are known at expansion time.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub(crate) enum Shape {
    #[default]
    Scalar,
    Vector(usize),
    Matrix(usize, usize),
}

/// The static type of a `math!` expression.
///
/// The extensions are independent axes rather than rival choices: a value may be
/// complex, dual, or both (a dual number whose coefficients are complex) and
/// any of those may be the element type of a tensor. Combining two types is just
/// the union of their axes.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub(crate) struct Ty {
    /// Coefficients are complex rather than real.
    pub(crate) complex: bool,
    /// Carries an infinitesimal `ε` part.
    pub(crate) dual: bool,
    pub(crate) shape: Shape,
}

/// Where tensor literals and operations emitted by a block live.
///
/// Host computes in `f64` unless a `dtype = f32;` directive asks for `f32`
/// (`HostF32`). Metal shaders are `f32`, so selecting Metal also selects `f32`
/// coefficients.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub(crate) enum BackendChoice {
    #[default]
    Host,
    HostF32,
    Metal,
}

impl BackendChoice {
    pub(crate) fn is_metal(self) -> bool {
        self == BackendChoice::Metal
    }

    /// Whether coefficients are `f32` rather than `f64`.
    fn is_single(self) -> bool {
        self != BackendChoice::Host
    }

    pub(crate) fn coefficient_type(self) -> TokenStream {
        if self.is_single() {
            quote!(f32)
        } else {
            quote!(f64)
        }
    }

    fn tensor_backend_type(self) -> TokenStream {
        if self.is_metal() {
            quote!(::tensorcrate::tensors::Metal)
        } else {
            quote!(::tensorcrate::tensors::Host)
        }
    }
}

impl Ty {
    pub(crate) const REAL: Ty = Ty {
        complex: false,
        dual: false,
        shape: Shape::Scalar,
    };
    const COMPLEX: Ty = Ty {
        complex: true,
        dual: false,
        shape: Shape::Scalar,
    };
    const DUAL: Ty = Ty {
        complex: false,
        dual: true,
        shape: Shape::Scalar,
    };

    /// This type's element type.
    pub(crate) fn element(self) -> Ty {
        Ty {
            shape: Shape::Scalar,
            ..self
        }
    }

    /// A tensor whose elements are this type.
    pub(crate) fn with_shape(self, shape: Shape) -> Ty {
        Ty { shape, ..self }
    }

    /// The Rust type of a single element.
    pub(crate) fn element_type(self, backend: BackendChoice) -> TokenStream {
        let real = backend.coefficient_type();
        let coefficient = if self.complex {
            quote!(::tensorcrate::numbers::Complex<#real>)
        } else {
            real
        };
        if self.dual {
            quote!(::tensorcrate::numbers::Dual<#coefficient>)
        } else {
            coefficient
        }
    }

    /// The Rust type this lowers to. Tensor dimensions are runtime values, so
    /// they are not part of it, though the macro knows them.
    pub(crate) fn rust_type(self, backend: BackendChoice) -> TokenStream {
        let element = self.element_type(backend);
        let storage = backend.tensor_backend_type();
        match self.shape {
            Shape::Scalar => element,
            Shape::Vector(_) => quote!(::tensorcrate::tensors::Vector<#element, #storage>),
            Shape::Matrix(_, _) => quote!(::tensorcrate::tensors::Matrix<#element, #storage>),
        }
    }

    /// The type of an expression combining `self` and `other`: the union of
    /// their axes. Every combination is meaningful, so this cannot fail.
    fn unify(self, other: Ty) -> Ty {
        Ty {
            complex: self.complex || other.complex,
            dual: self.dual || other.dual,
            shape: if self.shape == Shape::Scalar {
                other.shape
            } else {
                self.shape
            },
        }
    }

    pub(crate) fn is_tensor(self) -> bool {
        self.shape != Shape::Scalar
    }

    /// Real values: neither complex nor dual.
    pub(crate) fn is_real(self) -> bool {
        !self.complex && !self.dual
    }
}

impl LitKind {
    fn ty(self) -> Ty {
        match self {
            LitKind::Real => Ty::REAL,
            LitKind::Imaginary => Ty::COMPLEX,
            LitKind::Epsilon => Ty::DUAL,
        }
    }
}

/// The types of the bindings in scope.
type Scope = HashMap<String, Ty>;

/// Type every binding in order, then the result.
pub(crate) fn check_block(block: &Block, backend: BackendChoice) -> syn::Result<Block<Ty>> {
    let mut scope = Scope::new();
    let mut bindings = Vec::with_capacity(block.bindings.len());
    for binding in &block.bindings {
        let value = check_value(&binding.value, &scope, backend)?;
        scope.insert(binding.name.to_string(), value.ty);
        bindings.push(Binding {
            name: binding.name.clone(),
            value,
        });
    }
    let result = check_value(&block.result, &scope, backend)?;
    Ok(Block { bindings, result })
}

/// Type a binding's value or the result, which must be one the backend holds.
fn check_value(expr: &Expr, scope: &Scope, backend: BackendChoice) -> syn::Result<Expr<Ty>> {
    let typed = check(expr, scope)?;
    if backend.is_metal() && !typed.ty.is_real() {
        return Err(syn::Error::new(
            expr.span,
            "the Metal backend supports real f32 values only (no `i` or `d` literals)",
        ));
    }
    Ok(typed)
}

fn check(expr: &Expr, scope: &Scope) -> syn::Result<Expr<Ty>> {
    let span = expr.span;
    let checked = |expr: &Expr| check(expr, scope).map(Box::new);
    let (kind, ty) = match &expr.kind {
        ExprKind::Literal(literal) => (ExprKind::Literal(*literal), literal.kind.ty()),
        ExprKind::Tensor { rows, matrix } => {
            let rows = rows
                .iter()
                .map(|row| row.iter().map(|e| check(e, scope)).collect())
                .collect::<syn::Result<Vec<Vec<_>>>>()?;
            let element = rows
                .iter()
                .flatten()
                .fold(Ty::REAL, |element, e| element.unify(e.ty));
            if element.is_tensor() {
                return Err(syn::Error::new(span, "tensor elements must be scalars"));
            }
            let shape = if *matrix {
                Shape::Matrix(rows.len(), rows[0].len())
            } else {
                Shape::Vector(rows[0].len())
            };
            let matrix = *matrix;
            (ExprKind::Tensor { rows, matrix }, element.with_shape(shape))
        }
        // A name the block did not bind is an ordinary Rust `f64`.
        ExprKind::Var(ident) => (
            ExprKind::Var(ident.clone()),
            scope.get(&ident.to_string()).copied().unwrap_or(Ty::REAL),
        ),
        ExprKind::Group {
            inner,
            parenthesized,
        } => {
            let inner = checked(inner)?;
            let ty = inner.ty;
            let parenthesized = *parenthesized;
            (
                ExprKind::Group {
                    inner,
                    parenthesized,
                },
                ty,
            )
        }
        ExprKind::Neg(inner) => {
            let inner = checked(inner)?;
            let ty = inner.ty;
            (ExprKind::Neg(inner), ty)
        }
        ExprKind::Binary { op, left, right } => {
            let (left, right) = (checked(left)?, checked(right)?);
            let ty = match (op, left.ty.shape, right.ty.shape) {
                (Arith::Mul, Shape::Vector(left_len), Shape::Vector(right_len)) => {
                    if left_len != right_len {
                        return Err(syn::Error::new(
                            span,
                            "vector dot-product operands must have the same length",
                        ));
                    }
                    left.ty.unify(right.ty).element()
                }
                _ => elementwise(left.ty, right.ty, span)?,
            };
            let op = *op;
            (ExprKind::Binary { op, left, right }, ty)
        }
        ExprKind::MatMul(left, right) => {
            let (left, right) = (checked(left)?, checked(right)?);
            let ty = matmul(left.ty, right.ty).ok_or_else(|| {
                syn::Error::new(span, "`@` operands have incompatible matrix/vector shapes")
            })?;
            (ExprKind::MatMul(left, right), ty)
        }
        ExprKind::ElementwiseMul(left, right) => {
            let (left, right) = (checked(left)?, checked(right)?);
            let ty = elementwise(left.ty, right.ty, span)?;
            (ExprKind::ElementwiseMul(left, right), ty)
        }
        ExprKind::Call {
            builtin,
            args,
            name_span,
        } => {
            let args = args
                .iter()
                .map(|arg| check(arg, scope))
                .collect::<syn::Result<Vec<_>>>()?;
            let ty = call(*builtin, &args, span)?;
            (
                ExprKind::Call {
                    builtin: *builtin,
                    args,
                    name_span: *name_span,
                },
                ty,
            )
        }
    };
    Ok(Expr { kind, span, ty })
}

/// Two operands combined elementwise: tensors must agree in shape, and a
/// scalar broadcasts.
fn elementwise(left: Ty, right: Ty, span: Span) -> syn::Result<Ty> {
    if left.is_tensor() && right.is_tensor() && left.shape != right.shape {
        return Err(syn::Error::new(
            span,
            "elementwise tensor operands must have the same shape",
        ));
    }
    Ok(left.unify(right))
}

/// The type of a matrix/matrix, matrix/vector or vector/matrix product, or
/// `None` when the shapes do not meet.
fn matmul(left: Ty, right: Ty) -> Option<Ty> {
    let shape = match (left.shape, right.shape) {
        (Shape::Matrix(rows, inner), Shape::Matrix(inner2, columns)) if inner == inner2 => {
            Shape::Matrix(rows, columns)
        }
        (Shape::Matrix(rows, inner), Shape::Vector(len)) if inner == len => Shape::Vector(rows),
        (Shape::Vector(len), Shape::Matrix(rows, columns)) if len == rows => Shape::Vector(columns),
        _ => return None,
    };
    Some(left.unify(right).with_shape(shape))
}

/// The type of a call to `builtin` with these (already typed) arguments.
fn call(builtin: Builtin, args: &[Expr<Ty>], span: Span) -> syn::Result<Ty> {
    let name = builtin.name();
    let error = |message: String| syn::Error::new(span, message);
    let require_real = |i: usize| -> syn::Result<()> {
        if args[i].ty.is_real() {
            Ok(())
        } else {
            Err(syn::Error::new(
                args[i].span,
                format!("`{name}` requires real values"),
            ))
        }
    };
    let is_vector = |ty: Ty| matches!(ty.shape, Shape::Vector(_));
    match builtin {
        Builtin::Min | Builtin::Max => {
            let (a, b) = (args[0].ty, args[1].ty);
            require_real(0)?;
            require_real(1)?;
            if a.is_tensor() && b.is_tensor() && a.shape != b.shape {
                return Err(error(format!(
                    "`{name}` tensor operands must have the same shape"
                )));
            }
            Ok(a.unify(b))
        }
        Builtin::Clamp => {
            for i in 0..3 {
                require_real(i)?;
            }
            if args[1].ty.is_tensor() || args[2].ty.is_tensor() {
                return Err(error("`clamp` bounds must be scalars".to_string()));
            }
            Ok(args[0].ty)
        }
        Builtin::Sum | Builtin::Minimum | Builtin::Maximum => {
            let value = args[0].ty;
            if !is_vector(value) {
                return Err(error(format!("`{name}` expects a vector")));
            }
            if builtin != Builtin::Sum {
                require_real(0)?;
            }
            Ok(value.element())
        }
        Builtin::PrefixSum => {
            let value = args[0].ty;
            if !is_vector(value) {
                return Err(error("`prefix_sum` expects a vector".to_string()));
            }
            Ok(value)
        }
        Builtin::Sorted { .. } => {
            let value = args[0].ty;
            require_real(0)?;
            if !is_vector(value) {
                return Err(error("`sorted` expects a vector".to_string()));
            }
            Ok(value)
        }
        Builtin::Pow => {
            let (a, b) = (args[0].ty, args[1].ty);
            if a.is_tensor() && b.is_tensor() && a.shape != b.shape {
                return Err(error("`pow` expects tensors of the same shape".to_string()));
            }
            Ok(a.unify(b))
        }
        Builtin::Dot => {
            let (a, b) = (args[0].ty, args[1].ty);
            match (a.shape, b.shape) {
                (Shape::Vector(n), Shape::Vector(m)) if n == m => Ok(a.unify(b).element()),
                _ => Err(error(
                    "`dot` expects vectors of the same length".to_string(),
                )),
            }
        }
        Builtin::MatMul => matmul(args[0].ty, args[1].ty).ok_or_else(|| {
            error("`matmul` operands have incompatible matrix/vector shapes".to_string())
        }),
        Builtin::Det | Builtin::Inv => {
            let a = args[0].ty;
            match a.shape {
                Shape::Matrix(r, c) if r == c => Ok(if builtin == Builtin::Det {
                    a.element()
                } else {
                    a
                }),
                _ => Err(error(format!("`{name}` expects a square matrix"))),
            }
        }
        Builtin::Transpose => {
            let a = args[0].ty;
            match a.shape {
                Shape::Matrix(r, c) => Ok(a.with_shape(Shape::Matrix(c, r))),
                _ => Err(error("`transpose` expects a matrix".to_string())),
            }
        }
        // `conj` is only meaningful on complex numbers.
        Builtin::Conj => Ok(Ty::COMPLEX.unify(args[0].ty)),
        Builtin::Analytic(_) => Ok(args[0].ty),
    }
}
