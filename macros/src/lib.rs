//! The `math!` macro: a small mathematical language that expands to ordinary,
//! statically-typed Rust.
//!
//! ```ignore
//! let z = math! {
//!     let x = 1 + 2i;
//!     let y = 1 - 2i;
//!     x * y            // Complex<f64>: 5 + 0i
//! };
//! ```
//!
//! There is no interpreter and no dynamically-typed value: the macro works out
//! each binding's type at expansion time and emits concrete Rust (`f64`,
//! `Complex<f64>`, `Dual<f64>`, `Vector<_, N>`, `Matrix<_, R, C>`), so the Rust
//! compiler type-checks the result and the optimizer sees straight-line
//! arithmetic.
//!
//! Tensor products are selected symbolically from their inferred shapes:
//! `A @ B` is matrix multiplication (including matrix/vector and vector/matrix
//! products), while `v * u` is a vector dot product. Analytic functions such as
//! `sin(A)` and `cos(v)` map elementwise over matrices and vectors.
//!
//! The input is parsed with `syn` as Rust syntax, which gives operator
//! precedence, parentheses, grouping and array literals for free. The only
//! extensions are literal suffixes: `2i` is imaginary and `2d` is the dual
//! infinitesimal ε.
//!
//! (ε is spelled `d`, not `e`: rustc's lexer treats any `e` after digits as the
//! start of a float exponent, so `2e`/`2eps` cannot reach a macro at all.)

use std::collections::HashMap;

use proc_macro2::{Group, Ident, Literal, Punct, Spacing, Span, TokenStream, TokenTree};
use quote::quote;
use syn::parse::Parser;
use syn::spanned::Spanned;
use syn::{BinOp, Block, Expr, Lit, Pat, Stmt, UnOp};

/// The static type of a `math!` expression.
///
/// The extensions are independent axes rather than rival choices: a value may be
/// complex, dual, or both — a dual number whose coefficients are complex — and
/// any of those may be the element type of a tensor. Combining two types is just
/// the union of their axes.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
enum Shape {
    #[default]
    Scalar,
    Vector(usize),
    Matrix(usize, usize),
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
struct Ty {
    /// Coefficients are complex rather than real.
    complex: bool,
    /// Carries an infinitesimal `ε` part.
    dual: bool,
    shape: Shape,
}

impl Ty {
    const REAL: Ty = Ty {
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

    /// This type's element type — itself, with the tensor axis dropped.
    fn element(self) -> Ty {
        Ty {
            shape: Shape::Scalar,
            ..self
        }
    }

    /// A tensor whose elements are this type.
    fn with_shape(self, shape: Shape) -> Ty {
        Ty { shape, ..self }
    }

    /// The Rust type of a single element.
    fn element_type(self) -> TokenStream {
        let coefficient = if self.complex {
            quote!(::tensorcrate::numbers::Complex<f64>)
        } else {
            quote!(f64)
        };
        if self.dual {
            quote!(::tensorcrate::numbers::Dual<#coefficient>)
        } else {
            coefficient
        }
    }

    /// The Rust type this lowers to.
    fn rust_type(self) -> TokenStream {
        let element = self.element_type();
        match self.shape {
            Shape::Scalar => element,
            Shape::Vector(n) => quote!(::tensorcrate::tensors::Vector<#element, #n>),
            Shape::Matrix(r, c) => quote!(::tensorcrate::tensors::Matrix<#element, #r, #c>),
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

    fn is_tensor(self) -> bool {
        self.shape != Shape::Scalar
    }
}

/// Bindings introduced by `let` in the block, with their inferred types.
type Env = HashMap<String, Ty>;

#[proc_macro]
pub fn math(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    match expand(rewrite_matmul_operator(input.into())) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// `@` is not part of Rust's expression grammar. Rewrite it to a reserved
/// `lhs / marker / rhs` form before `syn` parses the block. Division has the
/// same precedence and associativity that matrix multiplication should have,
/// and [`matmul_operands`] recognizes the resulting AST without confusing an
/// ordinary `/`.
fn rewrite_matmul_operator(input: TokenStream) -> TokenStream {
    let mut output = TokenStream::new();
    for token in input {
        match token {
            TokenTree::Punct(at) if at.as_char() == '@' => {
                let span = at.span();
                let mut slash = Punct::new('/', Spacing::Alone);
                slash.set_span(span);
                output.extend([TokenTree::Punct(slash.clone())]);
                output.extend([TokenTree::Ident(Ident::new(
                    "__tensorcrate_matmul_operator__",
                    span,
                ))]);
                output.extend([TokenTree::Punct(slash)]);
            }
            TokenTree::Group(group) => {
                let mut rewritten =
                    Group::new(group.delimiter(), rewrite_matmul_operator(group.stream()));
                rewritten.set_span(group.span());
                output.extend([TokenTree::Group(rewritten)]);
            }
            token => output.extend([token]),
        }
    }
    output
}

fn is_matmul_marker(expr: &Expr) -> bool {
    matches!(expr, Expr::Path(path) if path.path.is_ident("__tensorcrate_matmul_operator__"))
}

/// Recover the operands from the reserved `(lhs / marker) / rhs` AST shape.
fn matmul_operands(expr: &Expr) -> Option<(&Expr, &Expr)> {
    let Expr::Binary(outer) = expr else {
        return None;
    };
    matmul_binary_operands(outer)
}

fn matmul_binary_operands(outer: &syn::ExprBinary) -> Option<(&Expr, &Expr)> {
    if !matches!(outer.op, BinOp::Div(_)) {
        return None;
    }
    let Expr::Binary(partial) = &*outer.left else {
        return None;
    };
    if matches!(partial.op, BinOp::Div(_)) && is_matmul_marker(&partial.right) {
        Some((&partial.left, &outer.right))
    } else {
        None
    }
}

fn expand(input: TokenStream) -> syn::Result<TokenStream> {
    let stmts = Block::parse_within.parse2(input)?;
    let Some((last, leading)) = stmts.split_last() else {
        return Err(syn::Error::new(
            Span::call_site(),
            "math! needs at least a final expression",
        ));
    };
    let Stmt::Expr(result, None) = last else {
        return Err(syn::Error::new(
            last.span(),
            "math! must end with an expression (no trailing semicolon)",
        ));
    };

    let fallible = block_mentions_inverse(leading, result);

    let mut env = Env::new();
    let mut out = Vec::new();
    for stmt in leading {
        out.push(lower_let(stmt, &mut env)?);
    }

    let ty = infer(result, &env)?;
    let body = lower(result, ty, &env)?;
    let annotated = ty.rust_type();

    Ok(if fallible {
        quote! {
            (|| -> ::core::result::Result<#annotated, ::tensorcrate::errors::Error> {
                #(#out)*
                let __math_result: #annotated = #body;
                ::core::result::Result::Ok(__math_result)
            })()
        }
    } else {
        quote! {
            {
                #(#out)*
                let __math_result: #annotated = #body;
                __math_result
            }
        }
    })
}

fn block_mentions_inverse(leading: &[Stmt], result: &Expr) -> bool {
    leading.iter().any(|stmt| match stmt {
        Stmt::Local(local) => local
            .init
            .as_ref()
            .is_some_and(|i| mentions_inverse(&i.expr)),
        _ => false,
    }) || mentions_inverse(result)
}

fn mentions_inverse(expr: &Expr) -> bool {
    match expr {
        Expr::Paren(p) => mentions_inverse(&p.expr),
        Expr::Group(g) => mentions_inverse(&g.expr),
        Expr::Unary(u) => mentions_inverse(&u.expr),
        Expr::Binary(b) => mentions_inverse(&b.left) || mentions_inverse(&b.right),
        Expr::Call(c) => {
            matches!(&*c.func, Expr::Path(p) if p.path.is_ident("inv"))
                || c.args.iter().any(mentions_inverse)
        }
        Expr::Array(a) => a.elems.iter().any(mentions_inverse),
        _ => false,
    }
}

/// Lower a `let` statement, recording the binding's inferred type.
fn lower_let(stmt: &Stmt, env: &mut Env) -> syn::Result<TokenStream> {
    let Stmt::Local(local) = stmt else {
        return Err(syn::Error::new(
            stmt.span(),
            "math! only allows `let` bindings before the final expression",
        ));
    };
    let Pat::Ident(pat) = &local.pat else {
        return Err(syn::Error::new(
            local.pat.span(),
            "expected a variable name",
        ));
    };
    let Some(init) = &local.init else {
        return Err(syn::Error::new(local.span(), "`let` needs an initializer"));
    };
    if init.diverge.is_some() {
        return Err(syn::Error::new(
            local.span(),
            "`let ... else` is not supported",
        ));
    }

    let ty = infer(&init.expr, env)?;
    let value = lower(&init.expr, ty, env)?;
    let name = &pat.ident;
    let annotated = ty.rust_type();
    env.insert(name.to_string(), ty);
    Ok(quote! { let #name: #annotated = #value; })
}

// ---- tensor literals --------------------------------------------------------

/// The rows of a tensor literal: one row for a vector, several for a matrix.
struct TensorLit<'a> {
    rows: Vec<Vec<&'a Expr>>,
    /// A vector is stored as a single row but keeps rank 1.
    is_matrix: bool,
}

impl<'a> TensorLit<'a> {
    fn elements(&self) -> impl Iterator<Item = &&'a Expr> {
        self.rows.iter().flatten()
    }
}

/// Read an array literal as a vector (`[1, 2]`) or a matrix (`[[1, 2], [3, 4]]`),
/// checking at expansion time that matrix rows all have the same length.
fn tensor_literal(arr: &syn::ExprArray) -> syn::Result<TensorLit<'_>> {
    if arr.elems.is_empty() {
        return Err(syn::Error::new(
            arr.span(),
            "an empty tensor literal has no element type",
        ));
    }
    let nested = arr
        .elems
        .iter()
        .filter(|e| matches!(e, Expr::Array(_)))
        .count();
    if nested == 0 {
        return Ok(TensorLit {
            rows: vec![arr.elems.iter().collect()],
            is_matrix: false,
        });
    }
    if nested != arr.elems.len() {
        return Err(syn::Error::new(
            arr.span(),
            "a tensor literal must be all scalars (a vector) or all rows (a matrix)",
        ));
    }

    let mut rows = Vec::new();
    for row in &arr.elems {
        let Expr::Array(row) = row else {
            unreachable!()
        };
        if row.elems.iter().any(|e| matches!(e, Expr::Array(_))) {
            return Err(syn::Error::new(
                row.span(),
                "tensor literals go at most two deep (vector or matrix)",
            ));
        }
        rows.push(row.elems.iter().collect::<Vec<_>>());
    }
    let width = rows[0].len();
    if let Some(bad) = rows.iter().find(|r| r.len() != width) {
        return Err(syn::Error::new(
            bad.first().map_or_else(|| arr.span(), |e| e.span()),
            format!("every matrix row must have {width} elements"),
        ));
    }
    Ok(TensorLit {
        rows,
        is_matrix: true,
    })
}

// ---- type inference ---------------------------------------------------------

fn infer_matmul(left: Ty, right: Ty, span: Span) -> syn::Result<Ty> {
    let shape = match (left.shape, right.shape) {
        (Shape::Matrix(rows, inner), Shape::Matrix(inner2, columns)) if inner == inner2 => {
            Shape::Matrix(rows, columns)
        }
        (Shape::Matrix(rows, inner), Shape::Vector(len)) if inner == len => Shape::Vector(rows),
        (Shape::Vector(len), Shape::Matrix(rows, columns)) if len == rows => Shape::Vector(columns),
        _ => {
            return Err(syn::Error::new(
                span,
                "`@` operands have incompatible matrix/vector shapes",
            ));
        }
    };
    Ok(left.unify(right).with_shape(shape))
}

/// Work out an expression's type without generating code.
fn infer(expr: &Expr, env: &Env) -> syn::Result<Ty> {
    match expr {
        Expr::Lit(lit) => Ok(classify_literal(&lit.lit)?.0.ty()),
        Expr::Array(arr) => {
            let literal = tensor_literal(arr)?;
            let mut element = Ty::REAL;
            for e in literal.elements() {
                element = element.unify(infer(e, env)?);
            }
            if element.is_tensor() {
                return Err(syn::Error::new(
                    arr.span(),
                    "tensor elements must be scalars",
                ));
            }
            let shape = if literal.is_matrix {
                Shape::Matrix(literal.rows.len(), literal.rows[0].len())
            } else {
                Shape::Vector(literal.rows[0].len())
            };
            Ok(element.with_shape(shape))
        }
        Expr::Path(p) => {
            let name = path_name(p)?;
            // An identifier the block did not bind is an ordinary Rust `f64`.
            Ok(env.get(&name).copied().unwrap_or(Ty::REAL))
        }
        Expr::Paren(p) => infer(&p.expr, env),
        Expr::Group(g) => infer(&g.expr, env),
        Expr::Unary(u) => match u.op {
            UnOp::Neg(_) => infer(&u.expr, env),
            _ => Err(syn::Error::new(expr.span(), "unsupported unary operator")),
        },
        Expr::Binary(b) => {
            if let Some((lhs, rhs)) = matmul_operands(expr) {
                return infer_matmul(infer(lhs, env)?, infer(rhs, env)?, b.span());
            }
            let left = infer(&b.left, env)?;
            let right = infer(&b.right, env)?;
            if matches!(b.op, BinOp::Mul(_))
                && let (Shape::Vector(left_len), Shape::Vector(right_len)) =
                    (left.shape, right.shape)
            {
                if left_len != right_len {
                    return Err(syn::Error::new(
                        b.span(),
                        "vector dot-product operands must have the same length",
                    ));
                }
                return Ok(left.unify(right).element());
            }
            if left.is_tensor() && right.is_tensor() && left.shape != right.shape {
                return Err(syn::Error::new(
                    b.span(),
                    "elementwise tensor operands must have the same shape",
                ));
            }
            Ok(left.unify(right))
        }
        Expr::Call(call) => {
            let name = call_name(call)?;
            let args: Vec<&Expr> = call.args.iter().collect();
            let arg_ty = |i: usize| infer(args[i], env);

            match (name.as_str(), args.len()) {
                ("pow", 2) => {
                    let (a, b) = (arg_ty(0)?, arg_ty(1)?);
                    if a.is_tensor() && b.is_tensor() {
                        return Err(syn::Error::new(
                            call.span(),
                            "`pow` accepts at most one tensor argument",
                        ));
                    }
                    Ok(a.unify(b))
                }
                ("dot", 2) => {
                    let (a, b) = (arg_ty(0)?, arg_ty(1)?);
                    match (a.shape, b.shape) {
                        (Shape::Vector(n), Shape::Vector(m)) if n == m => Ok(a.unify(b).element()),
                        _ => Err(syn::Error::new(
                            call.span(),
                            "`dot` expects vectors of the same length",
                        )),
                    }
                }
                ("matmul", 2) => {
                    let (a, b) = (arg_ty(0)?, arg_ty(1)?);
                    infer_matmul(a, b, call.span()).map_err(|_| {
                        syn::Error::new(
                            call.span(),
                            "`matmul` operands have incompatible matrix/vector shapes",
                        )
                    })
                }
                ("det" | "inv", 1) => {
                    let a = arg_ty(0)?;
                    match a.shape {
                        Shape::Matrix(r, c) if r == c => {
                            Ok(if name == "det" { a.element() } else { a })
                        }
                        _ => Err(syn::Error::new(
                            call.span(),
                            format!("`{name}` expects a square matrix"),
                        )),
                    }
                }
                ("transpose", 1) => {
                    let a = arg_ty(0)?;
                    match a.shape {
                        Shape::Matrix(r, c) => Ok(a.with_shape(Shape::Matrix(c, r))),
                        _ => Err(syn::Error::new(call.span(), "`transpose` expects a matrix")),
                    }
                }
                // `conj` is only meaningful on complex numbers.
                ("conj", 1) => Ok(Ty::COMPLEX.unify(arg_ty(0)?)),
                (_, 1) if ANALYTIC.iter().any(|(n, _)| *n == name) => arg_ty(0),
                (_, n) => Err(syn::Error::new(
                    call.span(),
                    format!("`{name}` does not take {n} argument(s) in math!"),
                )),
            }
        }
        other => Err(syn::Error::new(
            other.span(),
            "unsupported expression in math!",
        )),
    }
}

/// Which axis a literal's suffix introduces.
#[derive(Copy, Clone, PartialEq, Eq)]
enum LitKind {
    Real,
    Imaginary,
    Epsilon,
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

/// Classify a numeric literal by its suffix, returning its kind and digits.
fn classify_literal(lit: &Lit) -> syn::Result<(LitKind, String)> {
    let (suffix, digits) = match lit {
        Lit::Int(v) => (v.suffix().to_string(), v.base10_digits().to_string()),
        Lit::Float(v) => (v.suffix().to_string(), v.base10_digits().to_string()),
        other => {
            return Err(syn::Error::new(
                other.span(),
                "math! expects a numeric literal",
            ));
        }
    };
    match suffix.as_str() {
        "" => Ok((LitKind::Real, digits)),
        "i" => Ok((LitKind::Imaginary, digits)),
        "d" => Ok((LitKind::Epsilon, digits)),
        other => Err(syn::Error::new(
            lit.span(),
            format!("unknown literal suffix `{other}` (use `i` for imaginary, `d` for ε)"),
        )),
    }
}

// ---- code generation --------------------------------------------------------

/// The functions callable in `math!`, mapped to the trait and method that
/// implement them for every supported numeric type.
const ANALYTIC: &[(&str, &str)] = &[
    ("sin", "Sin"),
    ("cos", "Cos"),
    ("tan", "Tan"),
    ("sec", "Sec"),
    ("csc", "Csc"),
    ("arcsin", "Arcsin"),
    ("arccos", "Arccos"),
    ("arctan", "Arctan"),
    ("exp", "Exp"),
    ("ln", "Ln"),
    ("sinh", "Sinh"),
    ("cosh", "Cosh"),
    ("tanh", "Tanh"),
];

/// Emit `expr` as Rust of type `target`, widening subexpressions as needed.
fn lower(expr: &Expr, target: Ty, env: &Env) -> syn::Result<TokenStream> {
    match expr {
        Expr::Lit(lit) => lower_literal(lit, target),
        Expr::Array(arr) => {
            let literal = tensor_literal(arr)?;
            let element = target.element();
            let rows = literal
                .rows
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|e| lower(e, element, env))
                        .collect::<syn::Result<Vec<_>>>()
                })
                .collect::<syn::Result<Vec<_>>>()?;
            Ok(if literal.is_matrix {
                quote!(::tensorcrate::tensors::Matrix::from_rows([#([#(#rows),*]),*]))
            } else {
                let values = &rows[0];
                quote!(::tensorcrate::tensors::Vector::new([#(#values),*]))
            })
        }
        Expr::Path(p) => {
            let name = path_name(p)?;
            let ident = &p.path.segments[0].ident;
            let ty = env.get(&name).copied().unwrap_or(Ty::REAL);
            Ok(widen(quote!(#ident), ty, target))
        }
        Expr::Paren(p) => {
            let inner = lower(&p.expr, target, env)?;
            Ok(quote!((#inner)))
        }
        Expr::Group(g) => lower(&g.expr, target, env),
        Expr::Unary(u) => match u.op {
            UnOp::Neg(_) => {
                let inner = lower(&u.expr, target, env)?;
                Ok(quote!(-(#inner)))
            }
            _ => Err(syn::Error::new(expr.span(), "unsupported unary operator")),
        },
        Expr::Binary(b) => lower_binary(b, target, env),
        Expr::Call(call) => lower_call(call, target, env),
        other => Err(syn::Error::new(
            other.span(),
            "unsupported expression in math!",
        )),
    }
}

fn lower_literal(lit: &syn::ExprLit, target: Ty) -> syn::Result<TokenStream> {
    let (kind, digits) = classify_literal(&lit.lit)?;
    let value: f64 = digits
        .parse()
        .map_err(|_| syn::Error::new(lit.span(), "invalid numeric literal"))?;
    let value = Literal::f64_suffixed(value);

    // Build the coefficient first: `2i` is `0 + 2i`, anything else is purely
    // real.
    let coefficient = if target.complex {
        if kind == LitKind::Imaginary {
            quote!(::tensorcrate::numbers::Complex::imaginary(#value))
        } else {
            quote!(::tensorcrate::numbers::Complex::constant(#value))
        }
    } else {
        quote!(#value)
    };

    // Then place it on the real or the ε side of the dual.
    Ok(if target.dual {
        if kind == LitKind::Epsilon {
            let zero = zero_of(target.complex);
            quote!(::tensorcrate::numbers::Dual::new(#zero, #coefficient))
        } else {
            quote!(::tensorcrate::numbers::Dual::constant(#coefficient))
        }
    } else {
        coefficient
    })
}

fn lower_matmul(left: &Expr, right: &Expr, target: Ty, env: &Env) -> syn::Result<TokenStream> {
    let (left_ty, right_ty) = (infer(left, env)?, infer(right, env)?);
    let left = lower(left, target.element().with_shape(left_ty.shape), env)?;
    let right = lower(right, target.element().with_shape(right_ty.shape), env)?;
    Ok(match (left_ty.shape, right_ty.shape) {
        (Shape::Matrix(_, _), Shape::Matrix(_, _)) => quote!((#left).matmul(&(#right))),
        (Shape::Matrix(_, _), Shape::Vector(_)) => quote!((#left).matvec(&(#right))),
        (Shape::Vector(_), Shape::Matrix(_, _)) => quote!((#left).vecmat(&(#right))),
        _ => unreachable!("infer_matmul validated the operand shapes"),
    })
}

fn lower_binary(b: &syn::ExprBinary, target: Ty, env: &Env) -> syn::Result<TokenStream> {
    if let Some((left, right)) = matmul_binary_operands(b) {
        return lower_matmul(left, right, target, env);
    }

    let (left_ty, right_ty) = (infer(&b.left, env)?, infer(&b.right, env)?);
    if matches!(b.op, BinOp::Mul(_))
        && matches!(
            (left_ty.shape, right_ty.shape),
            (Shape::Vector(_), Shape::Vector(_))
        )
    {
        let vector_target = target.with_shape(left_ty.shape);
        let left = lower(&b.left, vector_target, env)?;
        let right = lower(&b.right, vector_target, env)?;
        return Ok(quote!((#left).dot(&(#right))));
    }

    let op = match b.op {
        BinOp::Add(_) => quote!(+),
        BinOp::Sub(_) => quote!(-),
        BinOp::Mul(_) => quote!(*),
        BinOp::Div(_) => quote!(/),
        BinOp::Rem(_) => quote!(%),
        BinOp::BitXor(_) => {
            return Err(syn::Error::new(
                b.op.span(),
                "`^` is not exponentiation here: Rust parses it looser than `*` and `+`, \
                 which would silently misread `a * b ^ 2`. Use `pow(base, exponent)`.",
            ));
        }
        other => {
            return Err(syn::Error::new(
                other.span(),
                "unsupported operator in math! (use + - * / % or pow(x, y))",
            ));
        }
    };
    let broadcast_op = match b.op {
        BinOp::Add(_) => quote!(::tensorcrate::tensors::BinaryOp::Add),
        BinOp::Sub(_) => quote!(::tensorcrate::tensors::BinaryOp::Sub),
        BinOp::Mul(_) => quote!(::tensorcrate::tensors::BinaryOp::Mul),
        BinOp::Div(_) => quote!(::tensorcrate::tensors::BinaryOp::Div),
        BinOp::Rem(_) => quote!(::tensorcrate::tensors::BinaryOp::Rem),
        _ => unreachable!("unsupported operators returned above"),
    };

    if !target.is_tensor() {
        let left = lower(&b.left, target, env)?;
        let right = lower(&b.right, target, env)?;
        return Ok(quote!((#left #op #right)));
    }

    // At least one side is a tensor. Two tensors combine elementwise (which can
    // fail on a shape mismatch); a tensor and a scalar broadcast the scalar.
    let element = target.element();
    Ok(match (left_ty.is_tensor(), right_ty.is_tensor()) {
        (true, true) => {
            let left = lower(&b.left, target, env)?;
            let right = lower(&b.right, target, env)?;
            quote!((#left #op #right))
        }
        (true, false) => {
            let left = lower(&b.left, target, env)?;
            let right = lower(&b.right, element, env)?;
            quote!((#left).broadcast_right(#right, #broadcast_op))
        }
        (false, true) => {
            let left = lower(&b.left, element, env)?;
            let right = lower(&b.right, target, env)?;
            quote!((#right).broadcast_left(#left, #broadcast_op))
        }
        (false, false) => unreachable!("target is a tensor only if an operand is"),
    })
}

fn lower_call(call: &syn::ExprCall, target: Ty, env: &Env) -> syn::Result<TokenStream> {
    let name = call_name(call)?;
    let args: Vec<&Expr> = call.args.iter().collect();

    // Linear algebra takes its arguments as tensors and is fallible.
    let tensor_arg = |i: usize, env: &Env| -> syn::Result<TokenStream> {
        let ty = infer(args[i], env)?;
        if !ty.is_tensor() {
            return Err(syn::Error::new(
                args[i].span(),
                format!("`{name}` expects a tensor argument"),
            ));
        }
        lower(args[i], target.element().with_shape(ty.shape), env)
    };

    match (name.as_str(), args.len()) {
        ("pow", 2) => {
            let base_ty = infer(args[0], env)?;
            if target.is_tensor() {
                let element = target.element();
                if base_ty.is_tensor() {
                    let base = lower(args[0], target, env)?;
                    let exponent = lower(args[1], element, env)?;
                    Ok(quote!({
                        let __exponent = #exponent;
                        (#base).map(|&__x| ::tensorcrate::numbers::Power::power(__x, __exponent))
                    }))
                } else {
                    let base = lower(args[0], element, env)?;
                    let exponent = lower(args[1], target, env)?;
                    Ok(quote!({
                        let __base = #base;
                        (#exponent).map(|&__x| ::tensorcrate::numbers::Power::power(__base, __x))
                    }))
                }
            } else {
                let base = lower(args[0], target, env)?;
                let exponent = lower(args[1], target, env)?;
                Ok(quote!(::tensorcrate::numbers::Power::power(#base, #exponent)))
            }
        }
        ("dot", 2) => {
            let (a, b) = (tensor_arg(0, env)?, tensor_arg(1, env)?);
            Ok(quote!((#a).dot(&(#b))))
        }
        ("matmul", 2) => {
            let (a, b) = (tensor_arg(0, env)?, tensor_arg(1, env)?);
            let (at, bt) = (infer(args[0], env)?, infer(args[1], env)?);
            Ok(match (at.shape, bt.shape) {
                (Shape::Matrix(_, _), Shape::Matrix(_, _)) => quote!((#a).matmul(&(#b))),
                (Shape::Matrix(_, _), Shape::Vector(_)) => quote!((#a).matvec(&(#b))),
                (Shape::Vector(_), Shape::Matrix(_, _)) => quote!((#a).vecmat(&(#b))),
                _ => unreachable!("infer validated matmul shapes"),
            })
        }
        ("transpose", 1) => {
            let a = tensor_arg(0, env)?;
            Ok(quote!((#a).transpose()))
        }
        ("inv", 1) => {
            let a = tensor_arg(0, env)?;
            Ok(quote!((#a).inverse()?))
        }
        ("det", 1) => {
            let a = tensor_arg(0, env)?;
            Ok(quote!((#a).determinant()))
        }
        ("conj", 1) => {
            let inner = lower(args[0], target, env)?;
            if target.dual {
                let conjugate_dual = |value: TokenStream| {
                    quote!({
                        let __dual = #value;
                        ::tensorcrate::numbers::Dual::new(
                            ::tensorcrate::numbers::Complex::conj(__dual.real),
                            ::tensorcrate::numbers::Complex::conj(__dual.dual),
                        )
                    })
                };
                Ok(if target.is_tensor() {
                    let conjugated = conjugate_dual(quote!(__x));
                    quote!((#inner).map(|&__x| #conjugated))
                } else {
                    conjugate_dual(inner)
                })
            } else {
                Ok(elementwise(
                    quote!(::tensorcrate::numbers::Complex::conj),
                    inner,
                    target,
                ))
            }
        }
        (_, 1) if ANALYTIC.iter().any(|(n, _)| *n == name) => {
            let (_, trait_name) = ANALYTIC.iter().find(|(n, _)| *n == name).unwrap();
            let trait_ident = syn::Ident::new(trait_name, call.func.span());
            let method = syn::Ident::new(&name, call.func.span());
            let inner = lower(args[0], target, env)?;
            Ok(elementwise(
                quote!(::tensorcrate::numbers::#trait_ident::#method),
                inner,
                target,
            ))
        }
        (_, n) => Err(syn::Error::new(
            call.span(),
            format!("`{name}` does not take {n} argument(s) in math!"),
        )),
    }
}

/// Apply a scalar function, mapping over the elements when the value is a
/// tensor.
fn elementwise(function: TokenStream, value: TokenStream, target: Ty) -> TokenStream {
    if target.is_tensor() {
        quote!((#value).map(|&__x| #function(__x)))
    } else {
        quote!(#function(#value))
    }
}

/// Zero of the coefficient type.
fn zero_of(complex: bool) -> TokenStream {
    if complex {
        quote!(::tensorcrate::numbers::Complex::constant(0f64))
    } else {
        quote!(0f64)
    }
}

/// Widen a value of type `from` so it can be used where `to` is expected.
///
/// The axes are handled in order: lift the coefficients to complex, then wrap in
/// a dual. A tensor is widened by mapping the same conversion over its elements.
/// Narrowing never happens — `infer` already unified to the wider type.
fn widen(value: TokenStream, from: Ty, to: Ty) -> TokenStream {
    if from.is_tensor() {
        // The element conversion, expressed on a bound element.
        let converted = widen_scalar(quote!(__x), from.element(), to.element());
        return if converted.to_string() == "__x" {
            value
        } else {
            quote!((#value).map(|&__x| #converted))
        };
    }
    widen_scalar(value, from, to)
}

fn widen_scalar(value: TokenStream, from: Ty, to: Ty) -> TokenStream {
    let mut out = value;
    if to.complex && !from.complex {
        out = if from.dual {
            // Dual<f64> -> Dual<Complex<f64>>: convert both coefficients.
            quote!((#out).map(::tensorcrate::numbers::Complex::constant))
        } else {
            quote!(::tensorcrate::numbers::Complex::constant(#out))
        };
    }
    if to.dual && !from.dual {
        out = quote!(::tensorcrate::numbers::Dual::constant(#out));
    }
    out
}

/// The single-segment name of a path expression.
fn path_name(p: &syn::ExprPath) -> syn::Result<String> {
    match p.path.get_ident() {
        Some(ident) => Ok(ident.to_string()),
        None => Err(syn::Error::new(
            p.span(),
            "math! expects a plain identifier",
        )),
    }
}

/// The name of a called function.
fn call_name(call: &syn::ExprCall) -> syn::Result<String> {
    match &*call.func {
        Expr::Path(p) => path_name(p),
        other => Err(syn::Error::new(other.span(), "expected a function name")),
    }
}
