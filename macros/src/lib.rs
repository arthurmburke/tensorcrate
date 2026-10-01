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
//! `Complex<f64>`, `Dual<f64>`, `Vector<_>`, `Matrix<_>`), so the Rust
//! compiler type-checks the result and the optimizer sees straight-line
//! arithmetic.
//!
//! Tensor products are selected symbolically from their inferred shapes:
//! `A @ B` is matrix multiplication (including matrix/vector and vector/matrix
//! products), while `v * u` is a vector dot product and `v .* u` is explicitly
//! elementwise. Analytic functions such as `sin(A)` and `cos(v)` map
//! elementwise over matrices and vectors.
//!
//! Tensors use the host backend by default, with `f64` coefficients. A leading
//! `backend = Metal;` directive selects resident `f32` Metal tensors instead;
//! `backend = Host;` is the explicit spelling of the default. A leading
//! `dtype = f32;` (or `dtype = f64;`) chooses the coefficient type, so a host
//! block can compute in `f32` too. The directives may come in either order;
//! Metal's shaders are `f32`, so `backend = Metal;` with `dtype = f64;` is an
//! error.
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
/// complex, dual, or both (a dual number whose coefficients are complex) and
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

/// Where tensor literals and operations emitted by a block live.
///
/// Host preserves the macro's original `f64` algebra unless a `dtype = f32;`
/// directive asks for `f32` (`HostF32`). Metal shaders are `f32`, so selecting
/// Metal also selects `f32` coefficients.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
enum BackendChoice {
    #[default]
    Host,
    HostF32,
    Metal,
}

impl BackendChoice {
    fn is_metal(self) -> bool {
        self == BackendChoice::Metal
    }

    /// Whether coefficients are `f32` rather than `f64`.
    fn is_single(self) -> bool {
        self != BackendChoice::Host
    }

    fn coefficient_type(self) -> TokenStream {
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

    /// This type's element type.
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
    fn element_type(self, backend: BackendChoice) -> TokenStream {
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

    /// The Rust type this lowers to.
    fn rust_type(self, backend: BackendChoice) -> TokenStream {
        let element = self.element_type(backend);
        let storage = backend.tensor_backend_type();
        match self.shape {
            Shape::Scalar => element,
            // The dimensions are runtime values now, so they are not part of
            // the type. The macro still *knows* them.
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

    fn is_tensor(self) -> bool {
        self.shape != Shape::Scalar
    }
}

/// Bindings introduced by `let` in the block, with their inferred types.
type Env = HashMap<String, Ty>;

#[proc_macro]
pub fn math(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    match expand(rewrite_custom_operators(input.into())) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// `@` and `.*` are not part of Rust's expression grammar. Rewrite each to a
/// reserved `lhs / marker / rhs` form before `syn` parses the block. Division
/// has the same precedence and associativity these product operators should
/// have, and [`marked_operands`] recovers the operands without confusing an
/// ordinary `/`.
fn rewrite_custom_operators(input: TokenStream) -> TokenStream {
    let mut output = TokenStream::new();
    let mut tokens = input.into_iter().peekable();
    while let Some(token) = tokens.next() {
        match token {
            TokenTree::Punct(at) if at.as_char() == '@' => {
                emit_marker(&mut output, "__tensorcrate_matmul_operator__", at.span());
            }
            TokenTree::Punct(dot)
                if dot.as_char() == '.'
                    && matches!(tokens.peek(), Some(TokenTree::Punct(star)) if star.as_char() == '*') =>
            {
                let star = tokens.next().expect("peeked at the elementwise `*`");
                emit_marker(
                    &mut output,
                    "__tensorcrate_elementwise_mul_operator__",
                    star.span(),
                );
            }
            TokenTree::Group(group) => {
                let mut rewritten =
                    Group::new(group.delimiter(), rewrite_custom_operators(group.stream()));
                rewritten.set_span(group.span());
                output.extend([TokenTree::Group(rewritten)]);
            }
            token => output.extend([token]),
        }
    }
    output
}

fn emit_marker(output: &mut TokenStream, name: &str, span: Span) {
    let mut slash = Punct::new('/', Spacing::Alone);
    slash.set_span(span);
    output.extend([TokenTree::Punct(slash.clone())]);
    output.extend([TokenTree::Ident(Ident::new(name, span))]);
    output.extend([TokenTree::Punct(slash)]);
}

/// Recover the operands from the reserved `(lhs / marker) / rhs` AST shape.
fn matmul_operands(expr: &Expr) -> Option<(&Expr, &Expr)> {
    marked_operands(expr, "__tensorcrate_matmul_operator__")
}

fn matmul_binary_operands(outer: &syn::ExprBinary) -> Option<(&Expr, &Expr)> {
    marked_binary_operands(outer, "__tensorcrate_matmul_operator__")
}

fn elementwise_mul_operands(expr: &Expr) -> Option<(&Expr, &Expr)> {
    marked_operands(expr, "__tensorcrate_elementwise_mul_operator__")
}

fn elementwise_mul_binary_operands(outer: &syn::ExprBinary) -> Option<(&Expr, &Expr)> {
    marked_binary_operands(outer, "__tensorcrate_elementwise_mul_operator__")
}

fn marked_operands<'a>(expr: &'a Expr, marker: &str) -> Option<(&'a Expr, &'a Expr)> {
    let Expr::Binary(outer) = expr else {
        return None;
    };
    marked_binary_operands(outer, marker)
}

fn marked_binary_operands<'a>(
    outer: &'a syn::ExprBinary,
    marker: &str,
) -> Option<(&'a Expr, &'a Expr)> {
    if !matches!(outer.op, BinOp::Div(_)) {
        return None;
    }
    let Expr::Binary(partial) = &*outer.left else {
        return None;
    };
    if matches!(partial.op, BinOp::Div(_))
        && matches!(&*partial.right, Expr::Path(path) if path.path.is_ident(marker))
    {
        Some((&partial.left, &outer.right))
    } else {
        None
    }
}

fn expand(input: TokenStream) -> syn::Result<TokenStream> {
    let mut stmts = Block::parse_within.parse2(input)?;
    let backend = take_backend_directive(&mut stmts)?;
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
        out.push(lower_let(stmt, &mut env, backend)?);
    }

    let ty = infer(result, &env)?;
    validate_backend_type(ty, backend, result.span())?;
    let body = lower(result, ty, &env, backend)?;
    let annotated = ty.rust_type(backend);

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

/// Remove the optional leading `backend = Host;` / `backend = Metal;` and
/// `dtype = f32;` / `dtype = f64;` directives, in either order.
fn take_backend_directive(stmts: &mut Vec<Stmt>) -> syn::Result<BackendChoice> {
    let mut metal: Option<bool> = None;
    let mut dtype: Option<(bool, Span)> = None;
    loop {
        let Some(Stmt::Expr(Expr::Assign(assign), Some(_))) = stmts.first() else {
            break;
        };
        let Expr::Path(left) = &*assign.left else {
            break;
        };
        let is_backend = left.path.is_ident("backend");
        let is_dtype = left.path.is_ident("dtype");
        if !is_backend && !is_dtype {
            break;
        }
        let message = if is_backend {
            "math! backend must be `Host` or `Metal`"
        } else {
            "math! dtype must be `f32` or `f64`"
        };
        let Expr::Path(right) = &*assign.right else {
            return Err(syn::Error::new(assign.right.span(), message));
        };
        if is_backend {
            if metal.is_some() {
                return Err(syn::Error::new(right.span(), "math! backend is set twice"));
            }
            metal = Some(if right.path.is_ident("Host") {
                false
            } else if right.path.is_ident("Metal") {
                true
            } else {
                return Err(syn::Error::new(right.span(), message));
            });
        } else {
            if dtype.is_some() {
                return Err(syn::Error::new(right.span(), "math! dtype is set twice"));
            }
            dtype = Some(if right.path.is_ident("f32") {
                (true, right.span())
            } else if right.path.is_ident("f64") {
                (false, right.span())
            } else {
                return Err(syn::Error::new(right.span(), message));
            });
        }
        stmts.remove(0);
    }
    match (metal, dtype) {
        (Some(true), Some((false, span))) => Err(syn::Error::new(
            span,
            "the Metal backend computes in f32; use `dtype = f32;` or the Host backend",
        )),
        (Some(true), _) => Ok(BackendChoice::Metal),
        (_, Some((true, _))) => Ok(BackendChoice::HostF32),
        _ => Ok(BackendChoice::Host),
    }
}

fn validate_backend_type(ty: Ty, backend: BackendChoice, span: Span) -> syn::Result<()> {
    if backend.is_metal() && (ty.complex || ty.dual) {
        return Err(syn::Error::new(
            span,
            "the Metal backend supports real f32 values only (no `i` or `d` literals)",
        ));
    }
    Ok(())
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
fn lower_let(stmt: &Stmt, env: &mut Env, backend: BackendChoice) -> syn::Result<TokenStream> {
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
    validate_backend_type(ty, backend, init.expr.span())?;
    let value = lower(&init.expr, ty, env, backend)?;
    let name = &pat.ident;
    let annotated = ty.rust_type(backend);
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
            if let Some((lhs, rhs)) = elementwise_mul_operands(expr) {
                let left = infer(lhs, env)?;
                let right = infer(rhs, env)?;
                if left.is_tensor() && right.is_tensor() && left.shape != right.shape {
                    return Err(syn::Error::new(
                        b.span(),
                        "elementwise tensor operands must have the same shape",
                    ));
                }
                return Ok(left.unify(right));
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
                ("min" | "max", 2) => {
                    let (a, b) = (arg_ty(0)?, arg_ty(1)?);
                    require_real(a, args[0].span(), &name)?;
                    require_real(b, args[1].span(), &name)?;
                    if a.is_tensor() && b.is_tensor() && a.shape != b.shape {
                        return Err(syn::Error::new(
                            call.span(),
                            format!("`{name}` tensor operands must have the same shape"),
                        ));
                    }
                    Ok(a.unify(b))
                }
                ("clamp", 3) => {
                    let value = arg_ty(0)?;
                    let low = arg_ty(1)?;
                    let high = arg_ty(2)?;
                    require_real(value, args[0].span(), "clamp")?;
                    require_real(low, args[1].span(), "clamp")?;
                    require_real(high, args[2].span(), "clamp")?;
                    if low.is_tensor() || high.is_tensor() {
                        return Err(syn::Error::new(
                            call.span(),
                            "`clamp` bounds must be scalars",
                        ));
                    }
                    Ok(value)
                }
                ("sum" | "minimum" | "maximum", 1) => {
                    let value = arg_ty(0)?;
                    if !matches!(value.shape, Shape::Vector(_)) {
                        return Err(syn::Error::new(
                            call.span(),
                            format!("`{name}` expects a vector"),
                        ));
                    }
                    if name != "sum" {
                        require_real(value, args[0].span(), &name)?;
                    }
                    Ok(value.element())
                }
                ("prefix_sum", 1) => {
                    let value = arg_ty(0)?;
                    if !matches!(value.shape, Shape::Vector(_)) {
                        return Err(syn::Error::new(
                            call.span(),
                            "`prefix_sum` expects a vector",
                        ));
                    }
                    Ok(value)
                }
                ("sorted", 1 | 2) => {
                    let value = arg_ty(0)?;
                    require_real(value, args[0].span(), "sorted")?;
                    if !matches!(value.shape, Shape::Vector(_)) {
                        return Err(syn::Error::new(call.span(), "`sorted` expects a vector"));
                    }
                    if args.len() == 2 {
                        sort_order(args[1])?;
                    }
                    Ok(value)
                }
                ("pow", 2) => {
                    let (a, b) = (arg_ty(0)?, arg_ty(1)?);
                    if a.is_tensor() && b.is_tensor() && a.shape != b.shape {
                        return Err(syn::Error::new(
                            call.span(),
                            "`pow` expects tensors of the same shape",
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

fn require_real(ty: Ty, span: Span, operation: &str) -> syn::Result<()> {
    if ty.complex || ty.dual {
        Err(syn::Error::new(
            span,
            format!("`{operation}` requires real values"),
        ))
    } else {
        Ok(())
    }
}

fn sort_order(expr: &Expr) -> syn::Result<bool> {
    let Expr::Path(path) = expr else {
        return Err(syn::Error::new(
            expr.span(),
            "sort order must be `ascending` or `descending`",
        ));
    };
    let name = path_name(path)?;
    match name.as_str() {
        "ascending" | "Ascending" => Ok(false),
        "descending" | "Descending" => Ok(true),
        _ => Err(syn::Error::new(
            expr.span(),
            "sort order must be `ascending` or `descending`",
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
    ("sqrt", "Sqrt"),
];

/// Emit `expr` as Rust of type `target`, widening subexpressions as needed.
fn lower(expr: &Expr, target: Ty, env: &Env, backend: BackendChoice) -> syn::Result<TokenStream> {
    match expr {
        Expr::Lit(lit) => lower_literal(lit, target, backend),
        Expr::Array(arr) => {
            let literal = tensor_literal(arr)?;
            let element = target.element();
            let rows = literal
                .rows
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|e| lower(e, element, env, backend))
                        .collect::<syn::Result<Vec<_>>>()
                })
                .collect::<syn::Result<Vec<_>>>()?;
            let element_type = element.element_type(backend);
            let host = if literal.is_matrix {
                quote!(::tensorcrate::tensors::Matrix::<#element_type, ::tensorcrate::tensors::Host>::from_rows([#([#(#rows),*]),*]))
            } else {
                let values = &rows[0];
                quote!(::tensorcrate::tensors::Vector::<#element_type, ::tensorcrate::tensors::Host>::new([#(#values),*]))
            };
            Ok(match backend {
                BackendChoice::Host | BackendChoice::HostF32 => host,
                BackendChoice::Metal => {
                    quote!((#host).to_backend::<::tensorcrate::tensors::Metal>())
                }
            })
        }
        Expr::Path(p) => {
            let name = path_name(p)?;
            let ident = &p.path.segments[0].ident;
            let ty = env.get(&name).copied().unwrap_or(Ty::REAL);
            Ok(widen(quote!(#ident), ty, target, backend))
        }
        Expr::Paren(p) => {
            let inner = lower(&p.expr, target, env, backend)?;
            Ok(quote!((#inner)))
        }
        Expr::Group(g) => lower(&g.expr, target, env, backend),
        Expr::Unary(u) => match u.op {
            UnOp::Neg(_) => {
                let inner = lower(&u.expr, target, env, backend)?;
                Ok(quote!(-(#inner)))
            }
            _ => Err(syn::Error::new(expr.span(), "unsupported unary operator")),
        },
        Expr::Binary(b) => lower_binary(b, target, env, backend),
        Expr::Call(call) => lower_call(call, target, env, backend),
        other => Err(syn::Error::new(
            other.span(),
            "unsupported expression in math!",
        )),
    }
}

fn lower_literal(
    lit: &syn::ExprLit,
    target: Ty,
    backend: BackendChoice,
) -> syn::Result<TokenStream> {
    let (kind, digits) = classify_literal(&lit.lit)?;
    let value: f64 = digits
        .parse()
        .map_err(|_| syn::Error::new(lit.span(), "invalid numeric literal"))?;
    let value = match backend {
        BackendChoice::Host => Literal::f64_suffixed(value),
        BackendChoice::HostF32 | BackendChoice::Metal => Literal::f32_suffixed(value as f32),
    };

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
            let zero = zero_of(target.complex, backend);
            quote!(::tensorcrate::numbers::Dual::new(#zero, #coefficient))
        } else {
            quote!(::tensorcrate::numbers::Dual::constant(#coefficient))
        }
    } else {
        coefficient
    })
}

fn lower_matmul(
    left: &Expr,
    right: &Expr,
    target: Ty,
    env: &Env,
    backend: BackendChoice,
) -> syn::Result<TokenStream> {
    let (left_ty, right_ty) = (infer(left, env)?, infer(right, env)?);
    let left = lower(
        left,
        target.element().with_shape(left_ty.shape),
        env,
        backend,
    )?;
    let right = lower(
        right,
        target.element().with_shape(right_ty.shape),
        env,
        backend,
    )?;
    Ok(match (left_ty.shape, right_ty.shape) {
        (Shape::Matrix(_, _), Shape::Matrix(_, _)) => quote!((#left).matmul(&(#right))),
        (Shape::Matrix(_, _), Shape::Vector(_)) => quote!((#left).matvec(&(#right))),
        (Shape::Vector(_), Shape::Matrix(_, _)) => quote!((#left).vecmat(&(#right))),
        _ => unreachable!("infer_matmul validated the operand shapes"),
    })
}

fn lower_binary(
    b: &syn::ExprBinary,
    target: Ty,
    env: &Env,
    backend: BackendChoice,
) -> syn::Result<TokenStream> {
    if let Some((left, right)) = matmul_binary_operands(b) {
        return lower_matmul(left, right, target, env, backend);
    }
    if let Some((left, right)) = elementwise_mul_binary_operands(b) {
        return lower_elementwise_binary(left, right, target, env, backend);
    }

    let (left_ty, right_ty) = (infer(&b.left, env)?, infer(&b.right, env)?);
    if matches!(b.op, BinOp::Mul(_))
        && matches!(
            (left_ty.shape, right_ty.shape),
            (Shape::Vector(_), Shape::Vector(_))
        )
    {
        let vector_target = target.with_shape(left_ty.shape);
        let left = lower(&b.left, vector_target, env, backend)?;
        let right = lower(&b.right, vector_target, env, backend)?;
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
        let left = lower(&b.left, target, env, backend)?;
        let right = lower(&b.right, target, env, backend)?;
        return Ok(quote!((#left #op #right)));
    }

    // At least one side is a tensor. Two tensors combine elementwise (which can
    // fail on a shape mismatch); a tensor and a scalar broadcast the scalar.
    let element = target.element();
    Ok(match (left_ty.is_tensor(), right_ty.is_tensor()) {
        (true, true) => {
            let left = lower(&b.left, target, env, backend)?;
            let right = lower(&b.right, target, env, backend)?;
            quote!((#left #op #right))
        }
        (true, false) => {
            let left = lower(&b.left, target, env, backend)?;
            let right = lower(&b.right, element, env, backend)?;
            quote!((#left).broadcast_right(#right, #broadcast_op))
        }
        (false, true) => {
            let left = lower(&b.left, element, env, backend)?;
            let right = lower(&b.right, target, env, backend)?;
            quote!((#right).broadcast_left(#left, #broadcast_op))
        }
        (false, false) => unreachable!("target is a tensor only if an operand is"),
    })
}

fn lower_elementwise_binary(
    left: &Expr,
    right: &Expr,
    target: Ty,
    env: &Env,
    backend: BackendChoice,
) -> syn::Result<TokenStream> {
    let (left_ty, right_ty) = (infer(left, env)?, infer(right, env)?);
    if !target.is_tensor() {
        let left = lower(left, target, env, backend)?;
        let right = lower(right, target, env, backend)?;
        return Ok(quote!((#left * #right)));
    }

    let element = target.element();
    Ok(match (left_ty.is_tensor(), right_ty.is_tensor()) {
        (true, true) => {
            let left = lower(left, target, env, backend)?;
            let right = lower(right, target, env, backend)?;
            quote!((#left * #right))
        }
        (true, false) => {
            let left = lower(left, target, env, backend)?;
            let right = lower(right, element, env, backend)?;
            quote!((#left).broadcast_right(#right, ::tensorcrate::tensors::BinaryOp::Mul))
        }
        (false, true) => {
            let left = lower(left, element, env, backend)?;
            let right = lower(right, target, env, backend)?;
            quote!((#right).broadcast_left(#left, ::tensorcrate::tensors::BinaryOp::Mul))
        }
        (false, false) => unreachable!("target is a tensor only if an operand is"),
    })
}

fn lower_call(
    call: &syn::ExprCall,
    target: Ty,
    env: &Env,
    backend: BackendChoice,
) -> syn::Result<TokenStream> {
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
        lower(args[i], target.element().with_shape(ty.shape), env, backend)
    };

    match (name.as_str(), args.len()) {
        ("min" | "max", 2) => lower_min_max(&name, args[0], args[1], target, env, backend),
        ("clamp", 3) => {
            let value = lower(args[0], target, env, backend)?;
            let low = lower(args[1], target.element(), env, backend)?;
            let high = lower(args[2], target.element(), env, backend)?;
            Ok(quote!((#value).clamp(#low, #high)))
        }
        ("sum", 1) => {
            let value = tensor_arg(0, env)?;
            Ok(quote!((#value).sum()))
        }
        ("minimum", 1) => {
            let value = tensor_arg(0, env)?;
            Ok(quote!((#value).minimum().expect("math! vectors are non-empty")))
        }
        ("maximum", 1) => {
            let value = tensor_arg(0, env)?;
            Ok(quote!((#value).maximum().expect("math! vectors are non-empty")))
        }
        ("prefix_sum", 1) => {
            let value = tensor_arg(0, env)?;
            Ok(quote!((#value).prefix_sum()))
        }
        ("sorted", 1 | 2) => {
            let value = tensor_arg(0, env)?;
            let descending = args.get(1).map_or(Ok(false), |order| sort_order(order))?;
            Ok(match backend {
                BackendChoice::Host | BackendChoice::HostF32 if descending => {
                    quote!((#value).sorted_by(|__a, __b| __b.total_cmp(__a)))
                }
                BackendChoice::Host | BackendChoice::HostF32 => {
                    quote!((#value).sorted_by(|__a, __b| __a.total_cmp(__b)))
                }
                BackendChoice::Metal if descending => quote!((#value).sorted(
                    ::tensorcrate::tensors::SortOrder::Descending
                )),
                BackendChoice::Metal => quote!((#value).sorted(
                    ::tensorcrate::tensors::SortOrder::Ascending
                )),
            })
        }
        ("pow", 2) => {
            let base_ty = infer(args[0], env)?;
            let exponent_ty = infer(args[1], env)?;
            if target.is_tensor() {
                let element = target.element();
                // Every order goes through `Power`, which the tensor references
                // implement alongside the numbers. Each has a resident kernel,
                // so a `backend = Metal` block stays on the GPU.
                match (base_ty.is_tensor(), exponent_ty.is_tensor()) {
                    (true, true) => {
                        let base = lower(args[0], target, env, backend)?;
                        let exponent = lower(args[1], target, env, backend)?;
                        Ok(quote!(::tensorcrate::numbers::Power::power(
                            &(#base),
                            &(#exponent)
                        )))
                    }
                    (true, false) => {
                        let base = lower(args[0], target, env, backend)?;
                        let exponent = lower(args[1], element, env, backend)?;
                        Ok(quote!(::tensorcrate::numbers::Power::power(
                            &(#base),
                            #exponent
                        )))
                    }
                    _ => {
                        let base = lower(args[0], element, env, backend)?;
                        let exponent = lower(args[1], target, env, backend)?;
                        Ok(quote!(::tensorcrate::numbers::Power::power(
                            #base,
                            &(#exponent)
                        )))
                    }
                }
            } else {
                let base = lower(args[0], target, env, backend)?;
                let exponent = lower(args[1], target, env, backend)?;
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
            Ok(match backend {
                BackendChoice::Host | BackendChoice::HostF32 => quote!((#a).inverse()?),
                BackendChoice::Metal => quote!((#a)
                    .to_backend::<::tensorcrate::tensors::Host>()
                    .inverse()?
                    .to_backend::<::tensorcrate::tensors::Metal>()),
            })
        }
        ("det", 1) => {
            let a = tensor_arg(0, env)?;
            Ok(match backend {
                BackendChoice::Host | BackendChoice::HostF32 => quote!((#a).determinant()),
                BackendChoice::Metal => quote!((#a)
                    .to_backend::<::tensorcrate::tensors::Host>()
                    .determinant()),
            })
        }
        ("conj", 1) => {
            let inner = lower(args[0], target, env, backend)?;
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
            let inner = lower(args[0], target, env, backend)?;
            // A tensor has the function as an inherent method on both backends:
            // the host maps the scalar op over its elements, a resident tensor
            // runs the unary kernel and stays on the GPU.
            Ok(if target.is_tensor() {
                quote!((#inner).#method())
            } else {
                quote!(::tensorcrate::numbers::#trait_ident::#method(#inner))
            })
        }
        (_, n) => Err(syn::Error::new(
            call.span(),
            format!("`{name}` does not take {n} argument(s) in math!"),
        )),
    }
}

fn lower_min_max(
    operation: &str,
    left: &Expr,
    right: &Expr,
    target: Ty,
    env: &Env,
    backend: BackendChoice,
) -> syn::Result<TokenStream> {
    let (left_ty, right_ty) = (infer(left, env)?, infer(right, env)?);
    let method = Ident::new(operation, left.span());
    let scalar_method = Ident::new(&format!("{operation}_scalar"), left.span());
    Ok(match (left_ty.is_tensor(), right_ty.is_tensor()) {
        (false, false) => {
            let left = lower(left, target, env, backend)?;
            let right = lower(right, target, env, backend)?;
            quote!((#left).#method(#right))
        }
        (true, true) => {
            let left = lower(left, target, env, backend)?;
            let right = lower(right, target, env, backend)?;
            quote!((#left).#method(&(#right)))
        }
        (true, false) => {
            let left = lower(left, target, env, backend)?;
            let right = lower(right, target.element(), env, backend)?;
            quote!((#left).#scalar_method(#right))
        }
        (false, true) => {
            let left = lower(left, target.element(), env, backend)?;
            let right = lower(right, target, env, backend)?;
            quote!((#right).#scalar_method(#left))
        }
    })
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
fn zero_of(complex: bool, backend: BackendChoice) -> TokenStream {
    let zero = match backend {
        BackendChoice::Host => quote!(0f64),
        BackendChoice::HostF32 | BackendChoice::Metal => quote!(0f32),
    };
    if complex {
        quote!(::tensorcrate::numbers::Complex::constant(#zero))
    } else {
        zero
    }
}

/// Widen a value of type `from` so it can be used where `to` is expected.
///
/// The axes are handled in order: lift the coefficients to complex, then wrap in
/// a dual. A tensor is widened by mapping the same conversion over its elements.
/// Narrowing never happens.
fn widen(value: TokenStream, from: Ty, to: Ty, backend: BackendChoice) -> TokenStream {
    if from.is_tensor() {
        // The element conversion, expressed on a bound element.
        let converted = widen_scalar(quote!(__x), from.element(), to.element());
        return if converted.to_string() == "__x" {
            value
        } else {
            match backend {
                BackendChoice::Host | BackendChoice::HostF32 => {
                    quote!((#value).map(|&__x| #converted))
                }
                BackendChoice::Metal => unreachable!("Metal values cannot require widening"),
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;

    fn expansion_error(input: TokenStream) -> String {
        expand(rewrite_custom_operators(input))
            .expect_err("the macro input should be rejected")
            .to_string()
    }

    #[test]
    fn backend_directives_are_validated() {
        assert!(
            expansion_error(quote! { backend = Cpu; [1, 2] })
                .contains("backend must be `Host` or `Metal`")
        );
        assert!(
            expansion_error(quote! { backend = Metal; [1 + 1i] })
                .contains("Metal backend supports real f32 values only")
        );
    }

    #[test]
    fn dtype_directives_are_validated() {
        assert!(expansion_error(quote! { dtype = f16; [1, 2] }).contains("dtype must be `f32`"));
        assert!(
            expansion_error(quote! { backend = Metal; dtype = f64; [1, 2] })
                .contains("Metal backend computes in f32")
        );
        assert!(expansion_error(quote! { dtype = f32; dtype = f64; [1, 2] }).contains("set twice"));
        assert!(
            expansion_error(quote! { backend = Host; backend = Host; [1, 2] })
                .contains("set twice")
        );
    }

    #[test]
    fn ordering_shapes_and_orders_are_checked_during_expansion() {
        assert!(
            expansion_error(quote! { min([1, 2], [1, 2, 3]) }).contains("must have the same shape")
        );
        assert!(expansion_error(quote! { sorted([[1, 2], [3, 4]]) }).contains("expects a vector"));
        assert!(
            expansion_error(quote! { sorted([1, 2], sideways) })
                .contains("must be `ascending` or `descending`")
        );
    }

    #[test]
    fn explicit_elementwise_products_check_shapes() {
        assert!(
            expansion_error(quote! { [1, 2] .* [1, 2, 3] })
                .contains("elementwise tensor operands must have the same shape")
        );
    }
}
