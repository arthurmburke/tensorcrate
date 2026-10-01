//! From the macro's tokens to an [`ast::Block`](crate::ast::Block): the
//! directives, the bindings and the expressions, with every syntax error
//! reported here.
//!
//! The input is parsed with `syn` as Rust syntax, which gives operator
//! precedence, parentheses, grouping and array literals for free. The only
//! extensions are the `@` and `.*` operators, rewritten to something `syn`
//! accepts before it sees them, and literal suffixes: `2i` is imaginary and
//! `2d` is the dual infinitesimal ε.
//!
//! (ε is spelled `d`, not `e`: rustc's lexer treats any `e` after digits as the
//! start of a float exponent, so `2e`/`2eps` cannot reach a macro at all.)

use proc_macro2::{Group, Ident, Punct, Spacing, Span, TokenStream, TokenTree};
use syn::parse::Parser;
use syn::spanned::Spanned;
use syn::{BinOp, Lit, Pat, Stmt, UnOp};

use crate::ast::{Arith, Binding, Block, Builtin, Expr, ExprKind, LitKind, Literal};
use crate::types::{BackendChoice, Dtype};

const MATMUL_MARKER: &str = "__tensorcrate_matmul_operator__";
const ELEMENTWISE_MUL_MARKER: &str = "__tensorcrate_elementwise_mul_operator__";

/// `@` and `.*` are not part of Rust's expression grammar. Rewrite each to a
/// reserved `lhs / marker / rhs` form before `syn` parses the block. Division
/// has the same precedence and associativity these product operators should
/// have, and [`marked_operands`] recovers the operands without confusing an
/// ordinary `/`.
pub(crate) fn rewrite_custom_operators(input: TokenStream) -> TokenStream {
    let mut output = TokenStream::new();
    let mut tokens = input.into_iter().peekable();
    while let Some(token) = tokens.next() {
        match token {
            TokenTree::Punct(at) if at.as_char() == '@' => {
                emit_marker(&mut output, MATMUL_MARKER, at.span());
            }
            TokenTree::Punct(dot)
                if dot.as_char() == '.'
                    && matches!(tokens.peek(), Some(TokenTree::Punct(star)) if star.as_char() == '*') =>
            {
                let star = tokens.next().expect("peeked at the elementwise `*`");
                emit_marker(&mut output, ELEMENTWISE_MUL_MARKER, star.span());
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
fn marked_operands<'a>(
    outer: &'a syn::ExprBinary,
    marker: &str,
) -> Option<(&'a syn::Expr, &'a syn::Expr)> {
    if !matches!(outer.op, BinOp::Div(_)) {
        return None;
    }
    let syn::Expr::Binary(partial) = &*outer.left else {
        return None;
    };
    if matches!(partial.op, BinOp::Div(_))
        && matches!(&*partial.right, syn::Expr::Path(path) if path.path.is_ident(marker))
    {
        Some((&partial.left, &outer.right))
    } else {
        None
    }
}

/// What the leading directives of a block chose.
pub(crate) struct Directives {
    pub(crate) backend: BackendChoice,
    pub(crate) fuse: bool,
    /// Whether fused kernels may regroup associative chains.
    pub(crate) reassociate: bool,
}

/// Parse a whole block, its operators already rewritten.
pub(crate) fn block(input: TokenStream) -> syn::Result<(Directives, Block)> {
    let mut stmts = syn::Block::parse_within.parse2(input)?;
    let directives = take_directives(&mut stmts)?;
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
    let bindings = leading.iter().map(binding).collect::<syn::Result<_>>()?;
    let result = expr(result)?;
    Ok((directives, Block { bindings, result }))
}

/// Remove the optional leading directives, in any order: `backend = Host;` /
/// `backend = Metal;`, `dtype = f64;` / `f32` / `f16` / `bf16`, `fuse = true;`
/// / `fuse = false;` and `reassociate = true;` / `reassociate = false;`.
fn take_directives(stmts: &mut Vec<Stmt>) -> syn::Result<Directives> {
    let mut metal: Option<bool> = None;
    let mut dtype: Option<(Dtype, Span)> = None;
    let mut fuse: Option<bool> = None;
    let mut reassociate: Option<bool> = None;
    while let Some(Stmt::Expr(syn::Expr::Assign(assign), Some(_))) = stmts.first() {
        let syn::Expr::Path(left) = &*assign.left else {
            break;
        };
        let flag = if left.path.is_ident("fuse") {
            Some(("fuse", &mut fuse))
        } else if left.path.is_ident("reassociate") {
            Some(("reassociate", &mut reassociate))
        } else {
            None
        };
        if let Some((name, slot)) = flag {
            let syn::Expr::Lit(syn::ExprLit {
                lit: Lit::Bool(value),
                ..
            }) = &*assign.right
            else {
                return Err(syn::Error::new(
                    assign.right.span(),
                    format!("math! {name} must be `true` or `false`"),
                ));
            };
            if slot.is_some() {
                return Err(syn::Error::new(
                    value.span(),
                    format!("math! {name} is set twice"),
                ));
            }
            *slot = Some(value.value);
            stmts.remove(0);
            continue;
        }
        let is_backend = left.path.is_ident("backend");
        let is_dtype = left.path.is_ident("dtype");
        if !is_backend && !is_dtype {
            break;
        }
        let message = if is_backend {
            "math! backend must be `Host` or `Metal`"
        } else {
            "math! dtype must be `f64`, `f32`, `f16` or `bf16`"
        };
        let syn::Expr::Path(right) = &*assign.right else {
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
            let chosen = [
                ("f64", Dtype::F64),
                ("f32", Dtype::F32),
                ("f16", Dtype::F16),
                ("bf16", Dtype::Bf16),
            ]
            .into_iter()
            .find(|(name, _)| right.path.is_ident(name))
            .map(|(_, dtype)| dtype)
            .ok_or_else(|| syn::Error::new(right.span(), message))?;
            dtype = Some((chosen, right.span()));
        }
        stmts.remove(0);
    }
    let metal = metal.unwrap_or(false);
    let dtype = match (metal, dtype) {
        (true, Some((Dtype::F64, span))) => {
            return Err(syn::Error::new(
                span,
                "the Metal backend computes in f32, f16 or bf16; use one of those or the Host backend",
            ));
        }
        (_, Some((dtype, _))) => dtype,
        (true, None) => Dtype::F32,
        (false, None) => Dtype::F64,
    };
    Ok(Directives {
        backend: BackendChoice { metal, dtype },
        fuse: fuse.unwrap_or(true),
        reassociate: reassociate.unwrap_or(true),
    })
}

/// A `let` statement, the only kind allowed before the result.
fn binding(stmt: &Stmt) -> syn::Result<Binding> {
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
    Ok(Binding {
        name: pat.ident.clone(),
        value: expr(&init.expr)?,
    })
}

/// Parse one expression.
fn expr(source: &syn::Expr) -> syn::Result<Expr> {
    let kind = match source {
        syn::Expr::Lit(lit) => ExprKind::Literal(literal(&lit.lit)?),
        syn::Expr::Array(array) => tensor(array)?,
        syn::Expr::Path(path) => ExprKind::Var(plain_ident(path)?.clone()),
        syn::Expr::Paren(paren) => ExprKind::Group {
            inner: Box::new(expr(&paren.expr)?),
            parenthesized: true,
        },
        syn::Expr::Group(group) => ExprKind::Group {
            inner: Box::new(expr(&group.expr)?),
            parenthesized: false,
        },
        syn::Expr::Unary(unary) => match unary.op {
            UnOp::Neg(_) => ExprKind::Neg(Box::new(expr(&unary.expr)?)),
            _ => {
                return Err(syn::Error::new(source.span(), "unsupported unary operator"));
            }
        },
        syn::Expr::Binary(binary) => self::binary(binary)?,
        syn::Expr::Call(call) => self::call(call)?,
        other => {
            return Err(syn::Error::new(
                other.span(),
                "unsupported expression in math!",
            ));
        }
    };
    Ok(Expr {
        kind,
        span: source.span(),
        ty: (),
    })
}

fn binary(binary: &syn::ExprBinary) -> syn::Result<ExprKind<()>> {
    let pair = |(left, right): (&syn::Expr, &syn::Expr)| -> syn::Result<_> {
        Ok((Box::new(expr(left)?), Box::new(expr(right)?)))
    };
    if let Some(operands) = marked_operands(binary, MATMUL_MARKER) {
        let (left, right) = pair(operands)?;
        return Ok(ExprKind::MatMul(left, right));
    }
    if let Some(operands) = marked_operands(binary, ELEMENTWISE_MUL_MARKER) {
        let (left, right) = pair(operands)?;
        return Ok(ExprKind::ElementwiseMul(left, right));
    }
    let op = match binary.op {
        BinOp::Add(_) => Arith::Add,
        BinOp::Sub(_) => Arith::Sub,
        BinOp::Mul(_) => Arith::Mul,
        BinOp::Div(_) => Arith::Div,
        BinOp::Rem(_) => Arith::Rem,
        BinOp::BitXor(_) => {
            return Err(syn::Error::new(
                binary.op.span(),
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
    let (left, right) = pair((&binary.left, &binary.right))?;
    Ok(ExprKind::Binary { op, left, right })
}

fn call(call: &syn::ExprCall) -> syn::Result<ExprKind<()>> {
    let syn::Expr::Path(function) = &*call.func else {
        return Err(syn::Error::new(
            call.func.span(),
            "expected a function name",
        ));
    };
    let name = plain_ident(function)?.to_string();
    let args: Vec<&syn::Expr> = call.args.iter().collect();
    let Some(mut builtin) = Builtin::resolve(&name, args.len()) else {
        return Err(syn::Error::new(
            call.span(),
            format!("`{name}` does not take {} argument(s) in math!", args.len()),
        ));
    };
    // `sorted`'s order is a keyword, not a value.
    let values = if let Builtin::Sorted { descending } = &mut builtin {
        if let Some(order) = args.get(1) {
            *descending = sort_order(order)?;
        }
        &args[..1]
    } else {
        &args[..]
    };
    Ok(ExprKind::Call {
        builtin,
        args: values
            .iter()
            .map(|arg| expr(arg))
            .collect::<syn::Result<_>>()?,
        name_span: call.func.span(),
    })
}

/// `ascending` or `descending` (either capitalized), as whether it is the
/// latter.
fn sort_order(order: &syn::Expr) -> syn::Result<bool> {
    let error = || {
        syn::Error::new(
            order.span(),
            "sort order must be `ascending` or `descending`",
        )
    };
    let syn::Expr::Path(path) = order else {
        return Err(error());
    };
    match plain_ident(path)?.to_string().as_str() {
        "ascending" | "Ascending" => Ok(false),
        "descending" | "Descending" => Ok(true),
        _ => Err(error()),
    }
}

/// A numeric literal, classified by its suffix.
fn literal(lit: &Lit) -> syn::Result<Literal> {
    let (suffix, digits) = match lit {
        Lit::Int(v) => (v.suffix(), v.base10_digits()),
        Lit::Float(v) => (v.suffix(), v.base10_digits()),
        other => {
            return Err(syn::Error::new(
                other.span(),
                "math! expects a numeric literal",
            ));
        }
    };
    let kind = match suffix {
        "" => LitKind::Real,
        "i" => LitKind::Imaginary,
        "d" => LitKind::Epsilon,
        other => {
            return Err(syn::Error::new(
                lit.span(),
                format!("unknown literal suffix `{other}` (use `i` for imaginary, `d` for ε)"),
            ));
        }
    };
    let value = digits
        .parse()
        .map_err(|_| syn::Error::new(lit.span(), "invalid numeric literal"))?;
    Ok(Literal { kind, value })
}

/// Read an array literal as a vector (`[1, 2]`) or a matrix (`[[1, 2], [3, 4]]`),
/// checking that matrix rows all have the same length.
fn tensor(array: &syn::ExprArray) -> syn::Result<ExprKind<()>> {
    if array.elems.is_empty() {
        return Err(syn::Error::new(
            array.span(),
            "an empty tensor literal has no element type",
        ));
    }
    let nested = array
        .elems
        .iter()
        .filter(|e| matches!(e, syn::Expr::Array(_)))
        .count();
    if nested == 0 {
        let row = array.elems.iter().map(expr).collect::<syn::Result<_>>()?;
        return Ok(ExprKind::Tensor {
            rows: vec![row],
            matrix: false,
        });
    }
    if nested != array.elems.len() {
        return Err(syn::Error::new(
            array.span(),
            "a tensor literal must be all scalars (a vector) or all rows (a matrix)",
        ));
    }

    let mut rows = Vec::new();
    for row in &array.elems {
        let syn::Expr::Array(row) = row else {
            unreachable!()
        };
        if row.elems.iter().any(|e| matches!(e, syn::Expr::Array(_))) {
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
            bad.first().map_or_else(|| array.span(), |e| e.span()),
            format!("every matrix row must have {width} elements"),
        ));
    }
    let rows = rows
        .into_iter()
        .map(|row| row.into_iter().map(expr).collect())
        .collect::<syn::Result<_>>()?;
    Ok(ExprKind::Tensor { rows, matrix: true })
}

/// The identifier of a single-segment path.
fn plain_ident(path: &syn::ExprPath) -> syn::Result<&Ident> {
    path.path
        .get_ident()
        .ok_or_else(|| syn::Error::new(path.span(), "math! expects a plain identifier"))
}
