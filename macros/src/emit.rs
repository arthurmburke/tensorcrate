//! Generating Rust from a checked block.
//!
//! Every expression arrives with its type, so generation only decides how to
//! spell each operation for the types and backend at hand. An expression is
//! lowered to a *target* type, which may be wider than its own (a real operand
//! of a complex sum, say), and values are widened where the two differ.

use std::collections::{HashMap, HashSet};

use proc_macro2::{Ident, TokenStream};
use quote::quote;

use crate::ast::{Arith, Binding, Block, Builtin, Expr, ExprKind, LitKind};
use crate::fusion;
use crate::types::{BackendChoice, Shape, Ty};

/// What lowering knows at a point in the block, beyond the types.
#[derive(Clone, Default)]
pub(crate) struct Env {
    /// Whether elementwise operations are fused; see [`fusion`].
    pub(crate) fuse: bool,
    /// Elementwise bindings that are never materialized: fused lowering
    /// substitutes their value wherever they are read.
    pub(crate) inline: HashMap<String, Expr<Ty>>,
    /// Whether fused kernels may regroup associative chains.
    pub(crate) reassociate: bool,
}

/// Lower the bindings and the result as one block expression, fused when
/// `inline` is given (naming the bindings to substitute rather than
/// materialize).
pub(crate) fn block(
    block: &Block<Ty>,
    backend: BackendChoice,
    inline: Option<HashMap<String, Expr<Ty>>>,
    reassociate: bool,
) -> syn::Result<TokenStream> {
    let env = Env {
        fuse: inline.is_some(),
        inline: inline.unwrap_or_default(),
        reassociate,
    };
    let bindings = &block.bindings;
    let mut out = Vec::new();
    let mut k = 0;
    while k < bindings.len() {
        if env.fuse
            && let Some((tokens, next)) = horizontal(bindings, k, &env, backend)?
        {
            out.push(tokens);
            k = next;
            continue;
        }
        out.push(binding(&bindings[k], &env, backend)?);
        k += 1;
    }
    let result = &block.result;
    let body = lower(result, result.ty, &env, backend)?;
    Ok(quote!({ #(#out)* #body }))
}

/// One `let`, or nothing for a binding that is inlined into its consumers.
fn binding(binding: &Binding<Ty>, env: &Env, backend: BackendChoice) -> syn::Result<TokenStream> {
    let name = &binding.name;
    if env.inline.contains_key(&name.to_string()) {
        return Ok(TokenStream::new());
    }
    let ty = binding.value.ty;
    let value = lower(&binding.value, ty, env, backend)?;
    let annotated = ty.rust_type(backend);
    Ok(quote! { let #name: #annotated = #value; })
}

/// Fuse the binding at `k` with the independent elementwise bindings of the
/// same shape that follow it, if there are any, returning the one `let` that
/// computes them all and the index of the first binding after them.
fn horizontal(
    bindings: &[Binding<Ty>],
    k: usize,
    env: &Env,
    backend: BackendChoice,
) -> syn::Result<Option<(TokenStream, usize)>> {
    let first = &bindings[k];
    if env.inline.contains_key(&first.name.to_string())
        || !fusion::joins_horizontally(&first.value, None, &HashSet::new(), env)
    {
        return Ok(None);
    }
    let shape = first.value.ty.shape;
    let mut members = vec![(first.name.clone(), &first.value)];
    let mut bound: HashSet<String> = HashSet::from([first.name.to_string()]);
    let mut end = k + 1;
    for (j, other) in bindings.iter().enumerate().skip(k + 1) {
        if members.len() == fusion::MAX_OUTPUTS {
            break;
        }
        // Inlined bindings emit nothing, so they do not separate the group.
        if env.inline.contains_key(&other.name.to_string()) {
            continue;
        }
        if bound.contains(&other.name.to_string())
            || !fusion::joins_horizontally(&other.value, Some(shape), &bound, env)
        {
            break;
        }
        members.push((other.name.clone(), &other.value));
        bound.insert(other.name.to_string());
        end = j + 1;
    }
    if members.len() < 2 {
        return Ok(None);
    }
    Ok(fusion::try_fuse_horizontally(&members, env, backend)?.map(|tokens| (tokens, end)))
}

/// Emit `expr` as Rust of type `target`, widening subexpressions as needed.
pub(crate) fn lower(
    expr: &Expr<Ty>,
    target: Ty,
    env: &Env,
    backend: BackendChoice,
) -> syn::Result<TokenStream> {
    if let Some(fused) = fusion::try_fuse(expr, target, env, backend)? {
        return Ok(fused);
    }
    match &expr.kind {
        ExprKind::Literal(literal) => Ok(self::literal(literal, target, backend)),
        ExprKind::Tensor { rows, matrix } => {
            let element = target.element();
            let rows = rows
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|e| lower(e, element, env, backend))
                        .collect::<syn::Result<Vec<_>>>()
                })
                .collect::<syn::Result<Vec<_>>>()?;
            let element_type = element.element_type(backend);
            let host = if *matrix {
                quote!(::tensorcrate::tensors::Matrix::<#element_type, ::tensorcrate::tensors::Host>::from_rows([#([#(#rows),*]),*]))
            } else {
                let values = &rows[0];
                quote!(::tensorcrate::tensors::Vector::<#element_type, ::tensorcrate::tensors::Host>::new([#(#values),*]))
            };
            Ok(if backend.is_metal() {
                quote!((#host).to_backend::<::tensorcrate::tensors::Metal>())
            } else {
                host
            })
        }
        ExprKind::Var(ident) => {
            if let Some(inlined) = env.inline.get(&ident.to_string()) {
                // Inlining only chooses bindings read inside fused groups, so
                // this is a fallback rather than a path taken: recompute.
                return lower(inlined, target, env, backend);
            }
            Ok(widen(quote!(#ident), expr.ty, target, backend))
        }
        ExprKind::Group {
            inner,
            parenthesized,
        } => {
            let inner = lower(inner, target, env, backend)?;
            Ok(if *parenthesized {
                quote!((#inner))
            } else {
                inner
            })
        }
        ExprKind::Neg(inner) => {
            let inner = lower(inner, target, env, backend)?;
            Ok(if target.is_tensor() {
                quote!(-&(#inner))
            } else {
                quote!(-(#inner))
            })
        }
        ExprKind::Binary { op, left, right } => binary(*op, left, right, target, env, backend),
        ExprKind::MatMul(left, right) => matmul(left, right, target, env, backend),
        ExprKind::ElementwiseMul(left, right) => {
            elementwise_binary(Arith::Mul, left, right, target, env, backend)
        }
        ExprKind::Call {
            builtin,
            args,
            name_span,
        } => call(*builtin, args, *name_span, target, env, backend),
    }
}

fn literal(literal: &crate::ast::Literal, target: Ty, backend: BackendChoice) -> TokenStream {
    let value = backend.dtype.literal(literal.value);

    // Build the coefficient first: `2i` is `0 + 2i`, anything else is purely
    // real.
    let coefficient = if target.complex {
        if literal.kind == LitKind::Imaginary {
            quote!(::tensorcrate::numbers::Complex::imaginary(#value))
        } else {
            quote!(::tensorcrate::numbers::Complex::constant(#value))
        }
    } else {
        quote!(#value)
    };

    // Then place it on the real or the ε side of the dual.
    if target.dual {
        if literal.kind == LitKind::Epsilon {
            let zero = zero_of(target.complex, backend);
            quote!(::tensorcrate::numbers::Dual::new(#zero, #coefficient))
        } else {
            quote!(::tensorcrate::numbers::Dual::constant(#coefficient))
        }
    } else {
        coefficient
    }
}

/// A product by `@` or `matmul`, whose operand shapes the type check matched.
fn matmul(
    left: &Expr<Ty>,
    right: &Expr<Ty>,
    target: Ty,
    env: &Env,
    backend: BackendChoice,
) -> syn::Result<TokenStream> {
    let (left_shape, right_shape) = (left.ty.shape, right.ty.shape);
    let left = lower(left, target.element().with_shape(left_shape), env, backend)?;
    let right = lower(
        right,
        target.element().with_shape(right_shape),
        env,
        backend,
    )?;
    Ok(match (left_shape, right_shape) {
        (Shape::Matrix(_, _), Shape::Matrix(_, _)) => quote!((#left).matmul(&(#right))),
        (Shape::Matrix(_, _), Shape::Vector(_)) => quote!((#left).matvec(&(#right))),
        (Shape::Vector(_), Shape::Matrix(_, _)) => quote!((#left).vecmat(&(#right))),
        _ => unreachable!("the type check matched the operand shapes"),
    })
}

fn binary(
    op: Arith,
    left: &Expr<Ty>,
    right: &Expr<Ty>,
    target: Ty,
    env: &Env,
    backend: BackendChoice,
) -> syn::Result<TokenStream> {
    if op == Arith::Mul
        && let (Shape::Vector(_), Shape::Vector(_)) = (left.ty.shape, right.ty.shape)
    {
        let vector_target = target.with_shape(left.ty.shape);
        let left = lower(left, vector_target, env, backend)?;
        let right = lower(right, vector_target, env, backend)?;
        return Ok(quote!((#left).dot(&(#right))));
    }
    elementwise_binary(op, left, right, target, env, backend)
}

/// An arithmetic operation between scalars, between tensors of one shape, or
/// between a tensor and a scalar it broadcasts.
fn elementwise_binary(
    op: Arith,
    left: &Expr<Ty>,
    right: &Expr<Ty>,
    target: Ty,
    env: &Env,
    backend: BackendChoice,
) -> syn::Result<TokenStream> {
    let operator = match op {
        Arith::Add => quote!(+),
        Arith::Sub => quote!(-),
        Arith::Mul => quote!(*),
        Arith::Div => quote!(/),
        Arith::Rem => quote!(%),
    };
    if !target.is_tensor() {
        let left = lower(left, target, env, backend)?;
        let right = lower(right, target, env, backend)?;
        return Ok(quote!((#left #operator #right)));
    }

    // At least one side is a tensor. Two tensors combine elementwise; a tensor
    // and a scalar broadcast the scalar.
    let element = target.element();
    let variant = Ident::new(op.variant(), proc_macro2::Span::call_site());
    let broadcast_op = quote!(::tensorcrate::tensors::BinaryOp::#variant);
    Ok(match (left.ty.is_tensor(), right.ty.is_tensor()) {
        // By reference, so a bound tensor stays usable after the operation:
        // `a + a`, or `a` read again on a later line. The by-value and
        // by-reference operators run the same kernel.
        (true, true) => {
            let left = lower(left, target, env, backend)?;
            let right = lower(right, target, env, backend)?;
            quote!((&(#left) #operator &(#right)))
        }
        (true, false) => {
            let left = lower(left, target, env, backend)?;
            let right = lower(right, element, env, backend)?;
            quote!((#left).broadcast_right(#right, #broadcast_op))
        }
        (false, true) => {
            let left = lower(left, element, env, backend)?;
            let right = lower(right, target, env, backend)?;
            quote!((#right).broadcast_left(#left, #broadcast_op))
        }
        (false, false) => unreachable!("target is a tensor only if an operand is"),
    })
}

fn call(
    builtin: Builtin,
    args: &[Expr<Ty>],
    name_span: proc_macro2::Span,
    target: Ty,
    env: &Env,
    backend: BackendChoice,
) -> syn::Result<TokenStream> {
    // Arguments the type check required to be tensors, lowered at their own
    // shape.
    let tensor_arg = |i: usize| -> syn::Result<TokenStream> {
        let arg = &args[i];
        lower(arg, target.element().with_shape(arg.ty.shape), env, backend)
    };
    let host = !backend.is_metal();

    match builtin {
        Builtin::Min | Builtin::Max => {
            min_max(builtin.name(), &args[0], &args[1], target, env, backend)
        }
        Builtin::Clamp => {
            let value = lower(&args[0], target, env, backend)?;
            let low = lower(&args[1], target.element(), env, backend)?;
            let high = lower(&args[2], target.element(), env, backend)?;
            Ok(quote!((#value).clamp(#low, #high)))
        }
        Builtin::Sum => {
            let value = tensor_arg(0)?;
            Ok(quote!((#value).sum()))
        }
        Builtin::Minimum => {
            let value = tensor_arg(0)?;
            Ok(quote!((#value).minimum().expect("math! vectors are non-empty")))
        }
        Builtin::Maximum => {
            let value = tensor_arg(0)?;
            Ok(quote!((#value).maximum().expect("math! vectors are non-empty")))
        }
        Builtin::PrefixSum => {
            let value = tensor_arg(0)?;
            Ok(quote!((#value).prefix_sum()))
        }
        Builtin::Sorted { descending } => {
            let value = tensor_arg(0)?;
            Ok(match (host, descending) {
                (true, true) => quote!((#value).sorted_by(|__a, __b| __b.total_cmp(__a))),
                (true, false) => quote!((#value).sorted_by(|__a, __b| __a.total_cmp(__b))),
                (false, true) => quote!((#value).sorted(
                    ::tensorcrate::tensors::SortOrder::Descending
                )),
                (false, false) => quote!((#value).sorted(
                    ::tensorcrate::tensors::SortOrder::Ascending
                )),
            })
        }
        Builtin::Pow => {
            let (base, exponent) = (&args[0], &args[1]);
            if !target.is_tensor() {
                let base = lower(base, target, env, backend)?;
                let exponent = lower(exponent, target, env, backend)?;
                return Ok(quote!(::tensorcrate::numbers::Power::power(#base, #exponent)));
            }
            // Every order goes through `Power`, which the tensor references
            // implement alongside the numbers. Each has a resident kernel, so a
            // `backend = Metal` block stays on the GPU.
            let element = target.element();
            Ok(match (base.ty.is_tensor(), exponent.ty.is_tensor()) {
                (true, true) => {
                    let base = lower(base, target, env, backend)?;
                    let exponent = lower(exponent, target, env, backend)?;
                    quote!(::tensorcrate::numbers::Power::power(
                        &(#base),
                        &(#exponent)
                    ))
                }
                (true, false) => {
                    let base = lower(base, target, env, backend)?;
                    let exponent = lower(exponent, element, env, backend)?;
                    quote!(::tensorcrate::numbers::Power::power(
                        &(#base),
                        #exponent
                    ))
                }
                _ => {
                    let base = lower(base, element, env, backend)?;
                    let exponent = lower(exponent, target, env, backend)?;
                    quote!(::tensorcrate::numbers::Power::power(
                        #base,
                        &(#exponent)
                    ))
                }
            })
        }
        Builtin::Dot => {
            let (a, b) = (tensor_arg(0)?, tensor_arg(1)?);
            Ok(quote!((#a).dot(&(#b))))
        }
        Builtin::MatMul => {
            let (a, b) = (tensor_arg(0)?, tensor_arg(1)?);
            Ok(match (args[0].ty.shape, args[1].ty.shape) {
                (Shape::Matrix(_, _), Shape::Matrix(_, _)) => quote!((#a).matmul(&(#b))),
                (Shape::Matrix(_, _), Shape::Vector(_)) => quote!((#a).matvec(&(#b))),
                (Shape::Vector(_), Shape::Matrix(_, _)) => quote!((#a).vecmat(&(#b))),
                _ => unreachable!("the type check matched the operand shapes"),
            })
        }
        Builtin::Transpose => {
            let a = tensor_arg(0)?;
            Ok(quote!((#a).transpose()))
        }
        // Inversion and the determinant run on the host.
        Builtin::Inv => {
            let a = tensor_arg(0)?;
            Ok(if host {
                quote!((#a).inverse()?)
            } else {
                quote!((#a)
                    .to_backend::<::tensorcrate::tensors::Host>()
                    .inverse()?
                    .to_backend::<::tensorcrate::tensors::Metal>())
            })
        }
        Builtin::Det => {
            let a = tensor_arg(0)?;
            Ok(if host {
                quote!((#a).determinant())
            } else {
                quote!((#a)
                    .to_backend::<::tensorcrate::tensors::Host>()
                    .determinant())
            })
        }
        Builtin::Conj => {
            let inner = lower(&args[0], target, env, backend)?;
            if !target.dual {
                return Ok(elementwise(
                    quote!(::tensorcrate::__private::conj),
                    inner,
                    target,
                ));
            }
            let conjugate_dual = |value: TokenStream| {
                quote!({
                    let __dual = #value;
                    ::tensorcrate::numbers::Dual::new(
                        ::tensorcrate::__private::conj(__dual.real),
                        ::tensorcrate::__private::conj(__dual.dual),
                    )
                })
            };
            Ok(if target.is_tensor() {
                let conjugated = conjugate_dual(quote!(__x));
                quote!((#inner).map(|&__x| #conjugated))
            } else {
                conjugate_dual(inner)
            })
        }
        Builtin::Analytic(function) => {
            let trait_ident = Ident::new(function.trait_name(), name_span);
            let method = Ident::new(function.name(), name_span);
            let inner = lower(&args[0], target, env, backend)?;
            // A tensor has the function as an inherent method on both backends:
            // the host maps the scalar op over its elements, a resident tensor
            // runs the unary kernel and stays on the GPU.
            Ok(if target.is_tensor() {
                quote!((#inner).#method())
            } else {
                quote!(::tensorcrate::numbers::#trait_ident::#method(#inner))
            })
        }
    }
}

fn min_max(
    operation: &str,
    left: &Expr<Ty>,
    right: &Expr<Ty>,
    target: Ty,
    env: &Env,
    backend: BackendChoice,
) -> syn::Result<TokenStream> {
    let method = Ident::new(operation, left.span);
    let scalar_method = Ident::new(&format!("{operation}_scalar"), left.span);
    Ok(match (left.ty.is_tensor(), right.ty.is_tensor()) {
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
    let zero = backend.dtype.literal(0.0);
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
            assert!(!backend.is_metal(), "Metal values cannot require widening");
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
