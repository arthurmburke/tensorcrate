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
//! `dtype = f64;`, `f32`, `f16` or `bf16` chooses the coefficient type. The
//! directives may come in any order; Metal's shaders compute in `f32`, `f16` and
//! `bf16`, so `backend = Metal;` with `dtype = f64;` is an error, and the 16-bit
//! types are real-only.
//!
//! Chains of elementwise tensor operations are fused into single kernels while
//! the block expands — see [`fusion`] — and each kernel is optimized by the
//! shared `tensorcrate-fusion` crate. A leading `fuse = false;` turns fusion off
//! for the block; `reassociate = false;` keeps it but forbids regrouping
//! associative chains, so results match the unfused block bit for bit.
//!
//! Expansion runs in three stages around one tree, `ast`:
//!
//! 1. `parse` reads the tokens into a block of bindings and expressions,
//!    resolving each function call to a `Builtin` and reporting every syntax
//!    error;
//! 2. `types` checks the block and annotates every expression with its type
//!    and shape, reporting every type error;
//! 3. `emit` (and `fusion` for the fused form) generate Rust from the typed
//!    tree, which no longer needs checking.

use proc_macro2::TokenStream;
use quote::quote;

mod ast;
mod emit;
mod fusion;
mod parse;
mod types;

use parse::rewrite_custom_operators;

#[proc_macro]
pub fn math(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    match expand(rewrite_custom_operators(input.into())) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Expand a block whose `@` and `.*` operators are already rewritten: parse it,
/// check its types, then generate the code.
fn expand(input: TokenStream) -> syn::Result<TokenStream> {
    let (directives, source) = parse::block(input)?;
    let backend = directives.backend;
    let block = types::check_block(&source, backend)?;
    let annotated = block.result.ty.rust_type(backend);

    let unfused = emit::block(&block, backend, None, directives.reassociate)?;
    // The block once as written and once fused. Both are emitted when they
    // differ, so `fused::Mode::Unfused` can still select the original at
    // runtime — the exact reference fusion is checked against, and a way to
    // rule fusion in or out when debugging.
    let body = if directives.fuse {
        let inline = fusion::plan_inlining(&block.bindings, &block.result);
        let fused = emit::block(&block, backend, Some(inline), directives.reassociate)?;
        if fused.to_string() == unfused.to_string() {
            unfused
        } else {
            quote! {
                if ::tensorcrate::tensors::fused::mode()
                    == ::tensorcrate::tensors::fused::Mode::Unfused
                {
                    #unfused
                } else {
                    #fused
                }
            }
        }
    } else {
        unfused
    };

    // `inv` fails on a singular matrix, so a block that calls it returns a
    // `Result` and its `?`s need a function to return from.
    let fallible = std::iter::once(&block.result)
        .chain(block.bindings.iter().map(|binding| &binding.value))
        .any(|expr| {
            expr.any(&|e| {
                matches!(
                    e.kind,
                    ast::ExprKind::Call {
                        builtin: ast::Builtin::Inv,
                        ..
                    }
                )
            })
        });
    Ok(if fallible {
        quote! {
            (|| -> ::core::result::Result<#annotated, ::tensorcrate::errors::Error> {
                let __math_result: #annotated = #body;
                ::core::result::Result::Ok(__math_result)
            })()
        }
    } else {
        quote! {
            {
                let __math_result: #annotated = #body;
                __math_result
            }
        }
    })
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
                .contains("Metal backend supports real values only")
        );
    }

    #[test]
    fn static_kernels_are_optimized() {
        // `exp(x)` is written twice and computed once.
        let shared = expansion(quote! { backend = Metal; let x = [1, 2]; exp(x) * 2 + exp(x) / 3 });
        assert_eq!(shared.matches("Analytic :: Exp").count(), 1, "{shared}");

        // `x · 2 · 3` multiplies by one folded constant when chains may be
        // regrouped, and twice when they may not.
        // Counted among the program's instructions, not the unfused fallback.
        let multiplies = |code: &str| {
            code.matches("op : :: tensorcrate :: tensors :: BinaryOp :: Mul")
                .count()
        };
        let folded = expansion(quote! { backend = Metal; let x = [1, 2]; x * 2 * 3 + x });
        assert_eq!(multiplies(&folded), 1, "{folded}");
        let exact = expansion(
            quote! { backend = Metal; reassociate = false; let x = [1, 2]; x * 2 * 3 + x },
        );
        assert_eq!(multiplies(&exact), 2, "{exact}");

        // On the host the folded constant is computed once, before the loop.
        let host = expansion(quote! { let x = [1, 2]; x * 2 * 3 + x });
        assert!(host.contains("__fused_c"), "{host}");
    }

    #[test]
    fn dtype_directives_are_validated() {
        assert!(
            expansion_error(quote! { dtype = f8; [1, 2] })
                .contains("dtype must be `f64`, `f32`, `f16` or `bf16`")
        );
        assert!(
            expansion_error(quote! { backend = Metal; dtype = f64; [1, 2] })
                .contains("Metal backend computes in f32, f16 or bf16")
        );
        assert!(
            expansion_error(quote! { dtype = bf16; [1 + 1i] })
                .contains("complex and dual values need")
        );
        // Every float type expands, on either backend where it exists.
        for dtype in [quote!(f64), quote!(f32), quote!(f16), quote!(bf16)] {
            expansion(quote! { dtype = #dtype; let a = [1, 2]; sin(a * 2 + 1) - a });
        }
        for dtype in [quote!(f32), quote!(f16), quote!(bf16)] {
            let metal = expansion(
                quote! { backend = Metal; dtype = #dtype; let a = [1, 2]; sin(a * 2 + 1) - a },
            );
            assert!(metal.contains("Program"), "{metal}");
        }
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

    fn expansion(input: TokenStream) -> String {
        expand(rewrite_custom_operators(input))
            .expect("the macro input should expand")
            .to_string()
    }

    #[test]
    fn fusion_directives_are_validated() {
        assert!(expansion_error(quote! { fuse = maybe; [1, 2] }).contains("`true` or `false`"));
        assert!(
            expansion_error(quote! { reassociate = 1; [1, 2] }).contains("reassociate must be")
        );
        assert!(
            expansion_error(quote! { reassociate = true; reassociate = false; [1, 2] })
                .contains("set twice")
        );
        assert!(
            expansion_error(quote! { fuse = true; fuse = false; [1, 2] }).contains("set twice")
        );
    }

    #[test]
    fn a_chain_of_elementwise_operations_becomes_one_kernel() {
        let host = expansion(quote! { let a = [1, 2]; sin(a * 2 + 1) - a });
        assert!(host.contains("record_kernel"), "{host}");
        // The unfused block is kept for `Mode::Unfused`.
        assert!(host.contains("Mode :: Unfused"), "{host}");

        let metal = expansion(quote! { backend = Metal; let a = [1, 2]; sin(a * 2 + 1) - a });
        assert_eq!(
            metal.matches("Program :: < f32 > :: new").count(),
            1,
            "{metal}"
        );
    }

    #[test]
    fn a_single_operation_is_left_alone() {
        // One kernel either way, and the existing kernel is already vectorized.
        let single = expansion(quote! { [1, 2] + [3, 4] });
        assert!(!single.contains("record_kernel"), "{single}");
        assert!(!single.contains("Mode :: Unfused"), "{single}");
    }

    #[test]
    fn fusion_can_be_switched_off_per_block() {
        let off = expansion(quote! { fuse = false; let a = [1, 2]; sin(a * 2 + 1) - a });
        assert!(!off.contains("record_kernel"), "{off}");
        assert!(!off.contains("Mode :: Unfused"), "{off}");
    }

    #[test]
    fn single_use_elementwise_bindings_are_not_materialized() {
        let fused = expansion(quote! { let a = [1, 2]; let b = a * 2; let c = b + 1; c - a });
        let fused_half = fused.split("else").nth(1).expect("a fused branch");
        assert!(!fused_half.contains("let b"), "{fused_half}");
        assert!(!fused_half.contains("let c"), "{fused_half}");
    }

    #[test]
    fn expensive_shared_bindings_are_materialized_once() {
        let fused = expansion(quote! { let a = [1, 2]; let e = exp(a) * 3; e .* a + e });
        let fused_half = fused.split("else").nth(1).expect("a fused branch");
        assert!(fused_half.contains("let e"), "{fused_half}");
        // ...but cheap ones are recomputed at each use.
        let fused = expansion(quote! { let a = [1, 2]; let c = a * 3; c .* a + c });
        let fused_half = fused.split("else").nth(1).expect("a fused branch");
        assert!(!fused_half.contains("let c"), "{fused_half}");
    }

    #[test]
    fn independent_bindings_fuse_horizontally() {
        let fused = expansion(quote! {
            backend = Metal;
            let x = [[1, 2], [3, 4]];
            let a = x * 2 + 1;
            let b = exp(x) - 1;
            a @ b
        });
        assert_eq!(
            fused.matches("Program :: < f32 > :: new").count(),
            1,
            "{fused}"
        );
        assert!(fused.contains("let (a , b)"), "{fused}");
    }

    #[test]
    fn programs_over_the_input_limit_are_split() {
        // Twenty distinct inputs cannot be one Metal program.
        let fused = expansion(quote! {
            backend = Metal;
            [1] + [2] + [3] + [4] + [5] + [6] + [7] + [8] + [9] + [10]
                + [11] + [12] + [13] + [14] + [15] + [16] + [17] + [18] + [19] + [20]
        });
        assert!(
            fused.matches("Program :: < f32 > :: new").count() >= 2,
            "{fused}"
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
