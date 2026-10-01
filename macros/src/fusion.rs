//! Static kernel fusion for `math!`.
//!
//! A `math!` block is a whole program the macro can see at once, with every
//! tensor's shape known at expansion time, so the fusion decisions that a lazy
//! backend would make at runtime can be made here instead, for free.
//!
//! # What fuses
//!
//! An *elementwise* operation is one whose result element `(i, j)` depends only
//! on element `(i, j)` of its real tensor operands: `+ − × ÷ %` between tensors
//! or with a scalar, `.*`, negation, the analytic functions, `min`, `max` and
//! `clamp`. A tree of them is a *group*, and a group becomes one kernel.
//! Whatever feeds a group and is not elementwise — a literal, a product, a
//! bound tensor — is a *leaf*, materialized as usual and read by the kernel.
//! Scalar subexpressions are evaluated once, before the kernel, and enter it as
//! constants.
//!
//! `transpose` is an index remap rather than arithmetic, and inside a group it
//! is pushed down to the leaves (`(A + B)ᵀ = Aᵀ + Bᵀ`), where it becomes a
//! transposed load. A transpose of a transpose cancels on the way.
//!
//! # Bindings
//!
//! A `let` whose value is elementwise is *inlined* into its consumers — never
//! materialized — when every use is itself inside a group and either there is
//! one use, or the value is cheap enough to recompute at each use (at most
//! [`CHEAP_OPERATIONS`] operations and no transcendental functions: memory
//! traffic costs more than a few additions, but not more than a `tan`).
//! Anything else is materialized once.
//!
//! Consecutive materialized bindings that are each elementwise, have the same
//! shape and do not depend on one another are fused *horizontally*: one kernel
//! computes them all, reading the inputs they share only once.
//!
//! # Code generation
//!
//! On the host a group becomes one closure over the element index, which LLVM
//! vectorizes; it performs exactly the operations the unfused kernels would, in
//! the same order, so its results are identical bit for bit. On Metal a group
//! becomes a register [`Program`](tensorcrate::tensors::fused::Program) for the
//! fused bytecode shader. The macro allocates its registers itself and splits a
//! group that would exceed the shader's register or input limits, so those
//! limits are met by construction rather than checked at runtime.
//!
//! When that program is the only kernel at its site and reads a product of two
//! matrices written inline (`tanh(x @ w + b)`), it runs as the product's
//! epilogue (`Program::run_matmul`): one dispatch, and the product never
//! reaches memory. A product read transposed as well is materialized instead.

use std::collections::{HashMap, HashSet};

use proc_macro2::{Ident, Literal, Span, TokenStream};
use quote::{format_ident, quote};

use crate::ast::{Arith, Binding, Builtin, Expr, ExprKind, Function};
use crate::emit::{Env, lower};
use crate::types::{BackendChoice, Shape, Ty};

/// Operations a value may take and still be recomputed at every use rather
/// than materialized.
const CHEAP_OPERATIONS: usize = 4;

/// The fused shader's limits; see `tensors::fused`.
const REGISTERS: usize = 16;
const MAX_INPUTS: usize = 16;
pub(crate) const MAX_OUTPUTS: usize = 8;
const MAX_INSTRUCTIONS: usize = 256;

/// The bindings fused lowering substitutes into their consumers.
type Inline = HashMap<String, Expr<Ty>>;

// ---- recognizing elementwise operations ----------------------------------------

/// One elementwise operation, with its operands.
enum Kind<'a> {
    /// Parentheses or an invisible group: no operation at all.
    Pass(&'a Expr<Ty>),
    Binary(Arith, &'a Expr<Ty>, &'a Expr<Ty>),
    Neg(&'a Expr<Ty>),
    Unary(Function, &'a Expr<Ty>),
    /// `max` when true.
    MinMax(bool, &'a Expr<Ty>, &'a Expr<Ty>),
    Clamp(&'a Expr<Ty>, &'a Expr<Ty>, &'a Expr<Ty>),
    Transpose(&'a Expr<Ty>),
}

fn is_real_tensor(ty: Ty) -> bool {
    ty.is_tensor() && ty.is_real()
}

/// `expr` as an elementwise operation, if it is one: a real tensor result built
/// by one of the operations a group can hold.
fn kind(expr: &Expr<Ty>) -> Option<Kind<'_>> {
    if !is_real_tensor(expr.ty) {
        return None;
    }
    match &expr.kind {
        ExprKind::Group { inner, .. } => Some(Kind::Pass(inner)),
        ExprKind::Neg(inner) => Some(Kind::Neg(inner)),
        ExprKind::ElementwiseMul(left, right) => Some(Kind::Binary(Arith::Mul, left, right)),
        // A product of two vectors is a dot product, which is not elementwise —
        // but then the result is a scalar, which the tensor check above has
        // already turned away.
        ExprKind::Binary { op, left, right } => Some(Kind::Binary(*op, left, right)),
        ExprKind::Call { builtin, args, .. } => match builtin {
            Builtin::Min => Some(Kind::MinMax(false, &args[0], &args[1])),
            Builtin::Max => Some(Kind::MinMax(true, &args[0], &args[1])),
            Builtin::Clamp => Some(Kind::Clamp(&args[0], &args[1], &args[2])),
            Builtin::Transpose => Some(Kind::Transpose(&args[0])),
            Builtin::Analytic(function) => Some(Kind::Unary(*function, &args[0])),
            _ => None,
        },
        _ => None,
    }
}

/// The operands of an elementwise operation, in order.
fn operands<'a>(kind: &Kind<'a>) -> Vec<&'a Expr<Ty>> {
    match *kind {
        Kind::Pass(e) | Kind::Neg(e) | Kind::Unary(_, e) | Kind::Transpose(e) => vec![e],
        Kind::Binary(_, a, b) | Kind::MinMax(_, a, b) => vec![a, b],
        Kind::Clamp(v, low, high) => vec![v, low, high],
    }
}

/// The value `expr` stands for when it names an inlined binding.
fn inlined<'a>(expr: &Expr<Ty>, inline: &'a Inline) -> Option<&'a Expr<Ty>> {
    match &expr.kind {
        ExprKind::Var(ident) => inline.get(&ident.to_string()),
        _ => None,
    }
}

/// How many operations fusing `expr` would save a kernel for, looking through
/// inlined bindings, and whether any of them is transcendental.
fn weigh(expr: &Expr<Ty>, inline: &Inline) -> (usize, bool) {
    if let Some(value) = inlined(expr, inline) {
        return weigh(value, inline);
    }
    let Some(kind) = kind(expr) else {
        return (0, false);
    };
    let (mut count, mut expensive) = match kind {
        Kind::Pass(_) => (0, false),
        Kind::Unary(..) => (1, true),
        _ => (1, false),
    };
    for operand in operands(&kind) {
        let (c, e) = weigh(operand, inline);
        count += c;
        expensive |= e;
    }
    (count, expensive)
}

// ---- deciding what to inline ------------------------------------------------------

/// Every identifier `expr` reads, function names aside.
fn names(expr: &Expr<Ty>, out: &mut HashSet<String>) {
    if let ExprKind::Var(ident) = &expr.kind {
        out.insert(ident.to_string());
    }
    for child in expr.children() {
        names(child, out);
    }
}

/// Record each use of a binding in `expr`, and whether it sits directly inside
/// an elementwise operation — the only place inlining can put it.
fn uses(expr: &Expr<Ty>, in_group: bool, out: &mut HashMap<String, Vec<bool>>) {
    if let ExprKind::Var(ident) = &expr.kind {
        out.entry(ident.to_string()).or_default().push(in_group);
        return;
    }
    if let Some(kind) = kind(expr) {
        // A real-tensor operand of an elementwise operation can be inlined into
        // it; a scalar operand only ever becomes a constant.
        let inside = match kind {
            Kind::Pass(_) => in_group,
            _ => true,
        };
        for operand in operands(&kind) {
            uses(operand, inside && is_real_tensor(operand.ty), out);
        }
        return;
    }
    // Anything else — a product's operands among them — reads its operands
    // whole, never elementwise.
    for child in expr.children() {
        uses(child, false, out);
    }
}

/// Decide which bindings fused lowering substitutes into their consumers.
///
/// `bindings` are the block's `let`s in order and `result` its final
/// expression.
pub(crate) fn plan_inlining(bindings: &[Binding<Ty>], result: &Expr<Ty>) -> Inline {
    let mut inline = Inline::new();
    for (k, binding) in bindings.iter().enumerate() {
        let (name, init) = (binding.name.to_string(), &binding.value);
        let Some(root) = kind(init) else {
            continue;
        };
        if matches!(root, Kind::Pass(_)) && weigh(init, &Inline::new()).0 == 0 {
            continue;
        }
        // Substitution moves the expression to its uses, so nothing it reads
        // may be rebound in between — and it must be the only binding of its
        // own name, or uses would be ambiguous.
        let mut read = HashSet::new();
        names(init, &mut read);
        let later = &bindings[k + 1..];
        if later.iter().any(|other| {
            let other = other.name.to_string();
            other == name || read.contains(&other)
        }) {
            continue;
        }

        let mut found: HashMap<String, Vec<bool>> = HashMap::new();
        for other in later {
            uses(&other.value, false, &mut found);
        }
        uses(result, false, &mut found);
        let Some(sites) = found.get(&name) else {
            continue; // unused: keep it as written
        };
        if !sites.iter().all(|&in_group| in_group) {
            continue;
        }
        let (operations, expensive) = weigh(init, &inline);
        if sites.len() == 1 || (operations <= CHEAP_OPERATIONS && !expensive) {
            inline.insert(name, init.clone());
        }
    }
    inline
}

// ---- groups --------------------------------------------------------------------------

/// A group's elementwise tree, over indices into its leaves and scalars.
#[derive(Clone, Debug)]
enum Node {
    Leaf(usize),
    Scalar(usize),
    Binary(Arith, Box<Node>, Box<Node>),
    Neg(Box<Node>),
    Unary(Function, Box<Node>),
    MinMax(bool, Box<Node>, Box<Node>),
    /// The value and the scalar indices of its bounds.
    Clamp(Box<Node>, usize, usize),
}

impl Node {
    fn size(&self) -> usize {
        1 + self.children().iter().map(|c| c.size()).sum::<usize>()
    }

    fn children(&self) -> Vec<&Node> {
        match self {
            Node::Leaf(_) | Node::Scalar(_) => vec![],
            Node::Neg(a) | Node::Unary(_, a) | Node::Clamp(a, ..) => vec![a],
            Node::Binary(_, a, b) | Node::MinMax(_, a, b) => vec![a, b],
        }
    }

    fn children_mut(&mut self) -> Vec<&mut Node> {
        match self {
            Node::Leaf(_) | Node::Scalar(_) => vec![],
            Node::Neg(a) | Node::Unary(_, a) | Node::Clamp(a, ..) => vec![a],
            Node::Binary(_, a, b) | Node::MinMax(_, a, b) => vec![a, b],
        }
    }

    fn leaves(&self, out: &mut Vec<usize>) {
        if let Node::Leaf(i) = self
            && !out.contains(i)
        {
            out.push(*i);
        }
        for child in self.children() {
            child.leaves(out);
        }
    }
}

/// A tensor read by a group: lowered once, before any kernel.
struct Leaf {
    /// The identifier bound to a reference to it.
    ident: Ident,
    /// What it is bound to, or `None` for a temporary a split produced.
    value: Option<TokenStream>,
    /// Read transposed: the leaf is stored `cols × rows`.
    transposed: bool,
    /// Its dedup key: the lowered tokens and the orientation.
    key: String,
    /// For a product of two matrices on Metal, its lowered operands: the
    /// group reading it may run as the product's epilogue instead.
    product: Option<(TokenStream, TokenStream)>,
    /// The product runs with the group as its epilogue, so only its operands
    /// are bound.
    epilogue: bool,
}

/// Several groups' worth of shared state: every leaf and scalar is lowered once
/// and may be read by any of the kernels emitted at one site.
struct Site<'e> {
    env: &'e Env,
    backend: BackendChoice,
    leaves: Vec<Leaf>,
    scalars: Vec<TokenStream>,
    /// `clamp` bounds that must be checked before the kernel runs.
    bounds: Vec<(usize, usize)>,
}

impl<'e> Site<'e> {
    fn new(env: &'e Env, backend: BackendChoice) -> Self {
        Site {
            env,
            backend,
            leaves: Vec::new(),
            scalars: Vec::new(),
            bounds: Vec::new(),
        }
    }

    fn scalar(&mut self, expr: &Expr<Ty>) -> syn::Result<Node> {
        let value = lower(expr, Ty::REAL, self.env, self.backend)?;
        self.scalars.push(value);
        Ok(Node::Scalar(self.scalars.len() - 1))
    }

    fn leaf(&mut self, expr: &Expr<Ty>, transposed: bool) -> syn::Result<Node> {
        let value = lower(expr, expr.ty, self.env, self.backend)?;
        let key = format!("{transposed}:{value}");
        if let Some(i) = self.leaves.iter().position(|leaf| leaf.key == key) {
            return Ok(Node::Leaf(i));
        }
        let ident = format_ident!("__fused_in{}", self.leaves.len());
        let product = match &expr.kind {
            ExprKind::MatMul(left, right)
                if self.backend.is_metal()
                    && matches!(
                        (left.ty.shape, right.ty.shape),
                        (Shape::Matrix(..), Shape::Matrix(..))
                    ) =>
            {
                Some((
                    lower(left, left.ty, self.env, self.backend)?,
                    lower(right, right.ty, self.env, self.backend)?,
                ))
            }
            _ => None,
        };
        self.leaves.push(Leaf {
            ident,
            value: Some(value),
            transposed,
            key,
            product,
            epilogue: false,
        });
        Ok(Node::Leaf(self.leaves.len() - 1))
    }

    /// Build the elementwise tree for `expr`, whose elements are read in
    /// transposed order when `transposed` is set.
    fn collect(&mut self, expr: &Expr<Ty>, transposed: bool) -> syn::Result<Node> {
        if !expr.ty.is_tensor() {
            return self.scalar(expr);
        }
        if let Some(value) = inlined(expr, &self.env.inline) {
            return self.collect(value, transposed);
        }
        let Some(kind) = kind(expr) else {
            return self.leaf(expr, transposed);
        };
        Ok(match kind {
            Kind::Pass(inner) => self.collect(inner, transposed)?,
            Kind::Transpose(inner) => self.collect(inner, !transposed)?,
            Kind::Binary(op, a, b) => Node::Binary(
                op,
                Box::new(self.collect(a, transposed)?),
                Box::new(self.collect(b, transposed)?),
            ),
            Kind::Neg(a) => Node::Neg(Box::new(self.collect(a, transposed)?)),
            Kind::Unary(function, a) => {
                Node::Unary(function, Box::new(self.collect(a, transposed)?))
            }
            Kind::MinMax(max, a, b) => Node::MinMax(
                max,
                Box::new(self.collect(a, transposed)?),
                Box::new(self.collect(b, transposed)?),
            ),
            Kind::Clamp(value, low, high) => {
                let value = self.collect(value, transposed)?;
                let (Node::Scalar(low), Node::Scalar(high)) =
                    (self.scalar(low)?, self.scalar(high)?)
                else {
                    unreachable!("scalars collect as scalars")
                };
                self.bounds.push((low, high));
                Node::Clamp(Box::new(value), low, high)
            }
        })
    }

    fn element_type(&self) -> TokenStream {
        self.backend.coefficient_type()
    }
}

/// The result of one group: its tree and the shape it is computed over.
struct Root {
    node: Node,
    shape: Shape,
}

/// Whether `expr` should be lowered as a fused group: an elementwise operation
/// on real tensors that saves at least one kernel.
fn worth_fusing(expr: &Expr<Ty>, target: Ty, env: &Env) -> bool {
    env.fuse
        && is_real_tensor(target)
        && expr.ty == target
        && kind(expr).is_some()
        && weigh(expr, &env.inline).0 >= 2
}

/// Lower `expr` as one fused kernel, or `None` to lower it as written.
pub(crate) fn try_fuse(
    expr: &Expr<Ty>,
    target: Ty,
    env: &Env,
    backend: BackendChoice,
) -> syn::Result<Option<TokenStream>> {
    if !worth_fusing(expr, target, env) {
        return Ok(None);
    }
    let mut site = Site::new(env, backend);
    let node = site.collect(expr, false)?;
    let roots = vec![Root {
        node,
        shape: target.shape,
    }];
    Ok(Some(emit(site, roots)?))
}

/// Lower several bindings as one horizontally fused kernel, or `None` when they
/// do not form one. Each binding is `(name, value)`; the result binds the names
/// in one `let`.
pub(crate) fn try_fuse_horizontally(
    bindings: &[(Ident, &Expr<Ty>)],
    env: &Env,
    backend: BackendChoice,
) -> syn::Result<Option<TokenStream>> {
    if bindings.len() < 2 {
        return Ok(None);
    }
    let mut site = Site::new(env, backend);
    let mut roots = Vec::new();
    for (_, expr) in bindings {
        roots.push(Root {
            node: site.collect(expr, false)?,
            shape: expr.ty.shape,
        });
    }
    if backend.is_metal() {
        let nodes: Vec<&Node> = roots.iter().map(|root| &root.node).collect();
        if allocate(&nodes).is_err() {
            // Splitting would only trade this kernel for several; leave the
            // bindings to fuse one by one.
            return Ok(None);
        }
    }
    let names = bindings.iter().map(|(name, _)| name);
    let types = bindings.iter().map(|(_, expr)| expr.ty.rust_type(backend));
    let value = emit(site, roots)?;
    Ok(Some(quote! { let (#(#names),*): (#(#types),*) = #value; }))
}

/// Whether binding `expr` may join a horizontal group of `shape`: an
/// elementwise operation over that shape that reads none of `earlier`.
pub(crate) fn joins_horizontally(
    expr: &Expr<Ty>,
    shape: Option<Shape>,
    earlier: &HashSet<String>,
    env: &Env,
) -> bool {
    if !env.fuse
        || !is_real_tensor(expr.ty)
        || shape.is_some_and(|shape| shape != expr.ty.shape)
        || kind(expr).is_none()
        || weigh(expr, &env.inline).0 == 0
    {
        return false;
    }
    let mut read = HashSet::new();
    expanded_names(expr, &env.inline, &mut read);
    read.is_disjoint(earlier)
}

/// The names `expr` reads once inlined bindings are substituted.
fn expanded_names(expr: &Expr<Ty>, inline: &Inline, out: &mut HashSet<String>) {
    let mut direct = HashSet::new();
    names(expr, &mut direct);
    for name in direct {
        if let Some(value) = inline.get(&name) {
            expanded_names(value, inline, out);
        }
        out.insert(name);
    }
}

// ---- emission ---------------------------------------------------------------------

fn extent(shape: Shape) -> (usize, usize) {
    match shape {
        Shape::Vector(n) => (1, n),
        Shape::Matrix(r, c) => (r, c),
        Shape::Scalar => unreachable!("groups are tensors"),
    }
}

/// Bind every leaf and scalar, check `clamp` bounds, then run the kernels.
fn emit(mut site: Site<'_>, roots: Vec<Root>) -> syn::Result<TokenStream> {
    let kernels = match site.backend {
        BackendChoice::Metal => emit_metal(&mut site, roots)?,
        BackendChoice::Host | BackendChoice::HostF32 => emit_host(&site, &roots),
    };
    let bind_leaves = site.leaves.iter().filter_map(|leaf| {
        let ident = &leaf.ident;
        if leaf.epilogue {
            let (left, right) = leaf.product.as_ref().expect("an epilogue reads a product");
            let (left_ident, right_ident) = operand_idents(ident);
            return Some(quote!(let #left_ident = &(#left); let #right_ident = &(#right);));
        }
        leaf.value
            .as_ref()
            .map(|value| quote!(let #ident = &(#value);))
    });
    let element = site.element_type();
    let bind_scalars = site.scalars.iter().enumerate().map(|(i, value)| {
        let ident = format_ident!("__fused_k{i}");
        quote!(let #ident: #element = #value;)
    });
    let checks = site.bounds.iter().map(|(low, high)| {
        let (low, high) = (
            format_ident!("__fused_k{low}"),
            format_ident!("__fused_k{high}"),
        );
        quote!(::tensorcrate::__private::assert_clamp_bounds(&#low, &#high);)
    });
    Ok(quote! {{
        #(#bind_leaves)*
        #(#bind_scalars)*
        #(#checks)*
        #kernels
    }})
}

/// The host form: one loop over the elements computing every root, which is
/// the same arithmetic as the unfused kernels in the same order.
fn emit_host(site: &Site<'_>, roots: &[Root]) -> TokenStream {
    let element = site.element_type();
    let (rows, cols) = extent(roots[0].shape);
    let len = rows * cols;

    let mut read = Vec::new();
    for root in roots {
        root.node.leaves(&mut read);
    }
    read.sort_unstable();
    let slices = read.iter().map(|&i| {
        let (ident, slice) = (&site.leaves[i].ident, format_ident!("__fused_s{i}"));
        quote!(let #slice: &[#element] = &#ident.as_slice()[..#len];)
    });
    let loads = read.iter().map(|&i| {
        let (slice, value) = (format_ident!("__fused_s{i}"), format_ident!("__fused_x{i}"));
        if site.leaves[i].transposed {
            quote!(let #value: #element = #slice[(__fused_i % #cols) * #rows + __fused_i / #cols];)
        } else {
            quote!(let #value: #element = #slice[__fused_i];)
        }
    });
    let bytes = (read.len() + roots.len()) * len;
    let size = quote!(::core::mem::size_of::<#element>());
    let record = {
        let outputs = roots.len();
        quote!(::tensorcrate::__private::record_kernel(#bytes * #size, #outputs);)
    };
    let wrap = |values: TokenStream, shape: Shape| match shape {
        Shape::Vector(_) => {
            quote!(::tensorcrate::tensors::Vector::<#element, ::tensorcrate::tensors::Host>::new(#values))
        }
        _ => {
            quote!(::tensorcrate::tensors::Matrix::<#element, ::tensorcrate::tensors::Host>::from_flat(#rows, #cols, #values))
        }
    };

    if let [root] = roots {
        let value = host_expr(&root.node);
        let tensor = wrap(quote!(__fused_out), root.shape);
        return quote! {
            #(#slices)*
            #record
            let __fused_out: ::std::vec::Vec<#element> = (0..#len)
                .map(|__fused_i| {
                    #(#loads)*
                    #value
                })
                .collect();
            #tensor
        };
    }

    let outputs: Vec<Ident> = (0..roots.len())
        .map(|k| format_ident!("__fused_out{k}"))
        .collect();
    let values = roots.iter().map(|root| host_expr(&root.node));
    let tensors = outputs
        .iter()
        .zip(roots)
        .map(|(out, root)| wrap(quote!(#out), root.shape));
    let zero = Literal::f64_unsuffixed(0.0);
    quote! {
        #(#slices)*
        #record
        #(let mut #outputs: ::std::vec::Vec<#element> = ::std::vec![#zero; #len];)*
        for __fused_i in 0..#len {
            #(#loads)*
            #(#outputs[__fused_i] = #values;)*
        }
        (#(#tensors),*)
    }
}

/// One element of a root, as Rust over the loaded leaves and the constants.
fn host_expr(node: &Node) -> TokenStream {
    match node {
        Node::Leaf(i) => {
            let value = format_ident!("__fused_x{i}");
            quote!(#value)
        }
        Node::Scalar(i) => {
            let value = format_ident!("__fused_k{i}");
            quote!(#value)
        }
        Node::Binary(op, a, b) => {
            let (a, b) = (host_expr(a), host_expr(b));
            match op {
                Arith::Add => quote!((#a + #b)),
                Arith::Sub => quote!((#a - #b)),
                Arith::Mul => quote!((#a * #b)),
                Arith::Div => quote!((#a / #b)),
                Arith::Rem => quote!((#a % #b)),
            }
        }
        Node::Neg(a) => {
            let a = host_expr(a);
            quote!((-#a))
        }
        Node::Unary(function, a) => {
            let a = host_expr(a);
            let (trait_ident, method) = (
                Ident::new(function.trait_name(), Span::call_site()),
                Ident::new(function.name(), Span::call_site()),
            );
            // The same scalar function the unfused host kernel maps.
            quote!(::tensorcrate::numbers::#trait_ident::#method(#a))
        }
        Node::MinMax(max, a, b) => {
            let (a, b) = (host_expr(a), host_expr(b));
            if *max {
                quote!((#a).max(#b))
            } else {
                quote!((#a).min(#b))
            }
        }
        Node::Clamp(a, low, high) => {
            let a = host_expr(a);
            let (low, high) = (
                format_ident!("__fused_k{low}"),
                format_ident!("__fused_k{high}"),
            );
            quote!((#a).max(#low).min(#high))
        }
    }
}

// ---- Metal: register allocation -------------------------------------------------------

/// An instruction, before it becomes tokens.
#[derive(Clone, Debug)]
enum Ins {
    Load { dst: u8, input: usize },
    Const { dst: u8, value: Constant },
    Binary { dst: u8, op: Arith, a: u8, b: u8 },
    Unary { dst: u8, function: Function, a: u8 },
    Cmp { dst: u8, max: bool, a: u8, b: u8 },
    Store { src: u8, output: usize },
}

#[derive(Copy, Clone, Debug)]
enum Constant {
    Scalar(usize),
    MinusOne,
}

/// Why a group did not fit one program.
struct DoesNotFit;

/// Linear register allocation over the trees, in Sethi–Ullman order: the
/// operand that needs more registers is evaluated first, and a register is
/// freed after the last instruction that reads it — which may then be the
/// instruction's own destination, as every backend reads operands before it
/// writes.
struct Allocator<'a> {
    inputs: &'a [usize],
    free: Vec<u8>,
    used: usize,
    code: Vec<Ins>,
    /// The register each input is loaded into, once it has been.
    loaded: Vec<Option<u8>>,
    /// Reads of each input still to come.
    remaining: Vec<usize>,
}

/// Who is responsible for freeing an operand's register.
enum Owner {
    Temporary,
    Input(usize),
}

fn count_reads(node: &Node, inputs: &[usize], counts: &mut [usize]) {
    if let Node::Leaf(i) = node {
        counts[inputs.iter().position(|j| j == i).unwrap()] += 1;
    }
    for child in node.children() {
        count_reads(child, inputs, counts);
    }
}

fn need(node: &Node) -> usize {
    match node {
        Node::Leaf(_) | Node::Scalar(_) => 1,
        Node::Neg(a) | Node::Unary(_, a) => need(a),
        Node::Clamp(a, ..) => need(a).max(2),
        Node::Binary(_, a, b) | Node::MinMax(_, a, b) => {
            let (a, b) = (need(a), need(b));
            if a == b { a + 1 } else { a.max(b) }
        }
    }
}

impl Allocator<'_> {
    fn allocate(&mut self) -> Result<u8, DoesNotFit> {
        if let Some(reg) = self.free.pop() {
            return Ok(reg);
        }
        if self.used == REGISTERS {
            return Err(DoesNotFit);
        }
        self.used += 1;
        Ok((self.used - 1) as u8)
    }

    fn release(&mut self, reg: u8, owner: Owner) {
        match owner {
            Owner::Temporary => self.free.push(reg),
            Owner::Input(i) => {
                if self.remaining[i] == 0 {
                    self.free.push(reg);
                }
            }
        }
    }

    fn constant(&mut self, value: Constant) -> Result<(u8, Owner), DoesNotFit> {
        let dst = self.allocate()?;
        self.code.push(Ins::Const { dst, value });
        Ok((dst, Owner::Temporary))
    }

    fn eval(&mut self, node: &Node) -> Result<(u8, Owner), DoesNotFit> {
        match node {
            Node::Leaf(leaf) => {
                let input = self.inputs.iter().position(|j| j == leaf).unwrap();
                self.remaining[input] -= 1;
                let reg = match self.loaded[input] {
                    Some(reg) => reg,
                    None => {
                        let dst = self.allocate()?;
                        self.code.push(Ins::Load { dst, input });
                        self.loaded[input] = Some(dst);
                        dst
                    }
                };
                Ok((reg, Owner::Input(input)))
            }
            Node::Scalar(i) => self.constant(Constant::Scalar(*i)),
            Node::Neg(a) => {
                // `x · −1` flips the sign exactly, zeros included, where
                // `0 − x` would turn `+0` into `+0`.
                let a = self.eval(a)?;
                let minus_one = self.constant(Constant::MinusOne)?;
                self.combine(a, minus_one, |dst, a, b| Ins::Binary {
                    dst,
                    op: Arith::Mul,
                    a,
                    b,
                })
            }
            Node::Unary(function, a) => {
                let (a, owner) = self.eval(a)?;
                self.release(a, owner);
                let dst = self.allocate()?;
                self.code.push(Ins::Unary {
                    dst,
                    function: *function,
                    a,
                });
                Ok((dst, Owner::Temporary))
            }
            Node::Clamp(a, low, high) => {
                // `x.max(low).min(high)`, as the clamp kernel computes it.
                let a = self.eval(a)?;
                let low = self.constant(Constant::Scalar(*low))?;
                let raised = self.combine(a, low, |dst, a, b| Ins::Cmp {
                    dst,
                    max: true,
                    a,
                    b,
                })?;
                let high = self.constant(Constant::Scalar(*high))?;
                self.combine(raised, high, |dst, a, b| Ins::Cmp {
                    dst,
                    max: false,
                    a,
                    b,
                })
            }
            Node::Binary(op, a, b) => {
                let (a, b) = self.both(a, b)?;
                let op = *op;
                self.combine(a, b, |dst, a, b| Ins::Binary { dst, op, a, b })
            }
            Node::MinMax(max, a, b) => {
                let (a, b) = self.both(a, b)?;
                let max = *max;
                self.combine(a, b, |dst, a, b| Ins::Cmp { dst, max, a, b })
            }
        }
    }

    /// Evaluate two operands, the hungrier first, returning them in order.
    #[allow(clippy::type_complexity)]
    fn both(&mut self, a: &Node, b: &Node) -> Result<((u8, Owner), (u8, Owner)), DoesNotFit> {
        if need(b) > need(a) {
            let b = self.eval(b)?;
            let a = self.eval(a)?;
            Ok((a, b))
        } else {
            let a = self.eval(a)?;
            let b = self.eval(b)?;
            Ok((a, b))
        }
    }

    fn combine(
        &mut self,
        (a, a_owner): (u8, Owner),
        (b, b_owner): (u8, Owner),
        instruction: impl FnOnce(u8, u8, u8) -> Ins,
    ) -> Result<(u8, Owner), DoesNotFit> {
        self.release(a, a_owner);
        if b != a {
            self.release(b, b_owner);
        }
        let dst = self.allocate()?;
        self.code.push(instruction(dst, a, b));
        Ok((dst, Owner::Temporary))
    }
}

/// Allocate one program computing every root, or report that it does not fit.
fn allocate(roots: &[&Node]) -> Result<(Vec<usize>, Vec<Ins>), DoesNotFit> {
    let mut inputs = Vec::new();
    for root in roots {
        root.leaves(&mut inputs);
    }
    if inputs.len() > MAX_INPUTS || roots.len() > MAX_OUTPUTS {
        return Err(DoesNotFit);
    }
    let mut remaining = vec![0; inputs.len()];
    for root in roots {
        count_reads(root, &inputs, &mut remaining);
    }
    let mut allocator = Allocator {
        inputs: &inputs,
        free: Vec::new(),
        used: 0,
        code: Vec::new(),
        loaded: vec![None; inputs.len()],
        remaining,
    };
    for (output, root) in roots.iter().enumerate() {
        let (src, owner) = allocator.eval(root)?;
        allocator.code.push(Ins::Store { src, output });
        allocator.release(src, owner);
    }
    let code = allocator.code;
    if code.len() > MAX_INSTRUCTIONS {
        return Err(DoesNotFit);
    }
    Ok((inputs, code))
}

/// The largest proper subtree of any root, by size, as a path of child
/// indices from that root.
fn largest_subtree(roots: &[Root]) -> Option<(usize, Vec<usize>)> {
    fn search(node: &Node, path: &mut Vec<usize>, best: &mut Option<(usize, Vec<usize>)>) {
        for (i, child) in node.children().into_iter().enumerate() {
            path.push(i);
            let size = child.size();
            if size > 1 && best.as_ref().is_none_or(|(s, _)| size > *s) {
                *best = Some((size, path.clone()));
            }
            search(child, path, best);
            path.pop();
        }
    }
    let mut best: Option<(usize, usize, Vec<usize>)> = None;
    for (r, root) in roots.iter().enumerate() {
        let mut found = None;
        search(&root.node, &mut Vec::new(), &mut found);
        if let Some((size, path)) = found
            && best.as_ref().is_none_or(|(s, ..)| size > *s)
        {
            best = Some((size, r, path));
        }
    }
    best.map(|(_, r, path)| (r, path))
}

/// The Metal form: the macro allocates the program, splitting off subtrees
/// into kernels of their own until every program fits the shader.
fn emit_metal(site: &mut Site<'_>, mut roots: Vec<Root>) -> syn::Result<TokenStream> {
    let mut kernels = TokenStream::new();
    let shape = roots[0].shape;
    loop {
        let nodes: Vec<&Node> = roots.iter().map(|root| &root.node).collect();
        if let Ok((mut inputs, mut code)) = allocate(&nodes) {
            // A program that is the site's only kernel can be the epilogue of
            // a product it reads, which then never reaches memory.
            let epilogue = kernels.is_empty() && as_epilogue(site, shape, &mut inputs, &mut code);
            let outputs: Vec<Shape> = roots.iter().map(|root| root.shape).collect();
            let run = metal_program(site, &inputs, &code, shape, &outputs, epilogue);
            return Ok(quote! { #kernels #run });
        }
        // Too big for one program: materialize the largest subtree as a kernel
        // of its own and read its result as a leaf.
        let Some((root, path)) = largest_subtree(&roots) else {
            return Err(syn::Error::new(
                Span::call_site(),
                "math! cannot fit this elementwise expression in a fused Metal kernel",
            ));
        };
        let mut node = &mut roots[root].node;
        for &step in &path {
            node = node.children_mut().into_iter().nth(step).unwrap();
        }
        let ident = format_ident!("__fused_t{}", site.leaves.len());
        let subtree = std::mem::replace(node, Node::Leaf(site.leaves.len()));
        site.leaves.push(Leaf {
            ident: ident.clone(),
            value: None,
            transposed: false,
            key: ident.to_string(),
            product: None,
            epilogue: false,
        });
        let sub = emit_metal(
            site,
            vec![Root {
                node: subtree,
                shape,
            }],
        )?;
        kernels.extend(quote! {
            let #ident = { #sub };
            let #ident = &#ident;
        });
    }
}

/// The identifiers a product's two operands are bound to.
fn operand_idents(ident: &Ident) -> (Ident, Ident) {
    (
        format_ident!("{ident}_left"),
        format_ident!("{ident}_right"),
    )
}

/// Make the program the epilogue of a matrix product it reads, if it reads
/// one untransposed over the program's own shape and no other kernel needs it:
/// that leaf moves to input slot 0, where `run_matmul` supplies the product.
fn as_epilogue(site: &mut Site<'_>, shape: Shape, inputs: &mut [usize], code: &mut [Ins]) -> bool {
    if !matches!(shape, Shape::Matrix(..)) {
        return false;
    }
    let Some(slot) = inputs.iter().position(|&leaf| {
        let leaf = &site.leaves[leaf];
        leaf.product.is_some() && !leaf.transposed && !leaf.epilogue
    }) else {
        return false;
    };
    // A product read both straight and transposed is two leaves; the
    // transposed read still needs it materialized.
    let key = site.leaves[inputs[slot]]
        .key
        .split_once(':')
        .map(|(_, value)| value.to_owned());
    let read_twice = site
        .leaves
        .iter()
        .filter(|leaf| leaf.key.split_once(':').map(|(_, value)| value.to_owned()) == key)
        .count()
        > 1;
    if read_twice {
        return false;
    }
    inputs.swap(0, slot);
    for ins in code.iter_mut() {
        if let Ins::Load { input, .. } = ins {
            if *input == 0 {
                *input = slot;
            } else if *input == slot {
                *input = 0;
            }
        }
    }
    site.leaves[inputs[0]].epilogue = true;
    true
}

/// Tokens running one allocated program and unpacking its outputs.
fn metal_program(
    site: &Site<'_>,
    inputs: &[usize],
    code: &[Ins],
    shape: Shape,
    outputs: &[Shape],
    epilogue: bool,
) -> TokenStream {
    let fused = quote!(::tensorcrate::tensors::fused);
    let instructions = code.iter().map(|ins| match *ins {
        Ins::Load { dst, input } => {
            let remap = if site.leaves[inputs[input]].transposed {
                quote!(Transpose)
            } else {
                quote!(Identity)
            };
            let input = input as u8;
            quote!(#fused::Instr::<f32>::Load { dst: #dst, input: #input, remap: #fused::Remap::#remap })
        }
        Ins::Const { dst, value } => {
            let value = match value {
                Constant::Scalar(i) => {
                    let ident = format_ident!("__fused_k{i}");
                    quote!(#ident)
                }
                Constant::MinusOne => quote!(-1.0f32),
            };
            quote!(#fused::Instr::<f32>::Const { dst: #dst, value: #value })
        }
        Ins::Binary { dst, op, a, b } => {
            let op = match op {
                Arith::Add => quote!(Add),
                Arith::Sub => quote!(Sub),
                Arith::Mul => quote!(Mul),
                Arith::Div => quote!(Div),
                Arith::Rem => quote!(Rem),
            };
            quote!(#fused::Instr::<f32>::Binary { dst: #dst, op: ::tensorcrate::tensors::BinaryOp::#op, a: #a, b: #b })
        }
        Ins::Unary { dst, function, a } => {
            let variant = Ident::new(function.trait_name(), Span::call_site());
            quote!(#fused::Instr::<f32>::Unary { dst: #dst, op: ::tensorcrate::tensors::Analytic::#variant, a: #a })
        }
        Ins::Cmp { dst, max, a, b } => {
            let op = if max { quote!(Max) } else { quote!(Min) };
            quote!(#fused::Instr::<f32>::Cmp { dst: #dst, op: ::tensorcrate::tensors::Compare::#op, a: #a, b: #b })
        }
        Ins::Store { src, output } => {
            let output = output as u8;
            quote!(#fused::Instr::<f32>::Store { src: #src, output: #output })
        }
    });
    let (rows, cols) = extent(shape);
    let (input_count, output_count) = (inputs.len(), outputs.len());
    let operands = inputs.iter().skip(usize::from(epilogue)).map(|&i| {
        let ident = &site.leaves[i].ident;
        quote!(#ident as &dyn #fused::Fusable<::tensorcrate::tensors::Metal>)
    });
    let unpack = outputs.iter().map(|shape| {
        let into = match shape {
            Shape::Vector(_) => quote!(into_vector::<f32>),
            _ => quote!(into_matrix::<f32>),
        };
        quote!(__fused_outputs.next().expect("one output per root").#into())
    });
    let result = if outputs.len() == 1 {
        quote!(#(#unpack)*)
    } else {
        quote!((#(#unpack),*))
    };
    let run = if epilogue {
        let (left, right) = operand_idents(&site.leaves[inputs[0]].ident);
        quote!(__fused_program.run_matmul::<::tensorcrate::tensors::Metal>(#left, #right, &[#(#operands),*]))
    } else {
        quote!(__fused_program.run::<::tensorcrate::tensors::Metal>((#rows, #cols), &[#(#operands),*], &mut []))
    };
    quote! {
        let __fused_program = #fused::Program::<f32>::new(
            ::std::vec![#(#instructions),*],
            ::std::vec![#fused::DType::F32; #input_count],
            ::std::vec![#fused::DType::F32; #output_count],
            0,
        )
        .expect("math! allocates programs that fit the fused kernel");
        let mut __fused_outputs = #run.into_iter();
        #result
    }
}
