# tensorcrate

`tensorcrate` is a scientific-computing library for Rust. It provides dynamically shaped vectors
and matrices, complex and dual numbers, forward- and reverse-mode automatic differentiation,
gradient-based optimizers, constrained minimization, and CPU or Apple Metal execution.

The [`math!`](#the-math-macro) macro is a small, statically typed mathematical language that
expands to ordinary Rust at compile time. The lower-level tensor API uses runtime shapes and is the
API to use for model training, dynamic data, and GPU execution.

> This project is under active development. APIs and the on-disk tensor format may change before
> a stable release.

## Highlights

- Dynamically shaped `Vector<T>` and row-major `Matrix<T>` values.
- Real, complex, dual, and complex-dual arithmetic.
- A `math!` macro with tensor literals, broadcasting, analytic functions, and `@` products.
- Reverse-mode autodiff with a tape and forward-mode autodiff with dual tensors.
- SGD, Momentum/Nesterov, AdaGrad, RMSProp, and Adam.
- Projected gradient descent for box, norm-ball, simplex, or custom constraints.
- Means, variances, and normal or inverse-Gaussian distribution functions, whole-tensor or by axis.
- A generic host backend, Accelerate and SIMD CPU paths, and resident Apple Metal storage.
- Generic element types: autodiff, optimizers, statistics, projections and fused programs run in
  `f32`, `f64`, `f16` or `bf16`, with `f32` the default.
- Saving and loading host tensors without an external serialization framework.

## Requirements and installation

The repository pins the stable Rust toolchain. To work on the crate itself:

```console
git clone https://github.com/arthurmburke/tensorcrate.git
cd tensorcrate
cargo test
```

To use the current Git version from another project:

```toml
[dependencies]
tensorcrate = { git = "https://github.com/arthurmburke/tensorcrate.git" }
```

The default features are `simd` and `metal`. The Metal code is only compiled on macOS; other
platforms continue to use the host backend. Building the Metal feature on macOS requires the Metal
4 compiler supplied with current Xcode because the M5 TensorOps library is compiled by `build.rs`.
The resulting crate still uses the older tiled kernel at runtime on pre-M5 GPUs. To request a
specific configuration:

```toml
# Portable scalar host implementation only.
tensorcrate = { git = "https://github.com/arthurmburke/tensorcrate.git", default-features = false }

# Host implementation with NEON or x86 SIMD.
tensorcrate = { git = "https://github.com/arthurmburke/tensorcrate.git", default-features = false, features = ["simd"] }
```

## Quick start

`Vector<T>` and `Matrix<T>` default to the `Host` backend. Shapes are runtime values and
shape-dependent operations validate them when called.

```rust
use tensorcrate::tensors::{Matrix, Vector};

let a = Matrix::<f32>::from_rows([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
]);
let x = Vector::new([0.5_f32, 1.0, -0.5]);

let y = a.matvec(&x);
assert_eq!(y.to_vec(), [1.0, 4.0]);

let shifted = &y + &Vector::new([2.0, -1.0]);
assert_eq!(shifted.to_vec(), [3.0, 3.0]);
```

In the tensor API, `+`, `-`, `*`, `/`, and `%` are elementwise. Linear-algebra products are
explicit:

- `matrix.matmul(&matrix)`
- `matrix.matvec(&vector)`
- `vector.vecmat(&matrix)`
- `vector.dot(&vector)`
- `vector.outer(&vector)` on differentiable variables

Borrow operands when you want to keep using them: `&a + &b` leaves both tensors intact.

## The `math!` macro

Import the macro and write a block whose last statement is an expression without a trailing
semicolon:

```rust
use tensorcrate::math;
use tensorcrate::numbers::Complex;

let z: Complex<f64> = math! {
    let x = 1 + 2i;
    let y = 1 - 2i;
    x * y
};

assert_eq!(z, Complex::new(5.0, 0.0));
```

The macro is not an interpreter. It infers a concrete type and expands into calls on `f64`,
`Complex<f64>`, `Dual<f64>`, `Vector<_>`, or `Matrix<_>`. Rust then type-checks and optimizes the
expanded code.

### Literals and types

- Unsuffixed numbers are `f64`, or the block's type when it starts with a `dtype` directive:
  `dtype = f32;`, `dtype = f16;` or `dtype = bf16;`.
- `2i` means the imaginary value `2i`.
- `1d` means the dual infinitesimal `1ε`; `d` is used because Rust reserves `e` in numeric
  exponents.
- Mixing axes widens automatically. For example, an expression containing both `i` and `d`
  produces `Dual<Complex<f64>>`.
- `[1, 2, 3]` is a vector and `[[1, 2], [3, 4]]` is a matrix.
- An outer Rust identifier may be referenced in a block and is treated as an `f64` value.

Dual numbers provide a compact way to evaluate a scalar derivative. Seed a variable with a dual
coefficient of one; the result's `dual` field is the derivative in that direction:

```rust
use tensorcrate::math;
use tensorcrate::numbers::Dual;

let y: Dual<f64> = math! {
    let x = 3 + 1d;
    x * x
};

assert_eq!(y.real, 9.0);
assert_eq!(y.dual, 6.0);
```

### Operators and functions

Inside `math!`:

- `+`, `-`, `/`, and `%` are scalar or elementwise tensor operations.
- Tensor/scalar operations broadcast the scalar.
- `A @ B`, `A @ v`, and `v @ A` select matrix-matrix, matrix-vector, and vector-matrix products.
- `v * u` is a dot product when both operands are vectors. This differs from the lower-level Rust
  tensor API, where `*` is elementwise.
- `a .* b` always multiplies elementwise, including when both operands are vectors. `.*`, `*`,
  `/`, and `@` have the same precedence and associate left-to-right.
- Use `pow(base, exponent)` for powers. `^` is Rust's XOR operator and is deliberately rejected.

The elementwise analytic functions are `sin`, `cos`, `tan`, `sec`, `csc`, `arcsin`, `arccos`,
`arctan`, `exp`, `ln`, `sinh`, `cosh`, and `tanh`. The macro also supports `conj`, `pow`, `dot`,
`matmul`, `transpose`, `det`, and `inv`.

Ordering and reduction operations use function syntax:

- `min(a, b)` and `max(a, b)` operate on scalars or elementwise on vectors and matrices; a scalar
  argument broadcasts.
- `clamp(value, low, high)` works on a scalar, vector, or matrix.
- `sum(v)`, `minimum(v)`, and `maximum(v)` reduce a vector to a scalar.
- `prefix_sum(v)` computes an inclusive scan.
- `sorted(v)` sorts ascending. `sorted(v, ascending)` and `sorted(v, descending)` select an
  explicit order.

```rust
use tensorcrate::math;
use tensorcrate::tensors::{Matrix, Vector};

let product: Matrix<f64> = math! {
    let a = [[1, 2, 3], [4, 5, 6]];
    let b = [[7, 8], [9, 10], [11, 12]];
    a @ b
};
assert_eq!(product.to_rows(), [[58.0, 64.0], [139.0, 154.0]]);

let scores: Vector<f64> = math! {
    let x = [0, 1, 2];
    tanh(2 * x - 1)
};

let ranked: Vector<f64> = math! {
    let values = [1, 2, 3];
    let weights = [0.5, 0.25, 2];
    let weighted = values .* weights;
    sorted(clamp(weighted, 0, 4), descending)
};
```

Matrix inversion can fail for a singular matrix, so any `math!` block containing `inv` returns a
`Result`:

```rust
use tensorcrate::math;
use tensorcrate::tensors::Matrix;

let inverse: Matrix<f64> = math! {
    inv([[4, 7], [2, 6]])
}
.expect("matrix should be invertible");
```

`math!` supports `let` bindings and a final expression, but not arbitrary statements, control
flow, assignments, or arbitrary function calls. Literal tensor shapes are checked while the macro
expands.

### Selecting a backend

Blocks use `Host` by default and preserve the original `f64` behavior. Put a backend directive at
the start of a block to make the choice explicit:

```rust
use tensorcrate::math;
use tensorcrate::tensors::{Host, Vector};

let values: Vector<f64, Host> = math! {
    backend = Host;
    max([1, -2, 3], 0)
};
```

On macOS with the `metal` feature, `build.rs` compiles `src/kernel.metal` into a `.metallib` and
embeds that library in the crate. Metal-enabled builds therefore require full Xcode with its Metal
Toolchain component, not only the standalone Command Line Tools. Select Xcode and install the
component if necessary:

```sh
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
xcodebuild -downloadComponent MetalToolchain
```

`backend = Metal;` emits real tensors directly on the Metal backend, `f32` unless a `dtype`
directive picks `f16` or `bf16`. Every tensor created in
the block and every tensor intermediate stays on that backend when the operation has a resident
kernel:

```rust
use tensorcrate::math;
use tensorcrate::tensors::{Metal, Vector};

let values: Vector<f32, Metal> = math! {
    backend = Metal;
    let values = [1, -2, 3];
    let weights = [4, 5, 6];
    sorted(clamp(values .* weights, 0, 10), descending)
};
```

Host blocks compute in `f64` unless a `dtype` directive says otherwise — `f32`, `f16` or `bf16`. It
may come before or after a backend directive, and changes only the coefficient type, so complex and
dual values become `Complex<f32>` and `Dual<f32>`; the 16-bit types are real-only:

```rust
use tensorcrate::math;
use tensorcrate::tensors::{Host, Matrix};

let product: Matrix<f32, Host> = math! {
    dtype = f32;
    [[1, 2], [3, 4]] @ [[0, 1], [1, 0]]
};
```

Metal blocks compute in `f32`, `f16` or `bf16`, the types the shaders are compiled for (so
`dtype = f64;` with `backend = Metal;` is rejected), and reject complex (`i`) and dual (`d`)
literals. A 16-bit block stays in its type end to end: its literals, its tensors and its fused
kernels. `pow`, `det`, and `inv` currently make an explicit Host round trip because they do not
have resident Metal kernels; the result is converted back to Metal when it is a tensor. Use the
regular tensor API for runtime-built shapes and autodiff tapes.

### Fusion

`math!` sees a whole block at once, and every tensor's shape at expansion time, so it fuses
elementwise work into single kernels while it expands the block. No runtime graph is involved:

```rust
use tensorcrate::math;
use tensorcrate::tensors::{Matrix, Metal};

let out: Matrix<f32, Metal> = math! {
    backend = Metal;
    let x = [[1, 2], [3, 4]];
    let w = [[0.5, -1], [2, 0.25]];
    let h = max(x @ w + 1, 0);      // fused into the product's consumer
    let a = h * 2 - 1;              // a and b: one kernel with two outputs
    let b = exp(h) / 10;
    sin(transpose(a @ b)) + 1       // the transpose becomes transposed loads
};
```

- A chain of elementwise operations becomes one kernel: `+ − × ÷ %` between tensors or with a
  scalar, `.*`, negation, the analytic functions, `min`, `max` and `clamp`. Scalars are computed
  once, before the kernel, and passed to it as constants.
- `transpose` inside such a chain is pushed down to the tensors it reads, which are then loaded in
  transposed order instead of being transposed into a new tensor.
- A `let` whose value is elementwise is never materialized when it is only read inside other fused
  chains. If it is read once, it is inlined. If it is read several times, it is inlined only when it
  is cheap: at most four operations and no transcendental functions.
- Consecutive independent elementwise `let`s of the same shape are computed together by one
  kernel that writes all of them.
- Every fused kernel is optimized as the block expands, by the same optimizer that compiles
  runtime programs (see [Fused elementwise programs](#fused-elementwise-programs)): a value
  written twice is computed once, chains are regrouped and their constants folded, and the
  cheapest equivalent program under a cost model is kept.
- On `Host` a fused chain becomes a single loop that LLVM vectorizes, in the block's type. With
  `reassociate = false;` it performs the same operations in the same order as the unfused code, so
  the result is identical bit for bit; by default it may regroup associative chains, and agrees to
  rounding. On `Metal` it becomes one fused program — `f32`, `f16` or `bf16` — which the GPU runs
  as a kernel compiled for it. The macro allocates its registers and splits any chain that would
  exceed the shader's limits, so a block that compiles always fits.
- On `Metal`, a chain that reads a matrix product written inline, such as `tanh(x @ w + b)`, runs
  as that product's epilogue: the product, bias and activation are one dispatch.

Products, reductions, sorts, `pow`, and complex or dual values are not fused; they run as before,
and the fused chains around them read their results. Fusion is on by default.
`fuse = false;` turns it off for one block, and `reassociate = false;` keeps it but forbids
regrouping, for results identical to the unfused block. `fused::with_mode(Mode::Unfused, || ...)` turns it off
at runtime for every block on the current thread. Each block keeps its unfused form for that, which
is also the reference the fused form is tested against.

## Automatic differentiation

Reverse mode records an evaluation on a `Tape`. It is usually the right choice for training: one
backward pass produces gradients for every parameter that contributed to a scalar loss.

```rust
use tensorcrate::tensors::{Tape, Vector};

let tape = Tape::new();
let x = tape.vector(Vector::new([1.0_f32, 2.0, 3.0]));
let loss = x.dot(&x);

loss.backward();
assert_eq!(x.grad().to_vec(), [2.0, 4.0, 6.0]);
```

Create a fresh tape for each optimizer step. A tape represents one evaluation; reusing it across
steps keeps appending graph nodes.

A scalar reduced from a tensor — a `sum`, `dot` or `frobenius_dot`, usually the loss — is not read
until something needs its value: `loss.value()`, or a backward rule whose derivative depends on
it. On `Metal` reading it waits for the GPU, and a step that only calls `loss.backward()` never
does, so training steps queue back to back. Read the loss every few steps rather than every step
when it is only for logging. Forward-mode drivers and dual tensors live in
`tensorcrate::tensors::dual`; they are useful for directional derivatives and functions with few
inputs and many outputs. Reverse-mode gradient and Jacobian helpers live in
`tensorcrate::tensors::tape`.

## Training a model

`minimize` owns the standard single-parameter training loop: it creates a fresh tape, records the
current parameter tensor, evaluates a scalar objective, backpropagates, and applies an optimizer.
The objective also receives the step number, which can be used to select a mini-batch.

This example fits `y = intercept + slope*x` with Adam. The intercept is represented by the first
column of the design matrix, so the model has one parameter vector.

```rust
use tensorcrate::optim::{Adam, minimize};
use tensorcrate::tensors::{Matrix, Vector};

let inputs = Matrix::<f32>::from_rows([
    [1.0, -1.0],
    [1.0, -0.5],
    [1.0,  0.0],
    [1.0,  0.5],
    [1.0,  1.0],
]);
let targets = Vector::new([-1.5_f32, -0.5, 0.5, 1.5, 2.5]);

let mut parameters = Vector::<f32>::zeros(2);
let mut optimizer = Adam::new(0.05);

let final_loss = minimize(
    &mut parameters,
    &mut optimizer,
    1_000,
    |parameters, _step| {
        let tape = parameters.tape();
        let predictions = tape.matrix(inputs.clone()).matvec(parameters);
        let residual = &predictions - &tape.vector(targets.clone());
        residual.dot(&residual).scale(1.0 / 5.0)
    },
);

println!("loss: {final_loss:e}");
println!("intercept: {}, slope: {}", parameters[0], parameters[1]);
```

Available rules are `Sgd`, `Momentum` (including Nesterov), `AdaGrad`, `RmsProp`, and `Adam`. The
rule owns its state and belongs to one parameter tensor. For a model with separate tensors such as
a weight matrix and bias vector, write the four-line tape loop directly and keep one rule instance
per tensor:

```rust
use tensorcrate::optim::{Adam, Rule};
use tensorcrate::tensors::{Matrix, Tape, Vector};

let input = Vector::new([0.25_f32, -0.75]);
let target = Vector::new([1.0_f32, 2.0]);
let mut weights = Matrix::<f32>::identity(2);
let mut bias = Vector::<f32>::zeros(2);
let mut weight_optimizer = Adam::new(0.01);
let mut bias_optimizer = Adam::new(0.01);

for _step in 0..1_000 {
    let tape = Tape::new();
    let w = tape.matrix(weights.clone());
    let b = tape.vector(bias.clone());
    let prediction = &w.matvec(&tape.vector(input.clone())) + &b;
    let residual = &prediction - &tape.vector(target.clone());
    residual.dot(&residual).backward();

    weight_optimizer.update(&mut weights, &w.grad());
    bias_optimizer.update(&mut bias, &b.grad());
}
```

Each rule's update is one fused program, built when the rule is made and run the way
`Program::run` runs one: the gradient is an input, only read, and the parameters and the rule's
state are updated in place. So a tensor gradient can be anything of the parameters' element type a
program reads — the parameters' own type, or a view of a larger matrix (`&grads.view(.., 0..d)`, or
`&grads.column_view(j)` for a vector). A matrix gradient must have the parameters' shape. Code
generic over the parameter type passes `gradient.as_gradient()`.

See `examples/gradient_descent.rs` for larger linear and nonlinear fits and
`examples/optimizers.rs` for mini-batches and multi-parameter training.

## Minimizing with constraints

`Constrained` wraps any optimizer rule and projects the parameters back into a feasible set after
every update. Start from a feasible value if the constraint must hold before the first step.

The following minimizes distance from a target while requiring nonnegative weights whose sum is
at most one:

```rust
use tensorcrate::optim::{Adam, Constrained, minimize};
use tensorcrate::projections::project_onto_capped_simplex;
use tensorcrate::tensors::Vector;

let target = Vector::new([0.8_f32, 0.4, 0.1]);
let mut weights = Vector::new([1.0_f32 / 3.0; 3]);

let mut optimizer = Constrained::new(
    "nonnegative weights with unit budget",
    Adam::new(0.05),
    |weights: &mut Vector<f32>| {
        *weights = project_onto_capped_simplex(weights, 1.0);
    },
);

minimize(&mut weights, &mut optimizer, 500, |weights, _step| {
    let target = weights.tape().vector(target.clone());
    let error = weights - &target;
    error.dot(&error)
});

assert!(weights.data().iter().all(|&weight| weight >= 0.0));
assert!(weights.sum() <= 1.0 + 1e-6);
```

Built-in vector projections are:

| Function | Feasible set |
| --- | --- |
| `project_onto_box(v, low, high)` | Every element is in `[low, high]` |
| `project_onto_ball(v, radius)` | `‖v‖₂ ≤ radius` |
| `project_onto_capped_simplex(v, cap)` | `vᵢ ≥ 0` and `Σvᵢ ≤ cap` |

The projection closure can implement any other constraint and may capture configuration. The
built-in projections are generic over the storage backend, so they work on host and resident Metal
vectors without changing the surrounding optimizer code.

## Statistics

`tensorcrate::statistics` adds moments and distribution functions to the tensor types. Summaries
come in a whole-tensor form and a per-axis form:

```rust
use tensorcrate::statistics::{Axis, Correction};
use tensorcrate::tensors::Matrix;

let observations = Matrix::<f64>::from_rows([
    [1.0, 2.0, 3.0, 4.0],
    [10.0, 12.0, 14.0, 16.0],
]);

assert_eq!(observations.mean(), 7.75);                      // every element
assert_eq!(observations.mean_axis(Axis::Rows).to_vec(), [2.5, 13.0]); // one per row

let spread = observations.stddev_axis(Axis::Columns, Correction::Sample);
assert_eq!(spread.len(), 4);                                // one per column
```

`Axis` names what is *folded*, not what survives: `Axis::Rows` folds each row and leaves one value
per row. Every variance and standard deviation takes a `Correction` - `Population` divides by `n`,
`Sample` by `n − 1`, because neither is a safe default. `moments()` and `moments_axis()` return
the count, mean, and summed squared deviations together, so asking for a mean and both variances
costs one traversal rather than three.

Two distribution families are available, each with a density, a distribution function, and a
quantile:

```rust
use tensorcrate::statistics::Distribution;
use tensorcrate::tensors::Vector;

let standard = Distribution::<f64>::standard_normal();
assert_eq!(standard.cdf(1.96), 0.9750021048517795);
assert_eq!(standard.ppf(0.975), 1.9599639845400536);

// The same functions apply elementwise to a tensor.
let z = Vector::new([-1.0_f64, 0.0, 1.0]);
let probabilities = z.cdf(&standard);
assert_eq!(probabilities.data()[1], 0.5);
assert_eq!(probabilities.ppf(&standard).data()[2].round(), 1.0);
```

`Distribution::InverseGaussian { mean, shape }` is the Wald distribution, the first-passage time
of a drifting Brownian motion. It lives on the positive reals and is right-skewed, which makes it
the counterpart to the normal for durations and latencies.

Fitting a distribution per row or column and mapping the elements through it is the probability
integral transform, which puts rows measured on different scales onto one `[0, 1]` scale:

```rust
use tensorcrate::statistics::{Axis, Correction, Family};
use tensorcrate::tensors::Matrix;

let raw = Matrix::<f64>::from_rows([
    [1.0, 2.0, 3.0, 4.0, 5.0],
    [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
]);

let fits = raw.fit_axis(Axis::Rows, Family::Normal, Correction::Population);
let ranked = raw.cdf_axis(Axis::Rows, &fits);

// Each row is now its own median at the centre, whatever its original scale.
assert!((ranked[(0, 2)] - 0.5).abs() < 1e-12);
assert!((ranked[(1, 2)] - 0.5).abs() < 1e-12);

// `ppf_axis` is the inverse, taking probabilities back to the original units.
let recovered = ranked.ppf_axis(Axis::Rows, &fits);
assert!((recovered[(1, 4)] - 5000.0).abs() < 1e-6);
```

The methods above are inherent on `Host` tensors of any float element type. On `Metal`, and in code
generic over the backend or the element type (`T: Real, B: Kernels<T>`), the same operations come
from the `Statistics` and `AxisStatistics` traits:

```rust
use tensorcrate::statistics::{Axis, AxisStatistics, Correction, Distribution, Statistics};
use tensorcrate::tensors::{Matrix, Metal};

let resident = Matrix::<f32>::from_rows([[1.0, 2.0], [3.0, 4.0]]).to_backend::<Metal>();

let mean = resident.mean();                        // one reduction, on the GPU
let spread = resident.stddev_axis(Axis::Rows, Correction::Sample);
let ranked = resident.cdf(&Distribution::standard_normal());

// Both results are still in GPU-shared memory; nothing came back to the CPU.
assert!(spread.is_device_resident());
assert!(ranked.is_device_resident());
```

A few things worth knowing about the numbers:

- Variances are computed in two passes: mean first, then squared deviations from it, rather than
  through the one-pass `E[x²] − E[x]²` identity, which loses every significant digit when the mean
  is large relative to the spread.
- A column-wise fold accumulates a whole row of partial sums at a time, so it reads the matrix in
  storage order with unit stride on both sides rather than striding down each column.
- The host evaluates the distribution functions in `f64` whatever the tensor's element type, so a
  `f32` tensor gets results correct to the last `f32` bit. The Metal shaders evaluate in `f32`
  throughout, so the two backends agree to about `1e-6` relative rather than exactly. Moments agree
  far more closely, differing only in summation order.
- `statistics::special` exposes the scalar functions underneath: `erf`, `erfc`, `ln_erfc`,
  `erf_inv`, and each family's density, distribution function, and quantile.

## Compute backends

A tensor's second type parameter selects where its values live:

| Configuration | Tensor type | Use it for |
| --- | --- | --- |
| Host, scalar | `Vector<T, Host>` / `Matrix<T, Host>` | Maximum portability and all supported element types |
| Host with `simd` | Same host types | Faster `f32`/`f64` CPU operations on AArch64 and x86-64 |
| Host on macOS | Same host types | Accelerate SGEMM/DGEMM for dispatched `f32`/`f64` products |
| Apple Metal | `Vector<T, Metal>` / `Matrix<T, Metal>` for `f32`, `f16`, `bf16` | Long operation chains that should remain GPU-resident |
| M5 Metal 4 | Same Metal types | TensorOps matrix products for all three types |

`Host` is the default, so `Vector<f32>` means `Vector<f32, Host>`. With the `simd` feature, host
operations select NEON on AArch64 or AVX2/FMA with an SSE2 fallback on x86-64. Small tensors and
unsupported operations use the ordinary scalar implementation automatically. SIMD is an execution
tier of `Host`, not a separate storage type. On macOS, Host `f32` and `f64` matrix products at the
existing 512 multiply-accumulate dispatch threshold use Accelerate SGEMM or DGEMM before the SIMD
path is considered.

Long host operations are split across the CPU's cores: elementwise arithmetic and comparisons,
the analytic functions, transposes, fused programs and the kernels `math!` fuses. A pool of worker
threads starts on first use; a tensor too short to be worth it — about 64K elements for cheap
arithmetic, a few thousand for a transcendental — stays on the calling thread, where it is likely
still in cache. Every split kernel computes each element on its own, so results are the same bit
for bit on any number of cores. Reductions are not split, since a sum's rounding would then depend
on the thread count. On an M5 Max a GELU over 16M `f32`s takes 2.1 ms against 20 ms on one core.

### Element types

The backend-generic layers are written once over any `Real` element type, `f32`, `f64`, `f16` or
`bf16`, through `Kernels<T>`. `Host` implements it for all four; `Metal` implements it for `f32`,
`f16` and `bf16`, with every shader compiled once per type. `T` defaults to `f32`, so `B: Kernels`, `Tape<B>`, `DualVector<B>`,
`Program` and `Sgd` keep meaning what they did, and choosing another type is a matter of what you
put in the tensors:

```rust
use tensorcrate::numbers::Real;
use tensorcrate::optim::{Adam, minimize};
use tensorcrate::tensors::{Kernels, Ordered, Vector};

// Generic over the element as well as the backend.
fn relu<T: Real, B: Kernels<T>>(v: &Vector<T, B>) -> Vector<T, B> {
    v.max_scalar(T::zero())
}
assert_eq!(relu(&Vector::new([-1.0f64, 2.0])).data(), [0.0, 2.0]);

// An `f64` parameter makes the whole step `f64`: the tape, the gradient, Adam's moments and its
// hyperparameters.
let target = Vector::new([0.1f64, 0.2]);
let mut parameters = Vector::<f64>::zeros(2);
let loss = minimize(&mut parameters, &mut Adam::new(0.05), 2000, |x, _| {
    let offset = x - &x.tape().vector(target.clone());
    offset.dot(&offset)
});
assert!(loss < 1e-12);
```

The scalar types `Dual<T>` and `Complex<T>`, the `math!` macro (`dtype = f32;`) and saved tensors
(`f16` and `bf16` included) follow the same element types.

`f16` and `bf16` are computed in, not only stored. Two rules hold on both backends:

- Elementwise operations round to the 16-bit type every time. On the GPU they are `half` and
  `bfloat` shader arithmetic. On AArch64 CPUs, `f16` uses the native FP16 vector instructions
  (`FADD v.8h`, `FSQRT`, `FMAXNM`, ...); `bf16` widens to `f32` exactly, computes, and rounds once
  with `BFCVTN` (M2 and later), which is the correctly rounded `bf16` result.
- Accumulations run in `f32` and round once: sums, dot and matrix products, prefix sums, moments
  and correlations. Dot products use the widening `FMLAL` and `BFMLALB`/`BFMLALT` instructions, and
  CPU matrix products widen once and use Accelerate's SGEMM. A sum of 10,000 `f16` ones is 10,000
  rather than the 2,048 a 16-bit running total stops at.

The elementary functions of `f32`, `f16` and `bf16` (`sin`, `exp`, `ln`, `tanh` and the rest of
`Analytic`) are tensorcrate's own on the host rather than the platform `libm`'s. They are written
so that LLVM vectorizes a whole tensor, which makes them two to six times faster, and they are
within one ulp of the correctly rounded result for `exp`, `ln`, `sin` and `cos`, two for most of
the rest and three for `tan`. The unfused kernels, fused programs, `math!` and the scalar `numbers`
traits all use them, so those paths still agree bit for bit. `f64` keeps `libm`.

```rust
use tensorcrate::numbers::f16;
use tensorcrate::tensors::Vector;

let ones = Vector::new(vec![f16::ONE; 10_000]);
assert_eq!(ones.sum(), f16::from_f32(10_000.0));

// Each addition rounds to f16: 2048 + 1 is 2048.
let big = Vector::new(vec![f16::from_f32(2048.0); 4]);
let one = Vector::new(vec![f16::ONE; 4]);
assert_eq!((&(&big + &one) - &big).data(), [f16::ZERO; 4]);
```

### Apple Metal

On macOS with the `metal` feature, convert `f32` tensors explicitly with `to_backend`. Keep a chain
on `Metal` and convert the final value back when host-owned data is needed:

```rust
use tensorcrate::tensors::{Host, Matrix, Metal};

let host = Matrix::<f32>::from_rows([
    [1.0, 2.0],
    [3.0, 4.0],
]);

let gpu = host.to_backend::<Metal>();       // upload once
let result = gpu.matmul(&gpu).matmul(&gpu); // intermediates stay resident
let result = result.to_backend::<Host>();   // return to host storage

assert_eq!(result, host.matmul(&host).matmul(&host));
```

Important backend details:

- On M5/Apple10 GPUs, `f32` matrix products use Metal 4 TensorOps: plain products, accumulating
  ones (`matmul_add`), and the products with one operand read transposed that reverse mode's
  backward pass needs (`Kernels::matmul_transposed_add`), so no transposed copy is made. Each
  threadgroup computes a tile whose size depends on the output's: a product with a small output
  gets smaller tiles so it still occupies every GPU core, which doubles the speed of a chain of
  products from 256³ to 1024³. Matrix–vector products have kernels of their own. Older GPUs use
  the tiled `f32` kernel.
- Operations are queued, not waited for: they are encoded into a shared command buffer that is
  committed in batches, and the CPU blocks only when it reads a result (about 140 µs for the round
  trip). A chain that stays on the device therefore costs a few microseconds per operation; one
  that reads a value back after every step pays the round trip each time. `to_backend::<Metal>()`
  on a tensor already on Metal is a GPU copy, not a round trip.
- Metal `f16` and `bf16` storage uses the Rust [`half`](https://crates.io/crates/half) crate, whose
  two-byte values map directly to Metal `half` and `bfloat` buffers.
- The full Metal operation set is available for `f32`, `f16` and `bf16`: every shader is compiled
  once per type, and `Kernels<T>` is implemented for each, so autodiff, optimizers, statistics and
  fused programs stay resident in 16-bit precision too.
- Operands in one operation must use the same backend.
- `to_backend` is the explicit transfer boundary; avoid moving back and forth inside a hot loop.
- Optimizers, projections, forward autodiff, and reverse autodiff are generic over `Host` and
  `Metal`. Put the parameters and data on Metal before building the tape.
- `is_device_resident()` reports whether a Metal tensor has a live GPU allocation. If no Metal
  device is available, the Metal backend falls back to CPU storage and preserves the same answers.
- Metal objects are thread-affine and are not `Send`.

Use `matmul` for a compact result, accumulated in FP32 and rounded once per element, or
`matmul_f32` when the product should stay in FP32:

```rust
use tensorcrate::numbers::{bf16, f16};
use tensorcrate::tensors::{Host, Matrix, Metal};

let a = Matrix::<f32>::from_rows([[1.0, 2.0], [3.0, 4.0]]);
let b = Matrix::<f32>::from_rows([[5.0, 6.0], [7.0, 8.0]]);

let a16 = Matrix::<f16, Metal>::from_f32(&a);
let b16 = Matrix::<f16, Metal>::from_f32(&b);
let compact: Matrix<f16, Metal> = a16.matmul(&b16);
let widened: Matrix<f32, Metal> = a16.matmul_f32(&b16);

let abf16 = Matrix::<bf16, Metal>::from_f32(&a);
let bbf16 = Matrix::<bf16, Metal>::from_f32(&b);
let compact_bf16: Matrix<bf16, Metal> = abf16.matmul(&bbf16);

let host_result = widened.to_backend::<Host>();
assert_eq!(host_result.shape(), (2, 2));
```

On a pre-M5 Mac, the compact APIs transparently convert through the Host backend; their return
types and numerical format stay the same.

On M5 the matrix units are much faster when they are allowed to work below full `f32` accuracy.
Large products measured on an M5 Max, queued back to back, in TFLOP/s:

| product                                   | 1024³ | 2048³ | typical error  |
|-------------------------------------------|------:|------:|---------------:|
| `f32`, exact (the default)                |  14.1 |  15.0 |           1e-6 |
| `f32`, `MatmulPrecision::Relaxed`         |  31.6 |  27.2 |           3e-3 |
| `f16` operands, `matmul_f32`              |  43.3 |  49.9 |           3e-3 |

The error column is relative to the size of the results: reduced precision errs in proportion to
the products being summed, so an element whose terms cancel to near zero carries the same absolute
error as its neighbours. Relaxed precision keeps `f32` storage and is opt-in per thread; it applies
to products with a fused epilogue too:

```rust
use tensorcrate::metal::{MatmulPrecision, set_matmul_precision};

set_matmul_precision(MatmulPrecision::Relaxed); // training: speed over the last digits
// ...
set_matmul_precision(MatmulPrecision::Exact);
```

Run the backend comparison on macOS with:

```console
cargo run --release --example metal_backend
```

### Fused elementwise programs

A chain of elementwise operations normally runs one kernel per operation, each
reading its operands from memory and writing a fresh result. `tensors::fused`
compiles such a chain into a register program that runs as **one** kernel on
either backend. Intermediates stay in registers, and tensors can be updated in
place:

```rust
use tensorcrate::tensors::fused::{Builder, DType};
use tensorcrate::tensors::{Analytic, Vector};

// y = sqrt(a·b + 1), in one pass.
let mut b = Builder::new();
let (x, w) = (b.input(DType::F32), b.input(DType::F32));
let product = b.mul(x, w);
let shifted = b.shift(product, 1.0);
let root = b.unary(Analytic::Sqrt, shifted);
b.output(root, DType::F32);
let program = b.build().unwrap();

let y = program.run_vectors(&[&Vector::new([3.0f32, 0.0]), &Vector::new([1.0f32, 5.0])]);
assert_eq!(y[0].as_slice(), [2.0, 1.0]);
```

- `build` optimizes the program. It considers equivalent programs — common subexpressions merged,
  exact identities such as `x · 1 = x` applied, associative chains regrouped (balanced for a
  shorter critical path, or left-deep for fewer registers) with their constants folded, constants
  and loads recomputed where that frees registers, and many instruction orders — and keeps the
  cheapest under the cost model

  ```text
  C = α · instructions + β · peak registers + γ · critical path + δ · memory traffic
    + ε · special operations
  ```

  `fused::CostModel` has presets for the host interpreter and for compiled GPU kernels;
  `Builder::build_with(&model, algebra)` takes any, and `program.cost(&model)` reports a program's
  terms. Plans are cached by program structure, so a program rebuilt every step with new
  constants — an optimizer's — is optimized once.
- Regrouping (`fused::Algebra::Reassociate`, the default) can change results by rounding;
  `fused::with_algebra(Algebra::Exact, || ...)` allows only rewrites that leave every bit as
  written. Either way a program fused and the same program run unfused agree exactly on the host.
- On `Host`, a program runs as a tile interpreter over the existing SIMD kernels, reading inputs
  and writing outputs in place, with ranges of tiles on separate cores for a long tensor; a program
  of a single operation runs as that operation's kernel. On `Metal`, a program seen once runs on a
  bytecode interpreter;
  from its second run it is compiled into a kernel of its own — straight-line code with every
  operation and storage type fixed — and cached. A 9-operation GELU over a million elements takes
  11 µs that way on an M5 Max, against 158 µs interpreted. Like the other shaders these use fast
  math, so Metal agrees with the host within tolerance rather than exactly.
- Loads can read a transposed matrix or broadcast a row or column vector without materializing it.
  Inputs can also be views, read in place through their strides: `m.view(rows, cols)` for a block,
  `m.row_view(i)` and `m.column_view(j)`, `m.transposed_view()`, and `.t()` and `.view(..)` of any
  view. A column of a matrix can be broadcast across the columns, a padded or offset layout read
  without a copy, and on Metal a view costs what a tensor of its size would. Views are read-only:
  in-place tensors (`FusableMut`) are whole vectors and matrices.
- `Builder::row_statistic(x, RowStatistic::Mean | Deviations)` gives a program each row's mean or
  sum of squared deviations of an input, which it computes itself. On the host that is
  `matrix_axis_moments` followed by the program, bit for bit; on `Metal` it is one kernel, a
  threadgroup per row, so a layer norm reads its input once — 11 µs for 1024×1024 on an M5 Max,
  what a single elementwise pass over it costs.
  Each input and output can be stored as `f32`, `f16`, `bf16` or `f64`, independently of the
  arithmetic, which runs in the program's element type: `Program<T>` and `Builder<T>` default to
  `f32`, and `Builder::<f16>::new()` builds one that computes in `f16`. The Metal shader is
  compiled for `f32`, `f16` and `bf16` arithmetic; `f64` programs run on the host.
- The optimizers use fused programs. An `Adam` step is one kernel that updates the parameters and
  both moments in place, where it used to be fourteen kernels and eleven temporary tensors.
- A program can be the epilogue of a matrix product: `program.run_matmul(&a, &b, &[&bias])` feeds
  `a·b` to the program as its input 0, so a dense layer's bias and activation cost nothing beyond
  the product. On `Metal` the product and the program are one dispatch (on TensorOps where the
  GPU has it) and the product is never written to memory; on `Host` the result equals the product
  followed by the program, bit for bit. On `Metal` the epilogue is compiled into the product's
  kernel from a program's second run, like any other program. A `relu(X·W + b)` layer of 512×1024
  by 1024×1024 takes 73 µs this way on an M5 Max, against 147 µs with the epilogue interpreted.
- `program.run_sum(shape, &inputs, axis)` sums a program's one output along each row or down each
  column without keeping it. On the host it is the program followed by the sums; on `Metal` it is
  one kernel, so a softmax is two passes over its input — `exp(x)`'s sums, then `exp(x) / sums` —
  with `exp(x)` never written to memory. A column softmax over 4096×512 takes 32 µs that way on an
  M5 Max, against 188 µs as an `exp` kernel, a product with ones and a fused division.
- For debugging: `println!("{program}")` prints a disassembly, `program.trace(...)` returns every
  intermediate, and `fused::with_mode(Mode::Unfused, || ...)` turns fusion off on the current
  thread, so a suspected fusion problem can be confirmed or ruled out without changing the code.

The `counters` feature counts the kernels, bytes moved and allocations on the current thread
(`tensorcrate::counters`). The `fusion_baseline` example uses it to compare training steps with
and without fusion:

```console
cargo run --release --features counters --example fusion_baseline
```

## Saving trained tensors

Host tensors can be written to a compact, versioned binary format and loaded with shape and type
validation:

```rust
use tensorcrate::tensors::Vector;

fn main() -> Result<(), tensorcrate::errors::Error> {
    let weights = Vector::new([0.5_f32, 2.0]);
    weights.save("weights.tensor")?;

    let restored = Vector::<f32>::load("weights.tensor")?;
    assert_eq!(restored, weights);
    Ok(())
}
```

Persistence is defined for host vectors and matrices whose elements implement `Storable`,
including primitive numeric, complex, and dual values. Move a Metal tensor to `Host` before saving
it.

## Examples and development

```console
cargo run --release --example gradient_descent
cargo run --release --example loss_choice
cargo run --release --example optimizers
cargo run --release --example simd_bench
cargo run --release --example metal_backend  # macOS
cargo run --release --features counters --example fusion_baseline

cargo test
cargo fmt --check
cargo clippy --all-targets --all-features
```

The tests cover tensor algebra, complex and dual arithmetic, both autodiff modes, convolutions,
optimization, persistence, ordering/projections, SIMD, and Metal/Host agreement.

### Benchmarks

`benches/nn_ops.rs` times the operations neural networks are made of — activations, layer norm,
softmax, dense layers, matrix products, reductions, the Adam update, and a whole MLP training
step — and compares the ways the crate can run each one:

- **fused and unfused**: the same fused program with fusion switched on and off, plus the plain
  `Kernels` calls a caller would write without it;
- **scalar, SIMD and Host API**: a plain loop, the architecture kernel called directly, and the
  tensor method, which dispatches through Accelerate and SIMD and allocates its result;
- **Host and Metal**: launch latency (waiting after every call), pipelined launches (waiting once
  per batch), and the cost of uploading and downloading around each call;
- **threads and compilation**: host rows run again on one thread, and Metal fused rows again on the
  bytecode interpreter, to show what splitting across cores and compiling each program buy.

```console
cargo bench --bench nn_ops                       # every size
cargo bench --bench nn_ops -- --quick            # every other size, fewer samples
cargo bench --bench nn_ops -- layer softmax      # only cases whose name contains a word
cargo bench --bench nn_ops --features counters   # add kernels, bytes moved and GPU dispatches
cargo test --release --bench nn_ops              # one quick pass of every case, as a check
```

Each case checks that its variants agree before timing them — fused with unfused bit for bit on
the host, SIMD and Metal within a tolerance — so the one-pass form doubles as a correctness test of
the fast paths. It is `std`-only and needs no benchmarking dependency; the module documentation in
the file explains how to read the tables.
