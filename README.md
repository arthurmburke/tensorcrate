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
- A generic host backend, Accelerate and SIMD CPU paths, and resident Apple Metal storage.
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

- Unsuffixed numbers are `f64`.
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

`backend = Metal;` emits real `f32` tensors directly on the Metal backend. Every tensor created in
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

Metal blocks use `f32` because the shaders are 32-bit and reject complex (`i`) and dual (`d`)
literals. `pow`, `det`, and `inv` currently make an explicit Host round trip because they do not
have resident Metal kernels; the result is converted back to Metal when it is a tensor. Use the
regular tensor API for runtime-built shapes and autodiff tapes.

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
steps keeps appending graph nodes. Forward-mode drivers and dual tensors live in
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

## Compute backends

A tensor's second type parameter selects where its values live:

| Configuration | Tensor type | Use it for |
| --- | --- | --- |
| Host, scalar | `Vector<T, Host>` / `Matrix<T, Host>` | Maximum portability and all supported element types |
| Host with `simd` | Same host types | Faster `f32`/`f64` CPU operations on AArch64 and x86-64 |
| Host on macOS | Same host types | Accelerate SGEMM/DGEMM for dispatched `f32`/`f64` products |
| Apple Metal | `Vector<f32, Metal>` / `Matrix<f32, Metal>` | Long `f32` operation chains that should remain GPU-resident |
| M5 Metal 4 | `Matrix<f16, Metal>` / `Matrix<bf16, Metal>` | Compact TensorOps matrix products |

`Host` is the default, so `Vector<f32>` means `Vector<f32, Host>`. With the `simd` feature, host
operations select NEON on AArch64 or AVX2/FMA with an SSE2 fallback on x86-64. Small tensors and
unsupported operations use the ordinary scalar implementation automatically. SIMD is an execution
tier of `Host`, not a separate storage type. On macOS, Host `f32` and `f64` matrix products at the
existing 512 multiply-accumulate dispatch threshold use Accelerate SGEMM or DGEMM before the SIMD
path is considered.

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

- On M5/Apple10 GPUs, ordinary `f32` matrix multiplication uses Metal 4 TensorOps. Older GPUs use
  the existing tiled `f32` kernel. Fused `matmul_add` also retains the tiled path.
- Metal `f16` and `bf16` storage uses the Rust [`half`](https://crates.io/crates/half) crate, whose
  two-byte values map directly to Metal `half` and `bfloat` buffers.
- The full Metal operation set is available for `f32`. Compact types currently provide storage,
  FP32 conversion, and matrix multiplication.
- Operands in one operation must use the same backend.
- `to_backend` is the explicit transfer boundary; avoid moving back and forth inside a hot loop.
- Optimizers, projections, forward autodiff, and reverse autodiff are generic over `Host` and
  `Metal`. Put the parameters and data on Metal before building the tape.
- `is_device_resident()` reports whether a Metal tensor has a live GPU allocation. If no Metal
  device is available, the Metal backend falls back to CPU storage and preserves the same answers.
- Metal objects are thread-affine and are not `Send`.

Use `matmul` when compact output is important, or `matmul_f32` when the product should accumulate
and remain in FP32:

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

Run the backend comparison on macOS with:

```console
cargo run --release --example metal_backend
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

cargo test
cargo fmt --check
cargo clippy --all-targets --all-features
```

The tests cover tensor algebra, complex and dual arithmetic, both autodiff modes, convolutions,
optimization, persistence, ordering/projections, SIMD, and Metal/Host agreement.
