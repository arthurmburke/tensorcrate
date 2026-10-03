# Repository Guidance

## Algebraic Operations

When adding an algebraic operation:

- Implement fused kernels for the operation.
- Implement the algebraic trait for every appropriate mathematical structure, including `Vector`, `Matrix`, `Tensor`, and `Coefficient`.
- Add support for the operation in `math!`.

## Tests and Benchmarks

- Add unit tests for new behavior.
- Add benchmarks for new behavior.
- Document any performance regression.

## Git and Worktrees

- Always work in a separate Git worktree.
- Do not add `Co-Authored-By` trailers to commits.

## Documentation Style

- Keep documentation concise.
- Avoid run-on sentences.
- Avoid excessive hyphenation and em dashes.

## Validation

Before handing off a change, run:

- `cargo test`
- `cargo fmt --check`
- `cargo clippy --all-targets --all-features`

For performance-sensitive changes, also run:

- `cargo bench --bench nn_ops`
- `cargo test --release --bench nn_ops`

Test relevant feature combinations, including `--no-default-features`, `--features simd`, and Metal builds on macOS.

## Backend Consistency

When changing tensor operations, keep the host, SIMD, and Metal implementations consistent.

Changes involving Metal must update and test both the Rust dispatch code and the corresponding `.metal` shaders.

Changes involving fusion must cover:

- The fused operation.
- The unfused kernel path.
- `math!` expansion.
- Host and Metal behavior where supported.

## Repository Areas

- `src/tensors/`: tensor types, operations, algebraic traits, kernels, and fusion.
- `src/numbers/`: numeric and coefficient types.
- `src/metal/` and `metal/`: Metal backend and shaders.
- `macros/`: `math!` parsing and expansion.
- `fusion/`: fusion graph optimization and scheduling.
- `tests/`: integration and regression tests.
- `benches/`: performance benchmarks.

## Performance Changes

Record benchmark results for performance-sensitive changes. Document meaningful regressions and explain whether they are expected, accepted, or unresolved.

## Compatibility

Preserve shape validation, broadcasting behavior, backend behavior, supported element types, and documented numerical tolerances unless the change explicitly updates the public contract.
