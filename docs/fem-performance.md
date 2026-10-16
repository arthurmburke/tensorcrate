# FEM operator performance

Measured on macOS with an Apple GPU on 2026-10-15. Default features enable
SIMD and Metal. These are warmed wall-clock medians, not hardware guarantees.
CPU and GPU are selected explicitly; operators do not migrate automatically.

Run:

```console
cargo bench --bench nn_ops
cargo bench --bench nn_ops -- fem
cargo test --release --bench nn_ops
```

The assembly benchmark uses overlapping two-DOF local matrices. It compares
equivalent COO, CSR, and element operators. This tests the execution machinery,
not the accuracy or performance of a Maxwell discretization. CSR and element
operators reuse their output; the existing COO API allocates an output each
time. Upload and setup costs are excluded from resident GPU measurements.

| DOFs | COO CPU | CSR CPU | Elements CPU | CSR Metal | Elements Metal |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 2.10 µs | 0.404 µs | 0.676 µs | 3.91 µs | 2.91 µs |
| 4,096 | 34.7 µs | 6.48 µs | 11.1 µs | 3.41 µs | 2.93 µs |
| 65,536 | 540 µs | 105 µs | 169 µs | 4.34 µs | 4.14 µs |

A mixed CSR/dense Host block operator with 1,024 DOFs took 1.02 µs.
For 65,536 complex values, fused Host `dotc` took 40.8 µs versus 48.1 µs for
conjugating then taking the dot product. Metal `dotc`, including the scalar
readback, took 149 µs. A 128-DOF Host GMRES solve with a prebuilt ILU(0)
preconditioner took 8.57 µs. This tridiagonal case has an exact ILU(0), so it
needs one iteration and is not representative of indefinite Maxwell solves.

## Accepted tradeoffs and remaining limits

- CPU element application is about 1.62 times slower than CSR at 65,536 DOFs.
  It repeats local contributions. This is expected for this low-order case;
  keeping local matrices enables later high-order and geometry-based kernels.
- At 256 DOFs, Metal CSR is about 9.7 times slower than Host CSR. Dispatch
  overhead dominates. This is accepted because backend selection is explicit.
- Standalone Metal scalar reductions synchronize. The complex dot benchmark is about
  3.6 times slower than the fused CPU operation. Resident GMRES avoids full
  vector downloads. Its default orthogonalization now batches CGS2 projections
  and norms into one synchronization per Arnoldi step, as measured below.
  Initial norms, residual checks, and CPU convergence decisions still synchronize.
- Metal timings above queue repeated applications and synchronize per batch.
  They are throughput measurements. An individual application followed by a
  CPU read pays additional latency, as the `dotc` measurement demonstrates.
  GPU CSR and element ordering varies between runs at 65,536 DOFs; the full
  suite measured 3.68 and 3.93 µs respectively. The table uses the final
  focused run. A crossover should be measured on the intended workload.
- CSR stores an auxiliary transposed index view for O(nnz) adjoint products.
  It costs two extra `u32` arrays per nonzero plus column offsets. Coefficients
  are stored once. Applications that never use adjoints may prefer a future
  optional transpose cache.
- Operators use one backend throughout a block tree. A custom adapter can
  bridge CPU and GPU blocks, but transfers are never inserted implicitly.
- ILU(0) has no pivoting or fill. Its inverse can now run on Metal using
  dependency-scheduled triangular solves. Strong Maxwell
  preconditioners remain the responsibility of the FEM/application layer.
- Dense operator blocks currently use direct row kernels. Large dense blocks
  may benefit from integrating the existing tuned dense matvec dispatch.

The existing tensor arithmetic implementations are unchanged. The full
`nn_ops` suite completed, but no controlled before/after baseline was recorded
for unrelated neural-network kernels, so no cross-revision speed claim is made.

## Solver and geometry update, 2026-10-16

Focused default-feature benchmarks, using the same warmed harness:

| Case | Reference | New path |
| --- | ---: | ---: |
| Metal GMRES, 128 DOFs | MGS2: 14.23 ms | Batched CGS2: 3.57 ms |
| 3D cross product | Dynamic vector: 15.9 ns | Stack vector: 1.1 ns |
| 3x3 pivoted solve | Factor and solve: 22.7 ns | Reused factors: 2.7 ns |
| Tetrahedral degree-2 integration | — | Cached rule/Jacobian: 3.8 ns |

The GPU GMRES comparison uses the same diagonal matrix, initial guess,
restart, and tolerance. The reference selects sequential modified
Gram-Schmidt through `gmres_with_orthogonalization`. Both paths reorthogonalize
and verify the true residual. The new path is about four times faster in this
case. It remains much slower than Host (25.9 µs) for a system this small.
This is not a claim of universal GPU superiority or a Maxwell benchmark.

The complete `cargo bench --bench nn_ops` run also passed. Its GMRES medians
were 12.56 ms for MGS2 and 2.78 ms for CGS2, a 4.5 times improvement. Timing
variation between runs does not change the small-system CPU advantage.

Metal CGS2 packs its basis once per restart and appends one vector per step.
Each step queues eight kernels for both projection passes and their norms.
Only the accumulated coefficients and two norms return to the CPU, in one
read. The normalized vectors and preconditioned basis remain resident. The
Host path retains MGS2 and does not allocate the packed GPU basis.

Regression tests verify that 70 small CSR applications submit two full
32-dispatch command buffers without a wait, then a third buffer at the final
read. Another test verifies one read for batched Arnoldi with twelve basis
vectors. Triangular-level tests verify that dependent levels remain queued until
the final read, including complex adjoints and block offsets.

GPU triangular solves are supported, but dependency width limits performance:

| Shape | Host | Metal, queued |
| --- | ---: | ---: |
| 128 rows in one dependency chain | 0.405 µs | 260 µs |
| 4,096 rows in 512 chains of length 8 | 6.13 µs | 20.2 µs |

The GPU path is about 641 times slower for the serial chain and 3.3 times
slower for the wider case. This is an accepted capability tradeoff for the
initial level-scheduled implementation. It avoids vector transfers and waits
between levels, but does not remove serial dependencies or per-level dispatch
overhead. Backend selection remains explicit. ILU factorization and scheduling
still happen on Host; only factor application runs on Metal.

The geometry timings compare different APIs or setup policies, not an
algorithmic speedup from changing numerical tolerances. Fixed-size values
avoid heap allocation. Reusing an LU factorization excludes factorization
cost; reusing a quadrature rule and Jacobian excludes their construction.

Algorithm references: [blocked GMRES operations](https://netlib.org/linalg/old_html_templates/subsection2.8.4.6.html)
and [Piola mappings](https://defelement.org/finite-elements.html).

Validation passed: workspace tests with Metal, scalar-only tests, SIMD tests,
format checks, all-target/all-feature Clippy, and the release benchmark smoke
run. Clippy still reports pre-existing warnings outside these changes.
