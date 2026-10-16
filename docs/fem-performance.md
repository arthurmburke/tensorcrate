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
- Metal scalar reductions synchronize. The complex dot benchmark is about
  3.6 times slower than the fused CPU operation. Resident GMRES avoids full
  vector downloads but still synchronizes for norms and orthogonalization.
  Pipelined or communication-avoiding Krylov methods remain future work.
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
- ILU(0) has no pivoting, fill, or GPU triangular solve. Use a resident Jacobi
  inverse or a custom inverse operator when solving on Metal. Strong Maxwell
  preconditioners remain the responsibility of the FEM/application layer.
- Dense operator blocks currently use direct row kernels. Large dense blocks
  may benefit from integrating the existing tuned dense matvec dispatch.

The existing tensor arithmetic implementations are unchanged. The full
`nn_ops` suite completed, but no controlled before/after baseline was recorded
for unrelated neural-network kernels, so no cross-revision speed claim is made.
