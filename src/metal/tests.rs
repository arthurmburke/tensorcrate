//! Tests of the Metal module against the device on the test machine.

use std::mem::size_of;

use half::{bf16, f16};
use objc2_metal::MTLBuffer;

use crate::tensors::{Analytic, BinaryOp};

use super::buffer::MetalBuffer;
use super::device::GPU;
use super::slices::{broadcast_f32, elementwise_f32, fft_f32_interleaved, ifft_f32_interleaved};
use super::sync::synchronize;

/// Operations encoded and blocking waits so far on this thread.
fn activity() -> Option<(u64, u64)> {
    GPU.with(|cell| {
        cell.get()
            .and_then(Option::as_ref)
            .map(|gpu| (gpu.operations.get(), gpu.waits.get()))
    })
}

fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

#[test]
fn gpu_elementwise_matches_cpu() {
    let a: Vec<f32> = (0..1000).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..1000).map(|i| (i % 9) as f32 + 1.0).collect();
    for (op, f) in [
        (BinaryOp::Add, (|x: f32, y| x + y) as fn(f32, f32) -> f32),
        (BinaryOp::Sub, |x, y| x - y),
        (BinaryOp::Mul, |x, y| x * y),
        (BinaryOp::Div, |x, y| x / y),
    ] {
        if let Some(gpu) = elementwise_f32(&a, &b, op) {
            for (i, g) in gpu.iter().enumerate() {
                let want = f(a[i], b[i]);
                assert!((g - want).abs() < 1e-3, "op {op:?} at {i}: {g} vs {want}");
            }
        }
    }
}

#[test]
fn gpu_broadcast_matches_cpu() {
    let values: Vec<f32> = (0..1000).map(|i| i as f32 * 0.125 - 3.0).collect();
    if let Some(gpu) = broadcast_f32(&values, 2.5, BinaryOp::Mul, false) {
        for (actual, value) in gpu.iter().zip(values) {
            assert!((actual - value * 2.5).abs() < 1e-5);
        }
    }
}

#[test]
fn gpu_fft_and_ifft_match_the_cpu_definition() {
    let count = 1024usize;
    let mut input = Vec::with_capacity(count * 2);
    for i in 0..count {
        input.push((i % 17) as f32 * 0.25 - 2.0);
        input.push((i % 11) as f32 * -0.125 + 0.5);
    }

    let Some(spectrum) = fft_f32_interleaved(&input) else {
        eprintln!("no Metal device; skipping GPU comparison");
        return;
    };
    let reconstructed = ifft_f32_interleaved(&spectrum).unwrap();
    for (actual, expected) in reconstructed.iter().zip(input) {
        assert!(
            (actual - expected).abs() < 2e-4,
            "gpu={actual} cpu={expected}"
        );
    }
}

#[test]
fn accumulating_matmul_adds_into_its_target() {
    let (m, k, n) = (3usize, 4usize, 2usize);
    let a: Vec<f32> = (0..m * k).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 3) as f32 * 0.5).collect();
    let Some(buf_a) = MetalBuffer::from_slice(&a) else {
        eprintln!("no Metal device; skipping accumulation comparison");
        return;
    };
    let buf_b = MetalBuffer::from_slice(&b).unwrap();

    let product = cpu_matmul(&a, &b, m, k, n);
    let seed: Vec<f32> = (0..m * n).map(|i| i as f32).collect();
    let mut target = MetalBuffer::from_slice(&seed).unwrap();
    buf_a
        .matmul_accumulate(&buf_b, &mut target, m, k, n)
        .expect("accumulating dispatch");

    for (index, (actual, base)) in target.as_slice().iter().zip(&seed).enumerate() {
        let expected = base + product[index];
        assert!((actual - expected).abs() < 1e-4, "{actual} vs {expected}");
    }
}

#[test]
fn tensorops_multiplies_half_and_bfloat_inputs() {
    // Cross both 64×64 tile boundaries and leave partial tiles on each
    // edge, so this covers the dispatch geometry as well as the data types.
    let (m, k, n) = (70usize, 33usize, 69usize);
    let left = (0..m * k)
        .map(|index| ((index % 7) as f32 - 3.0) * 0.25)
        .collect::<Vec<_>>();
    let right = (0..k * n)
        .map(|index| ((index % 5) as f32 - 2.0) * 0.125)
        .collect::<Vec<_>>();
    let expected = cpu_matmul(&left, &right, m, k, n);

    let left_f16 = left.iter().copied().map(f16::from_f32).collect::<Vec<_>>();
    let right_f16 = right.iter().copied().map(f16::from_f32).collect::<Vec<_>>();
    let Some(left_f16) = MetalBuffer::<f16>::from_slice(&left_f16) else {
        eprintln!("no Metal device; skipping TensorOps comparison");
        return;
    };
    let right_f16 = MetalBuffer::<f16>::from_slice(&right_f16).unwrap();
    let Some(compact_f16) = left_f16.matmul(&right_f16, m, k, n) else {
        eprintln!("no Metal 4 TensorOps support; skipping TensorOps comparison");
        return;
    };
    for (actual, expected) in compact_f16.as_slice().iter().zip(&expected) {
        assert!((f32::from(*actual) - expected).abs() < 0.02);
    }
    let wide_f16 = left_f16.matmul_f32(&right_f16, m, k, n).unwrap();
    for (actual, expected) in wide_f16.as_slice().iter().zip(&expected) {
        assert!((actual - expected).abs() < 1e-4);
    }

    let left_bf16 = left.iter().copied().map(bf16::from_f32).collect::<Vec<_>>();
    let right_bf16 = right
        .iter()
        .copied()
        .map(bf16::from_f32)
        .collect::<Vec<_>>();
    let left_bf16 = MetalBuffer::<bf16>::from_slice(&left_bf16).unwrap();
    let right_bf16 = MetalBuffer::<bf16>::from_slice(&right_bf16).unwrap();
    let compact_bf16 = left_bf16.matmul(&right_bf16, m, k, n).unwrap();
    for (actual, expected) in compact_bf16.as_slice().iter().zip(&expected) {
        assert!((f32::from(*actual) - expected).abs() < 0.1);
    }
    let wide_bf16 = left_bf16.matmul_f32(&right_bf16, m, k, n).unwrap();
    for (actual, expected) in wide_bf16.as_slice().iter().zip(&expected) {
        assert!((actual - expected).abs() < 1e-4);
    }
}

#[test]
fn unary_dual_applies_a_function_and_its_derivative() {
    let value: Vec<f32> = (0..64).map(|i| (i % 9) as f32 * 0.1 + 0.05).collect();
    let tangent: Vec<f32> = (0..64).map(|i| (i % 4) as f32 - 1.5).collect();
    let Some(buf_value) = MetalBuffer::from_slice(&value) else {
        eprintln!("no Metal device; skipping unary comparison");
        return;
    };
    let buf_tangent = MetalBuffer::from_slice(&tangent).unwrap();

    // Op 12 is tanh: f' = 1 − tanh².
    let (values, tangents) = buf_value.unary_dual(&buf_tangent, Analytic::Tanh).unwrap();
    for (index, (&actual, &expected)) in values.as_slice().iter().zip(&value).enumerate() {
        let want = expected.tanh();
        assert!((actual - want).abs() < 1e-4, "value at {index}");
        let derivative = 1.0 - want * want;
        let want_tangent = derivative * tangent[index];
        assert!(
            (tangents.as_slice()[index] - want_tangent).abs() < 1e-4,
            "tangent at {index}"
        );
    }

    assert_eq!(size_of::<BinaryOp>(), 2);
    assert_eq!(size_of::<Analytic>(), 2);
}

#[test]
fn a_recycled_allocation_is_never_clobbered_by_queued_work() {
    // Work is committed without waiting, so an allocation dropped while its
    // dispatch is still queued must not be handed straight back out: the
    // kernel would land on top of whatever the next owner put there. Ten
    // rounds, because the failure is a race the GPU can win by luck.
    let input: Vec<f32> = (0..256).map(|i| (i % 13) as f32 * 0.1).collect();
    let Some(source) = MetalBuffer::from_slice(&input) else {
        eprintln!("no Metal device; skipping the recycling check");
        return;
    };

    let known: Vec<f32> = (0..256).map(|i| i as f32).collect();
    for round in 0..10 {
        // Queue a dispatch and drop its output immediately.
        drop(source.unary(Analytic::Tanh).expect("unary dispatch"));
        // This may reuse that allocation; its contents must be what was
        // uploaded, not what the queued kernel owed its previous owner.
        let fresh = MetalBuffer::from_slice(&known).expect("upload");
        assert_eq!(fresh.to_vec(), known, "round {round}");
    }
}

#[test]
fn shared_buffers_keep_chained_operations_gpu_resident() {
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![5.0f32, 6.0, 7.0, 8.0];
    let Some(a) = MetalBuffer::from_slice(&a) else {
        eprintln!("no Metal device; skipping shared-buffer comparison");
        return;
    };
    let b = MetalBuffer::from_slice(&b).unwrap();
    let product = a.matmul(&b, 2, 2, 2).unwrap();
    let scaled = product.broadcast(0.5, BinaryOp::Mul, false).unwrap();
    assert_eq!(scaled.to_vec(), vec![9.5, 11.0, 21.5, 25.0]);

    let complex = MetalBuffer::from_slice(&[1.0, 0.0, 2.0, -1.0, 0.5, 3.0, -2.0, 0.25]).unwrap();
    let reconstructed = complex.fft().unwrap().ifft().unwrap().to_vec();
    for (actual, expected) in reconstructed.iter().zip(complex.to_vec()) {
        assert!((actual - expected).abs() < 1e-5);
    }
}

#[test]
fn stacking_reads_queued_device_results_without_host_staging() {
    let Some(first) = MetalBuffer::from_slice(&[1.0, 2.0, 3.0]) else {
        eprintln!("no Metal device; skipping device stacking comparison");
        return;
    };
    let second = MetalBuffer::from_slice(&[4.0, 5.0, 6.0]).unwrap();

    // Leave both inputs as pending GPU results. The stack dispatch must
    // consume those buffers directly, in command-queue order.
    let first = first.broadcast(10.0, BinaryOp::Add, false).unwrap();
    let second = second.broadcast(20.0, BinaryOp::Add, false).unwrap();

    let vertical = MetalBuffer::vstack(&[&first, &second], 3).unwrap();
    assert_eq!(vertical.to_vec(), vec![11.0, 12.0, 13.0, 24.0, 25.0, 26.0]);

    let horizontal = MetalBuffer::hstack(&[&first, &second], 3).unwrap();
    assert_eq!(
        horizontal.to_vec(),
        vec![11.0, 24.0, 12.0, 25.0, 13.0, 26.0]
    );
}

#[test]
fn tiled_transpose_stays_queued_and_handles_partial_tiles() {
    const ROWS: usize = 19;
    const COLS: usize = 23;
    synchronize();
    let before = activity().unwrap_or_default();

    let values = (0..ROWS * COLS)
        .map(|index| index as f32)
        .collect::<Vec<_>>();
    let Some(input) = MetalBuffer::from_slice(&values) else {
        eprintln!("no Metal device; skipping device transpose comparison");
        return;
    };
    let queued = input.broadcast(1.0, BinaryOp::Add, false).unwrap();
    let transposed = queued.transpose(ROWS, COLS).unwrap();

    let (operations, waits) = activity().unwrap();
    assert_eq!(
        waits, before.1,
        "transpose unexpectedly synchronized GPU work"
    );
    assert_eq!(operations - before.0, 2, "unexpected number of operations");

    let expected = (0..COLS)
        .flat_map(|col| (0..ROWS).map(move |row| (row * COLS + col) as f32 + 1.0))
        .collect::<Vec<_>>();
    assert_eq!(transposed.to_vec(), expected);

    let empty = MetalBuffer::<f32>::from_slice(&[]).unwrap();
    assert!(empty.transpose(0, COLS).unwrap().is_empty());
}

#[test]
fn matrix_concat_and_stack_stay_on_the_device() {
    const ROWS: usize = 19;
    const LEFT_COLS: usize = 13;
    const RIGHT_COLS: usize = 7;
    synchronize();
    let before = activity().unwrap_or_default();

    let left_values = (0..ROWS * LEFT_COLS)
        .map(|index| index as f32)
        .collect::<Vec<_>>();
    let right_values = (0..ROWS * RIGHT_COLS)
        .map(|index| 1_000.0 + index as f32)
        .collect::<Vec<_>>();
    let Some(left) = MetalBuffer::from_slice(&left_values) else {
        eprintln!("no Metal device; skipping matrix assembly comparison");
        return;
    };
    let right = MetalBuffer::from_slice(&right_values).unwrap();
    let left = left.broadcast(1.0, BinaryOp::Add, false).unwrap();
    let right = right.broadcast(2.0, BinaryOp::Add, false).unwrap();
    let concat = left
        .concat_matrix(&right, ROWS, LEFT_COLS, RIGHT_COLS)
        .unwrap();

    const TOP_ROWS: usize = 5;
    const BOTTOM_ROWS: usize = 7;
    const COLS: usize = 11;
    let top_values = (0..TOP_ROWS * COLS)
        .map(|index| index as f32)
        .collect::<Vec<_>>();
    let bottom_values = (0..BOTTOM_ROWS * COLS)
        .map(|index| 500.0 + index as f32)
        .collect::<Vec<_>>();
    let top = MetalBuffer::from_slice(&top_values)
        .unwrap()
        .broadcast(3.0, BinaryOp::Add, false)
        .unwrap();
    let bottom = MetalBuffer::from_slice(&bottom_values)
        .unwrap()
        .broadcast(4.0, BinaryOp::Add, false)
        .unwrap();
    let stack = top
        .stack_matrix(&bottom, TOP_ROWS, BOTTOM_ROWS, COLS)
        .unwrap();

    let (operations, waits) = activity().unwrap();
    assert_eq!(
        waits, before.1,
        "matrix assembly unexpectedly synchronized GPU work"
    );
    assert_eq!(operations - before.0, 6, "unexpected number of operations");

    let mut expected_concat = Vec::with_capacity(ROWS * (LEFT_COLS + RIGHT_COLS));
    for row in 0..ROWS {
        expected_concat.extend(
            left_values[row * LEFT_COLS..(row + 1) * LEFT_COLS]
                .iter()
                .map(|value| value + 1.0),
        );
        expected_concat.extend(
            right_values[row * RIGHT_COLS..(row + 1) * RIGHT_COLS]
                .iter()
                .map(|value| value + 2.0),
        );
    }
    assert_eq!(concat.to_vec(), expected_concat);

    let expected_stack = top_values
        .iter()
        .map(|value| value + 3.0)
        .chain(bottom_values.iter().map(|value| value + 4.0))
        .collect::<Vec<_>>();
    assert_eq!(stack.to_vec(), expected_stack);
}

#[test]
fn matrix_merges_consume_queued_device_buffers() {
    const MATRICES: usize = 3;
    const ROWS: usize = 19;
    const COLS: usize = 7;
    synchronize();
    let before = activity().unwrap_or_default();

    let host = (0..MATRICES)
        .map(|matrix| {
            (0..ROWS * COLS)
                .map(|index| matrix as f32 * 1_000.0 + index as f32)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let Some(inputs) = host
        .iter()
        .map(|values| MetalBuffer::from_slice(values))
        .collect::<Option<Vec<_>>>()
    else {
        eprintln!("no Metal device; skipping matrix merge comparison");
        return;
    };
    let queued = inputs
        .iter()
        .enumerate()
        .map(|(index, input)| {
            input
                .broadcast(index as f32 + 1.0, BinaryOp::Add, false)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let buffers = queued.iter().collect::<Vec<_>>();

    let horizontal = MetalBuffer::hmerge(&buffers, ROWS, COLS).unwrap();
    let vertical = MetalBuffer::vmerge(&buffers, ROWS, COLS).unwrap();

    let (operations, waits) = activity().unwrap();
    assert_eq!(
        waits, before.1,
        "matrix merge unexpectedly synchronized GPU work"
    );
    assert_eq!(operations - before.0, 5, "unexpected number of operations");

    let mut expected_horizontal = Vec::with_capacity(MATRICES * ROWS * COLS);
    for row in 0..ROWS {
        for (matrix, values) in host.iter().enumerate() {
            expected_horizontal.extend(
                values[row * COLS..(row + 1) * COLS]
                    .iter()
                    .map(|value| value + matrix as f32 + 1.0),
            );
        }
    }
    assert_eq!(horizontal.to_vec(), expected_horizontal);

    let expected_vertical = host
        .iter()
        .enumerate()
        .flat_map(|(matrix, values)| values.iter().map(move |value| value + matrix as f32 + 1.0))
        .collect::<Vec<_>>();
    assert_eq!(vertical.to_vec(), expected_vertical);
}

#[test]
fn strided_copies_run_on_the_device_without_waiting() {
    use crate::tensors::backend::{Backend, Region, Strided};
    use crate::tensors::{Metal, MetalStorage};

    // A `[2, 3, 4]` tensor permuted to `[4, 2, 3]`, read from a queued result.
    let values = (0..24).map(|index| index as f32).collect::<Vec<_>>();
    let Some(input) = MetalBuffer::from_slice(&values) else {
        eprintln!("no Metal device; skipping strided copy comparison");
        return;
    };
    synchronize();
    let before = activity().unwrap();
    let queued = input.broadcast(1.0, BinaryOp::Add, false).unwrap();
    let shape = [4, 2, 3];
    let from = Strided {
        offset: 0,
        strides: &[1, 12, 4],
    };
    let permuted = queued.strided_copy(&shape, from).unwrap();

    // Two blocks of a concatenation along the last axis of `[4, 2, 3 + 3]`.
    let target_strides = [12, 6, 1];
    let region = |offset| Region {
        source: &queued,
        shape: &shape,
        from,
        to: Strided {
            offset,
            strides: &target_strides,
        },
    };
    let joined = MetalBuffer::assemble(48, &[region(0), region(3)]).unwrap();

    // One of those blocks written again, in place, from the permuted copy.
    let mut target = MetalBuffer::from_slice(&[0.0f32; 48]).unwrap();
    let contiguous = [6, 3, 1];
    target
        .strided_write(Region {
            source: &permuted,
            shape: &shape,
            from: Strided {
                offset: 0,
                strides: &contiguous,
            },
            to: Strided {
                offset: 3,
                strides: &target_strides,
            },
        })
        .unwrap();

    let (operations, waits) = activity().unwrap();
    assert_eq!(waits, before.1, "a strided copy synchronized GPU work");
    assert_eq!(
        operations - before.0,
        5,
        "one broadcast, one copy, two assembled regions and one write"
    );

    let expected = (0..4)
        .flat_map(|d| {
            (0..2).flat_map(move |b| (0..3).map(move |t| (b * 12 + t * 4 + d) as f32 + 1.0))
        })
        .collect::<Vec<_>>();
    assert_eq!(permuted.to_vec(), expected);
    let doubled = expected
        .chunks(3)
        .flat_map(|run| run.iter().chain(run).copied())
        .collect::<Vec<_>>();
    assert_eq!(joined.to_vec(), doubled);
    let written = expected
        .chunks(3)
        .flat_map(|run| [0.0; 3].into_iter().chain(run.iter().copied()))
        .collect::<Vec<_>>();
    assert_eq!(target.to_vec(), written);

    // An index type has no arithmetic kernels but copies on the device.
    let ids = MetalStorage::from_slice(&(0..24u32).collect::<Vec<_>>());
    let copied = <Metal as Backend>::strided_copy(&ids, &shape, from);
    assert!(copied.is_device_resident());
    let expected = (0..4)
        .flat_map(|d| (0..2).flat_map(move |b| (0..3).map(move |t| b * 12 + t * 4 + d)))
        .collect::<Vec<u32>>();
    assert_eq!(copied.as_slice(), expected);
}

/// Operations are committed without waiting, so a long dependent chain is
/// the thing that would break if command buffers on one queue did not run in
/// commit order, or if a kernel could start before its input was written.
/// Each link here depends on the previous one and every link is exactly
/// representable, so any reordering, overlap, or dropped stage is an
/// unambiguous mismatch rather than a rounding difference.
#[test]
fn deferred_completion_preserves_the_order_of_a_dependent_chain() {
    const LINKS: usize = 250; // past the 64-buffer flush point, several times
    let start: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let Some(mut buffer) = MetalBuffer::from_slice(&start) else {
        eprintln!("no Metal device; skipping deferred-completion chain");
        return;
    };

    let ones = MetalBuffer::from_slice(&vec![1.0f32; 64]).unwrap();
    for _ in 0..LINKS {
        // +1 via broadcast, then +1 via elementwise: two kernels per link,
        // each reading what the one before it just wrote.
        buffer = buffer.broadcast(1.0, BinaryOp::Add, false).unwrap();
        buffer = buffer.elementwise(&ones, BinaryOp::Add).unwrap();
    }

    let expected: Vec<f32> = (0..64).map(|i| (i + 2 * LINKS) as f32).collect();
    assert_eq!(buffer.to_vec(), expected);
}

/// `synchronize` has to be enough on its own: after it returns, work queued
/// earlier must be visible to a later read that does not itself sync.
#[test]
fn synchronize_makes_queued_work_observable() {
    let Some(buffer) = MetalBuffer::from_slice(&[3.0f32; 32]) else {
        eprintln!("no Metal device; skipping synchronize check");
        return;
    };
    let doubled = buffer.broadcast(2.0, BinaryOp::Mul, false).unwrap();
    synchronize();
    assert_eq!(doubled.to_vec(), vec![6.0f32; 32]);
}

/// An allocation dropped while its kernel was queued comes back into service
/// once the GPU has finished with it, without anyone blocking on a sync — which
/// is what lets a long queued chain recycle its intermediates.
#[test]
fn a_finished_allocation_is_reused_without_a_sync() {
    let Some(source) = MetalBuffer::from_slice(&[1.0f32; 4096]) else {
        eprintln!("no Metal device; skipping the reclaim check");
        return;
    };
    synchronize();
    let before = activity().unwrap();
    let output = source.unary(Analytic::Exp).unwrap();
    let address = output.raw().contents();
    drop(output);

    // Wait for the GPU without `sync`: commit the batch, then poll until the
    // command buffer reports completion, as `reclaim` would see it.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    GPU.with(|cell| {
        let gpu = cell.get().and_then(Option::as_ref).unwrap();
        super::sync::flush(gpu).unwrap();
        while gpu.completed.get() < gpu.committed.get() {
            assert!(
                std::time::Instant::now() < deadline,
                "the GPU never finished"
            );
            super::sync::reclaim(gpu);
            std::thread::yield_now();
        }
    });

    let reused = MetalBuffer::<f32>::allocate(4096).unwrap();
    assert_eq!(
        reused.raw().contents(),
        address,
        "the finished allocation was not reused"
    );
    assert_eq!(
        activity().unwrap().1,
        before.1,
        "reclaiming blocked on the GPU"
    );
}

/// Copying a tensor to the backend it is already on — what the tape does to
/// duplicate an adjoint, and what `outer` does to its operands — is a GPU copy
/// queued behind the kernel still writing the source, not a sync.
#[test]
fn a_same_backend_copy_does_not_wait_for_the_gpu() {
    use crate::tensors::{Host, Matrix, Metal, Vector};

    let values: Vec<f32> = (0..4096).map(|i| i as f32 * 0.25).collect();
    let vector = Vector::new(values.clone()).to_backend::<Metal>();
    if !vector.is_device_resident() {
        eprintln!("no Metal device; skipping the same-backend copy check");
        return;
    }
    let matrix = Matrix::from_flat(64, 64, values.clone()).to_backend::<Metal>();
    synchronize();
    let before = activity().unwrap();

    let doubled = vector.scale(2.0);
    let copied = doubled.to_backend::<Metal>();
    let matrix_copy = matrix.scale(3.0).to_backend::<Metal>();
    let cloned = copied.clone();
    assert_eq!(
        activity().unwrap().1,
        before.1,
        "a same-backend copy blocked"
    );

    let twice: Vec<f32> = values.iter().map(|v| v * 2.0).collect();
    assert_eq!(copied.to_backend::<Host>().into_vec(), twice);
    assert_eq!(cloned.to_backend::<Host>().into_vec(), twice);
    let thrice: Vec<f32> = values.iter().map(|v| v * 3.0).collect();
    assert_eq!(matrix_copy.to_backend::<Host>().into_vec(), thrice);
}

/// A relaxed-precision product agrees with the exact one to the accuracy it
/// promises, and the setting is per thread and restorable.
#[test]
fn relaxed_matmul_stays_close_to_the_exact_product() {
    use super::device::{MatmulPrecision, matmul_precision, set_matmul_precision};
    use crate::tensors::{Host, Matrix, Metal};

    const N: usize = 192;
    let values = |seed: usize| -> Vec<f32> {
        (0..N * N)
            .map(|i| (((i * 2654435761 + seed * 40503) % 1000) as f32 / 500.0) - 1.0)
            .collect()
    };
    let a = Matrix::from_flat(N, N, values(1)).to_backend::<Metal>();
    let b = Matrix::from_flat(N, N, values(2)).to_backend::<Metal>();
    if !a.is_device_resident() {
        eprintln!("no Metal device; skipping the relaxed product check");
        return;
    }
    assert_eq!(matmul_precision(), MatmulPrecision::Exact);
    let exact = a.matmul(&b).to_backend::<Host>().into_vec();
    set_matmul_precision(MatmulPrecision::Relaxed);
    let relaxed = a.matmul(&b).to_backend::<Host>().into_vec();
    set_matmul_precision(MatmulPrecision::Exact);

    // Reduced precision errs in proportion to the size of the products being
    // summed, so measure against the result's scale, not each element's own
    // size — an element whose terms cancel to near zero errs as much as any.
    let scale = (exact.iter().map(|e| e * e).sum::<f32>() / exact.len() as f32).sqrt();
    for (index, (r, e)) in relaxed.iter().zip(&exact).enumerate() {
        assert!(
            (r - e).abs() <= 2e-2 * (1.0 + scale),
            "element {index}: relaxed {r} against exact {e} (scale {scale})"
        );
    }
}

/// A program run twice is compiled into a kernel of its own, which must
/// compute what the interpreter does: for every element type, through remapped
/// loads, an in-place update and compact storage.
#[test]
fn specialized_kernels_match_the_interpreter() {
    use super::codegen::set_fused_codegen;
    use super::device::Specialized;
    use crate::tensors::fused::{Builder, DType, Element, Fusable, Remap};
    use crate::tensors::{Analytic, Compare, Host, Matrix, Metal, Vector};

    fn check<T: super::MetalElement + Element>(tolerance: f64) {
        const ROWS: usize = 37;
        const COLS: usize = 53;
        let len = ROWS * COLS;
        let values = |seed: usize, n: usize| -> Vec<T> {
            (0..n)
                .map(|i| {
                    T::from_f64((((i * 2654435761 + seed * 40503) % 1000) as f64 / 500.0) - 1.0)
                })
                .collect()
        };
        let x = Matrix::<T>::from_flat(ROWS, COLS, values(1, len)).to_backend::<Metal>();
        if !x.is_device_resident() {
            eprintln!("no Metal device; skipping the specialized-kernel check");
            return;
        }
        let t = Matrix::<T>::from_flat(COLS, ROWS, values(2, len)).to_backend::<Metal>();
        let row = Vector::<T>::new(values(3, COLS)).to_backend::<Metal>();
        let start = Matrix::<T>::from_flat(ROWS, COLS, values(4, len)).to_backend::<Metal>();

        let mut b = Builder::<T>::new();
        let xv = b.input(T::DTYPE);
        let tv = b.input_remapped(T::DTYPE, Remap::Transpose);
        let rv = b.input_remapped(T::DTYPE, Remap::Row);
        let acc = b.update(T::DTYPE);
        let product = b.mul(xv, tv);
        let shifted = b.add(product, rv);
        let squashed = b.unary(Analytic::Tanh, shifted);
        let zero = b.constant(T::from_f64(0.25));
        let floor = b.compare(Compare::Max, squashed, zero);
        let step = b.scale(floor, T::from_f64(0.5));
        let next = b.add(acc, step);
        b.set(0, next);
        b.output(squashed, DType::F32);
        let program = b.build().unwrap();

        let run = |codegen: bool| {
            set_fused_codegen(codegen);
            let mut state = start.clone();
            let inputs: [&dyn Fusable<Metal>; 3] = [&x, &t, &row];
            let mut out = program.run((ROWS, COLS), &inputs, &mut [&mut state]);
            let fresh = out.remove(0).into_matrix::<f32>().to_backend::<Host>();
            set_fused_codegen(true);
            (state.to_backend::<Host>(), fresh)
        };
        let (want_state, want_fresh) = run(false);
        for round in 0..3 {
            let (state, fresh) = run(true);
            for (got, want) in state.as_slice().iter().zip(want_state.as_slice()) {
                let (got, want) = (got.into_f64(), want.into_f64());
                assert!(
                    (got - want).abs() <= tolerance * (1.0 + want.abs()),
                    "round {round}: state {got} vs {want}"
                );
            }
            for (got, want) in fresh.as_slice().iter().zip(want_fresh.as_slice()) {
                assert!(
                    (got - want).abs() <= tolerance as f32 * (1.0 + want.abs()),
                    "round {round}: output {got} vs {want}"
                );
            }
        }
        let compiled = GPU.with(|cell| {
            let gpu = cell.get().and_then(Option::as_ref).unwrap();
            gpu.specialized
                .borrow()
                .values()
                .any(|entry| matches!(entry, Specialized::Ready(_)))
        });
        assert!(compiled, "the repeated program was never compiled");
    }
    check::<f32>(1e-5);
    check::<f16>(4e-3);
    check::<bf16>(2e-2);
}

#[test]
fn general_products_read_operands_transposed_and_accumulate() {
    use super::device::with_gpu;
    use super::encode::encode_gemm;
    let (m, k, n) = (70, 45, 33);
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 5) as f32 - 2.0) * 0.5).collect();
    let c: Vec<f32> = (0..m * n).map(|i| (i % 3) as f32).collect();
    let transpose = |x: &[f32], rows: usize, cols: usize| -> Vec<f32> {
        (0..rows * cols)
            .map(|e| x[(e % rows) * cols + e / rows])
            .collect()
    };
    let product = cpu_matmul(&a, &b, m, k, n);
    let added: Vec<f32> = product.iter().zip(&c).map(|(p, c)| p + c).collect();
    // Every tile, each operand transposed and neither, accumulating or not.
    let tiles = [("s", (32, 32, 4)), ("m", (64, 32, 2)), ("l", (64, 64, 4))];
    let mut cases = Vec::new();
    for (tile, shape) in tiles {
        for (operands, stored_a, stored_b) in [
            ("nn", a.clone(), b.clone()),
            ("tn", transpose(&a, m, k), b.clone()),
            ("nt", a.clone(), transpose(&b, k, n)),
        ] {
            for accumulate in [false, true] {
                let suffix = if accumulate { "_acc" } else { "" };
                let kernel = format!("gemm_f32_{tile}_{operands}{suffix}");
                cases.push((
                    kernel,
                    shape,
                    stored_a.clone(),
                    stored_b.clone(),
                    accumulate,
                ));
            }
        }
    }
    for (kernel, tile, stored_a, stored_b, accumulate) in cases {
        let kernel = kernel.as_str();
        let (ga, gb) = (
            MetalBuffer::from_slice(&stored_a).unwrap(),
            MetalBuffer::from_slice(&stored_b).unwrap(),
        );
        let out = MetalBuffer::from_slice(&c).unwrap();
        let ran = with_gpu(|gpu| {
            let pipeline = gpu.tensorops_named(kernel)?;
            encode_gemm(gpu, &pipeline, &ga.raw, &gb.raw, &out.raw, (m, k, n), tile)
        });
        if ran.is_none() {
            return; // no TensorOps on this GPU
        }
        synchronize();
        let want = if accumulate { &added } else { &product };
        for (i, (g, w)) in out.as_slice().iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-4 * w.abs().max(1.0),
                "{kernel}: element {i} is {g}, expected {w}"
            );
        }
    }
}

/// The generator's epilogue and sum kernels compile — for every element type
/// and every product tile — rather than quietly leaving the program on the
/// interpreter.
#[test]
fn generated_kernels_compile_for_every_type_and_tile() {
    use super::codegen::Kernel;
    use super::device::with_gpu;
    use crate::tensors::Compare;
    use crate::tensors::fused::{Builder, DType, Element, Remap};

    fn check<T: super::MetalElement + Element>() {
        let mut b = Builder::<T>::new();
        let product = b.input(T::DTYPE);
        let bias = b.input_remapped(T::DTYPE, Remap::Row);
        let shifted = b.add(product, bias);
        let zero = b.constant(T::from_f64(0.0));
        let activated = b.compare(Compare::Max, shifted, zero);
        b.output(activated, DType::F32);
        let code = b.build().unwrap().encode();
        let mut sum = Builder::<T>::new();
        let x = sum.input(T::DTYPE);
        let row = sum.input_remapped(T::DTYPE, Remap::Row);
        let e = sum.unary(crate::tensors::Analytic::Exp, x);
        let y = sum.mul(e, row);
        sum.output(y, T::DTYPE);
        let sum_code = sum.build().unwrap().encode();
        // A layer norm's statistics, of an input stored as `f16` whatever `T`.
        let mut norm = Builder::<T>::new();
        let x = norm.input(DType::F16);
        let mean = norm.row_statistic(x, crate::tensors::fused::RowStatistic::Mean);
        let deviations = norm.row_statistic(x, crate::tensors::fused::RowStatistic::Deviations);
        let centered = norm.sub(x, mean);
        let scaled = norm.div(centered, deviations);
        norm.output(scaled, T::DTYPE);
        let norm = norm.build().unwrap();
        let mut statistics = [(0u8, 0u8, false); 16];
        statistics[0] = (0, DType::F16 as u8, false);
        statistics[1] = (0, DType::F16 as u8, true);
        let rows = Kernel::Rows(super::codegen::RowStatistics {
            first: 1,
            statistics,
            count: 2,
        });
        assert_eq!(norm.row_statistics().len(), 2);
        for kernel in [Kernel::RowSums, Kernel::ColumnSums, rows] {
            let code = if kernel == rows {
                norm.encode()
            } else {
                sum_code.clone()
            };
            let compiled = with_gpu(|gpu| {
                let _ = gpu.specialized::<T>(&code, kernel);
                Some(gpu.specialized::<T>(&code, kernel).is_some())
            });
            if let Some(compiled) = compiled {
                assert!(compiled, "{} {kernel:?}", T::SUFFIX);
            }
        }
        for tile in [(32, 32, 4), (64, 32, 2), (64, 64, 4)] {
            for relaxed in [false, true] {
                let kernel = Kernel::Epilogue { tile, relaxed };
                let compiled = with_gpu(|gpu| {
                    gpu.tensorops_library.as_ref()?;
                    // Compiled on the second sighting.
                    let _ = gpu.specialized::<T>(&code, kernel);
                    Some(gpu.specialized::<T>(&code, kernel).is_some())
                });
                if let Some(compiled) = compiled {
                    assert!(
                        compiled,
                        "{} epilogue, tile {tile:?}, relaxed {relaxed}",
                        T::SUFFIX
                    );
                }
            }
        }
    }
    check::<f32>();
    check::<f16>();
    check::<bf16>();
}

#[test]
fn broadcasts_and_axis_reductions_are_one_dispatch_each_without_waiting() {
    use crate::tensors::backend::Strided;
    use crate::tensors::kernels::{AxisReduction, Pairwise};
    use crate::tensors::layout::Split;
    use crate::tensors::{Compare, Reduce};

    // [2, 3, 4] scores plus a [3, 4] mask, read from a queued result.
    let values = (0..24).map(|index| index as f32).collect::<Vec<_>>();
    let Some(input) = MetalBuffer::from_slice(&values) else {
        eprintln!("no Metal device; skipping broadcast dispatch check");
        return;
    };
    let mask = MetalBuffer::from_slice(&values[..12]).unwrap();
    synchronize();
    let before = activity().unwrap();
    let queued = input.broadcast(1.0, BinaryOp::Add, false).unwrap();
    let shape = [2, 3, 4];
    let scores = Strided {
        offset: 0,
        strides: &[12, 4, 1],
    };
    let repeated = Strided {
        offset: 0,
        strides: &[0, 4, 1],
    };
    let add = Pairwise::Arithmetic(BinaryOp::Add);
    let sum = queued
        .strided_binary(scores, &mask, repeated, &shape, add)
        .unwrap();
    let greater = Pairwise::Compare(Compare::Greater);
    let larger = queued
        .strided_binary(scores, &mask, repeated, &shape, greater)
        .unwrap();
    // Sums over the middle axis, a variance over the last two, and the
    // position of each row's maximum.
    let middle = Split::new(&shape, &[12, 4, 1], 0, &[1]);
    let sums = sum
        .reduce_axes(&middle, AxisReduction::Fold(Reduce::Sum))
        .unwrap();
    let inner = Split::new(&shape, &[12, 4, 1], 0, &[1, 2]);
    let variances = queued
        .reduce_axes(&inner, AxisReduction::Variance { divisor: 12 })
        .unwrap();
    let last = Split::new(&shape, &[12, 4, 1], 0, &[2]);
    let positions = sum.arg_reduce(&last, Reduce::Max).unwrap();

    let (operations, waits) = activity().unwrap();
    assert_eq!(waits, before.1, "a broadcast or a reduction synchronized");
    assert_eq!(
        operations - before.0,
        6,
        "one scalar broadcast, two broadcasts, two reductions and one arg-reduction"
    );

    let added = (0..24)
        .map(|i| (i as f32 + 1.0) + (i % 12) as f32)
        .collect::<Vec<_>>();
    assert_eq!(sum.to_vec(), added);
    let expected = (0..24)
        .map(|i| {
            if i as f32 + 1.0 > (i % 12) as f32 {
                1.0
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(larger.to_vec(), expected);
    let column_sums = (0..2)
        .flat_map(|b| {
            let added = &added;
            (0..4).map(move |d| (0..3).map(|t| added[b * 12 + t * 4 + d]).sum::<f32>())
        })
        .collect::<Vec<_>>();
    assert_eq!(sums.to_vec(), column_sums);
    // Twelve consecutive values have variance (12² − 1)/12.
    assert_eq!(variances.to_vec(), [143.0 / 12.0; 2]);
    assert_eq!(positions.to_vec(), [3u32; 6]);
}
