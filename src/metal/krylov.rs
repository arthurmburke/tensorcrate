//! Resident Arnoldi projections and dependency-scheduled triangular solves.

use super::device::with_gpu;
use super::operators::{device, dispatch};
use super::sync::{compute, queued};
use super::{MetalBuffer, OperatorElement};
use crate::tensors::operator::Apply;
use crate::tensors::{Metal, MetalStorage, TriangularOperator};
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};
use std::{mem::size_of, ptr::NonNull};

#[cfg(test)]
mod tests;

pub(super) fn orthogonalize<T: OperatorElement>(
    basis: &MetalStorage<T>,
    n: usize,
    count: usize,
    work: &mut MetalStorage<T>,
) -> (Vec<T>, f64, f64) {
    assert!(
        n.checked_mul(count).is_some_and(|size| size <= basis.len()),
        "Arnoldi basis shape mismatch"
    );
    assert_eq!(work.len(), n);
    let chunks = n.div_ceil(256);
    let partial_len = chunks
        .checked_mul(count.checked_add(1).expect("Arnoldi count overflow"))
        .expect("Arnoldi scratch size overflow");
    let result_len = count
        .checked_mul(2)
        .and_then(|n| n.checked_add(2))
        .expect("Arnoldi result size overflow");
    for size in [basis.len(), n, count, chunks, partial_len, result_len] {
        assert!(
            u32::try_from(size).is_ok(),
            "Metal Arnoldi exceeds 32-bit indexing"
        );
    }
    if n == 0 {
        return (vec![T::zero(); count], 0.0, 0.0);
    }
    let partial =
        MetalBuffer::<T>::allocate(partial_len).expect("Arnoldi scratch allocation failed");
    let result = MetalBuffer::<T>::allocate(result_len).expect("Arnoldi result allocation failed");
    for (stage, threads) in [
        (0, partial_len),
        (1, count + 1),
        (2, n),
        (3, count * chunks),
        (4, count),
        (2, n),
        (6, chunks),
        (7, 1),
    ] {
        if threads == 0 {
            continue;
        }
        let parameters = [n as u32, count as u32, chunks as u32, stage];
        with_gpu(|gpu| {
            let encoder = compute(gpu)?;
            let pipeline = &gpu.krylov[T::OPERATOR_INDEX];
            encoder.setComputePipelineState(pipeline);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(device(basis).raw()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(device(work).raw()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(partial.raw()), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(result.raw()), 0, 3);
                encoder.setBytes_length_atIndex(
                    NonNull::from(&parameters).cast(),
                    size_of::<[u32; 4]>(),
                    4,
                );
            }
            let group = pipeline.threadExecutionWidth() * 4;
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: threads.div_ceil(group),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: group,
                    height: 1,
                    depth: 1,
                },
            );
            queued(gpu, threads)
        })
        .expect("Metal Arnoldi dispatch failed");
    }
    let result = result.as_slice();
    (
        result[count..2 * count].to_vec(),
        result[2 * count].magnitude(),
        result[2 * count + 1].magnitude(),
    )
}

pub(super) fn triangular<T: OperatorElement>(
    a: &TriangularOperator<T, Metal>,
    x: &MetalStorage<T>,
    y: &mut MetalStorage<T>,
    p: Apply<T>,
) {
    let n = a.matrix.rows();
    assert!(
        p.input.checked_add(n).is_some_and(|end| end <= x.len()),
        "triangular input window out of bounds"
    );
    assert!(
        p.output.checked_add(n).is_some_and(|end| end <= y.len()),
        "triangular output window out of bounds"
    );
    let temporary = MetalBuffer::<T>::allocate(n).expect("triangular scratch allocation failed");
    let (levels, rows, offsets, columns) = if p.adjoint {
        (
            &a.adjoint_levels,
            &a.adjoint_rows,
            &a.matrix.adjoint_offsets,
            &a.matrix.adjoint_columns,
        )
    } else {
        (&a.levels, &a.rows, &a.matrix.offsets, &a.matrix.columns)
    };
    let unit = u32::from(a.unit_diagonal);
    for level in levels.windows(2) {
        let count = (level[1] - level[0]) as usize;
        let parameters = [
            level[0],
            count as u32,
            u32::try_from(p.input).expect("triangular offset exceeds 32 bits"),
            u32::from(p.adjoint),
        ];
        with_gpu(|gpu| {
            let encoder = compute(gpu)?;
            let pipeline = &gpu.triangular[T::OPERATOR_INDEX];
            encoder.setComputePipelineState(pipeline);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(device(&a.matrix.values).raw()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(device(offsets).raw()), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(device(columns).raw()), 0, 2);
                encoder.setBuffer_offset_atIndex(
                    Some(device(&a.matrix.adjoint_positions).raw()),
                    0,
                    3,
                );
                encoder.setBuffer_offset_atIndex(Some(device(rows).raw()), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(device(x).raw()), 0, 5);
                encoder.setBuffer_offset_atIndex(Some(temporary.raw()), 0, 6);
                encoder.setBytes_length_atIndex(
                    NonNull::from(&parameters).cast(),
                    size_of::<[u32; 4]>(),
                    7,
                );
                encoder.setBytes_length_atIndex(NonNull::from(&unit).cast(), size_of::<u32>(), 8);
            }
            let group = pipeline.threadExecutionWidth() * 4;
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: count.div_ceil(group),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: group,
                    height: 1,
                    depth: 1,
                },
            );
            queued(gpu, count)
        })
        .expect("Metal triangular dispatch failed");
    }
    dispatch(
        &temporary,
        [None; 3],
        &temporary,
        device(y),
        8,
        (n, n),
        0,
        n,
        Apply { input: 0, ..p },
    );
}
