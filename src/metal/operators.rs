//! Resident real and complex operator dispatch.

use super::MetalBuffer;
use super::device::with_gpu;
use super::sync::{compute, queued};
use crate::numbers::{Complex, Field};
use crate::tensors::operator::Apply;
use crate::tensors::{CsrMatrix, ElementOperator, Metal, MetalStorage, OperatorBackend};
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};
use std::{mem::size_of, ptr::NonNull};

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for crate::numbers::Complex<f32> {}
}

/// Scalars supported by the Metal linear operator kernels.
pub trait OperatorElement: Field + sealed::Sealed {
    #[doc(hidden)]
    const OPERATOR_INDEX: usize;
}
impl OperatorElement for f32 {
    const OPERATOR_INDEX: usize = 0;
}
impl OperatorElement for Complex<f32> {
    const OPERATOR_INDEX: usize = 1;
}

pub(super) fn device<T: Copy + 'static>(storage: &MetalStorage<T>) -> &MetalBuffer<T> {
    storage
        .device()
        .expect("Metal operator requires device-resident storage")
}

#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch<T: OperatorElement>(
    values: &MetalBuffer<T>,
    indices: [Option<&MetalBuffer<u32>>; 3],
    x: &MetalBuffer<T>,
    y: &MetalBuffer<T>,
    kind: u32,
    (rows, cols): (usize, usize),
    width: usize,
    count: usize,
    p: Apply<T>,
) {
    assert!(
        values.len() <= u32::MAX as usize,
        "Metal operator values exceed 32-bit indexing"
    );
    let input = match kind {
        0..=2 => {
            if p.adjoint {
                rows
            } else {
                cols
            }
        }
        3 | 4 => rows,
        6 | 9 => cols,
        8 => count,
        _ => 0,
    };
    assert!(
        p.input.checked_add(input).is_some_and(|end| end <= x.len()),
        "operator input window out of bounds"
    );
    let parameters = [
        kind as usize,
        rows,
        cols,
        width,
        p.input,
        p.output,
        usize::from(p.adjoint),
        count,
    ]
    .map(|n| u32::try_from(n).expect("Metal operator exceeds 32-bit indexing"));
    assert!(
        p.output
            .checked_add(count)
            .is_some_and(|end| end <= y.len()),
        "operator output window out of bounds"
    );
    if count == 0 {
        return;
    }
    with_gpu(|gpu| {
        let encoder = compute(gpu)?;
        let pipeline = &gpu.operators[T::OPERATOR_INDEX];
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(values.raw()), 0, 0);
            for (index, buffer) in indices.into_iter().enumerate() {
                // Unused arguments still receive a valid buffer. The shader
                // only dereferences indices needed by its selected kind.
                encoder.setBuffer_offset_atIndex(
                    Some(buffer.map_or(values.raw(), MetalBuffer::raw)),
                    0,
                    index + 1,
                );
            }
            encoder.setBuffer_offset_atIndex(Some(x.raw()), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(y.raw()), 0, 5);
            encoder.setBytes_length_atIndex(
                NonNull::from(&parameters).cast(),
                size_of::<[u32; 8]>(),
                6,
            );
            encoder.setBytes_length_atIndex(NonNull::from(&p.alpha).cast(), size_of::<T>(), 7);
            encoder.setBytes_length_atIndex(NonNull::from(&p.beta).cast(), size_of::<T>(), 8);
        }
        let threads = pipeline.threadExecutionWidth() * 4;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: count.div_ceil(threads),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
        queued(gpu, count)
    })
    .expect("Metal linear operator dispatch failed");
}

impl<T: OperatorElement> OperatorBackend<T> for Metal {
    const BATCHED_ARNOLDI: bool = true;
    fn orthogonalize(
        basis: &MetalStorage<T>,
        n: usize,
        count: usize,
        work: &mut MetalStorage<T>,
    ) -> (Vec<T>, f64, f64) {
        super::krylov::orthogonalize(basis, n, count, work)
    }
    fn triangular(
        a: &crate::tensors::TriangularOperator<T, Self>,
        x: &MetalStorage<T>,
        y: &mut MetalStorage<T>,
        p: Apply<T>,
    ) {
        super::krylov::triangular(a, x, y, p);
    }
    fn dense(
        a: &MetalStorage<T>,
        shape: (usize, usize),
        x: &MetalStorage<T>,
        y: &mut MetalStorage<T>,
        p: Apply<T>,
    ) {
        dispatch(
            device(a),
            [None; 3],
            device(x),
            device(y),
            0,
            shape,
            0,
            if p.adjoint { shape.1 } else { shape.0 },
            p,
        );
    }
    fn csr(a: &CsrMatrix<T, Self>, x: &MetalStorage<T>, y: &mut MetalStorage<T>, p: Apply<T>) {
        let indices = if p.adjoint {
            [&a.adjoint_offsets, &a.adjoint_columns, &a.adjoint_positions]
        } else {
            [&a.offsets, &a.columns, &a.adjoint_positions]
        };
        dispatch(
            device(&a.values),
            indices.map(|v| Some(device(v))),
            device(x),
            device(y),
            if p.adjoint { 2 } else { 1 },
            a.shape(),
            0,
            if p.adjoint { a.cols() } else { a.rows() },
            p,
        );
    }
    fn diagonal(
        a: &MetalStorage<T>,
        n: usize,
        x: &MetalStorage<T>,
        y: &mut MetalStorage<T>,
        p: Apply<T>,
    ) {
        dispatch(
            device(a),
            [None; 3],
            device(x),
            device(y),
            4,
            (n, n),
            0,
            n,
            p,
        );
    }
    fn elements(
        a: &ElementOperator<T, Self>,
        x: &MetalStorage<T>,
        y: &mut MetalStorage<T>,
        p: Apply<T>,
    ) {
        dispatch(
            device(&a.values),
            [
                Some(device(&a.offsets)),
                Some(device(&a.contributors)),
                Some(device(&a.dofs)),
            ],
            device(x),
            device(y),
            3,
            (a.size, a.size),
            a.width,
            a.size,
            p,
        );
    }
    fn scale(y: &mut MetalStorage<T>, offset: usize, len: usize, beta: T) {
        dispatch(
            device(y),
            [None; 3],
            device(y),
            device(y),
            5,
            (len, len),
            0,
            len,
            Apply {
                input: 0,
                output: offset,
                alpha: T::zero(),
                beta,
                adjoint: false,
            },
        );
    }
    fn axpby(x: &MetalStorage<T>, y: &mut MetalStorage<T>, len: usize, alpha: T, beta: T) {
        dispatch(
            device(x),
            [None; 3],
            device(x),
            device(y),
            8,
            (len, len),
            0,
            len,
            Apply {
                input: 0,
                output: 0,
                alpha,
                beta,
                adjoint: false,
            },
        );
    }
    fn inner(x: &MetalStorage<T>, y: &MetalStorage<T>, len: usize) -> T {
        assert_eq!(x.len(), len);
        assert_eq!(y.len(), len);
        operator_dotc(x, y)
    }
    fn norm(x: &MetalStorage<T>, len: usize) -> f64 {
        assert_eq!(x.len(), len);
        let count = len.div_ceil(256);
        let output = MetalBuffer::<T>::allocate(count).expect("Metal norm allocation failed");
        dispatch(
            device(x),
            [None; 3],
            device(x),
            &output,
            9,
            (count, len),
            0,
            count,
            Apply {
                input: 0,
                output: 0,
                alpha: T::one(),
                beta: T::zero(),
                adjoint: false,
            },
        );
        output
            .as_slice()
            .iter()
            .fold(0.0, |n: f64, v| n.hypot(v.magnitude()))
    }
}

pub(crate) fn operator_dotc<T: OperatorElement>(a: &MetalStorage<T>, b: &MetalStorage<T>) -> T {
    let (a, b) = (device(a), device(b));
    let count = a.len().div_ceil(256);
    let output = MetalBuffer::allocate(count).expect("Metal dotc allocation failed");
    dispatch(
        a,
        [None; 3],
        b,
        &output,
        6,
        (count, a.len()),
        0,
        count,
        Apply {
            input: 0,
            output: 0,
            alpha: T::one(),
            beta: T::zero(),
            adjoint: false,
        },
    );
    output
        .as_slice()
        .iter()
        .fold(T::zero(), |sum, &value| sum + value)
}

pub(crate) fn operator_adjoint<T: OperatorElement>(
    a: &MetalStorage<T>,
    shape: (usize, usize),
) -> MetalStorage<T> {
    let a = device(a);
    let output = MetalBuffer::allocate(a.len()).expect("Metal adjoint allocation failed");
    dispatch(
        a,
        [None; 3],
        a,
        &output,
        7,
        shape,
        0,
        a.len(),
        Apply {
            input: 0,
            output: 0,
            alpha: T::one(),
            beta: T::zero(),
            adjoint: false,
        },
    );
    MetalStorage::from_device(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensors::{BlockOperator, Host, LinearOperator, Matrix, Vector};

    #[test]
    fn mixed_block_application_encodes_without_host_waits() {
        let mut a = BlockOperator::<Complex<f32>, Metal>::new(&[2, 1], &[2, 1]).unwrap();
        a.set_block(
            0,
            0,
            CsrMatrix::from_triplets(
                2,
                2,
                [
                    (0, 0, Complex::new(1.0f32, 2.0)),
                    (1, 1, Complex::new(2.0, -1.0)),
                ],
            )
            .to_backend::<Metal>(),
        )
        .unwrap();
        a.set_block(
            1,
            1,
            Matrix::from_rows([[Complex::new(3.0, 1.0)]]).to_backend::<Metal>(),
        )
        .unwrap();
        let x = Vector::new([Complex::new(1.0, 0.0); 3]).to_backend::<Metal>();
        let mut y = Vector::<Complex<f32>>::zeros(3).to_backend::<Metal>();
        super::super::synchronize();
        let before = with_gpu(|gpu| Some(gpu.waits.get())).unwrap();
        a.apply(&x, &mut y);
        let after = with_gpu(|gpu| Some(gpu.waits.get())).unwrap();
        assert_eq!(before, after);
        assert_eq!(
            y.to_backend::<Host>().as_slice(),
            [
                Complex::new(1.0, 2.0),
                Complex::new(2.0, -1.0),
                Complex::new(3.0, 1.0)
            ]
        );
    }

    #[test]
    fn repeated_operator_calls_share_command_buffers() {
        let a = CsrMatrix::from_triplets(2, 2, [(0, 0, 2.0f32), (1, 1, 3.0)]).to_backend::<Metal>();
        let x = Vector::new([1.0f32, 1.0]).to_backend::<Metal>();
        let mut y = Vector::<f32>::zeros(2).to_backend::<Metal>();
        super::super::synchronize();
        let before =
            with_gpu(|gpu| Some((gpu.committed.get(), gpu.operations.get(), gpu.waits.get())))
                .unwrap();
        for _ in 0..70 {
            a.apply_add(1.0, &x, 1.0, &mut y);
        }
        let queued =
            with_gpu(|gpu| Some((gpu.committed.get(), gpu.operations.get(), gpu.waits.get())))
                .unwrap();
        assert_eq!(
            queued.0 - before.0,
            2,
            "70 small dispatches should submit two full batches before the final read"
        );
        assert_eq!(queued.1 - before.1, 70);
        assert_eq!(queued.2, before.2);
        assert_eq!(y.as_slice(), [140.0, 210.0]);
        let committed = with_gpu(|gpu| Some(gpu.committed.get())).unwrap();
        assert_eq!(
            committed - before.0,
            3,
            "final read should submit the remaining six dispatches once"
        );
    }
}
