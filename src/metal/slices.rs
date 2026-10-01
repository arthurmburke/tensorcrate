//! Convenience functions over ordinary CPU slices: each uploads its inputs,
//! runs one operation and downloads the result.

use crate::tensors::BinaryOp;

use super::buffer::{download, upload};
use super::device::with_gpu;
use super::encode::{encode_broadcast, encode_elementwise, encode_fft};
use super::sync::sync;

/// GPU elementwise `f32` op over two equal-length buffers. `op` is 0=add,
/// 1=sub, 2=mul, 3=div. Returns `None` if no device is available.
pub fn elementwise_f32(a: &[f32], b: &[f32], op: BinaryOp) -> Option<Vec<f32>> {
    if a.len() != b.len() || op == BinaryOp::Rem {
        return None;
    }
    let len = a.len();
    if len == 0 {
        return Some(Vec::new());
    }
    with_gpu(|gpu| {
        let (buf_a, buf_b, buf_c) = {
            let mut pool = gpu.pool.borrow_mut();
            (
                pool.acquire(&gpu.device, len * 4)?,
                pool.acquire(&gpu.device, len * 4)?,
                pool.acquire(&gpu.device, len * 4)?,
            )
        };
        upload(&buf_a, a);
        upload(&buf_b, b);

        encode_elementwise::<f32>(gpu, &buf_a, &buf_b, &buf_c, len, op)?;

        sync(gpu)?;
        let out = download(&buf_c, len);
        let mut pool = gpu.pool.borrow_mut();
        pool.release(buf_a, false);
        pool.release(buf_b, false);
        pool.release(buf_c, false);
        Some(out)
    })
}

/// GPU `f32` broadcast operation between a buffer and a scalar. `op` uses the
/// same encoding as [`elementwise_f32`]; `scalar_left` controls operand order
/// for subtraction and division.
pub fn broadcast_f32(
    values: &[f32],
    scalar: f32,
    op: BinaryOp,
    scalar_left: bool,
) -> Option<Vec<f32>> {
    if op == BinaryOp::Rem {
        return None;
    }
    let len = values.len();
    if len == 0 {
        return Some(Vec::new());
    }
    with_gpu(|gpu| {
        let (input, output) = {
            let mut pool = gpu.pool.borrow_mut();
            (
                pool.acquire(&gpu.device, len * 4)?,
                pool.acquire(&gpu.device, len * 4)?,
            )
        };
        upload(&input, values);

        encode_broadcast::<f32>(gpu, &input, &output, len, scalar, op, scalar_left)?;

        sync(gpu)?;
        let out = download(&output, len);
        let mut pool = gpu.pool.borrow_mut();
        pool.release(input, false);
        pool.release(output, false);
        Some(out)
    })
}

/// GPU radix-2 FFT over interleaved complex `f32` values. The input and output
/// layout is `[real0, imag0, real1, imag1, ...]`.
pub fn fft_f32_interleaved(input: &[f32]) -> Option<Vec<f32>> {
    fourier_transform_f32_interleaved(input, false)
}

/// GPU normalized inverse radix-2 FFT over interleaved complex `f32` values.
pub fn ifft_f32_interleaved(input: &[f32]) -> Option<Vec<f32>> {
    fourier_transform_f32_interleaved(input, true)
}

pub(super) fn fourier_transform_f32_interleaved(input: &[f32], inverse: bool) -> Option<Vec<f32>> {
    if !input.len().is_multiple_of(2) {
        return None;
    }
    let count = input.len() / 2;
    if count == 0 {
        return Some(Vec::new());
    }
    if !count.is_power_of_two() || count > u32::MAX as usize {
        return None;
    }

    with_gpu(|gpu| {
        let (source, values) = {
            let mut pool = gpu.pool.borrow_mut();
            (
                pool.acquire(&gpu.device, input.len() * 4)?,
                pool.acquire(&gpu.device, input.len() * 4)?,
            )
        };
        upload(&source, input);

        encode_fft(gpu, &source, &values, count, inverse)?;
        sync(gpu)?;
        let out = download(&values, input.len());
        let mut pool = gpu.pool.borrow_mut();
        pool.release(source, false);
        pool.release(values, false);
        Some(out)
    })
}
