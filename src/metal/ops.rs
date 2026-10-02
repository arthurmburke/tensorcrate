//! The operations on [`MetalBuffer`]: each validates its shapes, allocates
//! the output and encodes one or more kernels. A `None` means the caller should
//! run the CPU kernel instead.

use half::{bf16, f16};

use crate::tensors::{
    Analytic, Axis, BinaryOp, Compare, Family, Reduce, SortOrder, Statistic, Transposed,
};

use super::MetalElement;
use super::buffer::MetalBuffer;
use super::device::{Gpu, Operands, Pipeline, Tile, with_gpu};
use super::encode::{
    REDUCE_GROUP, encode_axis_distribution, encode_axis_moments, encode_bitonic_stage,
    encode_broadcast, encode_clamp, encode_compare, encode_compare_scalar, encode_concat,
    encode_convert, encode_correlate, encode_deviation, encode_distribution, encode_elementwise,
    encode_fft, encode_flip, encode_gemm, encode_hmerge, encode_matmul, encode_matrix_stack,
    encode_matvec, encode_pad, encode_power, encode_power_scalar, encode_reduce, encode_scan,
    encode_sort_prepare, encode_stack, encode_transpose, encode_unary, encode_unary_dual,
    encode_vecmat, encode_vmerge, vecmat_bands,
};

impl<T: MetalElement> MetalBuffer<T> {
    /// Matrix multiplication, with both inputs and the result remaining in
    /// shared Metal buffers. The products accumulate in `f32` and each output
    /// element rounds to `T` once.
    pub fn matmul(&self, rhs: &Self, m: usize, k: usize, n: usize) -> Option<Self> {
        if self.len != m.checked_mul(k)? || rhs.len != k.checked_mul(n)? {
            return None;
        }
        let output_len = m.checked_mul(n)?;
        if output_len == 0 {
            return Self::from_slice(&[]);
        }
        if k == 0 {
            return Self::from_slice(&vec![T::zero(); output_len]);
        }
        let output = Self::allocate(output_len)?;
        self.product_into(rhs, &output, (m, k, n), false)?;
        Some(output)
    }

    /// Encode `target = A·B`, or `target += A·B`: a product with one column or
    /// one row of output on the matrix–vector kernels, which spread it across
    /// the GPU where the tiles of a product would not; any other on the
    /// product kernels.
    fn product_into(
        &self,
        rhs: &Self,
        target: &Self,
        (m, k, n): (usize, usize, usize),
        accumulate: bool,
    ) -> Option<()> {
        if n == 1 {
            return with_gpu(|gpu| {
                encode_matvec::<T>(gpu, &self.raw, &rhs.raw, &target.raw, (m, k), accumulate)
            });
        }
        if m == 1 {
            let (bands, _) = vecmat_bands(k, n);
            let partial = MetalBuffer::<f32>::allocate(bands.checked_mul(n)?)?;
            return with_gpu(|gpu| {
                encode_vecmat::<T>(
                    gpu,
                    &self.raw,
                    &rhs.raw,
                    &partial.raw,
                    &target.raw,
                    (k, n),
                    accumulate,
                )
            });
        }
        with_gpu(|gpu| {
            encode_matmul::<T>(gpu, &self.raw, &rhs.raw, &target.raw, m, k, n, accumulate)
        })
    }

    /// Transpose a row-major `rows × cols` matrix into a new shared buffer.
    pub(crate) fn transpose(&self, rows: usize, cols: usize) -> Option<Self> {
        if self.len != rows.checked_mul(cols)? {
            return None;
        }
        let output = Self::allocate(self.len)?;
        if self.len != 0 {
            with_gpu(|gpu| encode_transpose::<T>(gpu, &self.raw, &output.raw, rows, cols))?;
        }
        Some(output)
    }

    /// `target += A·B`, accumulated by the matmul kernel itself rather than by a
    /// second elementwise pass.
    ///
    /// The exclusive borrow of `target` is what makes the in-place GPU write
    /// sound: no [`as_slice`](Self::as_slice) borrow can be alive at the same
    /// time, and distinct buffers never share an allocation.
    pub fn matmul_accumulate(
        &self,
        rhs: &Self,
        target: &mut Self,
        m: usize,
        k: usize,
        n: usize,
    ) -> Option<()> {
        if self.len != m.checked_mul(k)? || rhs.len != k.checked_mul(n)? {
            return None;
        }
        if target.len != m.checked_mul(n)? {
            return None;
        }
        // An empty result, or an empty inner dimension, adds nothing.
        if target.len == 0 || k == 0 {
            return Some(());
        }
        self.product_into(rhs, target, (m, k, n), true)
    }

    /// `target += op(A)·op(B)` for an `m × n` target, where `op` transposes
    /// the operand `transposed` names, read where it lies: `A` is stored
    /// `k × m` when it is the one transposed, `B` `n × k`.
    ///
    /// `None` without TensorOps, whose general product is what reads an
    /// operand transposed; the caller then transposes it into a copy.
    pub(crate) fn matmul_transposed_accumulate(
        &self,
        rhs: &Self,
        target: &mut Self,
        transposed: Transposed,
        (m, k, n): (usize, usize, usize),
    ) -> Option<()> {
        if self.len != m.checked_mul(k)? || rhs.len != k.checked_mul(n)? {
            return None;
        }
        if target.len != m.checked_mul(n)? {
            return None;
        }
        if target.len == 0 || k == 0 {
            return Some(());
        }
        let operands = match transposed {
            Transposed::Left => Operands::LeftTransposed,
            Transposed::Right => Operands::RightTransposed,
        };
        with_gpu(|gpu| {
            let (pipeline, tile) = gpu.gemm::<T>(operands, true, m, n)?;
            encode_gemm(
                gpu,
                &pipeline,
                &self.raw,
                &rhs.raw,
                &target.raw,
                (m, k, n),
                tile,
            )
        })
    }

    /// Apply an analytic function elementwise.
    pub fn unary(&self, op: Analytic) -> Option<Self> {
        let output = Self::allocate(self.len)?;
        if self.len != 0 {
            with_gpu(|gpu| encode_unary::<T>(gpu, &self.raw, &output.raw, self.len, op))?;
        }
        Some(output)
    }

    /// Apply an analytic function to a value/tangent pair — forward-mode
    /// differentiation, `f(v) + f'(v)·d·ε` — returning `(value, tangent)`.
    ///
    /// One dispatch produces both parts.
    pub fn unary_dual(&self, tangent: &Self, op: Analytic) -> Option<(Self, Self)> {
        if self.len != tangent.len {
            return None;
        }
        let out_value = Self::allocate(self.len)?;
        let out_tangent = Self::allocate(self.len)?;
        if self.len != 0 {
            with_gpu(|gpu| {
                encode_unary_dual::<T>(
                    gpu,
                    &self.raw,
                    &tangent.raw,
                    &out_value.raw,
                    &out_tangent.raw,
                    self.len,
                    op,
                )
            })?;
        }
        Some((out_value, out_tangent))
    }

    /// Elementwise operation with another shared buffer.
    pub fn elementwise(&self, rhs: &Self, op: BinaryOp) -> Option<Self> {
        if self.len != rhs.len || op == BinaryOp::Rem {
            return None;
        }
        let output = Self::allocate(self.len)?;
        if self.len != 0 {
            with_gpu(|gpu| {
                encode_elementwise::<T>(gpu, &self.raw, &rhs.raw, &output.raw, self.len, op)
            })?;
        }
        Some(output)
    }

    /// Elementwise `self^rhs`.
    pub fn power(&self, rhs: &Self) -> Option<Self> {
        if self.len != rhs.len {
            return None;
        }
        let output = Self::allocate(self.len)?;
        if self.len != 0 {
            with_gpu(|gpu| encode_power::<T>(gpu, &self.raw, &rhs.raw, &output.raw, self.len))?;
        }
        Some(output)
    }

    /// Elementwise power with one operand fixed. `scalar_left` selects
    /// `scalar^x` over `x^scalar`.
    ///
    /// Scalars cross to the shader as `f32`, which holds every `T` exactly.
    pub fn power_scalar(&self, scalar: T, scalar_left: bool) -> Option<Self> {
        let scalar = scalar.into_f64() as f32;
        let output = Self::allocate(self.len)?;
        if self.len != 0 {
            with_gpu(|gpu| {
                encode_power_scalar::<T>(gpu, &self.raw, &output.raw, self.len, scalar, scalar_left)
            })?;
        }
        Some(output)
    }

    /// Valid cross-correlation of an `rows × cols` input with a
    /// `window_rows × window_cols` kernel, both already resident.
    ///
    /// `flip` reverses the window, giving convolution rather than correlation.
    pub fn correlate(
        &self,
        weights: &Self,
        rows: usize,
        cols: usize,
        window_rows: usize,
        window_cols: usize,
        flip: bool,
    ) -> Option<Self> {
        if self.len != rows.checked_mul(cols)?
            || weights.len != window_rows.checked_mul(window_cols)?
        {
            return None;
        }
        let out_rows = rows.checked_sub(window_rows)? + 1;
        let out_cols = cols.checked_sub(window_cols)? + 1;
        let output = Self::allocate(out_rows.checked_mul(out_cols)?)?;
        if output.len != 0 {
            with_gpu(|gpu| {
                encode_correlate::<T>(
                    gpu,
                    &self.raw,
                    &weights.raw,
                    &output.raw,
                    rows,
                    cols,
                    window_rows,
                    window_cols,
                    flip,
                )
            })?;
        }
        Some(output)
    }

    /// Surround an `rows × cols` matrix with `pad_rows`/`pad_cols` zeros.
    pub fn pad(&self, rows: usize, cols: usize, pad_rows: usize, pad_cols: usize) -> Option<Self> {
        if self.len != rows.checked_mul(cols)? {
            return None;
        }
        let output = Self::allocate((rows + 2 * pad_rows).checked_mul(cols + 2 * pad_cols)?)?;
        if output.len != 0 {
            with_gpu(|gpu| {
                encode_pad::<T>(gpu, &self.raw, &output.raw, rows, cols, pad_rows, pad_cols)
            })?;
        }
        Some(output)
    }

    /// Reverse both axes of an `rows × cols` matrix.
    pub fn flip(&self, rows: usize, cols: usize) -> Option<Self> {
        if self.len != rows.checked_mul(cols)? {
            return None;
        }
        let output = Self::allocate(self.len)?;
        if self.len != 0 {
            with_gpu(|gpu| encode_flip::<T>(gpu, &self.raw, &output.raw, rows, cols))?;
        }
        Some(output)
    }

    /// Elementwise comparison with another shared buffer.
    pub fn compare(&self, rhs: &Self, op: Compare) -> Option<Self> {
        if self.len != rhs.len {
            return None;
        }
        let output = Self::allocate(self.len)?;
        if self.len != 0 {
            with_gpu(|gpu| {
                encode_compare::<T>(gpu, &self.raw, &rhs.raw, &output.raw, self.len, op)
            })?;
        }
        Some(output)
    }

    /// Elementwise comparison against a scalar; `scalar_left` puts the scalar on
    /// the left, which matters for [`Compare::MaxShare`].
    pub fn compare_scalar(&self, scalar: T, op: Compare, scalar_left: bool) -> Option<Self> {
        let scalar = scalar.into_f64() as f32;
        let output = Self::allocate(self.len)?;
        if self.len != 0 {
            with_gpu(|gpu| {
                encode_compare_scalar::<T>(
                    gpu,
                    &self.raw,
                    &output.raw,
                    self.len,
                    scalar,
                    op,
                    scalar_left,
                )
            })?;
        }
        Some(output)
    }

    /// Confine every element to `[low, high]`, in one dispatch.
    pub fn clamp(&self, low: T, high: T) -> Option<Self> {
        let (low, high) = (low.into_f64() as f32, high.into_f64() as f32);
        let output = Self::allocate(self.len)?;
        if self.len != 0 {
            with_gpu(|gpu| encode_clamp::<T>(gpu, &self.raw, &output.raw, self.len, low, high))?;
        }
        Some(output)
    }

    /// Fold the whole buffer to one value with a tree reduction.
    ///
    /// Each round folds `REDUCE_GROUP` values per threadgroup, so the length
    /// falls by that factor per dispatch — three rounds for a million elements.
    /// The answer has to come back to the CPU, so this is one of the few
    /// operations that ends in a synchronization rather than leaving work
    /// queued.
    ///
    /// The fold runs in `f32` whatever `T` is, and the `f32` total is what
    /// comes back: rounding it to `T` is the caller's one rounding, and a mean
    /// or a variance built from it should use the unrounded value.
    pub fn reduce(&self, op: Reduce) -> Option<f32> {
        if self.len == 0 {
            return Some(op.identity());
        }
        if self.len == 1 {
            return Some(self.as_slice()[0].into_f64() as f32);
        }

        // Ping-pong between two `f32` scratch buffers: a round reads one and
        // writes the (much shorter) other. Only the first round reads `T`.
        let mut groups = self.len.div_ceil(REDUCE_GROUP);
        let mut front = MetalBuffer::<f32>::allocate(groups)?;
        with_gpu(|gpu| encode_reduce::<T>(gpu, &self.raw, &front.raw, self.len, groups, op))?;
        let mut count = groups;
        if count == 1 {
            return Some(front.as_slice()[0]);
        }

        let mut back = MetalBuffer::<f32>::allocate(count.div_ceil(REDUCE_GROUP))?;
        while count > 1 {
            groups = count.div_ceil(REDUCE_GROUP);
            with_gpu(|gpu| encode_reduce::<f32>(gpu, &front.raw, &back.raw, count, groups, op))?;
            std::mem::swap(&mut front, &mut back);
            count = groups;
        }
        Some(front.as_slice()[0])
    }

    /// Inclusive prefix sum, `log2(len)` dispatches deep.
    ///
    /// Every sweep reads one buffer and writes the other, so the two allocations
    /// alternate and the result is whichever one the last sweep wrote. The
    /// additions land in a different order from the CPU's running total, which
    /// is a rounding difference rather than a disagreement.
    ///
    /// The running totals are `f32`: a 16-bit buffer is widened into an `f32`
    /// one, scanned there, and each total rounds to `T` once on the way back.
    pub fn prefix_sum(&self) -> Option<Self> {
        if self.len <= 1 {
            return Self::from_slice(self.as_slice());
        }
        let mut front = MetalBuffer::<f32>::allocate(self.len)?;
        with_gpu(|gpu| {
            encode_convert(
                gpu,
                &gpu.kernels::<T>().widen,
                &self.raw,
                &front.raw,
                self.len,
            )
        })?;
        let mut back = MetalBuffer::<f32>::allocate(self.len)?;
        let mut offset = 1;
        while offset < self.len {
            with_gpu(|gpu| encode_scan(gpu, &front.raw, &back.raw, self.len, offset))?;
            std::mem::swap(&mut front, &mut back);
            offset *= 2;
        }
        let output = Self::allocate(self.len)?;
        with_gpu(|gpu| {
            encode_convert(
                gpu,
                &gpu.kernels::<T>().narrow,
                &front.raw,
                &output.raw,
                self.len,
            )
        })?;
        Some(output)
    }

    /// Sort the elements in IEEE total order, on the GPU.
    ///
    /// A bitonic sort: `log²` stages of compare-exchange over a power-of-two
    /// buffer, each stage one dispatch. The input is padded up to that length
    /// with a value that sorts to the far end, so trimming the tail afterwards
    /// leaves exactly the input's elements. The shader compares monotone
    /// integer keys rather than the floats themselves, which is what makes the
    /// result identical to a CPU total-order sort (`f32::total_cmp`, or the
    /// 16-bit types' own) rather than merely similar: NaNs and `−0.0` land where
    /// the total order puts them instead of wherever an unordered
    /// compare-exchange left them.
    pub fn sort(&self, order: SortOrder) -> Option<Self> {
        if self.len <= 1 {
            return Self::from_slice(self.as_slice());
        }
        let padded = self.len.checked_next_power_of_two()?;
        let ascending = order == SortOrder::Ascending;
        let padding = T::sort_padding(ascending);

        let buffer = Self::allocate(padded)?;
        with_gpu(|gpu| {
            encode_sort_prepare::<T>(gpu, &self.raw, &buffer.raw, padded, self.len, padding)?;
            let mut block = 2;
            while block <= padded {
                let mut stride = block / 2;
                while stride > 0 {
                    encode_bitonic_stage::<T>(gpu, &buffer.raw, padded, block, stride, ascending)?;
                    stride /= 2;
                }
                block *= 2;
            }
            Some(())
        })?;

        if padded == self.len {
            return Some(buffer);
        }
        // Trim the padding. The values are in shared memory, so this reads the
        // sorted prefix in place rather than downloading it.
        Self::from_slice(&buffer.as_slice()[..self.len])
    }

    /// `Σ(xᵢ − mean)²` over the whole buffer.
    ///
    /// The first round is its own kernel, which forms and squares each
    /// deviation as it reads the value; every round after it is the ordinary
    /// summing reduction over the partials. So the buffer is read once, not
    /// once to write an elementwise result and again to fold it.
    ///
    /// Like [`reduce`](Self::reduce), this accumulates in `f32` and returns the
    /// unrounded `f32` sum; `mean` is the unrounded `f32` mean.
    pub fn sum_squared_deviations(&self, mean: f32) -> Option<f32> {
        if self.len == 0 {
            return Some(0.0);
        }
        if self.len == 1 {
            let deviation = self.as_slice()[0].into_f64() as f32 - mean;
            return Some(deviation * deviation);
        }

        let mut groups = self.len.div_ceil(REDUCE_GROUP);
        let mut front = MetalBuffer::<f32>::allocate(groups)?;
        with_gpu(|gpu| encode_deviation::<T>(gpu, &self.raw, &front.raw, self.len, groups, mean))?;
        let mut count = groups;
        if count == 1 {
            return Some(front.as_slice()[0]);
        }

        let mut back = MetalBuffer::<f32>::allocate(count.div_ceil(REDUCE_GROUP))?;
        while count > 1 {
            groups = count.div_ceil(REDUCE_GROUP);
            with_gpu(|gpu| {
                encode_reduce::<f32>(gpu, &front.raw, &back.raw, count, groups, Reduce::Sum)
            })?;
            std::mem::swap(&mut front, &mut back);
            count = groups;
        }
        Some(front.as_slice()[0])
    }

    /// One mean and one `Σ(xᵢ − mean)²` per row or per column of a
    /// `rows × cols` matrix, as `(means, deviations)`.
    ///
    /// One dispatch with a thread per result, rather than one whole-buffer
    /// reduction per slice: a `1024 × 1024` matrix reduced by rows is a
    /// thousand folds of a thousand values each, and a thousand separate
    /// dispatches would cost more in command buffers than in arithmetic.
    pub fn axis_moments(&self, rows: usize, cols: usize, axis: Axis) -> Option<(Self, Self)> {
        if self.len != rows.checked_mul(cols)? {
            return None;
        }
        let extent = axis.extent((rows, cols));
        let means = Self::allocate(extent)?;
        let deviations = Self::allocate(extent)?;
        if extent != 0 && axis.depth((rows, cols)) != 0 {
            with_gpu(|gpu| {
                encode_axis_moments::<T>(
                    gpu,
                    &self.raw,
                    &means.raw,
                    &deviations.raw,
                    rows,
                    cols,
                    extent,
                    axis,
                )
            })?;
        }
        Some((means, deviations))
    }

    /// A distribution function applied elementwise, with one parameter pair for
    /// the whole buffer.
    pub fn distribution(
        &self,
        family: Family,
        statistic: Statistic,
        parameters: (f32, f32),
    ) -> Option<Self> {
        let output = Self::allocate(self.len)?;
        if self.len != 0 {
            with_gpu(|gpu| {
                encode_distribution::<T>(
                    gpu,
                    &self.raw,
                    &output.raw,
                    self.len,
                    family,
                    statistic,
                    parameters,
                )
            })?;
        }
        Some(output)
    }

    /// The same, with a parameter pair per row or per column.
    ///
    /// `first` and `second` hold one parameter each per slice along `axis`, in
    /// the order [`axis_moments`](Self::axis_moments) produces them.
    #[allow(clippy::too_many_arguments)]
    pub fn axis_distribution(
        &self,
        first: &Self,
        second: &Self,
        rows: usize,
        cols: usize,
        axis: Axis,
        family: Family,
        statistic: Statistic,
    ) -> Option<Self> {
        let extent = axis.extent((rows, cols));
        if self.len != rows.checked_mul(cols)? || first.len != extent || second.len != extent {
            return None;
        }
        let output = Self::allocate(self.len)?;
        if self.len != 0 {
            with_gpu(|gpu| {
                encode_axis_distribution::<T>(
                    gpu,
                    &self.raw,
                    &output.raw,
                    &first.raw,
                    &second.raw,
                    self.len,
                    cols,
                    axis,
                    family,
                    statistic,
                )
            })?;
        }
        Some(output)
    }

    /// Broadcast operation with a scalar. `op` has the same encoding as
    /// [`elementwise`](Self::elementwise).
    pub fn broadcast(&self, scalar: T, op: BinaryOp, scalar_left: bool) -> Option<Self> {
        if op == BinaryOp::Rem {
            return None;
        }
        let scalar = scalar.into_f64() as f32;
        let output = Self::allocate(self.len)?;
        if self.len != 0 {
            with_gpu(|gpu| {
                encode_broadcast::<T>(
                    gpu,
                    &self.raw,
                    &output.raw,
                    self.len,
                    scalar,
                    op,
                    scalar_left,
                )
            })?;
        }
        Some(output)
    }

    /// Stack equal-length buffers as rows of one row-major matrix.
    pub(crate) fn vstack(inputs: &[&Self], vector_len: usize) -> Option<Self> {
        Self::stack(inputs, vector_len, 1, |index| index * vector_len)
    }

    /// Stack equal-length buffers as columns of one row-major matrix.
    pub(crate) fn hstack(inputs: &[&Self], vector_len: usize) -> Option<Self> {
        let columns = inputs.len();
        Self::stack(inputs, vector_len, columns, |index| index)
    }

    fn stack(
        inputs: &[&Self],
        vector_len: usize,
        output_stride: usize,
        offset: impl Fn(usize) -> usize,
    ) -> Option<Self> {
        if inputs.iter().any(|input| input.len != vector_len) {
            return None;
        }
        let output = Self::allocate(inputs.len().checked_mul(vector_len)?)?;
        if output.len != 0 {
            with_gpu(|gpu| {
                encode_stack::<T>(gpu, inputs, &output.raw, vector_len, output_stride, offset)
            })?;
        }
        Some(output)
    }

    /// Concatenate two row-major matrices horizontally.
    pub(crate) fn concat_matrix(
        &self,
        rhs: &Self,
        rows: usize,
        left_cols: usize,
        right_cols: usize,
    ) -> Option<Self> {
        if self.len != rows.checked_mul(left_cols)? || rhs.len != rows.checked_mul(right_cols)? {
            return None;
        }
        let output_cols = left_cols.checked_add(right_cols)?;
        let output = Self::allocate(rows.checked_mul(output_cols)?)?;
        if output.len != 0 {
            with_gpu(|gpu| {
                encode_concat::<T>(
                    gpu,
                    &self.raw,
                    &rhs.raw,
                    &output.raw,
                    rows,
                    left_cols,
                    right_cols,
                )
            })?;
        }
        Some(output)
    }

    /// Concatenate two row-major matrices vertically using contiguous blits.
    pub(crate) fn stack_matrix(
        &self,
        rhs: &Self,
        top_rows: usize,
        bottom_rows: usize,
        cols: usize,
    ) -> Option<Self> {
        if self.len != top_rows.checked_mul(cols)? || rhs.len != bottom_rows.checked_mul(cols)? {
            return None;
        }
        let output = Self::allocate(self.len.checked_add(rhs.len)?)?;
        if output.len != 0 {
            with_gpu(|gpu| {
                encode_matrix_stack::<T>(gpu, &self.raw, &rhs.raw, &output.raw, self.len, rhs.len)
            })?;
        }
        Some(output)
    }

    /// Merge equally shaped row-major matrices horizontally.
    pub(crate) fn hmerge(inputs: &[&Self], rows: usize, cols: usize) -> Option<Self> {
        let matrix_len = rows.checked_mul(cols)?;
        if inputs.iter().any(|input| input.len != matrix_len) {
            return None;
        }
        let output = Self::allocate(matrix_len.checked_mul(inputs.len())?)?;
        if output.len != 0 {
            with_gpu(|gpu| encode_hmerge::<T>(gpu, inputs, &output.raw, rows, cols))?;
        }
        Some(output)
    }

    /// Merge equally shaped row-major matrices vertically with contiguous blits.
    pub(crate) fn vmerge(inputs: &[&Self], rows: usize, cols: usize) -> Option<Self> {
        let matrix_len = rows.checked_mul(cols)?;
        if inputs.iter().any(|input| input.len != matrix_len) {
            return None;
        }
        let output = Self::allocate(matrix_len.checked_mul(inputs.len())?)?;
        if output.len != 0 {
            with_gpu(|gpu| encode_vmerge::<T>(gpu, inputs, &output.raw, matrix_len))?;
        }
        Some(output)
    }
}

impl MetalBuffer<f32> {
    /// Radix-2 FFT over interleaved complex values. The layout is
    /// `[real0, imag0, real1, imag1, ...]`.
    pub fn fft(&self) -> Option<Self> {
        self.fourier_transform(false)
    }

    /// Normalized inverse radix-2 FFT over interleaved complex values.
    pub fn ifft(&self) -> Option<Self> {
        self.fourier_transform(true)
    }

    fn fourier_transform(&self, inverse: bool) -> Option<Self> {
        if !self.len.is_multiple_of(2) {
            return None;
        }
        let count = self.len / 2;
        if count == 0 {
            return Self::from_slice(&[]);
        }
        if !count.is_power_of_two() || count > u32::MAX as usize {
            return None;
        }
        let output = Self::allocate(self.len)?;
        with_gpu(|gpu| encode_fft(gpu, &self.raw, &output.raw, count, inverse))?;
        Some(output)
    }
}

impl MetalBuffer<f16> {
    /// Multiply FP16 inputs with TensorOps and accumulate into FP32 output.
    pub(crate) fn matmul_f32(
        &self,
        rhs: &Self,
        m: usize,
        k: usize,
        n: usize,
    ) -> Option<MetalBuffer<f32>> {
        matmul_tensorops(self, rhs, m, k, n, 0.0, |gpu| {
            gpu.gemm_widening("f16", m, n)
        })
    }
}

impl MetalBuffer<bf16> {
    /// Multiply BF16 inputs with TensorOps and accumulate into FP32 output.
    pub(crate) fn matmul_f32(
        &self,
        rhs: &Self,
        m: usize,
        k: usize,
        n: usize,
    ) -> Option<MetalBuffer<f32>> {
        matmul_tensorops(self, rhs, m, k, n, 0.0, |gpu| {
            gpu.gemm_widening("bf16", m, n)
        })
    }
}

pub(super) fn matmul_tensorops<T: Copy + 'static, U: Copy + 'static>(
    left: &MetalBuffer<T>,
    right: &MetalBuffer<T>,
    m: usize,
    k: usize,
    n: usize,
    zero: U,
    pipeline: impl Fn(&Gpu) -> Option<(Pipeline, Tile)>,
) -> Option<MetalBuffer<U>> {
    if left.len != m.checked_mul(k)? || right.len != k.checked_mul(n)? {
        return None;
    }
    let output_len = m.checked_mul(n)?;
    if output_len == 0 {
        return MetalBuffer::from_slice(&[]);
    }
    if k == 0 {
        return MetalBuffer::from_slice(&vec![zero; output_len]);
    }
    let output = MetalBuffer::<U>::allocate(output_len)?;
    with_gpu(|gpu| {
        let (state, tile) = pipeline(gpu)?;
        encode_gemm(
            gpu,
            &state,
            &left.raw,
            &right.raw,
            &output.raw,
            (m, k, n),
            tile,
        )
    })?;
    Some(output)
}
