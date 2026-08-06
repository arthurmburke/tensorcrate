//! Where a tensor's elements live.
//!
//! [`Vector`](super::Vector) and [`Matrix`](super::Matrix) carry a backend type
//! parameter that selects their storage. It defaults to [`Host`] — the flat
//! row-major `Vec` they normally use — so `Vector<f64>` and `Matrix<i64>` keep
//! meaning exactly what they meant before.
//!
//! The [`Metal`] backend (macOS, `metal` feature) instead keeps elements in
//! `MTLStorageModeShared` memory, which the GPU kernels read and write in place.
//! The full kernel set operates on `f32`; M5 TensorOps matrix products also use
//! compact `f16` and `bf16` storage. A chain of operations over `Metal`-backed
//! tensors therefore stays resident: no operand is uploaded and no result is
//! downloaded until you ask for one. The [`Host`] backend's automatic offload
//! (the `MIN_*` thresholds in [`crate::metal`]) cannot do that — it has to copy
//! both operands in and the result back out on every single call, which is why
//! its thresholds are so high.
//!
//! Move between backends explicitly with
//! [`Vector::to_backend`](super::Vector::to_backend) and
//! [`Matrix::to_backend`](super::Matrix::to_backend):
//!
//! ```
//! # #[cfg(all(feature = "metal", target_os = "macos"))] {
//! use tensorcrate::tensors::{Host, Matrix, Metal};
//!
//! let m: Matrix<f32> = Matrix::from_rows([[1.0, 2.0], [3.0, 4.0]]);
//! let resident = m.to_backend::<Metal>(); // one upload
//! let cubed = resident.matmul(&resident).matmul(&resident); // no copies
//! assert_eq!(cubed.to_backend::<Host>(), m.matmul(&m).matmul(&m));
//! # }
//! ```
//!
//! The backend never implicitly changes an element type. Compact matrices offer
//! both same-format [`Matrix::matmul`](super::Matrix::matmul) output and an
//! explicit `matmul_f32` widening path.
//!
//! # Shapes
//!
//! Storage here is shapeless: it is a run of values, and the extents live in the
//! [`Vector`](super::Vector) and [`Matrix`](super::Matrix) wrappers. That is why
//! the reshaping operations below take no arguments — a row vector, a column
//! vector and their flattening are all the same run of elements, so on either
//! backend the conversion is a move rather than a copy. Operations that really
//! do depend on the extents, like [`concat`](Backend::concat), take them as
//! ordinary parameters.

/// A tensor storage backend.
///
/// The trait is sealed: [`Host`] and [`Metal`] are the only implementations,
/// since the kernels behind them are part of this crate.
pub trait Backend: sealed::Sealed + Sized + 'static {
    /// Storage for a vector of `T`.
    type Vector<T>;

    /// Storage for a matrix of `T`, in row-major order.
    type Matrix<T>;

    /// Build vector storage from typed values.
    fn store_vector<T: Copy + 'static>(values: &[T]) -> Self::Vector<T>;

    /// Build matrix storage from typed row-major values.
    fn store_matrix<T: Copy + 'static>(values: &[T]) -> Self::Matrix<T>;

    /// Borrow this vector storage as a flat typed slice.
    ///
    /// Nothing is copied: for the [`Metal`] backend this borrows the shared
    /// allocation itself, which the CPU can read directly.
    fn vector_slice<T: Copy + 'static>(storage: &Self::Vector<T>) -> &[T];

    /// Borrow this matrix storage as a flat row-major typed slice, again without
    /// copying.
    fn matrix_slice<T: Copy + 'static>(storage: &Self::Matrix<T>) -> &[T];

    /// Reinterpret a vector as a matrix, filling rows in order.
    ///
    /// This and [`matrix_into_flattened`](Self::matrix_into_flattened) are what
    /// let a matrix input be differentiated by the vector machinery. Both are
    /// free on either backend — the elements are already in the right order —
    /// which is why they take ownership rather than borrowing.
    fn vector_into_matrix<T: Copy + 'static>(vector: Self::Vector<T>) -> Self::Matrix<T>;

    /// Reinterpret a matrix as its row-major flattening.
    fn matrix_into_flattened<T: Copy + 'static>(matrix: Self::Matrix<T>) -> Self::Vector<T>;

    /// Build a matrix from vectors stacked along the vertical axis (the vectors
    /// are rows), each of length `len`.
    fn vstack(vectors: &[Self::Vector<f32>], len: usize) -> Self::Matrix<f32>;

    /// Build a matrix from vectors stacked along the horizontal axis (the
    /// vectors are columns), each of length `len`.
    fn hstack(vectors: &[Self::Vector<f32>], len: usize) -> Self::Matrix<f32>;

    /// Build a matrix by placing two `rows`-tall matrices side by side.
    fn concat(
        a: &Self::Matrix<f32>,
        b: &Self::Matrix<f32>,
        rows: usize,
        left_cols: usize,
        right_cols: usize,
    ) -> Self::Matrix<f32>;

    /// Build a matrix by placing two `cols`-wide matrices one above the other.
    fn stack(
        a: &Self::Matrix<f32>,
        b: &Self::Matrix<f32>,
        top_rows: usize,
        bottom_rows: usize,
        cols: usize,
    ) -> Self::Matrix<f32>;

    /// Build a matrix by placing several `rows × cols` matrices side by side.
    fn hmerge(matrices: &[Self::Matrix<f32>], rows: usize, cols: usize) -> Self::Matrix<f32>;

    /// Build a matrix by stacking several `rows × cols` matrices vertically.
    fn vmerge(matrices: &[Self::Matrix<f32>], rows: usize, cols: usize) -> Self::Matrix<f32>;
}

mod sealed {
    pub trait Sealed {}
}

/// The default backend: elements live in a flat row-major [`Vec`].
///
/// Every element type is supported.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Host;

impl sealed::Sealed for Host {}

impl Backend for Host {
    type Vector<T> = Vec<T>;
    type Matrix<T> = Vec<T>;

    fn store_vector<T: Copy + 'static>(values: &[T]) -> Vec<T> {
        values.to_vec()
    }

    fn store_matrix<T: Copy + 'static>(values: &[T]) -> Vec<T> {
        values.to_vec()
    }

    fn vector_slice<T: Copy + 'static>(storage: &Vec<T>) -> &[T] {
        storage
    }

    fn matrix_slice<T: Copy + 'static>(storage: &Vec<T>) -> &[T] {
        storage
    }

    // A vector and a matrix are the same run of elements in the same order, so
    // both reshapes are the identity.
    fn vector_into_matrix<T: Copy + 'static>(vector: Vec<T>) -> Vec<T> {
        vector
    }

    fn matrix_into_flattened<T: Copy + 'static>(matrix: Vec<T>) -> Vec<T> {
        matrix
    }

    fn vstack(vectors: &[Vec<f32>], len: usize) -> Vec<f32> {
        let mut values = Vec::with_capacity(vectors.len() * len);
        for vector in vectors {
            debug_assert_eq!(vector.len(), len);
            values.extend_from_slice(vector);
        }
        values
    }

    fn hstack(vectors: &[Vec<f32>], len: usize) -> Vec<f32> {
        let cols = vectors.len();
        let mut values = vec![0.0; len * cols];
        for (col, vector) in vectors.iter().enumerate() {
            debug_assert_eq!(vector.len(), len);
            for (row, value) in vector.iter().enumerate() {
                values[row * cols + col] = *value;
            }
        }
        values
    }

    fn concat(
        a: &Vec<f32>,
        b: &Vec<f32>,
        rows: usize,
        left_cols: usize,
        right_cols: usize,
    ) -> Vec<f32> {
        let mut values = Vec::with_capacity(rows * (left_cols + right_cols));
        for row in 0..rows {
            values.extend_from_slice(&a[row * left_cols..(row + 1) * left_cols]);
            values.extend_from_slice(&b[row * right_cols..(row + 1) * right_cols]);
        }
        values
    }

    fn stack(
        a: &Vec<f32>,
        b: &Vec<f32>,
        top_rows: usize,
        bottom_rows: usize,
        cols: usize,
    ) -> Vec<f32> {
        let mut values = Vec::with_capacity((top_rows + bottom_rows) * cols);
        values.extend_from_slice(a);
        values.extend_from_slice(b);
        values
    }

    fn hmerge(matrices: &[Vec<f32>], rows: usize, cols: usize) -> Vec<f32> {
        let mut values = Vec::with_capacity(rows * cols * matrices.len());
        for row in 0..rows {
            for matrix in matrices {
                values.extend_from_slice(&matrix[row * cols..(row + 1) * cols]);
            }
        }
        values
    }

    fn vmerge(matrices: &[Vec<f32>], rows: usize, cols: usize) -> Vec<f32> {
        let mut values = Vec::with_capacity(rows * cols * matrices.len());
        for matrix in matrices {
            values.extend_from_slice(matrix);
        }
        values
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub use gpu::{Metal, MetalStorage};

#[cfg(all(feature = "metal", target_os = "macos"))]
mod gpu {
    use std::fmt;

    use half::{bf16, f16};

    use super::{Backend, sealed};
    use crate::metal::MetalBuffer;
    use crate::tensors::{Analytic, Axis, BinaryOp, Compare, Family, Reduce, SortOrder, Statistic};

    /// A backend that keeps elements in GPU-shared memory, so the Metal kernels
    /// read and write them in place.
    ///
    /// `f32` has the complete Metal operation set. `f16` and `bf16` use the
    /// crate's re-exported [`half`](https://docs.rs/half) types for compact M5
    /// TensorOps products.
    ///
    /// Metal objects are thread-affine, so these tensors are not `Send`. They do
    /// clone (into a fresh allocation with the same values) and compare by
    /// value.
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
    pub struct Metal;

    impl sealed::Sealed for Metal {}

    impl Backend for Metal {
        type Vector<T> = MetalStorage<T>;
        type Matrix<T> = MetalStorage<T>;

        fn store_vector<T: Copy + 'static>(values: &[T]) -> MetalStorage<T> {
            MetalStorage::from_slice(values)
        }

        fn store_matrix<T: Copy + 'static>(values: &[T]) -> MetalStorage<T> {
            MetalStorage::from_slice(values)
        }

        fn vector_slice<T: Copy + 'static>(storage: &MetalStorage<T>) -> &[T] {
            storage.as_slice()
        }

        fn matrix_slice<T: Copy + 'static>(storage: &MetalStorage<T>) -> &[T] {
            storage.as_slice()
        }

        // A vector and a matrix are the same shared allocation, and row-major
        // flattening is the identity on it: these move, they do not copy.
        fn vector_into_matrix<T: Copy + 'static>(vector: MetalStorage<T>) -> MetalStorage<T> {
            vector
        }

        fn matrix_into_flattened<T: Copy + 'static>(matrix: MetalStorage<T>) -> MetalStorage<T> {
            matrix
        }

        fn vstack(vectors: &[MetalStorage], len: usize) -> MetalStorage {
            if let Some(storage) = MetalStorage::vstack(vectors, len) {
                return storage;
            }
            let mut values = Vec::with_capacity(vectors.len() * len);
            for vector in vectors {
                values.extend_from_slice(vector.as_slice());
            }
            MetalStorage::from_slice(&values)
        }

        fn hstack(vectors: &[MetalStorage], len: usize) -> MetalStorage {
            if let Some(storage) = MetalStorage::hstack(vectors, len) {
                return storage;
            }
            let cols = vectors.len();
            let mut values = vec![0.0; len * cols];
            for (col, vector) in vectors.iter().enumerate() {
                for (row, value) in vector.as_slice().iter().enumerate() {
                    values[row * cols + col] = *value;
                }
            }
            MetalStorage::from_slice(&values)
        }

        fn concat(
            a: &MetalStorage,
            b: &MetalStorage,
            rows: usize,
            left_cols: usize,
            right_cols: usize,
        ) -> MetalStorage {
            if let Some(storage) = a.concat(b, rows, left_cols, right_cols) {
                return storage;
            }
            let (left, right) = (a.as_slice(), b.as_slice());
            let mut values = Vec::with_capacity(rows * (left_cols + right_cols));
            for row in 0..rows {
                values.extend_from_slice(&left[row * left_cols..(row + 1) * left_cols]);
                values.extend_from_slice(&right[row * right_cols..(row + 1) * right_cols]);
            }
            MetalStorage::from_slice(&values)
        }

        fn stack(
            a: &MetalStorage,
            b: &MetalStorage,
            top_rows: usize,
            bottom_rows: usize,
            cols: usize,
        ) -> MetalStorage {
            if let Some(storage) = a.stack(b, top_rows, bottom_rows, cols) {
                return storage;
            }
            let mut values = Vec::with_capacity((top_rows + bottom_rows) * cols);
            values.extend_from_slice(a.as_slice());
            values.extend_from_slice(b.as_slice());
            MetalStorage::from_slice(&values)
        }

        fn hmerge(matrices: &[MetalStorage], rows: usize, cols: usize) -> MetalStorage {
            if let Some(storage) = MetalStorage::hmerge(matrices, rows, cols) {
                return storage;
            }
            let mut values = Vec::with_capacity(rows * cols * matrices.len());
            for row in 0..rows {
                for matrix in matrices {
                    values.extend_from_slice(&matrix.as_slice()[row * cols..(row + 1) * cols]);
                }
            }
            MetalStorage::from_slice(&values)
        }

        fn vmerge(matrices: &[MetalStorage], rows: usize, cols: usize) -> MetalStorage {
            if let Some(storage) = MetalStorage::vmerge(matrices, rows, cols) {
                return storage;
            }
            let mut values = Vec::with_capacity(rows * cols * matrices.len());
            for matrix in matrices {
                values.extend_from_slice(matrix.as_slice());
            }
            MetalStorage::from_slice(&values)
        }
    }

    /// The allocation behind a [`Metal`]-backed tensor.
    ///
    /// Normally this is GPU-shared memory that kernels read in place. When the
    /// process has no Metal device — or an allocation fails — the values live in
    /// an ordinary `Vec` instead and operations fall back to the [`Host`] paths.
    /// Choosing a backend says where the data should live; it never changes the
    /// numbers that come out.
    ///
    /// [`Host`]: super::Host
    pub struct MetalStorage<T = f32>(Residency<T>);

    enum Residency<T> {
        /// A shared allocation the GPU can read without a copy.
        Device(MetalBuffer<T>),
        /// No Metal device was available; the CPU kernels run over this instead.
        Host(Vec<T>),
    }

    impl<T: Copy + 'static> MetalStorage<T> {
        pub(crate) fn from_slice(values: &[T]) -> Self {
            match MetalBuffer::from_slice(values) {
                Some(buffer) => Self(Residency::Device(buffer)),
                None => Self(Residency::Host(values.to_vec())),
            }
        }

        /// Whether the values really are in GPU-shared memory.
        ///
        /// `false` means no Metal device was available and this tensor is
        /// CPU-resident; its operations still produce the same results.
        pub fn is_device_resident(&self) -> bool {
            matches!(self.0, Residency::Device(_))
        }

        /// Number of stored values.
        pub fn len(&self) -> usize {
            match &self.0 {
                Residency::Device(buffer) => buffer.len(),
                Residency::Host(values) => values.len(),
            }
        }

        /// Whether this allocation holds no values.
        pub fn is_empty(&self) -> bool {
            self.len() == 0
        }

        /// Borrow the values as a flat slice, without copying: shared storage is
        /// ordinary cached memory as far as the CPU is concerned.
        pub(crate) fn as_slice(&self) -> &[T] {
            match &self.0 {
                Residency::Device(buffer) => buffer.as_slice(),
                Residency::Host(values) => values,
            }
        }

        /// The shared allocation, when there is one.
        fn device(&self) -> Option<&MetalBuffer<T>> {
            match &self.0 {
                Residency::Device(buffer) => Some(buffer),
                Residency::Host(_) => None,
            }
        }
    }

    impl MetalStorage<f32> {
        fn vstack(inputs: &[Self], vector_len: usize) -> Option<Self> {
            let buffers = inputs
                .iter()
                .map(Self::device)
                .collect::<Option<Vec<_>>>()?;
            Some(Self(Residency::Device(MetalBuffer::vstack(
                &buffers, vector_len,
            )?)))
        }

        fn hstack(inputs: &[Self], vector_len: usize) -> Option<Self> {
            let buffers = inputs
                .iter()
                .map(Self::device)
                .collect::<Option<Vec<_>>>()?;
            Some(Self(Residency::Device(MetalBuffer::hstack(
                &buffers, vector_len,
            )?)))
        }

        fn concat(
            &self,
            rhs: &Self,
            rows: usize,
            left_cols: usize,
            right_cols: usize,
        ) -> Option<Self> {
            Some(Self(Residency::Device(self.device()?.concat_matrix(
                rhs.device()?,
                rows,
                left_cols,
                right_cols,
            )?)))
        }

        fn stack(
            &self,
            rhs: &Self,
            top_rows: usize,
            bottom_rows: usize,
            cols: usize,
        ) -> Option<Self> {
            Some(Self(Residency::Device(self.device()?.stack_matrix(
                rhs.device()?,
                top_rows,
                bottom_rows,
                cols,
            )?)))
        }

        fn hmerge(inputs: &[Self], rows: usize, cols: usize) -> Option<Self> {
            let buffers = inputs
                .iter()
                .map(Self::device)
                .collect::<Option<Vec<_>>>()?;
            Some(Self(Residency::Device(MetalBuffer::hmerge(
                &buffers, rows, cols,
            )?)))
        }

        fn vmerge(inputs: &[Self], rows: usize, cols: usize) -> Option<Self> {
            let buffers = inputs
                .iter()
                .map(Self::device)
                .collect::<Option<Vec<_>>>()?;
            Some(Self(Residency::Device(MetalBuffer::vmerge(
                &buffers, rows, cols,
            )?)))
        }

        pub(crate) fn transpose(&self, rows: usize, cols: usize) -> Option<Self> {
            Some(Self(Residency::Device(
                self.device()?.transpose(rows, cols)?,
            )))
        }

        /// `C[m×n] = A[m×k] · B[k×n]`, entirely on the GPU. `None` when either
        /// operand is not device-resident or the dispatch failed, leaving the
        /// caller to use its host path.
        pub(crate) fn matmul(&self, rhs: &Self, m: usize, k: usize, n: usize) -> Option<Self> {
            let product = self.device()?.matmul(rhs.device()?, m, k, n)?;
            Some(Self(Residency::Device(product)))
        }

        /// Elementwise operation on the GPU. The shaders have no remainder
        /// kernel, so [`BinaryOp::Rem`] returns `None`.
        pub(crate) fn elementwise(&self, rhs: &Self, op: BinaryOp) -> Option<Self> {
            let output = self.device()?.elementwise(rhs.device()?, op)?;
            Some(Self(Residency::Device(output)))
        }

        /// Scalar broadcast on the GPU, with `op` encoded as in
        /// [`elementwise`](Self::elementwise).
        pub(crate) fn broadcast(
            &self,
            scalar: f32,
            op: BinaryOp,
            scalar_left: bool,
        ) -> Option<Self> {
            let output = self.device()?.broadcast(scalar, op, scalar_left)?;
            Some(Self(Residency::Device(output)))
        }

        /// `target += A[m×k]·B[k×n]`, accumulated by the matmul kernel itself.
        ///
        /// `None` when any of the three is not device-resident, leaving the
        /// caller to add with its host path.
        pub(crate) fn matmul_accumulate(
            &self,
            rhs: &Self,
            target: &mut Self,
            m: usize,
            k: usize,
            n: usize,
        ) -> Option<()> {
            let (a, b) = (self.device()?, rhs.device()?);
            let Residency::Device(target) = &mut target.0 else {
                return None;
            };
            a.matmul_accumulate(b, target, m, k, n)
        }

        /// Analytic function applied to a value/tangent pair, in one dispatch;
        pub(crate) fn unary_dual(&self, tangent: &Self, op: Analytic) -> Option<(Self, Self)> {
            let (value, tangent) = self.device()?.unary_dual(tangent.device()?, op)?;
            Some((
                Self(Residency::Device(value)),
                Self(Residency::Device(tangent)),
            ))
        }

        /// Valid cross-correlation, on the GPU.
        pub(crate) fn correlate(
            &self,
            weights: &Self,
            rows: usize,
            cols: usize,
            window_rows: usize,
            window_cols: usize,
            flip: bool,
        ) -> Option<Self> {
            let output = self.device()?.correlate(
                weights.device()?,
                rows,
                cols,
                window_rows,
                window_cols,
                flip,
            )?;
            Some(Self(Residency::Device(output)))
        }

        /// Reversal of both axes, on the GPU.
        pub(crate) fn flip(&self, rows: usize, cols: usize) -> Option<Self> {
            Some(Self(Residency::Device(self.device()?.flip(rows, cols)?)))
        }

        /// Zero padding, on the GPU.
        pub(crate) fn pad(
            &self,
            rows: usize,
            cols: usize,
            pad_rows: usize,
            pad_cols: usize,
        ) -> Option<Self> {
            let output = self.device()?.pad(rows, cols, pad_rows, pad_cols)?;
            Some(Self(Residency::Device(output)))
        }

        /// Elementwise comparison with another shared allocation.
        pub(crate) fn compare(&self, rhs: &Self, op: Compare) -> Option<Self> {
            let output = self.device()?.compare(rhs.device()?, op)?;
            Some(Self(Residency::Device(output)))
        }

        /// Elementwise comparison against a scalar.
        pub(crate) fn compare_scalar(
            &self,
            scalar: f32,
            op: Compare,
            scalar_left: bool,
        ) -> Option<Self> {
            let output = self.device()?.compare_scalar(scalar, op, scalar_left)?;
            Some(Self(Residency::Device(output)))
        }

        /// Elementwise clamp to `[low, high]`.
        pub(crate) fn clamp(&self, low: f32, high: f32) -> Option<Self> {
            Some(Self(Residency::Device(self.device()?.clamp(low, high)?)))
        }

        /// Whole-buffer fold. The result is a number rather than an allocation,
        /// so unlike its neighbours this one ends on the CPU by construction.
        pub(crate) fn reduce(&self, op: Reduce) -> Option<f32> {
            self.device()?.reduce(op)
        }

        /// Inclusive prefix sum, staying in shared memory.
        pub(crate) fn prefix_sum(&self) -> Option<Self> {
            Some(Self(Residency::Device(self.device()?.prefix_sum()?)))
        }

        /// Sort in IEEE total order, staying in shared memory.
        pub(crate) fn sort(&self, order: SortOrder) -> Option<Self> {
            Some(Self(Residency::Device(self.device()?.sort(order)?)))
        }

        /// Elementwise `self^rhs`.
        pub(crate) fn power(&self, rhs: &Self) -> Option<Self> {
            Some(Self(Residency::Device(self.device()?.power(rhs.device()?)?)))
        }

        /// Elementwise power with one operand fixed.
        pub(crate) fn power_scalar(&self, scalar: f32, scalar_left: bool) -> Option<Self> {
            let output = self.device()?.power_scalar(scalar, scalar_left)?;
            Some(Self(Residency::Device(output)))
        }

        /// Analytic function applied elementwise; `op` is an
        pub(crate) fn unary(&self, op: Analytic) -> Option<Self> {
            Some(Self(Residency::Device(self.device()?.unary(op)?)))
        }

        /// `Σ(xᵢ − mean)²` over the whole allocation. Like
        /// [`reduce`](Self::reduce) this ends on the CPU, since its result is a
        /// number.
        pub(crate) fn sum_squared_deviations(&self, mean: f32) -> Option<f32> {
            self.device()?.sum_squared_deviations(mean)
        }

        /// Per-row or per-column means and deviation sums, both staying in
        /// shared memory.
        pub(crate) fn axis_moments(
            &self,
            rows: usize,
            cols: usize,
            axis: Axis,
        ) -> Option<(Self, Self)> {
            let (means, deviations) = self.device()?.axis_moments(rows, cols, axis)?;
            Some((
                Self(Residency::Device(means)),
                Self(Residency::Device(deviations)),
            ))
        }

        /// A distribution function applied elementwise.
        pub(crate) fn distribution(
            &self,
            family: Family,
            statistic: Statistic,
            parameters: (f32, f32),
        ) -> Option<Self> {
            let output = self.device()?.distribution(family, statistic, parameters)?;
            Some(Self(Residency::Device(output)))
        }

        /// The same, with one parameter pair per row or column.
        pub(crate) fn axis_distribution(
            &self,
            first: &Self,
            second: &Self,
            shape: (usize, usize),
            axis: Axis,
            family: Family,
            statistic: Statistic,
        ) -> Option<Self> {
            let output = self.device()?.axis_distribution(
                first.device()?,
                second.device()?,
                shape.0,
                shape.1,
                axis,
                family,
                statistic,
            )?;
            Some(Self(Residency::Device(output)))
        }
    }

    macro_rules! low_precision_storage {
        ($ty:ty) => {
            impl MetalStorage<$ty> {
                pub(crate) fn matmul(
                    &self,
                    rhs: &Self,
                    m: usize,
                    k: usize,
                    n: usize,
                ) -> Option<Self> {
                    let product = self.device()?.matmul(rhs.device()?, m, k, n)?;
                    Some(Self(Residency::Device(product)))
                }

                pub(crate) fn matmul_f32(
                    &self,
                    rhs: &Self,
                    m: usize,
                    k: usize,
                    n: usize,
                ) -> Option<MetalStorage<f32>> {
                    let product = self.device()?.matmul_f32(rhs.device()?, m, k, n)?;
                    Some(MetalStorage(Residency::Device(product)))
                }
            }
        };
    }

    low_precision_storage!(f16);
    low_precision_storage!(bf16);

    impl<T: Copy + 'static> Clone for MetalStorage<T> {
        /// Clones the values into a fresh allocation on the same backend.
        fn clone(&self) -> Self {
            Self::from_slice(self.as_slice())
        }
    }

    impl<T: Copy + PartialEq + 'static> PartialEq for MetalStorage<T> {
        fn eq(&self, other: &Self) -> bool {
            self.as_slice() == other.as_slice()
        }
    }

    impl<T: Copy + fmt::Debug + 'static> fmt::Debug for MetalStorage<T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            self.as_slice().fmt(f)
        }
    }
}
