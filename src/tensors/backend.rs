//! Where a tensor's elements live.
//!
//! [`Vector`](super::Vector) and [`Matrix`](super::Matrix) carry a backend type
//! parameter that selects their storage. It defaults to [`Host`] — the plain
//! fixed-size array they have always used — so `Vector<f64, 3>` and
//! `Matrix<i64, 2, 2>` keep meaning exactly what they meant before.
//!
//! The [`Metal`] backend (macOS, `metal` feature) instead keeps `f32` elements
//! in `MTLStorageModeShared` memory, which the GPU kernels read and write in
//! place. A chain of products or elementwise operations over `Metal`-backed
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
//! use rinterp::tensors::{Host, Matrix, Metal};
//!
//! let m: Matrix<f32, 2, 2> = Matrix::from_rows([[1.0, 2.0], [3.0, 4.0]]);
//! let resident = m.to_backend::<Metal>(); // one upload
//! let cubed = resident.matmul(&resident).matmul(&resident); // no copies
//! assert_eq!(cubed.to_backend::<Host>(), m.matmul(&m).matmul(&m));
//! # }
//! ```
//!
//! Only `f32` tensors can be built on a non-[`Host`] backend: the Metal shaders
//! are 32-bit, and `f32` is the only element type they understand. The backend
//! never changes an answer — it only decides which memory the answer is
//! computed in.

/// A tensor storage backend.
///
/// The trait is sealed: [`Host`] and [`Metal`] are the only implementations,
/// since the kernels behind them are part of this crate.
pub trait Backend: sealed::Sealed + Sized + 'static {
    /// Storage for a length-`N` vector of `T`.
    type Vector<T, const N: usize>;

    /// Storage for an `R × C` matrix of `T`, in row-major order.
    type Matrix<T, const R: usize, const C: usize>;

    /// Build vector storage from `N` `f32` values.
    fn store_vector<const N: usize>(values: &[f32]) -> Self::Vector<f32, N>;

    /// Reinterpret a vector as a `1 × N` row matrix.
    fn vector_into_row<const N: usize>(vector: Self::Vector<f32, N>) -> Self::Matrix<f32, 1, N>;

    /// Reinterpret a vector as an `N × 1` column matrix.
    fn vector_into_column<const N: usize>(vector: Self::Vector<f32, N>) -> Self::Matrix<f32, N, 1>;

    /// Build a matrix from a collection of vectors stacked along the
    /// vertical axis (vectors are row vectors).
    fn vstack<const M: usize, const N: usize>(
        vectors: [Self::Vector<f32, N>; M],
    ) -> Self::Matrix<f32, M, N>;

    /// Build a matrix from a collection of vectors stacked along the
    /// horizontal axis (vectors are column vectors).
    fn hstack<const M: usize, const N: usize>(
        vectors: [Self::Vector<f32, M>; N],
    ) -> Self::Matrix<f32, M, N>;

    /// Builds a matrix by stacking two matrices horizontally.
    fn concat<const M: usize, const N1: usize, const N2: usize>(
        a: Self::Matrix<f32, M, N1>,
        b: Self::Matrix<f32, M, N2>,
    ) -> Self::Matrix<f32, M, { N1 + N2 }>;

    /// Builds a matrix by stacking two matrices vertically.
    fn stack<const M1: usize, const M2: usize, const N: usize>(
        a: Self::Matrix<f32, M1, N>,
        b: Self::Matrix<f32, M2, N>,
    ) -> Self::Matrix<f32, { M1 + M2 }, N>;

    /// Builds a matrix by concating a collection of matrices with
    /// the same number of columns along the horizontal axis.
    fn hmerge<const M: usize, const N: usize, const K: usize>(
        matrices: [Self::Matrix<f32, M, N>; K],
    ) -> Self::Matrix<f32, M, { K * N }>;

    /// Builds a matrix by concating a collection of matrices with
    /// the same number of columns along the vertical axis.
    fn vmerge<const M: usize, const N: usize, const K: usize>(
        matrices: [Self::Matrix<f32, M, N>; K],
    ) -> Self::Matrix<f32, { K * M }, N>;

    /// Borrow this vector storage as a flat `f32` slice of length `N`.
    ///
    /// Nothing is copied: for the [`Metal`] backend this borrows the shared
    /// allocation itself, which the CPU can read directly.
    fn vector_slice<const N: usize>(storage: &Self::Vector<f32, N>) -> &[f32];

    /// Build matrix storage from `R * C` row-major `f32` values.
    fn store_matrix<const R: usize, const C: usize>(values: &[f32]) -> Self::Matrix<f32, R, C>;

    /// Borrow this matrix storage as a flat row-major `f32` slice of length
    /// `R * C`, again without copying.
    fn matrix_slice<const R: usize, const C: usize>(storage: &Self::Matrix<f32, R, C>) -> &[f32];
}

mod sealed {
    pub trait Sealed {}
}

/// The default backend: elements live in a fixed-size array, on the stack.
///
/// Every element type is supported, and the tensors are `Copy` whenever the
/// element type is.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Host;

impl sealed::Sealed for Host {}

impl Backend for Host {
    type Vector<T, const N: usize> = [T; N];
    type Matrix<T, const R: usize, const C: usize> = [[T; C]; R];

    fn store_vector<const N: usize>(values: &[f32]) -> [f32; N] {
        debug_assert_eq!(values.len(), N);
        std::array::from_fn(|index| values[index])
    }

    fn vector_into_row<const N: usize>(vector: [f32; N]) -> [[f32; N]; 1] {
        [vector]
    }

    fn vector_into_column<const N: usize>(vector: [f32; N]) -> [[f32; 1]; N] {
        vector.map(|value| [value])
    }

    fn vector_slice<const N: usize>(storage: &[f32; N]) -> &[f32] {
        storage
    }

    fn store_matrix<const R: usize, const C: usize>(values: &[f32]) -> [[f32; C]; R] {
        debug_assert_eq!(values.len(), R * C);
        std::array::from_fn(|row| std::array::from_fn(|col| values[row * C + col]))
    }

    fn matrix_slice<const R: usize, const C: usize>(storage: &[[f32; C]; R]) -> &[f32] {
        // SAFETY: nested arrays are contiguous, so the rows are exactly R*C
        // adjacent `f32` with no padding.
        unsafe { std::slice::from_raw_parts(storage.as_ptr().cast::<f32>(), R * C) }
    }

    fn vstack<const M: usize, const N: usize>(
        vectors: [Self::Vector<f32, N>; M],
    ) -> Self::Matrix<f32, M, N> {
        std::array::from_fn(|row| vectors[row])
    }

    fn hstack<const M: usize, const N: usize>(
        vectors: [Self::Vector<f32, M>; N],
    ) -> Self::Matrix<f32, M, N> {
        std::array::from_fn(|row| std::array::from_fn(|col| vectors[col][row]))
    }

    fn concat<const M: usize, const N1: usize, const N2: usize>(
        a: Self::Matrix<f32, M, N1>,
        b: Self::Matrix<f32, M, N2>,
    ) -> Self::Matrix<f32, M, { N1 + N2 }> {
        std::array::from_fn(|row| {
            std::array::from_fn(|col| {
                if col < N1 {
                    a[row][col]
                } else {
                    b[row][col - N1]
                }
            })
        })
    }

    fn stack<const M1: usize, const M2: usize, const N: usize>(
        a: Self::Matrix<f32, M1, N>,
        b: Self::Matrix<f32, M2, N>,
    ) -> Self::Matrix<f32, { M1 + M2 }, N> {
        std::array::from_fn(|row| if row < M1 { a[row] } else { b[row - M1] })
    }

    fn hmerge<const M: usize, const N: usize, const K: usize>(
        matrices: [Self::Matrix<f32, M, N>; K],
    ) -> Self::Matrix<f32, M, { K * N }> {
        std::array::from_fn(|row| std::array::from_fn(|col| matrices[col / N][row][col % N]))
    }

    fn vmerge<const M: usize, const N: usize, const K: usize>(
        matrices: [Self::Matrix<f32, M, N>; K],
    ) -> Self::Matrix<f32, { K * M }, N> {
        std::array::from_fn(|row| matrices[row / M][row % M])
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub use gpu::{Metal, MetalStorage};

#[cfg(all(feature = "metal", target_os = "macos"))]
mod gpu {
    use std::fmt;

    use super::{Backend, sealed};
    use crate::metal::MetalBuffer;

    /// A backend that keeps `f32` elements in GPU-shared memory, so the Metal
    /// kernels read and write them in place.
    ///
    /// Only `f32` tensors can be built here: `Vector<f32, N, Metal>` and
    /// `Matrix<f32, R, C, Metal>` have constructors, any other element type
    /// names a type with none.
    ///
    /// Metal objects are thread-affine, so these tensors are not `Send`, and —
    /// a shared allocation not being a fixed-size array — not `Copy` either.
    /// They do clone (into a fresh allocation with the same values) and compare
    /// by value.
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
    pub struct Metal;

    impl sealed::Sealed for Metal {}

    impl Backend for Metal {
        type Vector<T, const N: usize> = MetalStorage;
        type Matrix<T, const R: usize, const C: usize> = MetalStorage;

        fn store_vector<const N: usize>(values: &[f32]) -> MetalStorage {
            debug_assert_eq!(values.len(), N);
            MetalStorage::from_slice(values)
        }

        fn vector_into_row<const N: usize>(vector: MetalStorage) -> MetalStorage {
            vector
        }

        fn vector_into_column<const N: usize>(vector: MetalStorage) -> MetalStorage {
            vector
        }

        fn vector_slice<const N: usize>(storage: &MetalStorage) -> &[f32] {
            storage.as_slice()
        }

        fn store_matrix<const R: usize, const C: usize>(values: &[f32]) -> MetalStorage {
            debug_assert_eq!(values.len(), R * C);
            MetalStorage::from_slice(values)
        }

        fn matrix_slice<const R: usize, const C: usize>(storage: &MetalStorage) -> &[f32] {
            storage.as_slice()
        }

        fn vstack<const M: usize, const N: usize>(
            vectors: [Self::Vector<f32, N>; M],
        ) -> Self::Matrix<f32, M, N> {
            if let Some(storage) = MetalStorage::vstack(&vectors, N) {
                return storage;
            }
            let mut values = Vec::with_capacity(M * N);
            for vector in &vectors {
                values.extend_from_slice(vector.as_slice());
            }
            MetalStorage::from_slice(&values)
        }

        fn hstack<const M: usize, const N: usize>(
            vectors: [Self::Vector<f32, M>; N],
        ) -> Self::Matrix<f32, M, N> {
            if let Some(storage) = MetalStorage::hstack(&vectors, M) {
                return storage;
            }
            let mut values = vec![0.0; M * N];
            for (col, vector) in vectors.iter().enumerate() {
                for (row, value) in vector.as_slice().iter().enumerate() {
                    values[row * N + col] = *value;
                }
            }
            MetalStorage::from_slice(&values)
        }

        fn concat<const M: usize, const N1: usize, const N2: usize>(
            a: Self::Matrix<f32, M, N1>,
            b: Self::Matrix<f32, M, N2>,
        ) -> Self::Matrix<f32, M, { N1 + N2 }> {
            if let Some(storage) = a.concat(&b, M, N1, N2) {
                return storage;
            }
            let (left, right) = (a.as_slice(), b.as_slice());
            let mut values = Vec::with_capacity(M * (N1 + N2));
            for row in 0..M {
                values.extend_from_slice(&left[row * N1..(row + 1) * N1]);
                values.extend_from_slice(&right[row * N2..(row + 1) * N2]);
            }
            MetalStorage::from_slice(&values)
        }

        fn stack<const M1: usize, const M2: usize, const N: usize>(
            a: Self::Matrix<f32, M1, N>,
            b: Self::Matrix<f32, M2, N>,
        ) -> Self::Matrix<f32, { M1 + M2 }, N> {
            if let Some(storage) = a.stack(&b, M1, M2, N) {
                return storage;
            }
            let mut values = Vec::with_capacity((M1 + M2) * N);
            values.extend_from_slice(a.as_slice());
            values.extend_from_slice(b.as_slice());
            MetalStorage::from_slice(&values)
        }

        fn hmerge<const M: usize, const N: usize, const K: usize>(
            matrices: [Self::Matrix<f32, M, N>; K],
        ) -> Self::Matrix<f32, M, { K * N }> {
            if let Some(storage) = MetalStorage::hmerge(&matrices, M, N) {
                return storage;
            }
            let slices = matrices
                .iter()
                .map(Self::matrix_slice::<M, N>)
                .collect::<Vec<_>>();
            let mut values = Vec::with_capacity(M * K * N);
            for row in 0..M {
                for matrix in &slices {
                    values.extend_from_slice(&matrix[row * N..(row + 1) * N]);
                }
            }
            MetalStorage::from_slice(&values)
        }

        fn vmerge<const M: usize, const N: usize, const K: usize>(
            matrices: [Self::Matrix<f32, M, N>; K],
        ) -> Self::Matrix<f32, { K * M }, N> {
            if let Some(storage) = MetalStorage::vmerge(&matrices, M, N) {
                return storage;
            }
            let mut values = Vec::with_capacity(K * M * N);
            for matrix in &matrices {
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
    pub struct MetalStorage(Residency);

    enum Residency {
        /// A shared allocation the GPU can read without a copy.
        Device(MetalBuffer),
        /// No Metal device was available; the CPU kernels run over this instead.
        Host(Vec<f32>),
    }

    impl MetalStorage {
        pub(crate) fn from_slice(values: &[f32]) -> Self {
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

        /// Number of stored `f32` values.
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
        pub(crate) fn as_slice(&self) -> &[f32] {
            match &self.0 {
                Residency::Device(buffer) => buffer.as_slice(),
                Residency::Host(values) => values,
            }
        }

        /// The shared allocation, when there is one.
        fn device(&self) -> Option<&MetalBuffer> {
            match &self.0 {
                Residency::Device(buffer) => Some(buffer),
                Residency::Host(_) => None,
            }
        }

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

        /// Elementwise operation on the GPU; `op` is 0=add, 1=sub, 2=mul, 3=div.
        /// The shaders have no remainder kernel, so `op == 4` returns `None`.
        pub(crate) fn elementwise(&self, rhs: &Self, op: u32) -> Option<Self> {
            let output = self.device()?.elementwise(rhs.device()?, op)?;
            Some(Self(Residency::Device(output)))
        }

        /// Scalar broadcast on the GPU, with `op` encoded as in
        /// [`elementwise`](Self::elementwise).
        pub(crate) fn broadcast(&self, scalar: f32, op: u32, scalar_left: bool) -> Option<Self> {
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
        /// `op` is an [`Analytic::code`](crate::tensors::Analytic::code).
        pub(crate) fn unary_dual(&self, tangent: &Self, op: u32) -> Option<(Self, Self)> {
            let (value, tangent) = self.device()?.unary_dual(tangent.device()?, op)?;
            Some((
                Self(Residency::Device(value)),
                Self(Residency::Device(tangent)),
            ))
        }

        /// Analytic function applied elementwise; `op` is an
        /// [`Analytic::code`](crate::tensors::Analytic::code).
        pub(crate) fn unary(&self, op: u32) -> Option<Self> {
            Some(Self(Residency::Device(self.device()?.unary(op)?)))
        }
    }

    impl Clone for MetalStorage {
        /// Clones the values into a fresh allocation on the same backend.
        fn clone(&self) -> Self {
            Self::from_slice(self.as_slice())
        }
    }

    impl PartialEq for MetalStorage {
        fn eq(&self, other: &Self) -> bool {
            self.as_slice() == other.as_slice()
        }
    }

    impl fmt::Debug for MetalStorage {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            self.as_slice().fmt(f)
        }
    }
}
