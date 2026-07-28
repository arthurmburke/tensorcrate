//! Statically-shaped vectors and matrices.
//!
//! [`Vector<T, N>`] and [`Matrix<T, R, C>`] carry their dimensions as const
//! generic parameters, so the shapes are part of the type and the compiler
//! checks them. A matrix product `Matrix<R, K> · Matrix<K, C>` only type-checks
//! when the inner dimensions agree; adding a `Matrix<2, 3>` to a `Matrix<3, 2>`
//! is a compile error, not a runtime one. The only operation that can still fail
//! at runtime is [`Matrix::inverse`], because singularity is a property of the
//! values, not the shape.
//!
//! Storage is a fixed-size array (`[T; N]` / `[[T; C]; R]`), so these live on the
//! stack and are `Copy` when `T` is. `+ - * /` are elementwise; the
//! linear-algebra products are the named methods.
//!
//! Both types take a third parameter, the storage [`Backend`], which defaults to
//! [`Host`] — the fixed-size array just described. On macOS with the `metal`
//! feature, `f32` tensors can instead be placed on the [`Metal`] backend, whose
//! elements live in GPU-shared memory so a chain of operations runs without
//! copying between CPU and GPU pools. [`Vector::to_backend`] and
//! [`Matrix::to_backend`] move between the two; see the [`backend`] module for
//! the details.

use std::cmp::Ordering;
use std::fmt::{self, Display};
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

use num_traits::{Float, NumCast};

use crate::errors::Error;
use crate::numbers::{Coefficient, Complex};

pub mod backend;
pub mod dual;
pub mod kernels;
pub mod tape;

/// The `Metal`-backed inherent operations. It declares no new types, so there is
/// nothing to re-export — naming the module is what puts the methods on the
/// tensors.
#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal_backend;

pub use backend::{Backend, Host};
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use backend::{Metal, MetalStorage};
pub use dual::{
    DualMatrix, DualVector, gradient, gradient_wrt_matrix, jacobian, jacobian_wrt_matrix,
    matrix_gradient,
};
pub use kernels::{Analytic, BinaryOp, Compare, Kernels};
pub use tape::{MatrixVar, ScalarVar, Tape, Var, VectorVar};

#[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
mod simd_dispatch {
    //! CPU SIMD tier: sits between `metal_dispatch` and the generic scalar
    //! loops. Each entry point downcasts the generic element type to a concrete
    //! float via `TypeId` (returning `None` — i.e. defer to scalar — for every
    //! other type), then calls the architecture-specific kernels in
    //! [`crate::simd`]. The size gates are deliberately small: CPU SIMD has
    //! almost no fixed cost, so it wins far below the Metal thresholds.

    use std::any::TypeId;

    use super::{BinaryOp, Complex, Matrix, Vector};
    use crate::numbers::Coefficient;

    // Below these lengths the generic scalar loop is already fine (and the
    // reinterpret/dispatch bookkeeping is not worth it).
    const MIN_ELEMENTS: usize = 16;
    const MIN_MATMUL_OPS: usize = 512;
    const MIN_FFT_LENGTH: usize = 8;

    fn is<T: 'static, U: 'static>() -> bool {
        TypeId::of::<T>() == TypeId::of::<U>()
    }

    /// Reinterpret `&[T]` as `&[U]` when `T` is exactly `U`. SAFETY rests on the
    /// `TypeId` equality: identical type ⇒ identical layout and lifetime.
    unsafe fn as_slice<T: 'static, U: 'static>(values: &[T]) -> Option<&[U]> {
        if !is::<T, U>() {
            return None;
        }
        Some(unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<U>(), values.len()) })
    }

    unsafe fn as_slice_mut<T: 'static, U: 'static>(values: &mut [T]) -> Option<&mut [U]> {
        if !is::<T, U>() {
            return None;
        }
        Some(unsafe {
            std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<U>(), values.len())
        })
    }

    fn from_f32<T: Copy + 'static>(v: f32) -> T {
        unsafe { std::ptr::read((&v as *const f32).cast::<T>()) }
    }
    fn from_f64<T: Copy + 'static>(v: f64) -> T {
        unsafe { std::ptr::read((&v as *const f64).cast::<T>()) }
    }

    pub fn dot<T: Coefficient, const N: usize>(a: &[T; N], b: &[T; N]) -> Option<T> {
        if N < MIN_ELEMENTS {
            return None;
        }
        unsafe {
            if let (Some(a), Some(b)) = (as_slice::<T, f32>(a), as_slice::<T, f32>(b)) {
                return Some(from_f32(crate::simd::f32k::dot(a, b)));
            }
            if let (Some(a), Some(b)) = (as_slice::<T, f64>(a), as_slice::<T, f64>(b)) {
                return Some(from_f64(crate::simd::f64k::dot(a, b)));
            }
        }
        None
    }

    pub fn elementwise<T: Coefficient>(a: &[T], b: &[T], op: BinaryOp) -> Option<Vec<T>> {
        if a.len() < MIN_ELEMENTS || op == BinaryOp::Rem {
            return None;
        }
        unsafe {
            if let (Some(a), Some(b)) = (as_slice::<T, f32>(a), as_slice::<T, f32>(b)) {
                let mut out = vec![0.0f32; a.len()];
                crate::simd::f32k::elementwise(a, b, op, &mut out);
                return Some(out.into_iter().map(from_f32).collect());
            }
            if let (Some(a), Some(b)) = (as_slice::<T, f64>(a), as_slice::<T, f64>(b)) {
                let mut out = vec![0.0f64; a.len()];
                crate::simd::f64k::elementwise(a, b, op, &mut out);
                return Some(out.into_iter().map(from_f64).collect());
            }
        }
        None
    }

    pub fn broadcast<T: Coefficient>(
        values: &[T],
        scalar: T,
        op: BinaryOp,
        scalar_left: bool,
    ) -> Option<Vec<T>> {
        if values.len() < MIN_ELEMENTS || op == BinaryOp::Rem {
            return None;
        }
        unsafe {
            if let (Some(v), Some(s)) = (
                as_slice::<T, f32>(values),
                as_slice::<T, f32>(std::slice::from_ref(&scalar)),
            ) {
                let mut out = vec![0.0f32; v.len()];
                crate::simd::f32k::broadcast(v, s[0], op, scalar_left, &mut out);
                return Some(out.into_iter().map(from_f32).collect());
            }
            if let (Some(v), Some(s)) = (
                as_slice::<T, f64>(values),
                as_slice::<T, f64>(std::slice::from_ref(&scalar)),
            ) {
                let mut out = vec![0.0f64; v.len()];
                crate::simd::f64k::broadcast(v, s[0], op, scalar_left, &mut out);
                return Some(out.into_iter().map(from_f64).collect());
            }
        }
        None
    }

    pub fn matmul<T: Coefficient, const R: usize, const K: usize, const C: usize>(
        a: &[[T; K]; R],
        b: &[[T; C]; K],
    ) -> Option<Matrix<T, R, C>> {
        if R.saturating_mul(K).saturating_mul(C) < MIN_MATMUL_OPS {
            return None;
        }
        unsafe {
            let a_flat = std::slice::from_raw_parts(a.as_ptr().cast::<T>(), R * K);
            let b_flat = std::slice::from_raw_parts(b.as_ptr().cast::<T>(), K * C);
            if let (Some(a), Some(b)) = (as_slice::<T, f32>(a_flat), as_slice::<T, f32>(b_flat)) {
                let mut out = vec![0.0f32; R * C];
                crate::simd::f32k::matmul(a, b, R, K, C, &mut out);
                return Some(Matrix::from_rows(std::array::from_fn(|i| {
                    std::array::from_fn(|j| from_f32(out[i * C + j]))
                })));
            }
            if let (Some(a), Some(b)) = (as_slice::<T, f64>(a_flat), as_slice::<T, f64>(b_flat)) {
                let mut out = vec![0.0f64; R * C];
                crate::simd::f64k::matmul(a, b, R, K, C, &mut out);
                return Some(Matrix::from_rows(std::array::from_fn(|i| {
                    std::array::from_fn(|j| from_f64(out[i * C + j]))
                })));
            }
        }
        None
    }

    pub fn matmul_add<T: Coefficient, const R: usize, const K: usize, const C: usize>(
        a: &[[T; K]; R],
        b: &[[T; C]; K],
        addend: &mut [[T; C]; R],
    ) -> bool {
        if R.saturating_mul(K).saturating_mul(C) < MIN_MATMUL_OPS {
            return false;
        }
        unsafe {
            let a_flat = std::slice::from_raw_parts(a.as_ptr().cast::<T>(), R * K);
            let b_flat = std::slice::from_raw_parts(b.as_ptr().cast::<T>(), K * C);
            let addend_flat = std::slice::from_raw_parts_mut(
                addend.as_mut_ptr().cast::<T>(),
                R.saturating_mul(C),
            );
            if let (Some(a), Some(b), Some(addend)) = (
                as_slice::<T, f32>(a_flat),
                as_slice::<T, f32>(b_flat),
                as_slice_mut::<T, f32>(addend_flat),
            ) {
                crate::simd::f32k::matmul_accumulate(a, b, R, K, C, addend);
                return true;
            }
            if let (Some(a), Some(b), Some(addend)) = (
                as_slice::<T, f64>(a_flat),
                as_slice::<T, f64>(b_flat),
                as_slice_mut::<T, f64>(addend_flat),
            ) {
                crate::simd::f64k::matmul_accumulate(a, b, R, K, C, addend);
                return true;
            }
        }
        false
    }

    pub fn matvec<T: Coefficient, const R: usize, const C: usize>(
        matrix: &[[T; C]; R],
        vector: &[T; C],
    ) -> Option<Vector<T, R>> {
        if R.saturating_mul(C) < MIN_MATMUL_OPS {
            return None;
        }
        // Row-times-vector: each output is a dot product of a matrix row with
        // the vector, so the per-row reduction kernel is the right shape here.
        unsafe {
            let m = std::slice::from_raw_parts(matrix.as_ptr().cast::<T>(), R * C);
            if let (Some(m), Some(v)) = (as_slice::<T, f32>(m), as_slice::<T, f32>(vector)) {
                return Some(Vector::new(std::array::from_fn(|i| {
                    from_f32(crate::simd::f32k::dot(&m[i * C..i * C + C], v))
                })));
            }
            if let (Some(m), Some(v)) = (as_slice::<T, f64>(m), as_slice::<T, f64>(vector)) {
                return Some(Vector::new(std::array::from_fn(|i| {
                    from_f64(crate::simd::f64k::dot(&m[i * C..i * C + C], v))
                })));
            }
        }
        None
    }

    pub fn matvec_add<T: Coefficient, const R: usize, const C: usize>(
        matrix: &[[T; C]; R],
        vector: &[T; C],
        addend: &[T; R],
    ) -> Option<Vector<T, R>> {
        if R.saturating_mul(C) < MIN_MATMUL_OPS {
            return None;
        }
        unsafe {
            let m = std::slice::from_raw_parts(matrix.as_ptr().cast::<T>(), R * C);
            if let (Some(m), Some(v), Some(addend)) = (
                as_slice::<T, f32>(m),
                as_slice::<T, f32>(vector),
                as_slice::<T, f32>(addend),
            ) {
                return Some(Vector::new(std::array::from_fn(|i| {
                    from_f32(crate::simd::f32k::dot(&m[i * C..i * C + C], v) + addend[i])
                })));
            }
            if let (Some(m), Some(v), Some(addend)) = (
                as_slice::<T, f64>(m),
                as_slice::<T, f64>(vector),
                as_slice::<T, f64>(addend),
            ) {
                return Some(Vector::new(std::array::from_fn(|i| {
                    from_f64(crate::simd::f64k::dot(&m[i * C..i * C + C], v) + addend[i])
                })));
            }
        }
        None
    }

    pub fn vecmat<T: Coefficient, const R: usize, const C: usize>(
        vector: &[T; R],
        matrix: &[[T; C]; R],
    ) -> Option<Vector<T, C>> {
        if R.saturating_mul(C) < MIN_MATMUL_OPS {
            return None;
        }
        // (1×R)·(R×C): the broadcast-A matmul vectorizes across the C columns.
        unsafe {
            let m = std::slice::from_raw_parts(matrix.as_ptr().cast::<T>(), R * C);
            if let (Some(v), Some(m)) = (as_slice::<T, f32>(vector), as_slice::<T, f32>(m)) {
                let mut out = vec![0.0f32; C];
                crate::simd::f32k::matmul(v, m, 1, R, C, &mut out);
                return Some(Vector::new(std::array::from_fn(|j| from_f32(out[j]))));
            }
            if let (Some(v), Some(m)) = (as_slice::<T, f64>(vector), as_slice::<T, f64>(m)) {
                let mut out = vec![0.0f64; C];
                crate::simd::f64k::matmul(v, m, 1, R, C, &mut out);
                return Some(Vector::new(std::array::from_fn(|j| from_f64(out[j]))));
            }
        }
        None
    }

    /// In-place radix-2 FFT for power-of-two `f32` lengths. Returns `false` (so
    /// the caller keeps the generic path) for every other element type or shape.
    /// `direction` is `-1.0` forward, `+1.0` inverse; normalization stays with
    /// the caller.
    pub fn radix2_fft<T: Coefficient, const N: usize>(
        output: &mut [Complex<T>; N],
        direction: f64,
    ) -> bool {
        if N < MIN_FFT_LENGTH || !N.is_power_of_two() || !is::<T, f32>() {
            return false;
        }
        // SAFETY: T is f32 and `Complex` is `#[repr(C)]`, so the buffer is
        // exactly `[re, im, …]` — `2*N` contiguous f32.
        let buf =
            unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr().cast::<f32>(), 2 * N) };
        crate::simd::fft_f32::radix2(buf, N, direction as f32);
        true
    }
}

// ---- vectors ----------------------------------------------------------------

/// A length-`N` vector, backed by `[T; N]` on the default [`Host`] backend.
pub struct Vector<T, const N: usize, B: Backend = Host> {
    data: B::Vector<T, N>,
}

// The storage type varies with the backend, so these are the derives written by
// hand: a `Host` tensor is `Copy` because an array of `Copy` elements is, and a
// `Metal` tensor is not because a shared allocation is not.
impl<T, const N: usize, B: Backend> Copy for Vector<T, N, B> where B::Vector<T, N>: Copy {}

impl<T, const N: usize, B: Backend> Clone for Vector<T, N, B>
where
    B::Vector<T, N>: Clone,
{
    fn clone(&self) -> Self {
        Vector {
            data: self.data.clone(),
        }
    }
}

impl<T, const N: usize, B: Backend> PartialEq for Vector<T, N, B>
where
    B::Vector<T, N>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<T, const N: usize, B: Backend> Eq for Vector<T, N, B> where B::Vector<T, N>: Eq {}

impl<T, const N: usize, B: Backend> fmt::Debug for Vector<T, N, B>
where
    B::Vector<T, N>: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Vector").field("data", &self.data).finish()
    }
}

/// `f32` vectors on any backend. These are the operations that do not depend on
/// where the elements live — everything else is implemented per backend.
impl<const N: usize, B: Backend> Vector<f32, N, B> {
    /// Move this vector's elements onto backend `B2`.
    ///
    /// This is the operation that copies: onto [`Metal`] it is an upload into
    /// shared memory, off it a download. Everything in between stays put.
    ///
    /// ```
    /// # #[cfg(all(feature = "metal", target_os = "macos"))] {
    /// use tensorcrate::tensors::{Host, Metal, Vector};
    ///
    /// let v = Vector::new([1.0f32, 2.0, 3.0]);
    /// let resident = v.to_backend::<Metal>();
    /// assert_eq!(resident.to_backend::<Host>(), v);
    /// # }
    /// ```
    pub fn to_backend<B2: Backend>(&self) -> Vector<f32, N, B2> {
        Vector {
            data: B2::store_vector::<N>(B::vector_slice::<N>(&self.data)),
        }
    }

    /// Every element set to `value`, allocated directly on backend `B`.
    ///
    /// The values are staged on the heap rather than written as an `[f32; N]`
    /// literal, so on a GPU backend this is how to build a tensor too large to
    /// sit on the stack: `Vector::<f32, 1_000_000, Metal>::filled(0.0)`.
    pub fn filled(value: f32) -> Self {
        Vector {
            data: B::store_vector::<N>(&vec![value; N]),
        }
    }

    /// Borrow the elements as a slice, without copying.
    pub fn as_slice(&self) -> &[f32] {
        B::vector_slice::<N>(&self.data)
    }

    /// Copy the elements into an array.
    pub fn to_array(&self) -> [f32; N] {
        let values = self.as_slice();
        std::array::from_fn(|index| values[index])
    }

    /// Consume this vector and view its elements as a `1 × N` row matrix.
    ///
    /// On the Metal backend this only changes the static shape; the existing
    /// allocation is reused without a copy or kernel dispatch.
    pub fn into_row_matrix(self) -> Matrix<f32, 1, N, B> {
        Matrix {
            data: B::vector_into_row(self.data),
        }
    }

    /// Consume this vector and view its elements as an `N × 1` column matrix.
    ///
    /// As with [`into_row_matrix`](Self::into_row_matrix), Metal reuses the
    /// existing allocation.
    pub fn into_column_matrix(self) -> Matrix<f32, N, 1, B> {
        Matrix {
            data: B::vector_into_column(self.data),
        }
    }
}

impl<T, const N: usize> Vector<T, N> {
    /// A vector from its elements.
    pub const fn new(data: [T; N]) -> Self {
        Vector { data }
    }

    pub fn data(&self) -> &[T; N] {
        &self.data
    }

    pub fn get(&self, i: usize) -> Option<&T> {
        self.data.get(i)
    }

    pub const fn len(&self) -> usize {
        N
    }

    pub const fn is_empty(&self) -> bool {
        N == 0
    }

    /// Apply `f` to every element, producing a vector of the new element type —
    /// e.g. lifting a `Vector<f64, N>` into a `Vector<Complex<f64>, N>`.
    pub fn map<U>(&self, f: impl Fn(&T) -> U) -> Vector<U, N> {
        Vector {
            data: std::array::from_fn(|i| f(&self.data[i])),
        }
    }
}

impl<T: Coefficient, const N: usize> Vector<T, N> {
    pub fn zeros() -> Self {
        Vector {
            data: std::array::from_fn(|_| T::zero()),
        }
    }

    /// Multiply every element by `scalar`.
    pub fn scale(&self, scalar: T) -> Self {
        self.broadcast_right(scalar, BinaryOp::Mul)
    }

    /// Implementation hook used by `math!` for tensor/scalar broadcasting.
    #[doc(hidden)]
    pub fn broadcast_right(&self, scalar: T, op: BinaryOp) -> Self {
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if let Some(output) = simd_dispatch::broadcast(&self.data, scalar, op, false) {
            return Vector::new(std::array::from_fn(|index| output[index]));
        }
        self.map(|&value| match op {
            BinaryOp::Add => value + scalar,
            BinaryOp::Sub => value - scalar,
            BinaryOp::Mul => value * scalar,
            BinaryOp::Div => value / scalar,
            BinaryOp::Rem => value % scalar,
        })
    }

    /// Implementation hook used by `math!` for scalar/tensor broadcasting.
    #[doc(hidden)]
    pub fn broadcast_left(&self, scalar: T, op: BinaryOp) -> Self {
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if let Some(output) = simd_dispatch::broadcast(&self.data, scalar, op, true) {
            return Vector::new(std::array::from_fn(|index| output[index]));
        }
        self.map(|&value| match op {
            BinaryOp::Add => scalar + value,
            BinaryOp::Sub => scalar - value,
            BinaryOp::Mul => scalar * value,
            BinaryOp::Div => scalar / value,
            BinaryOp::Rem => scalar % value,
        })
    }

    /// Dot product with a vector of the same length — the length match is
    /// enforced by the type.
    pub fn dot(&self, other: &Vector<T, N>) -> T {
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if let Some(output) = simd_dispatch::dot(&self.data, &other.data) {
            return output;
        }
        let mut sum = T::zero();
        for i in 0..N {
            sum = sum + self.data[i] * other.data[i];
        }
        sum
    }

    /// Row vector times matrix: `(1×N)·(N×C) = (1×C)`.
    pub fn vecmat<const C: usize>(&self, m: &Matrix<T, N, C>) -> Vector<T, C> {
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if let Some(output) = simd_dispatch::vecmat(&self.data, &m.data) {
            return output;
        }
        Vector {
            data: std::array::from_fn(|j| {
                let mut sum = T::zero();
                for p in 0..N {
                    sum = sum + self.data[p] * m.data[p][j];
                }
                sum
            }),
        }
    }

    /// Discrete Fourier transform.
    ///
    /// Power-of-two lengths use iterative radix-2 Cooley–Tukey (`O(N log N)`).
    /// Other composite lengths use a recursive mixed-radix Cooley–Tukey
    /// decomposition for radices up to 15. Sub-transforms without a factor in
    /// that range use the definition directly, so every const length—including
    /// zero and one—is supported.
    ///
    /// This is the conventional unnormalized forward transform:
    /// `X[k] = Σ x[n] exp(-2πikn/N)`.
    pub fn fft(&self) -> Vector<Complex<T>, N>
    where
        T: Float,
    {
        let mut output =
            std::array::from_fn(|i| Complex::new(self.data[i], <T as num_traits::Zero>::zero()));

        if N <= 1 {
            return Vector::new(output);
        }

        if !N.is_power_of_two() {
            let transformed = mixed_radix_fft(&output, false);
            output.copy_from_slice(&transformed);
            return Vector::new(output);
        }

        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if simd_dispatch::radix2_fft(&mut output, -1.0) {
            return Vector::new(output);
        }

        radix2_fft(&mut output, false);
        Vector::new(output)
    }
}

impl<T: Float + Coefficient, const N: usize> Vector<Complex<T>, N> {
    /// Inverse discrete Fourier transform.
    ///
    /// This uses the same radix-2 and mixed-radix Cooley–Tukey paths as
    /// [`Vector::fft`], with a direct DFT for leaves whose smallest factor
    /// exceeds 15. It is the conventional normalized inverse transform:
    /// `x[n] = (1/N) Σ X[k] exp(2πikn/N)`.
    pub fn ifft(&self) -> Vector<Complex<T>, N> {
        if N <= 1 {
            return *self;
        }

        let mut output = self.data;
        if N.is_power_of_two() {
            #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
            let vectorized = simd_dispatch::radix2_fft(&mut output, 1.0);
            #[cfg(not(all(
                feature = "simd",
                any(target_arch = "aarch64", target_arch = "x86_64")
            )))]
            let vectorized = false;
            if !vectorized {
                radix2_fft(&mut output, true);
            }
        } else {
            let transformed = mixed_radix_fft(&output, true);
            output.copy_from_slice(&transformed);
        }

        let normalization = cast::<T>(N);
        for value in &mut output {
            value.real = value.real / normalization;
            value.im = value.im / normalization;
        }
        Vector::new(output)
    }
}

fn cast<T: NumCast>(value: impl NumCast) -> T {
    NumCast::from(value).expect("usize and f64 Fourier constants fit supported float types")
}

fn radix2_fft<T: Float + Coefficient>(output: &mut [Complex<T>], inverse: bool) {
    let n = output.len();

    // Bit-reversal permutation puts inputs in the order consumed by the
    // iterative butterfly stages.
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            output.swap(i, j);
        }
    }

    let direction = if inverse {
        <T as num_traits::One>::one()
    } else {
        -<T as num_traits::One>::one()
    };
    let tau = cast::<T>(std::f64::consts::TAU);
    let mut len = 2;
    while len <= n {
        let angle = direction * tau / cast::<T>(len);
        let step = Complex::new(angle.cos(), angle.sin());
        for start in (0..n).step_by(len) {
            let mut twiddle = Complex::new(
                <T as num_traits::One>::one(),
                <T as num_traits::Zero>::zero(),
            );
            for offset in 0..len / 2 {
                let even = output[start + offset];
                let odd = output[start + offset + len / 2] * twiddle;
                output[start + offset] = even + odd;
                output[start + offset + len / 2] = even - odd;
                twiddle = twiddle * step;
            }
        }
        len *= 2;
    }
}

/// Cooley–Tukey decomposition for arbitrary composite lengths. Splitting by
/// the smallest supported factor uses radices up to 15. If no supported factor
/// divides the length, the transform uses the quadratic DFT rather than a
/// high-radix stage.
fn mixed_radix_fft<T: Float + Coefficient>(input: &[Complex<T>], inverse: bool) -> Vec<Complex<T>> {
    let n = input.len();
    if n <= 1 {
        return input.to_vec();
    }

    let Some(radix) = smallest_mixed_radix(n) else {
        return direct_dft(input, inverse);
    };

    let quotient = n / radix;
    let mut sub_transforms = Vec::with_capacity(radix);
    for residue in 0..radix {
        let subsequence = (0..quotient)
            .map(|index| input[residue + radix * index])
            .collect::<Vec<_>>();
        sub_transforms.push(mixed_radix_fft(&subsequence, inverse));
    }

    let zero = Complex::new(
        <T as num_traits::Zero>::zero(),
        <T as num_traits::Zero>::zero(),
    );
    let tau = cast::<T>(std::f64::consts::TAU);
    let direction = if inverse {
        <T as num_traits::One>::one()
    } else {
        -<T as num_traits::One>::one()
    };
    let mut output = vec![zero; n];
    for (high_frequency, frequency_band) in output.chunks_exact_mut(quotient).enumerate() {
        for (low_frequency, target) in frequency_band.iter_mut().enumerate() {
            let frequency = low_frequency + quotient * high_frequency;
            let mut sum = zero;
            for (residue, sub_transform) in sub_transforms.iter().enumerate() {
                let angle =
                    direction * tau * cast::<T>(residue) * cast::<T>(frequency) / cast::<T>(n);
                let twiddle = Complex::new(angle.cos(), angle.sin());
                sum = sum + sub_transform[low_frequency] * twiddle;
            }
            *target = sum;
        }
    }
    output
}

fn direct_dft<T: Float + Coefficient>(input: &[Complex<T>], inverse: bool) -> Vec<Complex<T>> {
    let n = input.len();
    let tau = cast::<T>(std::f64::consts::TAU);
    let direction = if inverse {
        <T as num_traits::One>::one()
    } else {
        -<T as num_traits::One>::one()
    };
    (0..n)
        .map(|frequency| {
            let mut sum = Complex::new(
                <T as num_traits::Zero>::zero(),
                <T as num_traits::Zero>::zero(),
            );
            for (index, &value) in input.iter().enumerate() {
                let angle =
                    direction * tau * cast::<T>(frequency) * cast::<T>(index) / cast::<T>(n);
                let twiddle = Complex::new(angle.cos(), angle.sin());
                sum = sum + value * twiddle;
            }
            sum
        })
        .collect()
}

const MAX_MIXED_RADIX: usize = 15;

fn smallest_mixed_radix(n: usize) -> Option<usize> {
    // These are all primes up to MAX_MIXED_RADIX. If none divides n, its
    // smallest prime factor necessarily exceeds the mixed-radix cutoff.
    [2, 3, 5, 7, 11, 13]
        .into_iter()
        .take_while(|&factor| factor <= MAX_MIXED_RADIX)
        .find(|&factor| n.is_multiple_of(factor))
}

#[cfg(test)]
mod fft_tests {
    use super::smallest_mixed_radix;

    #[test]
    fn mixed_radix_selection_stops_above_fifteen() {
        assert_eq!(smallest_mixed_radix(2 * 17), Some(2));
        assert_eq!(smallest_mixed_radix(13 * 17), Some(13));
        assert_eq!(smallest_mixed_radix(17 * 19), None);
        assert_eq!(smallest_mixed_radix(17), None);
    }
}

// ---- matrices ---------------------------------------------------------------

/// An `R × C` matrix, backed by `[[T; C]; R]` in row-major order on the default
/// [`Host`] backend.
pub struct Matrix<T, const R: usize, const C: usize, B: Backend = Host> {
    data: B::Matrix<T, R, C>,
}

// As for `Vector`: hand-written derives, because the storage type — and so which
// of these a tensor gets — depends on the backend.
impl<T, const R: usize, const C: usize, B: Backend> Copy for Matrix<T, R, C, B> where
    B::Matrix<T, R, C>: Copy
{
}

impl<T, const R: usize, const C: usize, B: Backend> Clone for Matrix<T, R, C, B>
where
    B::Matrix<T, R, C>: Clone,
{
    fn clone(&self) -> Self {
        Matrix {
            data: self.data.clone(),
        }
    }
}

impl<T, const R: usize, const C: usize, B: Backend> PartialEq for Matrix<T, R, C, B>
where
    B::Matrix<T, R, C>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<T, const R: usize, const C: usize, B: Backend> Eq for Matrix<T, R, C, B> where
    B::Matrix<T, R, C>: Eq
{
}

impl<T, const R: usize, const C: usize, B: Backend> fmt::Debug for Matrix<T, R, C, B>
where
    B::Matrix<T, R, C>: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Matrix").field("data", &self.data).finish()
    }
}

/// `f32` matrices on any backend, as for [`Vector`] above.
impl<const R: usize, const C: usize, B: Backend> Matrix<f32, R, C, B> {
    /// Move this matrix's elements onto backend `B2`.
    ///
    /// The counterpart to [`Vector::to_backend`], and the only place a
    /// `Metal`-backed chain copies between CPU and GPU memory.
    pub fn to_backend<B2: Backend>(&self) -> Matrix<f32, R, C, B2> {
        Matrix {
            data: B2::store_matrix::<R, C>(B::matrix_slice::<R, C>(&self.data)),
        }
    }

    /// Every element set to `value`, allocated directly on backend `B`, staged on
    /// the heap rather than as a `[[f32; C]; R]` literal — see
    /// [`Vector::filled`].
    pub fn filled(value: f32) -> Self {
        Matrix {
            data: B::store_matrix::<R, C>(&vec![value; R * C]),
        }
    }

    /// Borrow the elements as one flat row-major slice, without copying.
    pub fn as_slice(&self) -> &[f32] {
        B::matrix_slice::<R, C>(&self.data)
    }

    /// Copy the elements into an array of rows.
    pub fn to_rows(&self) -> [[f32; C]; R] {
        let values = self.as_slice();
        std::array::from_fn(|row| std::array::from_fn(|col| values[row * C + col]))
    }
}

/// A type-erased matrix used to assemble a heterogeneous const-generic matrix
/// chain. Construct one with `MatrixOperand::from(&matrix)`.
///
/// Rust slices cannot directly contain `Matrix<T, R, C>` values with differing
/// `R` and `C` parameters. This small owned adapter erases those intermediate
/// dimensions while [`chained_matmul`] restores the final dimensions in its
/// return type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatrixOperand<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T: Copy, const R: usize, const C: usize> From<&Matrix<T, R, C>> for MatrixOperand<T> {
    fn from(matrix: &Matrix<T, R, C>) -> Self {
        Self {
            rows: R,
            cols: C,
            data: matrix.data.iter().flatten().copied().collect(),
        }
    }
}

/// Multiply a heterogeneous chain in the optimal parenthesization.
///
/// The returned `R × C` shape is const-generic and is checked against the
/// chain's endpoints. The optimal order is found with the classic matrix-chain
/// dynamic program; use [`chained_matmul_cost`] when only the optimal cost is
/// needed, since that function uses the `O(n log n)` Hu–Shing solver.
pub fn chained_matmul<T: Coefficient, const R: usize, const C: usize>(
    matrices: &[MatrixOperand<T>],
) -> Result<Matrix<T, R, C>, Error> {
    let dims = validate_chain(matrices)?;
    if R != dims[0] || C != dims[dims.len() - 1] {
        return Err(Error::shape(format!(
            "chain result is {}×{}, but the requested Matrix type is {R}×{C}",
            dims[0],
            dims[dims.len() - 1]
        )));
    }

    let n = matrices.len();
    let mut costs = vec![vec![0u128; n]; n];
    let mut splits = vec![vec![0usize; n]; n];
    for span in 2..=n {
        for i in 0..=n - span {
            let j = i + span - 1;
            costs[i][j] = u128::MAX;
            for k in i..j {
                let multiplication = (dims[i] as u128)
                    .saturating_mul(dims[k + 1] as u128)
                    .saturating_mul(dims[j + 1] as u128);
                let candidate = costs[i][k]
                    .saturating_add(costs[k + 1][j])
                    .saturating_add(multiplication);
                if candidate < costs[i][j] {
                    costs[i][j] = candidate;
                    splits[i][j] = k;
                }
            }
        }
    }

    fn evaluate<T: Coefficient>(
        matrices: &[MatrixOperand<T>],
        splits: &[Vec<usize>],
        i: usize,
        j: usize,
    ) -> MatrixOperand<T> {
        if i == j {
            return matrices[i].clone();
        }
        let k = splits[i][j];
        let left = evaluate(matrices, splits, i, k);
        let right = evaluate(matrices, splits, k + 1, j);
        multiply_operands(&left, &right)
    }

    let result = evaluate(matrices, &splits, 0, n - 1);
    Ok(Matrix::from_rows(std::array::from_fn(|i| {
        std::array::from_fn(|j| result.data[i * C + j])
    })))
}

/// Minimum scalar-multiplication cost for a matrix chain, computed by the
/// Hu–Shing `O(n log n)` optimal polygon-triangulation algorithm.
pub fn chained_matmul_cost<T>(matrices: &[MatrixOperand<T>]) -> Result<u128, Error> {
    let dims = validate_chain(matrices)?;
    Ok(hu_shing::optimal_cost(&dims.iter().map(|&d| d as i128).collect::<Vec<_>>()) as u128)
}

fn validate_chain<T>(matrices: &[MatrixOperand<T>]) -> Result<Vec<usize>, Error> {
    if matrices.len() < 2 {
        return Err(Error::InvalidArgument(
            "expected at least 2 matrices to multiply".to_string(),
        ));
    }
    let mut dims = Vec::with_capacity(matrices.len() + 1);
    dims.push(matrices[0].rows);
    for (i, matrix) in matrices.iter().enumerate() {
        if i > 0 && matrix.rows != dims[i] {
            return Err(Error::shape(format!(
                "chained_matmul dimension mismatch: matrix {} has {} columns but matrix {i} has {} rows",
                i - 1,
                dims[i],
                matrix.rows
            )));
        }
        dims.push(matrix.cols);
    }
    if dims.contains(&0) {
        return Err(Error::InvalidArgument(
            "matrix-chain dimensions must be positive".to_string(),
        ));
    }
    Ok(dims)
}

fn multiply_operands<T: Coefficient>(
    left: &MatrixOperand<T>,
    right: &MatrixOperand<T>,
) -> MatrixOperand<T> {
    debug_assert_eq!(left.cols, right.rows);
    let mut data = vec![T::zero(); left.rows * right.cols];
    for i in 0..left.rows {
        for j in 0..right.cols {
            let mut sum = T::zero();
            for k in 0..left.cols {
                sum = sum + left.data[i * left.cols + k] * right.data[k * right.cols + j];
            }
            data[i * right.cols + j] = sum;
        }
    }
    MatrixOperand {
        rows: left.rows,
        cols: right.cols,
        data,
    }
}

/// Hu–Shing's optimal weighted-polygon triangulation algorithm, restored from
/// the original tensor implementation. A matrix chain's boundary dimensions
/// are the polygon weights.
mod hu_shing {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    #[derive(Clone, Copy)]
    struct HArc {
        u: usize,
        v: usize,
        low: usize,
        base: i128,
        mul: i128,
        num: i128,
        den: i128,
    }

    impl HArc {
        fn contains(&self, other: &HArc) -> bool {
            self.u <= other.u && other.v <= self.v
        }

        fn support(&self) -> i128 {
            self.num / self.den
        }
    }

    impl PartialEq for HArc {
        fn eq(&self, other: &Self) -> bool {
            self.support() == other.support()
        }
    }

    impl Eq for HArc {}

    impl PartialOrd for HArc {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for HArc {
        fn cmp(&self, other: &Self) -> Ordering {
            self.support().cmp(&other.support())
        }
    }

    struct Solver {
        n: usize,
        w: Vec<i128>,
        cp: Vec<i128>,
        h: Vec<HArc>,
        n_arcs: usize,
        sub: Vec<usize>,
        child: Vec<Vec<usize>>,
        n_pqs: usize,
        qid: Vec<usize>,
        pq: Vec<BinaryHeap<HArc>>,
        con: Vec<Vec<HArc>>,
    }

    impl Solver {
        fn new_arc(&mut self, u: usize, v: usize) {
            debug_assert!(u <= v);
            self.n_arcs += 1;
            let low = if self.w[u] < self.w[v] { u } else { v };
            let mul = self.w[u] * self.w[v];
            let base = self.cp[v] - self.cp[u] - mul;
            self.h[self.n_arcs] = HArc {
                u,
                v,
                low,
                base,
                mul,
                num: 0,
                den: 1,
            };
        }

        fn build_tree(&mut self, arcs: &[(usize, usize)]) {
            let mut stack = Vec::new();
            self.new_arc(1, self.n + 1);
            for &(a, b) in arcs {
                self.new_arc(a, b);
                let current = self.n_arcs;
                while let Some(&top) = stack.last() {
                    if self.h[current].contains(&self.h[top]) {
                        self.child[current].push(top);
                        stack.pop();
                    } else {
                        break;
                    }
                }
                stack.push(current);
            }
            while let Some(top) = stack.pop() {
                self.child[1].push(top);
            }
        }

        fn one_sweep(&mut self) {
            let mut stack = Vec::new();
            let mut arcs = Vec::new();
            for i in 1..=self.n {
                while stack.len() >= 2 && self.w[*stack.last().unwrap()] > self.w[i] {
                    arcs.push((stack[stack.len() - 2], i));
                    stack.pop();
                }
                stack.push(i);
            }
            while stack.len() >= 4 {
                arcs.push((1, stack[stack.len() - 2]));
                stack.pop();
            }
            let arcs = arcs
                .into_iter()
                .filter(|&(a, b)| a != 1 && b != 1)
                .collect::<Vec<_>>();
            self.build_tree(&arcs);
        }

        fn prepare(&mut self) {
            let mut first = 1;
            for i in 2..=self.n {
                if self.w[i] < self.w[first] {
                    first = i;
                }
            }
            self.w[1..=self.n].rotate_left(first - 1);
            self.w[self.n + 1] = self.w[1];
            for i in 1..=self.n + 1 {
                self.cp[i] = self.w[i] * self.w[i - 1] + self.cp[i - 1];
            }
        }

        fn minimum_neighbor_product(&self, node: usize) -> i128 {
            if node == 1 {
                return self.w[1] * self.w[2] + self.w[1] * self.w[self.n];
            }
            let current = self.h[node];
            if current.u == current.low {
                match self.con[current.u].last() {
                    Some(back) if current.contains(back) => back.mul,
                    _ => self.w[current.u] * self.w[current.u + 1],
                }
            } else {
                match self.con[current.v].last() {
                    Some(back) if current.contains(back) => back.mul,
                    _ => self.w[current.v] * self.w[current.v - 1],
                }
            }
        }

        fn add_arc(&mut self, node: usize, arc: HArc) {
            let queue = self.qid[node];
            self.pq[queue].push(arc);
            self.con[arc.u].push(arc);
            self.con[arc.v].push(arc);
        }

        fn remove_arc(&mut self, node: usize) {
            let queue = self.qid[node];
            let arc = *self.pq[queue].peek().expect("remove_arc on empty queue");
            self.con[arc.u].pop();
            self.con[arc.v].pop();
            self.pq[queue].pop();
        }

        fn merge_queues(&mut self, node: usize) {
            let mut largest = usize::MAX;
            for &child in &self.child[node] {
                if largest == usize::MAX || self.sub[largest] < self.sub[child] {
                    largest = child;
                }
            }
            self.qid[node] = self.qid[largest];
            let target = self.qid[node];
            for child in self.child[node].clone() {
                if child != largest {
                    let source = std::mem::take(&mut self.pq[self.qid[child]]);
                    self.pq[target].extend(source);
                }
            }
        }

        fn solve_subtree(&mut self, node: usize) {
            self.sub[node] = 1;
            let mul = self.h[node].mul;
            let low = self.h[node].low;

            if self.child[node].is_empty() {
                self.n_pqs += 1;
                self.qid[node] = self.n_pqs;
                let den = self.h[node].base;
                let num = self.w[low] * (den + mul - self.minimum_neighbor_product(node));
                self.h[node].num = num;
                self.h[node].den = den;
                self.add_arc(node, self.h[node]);
                return;
            }

            let mut den = self.h[node].base;
            for child in self.child[node].clone() {
                self.solve_subtree(child);
                self.sub[node] += self.sub[child];
                den -= self.h[child].base;
            }
            let mut num = self.w[low] * (den + mul - self.minimum_neighbor_product(node));
            self.merge_queues(node);
            let queue = self.qid[node];

            while matches!(self.pq[queue].peek(), Some(top) if top.support() >= self.w[low]) {
                den += self.pq[queue].peek().unwrap().den;
                self.remove_arc(node);
                num = self.w[low] * (den + mul - self.minimum_neighbor_product(node));
            }
            while matches!(self.pq[queue].peek(), Some(top) if num / den <= top.support()) {
                let top = *self.pq[queue].peek().unwrap();
                den += top.den;
                self.remove_arc(node);
                num += top.num;
            }

            self.h[node].num = num;
            self.h[node].den = den;
            self.add_arc(node, self.h[node]);
        }

        fn answer(&mut self) -> i128 {
            self.solve_subtree(1);
            let queue = std::mem::take(&mut self.pq[self.qid[1]]);
            queue.into_iter().map(|arc| arc.num).sum()
        }
    }

    pub fn optimal_cost(dims: &[i128]) -> i128 {
        match dims.len() {
            0 | 1 => return 0,
            2 => return dims[0] * dims[1],
            _ => {}
        }
        let n = dims.len();
        let len = n + 3;
        let empty_arc = HArc {
            u: 0,
            v: 0,
            low: 0,
            base: 0,
            mul: 0,
            num: 0,
            den: 1,
        };
        let mut solver = Solver {
            n,
            w: vec![0; len],
            cp: vec![0; len],
            h: vec![empty_arc; len],
            n_arcs: 0,
            sub: vec![0; len],
            child: vec![Vec::new(); len],
            n_pqs: 0,
            qid: vec![0; len],
            pq: (0..len).map(|_| BinaryHeap::new()).collect(),
            con: vec![Vec::new(); len],
        };
        solver.w[1..=n].copy_from_slice(dims);
        solver.prepare();
        solver.one_sweep();
        solver.answer()
    }
}

#[cfg(test)]
mod hu_shing_tests {
    use super::hu_shing;

    fn dynamic_programming_cost(dims: &[i128]) -> i128 {
        let n = dims.len() - 1;
        let mut costs = vec![vec![0i128; n]; n];
        for span in 2..=n {
            for i in 0..=n - span {
                let j = i + span - 1;
                costs[i][j] = i128::MAX;
                for k in i..j {
                    costs[i][j] = costs[i][j]
                        .min(costs[i][k] + costs[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]);
                }
            }
        }
        costs[0][n - 1]
    }

    #[test]
    fn hu_shing_matches_the_cubic_oracle_exhaustively() {
        let choices = [1i128, 2, 3, 4];
        for matrices in 2..=6usize {
            let dimension_count = matrices + 1;
            let cases = choices.len().pow(dimension_count as u32);
            for mut code in 0..cases {
                let dims = (0..dimension_count)
                    .map(|_| {
                        let dimension = choices[code % choices.len()];
                        code /= choices.len();
                        dimension
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    hu_shing::optimal_cost(&dims),
                    dynamic_programming_cost(&dims),
                    "dims={dims:?}"
                );
            }
        }
    }

    #[test]
    fn hu_shing_matches_the_cubic_oracle_for_random_long_chains() {
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut random = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        for _ in 0..1_000 {
            let matrices = 2 + (random() % 30) as usize;
            let dims = (0..=matrices)
                .map(|_| 1 + (random() % 50) as i128)
                .collect::<Vec<_>>();
            assert_eq!(
                hu_shing::optimal_cost(&dims),
                dynamic_programming_cost(&dims),
                "dims={dims:?}"
            );
        }
    }
}

impl<T, const R: usize, const C: usize> Matrix<T, R, C> {
    /// A matrix from its rows.
    pub const fn from_rows(data: [[T; C]; R]) -> Self {
        Matrix { data }
    }

    pub fn data(&self) -> &[[T; C]; R] {
        &self.data
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        self.data.get(row)?.get(col)
    }

    pub const fn shape(&self) -> (usize, usize) {
        (R, C)
    }

    /// Apply `f` to every element, producing a matrix of the new element type.
    pub fn map<U>(&self, f: impl Fn(&T) -> U) -> Matrix<U, R, C> {
        Matrix {
            data: std::array::from_fn(|i| std::array::from_fn(|j| f(&self.data[i][j]))),
        }
    }
}

impl<T: Coefficient, const R: usize, const C: usize> Matrix<T, R, C> {
    /// Multiply a heterogeneous matrix chain in its optimal order.
    ///
    /// Convert each differently-shaped matrix with [`MatrixOperand::from`].
    pub fn chained_matmul(matrices: &[MatrixOperand<T>]) -> Result<Self, Error> {
        crate::tensors::chained_matmul(matrices)
    }

    /// Return the optimal multiplication cost using the Hu–Shing algorithm.
    pub fn chained_matmul_cost(matrices: &[MatrixOperand<T>]) -> Result<u128, Error> {
        crate::tensors::chained_matmul_cost(matrices)
    }

    pub fn zeros() -> Self {
        Matrix {
            data: std::array::from_fn(|_| std::array::from_fn(|_| T::zero())),
        }
    }

    /// Multiply every element by `scalar`.
    pub fn scale(&self, scalar: T) -> Self {
        self.broadcast_right(scalar, BinaryOp::Mul)
    }

    /// Implementation hook used by `math!` for tensor/scalar broadcasting.
    #[doc(hidden)]
    pub fn broadcast_right(&self, scalar: T, op: BinaryOp) -> Self {
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            // SAFETY: nested arrays are contiguous and hold exactly R*C `T`.
            let input = unsafe {
                std::slice::from_raw_parts(self.data.as_ptr().cast::<T>(), R.saturating_mul(C))
            };
            if let Some(output) = simd_dispatch::broadcast(input, scalar, op, false) {
                return Matrix::from_rows(std::array::from_fn(|row| {
                    std::array::from_fn(|col| output[row * C + col])
                }));
            }
        }
        self.map(|&value| match op {
            BinaryOp::Add => value + scalar,
            BinaryOp::Sub => value - scalar,
            BinaryOp::Mul => value * scalar,
            BinaryOp::Div => value / scalar,
            BinaryOp::Rem => value % scalar,
        })
    }

    /// Implementation hook used by `math!` for scalar/tensor broadcasting.
    #[doc(hidden)]
    pub fn broadcast_left(&self, scalar: T, op: BinaryOp) -> Self {
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            // SAFETY: nested arrays are contiguous and hold exactly R*C `T`.
            let input = unsafe {
                std::slice::from_raw_parts(self.data.as_ptr().cast::<T>(), R.saturating_mul(C))
            };
            if let Some(output) = simd_dispatch::broadcast(input, scalar, op, true) {
                return Matrix::from_rows(std::array::from_fn(|row| {
                    std::array::from_fn(|col| output[row * C + col])
                }));
            }
        }
        self.map(|&value| match op {
            BinaryOp::Add => scalar + value,
            BinaryOp::Sub => scalar - value,
            BinaryOp::Mul => scalar * value,
            BinaryOp::Div => scalar / value,
            BinaryOp::Rem => scalar % value,
        })
    }

    /// Matrix product `(R×C)·(C×C2) = (R×C2)`. The shared inner dimension `C` is
    /// enforced by the type: a mismatch does not compile.
    pub fn matmul<const C2: usize>(&self, other: &Matrix<T, C, C2>) -> Matrix<T, R, C2> {
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if let Some(output) = simd_dispatch::matmul(&self.data, &other.data) {
            return output;
        }
        Matrix {
            data: std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    let mut sum = T::zero();
                    for p in 0..C {
                        sum = sum + self.data[i][p] * other.data[p][j];
                    }
                    sum
                })
            }),
        }
    }

    /// Fused matrix multiply-add: `self·other + addend`.
    ///
    /// `addend` is consumed and used as the accumulator, avoiding a separate
    /// product allocation and elementwise addition.
    pub fn matmul_add<const C2: usize>(
        &self,
        other: &Matrix<T, C, C2>,
        addend: Matrix<T, R, C2>,
    ) -> Matrix<T, R, C2> {
        let mut output = addend;
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if simd_dispatch::matmul_add(&self.data, &other.data, &mut output.data) {
            return output;
        }
        for i in 0..R {
            for p in 0..C {
                for j in 0..C2 {
                    output.data[i][j] = output.data[i][j] + self.data[i][p] * other.data[p][j];
                }
            }
        }
        output
    }

    /// Matrix times column vector: `(R×C)·(C×1) = (R×1)`.
    pub fn matvec(&self, v: &Vector<T, C>) -> Vector<T, R> {
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if let Some(output) = simd_dispatch::matvec(&self.data, &v.data) {
            return output;
        }
        Vector {
            data: std::array::from_fn(|i| {
                let mut sum = T::zero();
                for p in 0..C {
                    sum = sum + self.data[i][p] * v.data[p];
                }
                sum
            }),
        }
    }

    /// Fused matrix-vector multiply-add: `self·v + addend`.
    ///
    /// Each row uses the SIMD dot-product kernel when available, and the owned
    /// addend is updated in place.
    pub fn matvec_add(&self, v: &Vector<T, C>, addend: Vector<T, R>) -> Vector<T, R> {
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if let Some(output) = simd_dispatch::matvec_add(&self.data, &v.data, &addend.data) {
            return output;
        }
        let mut output = addend;
        for i in 0..R {
            let mut sum = T::zero();
            for p in 0..C {
                sum = sum + self.data[i][p] * v.data[p];
            }
            output.data[i] = output.data[i] + sum;
        }
        output
    }

    /// Transpose: an `R×C` matrix becomes `C×R`.
    pub fn transpose(&self) -> Matrix<T, C, R> {
        Matrix {
            data: std::array::from_fn(|i| std::array::from_fn(|j| self.data[j][i])),
        }
    }
}

impl<T: Coefficient, const N: usize> Matrix<T, N, N> {
    /// The `N × N` identity matrix.
    pub fn identity() -> Self {
        Matrix {
            data: std::array::from_fn(|i| {
                std::array::from_fn(|j| if i == j { T::one() } else { T::zero() })
            }),
        }
    }

    /// Determinant, by fraction-free (Bareiss) elimination — every division is
    /// exact, so an integer matrix keeps an exact integer determinant.
    pub fn determinant(&self) -> T {
        if N == 0 {
            return T::one();
        }
        let mut m = self.data;
        let mut prev = T::one();
        let mut negate = false;

        for k in 0..N - 1 {
            if m[k][k].is_zero() {
                match (k + 1..N).find(|&p| !m[p][k].is_zero()) {
                    Some(p) => {
                        m.swap(k, p);
                        negate = !negate;
                    }
                    None => return T::zero(),
                }
            }
            for i in k + 1..N {
                for j in k + 1..N {
                    let value = m[i][j] * m[k][k] - m[i][k] * m[k][j];
                    m[i][j] = value / prev;
                }
            }
            prev = m[k][k];
        }

        let det = m[N - 1][N - 1];
        if negate { T::zero() - det } else { det }
    }

    /// Inverse, by Gauss–Jordan elimination with partial pivoting on
    /// [`Coefficient::magnitude`] (so complex and dual elements, which have no
    /// ordering, still pivot sensibly). Returns [`Error::Singular`] when the
    /// matrix has no inverse. Coefficient domains with truncating division, such
    /// as primitive integers and integer-based complex or dual numbers, return
    /// [`Error::InvalidArgument`] because Gauss–Jordan requires fractions.
    pub fn inverse(&self) -> Result<Self, Error> {
        if !T::supports_fractional_division() {
            return Err(Error::InvalidArgument(
                "matrix inversion requires coefficients with fractional division".to_string(),
            ));
        }
        let mut a = self.data;
        let mut inv = Self::identity().data;

        for col in 0..N {
            let pivot = (col..N)
                .max_by(|&x, &y| {
                    a[x][col]
                        .magnitude()
                        .partial_cmp(&a[y][col].magnitude())
                        .unwrap_or(Ordering::Equal)
                })
                .expect("col < N, so the range is non-empty");
            if a[pivot][col].is_zero() {
                return Err(Error::Singular);
            }
            if pivot != col {
                a.swap(col, pivot);
                inv.swap(col, pivot);
            }

            let scale = a[col][col];
            for j in 0..N {
                a[col][j] = a[col][j] / scale;
                inv[col][j] = inv[col][j] / scale;
            }
            for r in 0..N {
                if r == col {
                    continue;
                }
                let factor = a[r][col];
                if factor.is_zero() {
                    continue;
                }
                for j in 0..N {
                    a[r][j] = a[r][j] - factor * a[col][j];
                    inv[r][j] = inv[r][j] - factor * inv[col][j];
                }
            }
        }
        Ok(Matrix { data: inv })
    }
}

macro_rules! elementwise {
    ($Type:ident < $($dim:ident),+ >, $Trait:ident, $method:ident, $op:tt) => {
        impl<T: Coefficient, $(const $dim: usize),+> $Trait for $Type<T, $($dim),+> {
            type Output = $Type<T, $($dim),+>;
            fn $method(self, rhs: Self) -> Self::Output {
                self.zip_with(&rhs, |a, b| a $op b)
            }
        }
    };
}

impl<T: Coefficient, const N: usize> Vector<T, N> {
    fn zip_with(&self, rhs: &Self, f: impl Fn(T, T) -> T) -> Self {
        Vector {
            data: std::array::from_fn(|i| f(self.data[i], rhs.data[i])),
        }
    }
}

impl<T: Coefficient, const R: usize, const C: usize> Matrix<T, R, C> {
    fn zip_with(&self, rhs: &Self, f: impl Fn(T, T) -> T) -> Self {
        Matrix {
            data: std::array::from_fn(|i| {
                std::array::from_fn(|j| f(self.data[i][j], rhs.data[i][j]))
            }),
        }
    }
}

elementwise!(Vector<N>, Add, add, +);
elementwise!(Vector<N>, Sub, sub, -);
elementwise!(Vector<N>, Div, div, /);
elementwise!(Vector<N>, Rem, rem, %);
elementwise!(Matrix<R, C>, Add, add, +);
elementwise!(Matrix<R, C>, Sub, sub, -);
elementwise!(Matrix<R, C>, Div, div, /);
elementwise!(Matrix<R, C>, Rem, rem, %);

impl<T: Coefficient, const N: usize> Mul for Vector<T, N> {
    type Output = Vector<T, N>;

    fn mul(self, rhs: Self) -> Self::Output {
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if let Some(output) = simd_dispatch::elementwise(&self.data, &rhs.data, BinaryOp::Mul) {
            return Vector::new(std::array::from_fn(|index| output[index]));
        }
        self.zip_with(&rhs, |a, b| a * b)
    }
}

impl<T: Coefficient, const R: usize, const C: usize> Mul for Matrix<T, R, C> {
    type Output = Matrix<T, R, C>;

    fn mul(self, rhs: Self) -> Self::Output {
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            // SAFETY: nested arrays are contiguous and hold exactly R*C `T`.
            let left = unsafe {
                std::slice::from_raw_parts(self.data.as_ptr().cast::<T>(), R.saturating_mul(C))
            };
            let right = unsafe {
                std::slice::from_raw_parts(rhs.data.as_ptr().cast::<T>(), R.saturating_mul(C))
            };
            if let Some(output) = simd_dispatch::elementwise(left, right, BinaryOp::Mul) {
                return Matrix::from_rows(std::array::from_fn(|row| {
                    std::array::from_fn(|col| output[row * C + col])
                }));
            }
        }
        self.zip_with(&rhs, |a, b| a * b)
    }
}

impl<T: Coefficient + Neg<Output = T>, const N: usize> Neg for Vector<T, N> {
    type Output = Vector<T, N>;
    fn neg(self) -> Self {
        self.map(|&x| -x)
    }
}

impl<T: Coefficient + Neg<Output = T>, const R: usize, const C: usize> Neg for Matrix<T, R, C> {
    type Output = Matrix<T, R, C>;
    fn neg(self) -> Self {
        self.map(|&x| -x)
    }
}

// ---- display ----------------------------------------------------------------

impl<T: Display, const N: usize> Display for Vector<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for x in &self.data {
            write!(f, " {x}")?;
        }
        write!(f, " ]")
    }
}

impl<T: Display, const R: usize, const C: usize> Display for Matrix<T, R, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (r, row) in self.data.iter().enumerate() {
            if r > 0 {
                writeln!(f)?;
            }
            write!(f, "[")?;
            for x in row {
                write!(f, " {x}")?;
            }
            write!(f, " ]")?;
        }
        Ok(())
    }
}
