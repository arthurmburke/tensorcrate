//! Dynamically-shaped vectors and matrices.
//!
//! [`Vector<T>`] and [`Matrix<T>`] carry their dimensions as ordinary fields, so
//! a shape can be computed at runtime: reading a length from a file, sizing a
//! layer from a batch, or building a matrix whose extent nobody knows until the
//! program runs. Storage is a flat row-major [`Vec`], so the tensors live on the
//! heap, are [`Clone`] rather than [`Copy`], and hand out contiguous slices that
//! the SIMD and GPU kernels read directly.
//!
//! Shapes are checked where the operation happens. A matrix product
//! `(R×K)·(K′×C)` panics unless `K == K′`, and adding a `2×3` to a `3×2` panics
//! too — the message names both shapes. The operators `+ - * /` cannot return a
//! `Result` (their signatures are fixed by `std`), so every shape-dependent
//! operation is consistent with them and panics. The operations that can fail
//! for reasons other than shape keep returning [`Result`]: [`Matrix::inverse`],
//! because singularity is a property of the values, and [`chained_matmul`],
//! which validates a whole chain before multiplying anything.
//!
//! `+ - * /` are elementwise; the linear-algebra products are the named methods.
//! Each is implemented for owned and borrowed operands, so `&a + &b` leaves both
//! usable.
//!
//! Both types take a second parameter, the storage [`Backend`], which defaults
//! to [`Host`] — the `Vec` just described. On macOS with the `metal` feature,
//! `f32` tensors can instead be placed on the [`Metal`] backend, whose elements
//! live in GPU-shared memory so a chain of operations runs without copying
//! between CPU and GPU pools. [`Vector::to_backend`] and [`Matrix::to_backend`]
//! move between the two; see the [`backend`] module for the details.

use std::cmp::Ordering;
use std::fmt::{self, Display};
use std::ops::{Add, Div, Index, Mul, Neg, Rem, Sub};

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
pub use kernels::{Analytic, BinaryOp, Compare, Kernels, Ordered, Reduce, SortOrder};
pub use tape::{MatrixVar, ScalarVar, Tape, Var, VectorVar};

// ---- shape checking ---------------------------------------------------------

/// Panics unless two tensors have the same length.
#[track_caller]
fn assert_same_len(left: usize, right: usize, operation: &str) {
    assert!(
        left == right,
        "{operation}: vector lengths differ, {left} and {right}"
    );
}

/// Panics unless two matrices have the same shape.
#[track_caller]
fn assert_same_shape(left: (usize, usize), right: (usize, usize), operation: &str) {
    assert!(
        left == right,
        "{operation}: matrix shapes differ, {}×{} and {}×{}",
        left.0,
        left.1,
        right.0,
        right.1
    );
}

/// Panics unless a clamp's bounds describe a non-empty range.
///
/// Written as "not greater" rather than "less or equal" so that a NaN bound —
/// which is unordered against everything, including itself — passes rather than
/// panicking: `x.max(NaN)` is `x` on every path here, so such a bound is inert
/// rather than wrong.
#[track_caller]
fn assert_ordered_bounds<T: PartialOrd>(low: &T, high: &T) {
    assert!(
        !matches!(low.partial_cmp(high), Some(Ordering::Greater)),
        "clamp: the lower bound exceeds the upper one"
    );
}

/// Panics unless the inner dimensions of a product agree.
#[track_caller]
fn assert_inner(left: (usize, usize), right: (usize, usize), operation: &str) {
    assert!(
        left.1 == right.0,
        "{operation}: inner dimensions differ, {}×{} times {}×{}",
        left.0,
        left.1,
        right.0,
        right.1
    );
}

#[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
mod simd_dispatch {
    //! CPU SIMD tier: sits between `metal_dispatch` and the generic scalar
    //! loops. Each entry point downcasts the generic element type to a concrete
    //! float via `TypeId` (returning `None` — i.e. defer to scalar — for every
    //! other type), then calls the architecture-specific kernels in
    //! [`crate::simd`]. The size gates are deliberately small: CPU SIMD has
    //! almost no fixed cost, so it wins far below the Metal thresholds.
    //!
    //! Every entry point takes flat slices and runtime extents, which is what
    //! the kernels underneath wanted all along — the shapes used to be const
    //! parameters threaded down from the tensor types purely to be read back out
    //! as numbers here.

    use std::any::TypeId;

    use super::{BinaryOp, Compare, Complex, Reduce};
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

    pub fn dot<T: Coefficient>(a: &[T], b: &[T]) -> Option<T> {
        if a.len() < MIN_ELEMENTS {
            return None;
        }
        debug_assert_eq!(a.len(), b.len());
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

    /// Writes `a op b` into `out`, returning whether the SIMD path ran.
    ///
    /// `out` is the caller's own storage. The earlier shape returned a `Vec`,
    /// which cost two allocations — one for the kernel's scratch buffer and a
    /// second for the `collect` that converted it back to `T` — plus a copy at
    /// the call site.
    pub fn elementwise<T: Coefficient>(a: &[T], b: &[T], op: BinaryOp, out: &mut [T]) -> bool {
        if a.len() < MIN_ELEMENTS || op == BinaryOp::Rem {
            return false;
        }
        debug_assert!(b.len() == a.len() && out.len() == a.len());
        unsafe {
            if let (Some(a), Some(b), Some(out)) = (
                as_slice::<T, f32>(a),
                as_slice::<T, f32>(b),
                as_slice_mut::<T, f32>(out),
            ) {
                crate::simd::f32k::elementwise(a, b, op, out);
                return true;
            }
            if let (Some(a), Some(b), Some(out)) = (
                as_slice::<T, f64>(a),
                as_slice::<T, f64>(b),
                as_slice_mut::<T, f64>(out),
            ) {
                crate::simd::f64k::elementwise(a, b, op, out);
                return true;
            }
        }
        false
    }

    /// Writes the tensor/scalar broadcast into `out`, returning whether the SIMD
    /// path ran. Allocation-free for the same reason as [`elementwise`].
    pub fn broadcast<T: Coefficient>(
        values: &[T],
        scalar: T,
        op: BinaryOp,
        scalar_left: bool,
        out: &mut [T],
    ) -> bool {
        if values.len() < MIN_ELEMENTS || op == BinaryOp::Rem {
            return false;
        }
        debug_assert_eq!(out.len(), values.len());
        unsafe {
            if let (Some(v), Some(s), Some(out)) = (
                as_slice::<T, f32>(values),
                as_slice::<T, f32>(std::slice::from_ref(&scalar)),
                as_slice_mut::<T, f32>(out),
            ) {
                crate::simd::f32k::broadcast(v, s[0], op, scalar_left, out);
                return true;
            }
            if let (Some(v), Some(s), Some(out)) = (
                as_slice::<T, f64>(values),
                as_slice::<T, f64>(std::slice::from_ref(&scalar)),
                as_slice_mut::<T, f64>(out),
            ) {
                crate::simd::f64k::broadcast(v, s[0], op, scalar_left, out);
                return true;
            }
        }
        false
    }

    /// Writes the elementwise comparison into `out`, returning whether the SIMD
    /// path ran.
    pub fn compare<T: Coefficient>(a: &[T], b: &[T], op: Compare, out: &mut [T]) -> bool {
        if a.len() < MIN_ELEMENTS {
            return false;
        }
        debug_assert!(b.len() == a.len() && out.len() == a.len());
        unsafe {
            if let (Some(a), Some(b), Some(out)) = (
                as_slice::<T, f32>(a),
                as_slice::<T, f32>(b),
                as_slice_mut::<T, f32>(out),
            ) {
                crate::simd::f32k::compare(a, b, op, out);
                return true;
            }
            if let (Some(a), Some(b), Some(out)) = (
                as_slice::<T, f64>(a),
                as_slice::<T, f64>(b),
                as_slice_mut::<T, f64>(out),
            ) {
                crate::simd::f64k::compare(a, b, op, out);
                return true;
            }
        }
        false
    }

    /// Writes the comparison against a splatted scalar into `out`, returning
    /// whether the SIMD path ran.
    pub fn compare_scalar<T: Coefficient>(
        values: &[T],
        scalar: T,
        op: Compare,
        scalar_left: bool,
        out: &mut [T],
    ) -> bool {
        if values.len() < MIN_ELEMENTS {
            return false;
        }
        debug_assert_eq!(out.len(), values.len());
        unsafe {
            if let (Some(v), Some(s), Some(out)) = (
                as_slice::<T, f32>(values),
                as_slice::<T, f32>(std::slice::from_ref(&scalar)),
                as_slice_mut::<T, f32>(out),
            ) {
                crate::simd::f32k::compare_scalar(v, s[0], op, scalar_left, out);
                return true;
            }
            if let (Some(v), Some(s), Some(out)) = (
                as_slice::<T, f64>(values),
                as_slice::<T, f64>(std::slice::from_ref(&scalar)),
                as_slice_mut::<T, f64>(out),
            ) {
                crate::simd::f64k::compare_scalar(v, s[0], op, scalar_left, out);
                return true;
            }
        }
        false
    }

    /// Writes the clamped values into `out`, returning whether the SIMD path
    /// ran.
    pub fn clamp<T: Coefficient>(values: &[T], low: T, high: T, out: &mut [T]) -> bool {
        if values.len() < MIN_ELEMENTS {
            return false;
        }
        debug_assert_eq!(out.len(), values.len());
        unsafe {
            if let (Some(v), Some(low), Some(high), Some(out)) = (
                as_slice::<T, f32>(values),
                as_slice::<T, f32>(std::slice::from_ref(&low)),
                as_slice::<T, f32>(std::slice::from_ref(&high)),
                as_slice_mut::<T, f32>(out),
            ) {
                crate::simd::f32k::clamp(v, low[0], high[0], out);
                return true;
            }
            if let (Some(v), Some(low), Some(high), Some(out)) = (
                as_slice::<T, f64>(values),
                as_slice::<T, f64>(std::slice::from_ref(&low)),
                as_slice::<T, f64>(std::slice::from_ref(&high)),
                as_slice_mut::<T, f64>(out),
            ) {
                crate::simd::f64k::clamp(v, low[0], high[0], out);
                return true;
            }
        }
        false
    }

    /// The whole-slice fold, or `None` for an element type the kernels do not
    /// cover (and for slices too short to be worth the dispatch).
    pub fn reduce<T: Coefficient>(values: &[T], op: Reduce) -> Option<T> {
        if values.len() < MIN_ELEMENTS {
            return None;
        }
        unsafe {
            if let Some(v) = as_slice::<T, f32>(values) {
                return Some(from_f32(crate::simd::f32k::reduce(v, op)));
            }
            if let Some(v) = as_slice::<T, f64>(values) {
                return Some(from_f64(crate::simd::f64k::reduce(v, op)));
            }
        }
        None
    }

    /// Writes `a·b` into `out`, returning whether the SIMD path ran.
    ///
    /// `out` is the caller's final storage, so the product lands in its
    /// destination directly. The earlier shape of this function returned an
    /// owned `Matrix` built from a `vec![0.0; R * C]` scratch buffer, which cost
    /// an allocation plus a second element-by-element pass to copy out — at
    /// `R = K = C = 8` that overhead was roughly twice the arithmetic itself.
    pub fn matmul<T: Coefficient>(
        a: &[T],
        b: &[T],
        rows: usize,
        inner: usize,
        cols: usize,
        out: &mut [T],
    ) -> bool {
        if rows.saturating_mul(inner).saturating_mul(cols) < MIN_MATMUL_OPS {
            return false;
        }
        unsafe {
            if let (Some(a), Some(b), Some(out)) = (
                as_slice::<T, f32>(a),
                as_slice::<T, f32>(b),
                as_slice_mut::<T, f32>(out),
            ) {
                crate::simd::f32k::matmul(a, b, rows, inner, cols, out);
                return true;
            }
            if let (Some(a), Some(b), Some(out)) = (
                as_slice::<T, f64>(a),
                as_slice::<T, f64>(b),
                as_slice_mut::<T, f64>(out),
            ) {
                crate::simd::f64k::matmul(a, b, rows, inner, cols, out);
                return true;
            }
        }
        false
    }

    pub fn matmul_add<T: Coefficient>(
        a: &[T],
        b: &[T],
        rows: usize,
        inner: usize,
        cols: usize,
        addend: &mut [T],
    ) -> bool {
        if rows.saturating_mul(inner).saturating_mul(cols) < MIN_MATMUL_OPS {
            return false;
        }
        unsafe {
            if let (Some(a), Some(b), Some(addend)) = (
                as_slice::<T, f32>(a),
                as_slice::<T, f32>(b),
                as_slice_mut::<T, f32>(addend),
            ) {
                crate::simd::f32k::matmul_accumulate(a, b, rows, inner, cols, addend);
                return true;
            }
            if let (Some(a), Some(b), Some(addend)) = (
                as_slice::<T, f64>(a),
                as_slice::<T, f64>(b),
                as_slice_mut::<T, f64>(addend),
            ) {
                crate::simd::f64k::matmul_accumulate(a, b, rows, inner, cols, addend);
                return true;
            }
        }
        false
    }

    /// Writes `matrix·vector` into `out`, returning whether the SIMD path ran.
    ///
    /// Row-times-vector: each output is a dot product of a matrix row with the
    /// vector, so the per-row reduction kernel is the right shape here.
    pub fn matvec<T: Coefficient>(
        matrix: &[T],
        vector: &[T],
        rows: usize,
        cols: usize,
        out: &mut [T],
    ) -> bool {
        if rows.saturating_mul(cols) < MIN_MATMUL_OPS {
            return false;
        }
        unsafe {
            if let (Some(m), Some(v), Some(out)) = (
                as_slice::<T, f32>(matrix),
                as_slice::<T, f32>(vector),
                as_slice_mut::<T, f32>(out),
            ) {
                for (row, slot) in out.iter_mut().enumerate() {
                    *slot = crate::simd::f32k::dot(&m[row * cols..row * cols + cols], v);
                }
                return true;
            }
            if let (Some(m), Some(v), Some(out)) = (
                as_slice::<T, f64>(matrix),
                as_slice::<T, f64>(vector),
                as_slice_mut::<T, f64>(out),
            ) {
                for (row, slot) in out.iter_mut().enumerate() {
                    *slot = crate::simd::f64k::dot(&m[row * cols..row * cols + cols], v);
                }
                return true;
            }
        }
        false
    }

    /// Adds `matrix·vector` into `addend`, returning whether the SIMD path ran.
    pub fn matvec_add<T: Coefficient>(
        matrix: &[T],
        vector: &[T],
        rows: usize,
        cols: usize,
        addend: &mut [T],
    ) -> bool {
        if rows.saturating_mul(cols) < MIN_MATMUL_OPS {
            return false;
        }
        unsafe {
            if let (Some(m), Some(v), Some(addend)) = (
                as_slice::<T, f32>(matrix),
                as_slice::<T, f32>(vector),
                as_slice_mut::<T, f32>(addend),
            ) {
                for (row, slot) in addend.iter_mut().enumerate() {
                    *slot += crate::simd::f32k::dot(&m[row * cols..row * cols + cols], v);
                }
                return true;
            }
            if let (Some(m), Some(v), Some(addend)) = (
                as_slice::<T, f64>(matrix),
                as_slice::<T, f64>(vector),
                as_slice_mut::<T, f64>(addend),
            ) {
                for (row, slot) in addend.iter_mut().enumerate() {
                    *slot += crate::simd::f64k::dot(&m[row * cols..row * cols + cols], v);
                }
                return true;
            }
        }
        false
    }

    /// Writes `vectorᵀ·matrix` into `out`, returning whether the SIMD path ran.
    pub fn vecmat<T: Coefficient>(
        vector: &[T],
        matrix: &[T],
        rows: usize,
        cols: usize,
        out: &mut [T],
    ) -> bool {
        if rows.saturating_mul(cols) < MIN_MATMUL_OPS {
            return false;
        }
        // (1×R)·(R×C): the broadcast-A matmul vectorizes across the C columns.
        matmul(vector, matrix, 1, rows, cols, out)
    }

    /// In-place radix-2 FFT for power-of-two `f32` lengths. Returns `false` (so
    /// the caller keeps the generic path) for every other element type or shape.
    /// `direction` is `-1.0` forward, `+1.0` inverse; normalization stays with
    /// the caller.
    pub fn radix2_fft<T: Coefficient>(output: &mut [Complex<T>], direction: f64) -> bool {
        let n = output.len();
        if n < MIN_FFT_LENGTH || !n.is_power_of_two() || !is::<T, f32>() {
            return false;
        }
        // SAFETY: T is f32 and `Complex` is `#[repr(C)]`, so the buffer is
        // exactly `[re, im, …]` — `2*n` contiguous f32.
        let buf =
            unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr().cast::<f32>(), 2 * n) };
        crate::simd::fft_f32::radix2(buf, n, direction as f32);
        true
    }
}

// ---- vectors ----------------------------------------------------------------

/// A vector whose length is fixed when it is built, backed by a `Vec<T>` on the
/// default [`Host`] backend.
pub struct Vector<T, B: Backend = Host> {
    len: usize,
    data: B::Vector<T>,
}

// The storage type varies with the backend, so these are the derives written by
// hand: which of them a tensor gets depends on what its storage supports.
impl<T, B: Backend> Clone for Vector<T, B>
where
    B::Vector<T>: Clone,
{
    fn clone(&self) -> Self {
        Vector {
            len: self.len,
            data: self.data.clone(),
        }
    }
}

impl<T, B: Backend> PartialEq for Vector<T, B>
where
    B::Vector<T>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.data == other.data
    }
}

impl<T, B: Backend> Eq for Vector<T, B> where B::Vector<T>: Eq {}

impl<T, B: Backend> fmt::Debug for Vector<T, B>
where
    B::Vector<T>: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Vector")
            .field("len", &self.len)
            .field("data", &self.data)
            .finish()
    }
}

/// The shape queries, which read a field and so need nothing of the backend or
/// the element type.
impl<T, B: Backend> Vector<T, B> {
    /// The number of elements.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether this vector holds no elements.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// `f32` vectors on any backend. These are the operations that do not depend on
/// where the elements live — everything else is implemented per backend.
impl<B: Backend> Vector<f32, B> {
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
    pub fn to_backend<B2: Backend>(&self) -> Vector<f32, B2> {
        Vector {
            len: self.len,
            data: B2::store_vector(B::vector_slice(&self.data)),
        }
    }

    /// A vector of `len` elements, every one set to `value`, allocated directly
    /// on backend `B`.
    pub fn filled(len: usize, value: f32) -> Self {
        Vector {
            len,
            data: B::store_vector(&vec![value; len]),
        }
    }

    /// An arithmetic progression: `start`, `start + step`, `start + 2·step`, …
    ///
    /// Index vectors are what make position-dependent arithmetic — "divide the
    /// running total by how many terms it covers", "which entry was this?" —
    /// expressible as whole-tensor operations rather than as a loop. Each
    /// element is computed from its own index rather than from the one before,
    /// so an integer ramp is exact.
    ///
    /// ```
    /// use tensorcrate::tensors::{Host, Vector};
    ///
    /// let counts = Vector::<f32, Host>::ramp(4, 1.0, 1.0);
    /// assert_eq!(counts.as_slice(), [1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn ramp(len: usize, start: f32, step: f32) -> Self {
        let values = (0..len)
            .map(|index| start + step * index as f32)
            .collect::<Vec<_>>();
        Vector {
            len,
            data: B::store_vector(&values),
        }
    }

    /// Borrow the elements as a slice, without copying.
    pub fn as_slice(&self) -> &[f32] {
        B::vector_slice(&self.data)
    }

    /// Build from `f32` values on backend `B`. The length is the slice's.
    pub(crate) fn build(values: &[f32]) -> Self {
        Vector {
            len: values.len(),
            data: B::store_vector(values),
        }
    }

    /// Attach a length to storage that already holds exactly that many values —
    /// how a tensor is rebuilt from the result of a kernel dispatch.
    pub(crate) fn from_storage(len: usize, data: B::Vector<f32>) -> Self {
        Vector { len, data }
    }

    /// The backend storage itself, which the kernels hand straight to a
    /// dispatch.
    pub(crate) fn storage(&self) -> &B::Vector<f32> {
        &self.data
    }

    /// The backend storage itself, for a dispatch that accumulates in place.
    pub(crate) fn storage_mut(&mut self) -> &mut B::Vector<f32> {
        &mut self.data
    }

    /// Consume this vector and take its storage, which is how a reshape moves
    /// the elements instead of copying them.
    pub(crate) fn into_storage(self) -> B::Vector<f32> {
        self.data
    }

    /// Copy the elements into a `Vec`.
    pub fn to_vec(&self) -> Vec<f32> {
        self.as_slice().to_vec()
    }

    /// Consume this vector and view its elements as a `1 × len` row matrix.
    ///
    /// On the Metal backend this only changes the recorded shape; the existing
    /// allocation is reused without a copy or kernel dispatch.
    pub fn into_row_matrix(self) -> Matrix<f32, B> {
        Matrix {
            rows: 1,
            cols: self.len,
            data: B::vector_into_matrix(self.data),
        }
    }

    /// Consume this vector and view its elements as a `len × 1` column matrix.
    ///
    /// As with [`into_row_matrix`](Self::into_row_matrix), Metal reuses the
    /// existing allocation.
    pub fn into_column_matrix(self) -> Matrix<f32, B> {
        Matrix {
            rows: self.len,
            cols: 1,
            data: B::vector_into_matrix(self.data),
        }
    }
}

impl<T> Vector<T, Host> {
    /// A vector from its elements.
    ///
    /// Both a fixed-size array and a `Vec` are accepted, so a length known in
    /// source and one computed at runtime are written the same way:
    ///
    /// ```
    /// use tensorcrate::tensors::Vector;
    ///
    /// let literal = Vector::new([1.0, 2.0, 3.0]);
    /// let computed = Vector::new((0..3).map(|i| i as f64 + 1.0).collect::<Vec<_>>());
    /// assert_eq!(literal, computed);
    /// ```
    pub fn new(data: impl Into<Vec<T>>) -> Self {
        let data = data.into();
        Vector {
            len: data.len(),
            data,
        }
    }

    /// Borrow the elements as a slice.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Borrow the elements as a mutable slice.
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Consume this vector and take its elements.
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    pub fn get(&self, i: usize) -> Option<&T> {
        self.data.get(i)
    }

    /// Apply `f` to every element, producing a vector of the new element type —
    /// e.g. lifting a `Vector<f64>` into a `Vector<Complex<f64>>`.
    pub fn map<U>(&self, f: impl Fn(&T) -> U) -> Vector<U, Host> {
        Vector {
            len: self.len,
            data: self.data.iter().map(f).collect(),
        }
    }
}

impl<T> Index<usize> for Vector<T, Host> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        &self.data[index]
    }
}

impl<T: Coefficient> Vector<T, Host> {
    /// A vector of `len` zeros.
    pub fn zeros(len: usize) -> Self {
        Vector {
            len,
            data: vec![T::zero(); len],
        }
    }

    /// A vector of `len` elements, every one set to `value`.
    pub fn repeat(len: usize, value: T) -> Self {
        Vector {
            len,
            data: vec![value; len],
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
            let mut out = vec![T::zero(); self.len];
            if simd_dispatch::broadcast(&self.data, scalar, op, false, &mut out) {
                return Vector::new(out);
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
            let mut out = vec![T::zero(); self.len];
            if simd_dispatch::broadcast(&self.data, scalar, op, true, &mut out) {
                return Vector::new(out);
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

    /// Dot product.
    ///
    /// # Panics
    ///
    /// If the two lengths differ.
    #[track_caller]
    pub fn dot(&self, other: &Vector<T, Host>) -> T {
        assert_same_len(self.len, other.len, "dot");
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if let Some(output) = simd_dispatch::dot(&self.data, &other.data) {
            return output;
        }
        let mut sum = T::zero();
        for i in 0..self.len {
            sum = sum + self.data[i] * other.data[i];
        }
        sum
    }

    /// Row vector times matrix: `(1×N)·(N×C) = (1×C)`.
    ///
    /// # Panics
    ///
    /// If this vector's length is not the matrix's row count.
    #[track_caller]
    pub fn vecmat(&self, m: &Matrix<T, Host>) -> Vector<T, Host> {
        assert_inner((1, self.len), m.shape(), "vecmat");
        let (rows, cols) = (m.rows, m.cols);
        let mut out = vec![T::zero(); cols];

        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if simd_dispatch::vecmat(&self.data, &m.data, rows, cols, &mut out) {
            return Vector::new(out);
        }

        for (j, slot) in out.iter_mut().enumerate() {
            let mut sum = T::zero();
            for p in 0..rows {
                sum = sum + self.data[p] * m.data[p * cols + j];
            }
            *slot = sum;
        }
        Vector::new(out)
    }

    /// Discrete Fourier transform.
    ///
    /// Power-of-two lengths use iterative radix-2 Cooley–Tukey (`O(N log N)`).
    /// Other composite lengths use a recursive mixed-radix Cooley–Tukey
    /// decomposition for radices up to 15. Sub-transforms without a factor in
    /// that range use the definition directly, so every length—including zero
    /// and one—is supported.
    ///
    /// This is the conventional unnormalized forward transform:
    /// `X[k] = Σ x[n] exp(-2πikn/N)`.
    pub fn fft(&self) -> Vector<Complex<T>, Host>
    where
        T: Float,
    {
        let n = self.len;
        let mut output = self
            .data
            .iter()
            .map(|&x| Complex::new(x, <T as num_traits::Zero>::zero()))
            .collect::<Vec<_>>();

        if n <= 1 {
            return Vector::new(output);
        }

        if !n.is_power_of_two() {
            mixed_radix_fft(&mut output, false);
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

/// The smaller of two values, ordering NaN the way [`f32::min`] does: an
/// unordered pair keeps whichever operand is not NaN.
///
/// The vector kernels use `fminnm`, which is that same rule in hardware, so the
/// two paths agree on every input rather than only on the ordered ones.
fn ordered_min<T: PartialOrd + Copy>(a: T, b: T) -> T {
    match a.partial_cmp(&b) {
        Some(Ordering::Greater) => b,
        Some(_) => a,
        // Unordered: a value that does not compare with itself is the NaN, so
        // the other operand wins.
        None if a.partial_cmp(&a).is_none() => b,
        None => a,
    }
}

/// The larger of two values; see [`ordered_min`].
fn ordered_max<T: PartialOrd + Copy>(a: T, b: T) -> T {
    match a.partial_cmp(&b) {
        Some(Ordering::Less) => b,
        Some(_) => a,
        None if a.partial_cmp(&a).is_none() => b,
        None => a,
    }
}

impl<T> Vector<T, Host> {
    /// Sort in place under an arbitrary comparator.
    ///
    /// This is [`slice::sort_by`], so the sort is stable and the comparator may
    /// be anything — a key projection, a reversed order, a tie-break across two
    /// fields. Floats have no total [`Ord`], which is exactly why the ordering
    /// is a parameter here; [`f32::total_cmp`] is the usual argument, and
    /// [`sort`](Vector::sort) is the name for it.
    ///
    /// ```
    /// use tensorcrate::tensors::Vector;
    ///
    /// let mut v = Vector::new([3.0f32, -1.0, 2.0]);
    /// v.sort_by(|left, right| right.total_cmp(left)); // descending
    /// assert_eq!(v.data(), [3.0, 2.0, -1.0]);
    /// ```
    pub fn sort_by(&mut self, compare: impl FnMut(&T, &T) -> Ordering) {
        self.data.sort_by(compare);
    }

    /// [`sort_by`](Vector::sort_by) into a new vector, leaving this one alone.
    pub fn sorted_by(&self, compare: impl FnMut(&T, &T) -> Ordering) -> Self
    where
        T: Clone,
    {
        let mut sorted = self.clone();
        sorted.sort_by(compare);
        sorted
    }
}

impl<T: Coefficient> Vector<T, Host> {
    /// The sum of every element; `0` for an empty vector.
    pub fn sum(&self) -> T {
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if let Some(total) = simd_dispatch::reduce(&self.data, Reduce::Sum) {
            return total;
        }
        self.data
            .iter()
            .fold(T::zero(), |total, &value| total + value)
    }

    /// Inclusive prefix sum: element `i` of the result is the sum of elements
    /// `0..=i`.
    ///
    /// The scan is the one operation here with a serial dependency — each output
    /// needs the one before it — so the CPU path is an ordinary running total,
    /// which is already about as fast as the memory it streams. The parallel
    /// version is the GPU one: see
    /// [`Kernels::vector_prefix_sum`](crate::tensors::Kernels::vector_prefix_sum),
    /// where `log n` sweeps beat the serial chain because there are thousands of
    /// threads to spend on it.
    pub fn prefix_sum(&self) -> Self {
        let mut running = T::zero();
        let mut out = Vec::with_capacity(self.len);
        for &value in &self.data {
            running = running + value;
            out.push(running);
        }
        Vector::new(out)
    }
}

impl<T: Coefficient + PartialOrd> Vector<T, Host> {
    /// Elementwise minimum with another vector.
    ///
    /// # Panics
    ///
    /// If the two lengths differ.
    #[track_caller]
    pub fn min(&self, other: &Self) -> Self {
        assert_same_len(self.len, other.len, "min");
        self.pairwise(other, Compare::Min, ordered_min)
    }

    /// Elementwise maximum with another vector.
    ///
    /// # Panics
    ///
    /// If the two lengths differ.
    #[track_caller]
    pub fn max(&self, other: &Self) -> Self {
        assert_same_len(self.len, other.len, "max");
        self.pairwise(other, Compare::Max, ordered_max)
    }

    /// The lesser of each element and `scalar`.
    pub fn min_scalar(&self, scalar: T) -> Self {
        self.against_scalar(scalar, Compare::Min, ordered_min)
    }

    /// The greater of each element and `scalar` — `max_scalar(0)` is a relu.
    pub fn max_scalar(&self, scalar: T) -> Self {
        self.against_scalar(scalar, Compare::Max, ordered_max)
    }

    /// Confine every element to `[low, high]`.
    ///
    /// One pass, rather than the two a `max_scalar` followed by a `min_scalar`
    /// would make.
    ///
    /// ```
    /// use tensorcrate::tensors::Vector;
    ///
    /// let v = Vector::new([-2.0f32, 0.5, 7.0]);
    /// assert_eq!(v.clamp(0.0, 1.0).data(), [0.0, 0.5, 1.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// If `low > high`, which has no answer to give. A NaN element clamps to
    /// `low`, following `x.max(low).min(high)`.
    #[track_caller]
    pub fn clamp(&self, low: T, high: T) -> Self {
        assert_ordered_bounds(&low, &high);
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let mut out = vec![T::zero(); self.len];
            if simd_dispatch::clamp(&self.data, low, high, &mut out) {
                return Vector::new(out);
            }
        }
        self.map(|&value| ordered_min(ordered_max(value, low), high))
    }

    /// The smallest element, or `None` when there are none.
    pub fn minimum(&self) -> Option<T> {
        self.fold_extreme(Reduce::Min, ordered_min)
    }

    /// The largest element, or `None` when there are none.
    pub fn maximum(&self) -> Option<T> {
        self.fold_extreme(Reduce::Max, ordered_max)
    }

    fn pairwise(&self, other: &Self, op: Compare, scalar: fn(T, T) -> T) -> Self {
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let mut out = vec![T::zero(); self.len];
            if simd_dispatch::compare(&self.data, &other.data, op, &mut out) {
                return Vector::new(out);
            }
        }
        let _ = op;
        Vector::new(
            self.data
                .iter()
                .zip(&other.data)
                .map(|(&left, &right)| scalar(left, right))
                .collect::<Vec<_>>(),
        )
    }

    fn against_scalar(&self, value: T, op: Compare, scalar: fn(T, T) -> T) -> Self {
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let mut out = vec![T::zero(); self.len];
            if simd_dispatch::compare_scalar(&self.data, value, op, false, &mut out) {
                return Vector::new(out);
            }
        }
        let _ = op;
        self.map(|&element| scalar(element, value))
    }

    fn fold_extreme(&self, op: Reduce, scalar: fn(T, T) -> T) -> Option<T> {
        if self.data.is_empty() {
            return None;
        }
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if let Some(extreme) = simd_dispatch::reduce(&self.data, op) {
            return Some(extreme);
        }
        let _ = op;
        let (&first, rest) = self.data.split_first()?;
        Some(rest.iter().fold(first, |best, &value| scalar(best, value)))
    }
}

impl Vector<f32, Host> {
    /// Elementwise comparison with another vector.
    ///
    /// The same operation the [`Metal`] backend runs as one dispatch, so code
    /// written against [`Kernels`] means the same thing on either.
    ///
    /// # Panics
    ///
    /// If the two lengths differ.
    #[track_caller]
    pub fn compare(&self, other: &Self, op: Compare) -> Self {
        assert_same_len(self.len, other.len, "compare");
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let mut out = vec![0.0; self.len];
            if simd_dispatch::compare(&self.data, &other.data, op, &mut out) {
                return Vector::new(out);
            }
        }
        Vector::new(
            self.data
                .iter()
                .zip(&other.data)
                .map(|(&left, &right)| op.value(left, right))
                .collect::<Vec<_>>(),
        )
    }

    /// Elementwise comparison against a scalar; `scalar_left` puts the scalar on
    /// the left, which matters for every op but `Min` and `Max`.
    pub fn compare_scalar(&self, scalar: f32, op: Compare, scalar_left: bool) -> Self {
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let mut out = vec![0.0; self.len];
            if simd_dispatch::compare_scalar(&self.data, scalar, op, scalar_left, &mut out) {
                return Vector::new(out);
            }
        }
        self.map(|&value| {
            if scalar_left {
                op.value(scalar, value)
            } else {
                op.value(value, scalar)
            }
        })
    }

    /// Fold the whole vector to one value. An empty vector gives
    /// [`op.identity()`](Reduce::identity).
    pub fn reduce(&self, op: Reduce) -> f32 {
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if let Some(total) = simd_dispatch::reduce(&self.data, op) {
            return total;
        }
        op.fold(&self.data)
    }

    /// Sort in place, in [`SortOrder`]'s total order.
    ///
    /// The comparator is [`f32::total_cmp`], so `−0.0` precedes `+0.0` and NaNs
    /// sort to the ends by sign instead of landing wherever the partial order
    /// left them. That is the same order the GPU sort produces.
    pub fn sort(&mut self, order: SortOrder) {
        self.data.sort_unstable_by(order.comparator());
    }

    /// [`sort`](Vector::sort) into a new vector.
    pub fn sorted(&self, order: SortOrder) -> Self {
        let mut sorted = self.clone();
        sorted.sort(order);
        sorted
    }
}

impl<T: Float + Coefficient> Vector<Complex<T>, Host> {
    /// Inverse discrete Fourier transform.
    ///
    /// This uses the same radix-2 and mixed-radix Cooley–Tukey paths as
    /// [`Vector::fft`], with a direct DFT for leaves whose smallest factor
    /// exceeds 15. It is the conventional normalized inverse transform:
    /// `x[n] = (1/N) Σ X[k] exp(2πikn/N)`.
    pub fn ifft(&self) -> Vector<Complex<T>, Host> {
        let n = self.len;
        if n <= 1 {
            return self.clone();
        }

        let mut output = self.data.clone();
        if n.is_power_of_two() {
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
            mixed_radix_fft(&mut output, true);
        }

        let normalization = cast::<T>(n);
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
fn mixed_radix_fft<T: Float + Coefficient>(data: &mut [Complex<T>], inverse: bool) {
    let n = data.len();
    if n <= 1 {
        return;
    }

    let direction = if inverse {
        <T as num_traits::One>::one()
    } else {
        -<T as num_traits::One>::one()
    };
    let zero = Complex::new(
        <T as num_traits::Zero>::zero(),
        <T as num_traits::Zero>::zero(),
    );

    // Two allocations for the whole transform: one workspace and the root
    // table. The recursion below borrows slices of these rather than allocating
    // per node — the earlier shape allocated a subsequence per residue plus an
    // output at every node of the tree, which for `n = 1000` (2³·5³, roughly
    // 1250 nodes) meant thousands of allocations per call.
    let unit = roots_of_unity(n, direction);
    let mut workspace = vec![zero; n];
    transform(data, &mut workspace, &unit, 1);
}

/// Transforms `data` in place, using `workspace` (same length) as scratch.
///
/// `unit` is the root table for the *top-level* length, shared by every node.
/// A node of length `m` needs the `m`-th roots, which are a stride-`stride`
/// subsequence of it: `exp(2πik/m) == unit[k · stride]` exactly when
/// `m · stride` equals the top-level length. Each descent multiplies `stride`
/// by the radix it split off, so no node ever needs a table of its own.
fn transform<T: Float + Coefficient>(
    data: &mut [Complex<T>],
    workspace: &mut [Complex<T>],
    unit: &[Complex<T>],
    stride: usize,
) {
    let m = data.len();
    if m <= 1 {
        return;
    }
    let zero = Complex::new(
        <T as num_traits::Zero>::zero(),
        <T as num_traits::Zero>::zero(),
    );

    let Some(radix) = smallest_mixed_radix(m) else {
        // No supported factor: evaluate the definition directly.
        for frequency in 0..m {
            let mut sum = zero;
            for (index, &value) in data.iter().enumerate() {
                sum = sum + value * unit[frequency * index % m * stride];
            }
            workspace[frequency] = sum;
        }
        data.copy_from_slice(&workspace[..m]);
        return;
    };

    let quotient = m / radix;

    // Gather each residue class into its own contiguous block of `workspace`.
    for residue in 0..radix {
        for index in 0..quotient {
            workspace[residue * quotient + index] = data[residue + radix * index];
        }
    }

    // Transform each block. `data` has been fully consumed by the gather, so
    // the matching block of it is free to serve as that sub-call's workspace.
    for residue in 0..radix {
        let (start, end) = (residue * quotient, (residue + 1) * quotient);
        transform(
            &mut workspace[start..end],
            &mut data[start..end],
            unit,
            stride * radix,
        );
    }

    // Combine: output frequency `low + quotient·high` sums one sample from each
    // residue class, phase-shifted by the corresponding root.
    for high in 0..radix {
        for low in 0..quotient {
            let frequency = low + quotient * high;
            let mut sum = zero;
            for residue in 0..radix {
                sum = sum
                    + workspace[residue * quotient + low] * unit[residue * frequency % m * stride];
            }
            data[frequency] = sum;
        }
    }
}

/// `exp(direction · 2πk/n)` for every `k` in `0..n`.
///
/// The transform only ever needs these `n` angles, because the exponent is
/// periodic modulo `n`. Computing them once and indexing by the reduced
/// exponent replaces the quadratic number of `cos`/`sin` calls the direct and
/// mixed-radix paths used to make — and is more accurate besides, since the
/// angle handed to `cos` stays inside one period instead of growing to
/// `2π·(n−1)²/n` and losing precision to argument reduction.
fn roots_of_unity<T: Float + Coefficient>(n: usize, direction: T) -> Vec<Complex<T>> {
    let tau = cast::<T>(std::f64::consts::TAU);
    (0..n)
        .map(|k| {
            let angle = direction * tau * cast::<T>(k) / cast::<T>(n);
            Complex::new(angle.cos(), angle.sin())
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

/// A matrix whose shape is fixed when it is built, backed by a flat row-major
/// `Vec<T>` on the default [`Host`] backend.
pub struct Matrix<T, B: Backend = Host> {
    rows: usize,
    cols: usize,
    data: B::Matrix<T>,
}

// As for `Vector`: hand-written derives, because the storage type — and so which
// of these a tensor gets — depends on the backend.
impl<T, B: Backend> Clone for Matrix<T, B>
where
    B::Matrix<T>: Clone,
{
    fn clone(&self) -> Self {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.clone(),
        }
    }
}

impl<T, B: Backend> PartialEq for Matrix<T, B>
where
    B::Matrix<T>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.shape() == other.shape() && self.data == other.data
    }
}

impl<T, B: Backend> Eq for Matrix<T, B> where B::Matrix<T>: Eq {}

impl<T, B: Backend> fmt::Debug for Matrix<T, B>
where
    B::Matrix<T>: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Matrix")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("data", &self.data)
            .finish()
    }
}

/// The shape queries, as for [`Vector`].
impl<T, B: Backend> Matrix<T, B> {
    /// The `(rows, columns)` extents.
    pub const fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// The number of rows.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// The number of columns.
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Whether this matrix holds no elements.
    pub const fn is_empty(&self) -> bool {
        self.rows == 0 || self.cols == 0
    }

    /// Whether this matrix is square, which the determinant and inverse need.
    pub const fn is_square(&self) -> bool {
        self.rows == self.cols
    }
}

/// `f32` matrices on any backend, as for [`Vector`] above.
impl<B: Backend> Matrix<f32, B> {
    /// Move this matrix's elements onto backend `B2`.
    ///
    /// The counterpart to [`Vector::to_backend`], and the only place a
    /// `Metal`-backed chain copies between CPU and GPU memory.
    pub fn to_backend<B2: Backend>(&self) -> Matrix<f32, B2> {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: B2::store_matrix(B::matrix_slice(&self.data)),
        }
    }

    /// A `rows × cols` matrix with every element set to `value`, allocated
    /// directly on backend `B`.
    pub fn filled(rows: usize, cols: usize, value: f32) -> Self {
        Matrix {
            rows,
            cols,
            data: B::store_matrix(&vec![value; rows * cols]),
        }
    }

    /// Borrow the elements as one flat row-major slice, without copying.
    pub fn as_slice(&self) -> &[f32] {
        B::matrix_slice(&self.data)
    }

    /// Build from row-major `f32` values on backend `B`.
    pub(crate) fn build(rows: usize, cols: usize, values: &[f32]) -> Self {
        debug_assert_eq!(values.len(), rows * cols);
        Matrix {
            rows,
            cols,
            data: B::store_matrix(values),
        }
    }

    /// Attach a shape to storage that already holds exactly `rows * cols`
    /// values — how a tensor is rebuilt from the result of a kernel dispatch.
    pub(crate) fn from_storage(rows: usize, cols: usize, data: B::Matrix<f32>) -> Self {
        Matrix { rows, cols, data }
    }

    /// The backend storage itself, which the kernels hand straight to a
    /// dispatch.
    pub(crate) fn storage(&self) -> &B::Matrix<f32> {
        &self.data
    }

    /// The backend storage itself, for a dispatch that accumulates in place.
    pub(crate) fn storage_mut(&mut self) -> &mut B::Matrix<f32> {
        &mut self.data
    }

    /// Consume this matrix and take its storage, which is how a reshape moves
    /// the elements instead of copying them.
    pub(crate) fn into_storage(self) -> B::Matrix<f32> {
        self.data
    }
}

/// A type-erased matrix used to assemble a heterogeneous matrix chain.
/// Construct one with `MatrixOperand::from(&matrix)`.
///
/// Rust slices cannot directly contain matrices whose shapes differ, because the
/// values would have different sizes. This small owned adapter carries the
/// shape alongside the elements so a chain can be held in one slice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatrixOperand<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T: Copy> From<&Matrix<T, Host>> for MatrixOperand<T> {
    fn from(matrix: &Matrix<T, Host>) -> Self {
        Self {
            rows: matrix.rows,
            cols: matrix.cols,
            data: matrix.data.clone(),
        }
    }
}

impl<T> MatrixOperand<T> {
    /// The `(rows, columns)` extents.
    pub const fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
}

/// Multiply a heterogeneous chain in the optimal parenthesization.
///
/// The optimal order is found with the classic matrix-chain dynamic program; use
/// [`chained_matmul_cost`] when only the optimal cost is needed, since that
/// function uses the `O(n log n)` Hu–Shing solver.
///
/// Returns [`Error::Shape`] when consecutive operands do not meet, and
/// [`Error::InvalidArgument`] for a chain shorter than two matrices or one with
/// a zero extent.
pub fn chained_matmul<T: Coefficient>(
    matrices: &[MatrixOperand<T>],
) -> Result<Matrix<T, Host>, Error> {
    let dims = validate_chain(matrices)?;

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
    Ok(Matrix {
        rows: result.rows,
        cols: result.cols,
        data: result.data,
    })
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

impl<T> Matrix<T, Host> {
    /// A matrix from its rows.
    ///
    /// Accepts anything that iterates over row-like values, so a literal and a
    /// runtime-built `Vec<Vec<T>>` are written the same way:
    ///
    /// ```
    /// use tensorcrate::tensors::Matrix;
    ///
    /// let literal = Matrix::from_rows([[1.0, 2.0], [3.0, 4.0]]);
    /// let computed = Matrix::from_rows((0..2).map(|r| vec![2.0 * r as f64 + 1.0, 2.0 * r as f64 + 2.0]));
    /// assert_eq!(literal, computed);
    /// ```
    ///
    /// # Panics
    ///
    /// If the rows are not all the same length.
    #[track_caller]
    pub fn from_rows<I, R>(rows: I) -> Self
    where
        I: IntoIterator<Item = R>,
        R: Into<Vec<T>>,
    {
        let mut data = Vec::new();
        let mut count = 0;
        let mut cols = 0;
        for row in rows {
            let mut row = row.into();
            if count == 0 {
                cols = row.len();
            } else {
                assert!(
                    row.len() == cols,
                    "from_rows: row {count} has {} elements, but row 0 has {cols}",
                    row.len()
                );
            }
            data.append(&mut row);
            count += 1;
        }
        Matrix {
            rows: count,
            cols,
            data,
        }
    }

    /// A matrix from its row-major elements and an explicit shape.
    ///
    /// # Panics
    ///
    /// If the element count is not `rows * cols`.
    #[track_caller]
    pub fn from_flat(rows: usize, cols: usize, data: impl Into<Vec<T>>) -> Self {
        let data = data.into();
        assert!(
            data.len() == rows * cols,
            "from_flat: {} elements cannot fill a {rows}×{cols} matrix",
            data.len()
        );
        Matrix { rows, cols, data }
    }

    /// Borrow the elements as one flat row-major slice.
    ///
    /// Row `i`, column `j` is at `i * cols + j`; [`row`](Self::row) and the
    /// `(row, column)` index do that arithmetic for you.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Borrow the elements as one flat row-major mutable slice.
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Consume this matrix and take its row-major elements.
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    /// Borrow one row.
    #[track_caller]
    pub fn row(&self, row: usize) -> &[T] {
        assert!(
            row < self.rows,
            "row {row} is out of range for {} rows",
            self.rows
        );
        &self.data[row * self.cols..(row + 1) * self.cols]
    }

    /// Iterate over the rows.
    pub fn row_iter(&self) -> impl Iterator<Item = &[T]> {
        (0..self.rows).map(move |row| self.row(row))
    }

    /// Copy the elements into a vector of rows.
    ///
    /// Host-only, since it hands back owned elements; a resident tensor reaches
    /// it through [`to_backend::<Host>()`](Self::to_backend).
    pub fn to_rows(&self) -> Vec<Vec<T>>
    where
        T: Clone,
    {
        self.row_iter().map(<[T]>::to_vec).collect()
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row >= self.rows || col >= self.cols {
            return None;
        }
        self.data.get(row * self.cols + col)
    }

    /// Apply `f` to every element, producing a matrix of the new element type.
    pub fn map<U>(&self, f: impl Fn(&T) -> U) -> Matrix<U, Host> {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(f).collect(),
        }
    }
}

impl<T> Index<(usize, usize)> for Matrix<T, Host> {
    type Output = T;

    #[track_caller]
    fn index(&self, (row, col): (usize, usize)) -> &T {
        assert!(
            row < self.rows && col < self.cols,
            "index ({row}, {col}) is out of range for a {}×{} matrix",
            self.rows,
            self.cols
        );
        &self.data[row * self.cols + col]
    }
}

impl<T: Coefficient> Matrix<T, Host> {
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

    /// A `rows × cols` matrix of zeros.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![T::zero(); rows * cols],
        }
    }

    /// A `rows × cols` matrix with every element set to `value`.
    pub fn repeat(rows: usize, cols: usize, value: T) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![value; rows * cols],
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
            let mut out = vec![T::zero(); self.data.len()];
            if simd_dispatch::broadcast(&self.data, scalar, op, false, &mut out) {
                return Matrix::from_flat(self.rows, self.cols, out);
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
            let mut out = vec![T::zero(); self.data.len()];
            if simd_dispatch::broadcast(&self.data, scalar, op, true, &mut out) {
                return Matrix::from_flat(self.rows, self.cols, out);
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

    /// Matrix product `(R×K)·(K×C) = (R×C)`.
    ///
    /// # Panics
    ///
    /// If this matrix's column count is not the other's row count.
    #[track_caller]
    pub fn matmul(&self, other: &Matrix<T, Host>) -> Matrix<T, Host> {
        assert_inner(self.shape(), other.shape(), "matmul");
        let (rows, inner, cols) = (self.rows, self.cols, other.cols);
        let mut out = vec![T::zero(); rows * cols];

        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if simd_dispatch::matmul(&self.data, &other.data, rows, inner, cols, &mut out) {
            return Matrix::from_flat(rows, cols, out);
        }

        for i in 0..rows {
            for j in 0..cols {
                let mut sum = T::zero();
                for p in 0..inner {
                    sum = sum + self.data[i * inner + p] * other.data[p * cols + j];
                }
                out[i * cols + j] = sum;
            }
        }
        Matrix::from_flat(rows, cols, out)
    }

    /// Fused matrix multiply-add: `self·other + addend`.
    ///
    /// `addend` is consumed and used as the accumulator, avoiding a separate
    /// product allocation and elementwise addition.
    ///
    /// # Panics
    ///
    /// If the inner dimensions disagree, or `addend` is not `rows × other.cols`.
    #[track_caller]
    pub fn matmul_add(&self, other: &Matrix<T, Host>, addend: Matrix<T, Host>) -> Matrix<T, Host> {
        assert_inner(self.shape(), other.shape(), "matmul_add");
        assert_same_shape(addend.shape(), (self.rows, other.cols), "matmul_add addend");
        let (rows, inner, cols) = (self.rows, self.cols, other.cols);
        let mut output = addend;

        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if simd_dispatch::matmul_add(&self.data, &other.data, rows, inner, cols, &mut output.data) {
            return output;
        }

        for i in 0..rows {
            for p in 0..inner {
                for j in 0..cols {
                    output.data[i * cols + j] = output.data[i * cols + j]
                        + self.data[i * inner + p] * other.data[p * cols + j];
                }
            }
        }
        output
    }

    /// Matrix times column vector: `(R×C)·(C×1) = (R×1)`.
    ///
    /// # Panics
    ///
    /// If the vector's length is not this matrix's column count.
    #[track_caller]
    pub fn matvec(&self, v: &Vector<T, Host>) -> Vector<T, Host> {
        assert_inner(self.shape(), (v.len, 1), "matvec");
        let (rows, cols) = (self.rows, self.cols);
        let mut out = vec![T::zero(); rows];

        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if simd_dispatch::matvec(&self.data, &v.data, rows, cols, &mut out) {
            return Vector::new(out);
        }

        for (i, slot) in out.iter_mut().enumerate() {
            let mut sum = T::zero();
            for p in 0..cols {
                sum = sum + self.data[i * cols + p] * v.data[p];
            }
            *slot = sum;
        }
        Vector::new(out)
    }

    /// Fused matrix-vector multiply-add: `self·v + addend`.
    ///
    /// Each row uses the SIMD dot-product kernel when available, and the owned
    /// addend is updated in place.
    ///
    /// # Panics
    ///
    /// If the vector's length is not this matrix's column count, or the addend's
    /// length is not its row count.
    #[track_caller]
    pub fn matvec_add(&self, v: &Vector<T, Host>, addend: Vector<T, Host>) -> Vector<T, Host> {
        assert_inner(self.shape(), (v.len, 1), "matvec_add");
        assert_same_len(addend.len, self.rows, "matvec_add addend");
        let (rows, cols) = (self.rows, self.cols);
        let mut output = addend;

        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if simd_dispatch::matvec_add(&self.data, &v.data, rows, cols, &mut output.data) {
            return output;
        }

        for i in 0..rows {
            let mut sum = T::zero();
            for p in 0..cols {
                sum = sum + self.data[i * cols + p] * v.data[p];
            }
            output.data[i] = output.data[i] + sum;
        }
        output
    }

    /// Transpose: an `R×C` matrix becomes `C×R`.
    pub fn transpose(&self) -> Matrix<T, Host> {
        let (rows, cols) = (self.rows, self.cols);
        let mut out = vec![T::zero(); rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                out[j * rows + i] = self.data[i * cols + j];
            }
        }
        Matrix::from_flat(cols, rows, out)
    }

    /// The `n × n` identity matrix.
    pub fn identity(n: usize) -> Self {
        let mut data = vec![T::zero(); n * n];
        for i in 0..n {
            data[i * n + i] = T::one();
        }
        Matrix::from_flat(n, n, data)
    }

    /// Determinant, by fraction-free (Bareiss) elimination — every division is
    /// exact, so an integer matrix keeps an exact integer determinant.
    ///
    /// # Panics
    ///
    /// If the matrix is not square.
    #[track_caller]
    pub fn determinant(&self) -> T {
        assert!(
            self.is_square(),
            "determinant: matrix is {}×{}, not square",
            self.rows,
            self.cols
        );
        let n = self.rows;
        if n == 0 {
            return T::one();
        }
        let mut m = self.data.clone();
        let at = |i: usize, j: usize| i * n + j;
        let mut prev = T::one();
        let mut negate = false;

        for k in 0..n - 1 {
            if m[at(k, k)].is_zero() {
                match (k + 1..n).find(|&p| !m[at(p, k)].is_zero()) {
                    Some(p) => {
                        for j in 0..n {
                            m.swap(at(k, j), at(p, j));
                        }
                        negate = !negate;
                    }
                    None => return T::zero(),
                }
            }
            for i in k + 1..n {
                for j in k + 1..n {
                    let value = m[at(i, j)] * m[at(k, k)] - m[at(i, k)] * m[at(k, j)];
                    m[at(i, j)] = value / prev;
                }
            }
            prev = m[at(k, k)];
        }

        let det = m[at(n - 1, n - 1)];
        if negate { T::zero() - det } else { det }
    }

    /// Inverse, by Gauss–Jordan elimination with partial pivoting on
    /// [`Coefficient::magnitude`] (so complex and dual elements, which have no
    /// ordering, still pivot sensibly). Returns [`Error::Singular`] when the
    /// matrix has no inverse, and [`Error::Shape`] when it is not square.
    /// Coefficient domains with truncating division, such as primitive integers
    /// and integer-based complex or dual numbers, return
    /// [`Error::InvalidArgument`] because Gauss–Jordan requires fractions.
    pub fn inverse(&self) -> Result<Self, Error> {
        if !self.is_square() {
            return Err(Error::shape(format!(
                "matrix is {}×{}, so it has no inverse",
                self.rows, self.cols
            )));
        }
        if !T::supports_fractional_division() {
            return Err(Error::InvalidArgument(
                "matrix inversion requires coefficients with fractional division".to_string(),
            ));
        }
        let n = self.rows;
        let at = |i: usize, j: usize| i * n + j;
        let mut a = self.data.clone();
        let mut inv = Self::identity(n).data;

        for col in 0..n {
            let pivot = (col..n)
                .max_by(|&x, &y| {
                    a[at(x, col)]
                        .magnitude()
                        .partial_cmp(&a[at(y, col)].magnitude())
                        .unwrap_or(Ordering::Equal)
                })
                .expect("col < n, so the range is non-empty");
            if a[at(pivot, col)].is_zero() {
                return Err(Error::Singular);
            }
            if pivot != col {
                for j in 0..n {
                    a.swap(at(col, j), at(pivot, j));
                    inv.swap(at(col, j), at(pivot, j));
                }
            }

            let scale = a[at(col, col)];
            for j in 0..n {
                a[at(col, j)] = a[at(col, j)] / scale;
                inv[at(col, j)] = inv[at(col, j)] / scale;
            }
            for r in 0..n {
                if r == col {
                    continue;
                }
                let factor = a[at(r, col)];
                if factor.is_zero() {
                    continue;
                }
                for j in 0..n {
                    a[at(r, j)] = a[at(r, j)] - factor * a[at(col, j)];
                    inv[at(r, j)] = inv[at(r, j)] - factor * inv[at(col, j)];
                }
            }
        }
        Ok(Matrix::from_flat(n, n, inv))
    }
}

impl<T: Coefficient + PartialOrd> Matrix<T, Host> {
    /// Confine every element to `[low, high]`; see
    /// [`Vector::clamp`](Vector::clamp).
    ///
    /// # Panics
    ///
    /// If `low > high`.
    #[track_caller]
    pub fn clamp(&self, low: T, high: T) -> Self {
        assert_ordered_bounds(&low, &high);
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let mut out = vec![T::zero(); self.data.len()];
            if simd_dispatch::clamp(&self.data, low, high, &mut out) {
                return Matrix::from_flat(self.rows, self.cols, out);
            }
        }
        self.map(|&value| ordered_min(ordered_max(value, low), high))
    }
}

impl Matrix<f32, Host> {
    /// Elementwise comparison with another matrix.
    ///
    /// # Panics
    ///
    /// If the two shapes differ.
    #[track_caller]
    pub fn compare(&self, other: &Self, op: Compare) -> Self {
        assert_same_shape(self.shape(), other.shape(), "compare");
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let mut out = vec![0.0; self.data.len()];
            if simd_dispatch::compare(&self.data, &other.data, op, &mut out) {
                return Matrix::from_flat(self.rows, self.cols, out);
            }
        }
        let values = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(&left, &right)| op.value(left, right))
            .collect::<Vec<_>>();
        Matrix::from_flat(self.rows, self.cols, values)
    }

    /// Elementwise comparison against a scalar.
    pub fn compare_scalar(&self, scalar: f32, op: Compare, scalar_left: bool) -> Self {
        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let mut out = vec![0.0; self.data.len()];
            if simd_dispatch::compare_scalar(&self.data, scalar, op, scalar_left, &mut out) {
                return Matrix::from_flat(self.rows, self.cols, out);
            }
        }
        self.map(|&value| {
            if scalar_left {
                op.value(scalar, value)
            } else {
                op.value(value, scalar)
            }
        })
    }
}

// ---- elementwise operators --------------------------------------------------

impl<T: Coefficient> Vector<T, Host> {
    #[track_caller]
    fn zip_with(&self, rhs: &Self, op: BinaryOp, f: impl Fn(T, T) -> T) -> Self {
        assert_same_len(self.len, rhs.len, op.name());

        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let mut out = vec![T::zero(); self.len];
            if simd_dispatch::elementwise(&self.data, &rhs.data, op, &mut out) {
                return Vector::new(out);
            }
        }

        Vector::new(
            self.data
                .iter()
                .zip(&rhs.data)
                .map(|(&a, &b)| f(a, b))
                .collect::<Vec<_>>(),
        )
    }
}

impl<T: Coefficient> Matrix<T, Host> {
    #[track_caller]
    fn zip_with(&self, rhs: &Self, op: BinaryOp, f: impl Fn(T, T) -> T) -> Self {
        assert_same_shape(self.shape(), rhs.shape(), op.name());

        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let mut out = vec![T::zero(); self.data.len()];
            if simd_dispatch::elementwise(&self.data, &rhs.data, op, &mut out) {
                return Matrix::from_flat(self.rows, self.cols, out);
            }
        }

        Matrix::from_flat(
            self.rows,
            self.cols,
            self.data
                .iter()
                .zip(&rhs.data)
                .map(|(&a, &b)| f(a, b))
                .collect::<Vec<_>>(),
        )
    }
}

/// One elementwise operator, for owned and borrowed operands.
///
/// Host tensors own a heap allocation, so they are not `Copy`; the reference
/// forms are what most code wants, since `&a + &b` leaves both usable.
macro_rules! elementwise {
    ($Type:ident, $Trait:ident, $method:ident, $op:expr, $apply:tt) => {
        impl<T: Coefficient> $Trait for $Type<T, Host> {
            type Output = $Type<T, Host>;
            #[track_caller]
            fn $method(self, rhs: Self) -> Self::Output {
                self.zip_with(&rhs, $op, |a, b| a $apply b)
            }
        }

        impl<T: Coefficient> $Trait<&$Type<T, Host>> for &$Type<T, Host> {
            type Output = $Type<T, Host>;
            #[track_caller]
            fn $method(self, rhs: &$Type<T, Host>) -> Self::Output {
                self.zip_with(rhs, $op, |a, b| a $apply b)
            }
        }
    };
}

elementwise!(Vector, Add, add, BinaryOp::Add, +);
elementwise!(Vector, Sub, sub, BinaryOp::Sub, -);
elementwise!(Vector, Mul, mul, BinaryOp::Mul, *);
elementwise!(Vector, Div, div, BinaryOp::Div, /);
elementwise!(Vector, Rem, rem, BinaryOp::Rem, %);
elementwise!(Matrix, Add, add, BinaryOp::Add, +);
elementwise!(Matrix, Sub, sub, BinaryOp::Sub, -);
elementwise!(Matrix, Mul, mul, BinaryOp::Mul, *);
elementwise!(Matrix, Div, div, BinaryOp::Div, /);
elementwise!(Matrix, Rem, rem, BinaryOp::Rem, %);

impl<T: Coefficient + Neg<Output = T>> Neg for Vector<T, Host> {
    type Output = Vector<T, Host>;
    fn neg(self) -> Self {
        self.map(|&x| -x)
    }
}

impl<T: Coefficient + Neg<Output = T>> Neg for &Vector<T, Host> {
    type Output = Vector<T, Host>;
    fn neg(self) -> Self::Output {
        self.map(|&x| -x)
    }
}

impl<T: Coefficient + Neg<Output = T>> Neg for Matrix<T, Host> {
    type Output = Matrix<T, Host>;
    fn neg(self) -> Self {
        self.map(|&x| -x)
    }
}

impl<T: Coefficient + Neg<Output = T>> Neg for &Matrix<T, Host> {
    type Output = Matrix<T, Host>;
    fn neg(self) -> Self::Output {
        self.map(|&x| -x)
    }
}

// ---- display ----------------------------------------------------------------

impl<T: Display> Display for Vector<T, Host> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for x in &self.data {
            write!(f, " {x}")?;
        }
        write!(f, " ]")
    }
}

impl<T: Display> Display for Matrix<T, Host> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (r, row) in self.row_iter().enumerate() {
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
