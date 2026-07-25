//! Tensor operations for the [`Metal`] backend.
//!
//! Everything here keeps its operands *and* its result in GPU-shared memory:
//! [`matmul`](Matrix::matmul), [`matvec`](Matrix::matvec),
//! [`vecmat`](Vector::vecmat), the elementwise operators and scalar
//! broadcasting all encode straight over the existing allocations, so a chain of
//! operations costs one upload at the start and one download at the end rather
//! than a pair per call.
//!
//! [`Vector::dot`] deliberately is not a GPU dispatch. It reduces on the CPU
//! over the shared allocations — still copy-free — because a `1×N·N×1` matmul
//! would put the entire reduction on one GPU thread.
//!
//! When a dispatch cannot run at all — no Metal device, or an operation the
//! shaders do not implement, like `%` — the operands move to the [`Host`]
//! backend, its kernels produce the result, and the result moves back. That
//! keeps the answers identical everywhere; it costs the copies this backend
//! exists to avoid, but only on a machine that could not have avoided them.
//!
//! Operations with no GPU kernel at all — [`Matrix::determinant`],
//! [`Matrix::inverse`], the Fourier transforms, and every element type other
//! than `f32` — are not implemented for this backend. Reach them through
//! [`to_backend::<Host>()`](Matrix::to_backend).
//!
//! [`Matrix::determinant`]: super::Matrix::determinant
//! [`Matrix::inverse`]: super::Matrix::inverse

use std::fmt::{self, Display};
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

use super::{Analytic, Host, Kernels, Matrix, Metal, Vector};

impl<const N: usize> Vector<f32, N, Metal> {
    /// Whether the elements really are in GPU-shared memory.
    ///
    /// `false` means the process has no Metal device, so this vector fell back
    /// to CPU storage and CPU kernels. Results are unaffected.
    pub fn is_device_resident(&self) -> bool {
        self.data.is_device_resident()
    }

    pub const fn len(&self) -> usize {
        N
    }

    pub const fn is_empty(&self) -> bool {
        N == 0
    }

    /// Dot product with a vector of the same length.
    ///
    /// The reduction runs on the CPU, reading both shared allocations in place —
    /// shared storage is ordinary cached memory from the CPU's side, so this
    /// still copies nothing. The GPU alternative available here, a `1×N` by
    /// `N×1` matmul, would run the whole sum on a single thread.
    pub fn dot(&self, other: &Self) -> f32 {
        reduce_dot(self.as_slice(), other.as_slice())
    }

    /// Row vector times matrix: `(1×N)·(N×C) = (1×C)`, on the GPU.
    pub fn vecmat<const C: usize>(&self, m: &Matrix<f32, N, C, Metal>) -> Vector<f32, C, Metal> {
        match self.data.matmul(&m.data, 1, N, C) {
            Some(data) => Vector { data },
            None => self
                .to_backend::<Host>()
                .vecmat(&m.to_backend::<Host>())
                .to_backend(),
        }
    }

    /// Apply an analytic function elementwise, on the GPU.
    ///
    /// The host backend reaches these through `math!` or
    /// [`map`](Vector::map) with the traits in [`crate::numbers`]; those go
    /// through `Float`, which the 32-bit shaders cannot, so resident tensors get
    /// this instead.
    pub fn analytic(&self, f: Analytic) -> Self {
        match self.data.unary(f.code()) {
            Some(data) => Vector { data },
            None => Host::vector_unary(&self.to_backend::<Host>(), f).to_backend(),
        }
    }

    /// Multiply every element by `scalar`, on the GPU.
    pub fn scale(&self, scalar: f32) -> Self {
        self.broadcast_right(scalar, 2)
    }

    /// Implementation hook used by `math!` for tensor/scalar broadcasting.
    #[doc(hidden)]
    pub fn broadcast_right(&self, scalar: f32, op: u32) -> Self {
        match self.data.broadcast(scalar, op, false) {
            Some(data) => Vector { data },
            None => self
                .to_backend::<Host>()
                .broadcast_right(scalar, op)
                .to_backend(),
        }
    }

    /// Implementation hook used by `math!` for scalar/tensor broadcasting.
    #[doc(hidden)]
    pub fn broadcast_left(&self, scalar: f32, op: u32) -> Self {
        match self.data.broadcast(scalar, op, true) {
            Some(data) => Vector { data },
            None => self
                .to_backend::<Host>()
                .broadcast_left(scalar, op)
                .to_backend(),
        }
    }
}

impl<const R: usize, const C: usize> Matrix<f32, R, C, Metal> {
    /// Whether the elements really are in GPU-shared memory; see
    /// [`Vector::is_device_resident`].
    pub fn is_device_resident(&self) -> bool {
        self.data.is_device_resident()
    }

    pub const fn shape(&self) -> (usize, usize) {
        (R, C)
    }

    /// Matrix product `(R×C)·(C×C2) = (R×C2)`, on the GPU, with both operands
    /// and the result staying in shared memory.
    pub fn matmul<const C2: usize>(
        &self,
        other: &Matrix<f32, C, C2, Metal>,
    ) -> Matrix<f32, R, C2, Metal> {
        match self.data.matmul(&other.data, R, C, C2) {
            Some(data) => Matrix { data },
            None => self
                .to_backend::<Host>()
                .matmul(&other.to_backend::<Host>())
                .to_backend(),
        }
    }

    /// Matrix times column vector: `(R×C)·(C×1) = (R×1)`, on the GPU.
    pub fn matvec(&self, v: &Vector<f32, C, Metal>) -> Vector<f32, R, Metal> {
        match self.data.matmul(&v.data, R, C, 1) {
            Some(data) => Vector { data },
            None => self
                .to_backend::<Host>()
                .matvec(&v.to_backend::<Host>())
                .to_backend(),
        }
    }

    /// Transpose: an `R×C` matrix becomes `C×R`.
    pub fn transpose(&self) -> Matrix<f32, C, R, Metal> {
        match self.data.transpose(R, C) {
            Some(data) => Matrix { data },
            None => self.to_backend::<Host>().transpose().to_backend(),
        }
    }

    /// Apply an analytic function elementwise, on the GPU; see
    /// [`Vector::analytic`].
    pub fn analytic(&self, f: Analytic) -> Self {
        match self.data.unary(f.code()) {
            Some(data) => Matrix { data },
            None => Host::matrix_unary(&self.to_backend::<Host>(), f).to_backend(),
        }
    }

    /// Multiply every element by `scalar`, on the GPU.
    pub fn scale(&self, scalar: f32) -> Self {
        self.broadcast_right(scalar, 2)
    }

    /// Implementation hook used by `math!` for tensor/scalar broadcasting.
    #[doc(hidden)]
    pub fn broadcast_right(&self, scalar: f32, op: u32) -> Self {
        match self.data.broadcast(scalar, op, false) {
            Some(data) => Matrix { data },
            None => self
                .to_backend::<Host>()
                .broadcast_right(scalar, op)
                .to_backend(),
        }
    }

    /// Implementation hook used by `math!` for scalar/tensor broadcasting.
    #[doc(hidden)]
    pub fn broadcast_left(&self, scalar: f32, op: u32) -> Self {
        match self.data.broadcast(scalar, op, true) {
            Some(data) => Matrix { data },
            None => self
                .to_backend::<Host>()
                .broadcast_left(scalar, op)
                .to_backend(),
        }
    }
}

/// Sum of products over two CPU-readable slices, vectorized where possible.
fn reduce_dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        crate::simd::f32k::dot(a, b)
    }
    #[cfg(not(all(feature = "simd", target_arch = "aarch64")))]
    {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }
}

/// Elementwise `op` over two resident vectors, in shared memory.
pub(super) fn vector_elementwise<const N: usize>(
    a: &Vector<f32, N, Metal>,
    b: &Vector<f32, N, Metal>,
    op: u32,
) -> Vector<f32, N, Metal> {
    match a.data.elementwise(&b.data, op) {
        Some(data) => Vector { data },
        None => Host::vector_elementwise(&a.to_backend::<Host>(), &b.to_backend::<Host>(), op)
            .to_backend(),
    }
}

/// Elementwise `op` over two resident matrices, in shared memory.
pub(super) fn matrix_elementwise<const R: usize, const C: usize>(
    a: &Matrix<f32, R, C, Metal>,
    b: &Matrix<f32, R, C, Metal>,
    op: u32,
) -> Matrix<f32, R, C, Metal> {
    match a.data.elementwise(&b.data, op) {
        Some(data) => Matrix { data },
        None => Host::matrix_elementwise(&a.to_backend::<Host>(), &b.to_backend::<Host>(), op)
            .to_backend(),
    }
}

/// Implements one operator for a resident tensor, by value and by reference.
///
/// `Metal`-backed tensors are not `Copy` — a shared allocation is not a
/// fixed-size array — so the by-reference forms are what most code wants:
/// `&a * &b` leaves both operands usable.
macro_rules! resident_operator {
    ($Type:ident < $($dim:ident),+ >, $Trait:ident, $method:ident, $op:expr, $apply:ident) => {
        impl<$(const $dim: usize),+> $Trait for $Type<f32, $($dim),+, Metal> {
            type Output = Self;
            fn $method(self, rhs: Self) -> Self::Output {
                $apply(&self, &rhs, $op)
            }
        }

        impl<$(const $dim: usize),+> $Trait<&$Type<f32, $($dim),+, Metal>>
            for &$Type<f32, $($dim),+, Metal>
        {
            type Output = $Type<f32, $($dim),+, Metal>;
            fn $method(self, rhs: &$Type<f32, $($dim),+, Metal>) -> Self::Output {
                $apply(self, rhs, $op)
            }
        }
    };
}

resident_operator!(Vector<N>, Add, add, 0, vector_elementwise);
resident_operator!(Vector<N>, Sub, sub, 1, vector_elementwise);
resident_operator!(Vector<N>, Mul, mul, 2, vector_elementwise);
resident_operator!(Vector<N>, Div, div, 3, vector_elementwise);
resident_operator!(Vector<N>, Rem, rem, 4, vector_elementwise);
resident_operator!(Matrix<R, C>, Add, add, 0, matrix_elementwise);
resident_operator!(Matrix<R, C>, Sub, sub, 1, matrix_elementwise);
resident_operator!(Matrix<R, C>, Mul, mul, 2, matrix_elementwise);
resident_operator!(Matrix<R, C>, Div, div, 3, matrix_elementwise);
resident_operator!(Matrix<R, C>, Rem, rem, 4, matrix_elementwise);

impl<const N: usize> Neg for Vector<f32, N, Metal> {
    type Output = Self;
    fn neg(self) -> Self {
        self.broadcast_right(-1.0, 2)
    }
}

impl<const N: usize> Neg for &Vector<f32, N, Metal> {
    type Output = Vector<f32, N, Metal>;
    fn neg(self) -> Self::Output {
        self.broadcast_right(-1.0, 2)
    }
}

impl<const R: usize, const C: usize> Neg for Matrix<f32, R, C, Metal> {
    type Output = Self;
    fn neg(self) -> Self {
        self.broadcast_right(-1.0, 2)
    }
}

impl<const R: usize, const C: usize> Neg for &Matrix<f32, R, C, Metal> {
    type Output = Matrix<f32, R, C, Metal>;
    fn neg(self) -> Self::Output {
        self.broadcast_right(-1.0, 2)
    }
}

// The same formatting as the host-backed tensors, so a backend switch does not
// change what a printed tensor looks like.
impl<const N: usize> Display for Vector<f32, N, Metal> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for x in self.as_slice() {
            write!(f, " {x}")?;
        }
        write!(f, " ]")
    }
}

impl<const R: usize, const C: usize> Display for Matrix<f32, R, C, Metal> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let values = self.as_slice();
        for r in 0..R {
            if r > 0 {
                writeln!(f)?;
            }
            write!(f, "[")?;
            for x in &values[r * C..r * C + C] {
                write!(f, " {x}")?;
            }
            write!(f, " ]")?;
        }
        Ok(())
    }
}
