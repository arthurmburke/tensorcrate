//! The `f32` tensor algebra, named per backend.
//!
//! [`Vector`] and [`Matrix`] implement their products and elementwise
//! operations as inherent methods, once for each backend, and nothing ties those
//! two sets together — `Matrix<f32, R, C, Host>::matmul` and
//! `Matrix<f32, R, C, Metal>::matmul` are unrelated functions that happen to
//! share a name. [`Kernels`] is that missing link: one trait naming every
//! operation both backends provide, so code written against it compiles for
//! either.
//!
//! Automatic differentiation is the reason it exists. The forward-mode layer in
//! [`dual`](super::dual) is written once against `Kernels` and instantiates on
//! both backends, which also means the `Host` instantiation is an exact
//! correctness oracle for the `Metal` one — same code, different memory.
//!
//! The trait is sealed, since [`Backend`] is.

use super::{Backend, Host, Matrix, Vector};

/// An elementwise binary operation.
///
/// The representation is part of the Metal shader ABI. Keep existing
/// discriminants stable and only append new operations.
#[repr(u16)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    Rem = 4,
}

impl From<BinaryOp> for u16 {
    fn from(op: BinaryOp) -> Self {
        op as u16
    }
}

impl TryFrom<u16> for BinaryOp {
    type Error = u16;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Add),
            1 => Ok(Self::Sub),
            2 => Ok(Self::Mul),
            3 => Ok(Self::Div),
            4 => Ok(Self::Rem),
            value => Err(value),
        }
    }
}

/// An elementwise comparison.
///
/// Like [`BinaryOp`], the representation is part of the Metal shader ABI: keep
/// existing discriminants stable and only append.
///
/// `Min` and `Max` are not differentiable where the operands tie, so a
/// convention is needed. This one splits the subgradient evenly, which is what
/// [`MaxShare`](Compare::MaxShare) computes — and it is the choice that keeps
/// `max(a, b)` and `max(b, a)` giving mirror-image gradients. Two consequences
/// worth knowing: `|x|` differentiates to `sign(x)` with `sign(0) = 0`, and
/// `relu` has slope `½` exactly at the kink rather than the `0` some frameworks
/// pick.
#[repr(u16)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Compare {
    /// The smaller of the two operands.
    Min = 0,
    /// The larger of the two operands.
    Max = 1,
    /// `∂max(a, b)/∂a`: one where `a` is larger, zero where it is smaller, and a
    /// half where they tie. The rule for `Min` is its complement, `1 − share`.
    MaxShare = 2,
}

impl Compare {
    /// Every comparison, in discriminant order.
    pub const ALL: [Compare; 3] = [Compare::Min, Compare::Max, Compare::MaxShare];

    /// Apply the comparison to a pair of values — the CPU counterpart of the
    /// `compare` shader, and the definition the GPU is tested against.
    pub fn value(self, a: f32, b: f32) -> f32 {
        match self {
            Compare::Min => a.min(b),
            Compare::Max => a.max(b),
            Compare::MaxShare => match a.partial_cmp(&b) {
                Some(std::cmp::Ordering::Greater) => 1.0,
                Some(std::cmp::Ordering::Less) => 0.0,
                _ => 0.5,
            },
        }
    }
}

impl From<Compare> for u16 {
    fn from(op: Compare) -> Self {
        op as u16
    }
}

impl TryFrom<u16> for Compare {
    type Error = u16;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        Self::ALL.get(value as usize).copied().ok_or(value)
    }
}

/// The analytic functions, paired with their derivatives.
///
/// This is the operation enum for the GPU `unary`/`unary_dual` kernels and the
/// dispatch table for the CPU path, so both sides stay in step. The variants
/// mirror the functions `math!` accepts, and each derivative is written the same
/// way as the matching [`Dual`](crate::numbers::Dual) implementation.
#[repr(u16)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Analytic {
    Sin = 0,
    Cos = 1,
    Tan = 2,
    Sec = 3,
    Csc = 4,
    Arcsin = 5,
    Arccos = 6,
    Arctan = 7,
    Exp = 8,
    Ln = 9,
    Sinh = 10,
    Cosh = 11,
    Tanh = 12,
    Sqrt = 13,
}

impl Analytic {
    /// Every function, in discriminant order.
    pub const ALL: [Analytic; 14] = [
        Analytic::Sin,
        Analytic::Cos,
        Analytic::Tan,
        Analytic::Sec,
        Analytic::Csc,
        Analytic::Arcsin,
        Analytic::Arccos,
        Analytic::Arctan,
        Analytic::Exp,
        Analytic::Ln,
        Analytic::Sinh,
        Analytic::Cosh,
        Analytic::Tanh,
        Analytic::Sqrt,
    ];

    /// `f(x)`.
    pub fn value(self, x: f32) -> f32 {
        match self {
            Analytic::Sin => x.sin(),
            Analytic::Cos => x.cos(),
            Analytic::Tan => x.tan(),
            Analytic::Sec => x.cos().recip(),
            Analytic::Csc => x.sin().recip(),
            Analytic::Arcsin => x.asin(),
            Analytic::Arccos => x.acos(),
            Analytic::Arctan => x.atan(),
            Analytic::Exp => x.exp(),
            Analytic::Ln => x.ln(),
            Analytic::Sinh => x.sinh(),
            Analytic::Cosh => x.cosh(),
            Analytic::Tanh => x.tanh(),
            Analytic::Sqrt => x.sqrt(),
        }
    }

    /// `f'(x)`.
    pub fn derivative(self, x: f32) -> f32 {
        match self {
            Analytic::Sin => x.cos(),
            Analytic::Cos => -x.sin(),
            Analytic::Tan => {
                let cos = x.cos();
                (cos * cos).recip()
            }
            Analytic::Sec => {
                let cos = x.cos();
                x.sin() / (cos * cos)
            }
            Analytic::Csc => {
                let sin = x.sin();
                -x.cos() / (sin * sin)
            }
            Analytic::Arcsin => (1.0 - x * x).sqrt().recip(),
            Analytic::Arccos => -(1.0 - x * x).sqrt().recip(),
            Analytic::Arctan => (1.0 + x * x).recip(),
            Analytic::Exp => x.exp(),
            Analytic::Ln => x.recip(),
            Analytic::Sinh => x.cosh(),
            Analytic::Cosh => x.sinh(),
            Analytic::Tanh => {
                let tanh = x.tanh();
                1.0 - tanh * tanh
            }
            Analytic::Sqrt => (2.0 * x.sqrt()).recip(),
        }
    }
}

impl From<Analytic> for u16 {
    fn from(op: Analytic) -> Self {
        op as u16
    }
}

impl TryFrom<u16> for Analytic {
    type Error = u16;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        Self::ALL.get(value as usize).copied().ok_or(value)
    }
}

#[cfg(test)]
mod operation_tests {
    use std::mem::size_of;

    use super::{Analytic, BinaryOp};

    #[test]
    fn operation_enums_have_a_stable_u16_representation() {
        assert_eq!(size_of::<BinaryOp>(), size_of::<u16>());
        assert_eq!(size_of::<Analytic>(), size_of::<u16>());

        for op in [
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Div,
            BinaryOp::Rem,
        ] {
            assert_eq!(BinaryOp::try_from(u16::from(op)), Ok(op));
        }
        for op in Analytic::ALL {
            assert_eq!(Analytic::try_from(u16::from(op)), Ok(op));
        }
        assert_eq!(BinaryOp::try_from(u16::MAX), Err(u16::MAX));
        assert_eq!(Analytic::try_from(u16::MAX), Err(u16::MAX));
    }
}

/// The `f32` tensor operations a [`Backend`] provides.
///
pub trait Kernels: Backend {
    // ---- vectors ----

    fn vector_elementwise<const N: usize>(
        a: &Vector<f32, N, Self>,
        b: &Vector<f32, N, Self>,
        op: BinaryOp,
    ) -> Vector<f32, N, Self>;

    fn vector_broadcast<const N: usize>(
        a: &Vector<f32, N, Self>,
        scalar: f32,
        op: BinaryOp,
        scalar_left: bool,
    ) -> Vector<f32, N, Self>;

    /// Elementwise comparison of two vectors.
    fn vector_compare<const N: usize>(
        a: &Vector<f32, N, Self>,
        b: &Vector<f32, N, Self>,
        op: Compare,
    ) -> Vector<f32, N, Self>;

    /// Elementwise comparison against a scalar.
    fn vector_compare_scalar<const N: usize>(
        a: &Vector<f32, N, Self>,
        scalar: f32,
        op: Compare,
        scalar_left: bool,
    ) -> Vector<f32, N, Self>;

    /// `f(a)`, elementwise.
    fn vector_unary<const N: usize>(a: &Vector<f32, N, Self>, f: Analytic) -> Vector<f32, N, Self>;

    /// `(f(value), f'(value) ⊙ tangent)` — one forward-mode step.
    fn vector_unary_dual<const N: usize>(
        value: &Vector<f32, N, Self>,
        tangent: &Vector<f32, N, Self>,
        f: Analytic,
    ) -> (Vector<f32, N, Self>, Vector<f32, N, Self>);

    fn dot<const N: usize>(a: &Vector<f32, N, Self>, b: &Vector<f32, N, Self>) -> f32;

    fn vecmat<const N: usize, const C: usize>(
        v: &Vector<f32, N, Self>,
        m: &Matrix<f32, N, C, Self>,
    ) -> Vector<f32, C, Self>;

    fn matvec<const R: usize, const C: usize>(
        m: &Matrix<f32, R, C, Self>,
        v: &Vector<f32, C, Self>,
    ) -> Vector<f32, R, Self>;

    /// `addend + m·v`, using `addend` as the accumulator when possible.
    fn matvec_add<const R: usize, const C: usize>(
        m: &Matrix<f32, R, C, Self>,
        v: &Vector<f32, C, Self>,
        addend: Vector<f32, R, Self>,
    ) -> Vector<f32, R, Self>;

    // ---- matrices ----

    fn matrix_elementwise<const R: usize, const C: usize>(
        a: &Matrix<f32, R, C, Self>,
        b: &Matrix<f32, R, C, Self>,
        op: BinaryOp,
    ) -> Matrix<f32, R, C, Self>;

    fn matrix_broadcast<const R: usize, const C: usize>(
        a: &Matrix<f32, R, C, Self>,
        scalar: f32,
        op: BinaryOp,
        scalar_left: bool,
    ) -> Matrix<f32, R, C, Self>;

    /// Elementwise comparison of two matrices.
    fn matrix_compare<const R: usize, const C: usize>(
        a: &Matrix<f32, R, C, Self>,
        b: &Matrix<f32, R, C, Self>,
        op: Compare,
    ) -> Matrix<f32, R, C, Self>;

    /// Elementwise comparison against a scalar.
    fn matrix_compare_scalar<const R: usize, const C: usize>(
        a: &Matrix<f32, R, C, Self>,
        scalar: f32,
        op: Compare,
        scalar_left: bool,
    ) -> Matrix<f32, R, C, Self>;

    /// `f(a)`, elementwise.
    fn matrix_unary<const R: usize, const C: usize>(
        a: &Matrix<f32, R, C, Self>,
        f: Analytic,
    ) -> Matrix<f32, R, C, Self>;

    /// `(f(value), f'(value) ⊙ tangent)` — one forward-mode step.
    fn matrix_unary_dual<const R: usize, const C: usize>(
        value: &Matrix<f32, R, C, Self>,
        tangent: &Matrix<f32, R, C, Self>,
        f: Analytic,
    ) -> (Matrix<f32, R, C, Self>, Matrix<f32, R, C, Self>);

    fn matmul<const R: usize, const K: usize, const C: usize>(
        a: &Matrix<f32, R, K, Self>,
        b: &Matrix<f32, K, C, Self>,
    ) -> Matrix<f32, R, C, Self>;

    /// `addend + a·b`.
    ///
    /// Backends that can accumulate inside the product kernel do so, which is
    /// what makes a dual matmul two dispatches instead of three. `addend` is
    /// consumed because it may become the accumulator.
    fn matmul_add<const R: usize, const K: usize, const C: usize>(
        a: &Matrix<f32, R, K, Self>,
        b: &Matrix<f32, K, C, Self>,
        addend: Matrix<f32, R, C, Self>,
    ) -> Matrix<f32, R, C, Self>;

    fn transpose<const R: usize, const C: usize>(
        m: &Matrix<f32, R, C, Self>,
    ) -> Matrix<f32, C, R, Self>;

    /// Valid cross-correlation: output `(i, j)` is the `KR × KC` window of
    /// `input` at `(i, j)` dotted with `window`.
    ///
    /// `flip` reverses the window, which is the difference between correlation
    /// (the machine-learning convention) and convolution (the signal-processing
    /// one) — and is also what the input-side gradient of either needs.
    fn correlate<const R: usize, const C: usize, const KR: usize, const KC: usize>(
        input: &Matrix<f32, R, C, Self>,
        window: &Matrix<f32, KR, KC, Self>,
        flip: bool,
    ) -> Matrix<f32, { R - KR + 1 }, { C - KC + 1 }, Self>;

    /// `∂L/∂window` for a valid correlation: the input windowed by the output
    /// adjoint, which comes out exactly the window's shape.
    ///
    /// The result shape is *named* rather than computed, which is the point.
    /// Writing it as `correlate::<R, C, {R−KR+1}, {C−KC+1}>` would leave the
    /// compiler needing to prove `R − (R−KR+1) + 1 == KR`, and const-expression
    /// equality is beyond what `generic_const_exprs` can do.
    fn correlate_window_gradient<const R: usize, const C: usize, const KR: usize, const KC: usize>(
        input: &Matrix<f32, R, C, Self>,
        adjoint: &Matrix<f32, { R - KR + 1 }, { C - KC + 1 }, Self>,
    ) -> Matrix<f32, KR, KC, Self>;

    /// `∂L/∂input` for a valid correlation: the full correlation of the adjoint
    /// with the window, which is the padded one. `forward_flip` says which
    /// convention the forward pass used; the gradient applies the window the
    /// other way round.
    fn correlate_input_gradient<const R: usize, const C: usize, const KR: usize, const KC: usize>(
        adjoint: &Matrix<f32, { R - KR + 1 }, { C - KC + 1 }, Self>,
        window: &Matrix<f32, KR, KC, Self>,
        forward_flip: bool,
    ) -> Matrix<f32, R, C, Self>;

    /// Surround a matrix with zeros.
    fn pad<const R: usize, const C: usize, const PR: usize, const PC: usize>(
        input: &Matrix<f32, R, C, Self>,
    ) -> Matrix<f32, { R + 2 * PR }, { C + 2 * PC }, Self>;

    /// Reverse both axes.
    fn flip<const R: usize, const C: usize>(
        input: &Matrix<f32, R, C, Self>,
    ) -> Matrix<f32, R, C, Self>;
}

/// Every operation here already exists as an inherent method or an operator on
/// the host-backed tensors; this is pure forwarding.
impl Kernels for Host {
    fn vector_elementwise<const N: usize>(
        a: &Vector<f32, N, Self>,
        b: &Vector<f32, N, Self>,
        op: BinaryOp,
    ) -> Vector<f32, N, Self> {
        match op {
            BinaryOp::Add => *a + *b,
            BinaryOp::Sub => *a - *b,
            BinaryOp::Mul => *a * *b,
            BinaryOp::Div => *a / *b,
            BinaryOp::Rem => *a % *b,
        }
    }

    fn vector_broadcast<const N: usize>(
        a: &Vector<f32, N, Self>,
        scalar: f32,
        op: BinaryOp,
        scalar_left: bool,
    ) -> Vector<f32, N, Self> {
        if scalar_left {
            a.broadcast_left(scalar, op)
        } else {
            a.broadcast_right(scalar, op)
        }
    }

    fn vector_compare<const N: usize>(
        a: &Vector<f32, N, Self>,
        b: &Vector<f32, N, Self>,
        op: Compare,
    ) -> Vector<f32, N, Self> {
        let (left, right) = (a.data(), b.data());
        Vector::new(std::array::from_fn(|i| op.value(left[i], right[i])))
    }

    fn vector_compare_scalar<const N: usize>(
        a: &Vector<f32, N, Self>,
        scalar: f32,
        op: Compare,
        scalar_left: bool,
    ) -> Vector<f32, N, Self> {
        a.map(|&x| {
            if scalar_left {
                op.value(scalar, x)
            } else {
                op.value(x, scalar)
            }
        })
    }

    fn vector_unary<const N: usize>(a: &Vector<f32, N, Self>, f: Analytic) -> Vector<f32, N, Self> {
        a.map(|&x| f.value(x))
    }

    fn vector_unary_dual<const N: usize>(
        value: &Vector<f32, N, Self>,
        tangent: &Vector<f32, N, Self>,
        f: Analytic,
    ) -> (Vector<f32, N, Self>, Vector<f32, N, Self>) {
        let values = value.data();
        (
            value.map(|&x| f.value(x)),
            Vector::new(std::array::from_fn(|i| {
                f.derivative(values[i]) * tangent.data()[i]
            })),
        )
    }

    fn dot<const N: usize>(a: &Vector<f32, N, Self>, b: &Vector<f32, N, Self>) -> f32 {
        a.dot(b)
    }

    fn vecmat<const N: usize, const C: usize>(
        v: &Vector<f32, N, Self>,
        m: &Matrix<f32, N, C, Self>,
    ) -> Vector<f32, C, Self> {
        v.vecmat(m)
    }

    fn matvec<const R: usize, const C: usize>(
        m: &Matrix<f32, R, C, Self>,
        v: &Vector<f32, C, Self>,
    ) -> Vector<f32, R, Self> {
        m.matvec(v)
    }

    fn matvec_add<const R: usize, const C: usize>(
        m: &Matrix<f32, R, C, Self>,
        v: &Vector<f32, C, Self>,
        addend: Vector<f32, R, Self>,
    ) -> Vector<f32, R, Self> {
        m.matvec_add(v, addend)
    }

    fn matrix_elementwise<const R: usize, const C: usize>(
        a: &Matrix<f32, R, C, Self>,
        b: &Matrix<f32, R, C, Self>,
        op: BinaryOp,
    ) -> Matrix<f32, R, C, Self> {
        match op {
            BinaryOp::Add => *a + *b,
            BinaryOp::Sub => *a - *b,
            BinaryOp::Mul => *a * *b,
            BinaryOp::Div => *a / *b,
            BinaryOp::Rem => *a % *b,
        }
    }

    fn matrix_broadcast<const R: usize, const C: usize>(
        a: &Matrix<f32, R, C, Self>,
        scalar: f32,
        op: BinaryOp,
        scalar_left: bool,
    ) -> Matrix<f32, R, C, Self> {
        if scalar_left {
            a.broadcast_left(scalar, op)
        } else {
            a.broadcast_right(scalar, op)
        }
    }

    fn matrix_compare<const R: usize, const C: usize>(
        a: &Matrix<f32, R, C, Self>,
        b: &Matrix<f32, R, C, Self>,
        op: Compare,
    ) -> Matrix<f32, R, C, Self> {
        let (left, right) = (a.data(), b.data());
        Matrix::from_rows(std::array::from_fn(|row| {
            std::array::from_fn(|col| op.value(left[row][col], right[row][col]))
        }))
    }

    fn matrix_compare_scalar<const R: usize, const C: usize>(
        a: &Matrix<f32, R, C, Self>,
        scalar: f32,
        op: Compare,
        scalar_left: bool,
    ) -> Matrix<f32, R, C, Self> {
        a.map(|&x| {
            if scalar_left {
                op.value(scalar, x)
            } else {
                op.value(x, scalar)
            }
        })
    }

    fn matrix_unary<const R: usize, const C: usize>(
        a: &Matrix<f32, R, C, Self>,
        f: Analytic,
    ) -> Matrix<f32, R, C, Self> {
        a.map(|&x| f.value(x))
    }

    fn matrix_unary_dual<const R: usize, const C: usize>(
        value: &Matrix<f32, R, C, Self>,
        tangent: &Matrix<f32, R, C, Self>,
        f: Analytic,
    ) -> (Matrix<f32, R, C, Self>, Matrix<f32, R, C, Self>) {
        let (values, tangents) = (value.data(), tangent.data());
        (
            value.map(|&x| f.value(x)),
            Matrix::from_rows(std::array::from_fn(|row| {
                std::array::from_fn(|col| f.derivative(values[row][col]) * tangents[row][col])
            })),
        )
    }

    fn matmul<const R: usize, const K: usize, const C: usize>(
        a: &Matrix<f32, R, K, Self>,
        b: &Matrix<f32, K, C, Self>,
    ) -> Matrix<f32, R, C, Self> {
        a.matmul(b)
    }

    fn matmul_add<const R: usize, const K: usize, const C: usize>(
        a: &Matrix<f32, R, K, Self>,
        b: &Matrix<f32, K, C, Self>,
        addend: Matrix<f32, R, C, Self>,
    ) -> Matrix<f32, R, C, Self> {
        a.matmul_add(b, addend)
    }

    fn transpose<const R: usize, const C: usize>(
        m: &Matrix<f32, R, C, Self>,
    ) -> Matrix<f32, C, R, Self> {
        m.transpose()
    }

    fn correlate<const R: usize, const C: usize, const KR: usize, const KC: usize>(
        input: &Matrix<f32, R, C, Self>,
        window: &Matrix<f32, KR, KC, Self>,
        flip: bool,
    ) -> Matrix<f32, { R - KR + 1 }, { C - KC + 1 }, Self> {
        let (values, taps) = (input.data(), window.data());
        Matrix::from_rows(std::array::from_fn(|row| {
            std::array::from_fn(|col| {
                let mut sum = 0.0;
                for window_row in 0..KR {
                    for window_col in 0..KC {
                        let (tap_row, tap_col) = if flip {
                            (KR - 1 - window_row, KC - 1 - window_col)
                        } else {
                            (window_row, window_col)
                        };
                        sum += values[row + window_row][col + window_col] * taps[tap_row][tap_col];
                    }
                }
                sum
            })
        }))
    }

    fn flip<const R: usize, const C: usize>(
        input: &Matrix<f32, R, C, Self>,
    ) -> Matrix<f32, R, C, Self> {
        let values = input.data();
        Matrix::from_rows(std::array::from_fn(|row| {
            std::array::from_fn(|col| values[R - 1 - row][C - 1 - col])
        }))
    }

    fn correlate_window_gradient<
        const R: usize,
        const C: usize,
        const KR: usize,
        const KC: usize,
    >(
        input: &Matrix<f32, R, C, Self>,
        adjoint: &Matrix<f32, { R - KR + 1 }, { C - KC + 1 }, Self>,
    ) -> Matrix<f32, KR, KC, Self> {
        // K̄[a][b] = Σᵢⱼ Ȳ[i][j]·X[i+a][j+b]
        let (values, upstream) = (input.data(), adjoint.data());
        Matrix::from_rows(std::array::from_fn(|tap_row| {
            std::array::from_fn(|tap_col| {
                let mut sum = 0.0;
                for row in 0..R - KR + 1 {
                    for col in 0..C - KC + 1 {
                        sum += upstream[row][col] * values[row + tap_row][col + tap_col];
                    }
                }
                sum
            })
        }))
    }

    fn correlate_input_gradient<
        const R: usize,
        const C: usize,
        const KR: usize,
        const KC: usize,
    >(
        adjoint: &Matrix<f32, { R - KR + 1 }, { C - KC + 1 }, Self>,
        window: &Matrix<f32, KR, KC, Self>,
        forward_flip: bool,
    ) -> Matrix<f32, R, C, Self> {
        // X̄[p][q] = Σᵤᵥ Ȳ[p−u][q−v]·K[u][v], with the taps reversed when the
        // forward pass reversed them. Out-of-range adjoint indices are the zeros
        // a full correlation pads with.
        let (upstream, taps) = (adjoint.data(), window.data());
        let (out_rows, out_cols) = (R - KR + 1, C - KC + 1);
        Matrix::from_rows(std::array::from_fn(|row| {
            std::array::from_fn(|col| {
                let mut sum = 0.0;
                for window_row in 0..KR {
                    for window_col in 0..KC {
                        if row < window_row || col < window_col {
                            continue;
                        }
                        let (source_row, source_col) = (row - window_row, col - window_col);
                        if source_row >= out_rows || source_col >= out_cols {
                            continue;
                        }
                        let (tap_row, tap_col) = if forward_flip {
                            (KR - 1 - window_row, KC - 1 - window_col)
                        } else {
                            (window_row, window_col)
                        };
                        sum += upstream[source_row][source_col] * taps[tap_row][tap_col];
                    }
                }
                sum
            })
        }))
    }

    fn pad<const R: usize, const C: usize, const PR: usize, const PC: usize>(
        input: &Matrix<f32, R, C, Self>,
    ) -> Matrix<f32, { R + 2 * PR }, { C + 2 * PC }, Self> {
        let values = input.data();
        Matrix::from_rows(std::array::from_fn(|row| {
            std::array::from_fn(|col| {
                let inside = row >= PR && row < PR + R && col >= PC && col < PC + C;
                if inside {
                    values[row - PR][col - PC]
                } else {
                    0.0
                }
            })
        }))
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod gpu {
    use super::{Analytic, BinaryOp, Compare, Kernels, Matrix, Vector};
    use crate::tensors::metal_backend::{matrix_elementwise, vector_elementwise};
    use crate::tensors::{Host, Metal};

    /// Forwarding again, but to the resident operations: every one of these
    /// leaves its result in GPU-shared memory.
    impl Kernels for Metal {
        fn vector_elementwise<const N: usize>(
            a: &Vector<f32, N, Self>,
            b: &Vector<f32, N, Self>,
            op: BinaryOp,
        ) -> Vector<f32, N, Self> {
            vector_elementwise(a, b, op)
        }

        fn vector_broadcast<const N: usize>(
            a: &Vector<f32, N, Self>,
            scalar: f32,
            op: BinaryOp,
            scalar_left: bool,
        ) -> Vector<f32, N, Self> {
            if scalar_left {
                a.broadcast_left(scalar, op)
            } else {
                a.broadcast_right(scalar, op)
            }
        }

        fn vector_compare<const N: usize>(
            a: &Vector<f32, N, Self>,
            b: &Vector<f32, N, Self>,
            op: Compare,
        ) -> Vector<f32, N, Self> {
            a.compare(b, op)
        }

        fn vector_compare_scalar<const N: usize>(
            a: &Vector<f32, N, Self>,
            scalar: f32,
            op: Compare,
            scalar_left: bool,
        ) -> Vector<f32, N, Self> {
            a.compare_scalar(scalar, op, scalar_left)
        }

        fn vector_unary<const N: usize>(
            a: &Vector<f32, N, Self>,
            f: Analytic,
        ) -> Vector<f32, N, Self> {
            a.analytic(f)
        }

        fn vector_unary_dual<const N: usize>(
            value: &Vector<f32, N, Self>,
            tangent: &Vector<f32, N, Self>,
            f: Analytic,
        ) -> (Vector<f32, N, Self>, Vector<f32, N, Self>) {
            match value.data.unary_dual(&tangent.data, f) {
                Some((value, tangent)) => (Vector { data: value }, Vector { data: tangent }),
                None => {
                    let (value, tangent) = Host::vector_unary_dual(
                        &value.to_backend::<Host>(),
                        &tangent.to_backend::<Host>(),
                        f,
                    );
                    (value.to_backend(), tangent.to_backend())
                }
            }
        }

        fn dot<const N: usize>(a: &Vector<f32, N, Self>, b: &Vector<f32, N, Self>) -> f32 {
            a.dot(b)
        }

        fn vecmat<const N: usize, const C: usize>(
            v: &Vector<f32, N, Self>,
            m: &Matrix<f32, N, C, Self>,
        ) -> Vector<f32, C, Self> {
            v.vecmat(m)
        }

        fn matvec<const R: usize, const C: usize>(
            m: &Matrix<f32, R, C, Self>,
            v: &Vector<f32, C, Self>,
        ) -> Vector<f32, R, Self> {
            m.matvec(v)
        }

        fn matvec_add<const R: usize, const C: usize>(
            m: &Matrix<f32, R, C, Self>,
            v: &Vector<f32, C, Self>,
            addend: Vector<f32, R, Self>,
        ) -> Vector<f32, R, Self> {
            m.matvec_add(v, addend)
        }

        fn matrix_elementwise<const R: usize, const C: usize>(
            a: &Matrix<f32, R, C, Self>,
            b: &Matrix<f32, R, C, Self>,
            op: BinaryOp,
        ) -> Matrix<f32, R, C, Self> {
            matrix_elementwise(a, b, op)
        }

        fn matrix_broadcast<const R: usize, const C: usize>(
            a: &Matrix<f32, R, C, Self>,
            scalar: f32,
            op: BinaryOp,
            scalar_left: bool,
        ) -> Matrix<f32, R, C, Self> {
            if scalar_left {
                a.broadcast_left(scalar, op)
            } else {
                a.broadcast_right(scalar, op)
            }
        }

        fn matrix_compare<const R: usize, const C: usize>(
            a: &Matrix<f32, R, C, Self>,
            b: &Matrix<f32, R, C, Self>,
            op: Compare,
        ) -> Matrix<f32, R, C, Self> {
            a.compare(b, op)
        }

        fn matrix_compare_scalar<const R: usize, const C: usize>(
            a: &Matrix<f32, R, C, Self>,
            scalar: f32,
            op: Compare,
            scalar_left: bool,
        ) -> Matrix<f32, R, C, Self> {
            a.compare_scalar(scalar, op, scalar_left)
        }

        fn matrix_unary<const R: usize, const C: usize>(
            a: &Matrix<f32, R, C, Self>,
            f: Analytic,
        ) -> Matrix<f32, R, C, Self> {
            a.analytic(f)
        }

        fn matrix_unary_dual<const R: usize, const C: usize>(
            value: &Matrix<f32, R, C, Self>,
            tangent: &Matrix<f32, R, C, Self>,
            f: Analytic,
        ) -> (Matrix<f32, R, C, Self>, Matrix<f32, R, C, Self>) {
            match value.data.unary_dual(&tangent.data, f) {
                Some((value, tangent)) => (Matrix { data: value }, Matrix { data: tangent }),
                None => {
                    let (value, tangent) = Host::matrix_unary_dual(
                        &value.to_backend::<Host>(),
                        &tangent.to_backend::<Host>(),
                        f,
                    );
                    (value.to_backend(), tangent.to_backend())
                }
            }
        }

        fn matmul<const R: usize, const K: usize, const C: usize>(
            a: &Matrix<f32, R, K, Self>,
            b: &Matrix<f32, K, C, Self>,
        ) -> Matrix<f32, R, C, Self> {
            a.matmul(b)
        }

        fn matmul_add<const R: usize, const K: usize, const C: usize>(
            a: &Matrix<f32, R, K, Self>,
            b: &Matrix<f32, K, C, Self>,
            addend: Matrix<f32, R, C, Self>,
        ) -> Matrix<f32, R, C, Self> {
            a.matmul_add(b, addend)
        }

        fn transpose<const R: usize, const C: usize>(
            m: &Matrix<f32, R, C, Self>,
        ) -> Matrix<f32, C, R, Self> {
            m.transpose()
        }

        fn correlate<const R: usize, const C: usize, const KR: usize, const KC: usize>(
            input: &Matrix<f32, R, C, Self>,
            window: &Matrix<f32, KR, KC, Self>,
            flip: bool,
        ) -> Matrix<f32, { R - KR + 1 }, { C - KC + 1 }, Self> {
            match input.data.correlate(&window.data, R, C, KR, KC, flip) {
                Some(data) => Matrix { data },
                None => Host::correlate(
                    &input.to_backend::<Host>(),
                    &window.to_backend::<Host>(),
                    flip,
                )
                .to_backend(),
            }
        }

        fn flip<const R: usize, const C: usize>(
            input: &Matrix<f32, R, C, Self>,
        ) -> Matrix<f32, R, C, Self> {
            match input.data.flip(R, C) {
                Some(data) => Matrix { data },
                None => Host::flip(&input.to_backend::<Host>()).to_backend(),
            }
        }

        fn correlate_window_gradient<
            const R: usize,
            const C: usize,
            const KR: usize,
            const KC: usize,
        >(
            input: &Matrix<f32, R, C, Self>,
            adjoint: &Matrix<f32, { R - KR + 1 }, { C - KC + 1 }, Self>,
        ) -> Matrix<f32, KR, KC, Self> {
            // Correlating the input with the adjoint as the window leaves
            // exactly `KR × KC`, and the storage layer takes those as numbers.
            match input
                .data
                .correlate(&adjoint.data, R, C, R - KR + 1, C - KC + 1, false)
            {
                Some(data) => Matrix { data },
                None => Host::correlate_window_gradient::<R, C, KR, KC>(
                    &input.to_backend::<Host>(),
                    &adjoint.to_backend::<Host>(),
                )
                .to_backend(),
            }
        }

        fn correlate_input_gradient<
            const R: usize,
            const C: usize,
            const KR: usize,
            const KC: usize,
        >(
            adjoint: &Matrix<f32, { R - KR + 1 }, { C - KC + 1 }, Self>,
            window: &Matrix<f32, KR, KC, Self>,
            forward_flip: bool,
        ) -> Matrix<f32, R, C, Self> {
            let padded = adjoint.data.pad(R - KR + 1, C - KC + 1, KR - 1, KC - 1);
            let full = padded.and_then(|padded| {
                padded.correlate(&window.data, R + KR - 1, C + KC - 1, KR, KC, !forward_flip)
            });
            match full {
                Some(data) => Matrix { data },
                None => Host::correlate_input_gradient::<R, C, KR, KC>(
                    &adjoint.to_backend::<Host>(),
                    &window.to_backend::<Host>(),
                    forward_flip,
                )
                .to_backend(),
            }
        }

        fn pad<const R: usize, const C: usize, const PR: usize, const PC: usize>(
            input: &Matrix<f32, R, C, Self>,
        ) -> Matrix<f32, { R + 2 * PR }, { C + 2 * PC }, Self> {
            match input.data.pad(R, C, PR, PC) {
                Some(data) => Matrix { data },
                None => Host::pad::<R, C, PR, PC>(&input.to_backend::<Host>()).to_backend(),
            }
        }
    }
}
