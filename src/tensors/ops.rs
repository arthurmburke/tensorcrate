//! The elementwise operators `+ - * / %` and negation on host tensors.

use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

use super::{BinaryOp, Host, Matrix, Vector};
use crate::numbers::Coefficient;

/// One elementwise operator, for owned and borrowed operands.
///
/// Host tensors own a heap allocation, so they are not `Copy`; the reference
/// forms are what most code wants, since `&a + &b` leaves both usable.
///
/// An owned left operand is consumed: its allocation holds the result, so
/// `a + b` and `a + &b` allocate nothing. Use `&a + &b` to keep `a`.
macro_rules! elementwise {
    ($Type:ident, $Trait:ident, $method:ident, $op:expr, $apply:tt) => {
        impl<T: Coefficient> $Trait for $Type<T, Host> {
            type Output = $Type<T, Host>;
            #[track_caller]
            fn $method(self, rhs: Self) -> Self::Output {
                self.zip_into(&rhs, $op, |a, b| a $apply b)
            }
        }

        impl<T: Coefficient> $Trait<&$Type<T, Host>> for $Type<T, Host> {
            type Output = $Type<T, Host>;
            #[track_caller]
            fn $method(self, rhs: &$Type<T, Host>) -> Self::Output {
                self.zip_into(rhs, $op, |a, b| a $apply b)
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
        self.into_map(|x| -x)
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
        self.into_map(|x| -x)
    }
}

impl<T: Coefficient + Neg<Output = T>> Neg for &Matrix<T, Host> {
    type Output = Matrix<T, Host>;
    fn neg(self) -> Self::Output {
        self.map(|&x| -x)
    }
}
