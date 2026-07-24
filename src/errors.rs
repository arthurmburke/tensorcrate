//! Errors from tensor operations.
//!
//! With the language handled entirely by the `math!` macro at compile time, the
//! only failures left are the ones that genuinely depend on runtime values:
//! mismatched shapes and singular matrices.

use std::fmt::{self, Display};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Shapes are incompatible for the requested operation.
    Shape(String),
    /// Invalid arguments
    InvalidArgument(String),
    /// A matrix has no inverse.
    Singular,
}

impl Error {
    pub fn shape(msg: impl Into<String>) -> Error {
        Error::Shape(msg.into())
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Shape(msg) => write!(f, "shape error: {msg}"),
            Error::Singular => write!(f, "matrix is singular and cannot be inverted"),
            Error::InvalidArgument(msg) => write!(f, "invalid arguments: {msg}"),
        }
    }
}

impl std::error::Error for Error {}
