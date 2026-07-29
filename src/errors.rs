//! Errors from tensor operations.
//!
//! With the language handled entirely by the `math!` macro at compile time, the
//! failures left are the ones that genuinely depend on runtime values:
//! mismatched shapes, singular matrices, and — once a tensor is read back from
//! a file rather than written in source — malformed input and IO failures.

use std::fmt::{self, Display};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Shapes are incompatible for the requested operation.
    Shape(String),
    /// Invalid arguments
    InvalidArgument(String),
    /// A matrix has no inverse.
    Singular,
    /// An underlying reader or writer failed.
    ///
    /// The message is [`std::io::Error`]'s, rendered eagerly: that type is
    /// neither `Clone` nor `PartialEq`, and this enum is both.
    Io(String),
    /// Stored bytes are not a tensor this version can read — a bad magic
    /// number, an unsupported format version, or an element type that does not
    /// match the one being loaded into.
    Format(String),
}

impl Error {
    pub fn shape(msg: impl Into<String>) -> Error {
        Error::Shape(msg.into())
    }

    pub fn format(msg: impl Into<String>) -> Error {
        Error::Format(msg.into())
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Error {
        Error::Io(error.to_string())
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Shape(msg) => write!(f, "shape error: {msg}"),
            Error::Singular => write!(f, "matrix is singular and cannot be inverted"),
            Error::InvalidArgument(msg) => write!(f, "invalid arguments: {msg}"),
            Error::Io(msg) => write!(f, "io error: {msg}"),
            Error::Format(msg) => write!(f, "malformed tensor data: {msg}"),
        }
    }
}

impl std::error::Error for Error {}
