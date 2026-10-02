//! Saving and loading tensors.
//!
//! [`Vector`] and [`Matrix`] carry their shapes as runtime fields, so the shape
//! of a tensor being loaded comes *out of the file* rather than having to be
//! known before it is opened. Reading is a parse: the header's extents become
//! the tensor's extents, and a file written by one program can be read by
//! another that has no idea how big it is.
//!
//! The element type needs protection that a shape does not, and is the harder
//! half. The numeric tower in [`numbers`](crate::numbers) has twelve leaf
//! types and lets [`Complex`] and [`Dual`] nest without limit, and it
//! deliberately carries no runtime type tag. Nothing in four `f32` distinguishes
//! them from four `i32` once they reach a file, so the format records an element
//! tag and loading checks it. Reading an `f32` file into a `Matrix<i32>` fails
//! with [`Error::Format`] instead of silently handing back reinterpreted bits.
//!
//! Loading a vector still checks one thing about the shape: the file has to hold
//! a vector rather than a matrix, which is the `kind` byte, not the extents.
//!
//! The entry points are [`Vector::write_to`] / [`Vector::read_from`] over any
//! [`Write`] / [`Read`], with [`Vector::save`] / [`Vector::load`] as filesystem
//! conveniences; [`Matrix`] and [`Tensor`] have the same four. A tensor reads
//! a vector or matrix file too, as a tensor of one or two axes. They are
//! defined for the
//! [`Host`](crate::tensors::Host) backend, which is where every element type
//! lives — move a `Metal` tensor across with
//! [`to_backend`](Matrix::to_backend) first.
//!
//! ```
//! use tensorcrate::tensors::Matrix;
//!
//! let m = Matrix::<f64>::from_rows([[1.0, 2.0], [3.0, 4.0]]);
//! let mut bytes = Vec::new();
//! m.write_to(&mut bytes)?;
//!
//! // The shape is read back from the file, not asserted beforehand.
//! let loaded = Matrix::<f64>::read_from(&bytes[..])?;
//! assert_eq!(loaded.shape(), (2, 2));
//! assert_eq!(loaded, m);
//!
//! // The element type is still checked, and still cannot slip through.
//! assert!(Matrix::<i32>::read_from(&bytes[..]).is_err());
//! # Ok::<(), tensorcrate::errors::Error>(())
//! ```
//!
//! # Format
//!
//! Little-endian throughout, so a file written on one architecture reads on
//! another.
//!
//! ```text
//! offset  size  field
//! 0       4     magic, b"TCR1"
//! 4       1     format version
//! 5       1     kind: 0 = vector, 1 = matrix, 2 = tensor
//! 6       1     element tag length, L
//! 7       L     element tag
//! 7+L     8     rows (u64); the length, for a vector
//! 15+L    8     columns (u64); 1, for a vector
//! 23+L    ..    elements, row-major
//! ```
//!
//! A [`Tensor`] stores its rank in place of the rows, and its extents in place
//! of the columns, one `u64` each:
//!
//! ```text
//! 7+L     8     rank, R (u64), at most MAX_RANK
//! 15+L    8·R   extents (u64), outermost first
//! 15+L+8R ..    elements, row-major
//! ```
//!
//! Element tags are a recursive prefix code — `Complex<Dual<f64>>` is
//! `[0x20, 0x21, 0x11]` — so nesting needs no cases of its own. Tags `0x80`
//! through `0xFF` are reserved for out-of-crate implementations of [`Storable`].

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::errors::Error;
use crate::numbers::{Coefficient, Complex, Dual, bf16, f16};
use crate::tensors::{Host, MAX_RANK, Matrix, Tensor, Vector};

const MAGIC: [u8; 4] = *b"TCR1";
const VERSION: u8 = 1;
const KIND_VECTOR: u8 = 0;
const KIND_MATRIX: u8 = 1;
const KIND_TENSOR: u8 = 2;

const TAG_COMPLEX: u8 = 0x20;
const TAG_DUAL: u8 = 0x21;

/// An element type that can be written to and read back from a byte stream.
///
/// Implemented for every leaf of the numeric tower and, recursively, for
/// [`Complex`] and [`Dual`], so any element type a tensor can hold is storable.
///
/// Downstream crates may implement this for their own coefficient types. Tag
/// bytes `0x00` through `0x7F` belong to this crate; use `0x80` and above.
pub trait Storable: Coefficient {
    /// Appends this type's tag. Composite types push their own byte and then
    /// recurse, which is what makes the tag a prefix code.
    fn write_tag(tag: &mut Vec<u8>);

    /// Writes one value. Must consume exactly what [`read_value`](Self::read_value)
    /// reads.
    fn write_value<W: Write>(&self, writer: &mut W) -> io::Result<()>;

    /// Reads one value back.
    fn read_value<Rd: Read>(reader: &mut Rd) -> io::Result<Self>;
}

macro_rules! primitive_storable {
    ($($t:ty => $tag:expr),+ $(,)?) => {$(
        impl Storable for $t {
            fn write_tag(tag: &mut Vec<u8>) {
                tag.push($tag);
            }

            fn write_value<W: Write>(&self, writer: &mut W) -> io::Result<()> {
                writer.write_all(&self.to_le_bytes())
            }

            fn read_value<Rd: Read>(reader: &mut Rd) -> io::Result<Self> {
                let mut bytes = [0u8; core::mem::size_of::<$t>()];
                reader.read_exact(&mut bytes)?;
                Ok(<$t>::from_le_bytes(bytes))
            }
        }
    )+};
}

// `to_le_bytes` on a float is its bit pattern, so signed zero, the infinities,
// and NaN payloads all survive a round trip unchanged.
primitive_storable! {
    i8 => 0x01, u8 => 0x02, i16 => 0x03, u16 => 0x04,
    i32 => 0x05, u32 => 0x06, i64 => 0x07, u64 => 0x08,
    f32 => 0x10, f64 => 0x11, f16 => 0x12, bf16 => 0x13,
}

impl<T: Storable> Storable for Complex<T> {
    fn write_tag(tag: &mut Vec<u8>) {
        tag.push(TAG_COMPLEX);
        T::write_tag(tag);
    }

    fn write_value<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        self.real.write_value(writer)?;
        self.im.write_value(writer)
    }

    fn read_value<Rd: Read>(reader: &mut Rd) -> io::Result<Self> {
        let real = T::read_value(reader)?;
        let im = T::read_value(reader)?;
        Ok(Complex::new(real, im))
    }
}

impl<T: Storable> Storable for Dual<T> {
    fn write_tag(tag: &mut Vec<u8>) {
        tag.push(TAG_DUAL);
        T::write_tag(tag);
    }

    fn write_value<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        self.real.write_value(writer)?;
        self.dual.write_value(writer)
    }

    fn read_value<Rd: Read>(reader: &mut Rd) -> io::Result<Self> {
        let real = T::read_value(reader)?;
        let dual = T::read_value(reader)?;
        Ok(Dual::new(real, dual))
    }
}

/// The tag for `T`, as bytes.
fn tag_of<T: Storable>() -> Vec<u8> {
    let mut tag = Vec::new();
    T::write_tag(&mut tag);
    tag
}

/// Renders a tag as a type name, for error messages only.
///
/// Unknown or truncated tags degrade to a placeholder rather than failing —
/// this runs on the error path, where the useful thing is to describe whatever
/// was actually found.
fn describe(tag: &[u8]) -> String {
    fn walk(tag: &[u8], at: &mut usize) -> String {
        let Some(&code) = tag.get(*at) else {
            return "?".to_string();
        };
        *at += 1;
        match code {
            0x01 => "i8".to_string(),
            0x02 => "u8".to_string(),
            0x03 => "i16".to_string(),
            0x04 => "u16".to_string(),
            0x05 => "i32".to_string(),
            0x06 => "u32".to_string(),
            0x07 => "i64".to_string(),
            0x08 => "u64".to_string(),
            0x10 => "f32".to_string(),
            0x11 => "f64".to_string(),
            0x12 => "f16".to_string(),
            0x13 => "bf16".to_string(),
            TAG_COMPLEX => format!("Complex<{}>", walk(tag, at)),
            TAG_DUAL => format!("Dual<{}>", walk(tag, at)),
            other => format!("<unknown tag 0x{other:02x}>"),
        }
    }

    let mut at = 0;
    walk(tag, &mut at)
}

fn write_header<W: Write, T: Storable>(
    writer: &mut W,
    kind: u8,
    rows: usize,
    cols: usize,
) -> Result<(), Error> {
    write_prefix::<W, T>(writer, kind)?;
    writer.write_all(&(rows as u64).to_le_bytes())?;
    writer.write_all(&(cols as u64).to_le_bytes())?;
    Ok(())
}

/// Writes a header up to its extents: magic, version, kind and element tag.
fn write_prefix<W: Write, T: Storable>(writer: &mut W, kind: u8) -> Result<(), Error> {
    let tag = tag_of::<T>();
    let length = u8::try_from(tag.len())
        .map_err(|_| Error::format(format!("element tag is {} bytes, limit 255", tag.len())))?;

    writer.write_all(&MAGIC)?;
    writer.write_all(&[VERSION, kind, length])?;
    writer.write_all(&tag)?;
    Ok(())
}

/// Reads a header up to its extents, validates everything that is not a shape,
/// and returns the stored kind, which must be one of `kinds`.
fn read_prefix<Rd: Read, T: Storable>(reader: &mut Rd, kinds: &[u8]) -> Result<u8, Error> {
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if magic != MAGIC {
        return Err(Error::format(format!(
            "not a tensor file: expected magic {:?}, found {:?}",
            MAGIC, magic
        )));
    }

    let mut fields = [0u8; 3];
    reader.read_exact(&mut fields)?;
    let [version, stored_kind, tag_length] = fields;

    if version != VERSION {
        return Err(Error::format(format!(
            "format version {version} is not supported; this build reads version {VERSION}"
        )));
    }

    if !kinds.contains(&stored_kind) {
        return Err(Error::format(format!(
            "expected a {}, found a {}",
            name_of_kind(kinds[0]),
            name_of_kind(stored_kind)
        )));
    }

    let mut stored_tag = vec![0u8; usize::from(tag_length)];
    reader.read_exact(&mut stored_tag)?;
    let expected = tag_of::<T>();
    if stored_tag != expected {
        return Err(Error::format(format!(
            "element type is {}, but the target holds {}",
            describe(&stored_tag),
            describe(&expected)
        )));
    }
    Ok(stored_kind)
}

/// Reads one stored extent.
fn read_extent<Rd: Read>(reader: &mut Rd) -> Result<usize, Error> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    let stored = u64::from_le_bytes(bytes);
    usize::try_from(stored).map_err(|_| too_large(stored))
}

/// Reads the rows and columns of a vector or matrix header.
fn read_extents<Rd: Read>(reader: &mut Rd) -> Result<(usize, usize), Error> {
    let (rows, cols) = (read_extent(reader)?, read_extent(reader)?);
    rows.checked_mul(cols).ok_or_else(|| {
        Error::format(format!(
            "stored tensor is {rows}\u{d7}{cols}, which overflows this platform's address space"
        ))
    })?;
    Ok((rows, cols))
}

/// Reads a header, validates everything that is not a shape, and returns the
/// stored extents for the caller to build a tensor of.
fn read_header<Rd: Read, T: Storable>(reader: &mut Rd, kind: u8) -> Result<(usize, usize), Error> {
    read_prefix::<Rd, T>(reader, &[kind])?;
    read_extents(reader)
}

fn too_large(extent: u64) -> Error {
    Error::format(format!(
        "stored extent {extent} does not fit in this platform's usize"
    ))
}

fn name_of_kind(kind: u8) -> &'static str {
    match kind {
        KIND_VECTOR => "vector",
        KIND_MATRIX => "matrix",
        KIND_TENSOR => "tensor",
        _ => "tensor of unknown kind",
    }
}

impl<T: Storable> Vector<T, Host> {
    /// Writes this vector to `writer`.
    ///
    /// Elements go out one at a time, so a `writer` that syscalls per write —
    /// a bare [`File`] — should be wrapped in a [`BufWriter`]. [`save`](Self::save)
    /// does that for you.
    pub fn write_to<W: Write>(&self, mut writer: W) -> Result<(), Error> {
        write_header::<W, T>(&mut writer, KIND_VECTOR, self.len(), 1)?;
        for value in self.data() {
            value.write_value(&mut writer)?;
        }
        Ok(())
    }

    /// Reads a vector of this element type, taking its length from the file.
    ///
    /// Fails with [`Error::Format`] if the stored element type differs, or if
    /// the bytes are a matrix rather than a vector.
    pub fn read_from<Rd: Read>(mut reader: Rd) -> Result<Self, Error> {
        let (len, _) = read_header::<Rd, T>(&mut reader, KIND_VECTOR)?;
        let mut data = Vec::with_capacity(len);
        for _ in 0..len {
            data.push(T::read_value(&mut reader)?);
        }
        Ok(Vector::new(data))
    }

    /// Writes this vector to a file, replacing it if it exists.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        let mut writer = BufWriter::new(File::create(path)?);
        self.write_to(&mut writer)?;
        // `BufWriter` swallows errors when it flushes on drop, so flush here.
        writer.flush()?;
        Ok(())
    }

    /// Reads a vector back from a file written by [`save`](Self::save).
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        Self::read_from(BufReader::new(File::open(path)?))
    }
}

impl<T: Storable> Matrix<T, Host> {
    /// Writes this matrix to `writer` in row-major order.
    ///
    /// As with [`Vector::write_to`], wrap an unbuffered sink.
    pub fn write_to<W: Write>(&self, mut writer: W) -> Result<(), Error> {
        let (rows, cols) = self.shape();
        write_header::<W, T>(&mut writer, KIND_MATRIX, rows, cols)?;
        for value in self.data() {
            value.write_value(&mut writer)?;
        }
        Ok(())
    }

    /// Reads a matrix of this element type, taking its shape from the file.
    ///
    /// Fails with [`Error::Format`] if the stored element type differs, or if
    /// the bytes are a vector rather than a matrix.
    pub fn read_from<Rd: Read>(mut reader: Rd) -> Result<Self, Error> {
        let (rows, cols) = read_header::<Rd, T>(&mut reader, KIND_MATRIX)?;
        let mut data = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            data.push(T::read_value(&mut reader)?);
        }
        Ok(Matrix::from_flat(rows, cols, data))
    }

    /// Writes this matrix to a file, replacing it if it exists.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        let mut writer = BufWriter::new(File::create(path)?);
        self.write_to(&mut writer)?;
        writer.flush()?;
        Ok(())
    }

    /// Reads a matrix back from a file written by [`save`](Self::save).
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        Self::read_from(BufReader::new(File::open(path)?))
    }
}

impl<T: Storable> Tensor<T, Host> {
    /// Writes this tensor to `writer`: its shape, then its elements in
    /// row-major order.
    ///
    /// As with [`Vector::write_to`], wrap an unbuffered sink.
    pub fn write_to<W: Write>(&self, mut writer: W) -> Result<(), Error> {
        write_prefix::<W, T>(&mut writer, KIND_TENSOR)?;
        writer.write_all(&(self.rank() as u64).to_le_bytes())?;
        for &extent in self.shape() {
            writer.write_all(&(extent as u64).to_le_bytes())?;
        }
        for value in self.data() {
            value.write_value(&mut writer)?;
        }
        Ok(())
    }

    /// Reads a tensor of this element type, taking its shape from the file.
    ///
    /// A file written by [`Vector::write_to`] or [`Matrix::write_to`] reads as
    /// a tensor of one or two axes. Fails with [`Error::Format`] if the stored
    /// element type differs, or the stored rank exceeds
    /// [`MAX_RANK`].
    pub fn read_from<Rd: Read>(mut reader: Rd) -> Result<Self, Error> {
        let kind = read_prefix::<Rd, T>(&mut reader, &[KIND_TENSOR, KIND_VECTOR, KIND_MATRIX])?;
        let shape = match kind {
            KIND_VECTOR => vec![read_extents(&mut reader)?.0],
            KIND_MATRIX => {
                let (rows, cols) = read_extents(&mut reader)?;
                vec![rows, cols]
            }
            _ => {
                let rank = read_extent(&mut reader)?;
                if rank > MAX_RANK {
                    return Err(Error::format(format!(
                        "stored tensor has {rank} axes; a tensor may have at most {MAX_RANK}"
                    )));
                }
                (0..rank)
                    .map(|_| read_extent(&mut reader))
                    .collect::<Result<Vec<_>, _>>()?
            }
        };
        let count = shape
            .iter()
            .try_fold(1usize, |count, &extent| count.checked_mul(extent))
            .ok_or_else(|| {
                Error::format(format!(
                    "stored tensor of shape {shape:?} overflows this platform's address space"
                ))
            })?;
        let mut data = Vec::with_capacity(count.min(1 << 20));
        for _ in 0..count {
            data.push(T::read_value(&mut reader)?);
        }
        Ok(Tensor::from_vec(&shape, data))
    }

    /// Writes this tensor to a file, replacing it if it exists.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        let mut writer = BufWriter::new(File::create(path)?);
        self.write_to(&mut writer)?;
        writer.flush()?;
        Ok(())
    }

    /// Reads a tensor back from a file written by [`save`](Self::save).
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        Self::read_from(BufReader::new(File::open(path)?))
    }
}

#[cfg(test)]
mod tag_tests {
    use super::*;

    #[test]
    fn tags_are_a_prefix_code_over_nesting() {
        assert_eq!(tag_of::<f64>(), vec![0x11]);
        assert_eq!(tag_of::<Complex<f64>>(), vec![TAG_COMPLEX, 0x11]);
        assert_eq!(
            tag_of::<Complex<Dual<f64>>>(),
            vec![TAG_COMPLEX, TAG_DUAL, 0x11]
        );
        assert_eq!(
            tag_of::<Dual<Complex<f32>>>(),
            vec![TAG_DUAL, TAG_COMPLEX, 0x10]
        );
    }

    #[test]
    fn descriptions_round_trip_the_tag() {
        assert_eq!(describe(&tag_of::<i16>()), "i16");
        assert_eq!(
            describe(&tag_of::<Complex<Dual<f64>>>()),
            "Complex<Dual<f64>>"
        );
        assert_eq!(
            describe(&tag_of::<Dual<Complex<f32>>>()),
            "Dual<Complex<f32>>"
        );
    }

    #[test]
    fn descriptions_survive_malformed_tags() {
        assert_eq!(describe(&[]), "?");
        assert_eq!(describe(&[TAG_COMPLEX]), "Complex<?>");
        assert_eq!(describe(&[0x7f]), "<unknown tag 0x7f>");
    }
}
