//! [`Tensor`]: an owned N-dimensional tensor whose shape is a runtime value.

use std::fmt::{self, Display};
use std::ops::{Index, IndexMut, RangeBounds};

use super::backend::Strided;
use super::layout::{
    AxisIndex, Dims, contiguous_strides, element_count, expect_position, position, resolve_axis,
};
use super::{Backend, Host, Matrix, TensorView, Vector};
use crate::numbers::Coefficient;

/// An N-dimensional tensor of up to [`MAX_RANK`](super::MAX_RANK) axes, whose
/// shape is fixed when it is built.
///
/// The elements are owned and lie in row-major order in the same flat storage
/// a [`Vector`] uses, on backend `B` — the [`Host`] `Vec` by default, or
/// GPU-shared memory on `Metal`. The shape is just the extents laid over that
/// run of elements, so changing it without changing the order —
/// [`reshape`](Self::reshape), [`squeeze`](Self::squeeze),
/// [`unsqueeze`](Self::unsqueeze), and the conversions to and from [`Vector`]
/// and [`Matrix`] — moves the storage rather than copying it.
///
/// Operations that reorder or pick out elements — [`permute`](Self::permute),
/// [`transpose`](Self::transpose), [`narrow`](Self::narrow),
/// [`select`](Self::select), [`split`](Self::split) — return a
/// [`TensorView`], which reads this tensor's storage in place through
/// strides. [`contiguous`](TensorView::contiguous) copies a view into a tensor
/// of its own, in one strided copy on the tensor's backend.
///
/// ```
/// use tensorcrate::tensors::Tensor;
///
/// // A batch of 2 sequences of 3 positions, each 4 features wide.
/// let x = Tensor::from_vec(&[2, 3, 4], (0..24).map(|i| i as f32).collect::<Vec<_>>());
///
/// // Split the features into 2 heads of 2, and bring the heads forward.
/// let heads = x.view_as(&[2, 3, 2, 2]).permute(&[0, 2, 1, 3]);
/// assert_eq!(heads.shape(), [2, 2, 3, 2]);
/// assert_eq!(heads.get(&[0, 1, 2, 0]), Some(10.0)); // position 2, feature 2·1 + 0
///
/// // And back: copy into order, undo the permutation, merge the heads.
/// let merged = heads
///     .contiguous()
///     .permute(&[0, 2, 1, 3])
///     .contiguous()
///     .reshape(&[2, 3, 4]);
/// assert_eq!(merged, x);
/// ```
pub struct Tensor<T, B: Backend = Host> {
    shape: Dims,
    data: Vector<T, B>,
}

// As for `Vector`: which of these a tensor gets depends on its storage.
impl<T, B: Backend> Clone for Tensor<T, B>
where
    Vector<T, B>: Clone,
{
    fn clone(&self) -> Self {
        Tensor {
            shape: self.shape,
            data: self.data.clone(),
        }
    }
}

impl<T, B: Backend> PartialEq for Tensor<T, B>
where
    Vector<T, B>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

impl<T, B: Backend> Eq for Tensor<T, B> where Vector<T, B>: Eq {}

impl<T, B: Backend> fmt::Debug for Tensor<T, B>
where
    Vector<T, B>: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("data", &self.data)
            .finish()
    }
}

/// The shape queries, which read the extents and need nothing else.
impl<T, B: Backend> Tensor<T, B> {
    /// The extent of each axis, outermost first. A scalar's is empty.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The number of axes.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// The number of elements: the product of the extents, so `1` for a
    /// scalar.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether this tensor holds no elements, because some axis is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The extent of `axis`.
    ///
    /// # Panics
    ///
    /// If there is no such axis.
    #[track_caller]
    pub fn dim(&self, axis: impl AxisIndex) -> usize {
        self.shape[resolve_axis(axis, &self.shape, "dim")]
    }

    /// The row-major strides of this tensor's shape: how many elements of its
    /// storage each axis steps over.
    pub fn strides(&self) -> Vec<usize> {
        contiguous_strides(&self.shape).to_vec()
    }
}

/// Copyable tensors on any backend: construction, layout and moving between
/// backends. The arithmetic is in the [`Kernels`](super::Kernels) section.
impl<T: Copy + 'static, B: Backend> Tensor<T, B> {
    /// Attach a shape to a vector's elements, which it takes in row-major
    /// order without copying them.
    ///
    /// # Panics
    ///
    /// If the vector's length is not the number of elements `shape` holds, or
    /// `shape` has more than [`MAX_RANK`](super::MAX_RANK) axes.
    #[track_caller]
    pub fn from_vector(shape: &[usize], values: Vector<T, B>) -> Self {
        let dims = Dims::new(shape, "from_vector");
        let count = element_count(shape, "from_vector");
        assert!(
            values.len() == count,
            "from_vector: {} elements cannot fill shape {shape:?}",
            values.len()
        );
        Tensor {
            shape: dims,
            data: values,
        }
    }

    /// A tensor of `shape` from its row-major elements, stored on backend
    /// `B`.
    ///
    /// # Panics
    ///
    /// As for [`from_vector`](Self::from_vector).
    #[track_caller]
    pub fn from_slice(shape: &[usize], values: &[T]) -> Self {
        let count = element_count(shape, "from_slice");
        assert!(
            values.len() == count,
            "from_slice: {} elements cannot fill shape {shape:?}",
            values.len()
        );
        Tensor {
            shape: Dims::new(shape, "from_slice"),
            data: Vector::build(values),
        }
    }

    /// A tensor of `shape` with every element set to `value`, allocated
    /// directly on backend `B`.
    #[track_caller]
    pub fn filled(shape: &[usize], value: T) -> Self {
        let count = element_count(shape, "filled");
        Tensor {
            shape: Dims::new(shape, "filled"),
            data: Vector::filled(count, value),
        }
    }

    /// Attach a checked shape to storage that holds exactly its elements.
    pub(crate) fn from_parts(shape: Dims, data: Vector<T, B>) -> Self {
        debug_assert_eq!(data.len(), element_count(&shape, "tensor"));
        Tensor { shape, data }
    }

    /// Move this tensor's elements onto backend `B2` — the one operation that
    /// copies between CPU and GPU memory.
    pub fn to_backend<B2: Backend>(&self) -> Tensor<T, B2> {
        Tensor {
            shape: self.shape,
            data: self.data.to_backend(),
        }
    }

    /// Borrow the elements as one flat row-major slice, without copying.
    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }

    /// Copy the elements, in row-major order, into a `Vec`.
    pub fn to_vec(&self) -> Vec<T> {
        self.data.to_vec()
    }

    /// The elements as the flat [`Vector`] they are stored in.
    pub fn as_vector(&self) -> &Vector<T, B> {
        &self.data
    }

    /// The element at `index`, one coordinate per axis, or `None` if it is
    /// out of range. On `Metal` this waits for queued work, as any read does.
    pub fn get(&self, index: &[usize]) -> Option<T> {
        let strides = contiguous_strides(&self.shape);
        position(&self.shape, &strides, 0, index).map(|at| self.as_slice()[at])
    }

    /// This whole tensor as a view.
    pub fn view(&self) -> TensorView<'_, T, B> {
        TensorView::of(&self.data, self.shape)
    }

    /// The same elements under a new shape, without copying: the elements stay
    /// in row-major order and the extents change.
    ///
    /// # Panics
    ///
    /// If `shape` holds a different number of elements.
    #[track_caller]
    pub fn reshape(self, shape: &[usize]) -> Self {
        let count = element_count(shape, "reshape");
        assert!(
            count == self.len(),
            "reshape: shape {:?} holds {} elements, which shape {shape:?} cannot",
            self.shape,
            self.len()
        );
        Tensor {
            shape: Dims::new(shape, "reshape"),
            data: self.data,
        }
    }

    /// This tensor read under a new shape, in place. Always possible, since a
    /// tensor is in row-major order.
    ///
    /// # Panics
    ///
    /// As for [`reshape`](Self::reshape).
    #[track_caller]
    pub fn view_as(&self, shape: &[usize]) -> TensorView<'_, T, B> {
        let count = element_count(shape, "view_as");
        assert!(
            count == self.len(),
            "view_as: shape {:?} holds {} elements, which shape {shape:?} cannot",
            self.shape,
            self.len()
        );
        TensorView::of(&self.data, Dims::new(shape, "view_as"))
    }

    /// Remove `axis`, which must have one element. A move, like
    /// [`reshape`](Self::reshape).
    ///
    /// # Panics
    ///
    /// If there is no such axis or its extent is not one.
    #[track_caller]
    pub fn squeeze(self, axis: impl AxisIndex) -> Self {
        let shape = *self.view().squeeze(axis).dims();
        Tensor {
            shape,
            data: self.data,
        }
    }

    /// Insert an axis of one element before `axis`, which may be the rank to
    /// append one. A move, like [`reshape`](Self::reshape).
    ///
    /// # Panics
    ///
    /// If `axis` is past the rank, or the tensor already has
    /// [`MAX_RANK`](super::MAX_RANK) axes.
    #[track_caller]
    pub fn unsqueeze(self, axis: impl AxisIndex) -> Self {
        let shape = *self.view().unsqueeze(axis).dims();
        Tensor {
            shape,
            data: self.data,
        }
    }

    /// The elements as a flat vector, without copying.
    pub fn into_vector(self) -> Vector<T, B> {
        self.data
    }

    /// The elements as a matrix, without copying.
    ///
    /// # Panics
    ///
    /// Unless the tensor has exactly two axes; [`reshape`](Self::reshape) to
    /// two first, which is free.
    #[track_caller]
    pub fn into_matrix(self) -> Matrix<T, B> {
        assert!(
            self.rank() == 2,
            "into_matrix: shape {:?} has {} axes, not 2",
            self.shape,
            self.rank()
        );
        let (rows, cols) = (self.shape[0], self.shape[1]);
        Matrix::from_storage(rows, cols, B::vector_into_matrix(self.data.into_storage()))
    }

    /// A copy of this tensor on the same backend — on `Metal`, made on the
    /// GPU without waiting.
    pub fn contiguous(&self) -> Self {
        self.to_backend()
    }

    /// The axes reordered: axis `k` of the view is axis `axes[k]` of this
    /// tensor. See [`TensorView::permute`].
    #[track_caller]
    pub fn permute(&self, axes: &[usize]) -> TensorView<'_, T, B> {
        self.view().permute(axes)
    }

    /// Axes `a` and `b` swapped. See [`TensorView::transpose`].
    #[track_caller]
    pub fn transpose(&self, a: impl AxisIndex, b: impl AxisIndex) -> TensorView<'_, T, B> {
        self.view().transpose(a, b)
    }

    /// `len` elements of `axis` from `start`. See [`TensorView::narrow`].
    #[track_caller]
    pub fn narrow(&self, axis: impl AxisIndex, start: usize, len: usize) -> TensorView<'_, T, B> {
        self.view().narrow(axis, start, len)
    }

    /// The elements of `axis` in `range`. See [`TensorView::slice`].
    #[track_caller]
    pub fn slice(
        &self,
        axis: impl AxisIndex,
        range: impl RangeBounds<usize>,
    ) -> TensorView<'_, T, B> {
        self.view().slice(axis, range)
    }

    /// Element `index` of `axis`, with the axis removed. See
    /// [`TensorView::select`].
    #[track_caller]
    pub fn select(&self, axis: impl AxisIndex, index: usize) -> TensorView<'_, T, B> {
        self.view().select(axis, index)
    }

    /// Consecutive pieces of `axis` with the given extents. See
    /// [`TensorView::split`].
    #[track_caller]
    pub fn split(&self, axis: impl AxisIndex, sizes: &[usize]) -> Vec<TensorView<'_, T, B>> {
        self.view().split(axis, sizes)
    }

    /// `parts` equal pieces of `axis`. See [`TensorView::chunk`].
    #[track_caller]
    pub fn chunk(&self, axis: impl AxisIndex, parts: usize) -> Vec<TensorView<'_, T, B>> {
        self.view().chunk(axis, parts)
    }

    /// Tensors joined along `axis`, which every other extent must agree on.
    ///
    /// Each piece may be a tensor or any view of one, read through its
    /// strides: the result is assembled in one copy per piece on the backend,
    /// with nothing made contiguous first.
    ///
    /// ```
    /// use tensorcrate::tensors::Tensor;
    ///
    /// let a = Tensor::from_vec(&[2, 1], vec![1.0f32, 2.0]);
    /// let b = Tensor::from_vec(&[2, 2], vec![3.0f32, 4.0, 5.0, 6.0]);
    /// let joined = Tensor::concat([&a, &b], 1);
    /// assert_eq!(joined, Tensor::from_vec(&[2, 3], vec![1.0, 3.0, 4.0, 2.0, 5.0, 6.0]));
    /// ```
    ///
    /// # Panics
    ///
    /// If there are no pieces, their ranks differ, or two of them differ in an
    /// extent other than `axis`'s.
    #[track_caller]
    pub fn concat<'a, V>(pieces: impl IntoIterator<Item = V>, axis: impl AxisIndex) -> Self
    where
        V: Into<TensorView<'a, T, B>>,
    {
        let pieces = pieces.into_iter().map(Into::into).collect::<Vec<_>>();
        let first = pieces
            .first()
            .unwrap_or_else(|| panic!("concat: there are no tensors to join"));
        let axis = resolve_axis(axis, first.shape(), "concat");
        let mut extent = 0;
        for piece in &pieces {
            let agrees = piece.rank() == first.rank()
                && (0..first.rank()).all(|k| k == axis || piece.shape()[k] == first.shape()[k]);
            assert!(
                agrees,
                "concat: shapes {:?} and {:?} differ outside axis {axis}",
                first.shape(),
                piece.shape()
            );
            extent += piece.shape()[axis];
        }
        let shape = first.dims().with(axis, extent);
        let strides = contiguous_strides(&shape);
        let len = element_count(&shape, "concat");
        let mut start = 0;
        let regions = pieces
            .iter()
            .map(|piece| {
                let region = piece.region(Strided {
                    offset: start * strides[axis],
                    strides: &strides,
                });
                start += piece.shape()[axis];
                region
            })
            .collect::<Vec<_>>();
        Tensor {
            shape,
            data: Vector::from_storage(len, B::assemble(len, &regions)),
        }
    }

    /// Tensors of one shape stacked along a new axis inserted before `axis`,
    /// which may be their rank to stack innermost.
    ///
    /// # Panics
    ///
    /// If there are no pieces, their shapes differ, or they already have
    /// [`MAX_RANK`](super::MAX_RANK) axes.
    #[track_caller]
    pub fn stack<'a, V>(pieces: impl IntoIterator<Item = V>, axis: impl AxisIndex) -> Self
    where
        V: Into<TensorView<'a, T, B>>,
    {
        let pieces = pieces.into_iter().map(Into::into).collect::<Vec<_>>();
        let first = pieces
            .first()
            .unwrap_or_else(|| panic!("stack: there are no tensors to stack"));
        for piece in &pieces {
            assert!(
                piece.shape() == first.shape(),
                "stack: shapes {:?} and {:?} differ",
                first.shape(),
                piece.shape()
            );
        }
        let axis = axis.resolve(first.rank() + 1).unwrap_or_else(|| {
            panic!(
                "stack: axis {axis:?} is out of range for stacking shape {:?}",
                first.shape()
            )
        });
        Self::concat(pieces.iter().map(|piece| piece.unsqueeze(axis)), axis)
    }

    /// Overwrite the elements of `axis` from `start` with `source`, in place:
    /// `source` has this tensor's shape except along `axis`, where it covers
    /// `start..start + source.dim(axis)`. One strided copy on the backend,
    /// reading `source` through its strides — on `Metal` a GPU write into the
    /// existing allocation, which is how a cache grows a step at a time.
    ///
    /// ```
    /// use tensorcrate::tensors::Tensor;
    ///
    /// let mut cache = Tensor::filled(&[2, 4], 0.0f32);
    /// let step = Tensor::from_vec(&[2, 1], vec![7.0, 8.0]);
    /// cache.write_slice(1, 2, &step);
    /// assert_eq!(cache.to_vec(), [0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 8.0, 0.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// If the shapes disagree outside `axis`, or the written range reaches
    /// past this tensor's extent.
    #[track_caller]
    pub fn write_slice<'s>(
        &mut self,
        axis: impl AxisIndex,
        start: usize,
        source: impl Into<TensorView<'s, T, B>>,
    ) {
        let source = source.into();
        let axis = resolve_axis(axis, &self.shape, "write_slice");
        let fits = source.rank() == self.rank()
            && (0..self.rank()).all(|k| k == axis || source.shape()[k] == self.shape[k])
            && start
                .checked_add(source.shape()[axis])
                .is_some_and(|end| end <= self.shape[axis]);
        assert!(
            fits,
            "write_slice: a {:?} source does not fit shape {:?} from {start} along axis {axis}",
            source.shape(),
            self.shape
        );
        let strides = contiguous_strides(&self.shape);
        let region = source.region(Strided {
            offset: start * strides[axis],
            strides: &strides,
        });
        B::strided_write(self.data.storage_mut(), region);
    }
}

impl<T: Coefficient, B: Backend> Tensor<T, B> {
    /// A tensor of `shape` filled with zeros, on backend `B`.
    #[track_caller]
    pub fn zeros(shape: &[usize]) -> Self {
        Self::filled(shape, T::zero())
    }

    /// A tensor of `shape` filled with ones, on backend `B`.
    #[track_caller]
    pub fn ones(shape: &[usize]) -> Self {
        Self::filled(shape, T::one())
    }
}

impl<T> Tensor<T, Host> {
    /// A host tensor of `shape` from its row-major elements, taking ownership
    /// of them.
    ///
    /// # Panics
    ///
    /// If `values` does not hold exactly the elements of `shape`, or `shape`
    /// has more than [`MAX_RANK`](super::MAX_RANK) axes.
    #[track_caller]
    pub fn from_vec(shape: &[usize], values: impl Into<Vec<T>>) -> Self {
        let values = values.into();
        let count = element_count(shape, "from_vec");
        assert!(
            values.len() == count,
            "from_vec: {} elements cannot fill shape {shape:?}",
            values.len()
        );
        Tensor {
            shape: Dims::new(shape, "from_vec"),
            data: Vector::new(values),
        }
    }

    /// Borrow the elements as one flat row-major slice.
    pub fn data(&self) -> &[T] {
        self.data.data()
    }

    /// Borrow the elements as one flat row-major mutable slice.
    pub fn data_mut(&mut self) -> &mut [T] {
        self.data.data_mut()
    }

    /// Consume this tensor and take its row-major elements.
    pub fn into_vec(self) -> Vec<T> {
        self.data.into_vec()
    }
}

impl<T, B: Backend> From<Vector<T, B>> for Tensor<T, B> {
    /// A vector as a tensor of one axis, without copying.
    fn from(vector: Vector<T, B>) -> Self {
        Tensor {
            shape: Dims::new(&[vector.len()], "from"),
            data: vector,
        }
    }
}

impl<T: Copy + 'static, B: Backend> From<Matrix<T, B>> for Tensor<T, B> {
    /// A matrix as a tensor of two axes, without copying.
    fn from(matrix: Matrix<T, B>) -> Self {
        let (rows, cols) = matrix.shape();
        let data = B::matrix_into_flattened(matrix.into_storage());
        Tensor {
            shape: Dims::new(&[rows, cols], "from"),
            data: Vector::from_storage(rows * cols, data),
        }
    }
}

impl<'a, T: Copy + 'static, B: Backend> From<&'a Tensor<T, B>> for TensorView<'a, T, B> {
    fn from(tensor: &'a Tensor<T, B>) -> Self {
        tensor.view()
    }
}

impl<T, const N: usize> Index<[usize; N]> for Tensor<T, Host> {
    type Output = T;

    /// The element at a full index, one coordinate per axis.
    #[track_caller]
    fn index(&self, index: [usize; N]) -> &T {
        let strides = contiguous_strides(&self.shape);
        &self.data.data()[expect_position(&self.shape, &strides, 0, &index)]
    }
}

impl<T, const N: usize> IndexMut<[usize; N]> for Tensor<T, Host> {
    #[track_caller]
    fn index_mut(&mut self, index: [usize; N]) -> &mut T {
        let strides = contiguous_strides(&self.shape);
        let at = expect_position(&self.shape, &strides, 0, &index);
        &mut self.data.data_mut()[at]
    }
}

impl<T: Display> Display for Tensor<T, Host> {
    /// Nested brackets, one level per axis, in the style of
    /// [`Vector`]'s and [`Matrix`]'s: the innermost axis on a line, and each
    /// outer one a block of lines.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_block(f, self.data.data(), &self.shape, 0)
    }
}

fn write_block<T: Display>(
    f: &mut fmt::Formatter<'_>,
    values: &[T],
    shape: &[usize],
    depth: usize,
) -> fmt::Result {
    match shape {
        [] => write!(f, "{}", values[0]),
        [_] => {
            write!(f, "[")?;
            for value in values {
                write!(f, " {value}")?;
            }
            write!(f, " ]")
        }
        [outer, inner @ ..] => {
            let step = values.len().checked_div(*outer).unwrap_or(0);
            write!(f, "[")?;
            for block in 0..*outer {
                if block > 0 {
                    writeln!(f)?;
                    write!(f, "{:width$}", "", width = depth + 1)?;
                }
                write_block(
                    f,
                    &values[block * step..(block + 1) * step],
                    inner,
                    depth + 1,
                )?;
            }
            write!(f, "]")
        }
    }
}
