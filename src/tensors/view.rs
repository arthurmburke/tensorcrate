//! [`TensorView`]: a tensor's elements read in place through a strided layout.

use std::fmt;
use std::ops::{Index, RangeBounds};

use super::backend::{Region, Strided};
use super::layout::{
    AxisIndex, Dims, Split, bounds, broadcast_strides, contiguous_strides, element_count,
    expect_position, is_contiguous, position, reshape_strides, resolve_axis,
};
use super::strided::Layout;
use super::{Backend, Host, Tensor, Vector};

/// Part or all of a [`Tensor`]'s elements, read in place: a shape, one stride
/// per axis and an offset over the tensor's storage. Element `[i₀, …, iₙ₋₁]`
/// of the view is storage element `offset + Σ iₖ·strides[k]`.
///
/// Every layout operation — [`permute`](Self::permute),
/// [`transpose`](Self::transpose), [`narrow`](Self::narrow),
/// [`slice`](Self::slice), [`select`](Self::select), [`split`](Self::split),
/// [`chunk`](Self::chunk), [`squeeze`](Self::squeeze),
/// [`unsqueeze`](Self::unsqueeze) and [`try_reshape`](Self::try_reshape) —
/// only rewrites those numbers, so it is free on either backend and never
/// waits for the GPU. [`contiguous`](Self::contiguous) is where elements move:
/// one strided copy into a tensor of their own, on the view's backend.
///
/// A view is `Copy` — it is a borrow and a few numbers — and the tensor it
/// reads cannot change while it lives. The arithmetic of
/// [`Tensor`] is defined on views too; a view that is not its tensor's whole
/// storage in order is copied into order first.
pub struct TensorView<'a, T, B: Backend = Host> {
    data: &'a Vector<T, B>,
    shape: Dims,
    strides: Dims,
    offset: usize,
}

impl<T, B: Backend> Clone for TensorView<'_, T, B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, B: Backend> Copy for TensorView<'_, T, B> {}

impl<T, B: Backend> fmt::Debug for TensorView<'_, T, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorView")
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .field("offset", &self.offset)
            .finish()
    }
}

/// The layout queries.
impl<'a, T, B: Backend> TensorView<'a, T, B> {
    /// The extent of each axis, outermost first.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// How many storage elements each axis steps over.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// The storage element the view's first element is.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// The number of axes.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// The number of elements the view reads.
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Whether the view reads no elements.
    pub fn is_empty(&self) -> bool {
        self.shape.contains(&0)
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

    /// Whether the view reads its elements in row-major order, one after the
    /// other — though not necessarily from the start of the storage.
    pub fn is_contiguous(&self) -> bool {
        is_contiguous(&self.shape, &self.strides)
    }

    pub(crate) fn dims(&self) -> &Dims {
        &self.shape
    }

    pub(crate) fn data(&self) -> &'a Vector<T, B> {
        self.data
    }

    /// Whether the view is its storage exactly: every element, in order.
    pub(crate) fn is_whole(&self) -> bool {
        self.offset == 0 && self.len() == self.data.len() && self.is_contiguous()
    }
}

impl<'a, T: Copy + 'static, B: Backend> TensorView<'a, T, B> {
    /// The whole of `data`, in row-major order, as `shape`.
    pub(crate) fn of(data: &'a Vector<T, B>, shape: Dims) -> Self {
        TensorView {
            data,
            shape,
            strides: contiguous_strides(&shape),
            offset: 0,
        }
    }

    fn with(&self, shape: Dims, strides: Dims, offset: usize) -> Self {
        TensorView {
            data: self.data,
            shape,
            strides,
            offset,
        }
    }

    /// The axes reordered: axis `k` of the result is axis `axes[k]` of this
    /// view, so `permute(&[0, 2, 1, 3])` turns `[B, T, H, D]` into
    /// `[B, H, T, D]`.
    ///
    /// # Panics
    ///
    /// Unless `axes` names every axis exactly once.
    #[track_caller]
    pub fn permute(&self, axes: &[usize]) -> Self {
        let rank = self.rank();
        let mut seen = [false; super::MAX_RANK];
        let valid = axes.len() == rank
            && axes
                .iter()
                .all(|&axis| axis < rank && !std::mem::replace(&mut seen[axis], true));
        assert!(
            valid,
            "permute: {axes:?} is not an order of the {rank} axes of shape {:?}",
            self.shape
        );
        let shape = axes
            .iter()
            .map(|&axis| self.shape[axis])
            .collect::<Vec<_>>();
        let strides = axes
            .iter()
            .map(|&axis| self.strides[axis])
            .collect::<Vec<_>>();
        self.with(
            Dims::new(&shape, "permute"),
            Dims::new(&strides, "permute"),
            self.offset,
        )
    }

    /// Axes `a` and `b` swapped; negative axes count from the last, so
    /// `transpose(-2, -1)` transposes the innermost matrices.
    ///
    /// # Panics
    ///
    /// If either axis does not exist.
    #[track_caller]
    pub fn transpose(&self, a: impl AxisIndex, b: impl AxisIndex) -> Self {
        let a = resolve_axis(a, &self.shape, "transpose");
        let b = resolve_axis(b, &self.shape, "transpose");
        let (mut shape, mut strides) = (self.shape, self.strides);
        let (extent_a, stride_a) = (shape[a], strides[a]);
        shape = shape.with(a, shape[b]).with(b, extent_a);
        strides = strides.with(a, strides[b]).with(b, stride_a);
        self.with(shape, strides, self.offset)
    }

    /// `len` elements of `axis`, from `start`.
    ///
    /// # Panics
    ///
    /// If the axis does not exist or the range reaches past its extent.
    #[track_caller]
    pub fn narrow(&self, axis: impl AxisIndex, start: usize, len: usize) -> Self {
        let axis = resolve_axis(axis, &self.shape, "narrow");
        let end = start.saturating_add(len);
        let range = bounds(start..end, self.shape[axis], axis, "narrow");
        self.restrict(axis, range.start, range.len())
    }

    /// The elements of `axis` in `range`.
    ///
    /// # Panics
    ///
    /// If the axis does not exist or the range reaches past its extent.
    #[track_caller]
    pub fn slice(&self, axis: impl AxisIndex, range: impl RangeBounds<usize>) -> Self {
        let axis = resolve_axis(axis, &self.shape, "slice");
        let range = bounds(range, self.shape[axis], axis, "slice");
        self.restrict(axis, range.start, range.len())
    }

    fn restrict(&self, axis: usize, start: usize, len: usize) -> Self {
        // An empty view reads nothing, so it keeps the offset it had rather
        // than one that may lie past the storage.
        let offset = if len == 0 {
            self.offset
        } else {
            self.offset + start * self.strides[axis]
        };
        self.with(self.shape.with(axis, len), self.strides, offset)
    }

    /// Element `index` of `axis`, with that axis removed: `select(0, i)` of a
    /// batch is its `i`th member.
    ///
    /// # Panics
    ///
    /// If the axis does not exist or `index` is past its extent.
    #[track_caller]
    pub fn select(&self, axis: impl AxisIndex, index: usize) -> Self {
        let axis = resolve_axis(axis, &self.shape, "select");
        assert!(
            index < self.shape[axis],
            "select: index {index} is out of range for axis {axis} of shape {:?}",
            self.shape
        );
        self.with(
            self.shape.removed(axis),
            self.strides.removed(axis),
            self.offset + index * self.strides[axis],
        )
    }

    /// Consecutive pieces of `axis`, of the extents in `sizes`.
    ///
    /// # Panics
    ///
    /// If the axis does not exist or `sizes` does not add up to its extent.
    #[track_caller]
    pub fn split(&self, axis: impl AxisIndex, sizes: &[usize]) -> Vec<Self> {
        let axis = resolve_axis(axis, &self.shape, "split");
        let total = sizes
            .iter()
            .try_fold(0usize, |total, &size| total.checked_add(size));
        assert!(
            total == Some(self.shape[axis]),
            "split: sizes {sizes:?} do not add up to axis {axis}'s extent of {} in shape {:?}",
            self.shape[axis],
            self.shape
        );
        let mut start = 0;
        sizes
            .iter()
            .map(|&size| {
                let piece = self.restrict(axis, start, size);
                start += size;
                piece
            })
            .collect()
    }

    /// `parts` equal pieces of `axis` — `chunk(-1, 3)` splits a fused
    /// query–key–value projection into its three parts.
    ///
    /// # Panics
    ///
    /// If the axis does not exist, or `parts` is zero or does not divide its
    /// extent.
    #[track_caller]
    pub fn chunk(&self, axis: impl AxisIndex, parts: usize) -> Vec<Self> {
        let axis = resolve_axis(axis, &self.shape, "chunk");
        let extent = self.shape[axis];
        assert!(
            parts != 0 && extent.is_multiple_of(parts),
            "chunk: axis {axis} of shape {:?} does not divide into {parts} equal parts",
            self.shape
        );
        self.split(axis, &vec![extent / parts; parts])
    }

    /// `axis` removed; it must have one element.
    ///
    /// # Panics
    ///
    /// If there is no such axis or its extent is not one.
    #[track_caller]
    pub fn squeeze(&self, axis: impl AxisIndex) -> Self {
        let axis = resolve_axis(axis, &self.shape, "squeeze");
        assert!(
            self.shape[axis] == 1,
            "squeeze: axis {axis} of shape {:?} has {} elements, not 1",
            self.shape,
            self.shape[axis]
        );
        self.with(
            self.shape.removed(axis),
            self.strides.removed(axis),
            self.offset,
        )
    }

    /// An axis of one element inserted before `axis`, which may be the rank to
    /// append one.
    ///
    /// # Panics
    ///
    /// If `axis` is past the rank, or the view already has
    /// [`MAX_RANK`](super::MAX_RANK) axes.
    #[track_caller]
    pub fn unsqueeze(&self, axis: impl AxisIndex) -> Self {
        let rank = self.rank();
        let axis = axis.resolve(rank + 1).unwrap_or_else(|| {
            panic!(
                "unsqueeze: axis {axis:?} is out of range for inserting into shape {:?}",
                self.shape
            )
        });
        // The new axis steps nowhere; giving it the stride a row-major layout
        // would keeps a contiguous view contiguous.
        let stride = if axis < rank {
            self.strides[axis] * self.shape[axis]
        } else {
            1
        };
        self.with(
            self.shape.inserted(axis, 1, "unsqueeze"),
            self.strides.inserted(axis, stride, "unsqueeze"),
            self.offset,
        )
    }

    /// This view repeated to `shape`, numpy-style, without copying: the two
    /// shapes are aligned at their last axes, and an axis of one element —
    /// or a leading axis `shape` adds — is read with a stride of zero, so
    /// every index along it reads the same elements. `[D]` broadcasts to
    /// `[B, T, D]`, and `[T, T]` to `[B, H, T, T]`.
    ///
    /// The tensor operations of two operands broadcast them this way
    /// themselves; this is for the cases that want the repeated view, such as
    /// a [fused program](super::fused::Program) input, whose leading axes fold
    /// into one strided axis when only those were added.
    ///
    /// ```
    /// use tensorcrate::tensors::Tensor;
    ///
    /// let bias = Tensor::from_vec(&[3], vec![1.0f32, 2.0, 3.0]);
    /// let repeated = bias.broadcast_to(&[2, 3]);
    /// assert_eq!(repeated.strides(), [0, 1]);
    /// assert_eq!(repeated.to_vec(), [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// If `shape` has fewer axes than the view, or an extent of the view is
    /// neither `1` nor the matching extent of `shape`.
    #[track_caller]
    pub fn broadcast_to(&self, shape: &[usize]) -> Self {
        let target = Dims::new(shape, "broadcast_to");
        let strides = broadcast_strides(&self.shape, &self.strides, shape).unwrap_or_else(|| {
            panic!(
                "broadcast_to: shape {:?} does not broadcast to {shape:?}",
                self.shape
            )
        });
        self.with(target, strides, self.offset)
    }

    /// The same elements in row-major order under `shape`, read in place, or
    /// `None` when the layout cannot express that without a copy — merging
    /// axes a permutation has separated, say. [`contiguous`](Self::contiguous)
    /// followed by [`Tensor::reshape`] always works.
    ///
    /// # Panics
    ///
    /// If `shape` holds a different number of elements, or has more than
    /// [`MAX_RANK`](super::MAX_RANK) axes.
    #[track_caller]
    pub fn try_reshape(&self, shape: &[usize]) -> Option<Self> {
        let new_shape = Dims::new(shape, "try_reshape");
        let count = element_count(shape, "try_reshape");
        assert!(
            count == self.len(),
            "try_reshape: shape {:?} holds {} elements, which shape {shape:?} cannot",
            self.shape,
            self.len()
        );
        let strides = reshape_strides(&self.shape, &self.strides, shape)?;
        Some(self.with(new_shape, strides, self.offset))
    }

    /// The view's elements copied into a tensor of their own, in row-major
    /// order — one strided copy on the view's backend, which on `Metal` runs on
    /// the GPU without waiting.
    pub fn contiguous(&self) -> Tensor<T, B> {
        if self.is_whole() {
            return Tensor::from_parts(self.shape, self.data.to_backend());
        }
        let len = self.len();
        let storage = B::strided_copy(
            self.data.storage(),
            &self.shape,
            Strided {
                offset: self.offset,
                strides: &self.strides,
            },
        );
        Tensor::from_parts(self.shape, Vector::from_storage(len, storage))
    }

    /// The view's elements as a tensor on backend `B2`.
    pub fn to_backend<B2: Backend>(&self) -> Tensor<T, B2> {
        if self.is_whole() {
            return Tensor::from_parts(self.shape, self.data.to_backend());
        }
        self.contiguous().to_backend()
    }

    /// The element at `index`, one coordinate per axis, or `None` if it is
    /// out of range.
    pub fn get(&self, index: &[usize]) -> Option<T> {
        position(&self.shape, &self.strides, self.offset, index).map(|at| self.data.as_slice()[at])
    }

    /// The view's elements, in row-major order, copied into a `Vec`.
    pub fn to_vec(&self) -> Vec<T> {
        super::backend::strided_copy_slice(
            self.data.as_slice(),
            &self.shape,
            Strided {
                offset: self.offset,
                strides: &self.strides,
            },
        )
    }

    /// This view's layout over its storage, read from the CPU — on `Metal`
    /// after waiting for queued work, as any read does.
    pub(crate) fn strided(&self) -> Layout<'_, T> {
        Layout {
            values: self.data.as_slice(),
            shape: &self.shape,
            strides: &self.strides,
            offset: self.offset,
        }
    }

    /// This view's layout split for folding `axes` — distinct, ascending.
    pub(crate) fn split_for(&self, axes: &[usize]) -> Split {
        Split::new(&self.shape, &self.strides, self.offset, axes)
    }

    /// The copy of this view's elements to `to`, for an assembly or a write.
    pub(crate) fn region<'r>(&'r self, to: Strided<'r>) -> Region<'r, B::Storage<T>> {
        Region {
            source: self.data.storage(),
            shape: &self.shape,
            from: Strided {
                offset: self.offset,
                strides: &self.strides,
            },
            to,
        }
    }
}

impl<T, const N: usize> Index<[usize; N]> for TensorView<'_, T, Host> {
    type Output = T;

    /// The element at a full index, one coordinate per axis.
    #[track_caller]
    fn index(&self, index: [usize; N]) -> &T {
        &self.data.data()[expect_position(&self.shape, &self.strides, self.offset, &index)]
    }
}
