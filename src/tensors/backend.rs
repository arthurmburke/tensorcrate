//! Where a tensor's elements live.
//!
//! [`Vector`](super::Vector), [`Matrix`](super::Matrix) and
//! [`Tensor`](super::Tensor) carry a backend type parameter that selects their
//! storage. It defaults to [`Host`] — the flat
//! row-major `Vec` they normally use — so `Vector<f64>` and `Matrix<i64>` keep
//! meaning exactly what they meant before.
//!
//! The [`Metal`] backend (macOS, `metal` feature) instead keeps elements in
//! `MTLStorageModeShared` memory, which the GPU kernels read and write in place.
//! The full kernel set is compiled for `f32`, `f16` and `bf16` — the
//! [`MetalElement`](crate::metal::MetalElement) types — so
//! [`Kernels<T>`](super::Kernels) is implemented for `Metal` at each of them,
//! where `Host` implements it for every float type (`f64` included). M5 GPUs
//! run all three types' matrix products on TensorOps. A chain of operations over `Metal`-backed
//! tensors therefore stays resident: no operand is uploaded and no result is
//! downloaded until you ask for one. The [`Host`] backend never moves work to
//! the GPU on its own; choosing a backend is the only thing that does.
//!
//! Move between backends explicitly with
//! [`Vector::to_backend`](super::Vector::to_backend) and
//! [`Matrix::to_backend`](super::Matrix::to_backend):
//!
//! ```
//! # #[cfg(all(feature = "metal", target_os = "macos"))] {
//! use tensorcrate::tensors::{Host, Matrix, Metal};
//!
//! let m: Matrix<f32> = Matrix::from_rows([[1.0, 2.0], [3.0, 4.0]]);
//! let resident = m.to_backend::<Metal>(); // one upload
//! let cubed = resident.matmul(&resident).matmul(&resident); // no copies
//! assert_eq!(cubed.to_backend::<Host>(), m.matmul(&m).matmul(&m));
//! # }
//! ```
//!
//! The backend never implicitly changes an element type. Compact matrices offer
//! both same-format [`Matrix::matmul`](super::Matrix::matmul) output and an
//! explicit `matmul_f32` widening path.
//!
//! # Storage and shape
//!
//! A backend knows one thing: [`Backend::Storage`], a flat, dense run of
//! values. It is shapeless — the extents live in the [`Vector`](super::Vector),
//! [`Matrix`](super::Matrix) and [`Tensor`](super::Tensor) wrappers laid over
//! it. That is why reshaping, or converting a vector to a matrix, takes no
//! backend call at all: the same storage is simply given a different shape, a
//! move rather than a copy on either backend.
//!
//! Metal operations never fall back to Host execution. If a device, allocation,
//! or kernel is unavailable, the operation panics. Transfer to [`Host`]
//! explicitly when CPU execution is intended.
//!
//! Anything that is not one dense run is several storages. A compressed sparse
//! row matrix, for instance, is its stored values plus two index arrays, each a
//! `Storage` on whichever backend the matrix lives on; it needs nothing from the
//! backend that a dense tensor does not. Joining shapes —
//! [`Matrix::vstack`](super::Matrix::vstack),
//! [`Matrix::hstack`](super::Matrix::hstack), [`Vector::vstack`](super::Vector::vstack),
//! [`Tensor::concat`](super::Tensor::concat) — likewise belongs to the shaped
//! types: each describes where its operands land with
//! [`assemble`](Backend::assemble), and the backend only copies.
//!
//! The one copy that takes a whole layout is the strided copy behind
//! [`Tensor`](super::Tensor) and [`TensorView`](super::TensorView): up to
//! [`MAX_RANK`] axes, each with its own stride, read from one storage and
//! written row-major into a new one, into part of a new one, or into part of
//! an existing one. It moves bits, so it is defined for every element type —
//! on `Metal` as one GPU dispatch for any element of 1, 2, 4, 8 or 16 bytes,
//! integer index types included.

use super::fused::Place;
use super::layout::{MAX_RANK, element_count, for_each_run, reach};

/// A tensor storage backend.
///
/// The trait is sealed: [`Host`] and [`Metal`] are the only implementations,
/// since the kernels behind them are part of this crate.
pub trait Backend: sealed::Sealed + Sized + 'static {
    /// A flat, dense run of `T`, in whatever memory this backend computes in.
    ///
    /// This is the only storage the backend knows about. Shape is not part of
    /// it: [`Vector`](super::Vector), [`Matrix`](super::Matrix) and
    /// [`Tensor`](super::Tensor) are extents laid over one `Storage`, which is
    /// why reshaping them is a move. A layout that is not one dense run — a
    /// compressed sparse row matrix, say — is several `Storage`s side by side
    /// (its values and its index arrays), each of which any backend can hold.
    type Storage<T>;

    /// Build storage from typed values.
    fn store<T: Copy + 'static>(values: &[T]) -> Self::Storage<T>;

    /// Take ownership of values as storage — a move on [`Host`], one upload on
    /// [`Metal`].
    fn from_vec<T: Copy + 'static>(values: Vec<T>) -> Self::Storage<T>;

    /// Borrow this storage as a flat typed slice.
    ///
    /// Nothing is copied: for the [`Metal`] backend this borrows the shared
    /// allocation itself, which the CPU can read directly.
    fn as_slice<T: Copy + 'static>(storage: &Self::Storage<T>) -> &[T];

    /// Borrow storage mutably. For [`Metal`] this waits for queued GPU work
    /// first, since a kernel may still be writing the allocation.
    #[doc(hidden)]
    fn as_mut_slice<T: Copy + 'static>(storage: &mut Self::Storage<T>) -> &mut [T];

    /// A second copy of some storage on this same backend — what
    /// [`Vector::to_backend`](super::Vector::to_backend) does when it is asked
    /// for the backend it is already on. On [`Metal`] the copy runs on the GPU
    /// and nothing waits for it.
    #[doc(hidden)]
    fn duplicate<T: Copy + 'static>(storage: &Self::Storage<T>) -> Self::Storage<T>;

    /// The `rows × cols` matrix whose element `(r, c)` is element
    /// `place.at(r, c)` of `storage`: a view, a broadcast or a permutation of
    /// it, copied into order. Every element is copied exactly. This is
    /// [`strided_copy`](Self::strided_copy) over the axes `place` walks.
    #[doc(hidden)]
    fn gather<T: Copy + 'static>(
        storage: &Self::Storage<T>,
        place: Place,
        (_, cols): (usize, usize),
    ) -> Self::Storage<T> {
        let (dims, steps) = place.layout(cols);
        let from = Strided {
            offset: place.offset,
            strides: &steps,
        };
        Self::strided_copy(storage, &dims, from)
    }

    /// The elements of `storage` that a layout of `shape` reads at `from`,
    /// copied into row-major order: a view, a permutation, a slice or a
    /// broadcast of it made contiguous. Every element is copied exactly — by
    /// its bits, so any element type is copied on either backend.
    ///
    /// # Panics
    ///
    /// If `shape` has more than [`MAX_RANK`](super::MAX_RANK) axes, `from`
    /// has a different number of strides, or the layout reads past `storage`.
    #[doc(hidden)]
    fn strided_copy<T: Copy + 'static>(
        storage: &Self::Storage<T>,
        shape: &[usize],
        from: Strided<'_>,
    ) -> Self::Storage<T>;

    /// Storage of `len` elements made by copying each region into it: region
    /// `r` copies `r.shape` from `r.source` at `r.from` to the new storage at
    /// `r.to`. The regions are expected to cover the storage — a concatenation
    /// is one region per operand — and any element none of them writes holds
    /// an unspecified value of `T`.
    ///
    /// # Panics
    ///
    /// As for [`strided_copy`](Self::strided_copy), for any region.
    #[doc(hidden)]
    fn assemble<T: Copy + 'static>(
        len: usize,
        regions: &[Region<'_, Self::Storage<T>>],
    ) -> Self::Storage<T>;

    /// Copy `region` into `target` in place, leaving every element the region
    /// does not write as it was — the write half of a strided copy, for
    /// updating part of a tensor such as one position of a cache.
    ///
    /// # Panics
    ///
    /// As for [`strided_copy`](Self::strided_copy), on either side.
    #[doc(hidden)]
    fn strided_write<T: Copy + 'static>(
        target: &mut Self::Storage<T>,
        region: Region<'_, Self::Storage<T>>,
    );
}

mod sealed {
    pub trait Sealed {}
}

/// `storage` copied onto backend `B2` — by [`Backend::duplicate`] when `B2` is
/// the backend it is already on, and otherwise through a slice of it.
pub(crate) fn transfer<T: Copy + 'static, B: Backend, B2: Backend>(
    storage: &B::Storage<T>,
) -> B2::Storage<T> {
    if std::any::TypeId::of::<B>() == std::any::TypeId::of::<B2>() {
        let copy = std::mem::ManuallyDrop::new(B::duplicate(storage));
        // SAFETY: `B` and `B2` are one type, so `B::Storage<T>` and
        // `B2::Storage<T>` are too; the copy is moved, not duplicated, because
        // the original is never dropped.
        return unsafe { std::mem::transmute_copy::<B::Storage<T>, B2::Storage<T>>(&copy) };
    }
    B2::store(B::as_slice(storage))
}

/// One side of a strided copy: over a copied shape, element `[i₀, …, iₙ₋₁]`
/// is storage element `offset + Σ iₖ·strides[k]`.
#[doc(hidden)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Strided<'a> {
    pub offset: usize,
    pub strides: &'a [usize],
}

/// One block of a [`Backend::assemble`] or [`Backend::strided_write`]:
/// `shape` copied from `source` at `from` to the target at `to`.
#[doc(hidden)]
#[derive(Debug)]
pub struct Region<'a, S: ?Sized> {
    pub source: &'a S,
    pub shape: &'a [usize],
    pub from: Strided<'a>,
    pub to: Strided<'a>,
}

impl<S: ?Sized> Clone for Region<'_, S> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<S: ?Sized> Copy for Region<'_, S> {}

/// Panics unless a layout of `shape` at `at` stays inside `len` elements of
/// storage; returns the number of elements the layout covers.
#[track_caller]
pub(crate) fn check_strided(
    len: usize,
    shape: &[usize],
    at: Strided<'_>,
    operation: &str,
) -> usize {
    assert!(
        shape.len() <= MAX_RANK,
        "{operation}: shape {shape:?} has more than {MAX_RANK} axes"
    );
    assert_eq!(
        shape.len(),
        at.strides.len(),
        "{operation}: shape {shape:?} has {} axes but {} strides",
        shape.len(),
        at.strides.len()
    );
    let count = element_count(shape, operation);
    let end = reach(shape, at.offset, at.strides).unwrap_or(usize::MAX);
    assert!(
        end <= len,
        "{operation}: shape {shape:?} with strides {:?} from offset {} reaches past {len} elements",
        at.strides,
        at.offset
    );
    count
}

/// [`Backend::strided_copy`] over a slice.
pub(crate) fn strided_copy_slice<T: Copy>(
    values: &[T],
    shape: &[usize],
    from: Strided<'_>,
) -> Vec<T> {
    let count = check_strided(values.len(), shape, from, "strided copy");
    let mut out = Vec::with_capacity(count);
    // The output is row-major, so the walk's order is the order to write in.
    for_each_run(
        shape,
        [from.offset],
        [from.strides],
        |[start], len, [step]| match step {
            1 => out.extend_from_slice(&values[start..start + len]),
            0 => out.extend(std::iter::repeat_n(values[start], len)),
            step => out.extend((0..len).map(|index| values[start + index * step])),
        },
    );
    out
}

/// [`Backend::strided_write`] over slices.
pub(crate) fn strided_write_slice<T: Copy>(target: &mut [T], region: Region<'_, [T]>) {
    let Region {
        source,
        shape,
        from,
        to,
    } = region;
    check_strided(source.len(), shape, from, "strided write source");
    check_strided(target.len(), shape, to, "strided write target");
    for_each_run(
        shape,
        [from.offset, to.offset],
        [from.strides, to.strides],
        |[start, target_start], len, [step, target_step]| match (step, target_step) {
            (1, 1) => target[target_start..target_start + len]
                .copy_from_slice(&source[start..start + len]),
            _ => {
                for index in 0..len {
                    target[target_start + index * target_step] = source[start + index * step];
                }
            }
        },
    );
}

/// [`Backend::assemble`] over slices.
pub(crate) fn assemble_slices<T: Copy>(len: usize, regions: &[Region<'_, [T]>]) -> Vec<T> {
    // Every element is written by some region, so the first element of any
    // region serves as the initial value: it keeps the fill exact without
    // asking `T` for a zero.
    let fill = regions.iter().find_map(|region| {
        let count = check_strided(region.source.len(), region.shape, region.from, "assemble");
        (count != 0).then(|| region.source[region.from.offset])
    });
    let Some(fill) = fill else {
        assert_eq!(
            len, 0,
            "assemble: {len} elements but no region to fill them"
        );
        return Vec::new();
    };
    let mut out = vec![fill; len];
    for &region in regions {
        strided_write_slice(&mut out, region);
    }
    out
}

/// `region` reading a `Vec` as reading its slice.
fn as_slices<'a, T>(region: &Region<'a, Vec<T>>) -> Region<'a, [T]> {
    Region {
        source: region.source.as_slice(),
        shape: region.shape,
        from: region.from,
        to: region.to,
    }
}

/// The default backend: elements live in a flat row-major [`Vec`].
///
/// Every element type is supported.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Host;

impl sealed::Sealed for Host {}

impl Backend for Host {
    type Storage<T> = Vec<T>;

    fn store<T: Copy + 'static>(values: &[T]) -> Vec<T> {
        values.to_vec()
    }

    fn from_vec<T: Copy + 'static>(values: Vec<T>) -> Vec<T> {
        values
    }

    fn as_slice<T: Copy + 'static>(storage: &Vec<T>) -> &[T] {
        storage
    }

    fn as_mut_slice<T: Copy + 'static>(storage: &mut Vec<T>) -> &mut [T] {
        storage
    }

    fn duplicate<T: Copy + 'static>(storage: &Vec<T>) -> Vec<T> {
        storage.clone()
    }

    fn strided_copy<T: Copy + 'static>(
        storage: &Vec<T>,
        shape: &[usize],
        from: Strided<'_>,
    ) -> Vec<T> {
        strided_copy_slice(storage, shape, from)
    }

    fn assemble<T: Copy + 'static>(len: usize, regions: &[Region<'_, Vec<T>>]) -> Vec<T> {
        let regions = regions.iter().map(as_slices).collect::<Vec<_>>();
        assemble_slices(len, &regions)
    }

    fn strided_write<T: Copy + 'static>(target: &mut Vec<T>, region: Region<'_, Vec<T>>) {
        strided_write_slice(target, as_slices(&region));
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub use gpu::{Metal, MetalStorage};

#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) use gpu::require_metal;

#[cfg(all(feature = "metal", target_os = "macos"))]
mod gpu {
    use std::any::TypeId;
    use std::fmt;

    use half::{bf16, f16};

    use super::{Backend, Region, Strided, check_strided, sealed};
    use crate::metal::{MetalBuffer, MetalElement};

    use crate::tensors::{Analytic, Axis, BinaryOp, Compare, Family, Reduce, SortOrder, Statistic};

    /// A backend that keeps elements in GPU-shared memory, so the Metal kernels
    /// read and write them in place.
    ///
    /// `f32`, `f16` and `bf16` — the [`MetalElement`] types — each have the
    /// complete Metal operation set, with every kernel compiled for that type.
    /// The 16-bit types are the crate's re-exported
    /// [`half`](https://docs.rs/half) types.
    ///
    /// Metal objects are thread-affine, so these tensors are not `Send`. They do
    /// clone (into a fresh allocation with the same values) and compare by
    /// value.
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
    pub struct Metal;

    impl sealed::Sealed for Metal {}

    /// Require an operation selected through the Metal backend to have encoded
    /// a Metal kernel. Host execution is only available through an explicit
    /// backend transfer.
    #[track_caller]
    pub(crate) fn require_metal<T>(operation: &str, result: Option<T>) -> T {
        result.unwrap_or_else(|| {
            panic!("Metal {operation} could not execute on the GPU; Host fallback is disabled")
        })
    }

    impl Backend for Metal {
        type Storage<T> = MetalStorage<T>;

        fn store<T: Copy + 'static>(values: &[T]) -> MetalStorage<T> {
            MetalStorage::from_slice(values)
        }

        fn from_vec<T: Copy + 'static>(values: Vec<T>) -> MetalStorage<T> {
            MetalStorage::from_slice(&values)
        }

        fn as_slice<T: Copy + 'static>(storage: &MetalStorage<T>) -> &[T] {
            storage.as_slice()
        }

        fn as_mut_slice<T: Copy + 'static>(storage: &mut MetalStorage<T>) -> &mut [T] {
            storage.as_mut_slice()
        }

        fn duplicate<T: Copy + 'static>(storage: &MetalStorage<T>) -> MetalStorage<T> {
            storage.duplicate()
        }

        // A strided copy moves bits, so the GPU copies every element type whose
        // width the kernel is instantiated for — not only the `MetalElement`s,
        // so index tensors move on the device too.
        fn strided_copy<T: Copy + 'static>(
            storage: &MetalStorage<T>,
            shape: &[usize],
            from: Strided<'_>,
        ) -> MetalStorage<T> {
            check_strided(storage.len(), shape, from, "strided copy");
            let buffer = require_metal(
                "strided copy",
                storage
                    .device()
                    .and_then(|buffer| buffer.strided_copy(shape, from)),
            );
            MetalStorage::from_device(buffer)
        }

        // The device path leaves an element no region writes as whatever the
        // recycled allocation held, so it is available only for types where
        // every bit pattern is a value.
        fn assemble<T: Copy + 'static>(
            len: usize,
            regions: &[Region<'_, MetalStorage<T>>],
        ) -> MetalStorage<T> {
            for region in regions {
                check_strided(region.source.len(), region.shape, region.from, "assemble");
                check_strided(len, region.shape, region.to, "assemble");
            }
            let device = regions
                .iter()
                .map(|region| {
                    Some(Region {
                        source: region.source.device()?,
                        shape: region.shape,
                        from: region.from,
                        to: region.to,
                    })
                })
                .collect::<Option<Vec<_>>>();
            let buffer = require_metal(
                "assemble",
                device
                    .filter(|_| any_bits_valid::<T>())
                    .and_then(|device| MetalBuffer::assemble(len, &device)),
            );
            MetalStorage::from_device(buffer)
        }

        fn strided_write<T: Copy + 'static>(
            target: &mut MetalStorage<T>,
            region: Region<'_, MetalStorage<T>>,
        ) {
            check_strided(
                region.source.len(),
                region.shape,
                region.from,
                "strided write",
            );
            check_strided(target.len(), region.shape, region.to, "strided write");
            let encoded = if let Some(source) = region.source.device()
                && let Residency::Device(buffer) = &mut target.0
            {
                buffer.strided_write(Region {
                    source,
                    shape: region.shape,
                    from: region.from,
                    to: region.to,
                })
            } else {
                None
            };
            require_metal("strided write", encoded);
        }
    }

    /// Whether every bit pattern of `T`'s width is a value of `T`: the
    /// primitive numbers.
    fn any_bits_valid<T: 'static>() -> bool {
        [
            TypeId::of::<f32>(),
            TypeId::of::<f64>(),
            TypeId::of::<f16>(),
            TypeId::of::<bf16>(),
            TypeId::of::<u8>(),
            TypeId::of::<u16>(),
            TypeId::of::<u32>(),
            TypeId::of::<u64>(),
            TypeId::of::<i8>(),
            TypeId::of::<i16>(),
            TypeId::of::<i32>(),
            TypeId::of::<i64>(),
            TypeId::of::<(u64, u64)>(),
        ]
        .contains(&TypeId::of::<T>())
    }

    /// The allocation behind a [`Metal`]-backed tensor.
    ///
    /// Normally this is GPU-shared memory that kernels read in place. If the
    /// process has no Metal device, values are retained temporarily so they can
    /// still be transferred explicitly to [`Host`](super::Host). Attempting a
    /// Metal operation on such storage panics rather than running on the CPU.
    pub struct MetalStorage<T = f32>(Residency<T>);

    enum Residency<T> {
        /// A shared allocation the GPU can read without a copy.
        Device(MetalBuffer<T>),
        /// No Metal device was available. Metal operations reject this storage.
        Host(Vec<T>),
    }

    impl<T: Copy + 'static> MetalStorage<T> {
        pub(crate) fn from_slice(values: &[T]) -> Self {
            match MetalBuffer::from_slice(values) {
                Some(buffer) => Self(Residency::Device(buffer)),
                None => Self(Residency::Host(values.to_vec())),
            }
        }

        /// A second copy on the same backend. A resident allocation is copied on
        /// the GPU, queued like any kernel, so nothing waits.
        pub(crate) fn duplicate(&self) -> Self {
            match &self.0 {
                Residency::Device(buffer) => Self(Residency::Device(require_metal(
                    "storage duplicate",
                    buffer.duplicate(),
                ))),
                Residency::Host(_) => require_metal("storage duplicate", None),
            }
        }

        /// Whether the values really are in GPU-shared memory.
        ///
        /// `false` means no Metal device was available. Metal operations on the
        /// storage will panic instead of executing on the CPU.
        pub fn is_device_resident(&self) -> bool {
            matches!(self.0, Residency::Device(_))
        }

        /// Number of stored values.
        pub fn len(&self) -> usize {
            match &self.0 {
                Residency::Device(buffer) => buffer.len(),
                Residency::Host(values) => values.len(),
            }
        }

        /// Whether this allocation holds no values.
        pub fn is_empty(&self) -> bool {
            self.len() == 0
        }

        /// Borrow the values as a flat slice, without copying: shared storage is
        /// ordinary cached memory as far as the CPU is concerned.
        pub(crate) fn as_slice(&self) -> &[T] {
            match &self.0 {
                Residency::Device(buffer) => buffer.as_slice(),
                Residency::Host(values) => values,
            }
        }

        /// Borrow the values mutably. A device allocation waits for queued
        /// GPU work first.
        pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
            match &mut self.0 {
                Residency::Device(buffer) => buffer.as_mut_slice(),
                Residency::Host(values) => values,
            }
        }

        /// The shared allocation, when there is one.
        pub(crate) fn device(&self) -> Option<&MetalBuffer<T>> {
            match &self.0 {
                Residency::Device(buffer) => Some(buffer),
                Residency::Host(_) => None,
            }
        }

        /// Wrap an allocation a kernel has written.
        pub(crate) fn from_device(buffer: MetalBuffer<T>) -> Self {
            Self(Residency::Device(buffer))
        }
    }

    /// The resident operations, for every element type the kernels are compiled
    /// for. Scalars and results are in `T`; the folds that end on the CPU
    /// return their unrounded `f32` accumulator.
    impl<T: MetalElement> MetalStorage<T> {
        pub(crate) fn transpose(&self, rows: usize, cols: usize) -> Option<Self> {
            Some(Self(Residency::Device(
                self.device()?.transpose(rows, cols)?,
            )))
        }

        /// `C[m×n] = A[m×k] · B[k×n]`, entirely on the GPU. `None` when either
        /// operand is not device-resident or the dispatch failed.
        pub(crate) fn matmul(&self, rhs: &Self, m: usize, k: usize, n: usize) -> Option<Self> {
            let product = self.device()?.matmul(rhs.device()?, m, k, n)?;
            Some(Self(Residency::Device(product)))
        }

        /// Elementwise operation on the GPU. The shaders have no remainder
        /// kernel, so [`BinaryOp::Rem`] returns `None`.
        pub(crate) fn elementwise(&self, rhs: &Self, op: BinaryOp) -> Option<Self> {
            let output = self.device()?.elementwise(rhs.device()?, op)?;
            Some(Self(Residency::Device(output)))
        }

        /// Scalar broadcast on the GPU, with `op` encoded as in
        /// [`elementwise`](Self::elementwise).
        pub(crate) fn broadcast(&self, scalar: T, op: BinaryOp, scalar_left: bool) -> Option<Self> {
            let output = self.device()?.broadcast(scalar, op, scalar_left)?;
            Some(Self(Residency::Device(output)))
        }

        /// `target += A[m×k]·B[k×n]`, accumulated by the matmul kernel itself.
        ///
        /// `None` when any of the three is not device-resident or dispatch fails.
        pub(crate) fn matmul_accumulate(
            &self,
            rhs: &Self,
            target: &mut Self,
            m: usize,
            k: usize,
            n: usize,
        ) -> Option<()> {
            let (a, b) = (self.device()?, rhs.device()?);
            let Residency::Device(target) = &mut target.0 else {
                return None;
            };
            a.matmul_accumulate(b, target, m, k, n)
        }

        /// `target += op(A)·op(B)`, reading the operand `transposed` names
        /// transposed where it lies — see
        /// [`MetalBuffer::matmul_transposed_accumulate`](crate::metal::MetalBuffer).
        ///
        /// `None` when any of the three is not device-resident, or the GPU
        /// cannot read an operand transposed.
        pub(crate) fn matmul_transposed_accumulate(
            &self,
            rhs: &Self,
            target: &mut Self,
            transposed: crate::tensors::Transposed,
            shape: (usize, usize, usize),
        ) -> Option<()> {
            let (a, b) = (self.device()?, rhs.device()?);
            let Residency::Device(target) = &mut target.0 else {
                return None;
            };
            a.matmul_transposed_accumulate(b, target, transposed, shape)
        }

        /// Analytic function applied to a value/tangent pair, in one dispatch.
        pub(crate) fn unary_dual(&self, tangent: &Self, op: Analytic) -> Option<(Self, Self)> {
            let (value, tangent) = self.device()?.unary_dual(tangent.device()?, op)?;
            Some((
                Self(Residency::Device(value)),
                Self(Residency::Device(tangent)),
            ))
        }

        /// Valid cross-correlation, on the GPU.
        pub(crate) fn correlate(
            &self,
            weights: &Self,
            rows: usize,
            cols: usize,
            window_rows: usize,
            window_cols: usize,
            flip: bool,
        ) -> Option<Self> {
            let output = self.device()?.correlate(
                weights.device()?,
                rows,
                cols,
                window_rows,
                window_cols,
                flip,
            )?;
            Some(Self(Residency::Device(output)))
        }

        /// Reversal of both axes, on the GPU.
        pub(crate) fn flip(&self, rows: usize, cols: usize) -> Option<Self> {
            Some(Self(Residency::Device(self.device()?.flip(rows, cols)?)))
        }

        /// Zero padding, on the GPU.
        pub(crate) fn pad(
            &self,
            rows: usize,
            cols: usize,
            pad_rows: usize,
            pad_cols: usize,
        ) -> Option<Self> {
            let output = self.device()?.pad(rows, cols, pad_rows, pad_cols)?;
            Some(Self(Residency::Device(output)))
        }

        /// `op(a, b)` over `shape`, each operand read in place through its
        /// layout, on the GPU.
        pub(crate) fn strided_binary(
            &self,
            a: Strided<'_>,
            rhs: &Self,
            b: Strided<'_>,
            shape: &[usize],
            op: crate::tensors::kernels::Pairwise,
        ) -> Option<Self> {
            let output = self
                .device()?
                .strided_binary(a, rhs.device()?, b, shape, op)?;
            Some(Self(Residency::Device(output)))
        }

        /// The fold, mean or variance of each slice of `split`, on the GPU.
        pub(crate) fn reduce_axes(
            &self,
            split: &crate::tensors::layout::Split,
            op: crate::tensors::kernels::AxisReduction,
        ) -> Option<Self> {
            Some(Self(Residency::Device(
                self.device()?.reduce_axes(split, op)?,
            )))
        }

        /// The position of each slice's extreme along its one folded axis,
        /// on the GPU.
        pub(crate) fn arg_reduce(
            &self,
            split: &crate::tensors::layout::Split,
            op: Reduce,
        ) -> Option<MetalStorage<u32>> {
            Some(MetalStorage(Residency::Device(
                self.device()?.arg_reduce(split, op)?,
            )))
        }

        /// Elementwise comparison with another shared allocation.
        pub(crate) fn compare(&self, rhs: &Self, op: Compare) -> Option<Self> {
            let output = self.device()?.compare(rhs.device()?, op)?;
            Some(Self(Residency::Device(output)))
        }

        /// Elementwise comparison against a scalar.
        pub(crate) fn compare_scalar(
            &self,
            scalar: T,
            op: Compare,
            scalar_left: bool,
        ) -> Option<Self> {
            let output = self.device()?.compare_scalar(scalar, op, scalar_left)?;
            Some(Self(Residency::Device(output)))
        }

        /// Elementwise clamp to `[low, high]`.
        pub(crate) fn clamp(&self, low: T, high: T) -> Option<Self> {
            Some(Self(Residency::Device(self.device()?.clamp(low, high)?)))
        }

        /// Whole-buffer fold. The result is a number rather than an allocation,
        /// so unlike its neighbours this one ends on the CPU by construction.
        /// It is the `f32` accumulator, unrounded.
        pub(crate) fn reduce(&self, op: Reduce) -> Option<f32> {
            self.device()?.reduce(op)
        }

        /// Inclusive prefix sum, staying in shared memory.
        pub(crate) fn prefix_sum(&self) -> Option<Self> {
            Some(Self(Residency::Device(self.device()?.prefix_sum()?)))
        }

        /// Sort in IEEE total order, staying in shared memory.
        pub(crate) fn sort(&self, order: SortOrder) -> Option<Self> {
            Some(Self(Residency::Device(self.device()?.sort(order)?)))
        }

        /// Elementwise `self^rhs`.
        pub(crate) fn power(&self, rhs: &Self) -> Option<Self> {
            Some(Self(Residency::Device(
                self.device()?.power(rhs.device()?)?,
            )))
        }

        /// Elementwise power with one operand fixed.
        pub(crate) fn power_scalar(&self, scalar: T, scalar_left: bool) -> Option<Self> {
            let output = self.device()?.power_scalar(scalar, scalar_left)?;
            Some(Self(Residency::Device(output)))
        }

        /// Analytic function applied elementwise.
        pub(crate) fn unary(&self, op: Analytic) -> Option<Self> {
            Some(Self(Residency::Device(self.device()?.unary(op)?)))
        }

        /// `Σ(xᵢ − mean)²` over the whole allocation. Like
        /// [`reduce`](Self::reduce) this ends on the CPU, since its result is a
        /// number.
        pub(crate) fn sum_squared_deviations(&self, mean: f32) -> Option<f32> {
            self.device()?.sum_squared_deviations(mean)
        }

        /// Per-row or per-column means and deviation sums, both staying in
        /// shared memory.
        pub(crate) fn axis_moments(
            &self,
            rows: usize,
            cols: usize,
            axis: Axis,
        ) -> Option<(Self, Self)> {
            let (means, deviations) = self.device()?.axis_moments(rows, cols, axis)?;
            Some((
                Self(Residency::Device(means)),
                Self(Residency::Device(deviations)),
            ))
        }

        /// A distribution function applied elementwise.
        pub(crate) fn distribution(
            &self,
            family: Family,
            statistic: Statistic,
            parameters: (f32, f32),
        ) -> Option<Self> {
            let output = self.device()?.distribution(family, statistic, parameters)?;
            Some(Self(Residency::Device(output)))
        }

        /// The same, with one parameter pair per row or column.
        pub(crate) fn axis_distribution(
            &self,
            first: &Self,
            second: &Self,
            shape: (usize, usize),
            axis: Axis,
            family: Family,
            statistic: Statistic,
        ) -> Option<Self> {
            let output = self.device()?.axis_distribution(
                first.device()?,
                second.device()?,
                shape.0,
                shape.1,
                axis,
                family,
                statistic,
            )?;
            Some(Self(Residency::Device(output)))
        }
    }

    macro_rules! low_precision_storage {
        ($ty:ty) => {
            impl MetalStorage<$ty> {
                pub(crate) fn matmul_f32(
                    &self,
                    rhs: &Self,
                    m: usize,
                    k: usize,
                    n: usize,
                ) -> Option<MetalStorage<f32>> {
                    let product = self.device()?.matmul_f32(rhs.device()?, m, k, n)?;
                    Some(MetalStorage(Residency::Device(product)))
                }
            }
        };
    }

    low_precision_storage!(f16);
    low_precision_storage!(bf16);

    impl<T: Copy + 'static> Clone for MetalStorage<T> {
        /// Clones the values into a fresh allocation on the same backend.
        fn clone(&self) -> Self {
            self.duplicate()
        }
    }

    impl<T: Copy + PartialEq + 'static> PartialEq for MetalStorage<T> {
        fn eq(&self, other: &Self) -> bool {
            self.as_slice() == other.as_slice()
        }
    }

    impl<T: Copy + fmt::Debug + 'static> fmt::Debug for MetalStorage<T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            self.as_slice().fmt(f)
        }
    }
}
