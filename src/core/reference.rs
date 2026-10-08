use std::hash::Hash;
use std::marker::PhantomData;
use std::ptr::NonNull;

#[derive(Debug)]
pub struct MutableRef<'a, T> {
    ptr: NonNull<T>,
    phantom: PhantomData<&'a T>,
}

impl<'a, T> MutableRef<'a, T> {
    pub fn of(reference: &'a mut T) -> Self {
        Self {
            ptr: NonNull::from_mut(reference),
            phantom: PhantomData,
        }
    }
}

impl<'a, T> AsMut<T> for MutableRef<'a, T> {
    fn as_mut(&mut self) -> &mut T {
        unsafe { self.ptr.as_mut() }
    }
}

impl<'a, T> AsRef<T> for MutableRef<'a, T> {
    fn as_ref(&self) -> &T {
        unsafe { self.ptr.as_ref() }
    }
}

impl<'a, T> Clone for MutableRef<'a, T> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr.clone(),
            phantom: self.phantom.clone(),
        }
    }
}

impl<'a, T> Copy for MutableRef<'a, T> {}

impl<'a, T> PartialEq for MutableRef<'a, T> {
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr && self.phantom == other.phantom
    }
}

impl<'a, T> Eq for MutableRef<'a, T> {}

impl<'a, T> Hash for MutableRef<'a, T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.ptr.hash(state);
    }
}
