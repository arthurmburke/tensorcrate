//! Recycling shared allocations between operations.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

/// How many allocations each list keeps; more are released to Metal.
const CAP: usize = 16;

/// A pool of reusable Metal buffers (recycled across calls to avoid repeated
/// allocation). A request takes the smallest free buffer that holds it, as long
/// as that buffer is not wastefully larger.
#[derive(Default)]
pub(super) struct Pool {
    /// Safe to hand out: every command buffer that could have touched these has
    /// completed.
    pub(super) free: Vec<Buffer>,
    /// Released while GPU work was still queued, each with the sequence number of
    /// the last command buffer committed at the time: a pending kernel up to and
    /// including that one may yet read or write it. [`promote`](Self::promote)
    /// moves an allocation into `free` once its command buffer has completed.
    ///
    /// Recycling one of these early is not a use-after-free — a command buffer
    /// retains the resources it references — but it would let an upload, or the
    /// next kernel, race the writes still owed to the previous owner.
    pub(super) retiring: Vec<(u64, Buffer)>,
}

impl Pool {
    /// A free allocation that holds `len` bytes, if there is one worth using.
    pub(super) fn take(&mut self, len: usize) -> Option<Buffer> {
        // A tiny tensor must not pin a huge allocation, which the next large
        // request would then have to make afresh.
        let limit = len.saturating_mul(4).max(len + (64 << 10));
        let best = self
            .free
            .iter()
            .enumerate()
            .filter(|(_, b)| (len..=limit).contains(&b.length()))
            .min_by_key(|(_, b)| b.length())
            .map(|(index, _)| index)?;
        Some(self.free.remove(best))
    }

    /// The oldest fence among the allocations waiting to retire that would
    /// satisfy a request for `len` bytes.
    pub(super) fn oldest_fitting(&self, len: usize) -> Option<u64> {
        let limit = len.saturating_mul(4).max(len + (64 << 10));
        self.retiring
            .iter()
            .filter(|(_, b)| (len..=limit).contains(&b.length()))
            .map(|&(fence, _)| fence)
            .min()
    }

    pub(super) fn allocate(device: &ProtocolObject<dyn MTLDevice>, len: usize) -> Option<Buffer> {
        device.newBufferWithLength_options(len.max(1), MTLResourceOptions::StorageModeShared)
    }

    /// Take an allocation back. `fence` is the sequence number of the last
    /// command buffer committed (or still being encoded) when it was released,
    /// or `None` when nothing was in flight; with a fence, the allocation waits
    /// until that command buffer completes before it can be handed out again.
    ///
    /// A full list makes room by letting its oldest allocation go back to
    /// Metal, not by refusing the newcomer: what was released last is what the
    /// work now running will ask for again, while what has sat longest belongs
    /// to work that has moved on — tensors of another size, say, which a
    /// request could never use and would otherwise crowd out the ones it can.
    pub(super) fn release(&mut self, buffer: Buffer, fence: Option<u64>) {
        match fence {
            Some(fence) => push_evicting(&mut self.retiring, (fence, buffer)),
            None => push_evicting(&mut self.free, buffer),
        }
    }

    /// Every command buffer up to and including `completed` has finished, so the
    /// allocations they held back are safe to reuse.
    pub(super) fn promote(&mut self, completed: u64) {
        let mut index = 0;
        while index < self.retiring.len() {
            if self.retiring[index].0 <= completed {
                // `remove`, not `swap_remove`: the lists stay oldest first.
                let (_, buffer) = self.retiring.remove(index);
                push_evicting(&mut self.free, buffer);
            } else {
                index += 1;
            }
        }
    }
}

/// Append to a list kept oldest first, dropping its oldest entry if it is full.
fn push_evicting<T>(list: &mut Vec<T>, item: T) {
    if list.len() >= CAP {
        list.remove(0);
    }
    list.push(item);
}
