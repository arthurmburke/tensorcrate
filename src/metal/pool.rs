//! Recycling shared allocations between operations.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

/// A pool of reusable Metal buffers (recycled across calls to avoid repeated
/// allocation). A buffer is reused when it is at least as large as requested.
#[derive(Default)]
pub(super) struct Pool {
    /// Safe to hand out: every command buffer that could have touched these has
    /// completed.
    pub(super) free: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Released while GPU work was still queued, so a pending kernel may yet
    /// read or write them. [`sync`](super::sync::sync) promotes these into `free`.
    ///
    /// Recycling one of these early is not a use-after-free — a command buffer
    /// retains the resources it references — but it would let an upload, or the
    /// next kernel, race the writes still owed to the previous owner.
    pub(super) retiring: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

impl Pool {
    pub(super) fn acquire(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        len: usize,
    ) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>> {
        if let Some(pos) = self.free.iter().position(|b| b.length() >= len) {
            return Some(self.free.swap_remove(pos));
        }
        device.newBufferWithLength_options(len.max(1), MTLResourceOptions::StorageModeShared)
    }

    /// Take an allocation back. `work_in_flight` says whether any command buffer
    /// has been committed but not yet waited on; if so the allocation waits for
    /// the next [`sync`](super::sync::sync) before it can be handed out again.
    pub(super) fn release(
        &mut self,
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        work_in_flight: bool,
    ) {
        const CAP: usize = 12;
        let bucket = if work_in_flight {
            &mut self.retiring
        } else {
            &mut self.free
        };
        if bucket.len() < CAP {
            bucket.push(buffer);
        }
    }

    /// Every queued command buffer has completed, so anything held back is now
    /// safe to reuse.
    pub(super) fn retire(&mut self) {
        self.free.append(&mut self.retiring);
    }
}
