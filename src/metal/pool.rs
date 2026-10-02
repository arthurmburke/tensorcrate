//! Recycling shared allocations between operations.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

/// How many allocations each list keeps; more are released to Metal.
///
/// Operations queued ahead of the GPU each hold their outputs until their
/// command buffer completes, and a pipelined loop of small operations has
/// hundreds of those in flight: a list shorter than that recycles nothing, and
/// every operation makes a new allocation, which costs more than a small kernel.
const CAP: usize = 512;

/// The most bytes of idle allocations kept for reuse; more are released to
/// Metal. Allocations still owed to queued work are bounded instead by
/// [`Gpu::acquire`](super::device::Gpu) waiting for the oldest of them.
const IDLE_BYTES: usize = 128 << 20;

/// A pool of reusable Metal buffers (recycled across calls to avoid repeated
/// allocation). A request takes the smallest free buffer that holds it, as long
/// as that buffer is not wastefully larger.
#[derive(Default)]
pub(super) struct Pool {
    /// Safe to hand out: every command buffer that could have touched these has
    /// completed. Each with its length, oldest first.
    free: Vec<(usize, Buffer)>,
    /// The bytes in `free`.
    free_bytes: usize,
    /// Released while GPU work was still queued, each with the sequence number of
    /// the last command buffer committed at the time — a pending kernel up to and
    /// including that one may yet read or write it — and its length.
    /// [`promote`](Self::promote) moves an allocation into `free` once its
    /// command buffer has completed.
    ///
    /// Recycling one of these early is not a use-after-free — a command buffer
    /// retains the resources it references — but it would let an upload, or the
    /// next kernel, race the writes still owed to the previous owner.
    retiring: Vec<(u64, usize, Buffer)>,
}

/// The largest allocation a request for `len` bytes may take: a tiny tensor
/// must not pin a huge allocation, which the next large request would then
/// have to make afresh.
fn limit(len: usize) -> usize {
    len.saturating_mul(4).max(len + (64 << 10))
}

impl Pool {
    /// A free allocation that holds `len` bytes, if there is one worth using.
    pub(super) fn take(&mut self, len: usize) -> Option<Buffer> {
        let best = self
            .free
            .iter()
            .enumerate()
            .filter(|(_, (length, _))| (len..=limit(len)).contains(length))
            .min_by_key(|(_, (length, _))| *length)
            .map(|(index, _)| index)?;
        let (length, buffer) = self.free.remove(best);
        self.free_bytes -= length;
        Some(buffer)
    }

    /// The oldest fence among the allocations waiting to retire that would
    /// satisfy a request for `len` bytes.
    pub(super) fn oldest_fitting(&self, len: usize) -> Option<u64> {
        self.retiring
            .iter()
            .filter(|(_, length, _)| (len..=limit(len)).contains(length))
            .map(|&(fence, ..)| fence)
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
        let length = buffer.length();
        match fence {
            Some(fence) => {
                if self.retiring.len() >= CAP {
                    self.retiring.remove(0);
                }
                self.retiring.push((fence, length, buffer));
            }
            None => self.release_free(length, buffer),
        }
    }

    fn release_free(&mut self, length: usize, buffer: Buffer) {
        self.free.push((length, buffer));
        self.free_bytes += length;
        while self.free.len() > CAP || (self.free_bytes > IDLE_BYTES && self.free.len() > 1) {
            let (length, _) = self.free.remove(0);
            self.free_bytes -= length;
        }
    }

    /// Every command buffer up to and including `completed` has finished, so the
    /// allocations they held back are safe to reuse.
    pub(super) fn promote(&mut self, completed: u64) {
        let mut index = 0;
        while index < self.retiring.len() {
            if self.retiring[index].0 <= completed {
                // `remove`, not `swap_remove`: the lists stay oldest first.
                let (_, length, buffer) = self.retiring.remove(index);
                self.release_free(length, buffer);
            } else {
                index += 1;
            }
        }
    }
}
