//! Submitting command buffers and waiting for them.

use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLComputeCommandEncoder,
};

use super::device::{GPU, Gpu, build_gpu};

/// Operations batched into one command buffer before it is committed.
const BATCH_OPERATIONS: usize = 32;

/// Queued work, in elements touched, past which a batch is committed at once so
/// the GPU can start on it while the CPU carries on.
const BATCH_WORK: usize = 1 << 20;

/// The command buffer operations are encoded into until it is committed.
///
/// A command buffer costs about ten microseconds of CPU time to create and
/// commit, and a dispatch within one well under a microsecond, so committing
/// one per operation made a chain of small operations ten times slower than its
/// encoding. Operations therefore share an open command buffer — and, while
/// they are compute work, one compute encoder — which is committed when
/// [`BATCH_OPERATIONS`] have accumulated, when they add up to enough work to be
/// worth starting ([`BATCH_WORK`]), or when anything waits for the GPU.
///
/// Sharing an encoder changes nothing about ordering: a compute encoder's
/// dispatches run in order, and Metal tracks the hazards between them and
/// between encoders exactly as it does between command buffers.
pub(crate) struct Batch {
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Encoder,
    operations: usize,
    work: usize,
}

enum Encoder {
    None,
    Compute(Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>),
    Blit(Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>),
}

/// A batch abandoned uncommitted — when the thread's device is torn down with
/// work still being encoded — must still end its encoder, which Metal requires
/// of every encoder before it is released.
impl Drop for Batch {
    fn drop(&mut self) {
        self.encoder.end();
    }
}

impl Encoder {
    fn end(&mut self) {
        match std::mem::replace(self, Encoder::None) {
            Encoder::None => {}
            Encoder::Compute(encoder) => encoder.endEncoding(),
            Encoder::Blit(encoder) => encoder.endEncoding(),
        }
    }
}

/// The open batch, starting one if there is none.
fn batch(gpu: &Gpu) -> Option<std::cell::RefMut<'_, Batch>> {
    let mut open = gpu.open.borrow_mut();
    if open.is_none() {
        *open = Some(Batch {
            command: gpu.queue.commandBuffer()?,
            encoder: Encoder::None,
            operations: 0,
            work: 0,
        });
    }
    Some(std::cell::RefMut::map(open, |open| open.as_mut().unwrap()))
}

/// A compute encoder in the open batch to encode one operation into. Finish the
/// operation with [`queued`].
pub(super) fn compute(gpu: &Gpu) -> Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>> {
    let mut batch = batch(gpu)?;
    if let Encoder::Compute(encoder) = &batch.encoder {
        return Some(encoder.clone());
    }
    batch.encoder.end();
    let encoder = batch.command.computeCommandEncoder()?;
    batch.encoder = Encoder::Compute(encoder.clone());
    Some(encoder)
}

/// A blit encoder in the open batch, as [`compute`].
pub(super) fn blit(gpu: &Gpu) -> Option<Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>> {
    let mut batch = batch(gpu)?;
    if let Encoder::Blit(encoder) = &batch.encoder {
        return Some(encoder.clone());
    }
    batch.encoder.end();
    let encoder = batch.command.blitCommandEncoder()?;
    batch.encoder = Encoder::Blit(encoder.clone());
    Some(encoder)
}

/// One operation has been encoded into the open batch, touching about `work`
/// elements. Commits the batch if it is now full or heavy enough to start.
pub(super) fn queued(gpu: &Gpu, work: usize) -> Option<()> {
    gpu.operations.set(gpu.operations.get() + 1);
    crate::counters::dispatch();
    let full = {
        let mut open = gpu.open.borrow_mut();
        let batch = open.as_mut()?;
        batch.operations += 1;
        batch.work = batch.work.saturating_add(work);
        batch.operations >= BATCH_OPERATIONS || batch.work >= BATCH_WORK
    };
    if full { flush(gpu) } else { Some(()) }
}

/// Commit the open batch, if there is one, without waiting for it.
pub(super) fn flush(gpu: &Gpu) -> Option<()> {
    let Some(mut batch) = gpu.open.borrow_mut().take() else {
        return Some(());
    };
    batch.encoder.end();
    let command = batch.command.clone();
    drop(batch);
    commit(gpu, command)
}

/// Submit `command` and return without waiting for the GPU.
///
/// Blocking here would cost a chain of resident operations a full round trip
/// per link (~145 µs each), when the whole point of the [`Metal`] backend is
/// that nothing is copied between them. Instead the buffer is parked
/// in [`Gpu::pending`] and the wait is deferred to [`sync`], which every path
/// that reads shared memory from the CPU calls first.
///
/// Deferring is safe because all of this module's work goes to a single
/// [`MTLCommandQueue`](objc2_metal::MTLCommandQueue) on one thread: buffers execute in commit order and Metal
/// tracks the hazards between them, so a kernel still sees its predecessor's
/// output. Resources stay alive too — a command buffer retains what its encoders
/// reference, so dropping a [`MetalBuffer`](super::MetalBuffer) with work in flight cannot free the
/// allocation early.
///
/// The one thing given up is per-call error reporting: a command that fails on
/// the GPU is noticed at the next `sync` rather than by the call that encoded
/// it. Encoding failures are still reported immediately.
///
/// [`Metal`]: crate::tensors::Metal
fn commit(gpu: &Gpu, command: Retained<ProtocolObject<dyn MTLCommandBuffer>>) -> Option<()> {
    command.commit();
    crate::counters::command_buffer();
    let sequence = gpu.committed.get() + 1;
    gpu.committed.set(sequence);
    gpu.pending.borrow_mut().push_back((sequence, command));
    // Cap the backlog so a long run of un-read operations cannot retain command
    // buffers without bound. Finished ones are dropped first, so the cap only
    // forces a wait when the GPU really is that far behind.
    if gpu.pending.borrow().len() >= 64 {
        reclaim(gpu);
        if gpu.pending.borrow().len() >= 64 {
            return sync(gpu);
        }
    }
    Some(())
}

/// Forget the command buffers at the front of the queue that have already
/// completed, without waiting for any, and return the allocations they were
/// holding back to the pool.
///
/// This is what lets a long chain of queued operations recycle its
/// intermediates: without it they would only come back at the next [`sync`],
/// and every operation in between would need a fresh allocation. A command
/// buffer that failed is left at the front, so the next `sync` still reports it.
pub(super) fn reclaim(gpu: &Gpu) {
    let Ok(mut pending) = gpu.pending.try_borrow_mut() else {
        return;
    };
    while let Some((sequence, command)) = pending.front() {
        if command.status() != MTLCommandBufferStatus::Completed {
            break;
        }
        gpu.completed.set(*sequence);
        pending.pop_front();
    }
    drop(pending);
    if let Ok(mut pool) = gpu.pool.try_borrow_mut() {
        pool.promote(gpu.completed.get());
    }
}

/// Block until every command buffer up to and including `fence` has finished —
/// not the whole queue — and return their allocations to the pool. Returns
/// whether they all completed; one that failed is left for [`sync`] to report.
pub(super) fn wait_through(gpu: &Gpu, fence: u64) -> bool {
    let Ok(mut pending) = gpu.pending.try_borrow_mut() else {
        return false;
    };
    crate::counters::sync();
    let mut ok = true;
    while let Some((sequence, command)) = pending.front() {
        if *sequence > fence {
            break;
        }
        command.waitUntilCompleted();
        if command.status() != MTLCommandBufferStatus::Completed {
            ok = false;
            break;
        }
        gpu.completed.set(*sequence);
        pending.pop_front();
    }
    drop(pending);
    if let Ok(mut pool) = gpu.pool.try_borrow_mut() {
        pool.promote(gpu.completed.get());
    }
    ok
}

/// Block until every command buffer committed so far has finished, and release
/// the allocations that were waiting on them.
///
/// Call this before the CPU reads any shared allocation the GPU may still be
/// writing. Returns `None` if any of them failed.
pub(super) fn sync(gpu: &Gpu) -> Option<()> {
    let flushed = flush(gpu);
    // Taken by value so a re-entrant call cannot see a half-drained list, and so
    // the buffers are released once they have been waited on.
    let pending = std::mem::take(&mut *gpu.pending.borrow_mut());
    if !pending.is_empty() {
        crate::counters::sync();
        gpu.waits.set(gpu.waits.get() + 1);
    }
    let mut ok = true;
    for (_, command) in &pending {
        command.waitUntilCompleted();
        if command.status() != MTLCommandBufferStatus::Completed {
            #[cfg(test)]
            eprintln!("Metal command buffer finished as {:?}", command.status());
            ok = false;
        }
    }
    // Nothing is in flight any more, so allocations released while it was can go
    // back into service. A busy borrow means an allocation is being handed out
    // right now; the next sync or reclaim will promote them instead.
    gpu.completed.set(gpu.committed.get());
    if let Ok(mut pool) = gpu.pool.try_borrow_mut() {
        pool.promote(gpu.completed.get());
    }
    (ok && flushed.is_some()).then_some(())
}

/// Synchronize queued work and stop before an incomplete allocation can be read.
pub(super) fn sync_or_panic(gpu: &Gpu) {
    assert!(
        sync(gpu).is_some(),
        "Metal command buffer failed; its output is invalid"
    );
}

/// Block until all GPU work submitted on this thread has completed.
///
/// Operations on [`Metal`](crate::tensors::Metal)-backed tensors are queued and
/// return before the GPU has run them, so this is the way to make outstanding
/// work observable — reading the values back already does it implicitly. It is
/// also what a benchmark needs in order to time GPU work rather than submission.
pub fn synchronize() {
    autoreleasepool(|_| {
        GPU.with(|cell| {
            if let Some(gpu) = cell.get_or_init(build_gpu).as_ref() {
                sync_or_panic(gpu);
            }
        });
    });
}
