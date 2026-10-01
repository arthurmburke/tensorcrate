//! Submitting command buffers and waiting for them.

use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandBufferStatus};

use super::device::{GPU, Gpu, build_gpu};

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
pub(super) fn commit(
    gpu: &Gpu,
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
) -> Option<()> {
    command.commit();
    crate::counters::command_buffer();
    let mut pending = gpu.pending.borrow_mut();
    pending.push(command);
    // Cap the backlog so a long run of un-read operations cannot retain command
    // buffers without bound.
    if pending.len() >= 64 {
        drop(pending);
        return sync(gpu);
    }
    Some(())
}

/// Block until every command buffer committed so far has finished, and release
/// the allocations that were waiting on them.
///
/// Call this before the CPU reads any shared allocation the GPU may still be
/// writing. Returns `None` if any of them failed.
pub(super) fn sync(gpu: &Gpu) -> Option<()> {
    // Taken by value so a re-entrant call cannot see a half-drained list, and so
    // the buffers are released once they have been waited on.
    let pending = std::mem::take(&mut *gpu.pending.borrow_mut());
    if !pending.is_empty() {
        crate::counters::sync();
    }
    let mut ok = true;
    for command in &pending {
        command.waitUntilCompleted();
        if command.status() != MTLCommandBufferStatus::Completed {
            #[cfg(test)]
            eprintln!("Metal command buffer finished as {:?}", command.status());
            ok = false;
        }
    }
    // Nothing is in flight any more, so allocations released while it was can go
    // back into service. A busy borrow means an allocation is being handed out
    // right now; the next sync will promote them instead.
    if let Ok(mut pool) = gpu.pool.try_borrow_mut() {
        pool.retire();
    }
    ok.then_some(())
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
