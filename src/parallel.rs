//! Splitting host work across cores.
//!
//! A pool of worker threads, started on first use, runs one job at a time: a
//! job is a number of chunks, which the workers and the submitting thread take
//! in turn until none are left. Chunks are taken dynamically, so the slower
//! efficiency cores simply take fewer of them.
//!
//! A job submitted while the pool is busy — by another thread, or from inside
//! a chunk — runs on its submitting thread alone, so concurrent or nested work
//! never waits for the pool, and never deadlocks on it.
//!
//! Every kernel split here computes each element independently of the others,
//! so its results are the same however it is split. Reductions are not split:
//! a sum's rounding would then depend on the number of threads.

use std::any::Any;
use std::cell::Cell;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// How long an idle worker watches for the next job before it sleeps. Kernels
/// tend to come in runs, and waking a sleeping thread costs several
/// microseconds.
const SPIN: Duration = Duration::from_micros(50);

/// Chunks per thread, so that a slow or preempted thread holds up little.
const CHUNKS_PER_THREAD: usize = 4;

thread_local! {
    /// The most threads this thread's jobs may use; zero for all of them.
    static LIMIT: Cell<usize> = const { Cell::new(0) };
}

/// Limit the host kernels this thread runs to `threads` threads, itself
/// included: 1 runs them all on this thread, and 0 restores the default, every
/// core.
#[doc(hidden)]
pub fn set_host_threads(threads: usize) {
    LIMIT.with(|limit| limit.set(threads));
}

/// The threads a job submitted from this thread may use.
pub(crate) fn threads() -> usize {
    let available = available();
    match LIMIT.with(Cell::get) {
        0 => available,
        limit => limit.min(available),
    }
}

fn available() -> usize {
    static AVAILABLE: OnceLock<usize> = OnceLock::new();
    *AVAILABLE.get_or_init(|| std::thread::available_parallelism().map_or(1, usize::from))
}

/// Whether `T` is one of the plain floats — `f32`, `f64`, `f16`, `bf16` —
/// which threads may share, so that a generic kernel knows it may split.
pub(crate) fn plain_float<T: 'static>() -> bool {
    use std::any::TypeId;
    [
        TypeId::of::<f32>(),
        TypeId::of::<f64>(),
        TypeId::of::<half::f16>(),
        TypeId::of::<half::bf16>(),
    ]
    .contains(&TypeId::of::<T>())
}

/// Run `work` over consecutive ranges covering `0..len`, on several threads
/// when there is enough of it: each range holds at least `grain` elements, and
/// every range but the last a multiple of `align`.
pub(crate) fn for_ranges(
    len: usize,
    grain: usize,
    align: usize,
    work: impl Fn(Range<usize>) + Sync,
) {
    let threads = threads();
    let chunks = (len / grain.max(1)).min(threads * CHUNKS_PER_THREAD);
    if threads <= 1 || chunks <= 1 {
        work(0..len);
        return;
    }
    let align = align.max(1);
    let size = len.div_ceil(chunks).div_ceil(align) * align;
    let chunks = len.div_ceil(size);
    run(chunks, threads, &|chunk| {
        let start = chunk * size;
        work(start..len.min(start + size));
    });
}

/// Write `out` a range at a time, on several threads when it is long enough:
/// `work` gets each range's start and its window of `out`.
pub(crate) fn for_slices<T: Send>(
    out: &mut [T],
    grain: usize,
    work: impl Fn(usize, &mut [T]) + Sync,
) {
    // SAFETY: the bounds say what the function needs.
    unsafe { for_slices_unchecked(out, grain, work) }
}

/// [`for_slices`] for an element type the compiler cannot see may be shared,
/// such as a generic one already known to be a float.
///
/// # Safety
///
/// `T`, and whatever `work` captures, must be safe to use from several threads
/// at once.
pub(crate) unsafe fn for_slices_unchecked<T>(
    out: &mut [T],
    grain: usize,
    work: impl Fn(usize, &mut [T]),
) {
    let len = out.len();
    let base = Shared(out.as_mut_ptr());
    // SAFETY: the caller's promise; `base` is shared as disjoint windows.
    unsafe {
        for_ranges_unchecked(len, grain, 16, |range| {
            // SAFETY: the ranges are disjoint windows of `out`, which is
            // borrowed mutably for the whole call.
            let window = std::slice::from_raw_parts_mut(base.at(range.start), range.len());
            work(range.start, window);
        });
    }
}

/// [`for_ranges`] for work the compiler cannot see may be shared.
///
/// # Safety
///
/// Whatever `work` captures must be safe to use from several threads at once.
pub(crate) unsafe fn for_ranges_unchecked(
    len: usize,
    grain: usize,
    align: usize,
    work: impl Fn(Range<usize>),
) {
    struct AssertSync<F>(F);
    // SAFETY: the caller's promise.
    unsafe impl<F> Sync for AssertSync<F> {}
    impl<F> AssertSync<F> {
        // A method, so that closures capture the whole wrapper.
        fn get(&self) -> &F {
            &self.0
        }
    }
    let work = AssertSync(work);
    for_ranges(len, grain, align, |range| (work.get())(range));
}

/// A pointer threads may share, each using a disjoint part of what it points
/// to.
pub(crate) struct Shared<T>(pub(crate) *mut T);

impl<T> Clone for Shared<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Shared<T> {}

// SAFETY: the users of `Shared` give each thread a disjoint part, of an element
// type they have checked may be shared.
unsafe impl<T> Send for Shared<T> {}
unsafe impl<T> Sync for Shared<T> {}

impl<T> Shared<T> {
    /// The pointer `offset` elements on. A method, so that closures capture the
    /// whole `Shared` — which is `Sync` — rather than its raw pointer field.
    pub(crate) fn at(self, offset: usize) -> *mut T {
        // SAFETY: callers stay within the allocation.
        unsafe { self.0.add(offset) }
    }
}

/// Run `work(0)` … `work(chunks - 1)` on up to `threads` threads, the caller
/// one of them, and return once all have finished. A panic in any chunk is
/// raised again here, after the others are done.
///
/// The job lives on this function's stack: workers find it through the pool's
/// pointer to it, announcing themselves in `inside` before they look, and this
/// function clears the pointer and waits for `inside` to drain before it
/// returns.
fn run(chunks: usize, threads: usize, work: &(dyn Fn(usize) + Sync)) {
    let Some(pool) = pool() else {
        (0..chunks).for_each(work);
        return;
    };
    if pool
        .busy
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        (0..chunks).for_each(work);
        return;
    }
    let job = Job {
        // SAFETY: only the lifetime changes, and no worker uses the job after
        // this function returns (see above).
        work: unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), Work>(work) },
        chunks,
        next: Padded(AtomicUsize::new(0)),
        finished: Padded(AtomicUsize::new(0)),
        panic: Mutex::new(None),
    };
    // Only as many workers as there are chunks for them join in.
    pool.wanted
        .store(threads.min(chunks).saturating_sub(1), Ordering::Relaxed);
    pool.job
        .store(std::ptr::from_ref(&job).cast_mut(), Ordering::SeqCst);
    pool.epoch.0.fetch_add(1, Ordering::SeqCst);
    if pool.sleeping.load(Ordering::SeqCst) > 0 {
        let _lock = pool.lock.lock().unwrap();
        pool.wake.notify_all();
    }
    job.help();
    wait_until(|| job.finished.0.load(Ordering::Acquire) == chunks);
    pool.job.store(std::ptr::null_mut(), Ordering::SeqCst);
    wait_until(|| pool.inside.0.load(Ordering::SeqCst) == 0);
    pool.busy.store(false, Ordering::Release);
    if let Some(payload) = job.panic.lock().unwrap().take() {
        std::panic::resume_unwind(payload);
    }
}

/// Spin until `done`, yielding the core if that takes a while.
fn wait_until(done: impl Fn() -> bool) {
    let mut spins = 0u32;
    while !done() {
        spins += 1;
        if spins < 1 << 14 {
            std::hint::spin_loop();
        } else {
            std::thread::yield_now();
        }
    }
}

type Work = &'static (dyn Fn(usize) + Sync);

/// A value alone on its cache line, so that threads updating it do not slow
/// threads reading its neighbours.
#[repr(align(128))]
struct Padded<T>(T);

struct Job {
    work: Work,
    chunks: usize,
    /// The next chunk to take.
    next: Padded<AtomicUsize>,
    finished: Padded<AtomicUsize>,
    panic: Mutex<Option<Box<dyn Any + Send>>>,
}

impl Job {
    /// Take chunks until none are left.
    fn help(&self) {
        loop {
            let chunk = self.next.0.fetch_add(1, Ordering::Relaxed);
            if chunk >= self.chunks {
                return;
            }
            let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (self.work)(chunk)));
            if let Err(payload) = ran {
                self.panic.lock().unwrap().get_or_insert(payload);
            }
            self.finished.0.fetch_add(1, Ordering::Release);
        }
    }
}

struct Pool {
    busy: AtomicBool,
    /// Bumped for every job, so that a spinning worker sees one arrive.
    epoch: Padded<AtomicU64>,
    /// The current job, or null.
    job: AtomicPtr<Job>,
    /// Workers numbered below this join the current job.
    wanted: AtomicUsize,
    /// Workers that may be looking at the current job.
    inside: Padded<AtomicUsize>,
    /// Workers asleep on `wake`, changed only under `lock`.
    sleeping: AtomicUsize,
    lock: Mutex<()>,
    wake: Condvar,
}

/// The pool, started on first use; `None` on a single core, or if no worker
/// thread could be started.
fn pool() -> Option<&'static Pool> {
    static POOL: OnceLock<Option<&'static Pool>> = OnceLock::new();
    *POOL.get_or_init(|| {
        let workers = available().saturating_sub(1);
        if workers == 0 {
            return None;
        }
        let pool: &'static Pool = Box::leak(Box::new(Pool {
            busy: AtomicBool::new(false),
            epoch: Padded(AtomicU64::new(0)),
            job: AtomicPtr::new(std::ptr::null_mut()),
            wanted: AtomicUsize::new(0),
            inside: Padded(AtomicUsize::new(0)),
            sleeping: AtomicUsize::new(0),
            lock: Mutex::new(()),
            wake: Condvar::new(),
        }));
        let mut started = 0;
        for index in 0..workers {
            let spawned = std::thread::Builder::new()
                .name(format!("tensorcrate-{index}"))
                .spawn(move || worker(pool, index));
            started += usize::from(spawned.is_ok());
        }
        (started > 0).then_some(pool)
    })
}

fn worker(pool: &'static Pool, index: usize) {
    let mut seen = 0u64;
    loop {
        // Watch for the next job a while, then sleep until one comes.
        let deadline = Instant::now() + SPIN;
        let mut spins = 0u32;
        while pool.epoch.0.load(Ordering::Acquire) == seen {
            spins += 1;
            if spins.is_multiple_of(64) && Instant::now() > deadline {
                let mut lock = pool.lock.lock().unwrap();
                pool.sleeping.fetch_add(1, Ordering::SeqCst);
                while pool.epoch.0.load(Ordering::SeqCst) == seen {
                    lock = pool.wake.wait(lock).unwrap();
                }
                pool.sleeping.fetch_sub(1, Ordering::SeqCst);
                break;
            }
            std::hint::spin_loop();
        }
        seen = pool.epoch.0.load(Ordering::Acquire);
        if index >= pool.wanted.load(Ordering::Relaxed) {
            continue;
        }
        pool.inside.0.fetch_add(1, Ordering::SeqCst);
        let job = pool.job.load(Ordering::SeqCst);
        // SAFETY: the job's submitter waits for `inside` to drain before its
        // job goes away, and clears the pointer first, so a job seen here is
        // alive until `inside` is decremented.
        if let Some(job) = unsafe { job.as_ref() } {
            job.help();
        }
        pool.inside.0.fetch_sub(1, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_cover_the_whole_length_once() {
        for len in [0, 1, 15, 1000, 100_003] {
            let hits: Vec<AtomicUsize> = (0..len).map(|_| AtomicUsize::new(0)).collect();
            for_ranges(len, 7, 4, |range| {
                assert!(range.start % 4 == 0);
                for i in range {
                    hits[i].fetch_add(1, Ordering::Relaxed);
                }
            });
            assert!(hits.iter().all(|hit| hit.load(Ordering::Relaxed) == 1));
        }
    }

    #[test]
    fn slices_are_written_in_place() {
        let mut out = vec![0usize; 100_000];
        for_slices(&mut out, 100, |start, window| {
            for (offset, value) in window.iter_mut().enumerate() {
                *value = start + offset;
            }
        });
        assert!(out.iter().enumerate().all(|(i, &v)| i == v));
    }

    #[test]
    fn nested_jobs_run_inline() {
        let total = AtomicUsize::new(0);
        for_ranges(64, 1, 1, |outer| {
            for_ranges(64, 1, 1, |inner| {
                total.fetch_add(outer.len() * inner.len(), Ordering::Relaxed);
            });
        });
        assert_eq!(total.load(Ordering::Relaxed), 64 * 64);
    }

    #[test]
    fn a_panicking_chunk_panics_the_caller() {
        let result = std::panic::catch_unwind(|| {
            for_ranges(1000, 1, 1, |range| {
                assert!(!range.contains(&500), "chunk failed")
            });
        });
        assert!(result.is_err());
        // The pool is usable afterwards.
        let total = AtomicUsize::new(0);
        for_ranges(1000, 1, 1, |range| {
            total.fetch_add(range.len(), Ordering::Relaxed);
        });
        assert_eq!(total.load(Ordering::Relaxed), 1000);
    }

    #[test]
    fn generated_kernels_fill_every_element() {
        let len = 1_000_003;
        let one = crate::__private::generate(len, 1, |i| i as f64 * 0.5);
        assert!(one.iter().enumerate().all(|(i, &v)| v == i as f64 * 0.5));
        let [a, b] = crate::__private::generate_many(len, 1, |i| [i as f32, -(i as f32)]);
        assert!(
            a.iter()
                .zip(&b)
                .enumerate()
                .all(|(i, (&a, &b))| a == i as f32 && b == -a)
        );
    }

    #[test]
    fn one_thread_runs_inline() {
        set_host_threads(1);
        let caller = std::thread::current().id();
        for_ranges(100_000, 1, 1, |_| {
            assert_eq!(std::thread::current().id(), caller)
        });
        set_host_threads(0);
    }
}
