//! Performance benchmarks for the operations neural networks are made of.
//!
//! ```text
//! cargo bench --bench nn_ops                       # everything
//! cargo bench --bench nn_ops -- --quick            # fewer sizes, fewer samples
//! cargo bench --bench nn_ops -- dense softmax      # only groups/cases containing a word
//! cargo bench --bench nn_ops --features counters   # add kernels, bytes moved and GPU dispatches
//! cargo test --release --benches                   # smoke run: one pass of each case
//! ```
//!
//! Five groups, each comparing one axis of the design:
//!
//! | group           | compares                                                           |
//! |-----------------|--------------------------------------------------------------------|
//! | `fusion`        | one [`Program`] run fused and unfused, on the host and on Metal    |
//! | `matmul`        | scalar loop, direct SIMD kernel, the Host API, and Metal           |
//! | `simd`          | scalar loop, direct SIMD kernel and the Host API, per kernel       |
//! | `dispatch`      | launch latency, pipelined launches, and host↔device transfers      |
//! | `training`      | a whole MLP step (forward, backward, Adam), fused and unfused      |
//!
//! # Reading the tables
//!
//! Every row is the median of several samples, each averaged over enough calls to
//! last tens of milliseconds, and `±` is the median absolute deviation. The last
//! column is the speed-up over the first row of the case.
//!
//! * **fused / unfused** is [`fused::with_mode`] on the same program, so the two
//!   rows run identical arithmetic and differ only in kernels launched and bytes
//!   moved. The unfused interpreter also copies each input in and each output out,
//!   so it is slightly slower than the same kernels called by hand. The
//!   activations therefore add a **direct kernels** row, which is exactly that:
//!   one `Kernels` call per operation, the baseline fusion is measured against.
//! * **host API** is the tensor method as a caller writes it. It dispatches
//!   through the tiers in `Matrix::matmul` and friends — Accelerate on macOS,
//!   then the SIMD kernels, then a scalar loop — and allocates its result, which
//!   the direct-kernel rows do not.
//! * **metal (latency)** waits for the GPU after every call, which is what a
//!   caller that reads the result pays. **metal (pipelined)** queues many calls
//!   and waits once per batch, which is what a chain of resident operations pays;
//!   the gap between the two is the cost of a blocking round trip.
//! * **metal + transfers** uploads the operands and downloads the result around
//!   every call, to show what keeping tensors resident saves.
//!
//! Before timing a case the benchmark checks that its variants agree — fused with
//! unfused bit for bit on the host, SIMD and Metal within a tolerance — because a
//! fast wrong answer is not a result. Everything is `f32`, the type the GPU
//! shaders compute in.
//!
//! These are wall-clock measurements on a machine that is also doing other things.
//! Compare rows within a table rather than numbers between runs, and expect the
//! last digit to move.

use std::hint::black_box;
use std::time::{Duration, Instant};

use tensorcrate::optim::{Adam, Rule};
use tensorcrate::tensors::fused::{
    self, Builder, DType, Fusable, Mode, Program, Remap, RowStatistic,
};
use tensorcrate::tensors::{
    Analytic, Axis, Backend, BinaryOp, Compare, Host, Kernels, Matrix, Tape, Vector,
};

#[cfg(all(feature = "metal", target_os = "macos"))]
use tensorcrate::tensors::Metal;

// ---- harness ------------------------------------------------------------------

/// How much work a run does.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Profile {
    /// `cargo bench`: every size, many samples.
    Full,
    /// `cargo bench -- --quick`: every other size, few samples.
    Quick,
    /// No `--bench` flag, which is how `cargo test --benches` runs us: the
    /// smallest size of each case, once. Enough to run every correctness check.
    Smoke,
}

struct Config {
    profile: Profile,
    filters: Vec<String>,
    /// How long one sample should last.
    target: Duration,
    samples: usize,
}

impl Config {
    fn from_args() -> Self {
        let (mut bench, mut quick) = (false, false);
        let mut filters = Vec::new();
        for arg in std::env::args().skip(1) {
            match arg.as_str() {
                "--bench" => bench = true,
                "--quick" => quick = true,
                // libtest flags such as `--nocapture` mean nothing here.
                flag if flag.starts_with("--") => {}
                word => filters.push(word.to_string()),
            }
        }
        let profile = match (bench, quick) {
            (false, _) => Profile::Smoke,
            (true, true) => Profile::Quick,
            (true, false) => Profile::Full,
        };
        let (target, samples) = match profile {
            Profile::Full => (Duration::from_millis(40), 15),
            Profile::Quick => (Duration::from_millis(15), 5),
            Profile::Smoke => (Duration::ZERO, 1),
        };
        Config {
            profile,
            filters,
            target,
            samples,
        }
    }

    /// The sizes this profile runs, from an ascending list.
    fn pick<T: Copy>(&self, all: &[T]) -> Vec<T> {
        match self.profile {
            Profile::Full => all.to_vec(),
            Profile::Quick => all.iter().step_by(2).copied().collect(),
            Profile::Smoke => all[..1].to_vec(),
        }
    }

    fn wants(&self, name: &str) -> bool {
        self.filters.is_empty() || self.filters.iter().any(|filter| name.contains(filter))
    }
}

/// When a case waits for the GPU. Host cases never do.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Drain {
    None,
    /// After every call: the latency of one operation, result in hand.
    PerCall,
    /// After every batch of calls: the cost of an operation in a queue.
    PerBatch,
}

/// Block until the GPU has finished everything queued on this thread.
fn gpu_sync() {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    tensorcrate::metal::synchronize();
}

/// What one call of a case accomplishes, for the throughput column.
#[derive(Copy, Clone)]
enum Work {
    Elements(usize),
    Bytes(usize),
    Flops(f64),
    Steps,
}

impl Work {
    fn rate(self, seconds: f64) -> String {
        match self {
            Work::Elements(n) => format!("{:.3} ns/elem", seconds / n as f64 * 1e9),
            Work::Bytes(b) => format!("{:.1} GB/s", b as f64 / seconds / 1e9),
            Work::Flops(f) => format!("{:.1} GFLOP/s", f / seconds / 1e9),
            Work::Steps => format!("{:.1} steps/s", 1.0 / seconds),
        }
    }
}

struct Stats {
    median: f64,
    min: f64,
    /// Median absolute deviation, as a fraction of the median.
    spread: f64,
    /// Kernels, bytes and GPU dispatches of one call; `counters` builds only.
    #[cfg_attr(not(feature = "counters"), allow(dead_code))]
    counted: Option<(u64, u64, u64)>,
}

/// Time `f`. Its result is passed through `black_box`, so the work cannot be
/// optimized away, and dropped inside the timed region, as a caller's would be.
fn measure<R>(cfg: &Config, drain: Drain, mut f: impl FnMut() -> R) -> Stats {
    let mut call = || {
        black_box(f());
        if drain == Drain::PerCall {
            gpu_sync();
        }
    };
    // The first call pays for shader compilation, buffer pools and page faults.
    call();
    gpu_sync();

    let start = Instant::now();
    call();
    if drain == Drain::PerBatch {
        gpu_sync();
    }
    let once = start.elapsed().as_secs_f64().max(1e-9);

    if cfg.profile == Profile::Smoke {
        return Stats {
            median: once,
            min: once,
            spread: 0.0,
            counted: None,
        };
    }

    let iters = ((cfg.target.as_secs_f64() / once).ceil() as u64).clamp(1, 1_000_000);
    let mut samples = Vec::with_capacity(cfg.samples);
    // Round 0 is a discarded warm-up at the real batch size.
    for round in 0..=cfg.samples {
        let start = Instant::now();
        for _ in 0..iters {
            call();
        }
        if drain == Drain::PerBatch {
            gpu_sync();
        }
        let per_call = start.elapsed().as_secs_f64() / iters as f64;
        if round > 0 {
            samples.push(per_call);
        }
    }

    #[cfg(feature = "counters")]
    let counted = {
        let ((), counts) = tensorcrate::counters::measure(|| {
            call();
            if drain == Drain::PerBatch {
                gpu_sync();
            }
        });
        Some((counts.kernels, counts.bytes, counts.dispatches))
    };
    #[cfg(not(feature = "counters"))]
    let counted = None;

    samples.sort_by(f64::total_cmp);
    let median = samples[samples.len() / 2];
    let mut deviations: Vec<f64> = samples.iter().map(|s| (s - median).abs()).collect();
    deviations.sort_by(f64::total_cmp);
    Stats {
        median,
        min: samples[0],
        spread: deviations[deviations.len() / 2] / median,
        counted,
    }
}

fn format_time(seconds: f64) -> String {
    if seconds >= 1.0 {
        format!("{seconds:.3} s")
    } else if seconds >= 1e-3 {
        format!("{:.3} ms", seconds * 1e3)
    } else if seconds >= 1e-6 {
        format!("{:.3} µs", seconds * 1e6)
    } else {
        format!("{:.1} ns", seconds * 1e9)
    }
}

struct Bench {
    cfg: Config,
    group: Option<String>,
    cases: usize,
}

impl Bench {
    /// Start a case, or `None` when the filter excludes it, so its setup is
    /// skipped as well as its timing.
    fn case(&mut self, group: &str, title: &str, work: Work) -> Option<Case<'_>> {
        if !self.cfg.wants(&format!("{group}/{title}")) {
            return None;
        }
        self.cases += 1;
        if self.group.as_deref() != Some(group) {
            println!("\n=== {group} ===");
            let counters = if cfg!(feature = "counters") {
                format!(" {:>7} {:>9} {:>7}", "kernels", "MB moved", "gpu ops")
            } else {
                String::new()
            };
            println!(
                "{:<38} {:>11} {:>11} {:>6} {:>15} {:>8}{counters}",
                "", "median", "min", "±", "throughput", "vs first"
            );
            self.group = Some(group.to_string());
        }
        println!("\n{title}");
        Some(Case {
            cfg: &self.cfg,
            work,
            baseline: None,
        })
    }
}

struct Case<'a> {
    cfg: &'a Config,
    work: Work,
    baseline: Option<f64>,
}

impl Case<'_> {
    fn row<R>(&mut self, label: &str, drain: Drain, f: impl FnMut() -> R) {
        let stats = measure(self.cfg, drain, f);
        let versus = match self.baseline {
            None => {
                self.baseline = Some(stats.median);
                "—".to_string()
            }
            Some(first) => format!("{:.2}×", first / stats.median),
        };
        #[cfg_attr(not(feature = "counters"), allow(unused_mut))]
        let mut line = format!(
            "  {label:<36} {:>11} {:>11} {:>5.1}% {:>15} {:>8}",
            format_time(stats.median),
            format_time(stats.min),
            stats.spread * 100.0,
            self.work.rate(stats.median),
            versus,
        );
        #[cfg(feature = "counters")]
        if let Some((kernels, bytes, dispatches)) = stats.counted {
            line += &format!(" {kernels:>7} {:>9.2} {dispatches:>7}", bytes as f64 / 1e6);
        }
        println!("{line}");
    }
}

// ---- checking -----------------------------------------------------------------

/// A result as host `f32`s, whatever tensor it is.
trait Flat {
    fn flat(&self) -> Vec<f32>;
}

impl<B: Backend> Flat for Vector<f32, B> {
    fn flat(&self) -> Vec<f32> {
        self.to_backend::<Host>().into_vec()
    }
}

impl<B: Backend> Flat for Matrix<f32, B> {
    fn flat(&self) -> Vec<f32> {
        self.to_backend::<Host>().into_vec()
    }
}

/// For workloads that update state instead of returning it.
impl Flat for () {
    fn flat(&self) -> Vec<f32> {
        Vec::new()
    }
}

#[track_caller]
fn assert_close(what: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: lengths differ");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol * (1.0 + w.abs()),
            "{what}: element {i} is {g}, expected {w} (tolerance {tol})"
        );
    }
}

/// As [`assert_close`], but measuring every error against the scale of the
/// whole result — its root mean square — rather than the element's own size.
/// That is the accuracy a reduced-precision product promises: an element whose
/// terms happen to cancel to near zero carries the same absolute error as the
/// rest, which would look huge relative to itself.
#[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(dead_code))]
#[track_caller]
fn assert_close_to_scale(what: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: lengths differ");
    let scale = (want.iter().map(|&w| w * w).sum::<f32>() / want.len().max(1) as f32).sqrt();
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol * (1.0 + scale),
            "{what}: element {i} is {g}, expected {w} (tolerance {tol} of scale {scale})"
        );
    }
}

#[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(dead_code))]
#[track_caller]
fn assert_modes_close<R: Flat>(
    what: &str,
    tol: f32,
    reference: &[f32],
    run: &mut impl FnMut() -> R,
) {
    for mode in [Mode::Unfused, Mode::Fused] {
        let got = fused::with_mode(mode, &mut *run).flat();
        assert_close(&format!("{what} ({mode:?})"), &got, reference, tol);
    }
}

/// Deterministic values in `[-1, 1)`.
fn values(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 2654435761 + seed * 40503) % 1000) as f32 / 500.0) - 1.0)
        .collect()
}

fn matrix<B: Backend>(rows: usize, cols: usize, seed: usize) -> Matrix<f32, B> {
    Matrix::from_flat(rows, cols, values(rows * cols, seed)).to_backend::<B>()
}

fn vector<B: Backend>(len: usize, seed: usize) -> Vector<f32, B> {
    Vector::new(values(len, seed)).to_backend::<B>()
}

// ---- fusion cases -------------------------------------------------------------

#[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(dead_code))]
const GPU: &[Drain] = &[Drain::PerCall, Drain::PerBatch];

/// Unfused then fused rows of one workload, once per way of waiting.
fn fusion_rows<R>(case: &mut Case, backend: &str, drains: &[Drain], mut run: impl FnMut() -> R) {
    for &drain in drains {
        let wait = match drain {
            Drain::None => "",
            Drain::PerCall => " (latency)",
            Drain::PerBatch => " (pipelined)",
        };
        for mode in [Mode::Unfused, Mode::Fused] {
            let name = match mode {
                Mode::Unfused => "unfused",
                Mode::Fused => "fused",
            };
            case.row(&format!("{backend} {name}{wait}"), drain, || {
                fused::with_mode(mode, &mut run)
            });
        }
        interpreted_row(case, backend, drain, &mut run);
        one_thread_row(case, backend, "fused", || {
            fused::with_mode(Mode::Fused, &mut run)
        });
    }
}

/// On the host, a row once more on one thread, to show what splitting kernels
/// across cores buys.
fn one_thread_row<R>(case: &mut Case, backend: &str, name: &str, run: impl FnMut() -> R) {
    if backend == "host" {
        tensorcrate::set_host_threads(1);
        case.row(&format!("host {name}, 1 thread"), Drain::None, run);
        tensorcrate::set_host_threads(0);
    }
}

/// On Metal, the fused program once more on the bytecode interpreter instead
/// of its specialized kernel, to show what compiling it buys.
fn interpreted_row<R>(case: &mut Case, backend: &str, drain: Drain, run: &mut impl FnMut() -> R) {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if backend == "metal" && drain == Drain::PerBatch {
        use tensorcrate::metal::set_fused_codegen;
        set_fused_codegen(false);
        case.row("metal fused, interpreted (pipelined)", drain, || {
            fused::with_mode(Mode::Fused, &mut *run)
        });
        set_fused_codegen(true);
    }
    let _ = (case, backend, drain, run);
}

/// A fused-versus-unfused case. `$build` is an expression that builds the
/// workload for the backend named by `$B`, which the macro defines as `Host`
/// and then as `Metal`. On the host the fused result must equal the unfused one
/// bit for bit; on Metal both must be within `$tol` of the host's.
macro_rules! fusion_case {
    ($bench:expr, $group:expr, $title:expr, $work:expr, $tol:expr, |$B:ident| $build:expr) => {
        if let Some(mut case) = $bench.case($group, &$title, $work) {
            let title: &str = &$title;
            let reference;
            {
                type $B = Host;
                let mut run = $build;
                let unfused = fused::with_mode(Mode::Unfused, &mut run).flat();
                let fused_result = fused::with_mode(Mode::Fused, &mut run).flat();
                assert_eq!(
                    unfused, fused_result,
                    "{title}: fused differs from unfused on the host"
                );
                reference = unfused;
                fusion_rows(&mut case, "host", &[Drain::None], run);
            }
            #[cfg(all(feature = "metal", target_os = "macos"))]
            {
                type $B = Metal;
                let mut run = $build;
                assert_modes_close(title, $tol, &reference, &mut run);
                fusion_rows(&mut case, "metal", GPU, run);
            }
            let _ = &reference;
        }
    };
}

fn program(build: impl FnOnce(&mut Builder, fused::Value) -> fused::Value) -> Program {
    let mut b = Builder::new();
    let x = b.input(DType::F32);
    let y = build(&mut b, x);
    b.output(y, DType::F32);
    b.build().unwrap()
}

fn relu() -> Program {
    program(|b, x| {
        let zero = b.constant(0.0);
        b.compare(Compare::Max, x, zero)
    })
}

/// `1 / (1 + exp(-x))`.
fn sigmoid() -> Program {
    program(|b, x| {
        let negated = b.scale(x, -1.0);
        let e = b.unary(Analytic::Exp, negated);
        let denominator = b.shift(e, 1.0);
        let one = b.constant(1.0);
        b.div(one, denominator)
    })
}

/// The tanh approximation of GELU: `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`.
fn gelu() -> Program {
    program(|b, x| {
        let square = b.mul(x, x);
        let cube = b.mul(square, x);
        let cubic = b.scale(cube, 0.044715);
        let inner = b.add(x, cubic);
        let scaled = b.scale(inner, 0.797_884_6);
        let tanh = b.unary(Analytic::Tanh, scaled);
        let shifted = b.shift(tanh, 1.0);
        let product = b.mul(x, shifted);
        b.scale(product, 0.5)
    })
}

/// `len` dependent elementwise operations that keep values bounded.
fn chain(len: usize) -> Program {
    program(|b, x| {
        (0..len).fold(x, |v, i| {
            if i % 2 == 0 {
                b.scale(v, 0.999)
            } else {
                b.shift(v, 0.001)
            }
        })
    })
}

/// `scale·(x − mean) / sqrt(variance + ε) + shift`, per row: the mean and the
/// sum of squared deviations are statistics the program computes of its input,
/// and the learned scale and shift are read across the rows.
fn layer_norm_program(cols: usize) -> Program {
    let mut b = Builder::new();
    let x = b.input(DType::F32);
    let gamma = b.input_remapped(DType::F32, Remap::Row);
    let beta = b.input_remapped(DType::F32, Remap::Row);
    let mean = b.row_statistic(x, RowStatistic::Mean);
    let deviations = b.row_statistic(x, RowStatistic::Deviations);
    let variance = b.scale(deviations, 1.0 / cols as f32);
    let stabilized = b.shift(variance, 1e-5);
    let deviation = b.unary(Analytic::Sqrt, stabilized);
    let centered = b.sub(x, mean);
    let normalized = b.div(centered, deviation);
    let scaled = b.mul(normalized, gamma);
    let shifted = b.add(scaled, beta);
    b.output(shifted, DType::F32);
    b.build().unwrap()
}

/// `exp(x)`, whose column sums are the softmax denominators.
fn exp_program() -> Program {
    program(|b, x| b.unary(Analytic::Exp, x))
}

/// `exp(x) / s`: normalizes `exp(x)` by each column's sum, computing it again
/// rather than reading it back.
fn normalize_program() -> Program {
    let mut b = Builder::new();
    let x = b.input(DType::F32);
    let sums = b.input_remapped(DType::F32, Remap::Row);
    let e = b.unary(Analytic::Exp, x);
    let softmax = b.div(e, sums);
    b.output(softmax, DType::F32);
    b.build().unwrap()
}

/// `relu(x·w + bias)`, the product's epilogue.
fn dense_program() -> Program {
    let mut b = Builder::new();
    let product = b.input(DType::F32);
    let bias = b.input_remapped(DType::F32, Remap::Row);
    let shifted = b.add(product, bias);
    let zero = b.constant(0.0);
    let activated = b.compare(Compare::Max, shifted, zero);
    b.output(activated, DType::F32);
    b.build().unwrap()
}

fn elementwise<B: Kernels>(program: &Program, len: usize) -> impl FnMut() -> Vector<f32, B> {
    let x = vector::<B>(len, 1);
    move || program.run_vectors(&[&x]).remove(0)
}

/// The activations, each as a [`Program`] and as the `Kernels` calls a caller
/// would write without one.
#[derive(Copy, Clone)]
enum Activation {
    Relu,
    Sigmoid,
    Gelu,
}

impl Activation {
    const ALL: [Activation; 3] = [Activation::Relu, Activation::Sigmoid, Activation::Gelu];

    fn name(self) -> &'static str {
        match self {
            Activation::Relu => "relu",
            Activation::Sigmoid => "sigmoid",
            Activation::Gelu => "gelu",
        }
    }

    /// Arithmetic operations, so what fusion has to remove.
    fn ops(self) -> usize {
        match self {
            Activation::Relu => 1,
            Activation::Sigmoid => 4,
            Activation::Gelu => 9,
        }
    }

    fn program(self) -> Program {
        match self {
            Activation::Relu => relu(),
            Activation::Sigmoid => sigmoid(),
            Activation::Gelu => gelu(),
        }
    }

    /// One kernel per operation, with the same operations in the same order as
    /// [`program`](Self::program), so on the host the two are bit for bit equal.
    fn direct<B: Kernels>(self, x: &Vector<f32, B>) -> Vector<f32, B> {
        use BinaryOp::{Add, Div, Mul};
        match self {
            Activation::Relu => B::vector_compare_scalar(x, 0.0, Compare::Max, false),
            Activation::Sigmoid => {
                let negated = B::vector_broadcast(x, -1.0, Mul, false);
                let e = B::vector_unary(&negated, Analytic::Exp);
                let denominator = B::vector_broadcast(&e, 1.0, Add, false);
                B::vector_broadcast(&denominator, 1.0, Div, true)
            }
            Activation::Gelu => {
                let square = B::vector_elementwise(x, x, Mul);
                let cube = B::vector_elementwise(&square, x, Mul);
                let cubic = B::vector_broadcast(&cube, 0.044715, Mul, false);
                let inner = B::vector_elementwise(x, &cubic, Add);
                let scaled = B::vector_broadcast(&inner, 0.797_884_6, Mul, false);
                let tanh = B::vector_unary(&scaled, Analytic::Tanh);
                let shifted = B::vector_broadcast(&tanh, 1.0, Add, false);
                let product = B::vector_elementwise(x, &shifted, Mul);
                B::vector_broadcast(&product, 0.5, Mul, false)
            }
        }
    }
}

/// One activation on one backend: direct kernel calls, the program unfused, and
/// the program fused. The first is what a caller pays today without fusion; the
/// second adds the copies the unfused interpreter makes when it loads and stores,
/// so it is the fused program's like-for-like reference, not the floor.
fn activation_rows<B: Kernels>(
    case: &mut Case,
    backend: &str,
    drains: &[Drain],
    activation: Activation,
    program: &Program,
    len: usize,
) {
    let x = vector::<B>(len, 1);
    for &drain in drains {
        let wait = match drain {
            Drain::None => "",
            Drain::PerCall => " (latency)",
            Drain::PerBatch => " (pipelined)",
        };
        case.row(&format!("{backend} direct kernels{wait}"), drain, || {
            activation.direct(&x)
        });
        for (name, mode) in [("unfused", Mode::Unfused), ("fused", Mode::Fused)] {
            case.row(&format!("{backend} {name}{wait}"), drain, || {
                fused::with_mode(mode, || program.run_vectors(&[&x]).remove(0))
            });
        }
        interpreted_row(case, backend, drain, &mut || {
            program.run_vectors(&[&x]).remove(0)
        });
        one_thread_row(case, backend, "direct kernels", || activation.direct(&x));
        one_thread_row(case, backend, "fused", || {
            fused::with_mode(Mode::Fused, || program.run_vectors(&[&x]).remove(0))
        });
    }
}

fn activation_case(bench: &mut Bench, activation: Activation, len: usize) {
    let ops = activation.ops();
    let title = format!(
        "{} ({ops} op{}), n={len}",
        activation.name(),
        if ops == 1 { "" } else { "s" }
    );
    let Some(mut case) = bench.case("fusion", &title, Work::Elements(len)) else {
        return;
    };
    let program = activation.program();

    let host = vector::<Host>(len, 1);
    let direct = activation.direct(&host).flat();
    // The program is optimized, and may be reassociated, so it agrees with the
    // direct kernels to rounding — and with itself, fused or not, exactly.
    let unfused =
        fused::with_mode(Mode::Unfused, || program.run_vectors(&[&host]).remove(0)).flat();
    let fused_result =
        fused::with_mode(Mode::Fused, || program.run_vectors(&[&host]).remove(0)).flat();
    assert_eq!(
        fused_result, unfused,
        "{title}: fused differs from unfused on the host"
    );
    assert_close(&title, &fused_result, &direct, 1e-5);
    activation_rows::<Host>(&mut case, "host", &[Drain::None], activation, &program, len);

    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        let gpu = host.to_backend::<Metal>();
        assert_close(&title, &activation.direct(&gpu).flat(), &direct, 2e-3);
        let mut run = elementwise::<Metal>(&program, len);
        assert_modes_close(&title, 2e-3, &direct, &mut run);
        activation_rows::<Metal>(&mut case, "metal", GPU, activation, &program, len);
    }
}

fn layer_norm<B: Kernels>(
    program: &Program,
    rows: usize,
    cols: usize,
) -> impl FnMut() -> Matrix<f32, B> {
    let x = matrix::<B>(rows, cols, 1);
    let gamma = vector::<B>(cols, 2);
    let beta = vector::<B>(cols, 3);
    move || {
        let inputs: [&dyn Fusable<B>; 3] = [&x, &gamma, &beta];
        program
            .run((rows, cols), &inputs, &mut [])
            .remove(0)
            .into_matrix::<f32>()
    }
}

/// Softmax down each column of a `features × batch` matrix: the column sums of
/// `exp(x)`, then the normalization. Fused, each is one kernel and `exp(x)` is
/// never stored; unfused, it is the `exp` kernel, a product with ones for the
/// sums, and the `exp` and division kernels again.
fn softmax<B: Kernels>(
    (exp, normalize): (&Program, &Program),
    features: usize,
    batch: usize,
) -> impl FnMut() -> Matrix<f32, B> {
    let x = matrix::<B>(features, batch, 1);
    move || {
        let sums = exp.run_sum((features, batch), &[&x], Axis::Columns);
        let inputs: [&dyn Fusable<B>; 2] = [&x, &sums];
        normalize
            .run((features, batch), &inputs, &mut [])
            .remove(0)
            .into_matrix::<f32>()
    }
}

fn dense_layer<B: Kernels>(
    program: &Program,
    batch: usize,
    inputs: usize,
    outputs: usize,
) -> impl FnMut() -> Matrix<f32, B> {
    let x = matrix::<B>(batch, inputs, 1);
    let w = matrix::<B>(inputs, outputs, 2);
    let bias = vector::<B>(outputs, 3);
    move || {
        program
            .run_matmul(&x, &w, &[&bias])
            .remove(0)
            .into_matrix::<f32>()
    }
}

fn adam_step<B: Kernels>(len: usize) -> impl FnMut() {
    let gradient = vector::<B>(len, 1);
    let mut parameters = vector::<B>(len, 2);
    let mut rule = Adam::new(1e-3);
    move || rule.update(&mut parameters, &gradient)
}

/// `W₂·relu(W₁·X + b₁) + b₂` against regression targets: forward, backward and an
/// Adam update of every parameter.
fn mlp_step<B: Kernels>(
    inputs: usize,
    hidden: usize,
    outputs: usize,
    batch: usize,
) -> impl FnMut() {
    let x = matrix::<B>(inputs, batch, 5);
    let targets = matrix::<B>(outputs, batch, 6);
    let ones = Vector::<f32>::filled(batch, 1.0).to_backend::<B>();
    let mut w1 = matrix::<B>(hidden, inputs, 7);
    let mut b1 = vector::<B>(hidden, 8);
    let mut w2 = matrix::<B>(outputs, hidden, 9);
    let mut b2 = vector::<B>(outputs, 10);
    let (mut r1, mut rb1, mut r2, mut rb2) = (
        Adam::new(1e-3),
        Adam::new(1e-3),
        Adam::new(1e-3),
        Adam::new(1e-3),
    );

    move || {
        let tape = Tape::<B>::new();
        let (w1v, b1v) = (
            tape.matrix(w1.to_backend::<B>()),
            tape.vector(b1.to_backend::<B>()),
        );
        let (w2v, b2v) = (
            tape.matrix(w2.to_backend::<B>()),
            tape.vector(b2.to_backend::<B>()),
        );
        let ones = tape.vector(ones.to_backend::<B>());
        let hidden = (&w1v.matmul(&tape.matrix(x.to_backend::<B>())) + &b1v.outer(&ones)).relu();
        let predicted = &w2v.matmul(&hidden) + &b2v.outer(&ones);
        let residual = &predicted - &tape.matrix(targets.to_backend::<B>());
        let loss = residual
            .frobenius_dot(&residual)
            .scale(1.0 / (outputs * batch) as f32);
        loss.backward();
        r1.update(&mut w1, &w1v.grad());
        rb1.update(&mut b1, &b1v.grad());
        r2.update(&mut w2, &w2v.grad());
        rb2.update(&mut b2, &b2v.grad());
    }
}

/// Fused against unfused, across the shapes neural networks run: activations of
/// growing length, normalization, softmax, a dense layer, and the optimizer.
fn fusion(bench: &mut Bench) {
    const G: &str = "fusion";

    // The longer the chain, the more traffic and launches fusion removes.
    // The last size is 64 MB a tensor, past the caches, where traffic is the cost.
    for len in bench.cfg.pick(&[4_096, 65_536, 1 << 20, 1 << 24]) {
        for activation in Activation::ALL {
            activation_case(bench, activation, len);
        }
    }

    for (rows, cols) in bench.cfg.pick(&[(64, 256), (256, 1024), (1024, 1024)]) {
        let program = layer_norm_program(cols);
        let title = format!("layer norm, {rows}×{cols}");
        fusion_case!(bench, G, title, Work::Elements(rows * cols), 2e-3, |B| {
            layer_norm::<B>(&program, rows, cols)
        });
        // Whatever the backend, each row must come out normalized.
        if bench.cfg.wants(&format!("{G}/{title}")) {
            let out = layer_norm::<Host>(&program, rows, cols)();
            let (gamma, beta) = (values(cols, 2), values(cols, 3));
            let x = values(rows * cols, 1);
            for r in [0, rows - 1] {
                let row = &x[r * cols..(r + 1) * cols];
                let mean = row.iter().map(|&v| v as f64).sum::<f64>() / cols as f64;
                let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / cols as f64;
                let want: Vec<f32> = (0..cols)
                    .map(|c| {
                        ((row[c] as f64 - mean) / (var + 1e-5).sqrt() * gamma[c] as f64
                            + beta[c] as f64) as f32
                    })
                    .collect();
                assert_close("layer norm vs f64", out.row(r), &want, 1e-3);
            }
        }
    }

    let (exp, normalize) = (exp_program(), normalize_program());
    for (features, batch) in bench.cfg.pick(&[(128, 128), (1024, 256), (4096, 512)]) {
        let title = format!("softmax, {features} classes × {batch} samples");
        fusion_case!(
            bench,
            G,
            title,
            Work::Elements(features * batch),
            2e-3,
            |B| softmax::<B>((&exp, &normalize), features, batch)
        );
        if bench.cfg.wants(&format!("{G}/{title}")) {
            let out = softmax::<Host>((&exp, &normalize), features, batch)();
            for c in [0, batch - 1] {
                let total: f32 = (0..features).map(|r| out[(r, c)]).sum();
                assert!(
                    (total - 1.0).abs() < 1e-3,
                    "softmax column {c} sums to {total}"
                );
            }
        }
    }

    let dense = dense_program();
    for (batch, inputs, outputs) in
        bench
            .cfg
            .pick(&[(32, 256, 256), (128, 512, 512), (512, 1024, 1024)])
    {
        let title = format!("dense relu(X·W+b), {batch}×{inputs}×{outputs}");
        let flops = 2.0 * (batch * inputs * outputs) as f64;
        fusion_case!(
            bench,
            G,
            title,
            Work::Flops(flops),
            1e-2,
            |B| dense_layer::<B>(&dense, batch, inputs, outputs)
        );
    }

    for len in bench.cfg.pick(&[65_536, 1 << 20]) {
        let title = format!("Adam update (14 ops), n={len}");
        fusion_case!(
            bench,
            G,
            title,
            Work::Elements(len),
            0.0,
            |B| adam_step::<B>(len)
        );
    }
}

// ---- matmul tiers -------------------------------------------------------------

fn naive_matmul(a: &[f32], b: &[f32], n: usize, out: &mut [f32]) {
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..n {
                sum += a[i * n + p] * b[p * n + j];
            }
            out[i * n + j] = sum;
        }
    }
}

/// Square products through every tier the library has: a scalar loop, the SIMD
/// kernel called directly, the Host API (Accelerate on macOS), and Metal.
fn matmul(bench: &mut Bench) {
    for n in bench.cfg.pick(&[16, 64, 256, 512, 1024, 2048]) {
        let flops = 2.0 * (n * n * n) as f64;
        let Some(mut case) = bench.case("matmul", &format!("{n}×{n}×{n}"), Work::Flops(flops))
        else {
            continue;
        };
        let (a_data, b_data) = (values(n * n, 1), values(n * n, 2));
        let (a, b) = (
            Matrix::<f32>::from_flat(n, n, a_data.clone()),
            Matrix::<f32>::from_flat(n, n, b_data.clone()),
        );
        let host = a.matmul(&b);
        let mut out = vec![0.0f32; n * n];

        // The scalar loop is cubic and unblocked; past 256 it only slows the run.
        if n <= 256 {
            naive_matmul(&a_data, &b_data, n, &mut out);
            assert_close("host matmul vs scalar", host.as_slice(), &out, 2e-3);
            case.row("scalar loop", Drain::None, || {
                naive_matmul(black_box(&a_data), black_box(&b_data), n, &mut out);
                black_box(&mut out);
            });
        }

        #[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            use tensorcrate::simd::f32k;
            f32k::matmul(&a_data, &b_data, n, n, n, &mut out);
            assert_close("simd matmul vs host", &out, host.as_slice(), 2e-3);
            case.row("simd kernel", Drain::None, || {
                f32k::matmul(black_box(&a_data), black_box(&b_data), n, n, n, &mut out);
                black_box(&mut out);
            });
        }

        case.row("host API", Drain::None, || a.matmul(&b));

        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            let (ga, gb) = (a.to_backend::<Metal>(), b.to_backend::<Metal>());
            assert_close(
                "metal matmul vs host",
                &ga.matmul(&gb).flat(),
                host.as_slice(),
                2e-3,
            );
            case.row("metal (latency)", Drain::PerCall, || ga.matmul(&gb));
            case.row("metal (pipelined)", Drain::PerBatch, || ga.matmul(&gb));

            // The matrix units' faster modes: `f32` with relaxed precision, and
            // `f16` operands accumulated into an `f32` result.
            use tensorcrate::metal::{MatmulPrecision, set_matmul_precision};
            set_matmul_precision(MatmulPrecision::Relaxed);
            assert_close_to_scale(
                "relaxed metal matmul vs host",
                &ga.matmul(&gb).flat(),
                host.as_slice(),
                2e-2,
            );
            case.row("metal f32 relaxed (pipelined)", Drain::PerBatch, || {
                ga.matmul(&gb)
            });
            set_matmul_precision(MatmulPrecision::Exact);

            let half = |m: &Matrix<f32>| {
                Matrix::from_flat(
                    n,
                    n,
                    m.as_slice()
                        .iter()
                        .map(|&x| half::f16::from_f32(x))
                        .collect::<Vec<_>>(),
                )
                .to_backend::<Metal>()
            };
            let (ha, hb) = (half(&a), half(&b));
            assert_close_to_scale(
                "f16 metal matmul vs host",
                &ha.matmul_f32(&hb).flat(),
                host.as_slice(),
                2e-2,
            );
            case.row("metal f16 → f32 (pipelined)", Drain::PerBatch, || {
                ha.matmul_f32(&hb)
            });
            assert_close_to_scale(
                "relaxed metal matmul vs host",
                &ha.matmul(&hb).to_f32::<Host>().flat(),
                host.as_slice(),
                2e-2,
            );
            case.row("metal f16 → f16 (pipelined)", Drain::PerBatch, || {
                ha.matmul(&hb)
            });

            // The products a 16-bit matmul's backward pass runs: `C += Aᵀ·B` and
            // `C += A·Bᵀ`, read transposed where they lie and accumulated into
            // `C`. Without TensorOps they are a transpose copy and the tiled
            // kernel, so the third row is what the matrix units save.
            use tensorcrate::metal::set_tensorops;
            use tensorcrate::tensors::Transposed;
            let zero =
                || Matrix::from_flat(n, n, vec![half::f16::ZERO; n * n]).to_backend::<Metal>();
            assert_close_to_scale(
                "f16 metal Aᵀ·B vs host",
                &Metal::matmul_transposed_add(&ha, &hb, Transposed::Left, zero())
                    .to_f32::<Host>()
                    .flat(),
                a.transpose().matmul(&b).as_slice(),
                2e-2,
            );
            for (label, transposed) in [
                ("metal f16 C += Aᵀ·B (pipelined)", Transposed::Left),
                ("metal f16 C += A·Bᵀ (pipelined)", Transposed::Right),
            ] {
                // Accumulating in place, so the timing holds no copy of `C`.
                let mut acc = Some(zero());
                case.row(label, Drain::PerBatch, || {
                    acc = Some(Metal::matmul_transposed_add(
                        &ha,
                        &hb,
                        transposed,
                        acc.take().unwrap(),
                    ));
                });
            }
            set_tensorops(false);
            let mut acc = Some(zero());
            case.row(
                "metal f16 C += Aᵀ·B, tiled kernel (pipelined)",
                Drain::PerBatch,
                || {
                    acc = Some(Metal::matmul_transposed_add(
                        &ha,
                        &hb,
                        Transposed::Left,
                        acc.take().unwrap(),
                    ));
                },
            );
            set_tensorops(true);
        }
    }
}

// ---- simd kernels -------------------------------------------------------------

/// Each kernel as a plain loop, as the direct SIMD kernel, and through the Host
/// API. The gap between the last two is the cost of dispatch and allocation.
#[cfg(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64")))]
fn simd(bench: &mut Bench) {
    use tensorcrate::simd::f32k;
    use tensorcrate::tensors::Reduce;

    const G: &str = "simd";
    let close = |what: &str, got: f32, want: f64| {
        assert!(
            (got as f64 - want).abs() <= 1e-3 * (1.0 + want.abs()),
            "{what}: {got} against {want}"
        );
    };

    for n in bench.cfg.pick(&[4_096, 65_536, 1 << 20]) {
        let (x, y) = (values(n, 1), values(n, 2));
        let (hx, hy) = (Vector::new(x.clone()), Vector::new(y.clone()));
        let mut out = vec![0.0f32; n];
        let mut check = vec![0.0f32; n];
        let bytes = |streams: usize| Work::Bytes(streams * n * 4);

        if let Some(mut case) = bench.case(G, &format!("add, n={n}"), bytes(3)) {
            for i in 0..n {
                check[i] = x[i] + y[i];
            }
            f32k::elementwise(&x, &y, BinaryOp::Add, &mut out);
            assert_eq!(out, check, "simd add");
            assert_eq!((&hx + &hy).as_slice(), &check[..], "host add");
            assert_eq!(
                (hx.clone() + &hy).as_slice(),
                &check[..],
                "host consuming add"
            );
            case.row("scalar loop", Drain::None, || {
                let (x, y) = (black_box(&x), black_box(&y));
                for i in 0..n {
                    out[i] = x[i] + y[i];
                }
                black_box(&mut out);
            });
            case.row("simd kernel", Drain::None, || {
                f32k::elementwise(black_box(&x), black_box(&y), BinaryOp::Add, &mut out);
                black_box(&mut out);
            });
            case.row("host API", Drain::None, || &hx + &hy);
            // The accumulator is moved through the operator and back, so the
            // one allocation is overwritten every call.
            let mut state = Some(hx.clone());
            case.row("host API, consuming", Drain::None, || {
                let sum = state.take().unwrap() + black_box(&hy);
                state = Some(black_box(sum));
            });
        }

        let reference_dot: f64 = x.iter().zip(&y).map(|(&a, &b)| a as f64 * b as f64).sum();
        if let Some(mut case) = bench.case(G, &format!("dot, n={n}"), bytes(2)) {
            close("simd dot", f32k::dot(&x, &y), reference_dot);
            close("host dot", hx.dot(&hy), reference_dot);
            #[cfg(all(feature = "metal", target_os = "macos"))]
            let (gx, gy) = (hx.to_backend::<Metal>(), hy.to_backend::<Metal>());
            #[cfg(all(feature = "metal", target_os = "macos"))]
            close("metal dot", gx.dot(&gy), reference_dot);
            case.row("scalar loop (serial sum)", Drain::None, || {
                let (x, y) = (black_box(&x), black_box(&y));
                let mut sum = 0.0f32;
                for i in 0..n {
                    sum += x[i] * y[i];
                }
                sum
            });
            case.row("simd kernel", Drain::None, || {
                f32k::dot(black_box(&x), black_box(&y))
            });
            case.row("host API", Drain::None, || hx.dot(&hy));
            #[cfg(all(feature = "metal", target_os = "macos"))]
            case.row("metal API (latency)", Drain::None, || gx.dot(&gy));
        }

        let reference_sum: f64 = x.iter().map(|&v| v as f64).sum();
        if let Some(mut case) = bench.case(G, &format!("sum, n={n}"), bytes(1)) {
            close("simd sum", f32k::reduce(&x, Reduce::Sum), reference_sum);
            close("host sum", hx.sum(), reference_sum);
            case.row("scalar loop (serial sum)", Drain::None, || {
                black_box(&x).iter().sum::<f32>()
            });
            case.row("simd kernel", Drain::None, || {
                f32k::reduce(black_box(&x), Reduce::Sum)
            });
            case.row("host API", Drain::None, || hx.sum());
        }

        if let Some(mut case) = bench.case(G, &format!("max, n={n}"), bytes(1)) {
            let want = x.iter().copied().fold(f32::MIN, f32::max);
            assert_eq!(f32k::reduce(&x, Reduce::Max), want, "simd max");
            assert_eq!(hx.reduce(Reduce::Max), want, "host max");
            case.row("scalar loop", Drain::None, || {
                black_box(&x).iter().copied().fold(f32::MIN, f32::max)
            });
            case.row("simd kernel", Drain::None, || {
                f32k::reduce(black_box(&x), Reduce::Max)
            });
            case.row("host API", Drain::None, || hx.reduce(Reduce::Max));
        }

        if let Some(mut case) = bench.case(G, &format!("relu, n={n}"), bytes(2)) {
            for i in 0..n {
                check[i] = x[i].max(0.0);
            }
            f32k::compare_scalar(&x, 0.0, Compare::Max, false, &mut out);
            assert_eq!(out, check, "simd relu");
            assert_eq!(hx.max_scalar(0.0).as_slice(), &check[..], "host relu");
            assert_eq!(
                hx.clone().into_max_scalar(0.0).as_slice(),
                &check[..],
                "host consuming relu"
            );
            case.row("scalar loop", Drain::None, || {
                let x = black_box(&x);
                for i in 0..n {
                    out[i] = x[i].max(0.0);
                }
                black_box(&mut out);
            });
            case.row("simd kernel", Drain::None, || {
                f32k::compare_scalar(black_box(&x), 0.0, Compare::Max, false, &mut out);
                black_box(&mut out);
            });
            case.row("host API", Drain::None, || hx.max_scalar(0.0));
            let mut state = Some(hx.clone());
            case.row("host API, consuming", Drain::None, || {
                state = Some(black_box(state.take().unwrap().into_max_scalar(0.0)));
            });
        }

        // No SIMD entry point for these: the Host API runs the crate's own
        // vectorized `exp` and `tanh`, against the platform's scalar ones.
        for (name, op, scalar) in [
            ("exp", Analytic::Exp, f32::exp as fn(f32) -> f32),
            ("tanh", Analytic::Tanh, f32::tanh as fn(f32) -> f32),
        ] {
            if let Some(mut case) = bench.case(G, &format!("{name}, n={n}"), Work::Elements(n)) {
                for i in 0..n {
                    check[i] = scalar(x[i]);
                }
                assert_close(
                    &format!("host {name}"),
                    hx.analytic(op).as_slice(),
                    &check,
                    1e-5,
                );
                case.row("scalar loop (libm)", Drain::None, || {
                    let x = black_box(&x);
                    for i in 0..n {
                        out[i] = scalar(x[i]);
                    }
                    black_box(&mut out);
                });
                case.row("host API (vectorized)", Drain::None, || hx.analytic(op));
            }
        }
    }
}

#[cfg(not(all(feature = "simd", any(target_arch = "aarch64", target_arch = "x86_64"))))]
fn simd(_: &mut Bench) {
    println!("\n=== simd ===\nskipped: needs the `simd` feature on aarch64 or x86_64");
}

// ---- dispatch -----------------------------------------------------------------

/// What launching work costs: one operation from size 1 to a million elements, a
/// chain of operations growing from one to sixty-four, and the copies between
/// host and device memory.
fn dispatch(bench: &mut Bench) {
    const G: &str = "dispatch";

    // A single multiply. At small sizes the arithmetic is nothing and every row
    // is its launch cost; the resident GPU only wins once the data is large.
    for n in bench.cfg.pick(&[1, 256, 4_096, 65_536, 1 << 20]) {
        let Some(mut case) = bench.case(G, &format!("one multiply, n={n}"), Work::Elements(n))
        else {
            continue;
        };
        let (a, b) = (vector::<Host>(n, 1), vector::<Host>(n, 2));
        let want = (&a * &b).into_vec();
        case.row("host", Drain::None, || &a * &b);

        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            type M = Metal;
            let (ga, gb) = (a.to_backend::<M>(), b.to_backend::<M>());
            let multiply = |x: &Vector<f32, M>, y: &Vector<f32, M>| {
                <M as Kernels>::vector_elementwise(x, y, BinaryOp::Mul)
            };
            assert_close("metal multiply", &multiply(&ga, &gb).flat(), &want, 1e-5);
            case.row("metal resident (latency)", Drain::PerCall, || {
                multiply(&ga, &gb)
            });
            case.row("metal resident (pipelined)", Drain::PerBatch, || {
                multiply(&ga, &gb)
            });
            case.row("metal + transfers", Drain::None, || {
                let (x, y) = (a.to_backend::<M>(), b.to_backend::<M>());
                multiply(&x, &y).to_backend::<Host>()
            });
        }
        let _ = want;
    }

    // A chain of k dependent operations over a small tensor. Unfused, the host
    // makes k passes and Metal k launches; fused, one of each. The slope of the
    // unfused rows is the cost of one more operation.
    let len = 16_384;
    for ops in bench.cfg.pick(&[1, 4, 16, 64]) {
        let program = chain(ops);
        let title = format!(
            "chain of {ops} operation{}, n={len}",
            if ops == 1 { "" } else { "s" }
        );
        fusion_case!(
            bench,
            G,
            title,
            Work::Elements(len),
            2e-3,
            |B| elementwise::<B>(&program, len)
        );
    }

    // Moving data, with nothing computed. Unified memory makes both directions a
    // copy rather than a bus transfer, so these bound what `metal + transfers`
    // adds above.
    for n in bench.cfg.pick(&[65_536, 1 << 20]) {
        let Some(mut case) = bench.case(G, &format!("copies, n={n}"), Work::Bytes(n * 4)) else {
            continue;
        };
        let host = vector::<Host>(n, 1);
        case.row("host clone", Drain::None, || host.clone());
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            let device = host.to_backend::<Metal>();
            assert_eq!(
                device.to_backend::<Host>(),
                host,
                "round trip through Metal"
            );
            case.row("host → metal", Drain::None, || host.to_backend::<Metal>());
            case.row("metal → host", Drain::None, || {
                device.to_backend::<Host>()
            });
        }
    }
}

// ---- training -----------------------------------------------------------------

/// Whole training steps, which is where the other groups' effects compound: the
/// products, the activations, the gradients and the optimizer.
fn training(bench: &mut Bench) {
    for (inputs, hidden, outputs, batch) in
        bench.cfg.pick(&[(64, 256, 16, 128), (256, 1024, 64, 256)])
    {
        let title = format!("MLP step, {inputs}→{hidden}→{outputs}, batch {batch}");
        fusion_case!(
            bench,
            "training",
            title,
            Work::Steps,
            0.0,
            |B| mlp_step::<B>(inputs, hidden, outputs, batch)
        );
    }
}

fn main() {
    let mut bench = Bench {
        cfg: Config::from_args(),
        group: None,
        cases: 0,
    };
    let metal = cfg!(all(feature = "metal", target_os = "macos"));
    let profile = match bench.cfg.profile {
        Profile::Full => "full",
        Profile::Quick => "quick",
        Profile::Smoke => "smoke (one pass; run `cargo bench` to time)",
    };
    println!(
        "tensorcrate nn_ops, {profile} — metal: {}, simd: {}, counters: {}",
        if metal { "on" } else { "off" },
        if cfg!(feature = "simd") { "on" } else { "off" },
        if cfg!(feature = "counters") {
            "on"
        } else {
            "off"
        },
    );

    fusion(&mut bench);
    matmul(&mut bench);
    simd(&mut bench);
    dispatch(&mut bench);
    training(&mut bench);

    if bench.cases == 0 {
        println!("\nno case matches {:?}", bench.cfg.filters);
    }
}
