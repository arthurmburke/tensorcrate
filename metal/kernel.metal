#include <metal_stdlib>
using namespace metal;

#define TILE 16
#define REDUCE_GROUP 256

enum class BinaryOp : ushort {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    Rem = 4
};

// Comparisons and the subgradient they imply. `MaxShare` is the derivative of
// `Max` with respect to its left operand: one where the left is larger, zero
// where it is smaller, and a half where they tie, so a tied maximum splits its
// gradient evenly between the two.
// The last four are predicates, answering 1.0 or 0.0 — the mask a tensor
// algebra with no boolean element type uses. They are the *ordered* comparisons,
// so a NaN operand answers 0.0, which is what `a < b` does on the CPU side.
enum class CompareOp : ushort {
    Min = 0,
    Max = 1,
    MaxShare = 2,
    Less = 3,
    LessEqual = 4,
    Greater = 5,
    GreaterEqual = 6
};

enum class ReduceOp : ushort {
    Sum = 0,
    Min = 1,
    Max = 2
};

enum class AnalyticOp : ushort {
    Sin = 0,
    Cos = 1,
    Tan = 2,
    Sec = 3,
    Csc = 4,
    Arcsin = 5,
    Arccos = 6,
    Arctan = 7,
    Exp = 8,
    Ln = 9,
    Sinh = 10,
    Cosh = 11,
    Tanh = 12,
    Sqrt = 13
};

// Which way a matrix reduction folds. `Rows` folds each row and leaves one
// value per row, matching `tensors::kernels::Axis`.
enum class AxisOp : ushort {
    Rows = 0,
    Columns = 1
};

enum class FamilyOp : ushort {
    Normal = 0,
    InverseGaussian = 1
};

enum class StatisticOp : ushort {
    Pdf = 0,
    Cdf = 1,
    Ppf = 2
};

kernel void matmul_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& K      [[buffer(4)]],
    constant uint& N      [[buffer(5)]],
    constant uint& accumulate [[buffer(6)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]])
{
    threadgroup float Asub[TILE][TILE];
    threadgroup float Bsub[TILE][TILE];

    uint row = gid.y;
    uint col = gid.x;
    float acc = 0.0f;

    uint tiles = (K + TILE - 1) / TILE;
    for (uint t = 0; t < tiles; t++) {
        uint a_col = t * TILE + tid.x;
        uint b_row = t * TILE + tid.y;
        Asub[tid.y][tid.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bsub[tid.y][tid.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint p = 0; p < TILE; p++) {
            acc += Asub[tid.y][p] * Bsub[p][tid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        // `accumulate` adds into C instead of overwriting it, which is what the
        // forward-mode tangent `A'B + AB'` and reverse-mode gradient
        // accumulation both want — one dispatch instead of a separate add.
        C[row * N + col] = accumulate ? C[row * N + col] + acc : acc;
    }
}

kernel void elementwise(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant BinaryOp& op [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    float a = A[i];
    float b = B[i];
    switch (op) {
        case BinaryOp::Add: C[i] = a + b; break;
        case BinaryOp::Sub: C[i] = a - b; break;
        case BinaryOp::Mul: C[i] = a * b; break;
        default: C[i] = a / b; break;
    }
}

// Elementwise `a^b`. Kept out of `elementwise` above rather than added to
// `BinaryOp`, because that enum's contract is that every variant is an operator
// defined for every `Coefficient` — and a power is not: raising an integer to an
// integer leaves the integers. The host side has the same split for the same
// reason.
kernel void power(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    C[i] = pow(A[i], B[i]);
}

// The same with one operand held fixed: `scalar_left` selects `scalar^A[i]`
// over `A[i]^scalar`.
kernel void power_scalar(
    device const float* A      [[buffer(0)]],
    device float* C            [[buffer(1)]],
    constant float& scalar     [[buffer(2)]],
    constant uint& scalar_left [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    float a = scalar_left ? scalar : A[i];
    float b = scalar_left ? A[i] : scalar;
    C[i] = pow(a, b);
}

kernel void broadcast(
    device const float* A [[buffer(0)]],
    device float* C       [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    constant BinaryOp& op  [[buffer(3)]],
    constant uint& scalar_left [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    float a = scalar_left ? scalar : A[i];
    float b = scalar_left ? A[i] : scalar;
    switch (op) {
        case BinaryOp::Add: C[i] = a + b; break;
        case BinaryOp::Sub: C[i] = a - b; break;
        case BinaryOp::Mul: C[i] = a * b; break;
        default: C[i] = a / b; break;
    }
}

inline float compare_values(CompareOp op, float a, float b) {
    switch (op) {
        case CompareOp::Min: return fmin(a, b);
        case CompareOp::Max: return fmax(a, b);
        case CompareOp::MaxShare: return a > b ? 1.0f : (a < b ? 0.0f : 0.5f);
        case CompareOp::Less: return a < b ? 1.0f : 0.0f;
        case CompareOp::LessEqual: return a <= b ? 1.0f : 0.0f;
        case CompareOp::Greater: return a > b ? 1.0f : 0.0f;
        default: return a >= b ? 1.0f : 0.0f;
    }
}

kernel void compare(
    device const float* A  [[buffer(0)]],
    device const float* B  [[buffer(1)]],
    device float* C        [[buffer(2)]],
    constant CompareOp& op [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    C[i] = compare_values(op, A[i], B[i]);
}

kernel void compare_scalar(
    device const float* A      [[buffer(0)]],
    device float* C            [[buffer(1)]],
    constant float& scalar     [[buffer(2)]],
    constant CompareOp& op     [[buffer(3)]],
    constant uint& scalar_left [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    float a = scalar_left ? scalar : A[i];
    float b = scalar_left ? A[i] : scalar;
    C[i] = compare_values(op, a, b);
}

// `fmin`/`fmax` rather than the `clamp` builtin, whose behaviour on a NaN input
// is unspecified: this pair is `x.max(low).min(high)`, the CPU definition.
kernel void clamp_values(
    device const float* A  [[buffer(0)]],
    device float* C        [[buffer(1)]],
    constant float& low    [[buffer(2)]],
    constant float& high   [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    C[i] = fmin(fmax(A[i], low), high);
}

inline float reduce_values(ReduceOp op, float a, float b) {
    switch (op) {
        case ReduceOp::Sum: return a + b;
        case ReduceOp::Min: return fmin(a, b);
        default: return fmax(a, b);
    }
}

inline float reduce_identity(ReduceOp op) {
    switch (op) {
        case ReduceOp::Sum: return 0.0f;
        case ReduceOp::Min: return INFINITY;
        default: return -INFINITY;
    }
}

// One round of a tree reduction: each threadgroup folds its own slice and writes
// a single partial, so the host re-dispatches over the partials until one value
// is left. Threads past the end read the identity, which is why the fold has to
// be over an associative operation with one.
//
// Dispatched as whole threadgroups (never `dispatchThreads`): every thread in a
// group must reach the barriers, and a ragged final group would not.
kernel void reduce_partial(
    device const float* input  [[buffer(0)]],
    device float* partials     [[buffer(1)]],
    constant uint& count       [[buffer(2)]],
    constant ReduceOp& op      [[buffer(3)]],
    uint gid   [[thread_position_in_grid]],
    uint tid   [[thread_position_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]],
    uint width [[threads_per_threadgroup]])
{
    threadgroup float scratch[REDUCE_GROUP];
    scratch[tid] = gid < count ? input[gid] : reduce_identity(op);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = width / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = reduce_values(op, scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        partials[group] = scratch[0];
    }
}

// One sweep of an inclusive Hillis–Steele scan, from `input` into `output`. The
// host runs it for offsets 1, 2, 4, … and swaps the buffers between sweeps: the
// pass cannot be done in place, since a thread reading `i - offset` would race
// the thread writing it.
//
// `log n` sweeps of `n` adds is more arithmetic than the serial `n`, which is
// the trade a scan makes to have any parallelism at all.
kernel void scan_step(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    constant uint& offset     [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    output[i] = i >= offset ? input[i] + input[i - offset] : input[i];
}

// The unsigned key that sorts floats in IEEE total order — the order
// `f32::total_cmp` gives, and the reason this sort agrees with the CPU one on
// NaNs and on -0.0 rather than only on ordinary values. Positive floats already
// compare correctly as integers once the sign bit is set; negative ones need
// every bit inverted, which both flips the sign bit and reverses the magnitude.
inline uint sort_key(float value) {
    uint bits = as_type<uint>(value);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Copy into a power-of-two buffer, filling the tail with a value that sorts to
// the end so the padding trims cleanly afterwards.
kernel void sort_prepare(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    constant uint& count      [[buffer(2)]],
    constant float& padding   [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    output[i] = i < count ? input[i] : padding;
}

// One compare-exchange stage of a bitonic sort. `block` is the width of the
// bitonic sequence being merged and `stride` the distance between partners;
// within a block the direction alternates, which is what builds the next
// sequence up. Only the lower index of each pair does the work, so the stage's
// writes are disjoint and need no synchronization beyond the dispatch boundary.
kernel void bitonic_stage(
    device float* values   [[buffer(0)]],
    constant uint& block   [[buffer(1)]],
    constant uint& stride  [[buffer(2)]],
    constant uint& ascending [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    uint partner = i ^ stride;
    if (partner <= i) return;

    bool up = ((i & block) == 0) == (ascending != 0);
    float a = values[i];
    float b = values[partner];
    if ((sort_key(a) > sort_key(b)) == up) {
        values[i] = b;
        values[partner] = a;
    }
}

// Valid cross-correlation: every output element is the window of `input` under
// `weights`, summed. `flip` reverses the window, which turns cross-correlation
// into convolution proper — and is what the input-side gradient needs.
kernel void correlate(
    device const float* input   [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output        [[buffer(2)]],
    constant uint& rows         [[buffer(3)]],
    constant uint& cols         [[buffer(4)]],
    constant uint& window_rows  [[buffer(5)]],
    constant uint& window_cols  [[buffer(6)]],
    constant uint& flip         [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint out_rows = rows - window_rows + 1;
    uint out_cols = cols - window_cols + 1;
    if (gid.y >= out_rows || gid.x >= out_cols) return;

    float acc = 0.0f;
    for (uint wr = 0; wr < window_rows; wr++) {
        for (uint wc = 0; wc < window_cols; wc++) {
            uint tap_row = flip ? window_rows - 1 - wr : wr;
            uint tap_col = flip ? window_cols - 1 - wc : wc;
            acc += input[(gid.y + wr) * cols + (gid.x + wc)]
                 * weights[tap_row * window_cols + tap_col];
        }
    }
    output[gid.y * out_cols + gid.x] = acc;
}

// Reverse both axes. The kernel-side gradient of a convolution is the flip of
// the correlation's, so this is what lets both conventions differentiate.
kernel void flip_both(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    constant uint& rows       [[buffer(2)]],
    constant uint& cols       [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.y >= rows || gid.x >= cols) return;
    output[gid.y * cols + gid.x] = input[(rows - 1 - gid.y) * cols + (cols - 1 - gid.x)];
}

// Surround a matrix with zeros. The input-side gradient of a valid correlation
// is a full one, and padding is how a full correlation is spelled.
kernel void pad_zeros(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    constant uint& rows       [[buffer(2)]],
    constant uint& cols       [[buffer(3)]],
    constant uint& pad_rows   [[buffer(4)]],
    constant uint& pad_cols   [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint out_rows = rows + 2 * pad_rows;
    uint out_cols = cols + 2 * pad_cols;
    if (gid.y >= out_rows || gid.x >= out_cols) return;

    bool inside = gid.y >= pad_rows && gid.y < pad_rows + rows
               && gid.x >= pad_cols && gid.x < pad_cols + cols;
    output[gid.y * out_cols + gid.x] =
        inside ? input[(gid.y - pad_rows) * cols + (gid.x - pad_cols)] : 0.0f;
}

// Copy one input vector into a row or column of a row-major output matrix.
// `output_stride == 1` writes a contiguous row; otherwise it scatters a column.
kernel void stack_vector(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    constant uint& count      [[buffer(2)]],
    constant uint& offset     [[buffer(3)]],
    constant uint& output_stride [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i < count) {
        output[offset + i * output_stride] = input[i];
    }
}

// Concatenate two row-major matrices with the same row count. Each thread
// writes one output element; both input reads and output writes are coalesced.
kernel void concat_horizontal(
    device const float* left  [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device float* output      [[buffer(2)]],
    constant uint& rows       [[buffer(3)]],
    constant uint& left_cols  [[buffer(4)]],
    constant uint& right_cols [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint output_cols = left_cols + right_cols;
    uint row = gid.y;
    uint col = gid.x;
    if (row >= rows || col >= output_cols) return;

    output[row * output_cols + col] = col < left_cols
        ? left[row * left_cols + col]
        : right[row * right_cols + col - left_cols];
}

// Place one matrix into a horizontal block of a wider row-major matrix. The
// encoder dispatches this once per input matrix in the same command buffer.
kernel void merge_horizontal(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    constant uint& rows       [[buffer(2)]],
    constant uint& input_cols [[buffer(3)]],
    constant uint& output_cols [[buffer(4)]],
    constant uint& col_offset [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    if (row < rows && col < input_cols) {
        output[row * output_cols + col_offset + col] =
            input[row * input_cols + col];
    }
}

// Transpose through a padded threadgroup tile. Adjacent threads read adjacent
// input values and write adjacent output values; the extra column avoids bank
// conflicts when the tile is read in the opposite direction.
kernel void transpose_tiled(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    constant uint& rows       [[buffer(2)]],
    constant uint& cols       [[buffer(3)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]])
{
    threadgroup float tile[TILE][TILE + 1];

    uint input_col = group.x * TILE + tid.x;
    uint input_row = group.y * TILE + tid.y;
    if (input_row < rows && input_col < cols) {
        tile[tid.y][tid.x] = input[input_row * cols + input_col];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint output_col = group.y * TILE + tid.x;
    uint output_row = group.x * TILE + tid.y;
    if (output_row < cols && output_col < rows) {
        output[output_row * rows + output_col] = tile[tid.x][tid.y];
    }
}

// These variants must agree with `tensors::kernels::Analytic`, and each
// derivative must match the corresponding `Dual` implementation.
inline float analytic_value(AnalyticOp op, float x) {
    switch (op) {
        case AnalyticOp::Sin:    return sin(x);
        case AnalyticOp::Cos:    return cos(x);
        case AnalyticOp::Tan:    return tan(x);
        case AnalyticOp::Sec:    return 1.0f / cos(x);
        case AnalyticOp::Csc:    return 1.0f / sin(x);
        case AnalyticOp::Arcsin: return asin(x);
        case AnalyticOp::Arccos: return acos(x);
        case AnalyticOp::Arctan: return atan(x);
        case AnalyticOp::Exp:    return exp(x);
        case AnalyticOp::Ln:     return log(x);
        case AnalyticOp::Sinh:   return sinh(x);
        case AnalyticOp::Cosh:   return cosh(x);
        case AnalyticOp::Tanh:   return tanh(x);
        case AnalyticOp::Sqrt:   return sqrt(x);
        default: return NAN;
    }
}

inline float analytic_derivative(AnalyticOp op, float x) {
    switch (op) {
        case AnalyticOp::Sin:    return cos(x);
        case AnalyticOp::Cos:    return -sin(x);
        case AnalyticOp::Tan:    { float c = cos(x); return 1.0f / (c * c); }
        case AnalyticOp::Sec:    { float c = cos(x); return sin(x) / (c * c); }
        case AnalyticOp::Csc:    { float s = sin(x); return -cos(x) / (s * s); }
        case AnalyticOp::Arcsin: return 1.0f / sqrt(1.0f - x * x);
        case AnalyticOp::Arccos: return -1.0f / sqrt(1.0f - x * x);
        case AnalyticOp::Arctan: return 1.0f / (1.0f + x * x);
        case AnalyticOp::Exp:    return exp(x);
        case AnalyticOp::Ln:     return 1.0f / x;
        case AnalyticOp::Sinh:   return cosh(x);
        case AnalyticOp::Cosh:   return sinh(x);
        case AnalyticOp::Tanh:   { float t = tanh(x); return 1.0f - t * t; }
        case AnalyticOp::Sqrt:   return 0.5f * rsqrt(x);
        default: return NAN;
    }
}

kernel void unary(
    device const float* A [[buffer(0)]],
    device float* C       [[buffer(1)]],
    constant AnalyticOp& op [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    C[i] = analytic_value(op, A[i]);
}

// Forward-mode AD: `f(v + d·ε) = f(v) + f'(v)·d·ε`. Both parts come out of one
// dispatch, which also reads `v` only once.
kernel void unary_dual(
    device const float* value    [[buffer(0)]],
    device const float* tangent  [[buffer(1)]],
    device float* out_value      [[buffer(2)]],
    device float* out_tangent    [[buffer(3)]],
    constant AnalyticOp& op      [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    float v = value[i];
    out_value[i] = analytic_value(op, v);
    out_tangent[i] = analytic_derivative(op, v) * tangent[i];
}

kernel void fft_bit_reverse(
    device const float2* input [[buffer(0)]],
    device float2* output      [[buffer(1)]],
    constant uint& count       [[buffer(2)]],
    constant uint& bits        [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= count) return;
    uint value = i;
    uint reversed = 0;
    for (uint bit = 0; bit < bits; bit++) {
        reversed = (reversed << 1) | (value & 1);
        value >>= 1;
    }
    output[reversed] = input[i];
}

kernel void fft_stage(
    device float2* values       [[buffer(0)]],
    constant uint& count        [[buffer(1)]],
    constant uint& stage_length [[buffer(2)]],
    constant uint& inverse      [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    // `half` is a type name in MSL, so the span cannot be called that.
    uint half_length = stage_length / 2;
    uint butterflies = count / 2;
    if (i >= butterflies) return;

    uint group = i / half_length;
    uint offset = i % half_length;
    uint even_index = group * stage_length + offset;
    uint odd_index = even_index + half_length;

    float direction = inverse ? 1.0f : -1.0f;
    float angle = direction * 2.0f * M_PI_F * float(offset) / float(stage_length);
    float2 twiddle = float2(cos(angle), sin(angle));
    float2 odd_value = values[odd_index];
    float2 rotated = float2(
        odd_value.x * twiddle.x - odd_value.y * twiddle.y,
        odd_value.y * twiddle.x + odd_value.x * twiddle.y
    );
    float2 even_value = values[even_index];
    float normalization = (inverse && stage_length == count) ? float(count) : 1.0f;
    values[even_index] = (even_value + rotated) / normalization;
    values[odd_index] = (even_value - rotated) / normalization;
}
// ---- statistics -------------------------------------------------------------

// The second pass of a two-pass variance, fused with the first round of the
// tree reduction: the deviation is formed and squared as the value is read, so
// the whole pass costs one traversal rather than an elementwise kernel plus a
// separate fold. Later rounds are ordinary `reduce_partial` sums over the
// partials this leaves behind.
//
// Dispatched as whole threadgroups, for the reason `reduce_partial` is: every
// thread has to reach the barriers.
kernel void deviation_partial(
    device const float* input [[buffer(0)]],
    device float* partials    [[buffer(1)]],
    constant uint& count      [[buffer(2)]],
    constant float& mean      [[buffer(3)]],
    uint gid   [[thread_position_in_grid]],
    uint tid   [[thread_position_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]],
    uint width [[threads_per_threadgroup]])
{
    threadgroup float scratch[REDUCE_GROUP];
    float deviation = gid < count ? input[gid] - mean : 0.0f;
    scratch[tid] = deviation * deviation;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = width / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        partials[group] = scratch[0];
    }
}

// One thread per row or per column, each taking both passes over its own slice.
//
// A thread per output rather than a threadgroup per output: the slices are
// usually short next to the number of them, so the parallelism is across
// slices. Along `Rows` the reads are contiguous per thread; along `Columns`
// they are strided per thread but *adjacent threads read adjacent elements*,
// which is the coalesced pattern — the two axes trade which of the two
// localities they get, and neither is the pathological case.
kernel void axis_moments(
    device const float* input  [[buffer(0)]],
    device float* means        [[buffer(1)]],
    device float* deviations   [[buffer(2)]],
    constant uint& rows        [[buffer(3)]],
    constant uint& cols        [[buffer(4)]],
    constant AxisOp& axis      [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    bool by_row = axis == AxisOp::Rows;
    uint count = by_row ? cols : rows;
    uint stride = by_row ? 1u : cols;
    uint base = by_row ? i * cols : i;

    float total = 0.0f;
    for (uint k = 0; k < count; ++k) {
        total += input[base + k * stride];
    }
    float mean = total / float(count);

    float deviation = 0.0f;
    for (uint k = 0; k < count; ++k) {
        float d = input[base + k * stride] - mean;
        deviation += d * d;
    }

    means[i] = mean;
    deviations[i] = deviation;
}

// These must agree with `statistics::special`, which is the same mathematics in
// double precision. They do not agree bit for bit and are not meant to: the
// host rounds a `f64` result once, while everything here is `f32` throughout.

// MSL has no `erf` or `erfc`, so this is the Chebyshev-fitted form: one
// exponential and a ninth-degree polynomial in `t = 2/(2 + x)`, with a
// fractional error below 1.2e-7 everywhere — about one `f32` ulp, which is as
// close as this tier can get. The host's rational approximation is far more
// accurate and far more branchy; both are the right choice for their precision.
inline float error_function_complement(float x) {
    float magnitude = fabs(x);
    float t = 2.0f / (2.0f + magnitude);
    float poly = -1.26551223f + t * (1.00002368f + t * (0.37409196f + t * (0.09678418f
        + t * (-0.18628806f + t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f
        + t * (-0.82215223f + t * 0.17087277f))))))));
    float value = t * exp(-magnitude * magnitude + poly);
    return x >= 0.0f ? value : 2.0f - value;
}

inline float standard_normal_pdf(float z) {
    return 0.3989422804014327f * exp(-0.5f * z * z);
}

inline float standard_normal_cdf(float z) {
    return 0.5f * error_function_complement(-z * M_SQRT1_2_F);
}

// `log(Phi(z))`, which stays finite past the point where `Phi` itself
// underflows. Below the switch the asymptotic expansion of the tail is used;
// above it, the logarithm of the ordinary value.
inline float standard_normal_ln_cdf(float z) {
    if (z > -5.0f) {
        return log(0.5f * error_function_complement(-z * M_SQRT1_2_F));
    }
    float w = 1.0f / (z * z);
    float series = 1.0f + w * (-1.0f + w * (3.0f - 15.0f * w));
    return -0.5f * z * z - 0.9189385332046727f - log(-z) + log(series);
}

// Wichura's AS 241, in single precision. The three branches meet where the
// central rational stops being accurate and where the tail parameterization
// switches; each is a plain polynomial ratio.
inline float standard_normal_ppf(float p) {
    if (isnan(p) || p < 0.0f || p > 1.0f) return NAN;
    if (p <= 0.0f) return -INFINITY;
    if (p >= 1.0f) return INFINITY;

    float q = p - 0.5f;
    if (fabs(q) <= 0.425f) {
        float r = 0.180625f - q * q;
        float num = 2509.0809287301226727f;
        num = num * r + 33430.575583588128105f;
        num = num * r + 67265.770927008700853f;
        num = num * r + 45921.953931549871457f;
        num = num * r + 13731.693765509461125f;
        num = num * r + 1971.5909503065514427f;
        num = num * r + 133.14166789178437745f;
        num = num * r + 3.387132872796366608f;
        float den = 5226.495278852854561f;
        den = den * r + 28729.085735721942674f;
        den = den * r + 39307.89580009271061f;
        den = den * r + 21213.794301586595867f;
        den = den * r + 5394.1960214247511077f;
        den = den * r + 687.1870074920579083f;
        den = den * r + 42.313330701600911252f;
        den = den * r + 1.0f;
        return q * num / den;
    }

    float tail = q < 0.0f ? p : 1.0f - p;
    float r = sqrt(-log(tail));
    float value;
    if (r <= 5.0f) {
        r -= 1.6f;
        float num = 7.7454501427834140764e-4f;
        num = num * r + 0.0227238449892691845833f;
        num = num * r + 0.24178072517745061177f;
        num = num * r + 1.27045825245236838258f;
        num = num * r + 3.64784832476320460504f;
        num = num * r + 5.7694972214606914055f;
        num = num * r + 4.6303378461565452959f;
        num = num * r + 1.42343711074968357734f;
        float den = 1.05075007164441684324e-9f;
        den = den * r + 5.475938084995344946e-4f;
        den = den * r + 0.0151986665636164571966f;
        den = den * r + 0.14810397642748007459f;
        den = den * r + 0.68976733498510000455f;
        den = den * r + 1.6763848301838038494f;
        den = den * r + 2.05319162663775882187f;
        den = den * r + 1.0f;
        value = num / den;
    } else {
        r -= 5.0f;
        float num = 2.01033439929228813265e-7f;
        num = num * r + 2.71155556874348757815e-5f;
        num = num * r + 0.0012426609473880784386f;
        num = num * r + 0.026532189526576123093f;
        num = num * r + 0.29656057182850489123f;
        num = num * r + 1.7848265399172913358f;
        num = num * r + 5.4637849111641143699f;
        num = num * r + 6.6579046435011037772f;
        float den = 2.04426310338993978564e-15f;
        den = den * r + 1.4215117583164458887e-7f;
        den = den * r + 1.8463183175100546818e-5f;
        den = den * r + 7.868691311456132591e-4f;
        den = den * r + 0.0148753612908506148525f;
        den = den * r + 0.13692988092273580531f;
        den = den * r + 0.59983220655588793769f;
        den = den * r + 1.0f;
        value = num / den;
    }
    return q < 0.0f ? -value : value;
}

inline float normal_pdf(float x, float mean, float stddev) {
    if (!(stddev > 0.0f)) return NAN;
    return standard_normal_pdf((x - mean) / stddev) / stddev;
}

inline float normal_cdf(float x, float mean, float stddev) {
    if (!(stddev > 0.0f)) return NAN;
    return standard_normal_cdf((x - mean) / stddev);
}

inline float normal_ppf(float p, float mean, float stddev) {
    if (!(stddev > 0.0f)) return NAN;
    return mean + stddev * standard_normal_ppf(p);
}

inline float inverse_gaussian_pdf(float x, float mean, float shape) {
    if (!(mean > 0.0f) || !(shape > 0.0f)) return NAN;
    if (x <= 0.0f) return 0.0f;
    float deviation = x - mean;
    float exponent = -shape * deviation * deviation / (2.0f * mean * mean * x);
    return sqrt(shape / (2.0f * M_PI_F * x * x * x)) * exp(exponent);
}

// `Phi(a) + exp(2*shape/mean) * Phi(-b)`, with the second term formed in
// logarithms: the factor overflows and the probability underflows at the same
// rate, and only their product is representable.
inline float inverse_gaussian_cdf(float x, float mean, float shape) {
    if (!(mean > 0.0f) || !(shape > 0.0f)) return NAN;
    if (x <= 0.0f) return 0.0f;
    float scale = sqrt(shape / x);
    float ratio = x / mean;
    float lower = scale * (ratio - 1.0f);
    float upper = scale * (ratio + 1.0f);
    float tail = 2.0f * shape / mean + standard_normal_ln_cdf(-upper);
    return fmin(standard_normal_cdf(lower) + exp(tail), 1.0f);
}

// No elementary inverse, so this brackets the answer and then takes Newton
// steps that are rejected in favour of a bisection whenever they would leave
// the bracket. Thirty passes is past convergence for `f32` and bounds the
// worst case at a bracket narrower than the floats in it.
inline float inverse_gaussian_ppf(float p, float mean, float shape) {
    if (!(mean > 0.0f) || !(shape > 0.0f) || isnan(p) || p < 0.0f || p > 1.0f) return NAN;
    if (p <= 0.0f) return 0.0f;
    if (p >= 1.0f) return INFINITY;

    float low = 0.0f;
    float high = mean;
    for (uint i = 0; i < 128u; ++i) {
        if (inverse_gaussian_cdf(high, mean, shape) >= p) break;
        low = high;
        high *= 2.0f;
        if (isinf(high)) return INFINITY;
    }

    float guess = 0.5f * (low + high);
    for (uint i = 0; i < 30u; ++i) {
        float error = inverse_gaussian_cdf(guess, mean, shape) - p;
        if (error > 0.0f) {
            high = guess;
        } else {
            low = guess;
        }
        float density = inverse_gaussian_pdf(guess, mean, shape);
        float step = density > 0.0f ? guess - error / density : NAN;
        float next = (isfinite(step) && step > low && step < high)
            ? step
            : 0.5f * (low + high);
        if (next == guess) break;
        guess = next;
    }
    return guess;
}

inline float distribution_value(
    FamilyOp family,
    StatisticOp statistic,
    float x,
    float first,
    float second)
{
    if (family == FamilyOp::Normal) {
        switch (statistic) {
            case StatisticOp::Pdf: return normal_pdf(x, first, second);
            case StatisticOp::Cdf: return normal_cdf(x, first, second);
            default:               return normal_ppf(x, first, second);
        }
    }
    switch (statistic) {
        case StatisticOp::Pdf: return inverse_gaussian_pdf(x, first, second);
        case StatisticOp::Cdf: return inverse_gaussian_cdf(x, first, second);
        default:               return inverse_gaussian_ppf(x, first, second);
    }
}

// One distribution over the whole tensor.
kernel void distribution(
    device const float* input     [[buffer(0)]],
    device float* output          [[buffer(1)]],
    constant FamilyOp& family     [[buffer(2)]],
    constant StatisticOp& stat    [[buffer(3)]],
    constant float& first         [[buffer(4)]],
    constant float& second        [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    output[i] = distribution_value(family, stat, input[i], first, second);
}

// One distribution per row or per column: the parameter pair is looked up by
// the element's position along the axis, which is what turns a per-row fit into
// a single dispatch rather than one per row.
kernel void axis_distribution(
    device const float* input     [[buffer(0)]],
    device float* output          [[buffer(1)]],
    device const float* first     [[buffer(2)]],
    device const float* second    [[buffer(3)]],
    constant uint& cols           [[buffer(4)]],
    constant AxisOp& axis         [[buffer(5)]],
    constant FamilyOp& family     [[buffer(6)]],
    constant StatisticOp& stat    [[buffer(7)]],
    uint i [[thread_position_in_grid]])
{
    uint along = axis == AxisOp::Rows ? i / cols : i % cols;
    output[i] = distribution_value(family, stat, input[i], first[along], second[along]);
}

// ---- fused elementwise programs ----------------------------------------------
//
// One thread runs a whole register program for one element; see
// `tensors::fused`. Every thread runs the same instructions in the same order,
// so the `switch` on each opcode never diverges within a SIMD group, and the
// intermediates stay in the `r` array rather than going back to memory.

#define FUSED_REGISTERS 16

// Must match `fused::Encoded`: twelve bytes, four-aligned.
struct FusedInstr {
    ushort kind;   // 0 load, 1 const, 2 binary, 3 unary, 4 compare, 5 store
    ushort op;     // remap, BinaryOp, AnalyticOp or CompareOp
    uchar dst;
    uchar a;       // load: input slot; others: register
    uchar b;       // load/store: storage type; binary/compare: register
    uchar aux;     // store: output slot
    float value;   // const
};

struct FusedShape {
    uint rows;
    uint cols;
    uint count;    // instructions
};

// Storage types, matching `fused::DType`.
inline float fused_load(device const uchar* base, uint dtype, uint index) {
    switch (dtype) {
        case 1: return float(((device const half*)base)[index]);
        // bf16 is the top half of an f32, so widening is a shift.
        case 2: return as_type<float>(uint(((device const ushort*)base)[index]) << 16);
        default: return ((device const float*)base)[index];
    }
}

// Round to nearest, ties to even, quieting NaNs — the same rule as
// `half::bf16::from_f32`, so both backends narrow a given f32 identically.
inline ushort fused_to_bf16(float value) {
    uint x = as_type<uint>(value);
    if ((x & 0x7fffffffu) > 0x7f800000u) {
        return ushort((x >> 16) | 0x0040u);
    }
    uint round_bit = 0x00008000u;
    if ((x & round_bit) != 0 && (x & (3u * round_bit - 1u)) != 0) {
        return ushort(x >> 16) + 1;
    }
    return ushort(x >> 16);
}

inline void fused_store(device uchar* base, uint dtype, uint index, float value) {
    switch (dtype) {
        case 1: ((device half*)base)[index] = half(value); break;
        case 2: ((device ushort*)base)[index] = fused_to_bf16(value); break;
        default: ((device float*)base)[index] = value; break;
    }
}

// Which input element feeds output element `i`, matching `fused::Remap`.
inline uint fused_remap(ushort remap, uint i, uint rows, uint cols) {
    switch (remap) {
        case 1: return (i % cols) * rows + i / cols;  // transpose
        case 2: return i % cols;                      // row vector, down every row
        case 3: return i / cols;                      // column vector, across every column
        default: return i;
    }
}

inline float fused_binary(BinaryOp op, float a, float b) {
    switch (op) {
        case BinaryOp::Add: return a + b;
        case BinaryOp::Sub: return a - b;
        case BinaryOp::Mul: return a * b;
        default: return a / b;
    }
}

// Sixteen input and eight output slots, plus the program and its shape: 26 of
// Metal's 31 buffer arguments. Unused slots are bound to a used buffer, which is
// never touched because no instruction names them. A tensor updated in place is
// bound to both an input and an output slot; the program reads it before it
// stores it, and each thread touches only its own element.
kernel void fused_elementwise(
    constant FusedInstr* code   [[buffer(0)]],
    constant FusedShape& shape  [[buffer(1)]],
    device const uchar* in0  [[buffer(2)]],
    device const uchar* in1  [[buffer(3)]],
    device const uchar* in2  [[buffer(4)]],
    device const uchar* in3  [[buffer(5)]],
    device const uchar* in4  [[buffer(6)]],
    device const uchar* in5  [[buffer(7)]],
    device const uchar* in6  [[buffer(8)]],
    device const uchar* in7  [[buffer(9)]],
    device const uchar* in8  [[buffer(10)]],
    device const uchar* in9  [[buffer(11)]],
    device const uchar* in10 [[buffer(12)]],
    device const uchar* in11 [[buffer(13)]],
    device const uchar* in12 [[buffer(14)]],
    device const uchar* in13 [[buffer(15)]],
    device const uchar* in14 [[buffer(16)]],
    device const uchar* in15 [[buffer(17)]],
    device uchar* out0 [[buffer(18)]],
    device uchar* out1 [[buffer(19)]],
    device uchar* out2 [[buffer(20)]],
    device uchar* out3 [[buffer(21)]],
    device uchar* out4 [[buffer(22)]],
    device uchar* out5 [[buffer(23)]],
    device uchar* out6 [[buffer(24)]],
    device uchar* out7 [[buffer(25)]],
    uint i [[thread_position_in_grid]])
{
    device const uchar* inputs[16] = {
        in0, in1, in2, in3, in4, in5, in6, in7,
        in8, in9, in10, in11, in12, in13, in14, in15
    };
    device uchar* outputs[8] = { out0, out1, out2, out3, out4, out5, out6, out7 };

    float r[FUSED_REGISTERS];
    for (uint pc = 0; pc < shape.count; pc++) {
        FusedInstr instr = code[pc];
        switch (instr.kind) {
            case 0:
                r[instr.dst] = fused_load(
                    inputs[instr.a], instr.b,
                    fused_remap(instr.op, i, shape.rows, shape.cols));
                break;
            case 1:
                r[instr.dst] = instr.value;
                break;
            case 2:
                r[instr.dst] = fused_binary(BinaryOp(instr.op), r[instr.a], r[instr.b]);
                break;
            case 3:
                r[instr.dst] = analytic_value(AnalyticOp(instr.op), r[instr.a]);
                break;
            case 4:
                r[instr.dst] = compare_values(CompareOp(instr.op), r[instr.a], r[instr.b]);
                break;
            default:
                fused_store(outputs[instr.aux], instr.b, i, r[instr.a]);
                break;
        }
    }
}
