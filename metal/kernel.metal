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