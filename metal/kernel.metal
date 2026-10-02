#include "common.h"

#define TILE 16
#define REDUCE_GROUP 256

// Every typed kernel below is a template over its element type, instantiated
// once per type the `Metal` backend computes in. The host picks the pipeline by
// appending the suffix: `elementwise_f16`, `reduce_partial_bf16`, …
//
// Arithmetic runs in the element type itself — `half` and `bfloat` operators
// round every result to that type, exactly as the host's `f16`/`bf16` do. The
// one exception is accumulation: sums, products, convolutions and moments fold
// in `float` and round once at the end, so a long reduction keeps `float`'s
// precision instead of losing it to a 16-bit running total. MSL's math library
// has no `bfloat` overloads, so `bfloat` transcendentals evaluate in `float`
// and round once, as the host's do.
#define FOR_EACH_ELEMENT(M) M(float, f32) M(half, f16) M(bfloat, bf16)

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

// One output element of `A·B`, accumulated in `float` over `TILE`-wide
// tiles staged in threadgroup memory. Every thread of the group must call it,
// in or out of range, because of the barriers.
template <typename T>
inline float tiled_product(
    device const T* A,
    device const T* B,
    uint M,
    uint K,
    uint N,
    uint2 tid,
    uint2 gid,
    threadgroup float (*Asub)[TILE],
    threadgroup float (*Bsub)[TILE])
{
    uint row = gid.y;
    uint col = gid.x;
    float acc = 0.0f;

    uint tiles = (K + TILE - 1) / TILE;
    for (uint t = 0; t < tiles; t++) {
        uint a_col = t * TILE + tid.x;
        uint b_row = t * TILE + tid.y;
        // Staged in `float`: the products of two 16-bit values are exact there,
        // and the running sum is the accumulator.
        Asub[tid.y][tid.x] = (row < M && a_col < K) ? float(A[row * K + a_col]) : 0.0f;
        Bsub[tid.y][tid.x] = (b_row < K && col < N) ? float(B[b_row * N + col]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint p = 0; p < TILE; p++) {
            acc += Asub[tid.y][p] * Bsub[p][tid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return acc;
}

template <typename T>
kernel void matmul_tiled(
    device const T* A [[buffer(0)]],
    device const T* B [[buffer(1)]],
    device T* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& K      [[buffer(4)]],
    constant uint& N      [[buffer(5)]],
    constant uint& accumulate [[buffer(6)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]])
{
    threadgroup float Asub[TILE][TILE];
    threadgroup float Bsub[TILE][TILE];
    float acc = tiled_product(A, B, M, K, N, tid, gid, Asub, Bsub);

    uint row = gid.y;
    uint col = gid.x;
    if (row < M && col < N) {
        // `accumulate` adds into C instead of overwriting it, which is what the
        // forward-mode tangent `A'B + AB'` and reverse-mode gradient
        // accumulation both want — one dispatch instead of a separate add.
        C[row * N + col] = T(accumulate ? float(C[row * N + col]) + acc : acc);
    }
}

#define INSTANTIATE_MATMUL(T, S)                                               \
template [[host_name("matmul_tiled_" #S)]] kernel void matmul_tiled<T>(        \
    device const T*, device const T*, device T*, constant uint&,               \
    constant uint&, constant uint&, constant uint&, uint2, uint2);
FOR_EACH_ELEMENT(INSTANTIATE_MATMUL)

// Matrix–vector products, which as matrix products have a single row or
// column of output and so too few tiles to occupy the GPU.
//
// `y = M·x`, plus `y` if accumulating: one SIMD group per row of `M`, its
// lanes reading the row a stride apart so that together they read whole
// cache lines, then summing their partial results.
template <typename T>
kernel void matvec_rows(
    device const T* M     [[buffer(0)]],
    device const T* x     [[buffer(1)]],
    device T* y           [[buffer(2)]],
    constant uint& rows   [[buffer(3)]],
    constant uint& cols   [[buffer(4)]],
    constant uint& accumulate [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simds [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint row = group * simds + simd;
    if (row >= rows) {
        return;
    }
    device const T* line = M + ulong(row) * cols;
    float acc = 0.0f;
    for (uint c = lane; c < cols; c += 32) {
        acc += float(line[c]) * float(x[c]);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        y[row] = T(accumulate ? float(y[row]) + acc : acc);
    }
}

// `y = v·M` in two passes. The first sums each column over a band of rows —
// adjacent threads take adjacent columns, so reads are whole cache lines — and
// the second adds the bands of each column in order, so the result does not
// depend on how the work was scheduled.
template <typename T>
kernel void vecmat_bands(
    device const T* v        [[buffer(0)]],
    device const T* M        [[buffer(1)]],
    device float* partial    [[buffer(2)]],
    constant uint& rows      [[buffer(3)]],
    constant uint& cols      [[buffer(4)]],
    constant uint& band      [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint col = gid.x;
    if (col >= cols) {
        return;
    }
    uint first = gid.y * band;
    uint last = min(first + band, rows);
    float acc = 0.0f;
    for (uint r = first; r < last; r++) {
        acc += float(v[r]) * float(M[ulong(r) * cols + col]);
    }
    partial[ulong(gid.y) * cols + col] = acc;
}

template <typename T>
kernel void vecmat_finish(
    device const float* partial [[buffer(0)]],
    device T* y                 [[buffer(1)]],
    constant uint& cols         [[buffer(2)]],
    constant uint& bands        [[buffer(3)]],
    constant uint& accumulate   [[buffer(4)]],
    uint col [[thread_position_in_grid]])
{
    if (col >= cols) {
        return;
    }
    float acc = 0.0f;
    for (uint b = 0; b < bands; b++) {
        acc += partial[ulong(b) * cols + col];
    }
    y[col] = T(accumulate ? float(y[col]) + acc : acc);
}

#define INSTANTIATE_GEMV(T, S)                                                 \
template [[host_name("matvec_rows_" #S)]] kernel void matvec_rows<T>(          \
    device const T*, device const T*, device T*, constant uint&,               \
    constant uint&, constant uint&, uint, uint, uint, uint);                   \
template [[host_name("vecmat_bands_" #S)]] kernel void vecmat_bands<T>(        \
    device const T*, device const T*, device float*, constant uint&,           \
    constant uint&, constant uint&, uint2);                                    \
template [[host_name("vecmat_finish_" #S)]] kernel void vecmat_finish<T>(      \
    device const float*, device T*, constant uint&, constant uint&,            \
    constant uint&, uint);
FOR_EACH_ELEMENT(INSTANTIATE_GEMV)

// `out[r][c] = in[offset + r·row + c·col]`: a strided view, a broadcast or a
// transpose of `in`, copied into order. A copy, so exact in every type.
template <typename T>
kernel void gather_place(
    device const T* input   [[buffer(0)]],
    device T* output        [[buffer(1)]],
    constant uint4& place   [[buffer(2)]],   // offset, row step, column step, cols
    uint i [[thread_position_in_grid]])
{
    uint row = i / place.w;
    uint col = i - row * place.w;
    output[i] = input[place.x + row * place.y + col * place.z];
}

#define INSTANTIATE_GATHER(T, S)                                               \
template [[host_name("gather_place_" #S)]] kernel void gather_place<T>(        \
    device const T*, device T*, constant uint4&, uint);
FOR_EACH_ELEMENT(INSTANTIATE_GATHER)

// `+ - * /` in the element type. There is no remainder kernel; the host keeps
// `BinaryOp::Rem` for itself.
template <typename T>
kernel void elementwise(
    device const T* A [[buffer(0)]],
    device const T* B [[buffer(1)]],
    device T* C       [[buffer(2)]],
    constant BinaryOp& op [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    C[i] = binary_values(op, A[i], B[i]);
}

#define INSTANTIATE_ELEMENTWISE(T, S)                                          \
template [[host_name("elementwise_" #S)]] kernel void elementwise<T>(          \
    device const T*, device const T*, device T*, constant BinaryOp&, uint);
FOR_EACH_ELEMENT(INSTANTIATE_ELEMENTWISE)

// Elementwise `a^b`. Kept out of `elementwise` above rather than added to
// `BinaryOp`, because that enum's contract is that every variant is an operator
// defined for every `Coefficient` — and a power is not: raising an integer to an
// integer leaves the integers. The host side has the same split for the same
// reason.
template <typename T>
kernel void power(
    device const T* A [[buffer(0)]],
    device const T* B [[buffer(1)]],
    device T* C       [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    C[i] = T(pow(A[i], B[i]));
}

#define INSTANTIATE_POWER(T, S)                                                \
template [[host_name("power_" #S)]] kernel void power<T>(                      \
    device const T*, device const T*, device T*, uint);
FOR_EACH_ELEMENT(INSTANTIATE_POWER)

// The same with one operand held fixed: `scalar_left` selects `scalar^A[i]`
// over `A[i]^scalar`.
//
// Scalars arrive as `float` for every element type: the host only ever passes a
// value of the element type, which `float` holds exactly, so `T(scalar)` is that
// value again.
template <typename T>
kernel void power_scalar(
    device const T* A          [[buffer(0)]],
    device T* C                [[buffer(1)]],
    constant float& scalar     [[buffer(2)]],
    constant uint& scalar_left [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    T s = T(scalar);
    T a = scalar_left ? s : A[i];
    T b = scalar_left ? A[i] : s;
    C[i] = T(pow(a, b));
}

#define INSTANTIATE_POWER_SCALAR(T, S)                                         \
template [[host_name("power_scalar_" #S)]] kernel void power_scalar<T>(        \
    device const T*, device T*, constant float&, constant uint&, uint);
FOR_EACH_ELEMENT(INSTANTIATE_POWER_SCALAR)

template <typename T>
kernel void broadcast(
    device const T* A      [[buffer(0)]],
    device T* C            [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    constant BinaryOp& op  [[buffer(3)]],
    constant uint& scalar_left [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    T s = T(scalar);
    C[i] = scalar_left ? binary_values(op, s, A[i]) : binary_values(op, A[i], s);
}

#define INSTANTIATE_BROADCAST(T, S)                                            \
template [[host_name("broadcast_" #S)]] kernel void broadcast<T>(              \
    device const T*, device T*, constant float&, constant BinaryOp&,           \
    constant uint&, uint);
FOR_EACH_ELEMENT(INSTANTIATE_BROADCAST)

template <typename T>
kernel void compare(
    device const T* A      [[buffer(0)]],
    device const T* B      [[buffer(1)]],
    device T* C            [[buffer(2)]],
    constant CompareOp& op [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    C[i] = compare_values(op, A[i], B[i]);
}

#define INSTANTIATE_COMPARE(T, S)                                              \
template [[host_name("compare_" #S)]] kernel void compare<T>(                  \
    device const T*, device const T*, device T*, constant CompareOp&, uint);
FOR_EACH_ELEMENT(INSTANTIATE_COMPARE)

template <typename T>
kernel void compare_scalar(
    device const T* A          [[buffer(0)]],
    device T* C                [[buffer(1)]],
    constant float& scalar     [[buffer(2)]],
    constant CompareOp& op     [[buffer(3)]],
    constant uint& scalar_left [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    T s = T(scalar);
    C[i] = scalar_left ? compare_values(op, s, A[i]) : compare_values(op, A[i], s);
}

#define INSTANTIATE_COMPARE_SCALAR(T, S)                                       \
template [[host_name("compare_scalar_" #S)]] kernel void compare_scalar<T>(    \
    device const T*, device T*, constant float&, constant CompareOp&,          \
    constant uint&, uint);
FOR_EACH_ELEMENT(INSTANTIATE_COMPARE_SCALAR)

// `fmin`/`fmax` rather than the `clamp` builtin, whose behaviour on a NaN input
// is unspecified: this pair is `x.max(low).min(high)`, the CPU definition.
template <typename T>
kernel void clamp_values(
    device const T* A      [[buffer(0)]],
    device T* C            [[buffer(1)]],
    constant float& low    [[buffer(2)]],
    constant float& high   [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    C[i] = element_min(element_max(A[i], T(low)), T(high));
}

#define INSTANTIATE_CLAMP(T, S)                                                \
template [[host_name("clamp_values_" #S)]] kernel void clamp_values<T>(        \
    device const T*, device T*, constant float&, constant float&, uint);
FOR_EACH_ELEMENT(INSTANTIATE_CLAMP)

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
//
// The fold is always in `float`. The first round reads the tensor's own element
// type; every later round reads the `float` partials, through the `_f32`
// instance.
template <typename T>
kernel void reduce_partial(
    device const T* input      [[buffer(0)]],
    device float* partials     [[buffer(1)]],
    constant uint& count       [[buffer(2)]],
    constant ReduceOp& op      [[buffer(3)]],
    uint gid   [[thread_position_in_grid]],
    uint tid   [[thread_position_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]],
    uint width [[threads_per_threadgroup]])
{
    threadgroup float scratch[REDUCE_GROUP];
    scratch[tid] = gid < count ? float(input[gid]) : reduce_identity(op);
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

#define INSTANTIATE_REDUCE(T, S)                                               \
template [[host_name("reduce_partial_" #S)]] kernel void reduce_partial<T>(    \
    device const T*, device float*, constant uint&, constant ReduceOp&,        \
    uint, uint, uint, uint);
FOR_EACH_ELEMENT(INSTANTIATE_REDUCE)

// Conversions into and out of the `float` accumulator, for the operations that
// need a whole `float` buffer: the prefix sum scans in `float` and rounds each
// running total once on the way back.
template <typename T>
kernel void widen(
    device const T* input [[buffer(0)]],
    device float* output  [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    output[i] = float(input[i]);
}

template <typename T>
kernel void narrow(
    device const float* input [[buffer(0)]],
    device T* output          [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    output[i] = T(input[i]);
}

#define INSTANTIATE_CONVERT(T, S)                                              \
template [[host_name("widen_" #S)]] kernel void widen<T>(                      \
    device const T*, device float*, uint);                                     \
template [[host_name("narrow_" #S)]] kernel void narrow<T>(                    \
    device const float*, device T*, uint);
FOR_EACH_ELEMENT(INSTANTIATE_CONVERT)

// One sweep of an inclusive Hillis–Steele scan, from `input` into `output`. The
// host runs it for offsets 1, 2, 4, … and swaps the buffers between sweeps: the
// pass cannot be done in place, since a thread reading `i - offset` would race
// the thread writing it.
//
// `log n` sweeps of `n` adds is more arithmetic than the serial `n`, which is
// the trade a scan makes to have any parallelism at all.
//
// `float` only: a 16-bit tensor is widened first and narrowed after, so the
// running totals carry `float` precision.
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

// The same key for the 16-bit types, whose sign bit is bit 15.
inline ushort sort_key_16(ushort bits) {
    return (bits & 0x8000u) ? ushort(~bits) : ushort(bits | 0x8000u);
}
inline ushort sort_key(half value) { return sort_key_16(as_type<ushort>(value)); }
inline ushort sort_key(bfloat value) { return sort_key_16(as_type<ushort>(value)); }

// Copy into a power-of-two buffer, filling the tail with a value that sorts to
// the end so the padding trims cleanly afterwards.
template <typename T>
kernel void sort_prepare(
    device const T* input     [[buffer(0)]],
    device T* output          [[buffer(1)]],
    constant uint& count      [[buffer(2)]],
    constant T& padding       [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    output[i] = i < count ? input[i] : padding;
}

#define INSTANTIATE_SORT_PREPARE(T, S)                                         \
template [[host_name("sort_prepare_" #S)]] kernel void sort_prepare<T>(        \
    device const T*, device T*, constant uint&, constant T&, uint);
FOR_EACH_ELEMENT(INSTANTIATE_SORT_PREPARE)

// One compare-exchange stage of a bitonic sort. `block` is the width of the
// bitonic sequence being merged and `stride` the distance between partners;
// within a block the direction alternates, which is what builds the next
// sequence up. Only the lower index of each pair does the work, so the stage's
// writes are disjoint and need no synchronization beyond the dispatch boundary.
template <typename T>
kernel void bitonic_stage(
    device T* values       [[buffer(0)]],
    constant uint& block   [[buffer(1)]],
    constant uint& stride  [[buffer(2)]],
    constant uint& ascending [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    uint partner = i ^ stride;
    if (partner <= i) return;

    bool up = ((i & block) == 0) == (ascending != 0);
    T a = values[i];
    T b = values[partner];
    if ((sort_key(a) > sort_key(b)) == up) {
        values[i] = b;
        values[partner] = a;
    }
}

#define INSTANTIATE_BITONIC(T, S)                                              \
template [[host_name("bitonic_stage_" #S)]] kernel void bitonic_stage<T>(      \
    device T*, constant uint&, constant uint&, constant uint&, uint);
FOR_EACH_ELEMENT(INSTANTIATE_BITONIC)

// Valid cross-correlation: every output element is the window of `input` under
// `weights`, summed. `flip` reverses the window, which turns cross-correlation
// into convolution proper — and is what the input-side gradient needs.
template <typename T>
kernel void correlate(
    device const T* input       [[buffer(0)]],
    device const T* weights     [[buffer(1)]],
    device T* output            [[buffer(2)]],
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
            acc += float(input[(gid.y + wr) * cols + (gid.x + wc)])
                 * float(weights[tap_row * window_cols + tap_col]);
        }
    }
    output[gid.y * out_cols + gid.x] = T(acc);
}

#define INSTANTIATE_CORRELATE(T, S)                                            \
template [[host_name("correlate_" #S)]] kernel void correlate<T>(              \
    device const T*, device const T*, device T*, constant uint&,               \
    constant uint&, constant uint&, constant uint&, constant uint&, uint2);
FOR_EACH_ELEMENT(INSTANTIATE_CORRELATE)

// Reverse both axes. The kernel-side gradient of a convolution is the flip of
// the correlation's, so this is what lets both conventions differentiate.
template <typename T>
kernel void flip_both(
    device const T* input     [[buffer(0)]],
    device T* output          [[buffer(1)]],
    constant uint& rows       [[buffer(2)]],
    constant uint& cols       [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.y >= rows || gid.x >= cols) return;
    output[gid.y * cols + gid.x] = input[(rows - 1 - gid.y) * cols + (cols - 1 - gid.x)];
}

#define INSTANTIATE_FLIP(T, S)                                                 \
template [[host_name("flip_both_" #S)]] kernel void flip_both<T>(              \
    device const T*, device T*, constant uint&, constant uint&, uint2);
FOR_EACH_ELEMENT(INSTANTIATE_FLIP)

// Surround a matrix with zeros. The input-side gradient of a valid correlation
// is a full one, and padding is how a full correlation is spelled.
template <typename T>
kernel void pad_zeros(
    device const T* input     [[buffer(0)]],
    device T* output          [[buffer(1)]],
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
        inside ? input[(gid.y - pad_rows) * cols + (gid.x - pad_cols)] : T(0.0f);
}

#define INSTANTIATE_PAD(T, S)                                                  \
template [[host_name("pad_zeros_" #S)]] kernel void pad_zeros<T>(              \
    device const T*, device T*, constant uint&, constant uint&,                \
    constant uint&, constant uint&, uint2);
FOR_EACH_ELEMENT(INSTANTIATE_PAD)

// Copy one input vector into a row or column of a row-major output matrix.
// `output_stride == 1` writes a contiguous row; otherwise it scatters a column.
template <typename T>
kernel void stack_vector(
    device const T* input     [[buffer(0)]],
    device T* output          [[buffer(1)]],
    constant uint& count      [[buffer(2)]],
    constant uint& offset     [[buffer(3)]],
    constant uint& output_stride [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i < count) {
        output[offset + i * output_stride] = input[i];
    }
}

#define INSTANTIATE_STACK(T, S)                                                \
template [[host_name("stack_vector_" #S)]] kernel void stack_vector<T>(        \
    device const T*, device T*, constant uint&, constant uint&,                \
    constant uint&, uint);
FOR_EACH_ELEMENT(INSTANTIATE_STACK)

// Concatenate two row-major matrices with the same row count. Each thread
// writes one output element; both input reads and output writes are coalesced.
template <typename T>
kernel void concat_horizontal(
    device const T* left      [[buffer(0)]],
    device const T* right     [[buffer(1)]],
    device T* output          [[buffer(2)]],
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

#define INSTANTIATE_CONCAT(T, S)                                               \
template [[host_name("concat_horizontal_" #S)]] kernel void concat_horizontal<T>( \
    device const T*, device const T*, device T*, constant uint&,               \
    constant uint&, constant uint&, uint2);
FOR_EACH_ELEMENT(INSTANTIATE_CONCAT)

// Place one matrix into a horizontal block of a wider row-major matrix. The
// encoder dispatches this once per input matrix in the same command buffer.
template <typename T>
kernel void merge_horizontal(
    device const T* input     [[buffer(0)]],
    device T* output          [[buffer(1)]],
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

#define INSTANTIATE_MERGE(T, S)                                                \
template [[host_name("merge_horizontal_" #S)]] kernel void merge_horizontal<T>( \
    device const T*, device T*, constant uint&, constant uint&,                \
    constant uint&, constant uint&, uint2);
FOR_EACH_ELEMENT(INSTANTIATE_MERGE)

// Transpose through a padded threadgroup tile. Adjacent threads read adjacent
// input values and write adjacent output values; the extra column avoids bank
// conflicts when the tile is read in the opposite direction.
template <typename T>
kernel void transpose_tiled(
    device const T* input     [[buffer(0)]],
    device T* output          [[buffer(1)]],
    constant uint& rows       [[buffer(2)]],
    constant uint& cols       [[buffer(3)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]])
{
    threadgroup T tile[TILE][TILE + 1];

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

#define INSTANTIATE_TRANSPOSE(T, S)                                            \
template [[host_name("transpose_tiled_" #S)]] kernel void transpose_tiled<T>(  \
    device const T*, device T*, constant uint&, constant uint&, uint2, uint2);
FOR_EACH_ELEMENT(INSTANTIATE_TRANSPOSE)

template <typename T>
inline T analytic_derivative(AnalyticOp op, T x) {
    const T one = T(1.0f);
    switch (op) {
        case AnalyticOp::Sin:    return T(cos(x));
        case AnalyticOp::Cos:    return -T(sin(x));
        case AnalyticOp::Tan:    { T c = T(cos(x)); return one / (c * c); }
        case AnalyticOp::Sec:    { T c = T(cos(x)); return T(sin(x)) / (c * c); }
        case AnalyticOp::Csc:    { T s = T(sin(x)); return -T(cos(x)) / (s * s); }
        case AnalyticOp::Arcsin: return one / T(sqrt(one - x * x));
        case AnalyticOp::Arccos: return -(one / T(sqrt(one - x * x)));
        case AnalyticOp::Arctan: return one / (one + x * x);
        case AnalyticOp::Exp:    return T(exp(x));
        case AnalyticOp::Ln:     return one / x;
        case AnalyticOp::Sinh:   return T(precise::cosh(x));
        case AnalyticOp::Cosh:   return T(precise::sinh(x));
        case AnalyticOp::Tanh:   { T t = T(precise::tanh(x)); return one - t * t; }
        case AnalyticOp::Sqrt:   return one / ((one + one) * T(sqrt(x)));
        default: return T(NAN);
    }
}

template <typename T>
kernel void unary(
    device const T* A [[buffer(0)]],
    device T* C       [[buffer(1)]],
    constant AnalyticOp& op [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    C[i] = analytic_value(op, A[i]);
}

#define INSTANTIATE_UNARY(T, S)                                                \
template [[host_name("unary_" #S)]] kernel void unary<T>(                      \
    device const T*, device T*, constant AnalyticOp&, uint);
FOR_EACH_ELEMENT(INSTANTIATE_UNARY)

// Forward-mode AD: `f(v + d·ε) = f(v) + f'(v)·d·ε`. Both parts come out of one
// dispatch, which also reads `v` only once.
template <typename T>
kernel void unary_dual(
    device const T* value        [[buffer(0)]],
    device const T* tangent      [[buffer(1)]],
    device T* out_value          [[buffer(2)]],
    device T* out_tangent        [[buffer(3)]],
    constant AnalyticOp& op      [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    T v = value[i];
    out_value[i] = analytic_value(op, v);
    out_tangent[i] = analytic_derivative(op, v) * tangent[i];
}

#define INSTANTIATE_UNARY_DUAL(T, S)                                           \
template [[host_name("unary_dual_" #S)]] kernel void unary_dual<T>(            \
    device const T*, device const T*, device T*, device T*,                    \
    constant AnalyticOp&, uint);
FOR_EACH_ELEMENT(INSTANTIATE_UNARY_DUAL)

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
template <typename T>
kernel void deviation_partial(
    device const T* input     [[buffer(0)]],
    device float* partials    [[buffer(1)]],
    constant uint& count      [[buffer(2)]],
    constant float& mean      [[buffer(3)]],
    uint gid   [[thread_position_in_grid]],
    uint tid   [[thread_position_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]],
    uint width [[threads_per_threadgroup]])
{
    threadgroup float scratch[REDUCE_GROUP];
    // The mean is the unrounded `float` one, so the deviations are measured
    // from the true centre rather than its 16-bit neighbour.
    float deviation = gid < count ? float(input[gid]) - mean : 0.0f;
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

#define INSTANTIATE_DEVIATION(T, S)                                            \
template [[host_name("deviation_partial_" #S)]] kernel void deviation_partial<T>( \
    device const T*, device float*, constant uint&, constant float&,           \
    uint, uint, uint, uint);
FOR_EACH_ELEMENT(INSTANTIATE_DEVIATION)

// One thread per row or per column, each taking both passes over its own slice.
//
// A thread per output rather than a threadgroup per output: the slices are
// usually short next to the number of them, so the parallelism is across
// slices. Along `Rows` the reads are contiguous per thread; along `Columns`
// they are strided per thread but *adjacent threads read adjacent elements*,
// which is the coalesced pattern — the two axes trade which of the two
// localities they get, and neither is the pathological case.
//
// Both passes accumulate in `float`; the mean and the deviation sum round to
// the element type once, when they are written.
template <typename T>
kernel void axis_moments(
    device const T* input      [[buffer(0)]],
    device T* means            [[buffer(1)]],
    device T* deviations       [[buffer(2)]],
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
        total += float(input[base + k * stride]);
    }
    float mean = total / float(count);

    float deviation = 0.0f;
    for (uint k = 0; k < count; ++k) {
        float d = float(input[base + k * stride]) - mean;
        deviation += d * d;
    }

    means[i] = T(mean);
    deviations[i] = T(deviation);
}

#define INSTANTIATE_AXIS_MOMENTS(T, S)                                         \
template [[host_name("axis_moments_" #S)]] kernel void axis_moments<T>(        \
    device const T*, device T*, device T*, constant uint&, constant uint&,     \
    constant AxisOp&, uint);
FOR_EACH_ELEMENT(INSTANTIATE_AXIS_MOMENTS)

// The same moments of each row, by one SIMD group per row: its lanes read the
// row four elements at a time, a stride apart, so that together they read
// whole cache lines, and `simd_sum` combines them. The second pass reads the
// row again from cache. One thread per row, above, would leave a 1024-row
// matrix with 1024 threads, each reading lines no neighbour shares.
template <typename T>
kernel void row_moments(
    device const T* input      [[buffer(0)]],
    device T* means            [[buffer(1)]],
    device T* deviations       [[buffer(2)]],
    constant uint& rows        [[buffer(3)]],
    constant uint& cols        [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simds [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint row = group * simds + simd;
    if (row >= rows) {
        return;
    }
    device const T* line = input + ulong(row) * cols;
    // Whole groups of four when every row starts on one.
    uint wide = cols % 4 == 0 ? cols / 4 : 0;
    device const vec<T, 4>* quads = (device const vec<T, 4>*)line;

    float total = 0.0f;
    for (uint q = lane; q < wide; q += 32) {
        float4 v = float4(quads[q]);
        total += (v.x + v.y) + (v.z + v.w);
    }
    for (uint c = wide * 4 + lane; c < cols; c += 32) {
        total += float(line[c]);
    }
    float mean = simd_sum(total) / float(cols);

    float deviation = 0.0f;
    for (uint q = lane; q < wide; q += 32) {
        float4 d = float4(quads[q]) - mean;
        deviation += dot(d, d);
    }
    for (uint c = wide * 4 + lane; c < cols; c += 32) {
        float d = float(line[c]) - mean;
        deviation += d * d;
    }
    deviation = simd_sum(deviation);
    if (lane == 0) {
        means[row] = T(mean);
        deviations[row] = T(deviation);
    }
}

#define INSTANTIATE_ROW_MOMENTS(T, S)                                          \
template [[host_name("row_moments_" #S)]] kernel void row_moments<T>(          \
    device const T*, device T*, device T*, constant uint&, constant uint&,     \
    uint, uint, uint, uint);
FOR_EACH_ELEMENT(INSTANTIATE_ROW_MOMENTS)

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
//
// The special functions are `float` polynomials; a 16-bit element widens
// exactly on the way in and the result rounds once on the way out, which is
// also what the host does from `f64`.
template <typename T>
kernel void distribution(
    device const T* input         [[buffer(0)]],
    device T* output              [[buffer(1)]],
    constant FamilyOp& family     [[buffer(2)]],
    constant StatisticOp& stat    [[buffer(3)]],
    constant float& first         [[buffer(4)]],
    constant float& second        [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    output[i] = T(distribution_value(family, stat, float(input[i]), first, second));
}

#define INSTANTIATE_DISTRIBUTION(T, S)                                         \
template [[host_name("distribution_" #S)]] kernel void distribution<T>(        \
    device const T*, device T*, constant FamilyOp&, constant StatisticOp&,     \
    constant float&, constant float&, uint);
FOR_EACH_ELEMENT(INSTANTIATE_DISTRIBUTION)

// One distribution per row or per column: the parameter pair is looked up by
// the element's position along the axis, which is what turns a per-row fit into
// a single dispatch rather than one per row.
template <typename T>
kernel void axis_distribution(
    device const T* input         [[buffer(0)]],
    device T* output              [[buffer(1)]],
    device const T* first         [[buffer(2)]],
    device const T* second        [[buffer(3)]],
    constant uint& cols           [[buffer(4)]],
    constant AxisOp& axis         [[buffer(5)]],
    constant FamilyOp& family     [[buffer(6)]],
    constant StatisticOp& stat    [[buffer(7)]],
    uint i [[thread_position_in_grid]])
{
    uint along = axis == AxisOp::Rows ? i / cols : i % cols;
    output[i] = T(distribution_value(
        family, stat, float(input[i]), float(first[along]), float(second[along])));
}

#define INSTANTIATE_AXIS_DISTRIBUTION(T, S)                                    \
template [[host_name("axis_distribution_" #S)]] kernel void axis_distribution<T>( \
    device const T*, device T*, device const T*, device const T*,              \
    constant uint&, constant AxisOp&, constant FamilyOp&, constant StatisticOp&, uint);
FOR_EACH_ELEMENT(INSTANTIATE_AXIS_DISTRIBUTION)

// ---- fused elementwise programs ----------------------------------------------
//
// One thread runs a whole register program for one element; see
// `tensors::fused`. The registers hold the program's element type — the `T` of
// `Program<T>` — so a `Program<f16>` computes in `half`, and loads and stores
// convert between that and each operand's storage type. Every thread runs the same instructions in the same order,
// so the `switch` on each opcode never diverges within a SIMD group, and the
// intermediates stay in the `r` array rather than going back to memory.

// Sixteen input and eight output slots, plus the program and its shape: 26 of
// Metal's 31 buffer arguments. Unused slots are bound to a used buffer, which is
// never touched because no instruction names them. A tensor updated in place is
// bound to both an input and an output slot; the program reads it before it
// stores it, and each thread touches only its own element.
template <typename T>
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
    constant FusedPlace* places [[buffer(26)]],
    uint i [[thread_position_in_grid]])
{
    FusedBuffers buffers = {
        { in0, in1, in2, in3, in4, in5, in6, in7,
          in8, in9, in10, in11, in12, in13, in14, in15 },
        { out0, out1, out2, out3, out4, out5, out6, out7 },
        places
    };
    fused_run<T>(code, shape.count, buffers, i, shape.rows, shape.cols, false, 0.0f);
}

#define INSTANTIATE_FUSED(T, S)                                                \
template [[host_name("fused_elementwise_" #S)]] kernel void fused_elementwise<T>( \
    constant FusedInstr*, constant FusedShape&,                                \
    device const uchar*, device const uchar*, device const uchar*,             \
    device const uchar*, device const uchar*, device const uchar*,             \
    device const uchar*, device const uchar*, device const uchar*,             \
    device const uchar*, device const uchar*, device const uchar*,             \
    device const uchar*, device const uchar*, device const uchar*,             \
    device const uchar*,                                                       \
    device uchar*, device uchar*, device uchar*, device uchar*,                \
    device uchar*, device uchar*, device uchar*, device uchar*,                \
    constant FusedPlace*, uint);
FOR_EACH_ELEMENT(INSTANTIATE_FUSED)

// ---- matrix products with an epilogue -----------------------------------------
//
// `A·B` whose every element goes straight into a fused program as its input 0,
// before anything is stored: bias, activation, scaling — whatever the program
// says — at the cost of the product alone. The shape is the product's
// (`rows = M`, `cols = N`), so the program's other inputs are addressed, and
// its remaps resolved, exactly as `fused_elementwise` would for the stored
// product. Slot 0's buffer is never read.
template <typename T>
kernel void matmul_epilogue(
    constant FusedInstr* code   [[buffer(0)]],
    constant FusedShape& shape  [[buffer(1)]],
    device const T* A  [[buffer(2)]],
    device const T* B  [[buffer(3)]],
    constant uint& K   [[buffer(4)]],
    device const uchar* in1  [[buffer(5)]],
    device const uchar* in2  [[buffer(6)]],
    device const uchar* in3  [[buffer(7)]],
    device const uchar* in4  [[buffer(8)]],
    device const uchar* in5  [[buffer(9)]],
    device const uchar* in6  [[buffer(10)]],
    device const uchar* in7  [[buffer(11)]],
    device const uchar* in8  [[buffer(12)]],
    device const uchar* in9  [[buffer(13)]],
    device const uchar* in10 [[buffer(14)]],
    device const uchar* in11 [[buffer(15)]],
    device const uchar* in12 [[buffer(16)]],
    device const uchar* in13 [[buffer(17)]],
    device const uchar* in14 [[buffer(18)]],
    device const uchar* in15 [[buffer(19)]],
    device uchar* out0 [[buffer(20)]],
    device uchar* out1 [[buffer(21)]],
    device uchar* out2 [[buffer(22)]],
    device uchar* out3 [[buffer(23)]],
    device uchar* out4 [[buffer(24)]],
    device uchar* out5 [[buffer(25)]],
    device uchar* out6 [[buffer(26)]],
    device uchar* out7 [[buffer(27)]],
    constant FusedPlace* places [[buffer(28)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]])
{
    threadgroup float Asub[TILE][TILE];
    threadgroup float Bsub[TILE][TILE];
    uint M = shape.rows;
    uint N = shape.cols;
    float acc = tiled_product(A, B, M, K, N, tid, gid, Asub, Bsub);

    uint row = gid.y;
    uint col = gid.x;
    if (row < M && col < N) {
        FusedBuffers buffers = {
            { (device const uchar*)A, in1, in2, in3, in4, in5, in6, in7,
              in8, in9, in10, in11, in12, in13, in14, in15 },
            { out0, out1, out2, out3, out4, out5, out6, out7 },
            places
        };
        fused_run<T>(code, shape.count, buffers, row * N + col, M, N, true, acc);
    }
}

#define INSTANTIATE_MATMUL_EPILOGUE(T, S)                                      \
template [[host_name("matmul_epilogue_" #S)]] kernel void matmul_epilogue<T>(  \
    constant FusedInstr*, constant FusedShape&, device const T*,               \
    device const T*, constant uint&,                                           \
    device const uchar*, device const uchar*, device const uchar*,             \
    device const uchar*, device const uchar*, device const uchar*,             \
    device const uchar*, device const uchar*, device const uchar*,             \
    device const uchar*, device const uchar*, device const uchar*,             \
    device const uchar*, device const uchar*, device const uchar*,             \
    device uchar*, device uchar*, device uchar*, device uchar*,                \
    device uchar*, device uchar*, device uchar*, device uchar*,                \
    constant FusedPlace*, uint2, uint2);
FOR_EACH_ELEMENT(INSTANTIATE_MATMUL_EPILOGUE)
