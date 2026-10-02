#include "common.h"
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace mpp::tensor_ops;

// A 16-bit product with a `float` accumulator and a 16-bit result: the policy
// every other reduction in the crate follows. TensorOps stores a cooperative
// tensor only to a tensor of its own element type, so the `float` tile goes to
// threadgroup memory first, and each thread then rounds its share of it into
// the output. The guard on `M` and `N` is what trims a partial edge tile.
template <typename Input, typename Output>
METAL_FUNC void tensorcrate_matmul_narrow(
    device Input* A,
    device Input* B,
    device Output* C,
    uint M,
    uint K,
    uint N,
    uint2 group,
    uint thread_index,
    uint threads,
    threadgroup float* staging)
{
    constexpr auto descriptor = matmul2d_descriptor(
        64,
        64,
        static_cast<int>(dynamic_extent),
        false,
        false,
        false
    );
    auto tensor_a = tensor(
        A,
        dextents<int32_t, 2>{static_cast<int32_t>(K), static_cast<int32_t>(M)},
        array<int32_t, 2>{1, static_cast<int32_t>(K)}
    );
    auto tensor_b = tensor(
        B,
        dextents<int32_t, 2>{static_cast<int32_t>(N), static_cast<int32_t>(K)},
        array<int32_t, 2>{1, static_cast<int32_t>(N)}
    );
    auto staged = tensor(
        staging,
        dextents<int32_t, 2>{64, 64},
        array<int32_t, 2>{1, 64}
    );

    matmul2d<descriptor, execution_simdgroups<4>> operation;
    auto tile_a = tensor_a.slice(0, group.y * 64);
    auto tile_b = tensor_b.slice(group.x * 64, 0);
    auto accumulator = operation.get_destination_cooperative_tensor<
        decltype(tile_a),
        decltype(tile_b),
        float
    >();
    operation.run(tile_a, tile_b, accumulator);
    accumulator.store(staged);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = thread_index; e < 64 * 64; e += threads) {
        uint row = group.y * 64 + e / 64;
        uint col = group.x * 64 + e % 64;
        if (row < M && col < N) {
            C[row * N + col] = Output(staging[e]);
        }
    }
}

#define TENSORCRATE_MATMUL_NARROW_KERNEL(NAME, INPUT, OUTPUT)                 \
kernel void NAME(                                                            \
    device INPUT* A [[buffer(0)]],                                            \
    device INPUT* B [[buffer(1)]],                                            \
    device OUTPUT* C [[buffer(2)]],                                           \
    constant uint& M [[buffer(3)]],                                           \
    constant uint& K [[buffer(4)]],                                           \
    constant uint& N [[buffer(5)]],                                           \
    uint2 group [[threadgroup_position_in_grid]],                             \
    uint2 lane [[thread_position_in_threadgroup]],                            \
    uint2 threads [[threads_per_threadgroup]])                                \
{                                                                            \
    threadgroup float staging[64 * 64];                                      \
    tensorcrate_matmul_narrow(                                               \
        A, B, C, M, K, N, group, lane.x, threads.x, staging);                \
}

TENSORCRATE_MATMUL_NARROW_KERNEL(matmul_tensorops_f16, half, half)
TENSORCRATE_MATMUL_NARROW_KERNEL(matmul_tensorops_bf16, bfloat, bfloat)

// The same product with an epilogue: the `float` tile is staged as in the
// narrow kernels, and each thread runs the fused program on its share of it
// instead of a plain store. Buffer layout as `matmul_epilogue` in
// `kernel.metal`.
template <typename T, bool Relaxed = false>
METAL_FUNC void tensorcrate_matmul_epilogue(
    constant FusedInstr* code,
    constant FusedShape& shape,
    device T* A,
    device T* B,
    uint K,
    thread const FusedBuffers& buffers,
    uint2 group,
    uint thread_index,
    uint threads,
    threadgroup float* staging)
{
    uint M = shape.rows;
    uint N = shape.cols;
    constexpr auto descriptor = matmul2d_descriptor(
        64,
        64,
        static_cast<int>(dynamic_extent),
        false,
        false,
        Relaxed
    );
    auto tensor_a = tensor(
        A,
        dextents<int32_t, 2>{static_cast<int32_t>(K), static_cast<int32_t>(M)},
        array<int32_t, 2>{1, static_cast<int32_t>(K)}
    );
    auto tensor_b = tensor(
        B,
        dextents<int32_t, 2>{static_cast<int32_t>(N), static_cast<int32_t>(K)},
        array<int32_t, 2>{1, static_cast<int32_t>(N)}
    );
    auto staged = tensor(
        staging,
        dextents<int32_t, 2>{64, 64},
        array<int32_t, 2>{1, 64}
    );

    matmul2d<descriptor, execution_simdgroups<4>> operation;
    auto tile_a = tensor_a.slice(0, group.y * 64);
    auto tile_b = tensor_b.slice(group.x * 64, 0);
    auto accumulator = operation.template get_destination_cooperative_tensor<
        decltype(tile_a),
        decltype(tile_b),
        float
    >();
    operation.run(tile_a, tile_b, accumulator);
    accumulator.store(staged);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = thread_index; e < 64 * 64; e += threads) {
        uint row = group.y * 64 + e / 64;
        uint col = group.x * 64 + e % 64;
        if (row < M && col < N) {
            fused_run<T>(code, shape.count, buffers, row * N + col, M, N, true, staging[e]);
        }
    }
}

#define TENSORCRATE_MATMUL_EPILOGUE_KERNEL(NAME, T, RELAXED)                  \
kernel void NAME(                                                            \
    constant FusedInstr* code [[buffer(0)]],                                  \
    constant FusedShape& shape [[buffer(1)]],                                 \
    device T* A [[buffer(2)]],                                                \
    device T* B [[buffer(3)]],                                                \
    constant uint& K [[buffer(4)]],                                           \
    device const uchar* in1  [[buffer(5)]],                                   \
    device const uchar* in2  [[buffer(6)]],                                   \
    device const uchar* in3  [[buffer(7)]],                                   \
    device const uchar* in4  [[buffer(8)]],                                   \
    device const uchar* in5  [[buffer(9)]],                                   \
    device const uchar* in6  [[buffer(10)]],                                  \
    device const uchar* in7  [[buffer(11)]],                                  \
    device const uchar* in8  [[buffer(12)]],                                  \
    device const uchar* in9  [[buffer(13)]],                                  \
    device const uchar* in10 [[buffer(14)]],                                  \
    device const uchar* in11 [[buffer(15)]],                                  \
    device const uchar* in12 [[buffer(16)]],                                  \
    device const uchar* in13 [[buffer(17)]],                                  \
    device const uchar* in14 [[buffer(18)]],                                  \
    device const uchar* in15 [[buffer(19)]],                                  \
    device uchar* out0 [[buffer(20)]],                                        \
    device uchar* out1 [[buffer(21)]],                                        \
    device uchar* out2 [[buffer(22)]],                                        \
    device uchar* out3 [[buffer(23)]],                                        \
    device uchar* out4 [[buffer(24)]],                                        \
    device uchar* out5 [[buffer(25)]],                                        \
    device uchar* out6 [[buffer(26)]],                                        \
    device uchar* out7 [[buffer(27)]],                                        \
    constant FusedPlace* places [[buffer(28)]],                               \
    uint2 group [[threadgroup_position_in_grid]],                             \
    uint2 lane [[thread_position_in_threadgroup]],                            \
    uint2 threads [[threads_per_threadgroup]])                                \
{                                                                            \
    threadgroup float staging[64 * 64];                                      \
    FusedBuffers buffers = {                                                 \
        { (device const uchar*)A, in1, in2, in3, in4, in5, in6, in7,         \
          in8, in9, in10, in11, in12, in13, in14, in15 },                    \
        { out0, out1, out2, out3, out4, out5, out6, out7 },                  \
        places                                                               \
    };                                                                       \
    tensorcrate_matmul_epilogue<T, RELAXED>(                                 \
        code, shape, A, B, K, buffers, group, lane.x, threads.x, staging);   \
}

TENSORCRATE_MATMUL_EPILOGUE_KERNEL(matmul_tensorops_epilogue_f32, float, false)
TENSORCRATE_MATMUL_EPILOGUE_KERNEL(matmul_tensorops_epilogue_f16, half, false)
TENSORCRATE_MATMUL_EPILOGUE_KERNEL(matmul_tensorops_epilogue_bf16, bfloat, false)
TENSORCRATE_MATMUL_EPILOGUE_KERNEL(matmul_tensorops_epilogue_f32_relaxed, float, true)

// ---- the general product ------------------------------------------------------
//
// `C = op(A)·op(B)`, or `C += op(A)·op(B)` when accumulating, where `op`
// transposes an operand whose flag is set — the products a matrix product's
// backward pass needs, read where they lie instead of transposed into a copy
// first. Each threadgroup computes one `TM × TN` tile with `G` SIMD groups;
// smaller tiles give a product with a small output more threadgroups to spread
// across the GPU.
//
// A transposed operand is passed in its stored layout and the descriptor reads
// it transposed: `A` stored `K × M` when `TA`, `B` stored `N × K` when `TB`.
template <typename Input, typename Output, int TM, int TN, int G, bool Relaxed,
          bool TA, bool TB, bool Accumulate>
METAL_FUNC void tensorcrate_gemm(
    device Input* A,
    device Input* B,
    device Output* C,
    uint M,
    uint K,
    uint N,
    uint2 group)
{
    constexpr auto descriptor = matmul2d_descriptor(
        TM,
        TN,
        static_cast<int>(dynamic_extent),
        TA,
        TB,
        Relaxed,
        Accumulate ? matmul2d_descriptor::mode::multiply_accumulate
                   : matmul2d_descriptor::mode::multiply
    );
    const int32_t m = int32_t(M), k = int32_t(K), n = int32_t(N);
    auto tensor_a = tensor(
        A,
        dextents<int32_t, 2>{TA ? m : k, TA ? k : m},
        array<int32_t, 2>{1, TA ? m : k}
    );
    auto tensor_b = tensor(
        B,
        dextents<int32_t, 2>{TB ? k : n, TB ? n : k},
        array<int32_t, 2>{1, TB ? k : n}
    );
    auto tensor_c = tensor(
        C,
        dextents<int32_t, 2>{n, m},
        array<int32_t, 2>{1, n}
    );

    matmul2d<descriptor, execution_simdgroups<G>> operation;
    auto tile_a = TA ? tensor_a.slice(int32_t(group.y) * TM, 0)
                     : tensor_a.slice(0, int32_t(group.y) * TM);
    auto tile_b = TB ? tensor_b.slice(0, int32_t(group.x) * TN)
                     : tensor_b.slice(int32_t(group.x) * TN, 0);
    auto tile_c = tensor_c.slice(int32_t(group.x) * TN, int32_t(group.y) * TM);
    auto accumulator = operation.template get_destination_cooperative_tensor<
        decltype(tile_a),
        decltype(tile_b),
        Output
    >();
    if (Accumulate) {
        accumulator.load(tile_c);
    }
    operation.run(tile_a, tile_b, accumulator);
    accumulator.store(tile_c);
}

#define TENSORCRATE_GEMM(NAME, INPUT, OUTPUT, TM, TN, G, RELAXED, TA, TB, ACC) \
kernel void NAME(                                                            \
    device INPUT* A [[buffer(0)]],                                            \
    device INPUT* B [[buffer(1)]],                                            \
    device OUTPUT* C [[buffer(2)]],                                           \
    constant uint& M [[buffer(3)]],                                           \
    constant uint& K [[buffer(4)]],                                           \
    constant uint& N [[buffer(5)]],                                           \
    uint2 group [[threadgroup_position_in_grid]])                             \
{                                                                            \
    tensorcrate_gemm<INPUT, OUTPUT, TM, TN, G, RELAXED, TA, TB, ACC>(        \
        A, B, C, M, K, N, group);                                            \
}

// Every product the crate runs on the general kernel, named
// `gemm_<input>_<tile>_<operands>`. The tiles are `s` (32 × 32 on four SIMD
// groups), `m` (64 × 32 on two) and `l` (64 × 64 on four); the operands are
// `nn`, `tn` (`A` transposed) or `nt` (`B` transposed), with `_acc` when the
// product adds into `C`. `f32r` is `float` at relaxed precision.
#define TENSORCRATE_GEMM_TILE(PREFIX, INPUT, OUTPUT, RELAXED, TILE, TM, TN, G)              \
TENSORCRATE_GEMM(PREFIX##_##TILE##_nn, INPUT, OUTPUT, TM, TN, G, RELAXED, false, false, false) \
TENSORCRATE_GEMM(PREFIX##_##TILE##_tn, INPUT, OUTPUT, TM, TN, G, RELAXED, true, false, false)  \
TENSORCRATE_GEMM(PREFIX##_##TILE##_nt, INPUT, OUTPUT, TM, TN, G, RELAXED, false, true, false)  \
TENSORCRATE_GEMM(PREFIX##_##TILE##_nn_acc, INPUT, OUTPUT, TM, TN, G, RELAXED, false, false, true) \
TENSORCRATE_GEMM(PREFIX##_##TILE##_tn_acc, INPUT, OUTPUT, TM, TN, G, RELAXED, true, false, true)  \
TENSORCRATE_GEMM(PREFIX##_##TILE##_nt_acc, INPUT, OUTPUT, TM, TN, G, RELAXED, false, true, true)

#define TENSORCRATE_GEMM_TILES(PREFIX, INPUT, OUTPUT, RELAXED)                  \
TENSORCRATE_GEMM_TILE(PREFIX, INPUT, OUTPUT, RELAXED, s, 32, 32, 4)           \
TENSORCRATE_GEMM_TILE(PREFIX, INPUT, OUTPUT, RELAXED, m, 64, 32, 2)           \
TENSORCRATE_GEMM_TILE(PREFIX, INPUT, OUTPUT, RELAXED, l, 64, 64, 4)

TENSORCRATE_GEMM_TILES(gemm_f32, float, float, false)
TENSORCRATE_GEMM_TILES(gemm_f32r, float, float, true)

// The 16-bit inputs with a `float` result behind `matmul_f32`.
#define TENSORCRATE_GEMM_NN(PREFIX, INPUT, OUTPUT)                              \
TENSORCRATE_GEMM(PREFIX##_s_nn, INPUT, OUTPUT, 32, 32, 4, false, false, false, false) \
TENSORCRATE_GEMM(PREFIX##_m_nn, INPUT, OUTPUT, 64, 32, 2, false, false, false, false) \
TENSORCRATE_GEMM(PREFIX##_l_nn, INPUT, OUTPUT, 64, 64, 4, false, false, false, false)

TENSORCRATE_GEMM_NN(gemm_f16_f32, half, float)
TENSORCRATE_GEMM_NN(gemm_bf16_f32, bfloat, float)
