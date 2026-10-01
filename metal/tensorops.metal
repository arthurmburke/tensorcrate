#include "common.h"
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace mpp::tensor_ops;

template <typename Input, typename Output>
METAL_FUNC void tensorcrate_matmul(
    device Input* A,
    device Input* B,
    device Output* C,
    uint M,
    uint K,
    uint N,
    uint2 group)
{
    // Four SIMD groups cooperatively produce a 2x2 arrangement of 32x32 tiles,
    // Apple's recommended starting point for 16-bit M5 matrix products. The
    // reduction extent stays dynamic so one pipeline covers every shape.
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
    auto tensor_c = tensor(
        C,
        dextents<int32_t, 2>{static_cast<int32_t>(N), static_cast<int32_t>(M)},
        array<int32_t, 2>{1, static_cast<int32_t>(N)}
    );

    matmul2d<descriptor, execution_simdgroups<4>> operation;
    auto tile_a = tensor_a.slice(0, group.y * 64);
    auto tile_b = tensor_b.slice(group.x * 64, 0);
    auto tile_c = tensor_c.slice(group.x * 64, group.y * 64);
    auto accumulator = operation.get_destination_cooperative_tensor<
        decltype(tile_a),
        decltype(tile_b),
        Output
    >();
    operation.run(tile_a, tile_b, accumulator);
    accumulator.store(tile_c);
}

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

#define TENSORCRATE_MATMUL_KERNEL(NAME, INPUT, OUTPUT)                        \
kernel void NAME(                                                            \
    device INPUT* A [[buffer(0)]],                                            \
    device INPUT* B [[buffer(1)]],                                            \
    device OUTPUT* C [[buffer(2)]],                                           \
    constant uint& M [[buffer(3)]],                                           \
    constant uint& K [[buffer(4)]],                                           \
    constant uint& N [[buffer(5)]],                                           \
    uint2 group [[threadgroup_position_in_grid]])                             \
{                                                                            \
    tensorcrate_matmul(A, B, C, M, K, N, group);                             \
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

TENSORCRATE_MATMUL_KERNEL(matmul_tensorops_f32, float, float)
TENSORCRATE_MATMUL_NARROW_KERNEL(matmul_tensorops_f16, half, half)
TENSORCRATE_MATMUL_KERNEL(matmul_tensorops_f16_f32, half, float)
TENSORCRATE_MATMUL_NARROW_KERNEL(matmul_tensorops_bf16, bfloat, bfloat)
TENSORCRATE_MATMUL_KERNEL(matmul_tensorops_bf16_f32, bfloat, float)

// The same product with an epilogue: the `float` tile is staged as in the
// narrow kernels, and each thread runs the fused program on its share of it
// instead of a plain store. Buffer layout as `matmul_epilogue` in
// `kernel.metal`.
template <typename T>
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
            fused_run<T>(code, shape.count, buffers, row * N + col, M, N, true, staging[e]);
        }
    }
}

#define TENSORCRATE_MATMUL_EPILOGUE_KERNEL(NAME, T)                           \
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
    uint2 group [[threadgroup_position_in_grid]],                             \
    uint2 lane [[thread_position_in_threadgroup]],                            \
    uint2 threads [[threads_per_threadgroup]])                                \
{                                                                            \
    threadgroup float staging[64 * 64];                                      \
    FusedBuffers buffers = {                                                 \
        { (device const uchar*)A, in1, in2, in3, in4, in5, in6, in7,         \
          in8, in9, in10, in11, in12, in13, in14, in15 },                    \
        { out0, out1, out2, out3, out4, out5, out6, out7 }                   \
    };                                                                       \
    tensorcrate_matmul_epilogue<T>(                                          \
        code, shape, A, B, K, buffers, group, lane.x, threads.x, staging);   \
}

TENSORCRATE_MATMUL_EPILOGUE_KERNEL(matmul_tensorops_epilogue_f32, float)
TENSORCRATE_MATMUL_EPILOGUE_KERNEL(matmul_tensorops_epilogue_f16, half)
TENSORCRATE_MATMUL_EPILOGUE_KERNEL(matmul_tensorops_epilogue_bf16, bfloat)
