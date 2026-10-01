// Shared by `kernel.metal` and `tensorops.metal`: the operation enums, the
// elementwise value functions, and the fused-program interpreter, so a matrix
// product's epilogue runs exactly the code `fused_elementwise` does.
#pragma once

#include <metal_stdlib>
using namespace metal;

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

template <typename T>
inline T binary_values(BinaryOp op, T a, T b) {
    switch (op) {
        case BinaryOp::Add: return a + b;
        case BinaryOp::Sub: return a - b;
        case BinaryOp::Mul: return a * b;
        default: return a / b;
    }
}

// `fmin`/`fmax` in the element type. MSL returns `float` for `bfloat` operands,
// which the conversion rounds straight back — exactly, since the result is one
// of the two operands.
template <typename T> inline T element_min(T a, T b) { return T(fmin(a, b)); }
template <typename T> inline T element_max(T a, T b) { return T(fmax(a, b)); }

template <typename T>
inline T compare_values(CompareOp op, T a, T b) {
    switch (op) {
        case CompareOp::Min: return element_min(a, b);
        case CompareOp::Max: return element_max(a, b);
        case CompareOp::MaxShare: return a > b ? T(1.0f) : (a < b ? T(0.0f) : T(0.5f));
        case CompareOp::Less: return a < b ? T(1.0f) : T(0.0f);
        case CompareOp::LessEqual: return a <= b ? T(1.0f) : T(0.0f);
        case CompareOp::Greater: return a > b ? T(1.0f) : T(0.0f);
        default: return a >= b ? T(1.0f) : T(0.0f);
    }
}

// The shader builds with fast math, whose hyperbolics are formed from `exp`:
// `tanh` becomes `inf / inf = NaN` once `exp(2x)` overflows (|x| > ~44), and
// `sinh`/`cosh` reach `inf` near |x| = 89 where the true value is still finite.
// The `precise::` forms saturate and overflow where the host's do, and the rest
// of the shader keeps fast math.
//
// These variants must agree with `tensors::kernels::Analytic`, and each
// derivative must match the corresponding `Dual` implementation.
//
// Each function is evaluated in the element type and every intermediate rounds
// to it, in the order the host's `Analytic::value` and `derivative` use, so a
// `half` derivative is built from `half` operations exactly as the CPU's `f16`
// one is. `T(...)` around a library call is the identity for `float` and
// `half`, and the single rounding of a `float` result for `bfloat`.
template <typename T>
inline T analytic_value(AnalyticOp op, T x) {
    const T one = T(1.0f);
    switch (op) {
        case AnalyticOp::Sin:    return T(sin(x));
        case AnalyticOp::Cos:    return T(cos(x));
        case AnalyticOp::Tan:    return T(tan(x));
        case AnalyticOp::Sec:    return one / T(cos(x));
        case AnalyticOp::Csc:    return one / T(sin(x));
        case AnalyticOp::Arcsin: return T(asin(x));
        case AnalyticOp::Arccos: return T(acos(x));
        case AnalyticOp::Arctan: return T(atan(x));
        case AnalyticOp::Exp:    return T(exp(x));
        case AnalyticOp::Ln:     return T(log(x));
        case AnalyticOp::Sinh:   return T(precise::sinh(x));
        case AnalyticOp::Cosh:   return T(precise::cosh(x));
        case AnalyticOp::Tanh:   return T(precise::tanh(x));
        case AnalyticOp::Sqrt:   return T(sqrt(x));
        default: return T(NAN);
    }
}

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


// The registers a program runs over, and its buffers.
struct FusedBuffers {
    device const uchar* inputs[16];
    device uchar* outputs[8];
};

// Run a program for element `i` of a `rows × cols` space. With `has_product`,
// a load from input slot 0 is `product` — a matrix product's `float`
// accumulator, rounded to `T` exactly as the product's own store would round
// it — so the product's epilogue never writes it to memory.
template <typename T>
inline void fused_run(
    constant FusedInstr* code,
    uint count,
    thread const FusedBuffers& buffers,
    uint i,
    uint rows,
    uint cols,
    bool has_product,
    float product)
{
    T r[FUSED_REGISTERS];
    for (uint pc = 0; pc < count; pc++) {
        FusedInstr instr = code[pc];
        switch (instr.kind) {
            case 0:
                if (has_product && instr.a == 0) {
                    r[instr.dst] = T(product);
                } else {
                    // Every storage type widens to `float` exactly, so this is
                    // one rounding, to `T`.
                    r[instr.dst] = T(fused_load(
                        buffers.inputs[instr.a], instr.b,
                        fused_remap(instr.op, i, rows, cols)));
                }
                break;
            case 1:
                // A `Program<T>` constant is a `T`, which `float` holds exactly.
                r[instr.dst] = T(instr.value);
                break;
            case 2:
                r[instr.dst] = binary_values(BinaryOp(instr.op), r[instr.a], r[instr.b]);
                break;
            case 3:
                r[instr.dst] = analytic_value(AnalyticOp(instr.op), r[instr.a]);
                break;
            case 4:
                r[instr.dst] = compare_values(CompareOp(instr.op), r[instr.a], r[instr.b]);
                break;
            default:
                fused_store(buffers.outputs[instr.aux], instr.b, i, float(r[instr.a]));
                break;
        }
    }
}
