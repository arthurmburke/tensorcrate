// Linear operator kernels. Complex values use the existing interleaved ABI.
template<typename T> inline T operator_mul(T a, T b) { return a * b; }
template<> inline float2 operator_mul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
template<typename T> inline T operator_conj(T a) { return a; }
template<> inline float2 operator_conj(float2 a) { return float2(a.x, -a.y); }
template<typename T> inline bool operator_zero(T a) { return a == T(0); }
template<> inline bool operator_zero(float2 a) { return all(a == float2(0)); }
inline float operator_length(float a) { return abs(a); }
inline float operator_length(float2 a) {
    if (any(isnan(a))) return NAN;
    if (any(isinf(a))) return INFINITY;
    float scale = max(abs(a.x), abs(a.y));
    if (scale == 0.0f || !isfinite(scale)) return scale;
    float2 scaled = a / scale;
    return scale * sqrt(scaled.x * scaled.x + scaled.y * scaled.y);
}
template<typename T> inline T operator_real(float a) { return T(a); }
template<> inline float2 operator_real(float a) { return float2(a, 0.0f); }

// p: kind, rows, cols, local width, input offset, output offset, adjoint, count.
template<typename T>
kernel void linear_operator(
    device const T* values [[buffer(0)]],
    device const uint* indices0 [[buffer(1)]],
    device const uint* indices1 [[buffer(2)]],
    device const uint* indices2 [[buffer(3)]],
    device const T* x [[buffer(4)]],
    device T* y [[buffer(5)]],
    constant uint* p [[buffer(6)]],
    constant T& alpha [[buffer(7)]],
    constant T& beta [[buffer(8)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= p[7]) return;
    uint kind = p[0];
    T sum = T(0);
    if (kind == 0) { // Dense matrix or its adjoint.
        uint inner = p[6] ? p[1] : p[2];
        for (uint j = 0; j < inner; ++j) {
            T a = p[6] ? operator_conj(values[j * p[2] + row]) : values[row * p[2] + j];
            sum += operator_mul(a, x[p[4] + j]);
        }
    } else if (kind == 1 || kind == 2) { // CSR or transposed index view.
        for (uint k = indices0[row]; k < indices0[row + 1]; ++k) {
            T a = kind == 2 ? operator_conj(values[indices2[k]]) : values[k];
            sum += operator_mul(a, x[p[4] + indices1[k]]);
        }
    } else if (kind == 3) { // Grouped element contributions, no scatter.
        uint width = p[3];
        for (uint k = indices0[row]; k < indices0[row + 1]; ++k) {
            uint local = indices1[k], element = local / width, i = local % width;
            uint base = element * width * width;
            for (uint j = 0; j < width; ++j) {
                T a = p[6] ? operator_conj(values[base + j * width + i]) : values[base + i * width + j];
                sum += operator_mul(a, x[p[4] + indices2[element * width + j]]);
            }
        }
    } else if (kind == 4) {
        T a = p[6] ? operator_conj(values[row]) : values[row];
        sum = operator_mul(a, x[p[4] + row]);
    } else if (kind == 6) { // Partial conjugate inner product.
        uint begin = row * 256;
        for (uint j = begin; j < min(begin + 256, p[2]); ++j)
            sum += operator_mul(operator_conj(values[j]), x[j]);
    } else if (kind == 7) { // Materialized dense adjoint.
        uint batch = p[1] * p[2];
        uint local = row % batch;
        sum = operator_conj(values[(row / batch) * batch + (local % p[1]) * p[2] + local / p[1]]);
    } else if (kind == 8) { // Vector AXPBY, fused without an intermediate.
        sum = x[p[4] + row];
    } else if (kind == 9) { // Scaled chunk norms avoid squaring large values.
        uint begin = row * 256;
        float length = 0.0f;
        for (uint j = begin; j < min(begin + 256, p[2]); ++j) {
            float value = operator_length(values[j]);
            float scale = max(length, value);
            if (!isfinite(value)) { length = value; break; }
            if (scale != 0.0f) {
                float a = length / scale, b = value / scale;
                length = scale * sqrt(a * a + b * b);
            }
        }
        sum = operator_real<T>(length);
    }
    uint target = p[5] + row;
    T old = operator_zero(beta) ? T(0) : operator_mul(beta, y[target]);
    y[target] = operator_mul(alpha, sum) + old;
}

#define INSTANTIATE_OPERATOR(T, S) \
template [[host_name("linear_operator_" #S)]] kernel void linear_operator<T>( \
    device const T*, device const uint*, device const uint*, device const uint*, \
    device const T*, device T*, constant uint*, constant T&, constant T&, uint);
INSTANTIATE_OPERATOR(float, f32)
INSTANTIATE_OPERATOR(float2, c32)
