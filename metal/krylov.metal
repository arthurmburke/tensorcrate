// CGS2 coefficients and norms stay resident until both passes finish.
inline float krylov_hypot(float a, float b) {
    if (!isfinite(a) || !isfinite(b)) return a+b;
    float scale=max(a,b);
    if(scale==0.0f) return 0.0f;
    return scale*sqrt((a/scale)*(a/scale)+(b/scale)*(b/scale));
}
template<typename T> kernel void krylov_step(
    device const T* basis [[buffer(0)]],device T* work [[buffer(1)]],
    device T* partial [[buffer(2)]],device T* coefficients [[buffer(3)]],
    constant uint4& p [[buffer(4)]],uint tid [[thread_position_in_grid]]) {
    uint n=p.x,count=p.y,chunks=p.z,stage=p.w;
    if(stage==0 || stage==3) {
        if(tid>=chunks*(count+(stage==0?1:0))) return;
        uint k=tid/chunks,chunk=tid%chunks,begin=chunk*256;
        T sum=T(0); float length=0.0f;
        uint end=begin+min(256u,n-begin);
        for(uint i=begin;i<end;++i) {
            if(k==count) length=krylov_hypot(length,operator_length(work[i]));
            else sum+=operator_mul(operator_conj(basis[k*n+i]),work[i]);
        }
        partial[tid]=k==count?operator_real<T>(length):sum;
    } else if(stage==1 || stage==4) {
        if(tid>=count+(stage==1?1:0)) return;
        T sum=T(0); float length=0.0f;
        for(uint c=0;c<chunks;++c) {
            if(tid==count) length=krylov_hypot(length,operator_length(partial[tid*chunks+c]));
            else sum+=partial[tid*chunks+c];
        }
        if(tid==count) coefficients[2*count]=operator_real<T>(length);
        else {
            coefficients[tid]=sum;
            if(stage==1) coefficients[count+tid]=sum;
            else coefficients[count+tid]+=sum;
        }
    } else if(stage==2) {
        if(tid>=n) return;
        T sum=T(0);
        for(uint k=0;k<count;++k) sum+=operator_mul(basis[k*n+tid],coefficients[k]);
        work[tid]-=sum;
    } else if(stage==6) {
        if(tid>=chunks) return;
        float length=0.0f;
        uint begin=tid*256,end=begin+min(256u,n-begin);
        for(uint i=begin;i<end;++i) length=krylov_hypot(length,operator_length(work[i]));
        partial[tid]=operator_real<T>(length);
    } else if(stage==7) {
        if(tid!=0) return;
        float length=0.0f;
        for(uint c=0;c<chunks;++c) length=krylov_hypot(length,operator_length(partial[c]));
        coefficients[2*count+1]=operator_real<T>(length);
    }
}
#define INSTANTIATE_KRYLOV(T,S) \
template [[host_name("krylov_step_" #S)]] kernel void krylov_step<T>(device const T*,device T*,device T*,device T*,constant uint4&,uint);
INSTANTIATE_KRYLOV(float,f32)
INSTANTIATE_KRYLOV(float2,c32)

template<typename T> inline T triangular_div(T a,T b) { return a/b; }
template<> inline float2 triangular_div(float2 a,float2 b) {
    float scale=max(abs(b.x),abs(b.y));
    float2 scaled=b/scale,numerator=a/scale;
    float denominator=scaled.x*scaled.x+scaled.y*scaled.y;
    return float2(numerator.x*scaled.x+numerator.y*scaled.y,numerator.y*scaled.x-numerator.x*scaled.y)/denominator;
}
template<typename T> kernel void triangular_level(
    device const T* values [[buffer(0)]],device const uint* offsets [[buffer(1)]],
    device const uint* columns [[buffer(2)]],device const uint* positions [[buffer(3)]],
    device const uint* rows [[buffer(4)]],device const T* x [[buffer(5)]],device T* result [[buffer(6)]],
    constant uint4& p [[buffer(7)]],constant uint& unit_diagonal [[buffer(8)]],uint tid [[thread_position_in_grid]]) {
    if(tid>=p.y) return;
    uint row=rows[p.x+tid];
    T rhs=x[p.z+row],diagonal=operator_real<T>(1.0f);
    for(uint k=offsets[row];k<offsets[row+1];++k) {
        uint col=columns[k];
        T value=p.w?operator_conj(values[positions[k]]):values[k];
        if(col==row) { if(!unit_diagonal) diagonal=value; }
        else if(!operator_zero(value)) rhs-=operator_mul(value,result[col]);
    }
    result[row]=triangular_div(rhs,diagonal);
}
#define INSTANTIATE_TRIANGULAR(T,S) \
template [[host_name("triangular_level_" #S)]] kernel void triangular_level<T>(device const T*,device const uint*,device const uint*,device const uint*,device const uint*,device const T*,device T*,constant uint4&,constant uint&,uint);
INSTANTIATE_TRIANGULAR(float,f32)
INSTANTIATE_TRIANGULAR(float2,c32)

template<typename T> kernel void cross_values(device const T* a [[buffer(0)]],device const T* b [[buffer(1)]],device T* result [[buffer(2)]],constant uint& n [[buffer(3)]],uint tid [[thread_position_in_grid]]) {
    if(tid>=n) return;
    uint base=(tid/3)*3, i=(tid%3+1)%3, j=(tid%3+2)%3;
    result[tid]=operator_mul(a[base+i],b[base+j])-operator_mul(a[base+j],b[base+i]);
}
#define INSTANTIATE_CROSS(T,S) \
template [[host_name("cross_values_" #S)]] kernel void cross_values<T>(device const T*,device const T*,device T*,constant uint&,uint);
INSTANTIATE_CROSS(float,f32)
INSTANTIATE_CROSS(half,f16)
INSTANTIATE_CROSS(bfloat,bf16)
INSTANTIATE_CROSS(float2,c32)
