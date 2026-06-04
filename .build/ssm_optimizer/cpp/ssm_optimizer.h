/*
 * SSM Optimizer - High-performance Selective Scan for DiffSinger
 * 
 * This library provides optimized CPU implementations of SSM selective scan
 * for use with ONNX Runtime custom operators.
 */

#ifndef SSM_OPTIMIZER_H
#define SSM_OPTIMIZER_H

#ifdef _WIN32
    #ifdef SSM_OPTIMIZER_EXPORTS
        #define SSM_API __declspec(dllexport)
    #else
        #define SSM_API __declspec(dllimport)
    #endif
#else
    #define SSM_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Version information
 */
SSM_API const char* SSM_GetVersion(void);
SSM_API int SSM_GetMajorVersion(void);
SSM_API int SSM_GetMinorVersion(void);
SSM_API int SSM_GetPatchVersion(void);

/*
 * Configuration
 */
typedef struct {
    int use_simd;        // Enable SIMD optimizations (AVX/AVX2)
    int use_openmp;      // Enable OpenMP parallelization
    int chunk_size;      // Processing chunk size for cache efficiency
    int num_threads;     // Number of threads for parallel execution
} SSM_Config;

SSM_API void SSM_GetDefaultConfig(SSM_Config* config);
SSM_API void SSM_SetGlobalConfig(const SSM_Config* config);

/*
 * Selective Scan Implementation
 * 
 * Computes: h_t = A_t * h_{t-1} + B_t * x_t
 *           y_t = C_t * h_t
 * 
 * Args:
 *   input: Input tensor [batch, seq_len, d_inner]
 *   dt: Time delta [batch, seq_len, n_heads]
 *   A: State transition [n_heads, d_state]
 *   B: Input matrix [batch, seq_len, d_state]
 *   C: Output matrix [batch, seq_len, d_state]
 *   D: Skip connection [n_heads] (optional, can be NULL)
 *   output: Output tensor [batch, seq_len, d_inner]
 *   batch_size: Batch dimension
 *   seq_len: Sequence length
 *   d_inner: Inner dimension (n_heads * head_dim)
 *   n_heads: Number of heads
 *   d_state: State dimension
 *   head_dim: Dimension per head
 * 
 * Returns:
 *   0 on success, non-zero on error
 */
SSM_API int SSM_SelectiveScan_F32(
    const float* input,
    const float* dt,
    const float* A,
    const float* B,
    const float* C,
    const float* D,
    float* output,
    int batch_size,
    int seq_len,
    int d_inner,
    int n_heads,
    int d_state,
    int head_dim
);

/*
 * Optimized version with pre-allocated buffers
 */
SSM_API int SSM_SelectiveScan_F32_Optimized(
    const float* input,
    const float* dt,
    const float* A,
    const float* B,
    const float* C,
    const float* D,
    float* output,
    float* buffer,  // Pre-allocated buffer for intermediate results
    int batch_size,
    int seq_len,
    int d_inner,
    int n_heads,
    int d_state,
    int head_dim
);

/*
 * Batch processing - multiple sequences at once
 */
SSM_API int SSM_SelectiveScan_Batch_F32(
    const float* const* inputs,
    const float* const* dts,
    const float* const* As,
    const float* const* Bs,
    const float* const* Cs,
    const float* const* Ds,
    float** outputs,
    int num_sequences,
    const int* batch_sizes,
    const int* seq_lens,
    int d_inner,
    int n_heads,
    int d_state,
    int head_dim
);

/*
 * State caching for incremental inference
 */
typedef struct SSM_StateCache SSM_StateCache;

SSM_API SSM_StateCache* SSM_StateCache_Create(int max_size);
SSM_API void SSM_StateCache_Destroy(SSM_StateCache* cache);
SSM_API int SSM_StateCache_Get(
    SSM_StateCache* cache,
    const char* key,
    float* state,
    int state_size
);
SSM_API void SSM_StateCache_Set(
    SSM_StateCache* cache,
    const char* key,
    const float* state,
    int state_size
);
SSM_API void SSM_StateCache_Clear(SSM_StateCache* cache);

#ifdef __cplusplus
}
#endif

#endif /* SSM_OPTIMIZER_H */
