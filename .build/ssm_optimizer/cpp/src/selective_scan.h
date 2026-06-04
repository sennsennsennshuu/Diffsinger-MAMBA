/*
 * Selective Scan Implementation
 * 
 * High-performance SSM selective scan with SIMD and OpenMP optimizations.
 */

#ifndef SSM_SELECTIVE_SCAN_H
#define SSM_SELECTIVE_SCAN_H

#include <cstddef>

namespace ssm {

// Configuration for selective scan
struct ScanConfig {
    bool use_simd = true;
    bool use_openmp = true;
    int num_threads = 0;  // 0 = use all available
    int chunk_size = 64;  // Process state dimensions in chunks
};

/*
 * Selective scan for a single head
 * 
 * Computes: h_t = A_t * h_{t-1} + B_t * x_t
 *           y_t = C_t * h_t
 * 
 * Args:
 *   x: Input [seq_len, head_dim]
 *   dt: Time delta [seq_len]
 *   A: State transition [d_state]
 *   B: Input matrix [seq_len, d_state]
 *   C: Output matrix [seq_len, d_state]
 *   y: Output [seq_len, head_dim]
 *   seq_len: Sequence length
 *   head_dim: Dimension per head
 *   d_state: State dimension
 */
void selective_scan_head(
    const float* x,
    const float* dt,
    const float* A,
    const float* B,
    const float* C,
    float* y,
    int seq_len,
    int head_dim,
    int d_state,
    const ScanConfig& config
);

/*
 * Selective scan for batch
 * 
 * Args:
 *   input: Input [batch, seq_len, d_inner]
 *   dt: Time delta [batch, seq_len, n_heads]
 *   A: State transition [n_heads, d_state]
 *   B: Input matrix [batch, seq_len, d_state]
 *   C: Output matrix [batch, seq_len, d_state]
 *   output: Output [batch, seq_len, d_inner]
 *   batch_size: Batch dimension
 *   seq_len: Sequence length
 *   d_inner: Inner dimension (n_heads * head_dim)
 *   n_heads: Number of heads
 *   d_state: State dimension
 *   head_dim: Dimension per head
 */
void selective_scan(
    const float* input,
    const float* dt,
    const float* A,
    const float* B,
    const float* C,
    const float* D,  // Optional skip connection
    float* output,
    int batch_size,
    int seq_len,
    int d_inner,
    int n_heads,
    int d_state,
    int head_dim,
    const ScanConfig& config
);

/*
 * Optimized version with pre-allocated buffers
 */
void selective_scan_optimized(
    const float* input,
    const float* dt,
    const float* A,
    const float* B,
    const float* C,
    const float* D,
    float* output,
    float* buffer,  // Pre-allocated buffer
    int batch_size,
    int seq_len,
    int d_inner,
    int n_heads,
    int d_state,
    int head_dim,
    const ScanConfig& config
);

} // namespace ssm

#endif // SSM_SELECTIVE_SCAN_H
