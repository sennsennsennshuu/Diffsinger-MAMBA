/*
 * Selective Scan Implementation
 * 
 * High-performance SSM selective scan with SIMD and OpenMP optimizations.
 * 
 * Math: h_s = A_bar * h_s + B_bar * x  (vector update)
 *       y += C_s * h_s                  (vector contribution)
 * 
 * Where h_s is a head_dim-dimensional vector (NOT a scalar).
 * This matches SimpleSSM's outer product: u = B * x (L,S,P)
 */
 
#include "selective_scan.h"
#include "simd_utils.h"
#include <cmath>
#include <cstring>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ssm {

// Fast approximate exp for CPU
inline float fast_exp(float x) {
    // Use std::exp for accuracy in production
    return std::exp(x);
}

// Selective scan for a single head with SIMD optimization
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
) {
    // State buffer: d_state vectors, each head_dim long
    // h[s * head_dim + d] = state for state_dim s, head_dim_pos d
    float* h = new float[d_state * head_dim]();
    
    // Process each timestep
    for (int t = 0; t < seq_len; t++) {
        const float dt_t = dt[t];
        const float* x_t = x + t * head_dim;
        const float* B_t = B + t * d_state;
        const float* C_t = C + t * d_state;
        float* y_t = y + t * head_dim;
        
        // Process each state dimension
        for (int s = 0; s < d_state; s++) {
            // A_bar = exp(dt * A[s]): dt only affects A, NOT B
            // SimpleSSM uses u = B * x (no dt multiplication on B)
            float A_bar = fast_exp(dt_t * A[s]);
            float B_s = B_t[s];
            float C_s = C_t[s];
            
            // h_s is head_dim-long vector: h[s * head_dim + d]
            float* h_s = h + s * head_dim;
            
            // Vector update: h_s = A_bar * (h_s + B_s * x_t)
            // SimpleSSM uses parallel scan which is equivalent to:
            //   h[t] = a_t * (h[t-1] + u_t)  where u_t = B[t,s] * x[t]
            // NOT the standard: h[t] = a_t * h[t-1] + u_t
            // The extra a_t factor on u_t is from the cumprod_a/cumprod_a_pad offset
            for (int d = 0; d < head_dim; d++) {
                float u = B_s * x_t[d];
                h_s[d] = A_bar * (h_s[d] + u);
                y_t[d] += C_s * h_s[d];
            }
        }
    }
    
    delete[] h;
}

// SIMD-optimized version for single head
void selective_scan_head_simd(
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
) {
    // State buffer: d_state vectors, each head_dim long
    float* h = new float[d_state * head_dim]();
    
    for (int t = 0; t < seq_len; t++) {
        const float dt_t = dt[t];
        const float* x_t = x + t * head_dim;
        const float* B_t = B + t * d_state;
        const float* C_t = C + t * d_state;
        float* y_t = y + t * head_dim;
        
        // Process each state dimension
        for (int s = 0; s < d_state; s++) {
            // A_bar = exp(dt * A[s]): dt only affects A, NOT B
            float A_bar = fast_exp(dt_t * A[s]);
            float B_s = B_t[s];
            float C_s = C_t[s];
            
            // h_s is head_dim-long vector
            float* h_s = h + s * head_dim;
            
            // SIMD vectorized: h = A * (h + B_s * x), y += C * h
            int d = 0;
            for (; d + SIMD_FLOAT_WIDTH <= head_dim; d += SIMD_FLOAT_WIDTH) {
                FloatVec h_vec = simd_load(h_s + d);
                FloatVec x_vec = simd_load(x_t + d);
                FloatVec B_vec = simd_set1(B_s);
                FloatVec A_vec = simd_set1(A_bar);
                
                // h = A * (h + B_s * x)
                h_vec = simd_mul(A_vec, simd_add(h_vec, simd_mul(B_vec, x_vec)));
                simd_store(h_s + d, h_vec);
                
                // y += C_s * h
                FloatVec y_vec = simd_load(y_t + d);
                y_vec = simd_add(y_vec, simd_mul(simd_set1(C_s), h_vec));
                simd_store(y_t + d, y_vec);
            }
            
            // Handle remaining elements
            for (; d < head_dim; d++) {
                float u = B_s * x_t[d];
                h_s[d] = A_bar * (h_s[d] + u);
                y_t[d] += C_s * h_s[d];
            }
        }
    }
    
    delete[] h;
}

// Main selective scan function
void selective_scan(
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
    int head_dim,
    const ScanConfig& config
) {
    const int use_simd = config.use_simd;
    
    #ifdef _OPENMP
    if (config.use_openmp) {
        int num_threads = config.num_threads > 0 ? config.num_threads : omp_get_max_threads();
        omp_set_num_threads(num_threads);
    }
    #endif
    
    // Process each batch
    #pragma omp parallel for if(config.use_openmp && batch_size > 1)
    for (int b = 0; b < batch_size; b++) {
        const float* input_b = input + b * seq_len * d_inner;
        const float* dt_b = dt + b * seq_len * n_heads;
        const float* B_b = B + b * seq_len * d_state;
        const float* C_b = C + b * seq_len * d_state;
        float* output_b = output + b * seq_len * d_inner;
        
        // Temporary output for each head
        float* y_head = new float[seq_len * head_dim];
        
        // Process each head
        for (int h = 0; h < n_heads; h++) {
            const float* x_h = input_b + h * head_dim;  // Interleaved
            const float* dt_h = dt_b + h;  // Interleaved
            const float* A_h = A + h * d_state;
            const float* B_h = B_b;
            const float* C_h = C_b;
            
            // Gather input for this head
            float* x_gathered = new float[seq_len * head_dim];
            float* dt_gathered = new float[seq_len];
            
            for (int t = 0; t < seq_len; t++) {
                dt_gathered[t] = dt_h[t * n_heads];
                for (int d = 0; d < head_dim; d++) {
                    x_gathered[t * head_dim + d] = input_b[t * d_inner + h * head_dim + d];
                }
            }
            
            // Initialize y_head to zero
            std::fill(y_head, y_head + seq_len * head_dim, 0.0f);
            
            // Compute selective scan for this head
            if (use_simd && head_dim >= SIMD_FLOAT_WIDTH) {
                selective_scan_head_simd(
                    x_gathered, dt_gathered, A_h, B_h, C_h, y_head,
                    seq_len, head_dim, d_state, config
                );
            } else {
                selective_scan_head(
                    x_gathered, dt_gathered, A_h, B_h, C_h, y_head,
                    seq_len, head_dim, d_state, config
                );
            }
            
            // Scatter output back
            for (int t = 0; t < seq_len; t++) {
                for (int d = 0; d < head_dim; d++) {
                    output_b[t * d_inner + h * head_dim + d] = y_head[t * head_dim + d];
                }
            }
            
            // Add D skip connection if provided
            if (D != nullptr) {
                float D_h = D[h];
                for (int t = 0; t < seq_len; t++) {
                    for (int d = 0; d < head_dim; d++) {
                        output_b[t * d_inner + h * head_dim + d] += 
                            D_h * input_b[t * d_inner + h * head_dim + d];
                    }
                }
            }
            
            delete[] x_gathered;
            delete[] dt_gathered;
        }
        
        delete[] y_head;
    }
}

// Optimized version with pre-allocated buffers
void selective_scan_optimized(
    const float* input,
    const float* dt,
    const float* A,
    const float* B,
    const float* C,
    const float* D,
    float* output,
    float* buffer,
    int batch_size,
    int seq_len,
    int d_inner,
    int n_heads,
    int d_state,
    int head_dim,
    const ScanConfig& config
) {
    // Delegate to regular scan for now
    selective_scan(
        input, dt, A, B, C, D, output,
        batch_size, seq_len, d_inner, n_heads, d_state, head_dim,
        config
    );
}

} // namespace ssm

/* ─── C-linkage wrapper for ONNX Custom Op ─── */
extern "C" {
void ssm_selective_scan_f32(
    const float* input, const float* dt, const float* A,
    const float* B, const float* C, const float* D,
    float* output,
    int batch_size, int seq_len, int d_inner,
    int n_heads, int d_state, int head_dim,
    int use_simd)
{
    ssm::ScanConfig config;
    config.use_simd = (use_simd != 0);
    config.use_openmp = true;
    config.chunk_size = d_state;
    config.num_threads = 0;
    
    ssm::selective_scan(
        input, dt, A, B, C, D, output,
        batch_size, seq_len, d_inner, n_heads, d_state, head_dim,
        config
    );
}
}
