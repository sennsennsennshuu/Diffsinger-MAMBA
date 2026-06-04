/*
 * SSM Optimizer - Main API Implementation
 *
 * C API implementation for the SSM Optimizer library.
 */

#include "../ssm_optimizer.h"
#include "selective_scan.h"
#include "state_cache.h"
#include <cstring>
#include <memory>

// Version information
#define SSM_VERSION_MAJOR 1
#define SSM_VERSION_MINOR 0
#define SSM_VERSION_PATCH 0

static const char* SSM_VERSION_STRING = "1.0.0";

// Global configuration
static ssm::ScanConfig g_config;
static std::unique_ptr<ssm::StateCache> g_state_cache;
static std::mutex g_config_mutex;

// C API Implementation
extern "C" {

const char* SSM_GetVersion(void) {
    return SSM_VERSION_STRING;
}

int SSM_GetMajorVersion(void) {
    return SSM_VERSION_MAJOR;
}

int SSM_GetMinorVersion(void) {
    return SSM_VERSION_MINOR;
}

int SSM_GetPatchVersion(void) {
    return SSM_VERSION_PATCH;
}

void SSM_GetDefaultConfig(SSM_Config* config) {
    if (!config) return;
    
    config->use_simd = 1;
    config->use_openmp = 1;
    config->chunk_size = 64;
    config->num_threads = 0;  // Use all available
}

void SSM_SetGlobalConfig(const SSM_Config* config) {
    if (!config) return;
    
    std::lock_guard<std::mutex> lock(g_config_mutex);
    g_config.use_simd = config->use_simd != 0;
    g_config.use_openmp = config->use_openmp != 0;
    g_config.chunk_size = config->chunk_size;
    g_config.num_threads = config->num_threads;
}

int SSM_SelectiveScan_F32(
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
) {
    if (!input || !dt || !A || !B || !C || !output) {
        return -1;  // Invalid argument
    }
    
    if (batch_size <= 0 || seq_len <= 0 || d_inner <= 0 || 
        n_heads <= 0 || d_state <= 0 || head_dim <= 0) {
        return -2;  // Invalid dimensions
    }
    
    if (d_inner != n_heads * head_dim) {
        return -3;  // Dimension mismatch
    }
    
    try {
        ssm::ScanConfig config;
        {
            std::lock_guard<std::mutex> lock(g_config_mutex);
            config = g_config;
        }
        
        ssm::selective_scan(
            input, dt, A, B, C, D,
            output, batch_size, seq_len, d_inner, n_heads, d_state, head_dim,
            config
        );
        
        return 0;  // Success
    } catch (...) {
        return -4;  // Runtime error
    }
}

int SSM_SelectiveScan_F32_Optimized(
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
    int head_dim
) {
    // For now, optimized version just calls the regular version
    // The buffer parameter is reserved for future optimizations
    (void)buffer;  // Unused for now
    
    return SSM_SelectiveScan_F32(
        input, dt, A, B, C, D,
        output, batch_size, seq_len, d_inner, n_heads, d_state, head_dim
    );
}

int SSM_SelectiveScan_Batch_F32(
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
) {
    if (!inputs || !dts || !As || !Bs || !Cs || !outputs || 
        !batch_sizes || !seq_lens) {
        return -1;
    }
    
    if (num_sequences <= 0) {
        return -2;
    }
    
    try {
        for (int i = 0; i < num_sequences; i++) {
            int ret = SSM_SelectiveScan_F32(
                inputs[i], dts[i], As[i], Bs[i], Cs[i], Ds ? Ds[i] : nullptr,
                outputs[i], batch_sizes[i], seq_lens[i], d_inner, n_heads, d_state, head_dim
            );
            if (ret != 0) {
                return ret;
            }
        }
        return 0;
    } catch (...) {
        return -4;
    }
}

// State cache C API
struct SSM_StateCache {
    ssm::StateCache cache;
    explicit SSM_StateCache(int max_size) : cache(max_size) {}
};

SSM_StateCache* SSM_StateCache_Create(int max_size) {
    try {
        return new SSM_StateCache(max_size);
    } catch (...) {
        return nullptr;
    }
}

void SSM_StateCache_Destroy(SSM_StateCache* cache) {
    delete cache;
}

int SSM_StateCache_Get(
    SSM_StateCache* cache,
    const char* key,
    float* state,
    int state_size
) {
    if (!cache || !key || !state || state_size <= 0) {
        return -1;
    }
    
    try {
        bool found = cache->cache.get(key, state, state_size);
        return found ? 0 : 1;  // 0 = found, 1 = not found
    } catch (...) {
        return -4;
    }
}

void SSM_StateCache_Set(
    SSM_StateCache* cache,
    const char* key,
    const float* state,
    int state_size
) {
    if (!cache || !key || !state || state_size <= 0) {
        return;
    }
    
    try {
        cache->cache.set(key, state, state_size);
    } catch (...) {
        // Silently ignore errors
    }
}

void SSM_StateCache_Clear(SSM_StateCache* cache) {
    if (!cache) return;
    
    try {
        cache->cache.clear();
    } catch (...) {
        // Silently ignore errors
    }
}

} // extern "C"
