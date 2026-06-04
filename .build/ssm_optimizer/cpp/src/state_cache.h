/*
 * State Cache for Incremental SSM Inference
 * 
 * LRU cache for storing and reusing SSM states across diffusion steps.
 */

#ifndef SSM_STATE_CACHE_H
#define SSM_STATE_CACHE_H

#include <string>
#include <unordered_map>
#include <list>
#include <vector>
#include <mutex>

namespace ssm {

// LRU Cache implementation for SSM states
class StateCache {
public:
    explicit StateCache(size_t max_size);
    ~StateCache();
    
    // Get cached state
    bool get(const std::string& key, float* state, size_t state_size);
    
    // Set cached state
    void set(const std::string& key, const float* state, size_t state_size);
    
    // Clear all cached states
    void clear();
    
    // Get current cache size
    size_t size() const;
    
    // Get maximum cache size
    size_t max_size() const;

private:
    struct CacheEntry {
        std::vector<float> data;
        std::list<std::string>::iterator lru_iter;
    };
    
    size_t max_size_;
    std::unordered_map<std::string, CacheEntry> cache_;
    std::list<std::string> lru_list_;  // Most recent at front
    mutable std::mutex mutex_;
};

// Thread-safe wrapper
class ThreadSafeStateCache {
public:
    explicit ThreadSafeStateCache(size_t max_size);
    ~ThreadSafeStateCache();
    
    bool get(const std::string& key, float* state, size_t state_size);
    void set(const std::string& key, const float* state, size_t state_size);
    void clear();

private:
    StateCache cache_;
    std::mutex mutex_;
};

} // namespace ssm

#endif // SSM_STATE_CACHE_H
