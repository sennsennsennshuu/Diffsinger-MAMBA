/*
 * State Cache Implementation
 */

#include "state_cache.h"
#include <algorithm>
#include <cstring>

namespace ssm {

// StateCache implementation
StateCache::StateCache(size_t max_size) : max_size_(max_size) {}

StateCache::~StateCache() = default;

bool StateCache::get(const std::string& key, float* state, size_t state_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = cache_.find(key);
    if (it == cache_.end()) {
        return false;
    }
    
    // Copy data
    if (it->second.data.size() != state_size) {
        return false;
    }
    std::memcpy(state, it->second.data.data(), state_size * sizeof(float));
    
    // Update LRU: move to front
    lru_list_.erase(it->second.lru_iter);
    lru_list_.push_front(key);
    it->second.lru_iter = lru_list_.begin();
    
    return true;
}

void StateCache::set(const std::string& key, const float* state, size_t state_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Check if key already exists
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        // Update existing entry
        it->second.data.assign(state, state + state_size);
        // Update LRU
        lru_list_.erase(it->second.lru_iter);
        lru_list_.push_front(key);
        it->second.lru_iter = lru_list_.begin();
        return;
    }
    
    // Evict if necessary
    if (cache_.size() >= max_size_ && !lru_list_.empty()) {
        const std::string& lru_key = lru_list_.back();
        cache_.erase(lru_key);
        lru_list_.pop_back();
    }
    
    // Insert new entry
    lru_list_.push_front(key);
    CacheEntry entry;
    entry.data.assign(state, state + state_size);
    entry.lru_iter = lru_list_.begin();
    cache_[key] = std::move(entry);
}

void StateCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
    lru_list_.clear();
}

size_t StateCache::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_.size();
}

size_t StateCache::max_size() const {
    return max_size_;
}

// ThreadSafeStateCache implementation
ThreadSafeStateCache::ThreadSafeStateCache(size_t max_size) : cache_(max_size) {}

ThreadSafeStateCache::~ThreadSafeStateCache() = default;

bool ThreadSafeStateCache::get(const std::string& key, float* state, size_t state_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_.get(key, state, state_size);
}

void ThreadSafeStateCache::set(const std::string& key, const float* state, size_t state_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.set(key, state, state_size);
}

void ThreadSafeStateCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
}

} // namespace ssm
