"""SSM State Cache for Incremental Inference"""
import numpy as np
from typing import Dict, Optional


class SSMStateCache:
    """Cache and reuse SSM states across diffusion steps
    
    Uses LRU (Least Recently Used) eviction policy to manage memory.
    """
    
    def __init__(self, max_cache_size: int = 100):
        """Initialize state cache
        
        Args:
            max_cache_size: Maximum number of states to cache
        """
        self.max_cache_size = max_cache_size
        self._cache: Dict[str, np.ndarray] = {}
        self._access_count: Dict[str, int] = {}
        self._access_counter = 0
    
    def get_state(self, key: str) -> Optional[np.ndarray]:
        """Get cached SSM state
        
        Args:
            key: Cache key (e.g., "step_5")
            
        Returns:
            Cached state array or None if not found
        """
        if key in self._cache:
            self._access_counter += 1
            self._access_count[key] = self._access_counter
            return self._cache[key].copy()
        return None
    
    def set_state(self, key: str, state: np.ndarray):
        """Cache SSM state with LRU eviction
        
        Args:
            key: Cache key
            state: State array to cache
        """
        # Evict if cache is full
        if len(self._cache) >= self.max_cache_size and key not in self._cache:
            # Find LRU key
            lru_key = min(self._access_count, key=self._access_count.get)
            del self._cache[lru_key]
            del self._access_count[lru_key]
        
        # Store state (make a copy to prevent external modification)
        self._cache[key] = state.copy()
        self._access_counter += 1
        self._access_count[key] = self._access_counter
    
    def clear(self):
        """Clear all cached states"""
        self._cache.clear()
        self._access_count.clear()
        self._access_counter = 0
    
    def __len__(self) -> int:
        """Return number of cached states"""
        return len(self._cache)
