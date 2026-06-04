"""SSM Optimizer for DiffSinger

High-performance SSM implementations for OpenUtau DiffSinger voicebanks.
Provides optimized CPU execution without reducing sampling steps.
"""

from .optimized_ssm import OptimizedSSM
from .state_cache import SSMStateCache
from .incremental_ssm import IncrementalSSM

__all__ = [
    'OptimizedSSM',
    'SSMStateCache', 
    'IncrementalSSM',
]

__version__ = '1.0.0'
