"""Incremental SSM computation across diffusion steps"""
import numpy as np
from typing import Dict, Optional


class IncrementalSSM:
    """Reuse SSM computations between diffusion steps
    
    SSM formula: h_t = A * h_{t-1} + B * x_t
    By caching states from previous steps, we can avoid redundant computation.
    """
    
    def __init__(self, d_model: int, d_state: int = 128):
        """Initialize incremental SSM
        
        Args:
            d_model: Model dimension
            d_state: State dimension
        """
        self.d_model = d_model
        self.d_state = d_state
        self._prev_states: Dict[int, np.ndarray] = {}
    
    def compute(self, step: int, x: np.ndarray, dt: np.ndarray, 
                A: np.ndarray, B: np.ndarray, C: np.ndarray,
                D: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute SSM with state reuse
        
        Args:
            step: Current diffusion step
            x: Input tensor [batch, seq_len, d_model]
            dt: Time delta [batch, seq_len, d_state]
            A: State transition matrix [d_state]
            B: Input matrix [batch, seq_len, d_state]
            C: Output matrix [batch, seq_len, d_state]
            D: Skip connection (optional)
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Check if we can reuse state from previous step
        if step > 0 and (step - 1) in self._prev_states:
            # Incremental computation
            prev_state = self._prev_states[step - 1]
            output = self._incremental_scan(x, dt, A, B, C, prev_state, D)
        else:
            # Full computation
            output = self._full_scan(x, dt, A, B, C, D)
        
        # Cache state for next step
        self._prev_states[step] = output.copy()
        return output
    
    def _full_scan(self, x: np.ndarray, dt: np.ndarray, A: np.ndarray,
                   B: np.ndarray, C: np.ndarray, 
                   D: Optional[np.ndarray] = None) -> np.ndarray:
        """Full selective scan computation"""
        batch_size, seq_len, _ = x.shape
        
        # Discretize: A_bar = exp(dt * A)
        A_bar = np.exp(dt * A[None, None, :])  # [batch, seq, d_state]
        
        # B_bar = dt * B
        B_bar = dt * B  # [batch, seq, d_state]
        
        # Sequential scan
        h = np.zeros((batch_size, self.d_state))
        outputs = []
        
        for t in range(seq_len):
            # h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
            h = A_bar[:, t, :] * h + B_bar[:, t, :] * x[:, t, :self.d_state]
            
            # y_t = C_t * h_t (+ D * x_t)
            y = C[:, t, :] * h
            if D is not None:
                y = y + D[None, :] * x[:, t, :self.d_state]
            
            outputs.append(y)
        
        return np.stack(outputs, axis=1)  # [batch, seq, d_state]
    
    def _incremental_scan(self, x: np.ndarray, dt: np.ndarray, A: np.ndarray,
                          B: np.ndarray, C: np.ndarray, 
                          prev_state: np.ndarray,
                          D: Optional[np.ndarray] = None) -> np.ndarray:
        """Incremental scan reusing previous state"""
        # For now, same as full scan but starting from prev_state
        # In a more optimized version, we could reuse intermediate computations
        batch_size, seq_len, _ = x.shape
        
        A_bar = np.exp(dt * A[None, None, :])
        B_bar = dt * B
        
        h = prev_state[:, -1, :].copy()  # Start from last state of previous step
        outputs = []
        
        for t in range(seq_len):
            h = A_bar[:, t, :] * h + B_bar[:, t, :] * x[:, t, :self.d_state]
            y = C[:, t, :] * h
            if D is not None:
                y = y + D[None, :] * x[:, t, :self.d_state]
            outputs.append(y)
        
        return np.stack(outputs, axis=1)
    
    def reset(self):
        """Reset cached states for new inference"""
        self._prev_states.clear()
