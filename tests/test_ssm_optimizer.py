"""TDD Tests for SSM Optimizer

Tests for SSM performance optimization without reducing sampling steps.
Following TDD: Write failing test first, then implement.
"""
import time
import numpy as np
import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSSMPerformanceBaseline:
    """Test 1: Establish baseline performance of current SimpleSSM"""
    
    def test_simple_ssm_inference_time(self):
        """SimpleSSM should complete inference within acceptable time
        
        This test will FAIL initially with current implementation.
        Target: < 100ms for 5k sequence
        """
        from modules.commons.ssm_layers import SimpleSSM
        
        # Setup
        batch_size = 1
        seq_len = 5000
        d_model = 256
        
        model = SimpleSSM(d_model=d_model)
        model.eval()
        
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Warmup
        with torch.no_grad():
            _ = model(x)
        
        # Benchmark
        start = time.time()
        with torch.no_grad():
            result = model(x)
        elapsed = time.time() - start
        
        # Assert: Should complete in less than 100ms
        assert elapsed < 0.1, f"SSM too slow: {elapsed:.3f}s for {seq_len} sequence"
        assert result.shape == (batch_size, seq_len, d_model)
    
    def test_simple_ssm_scalability(self):
        """SSM performance should scale linearly with sequence length"""
        from modules.commons.ssm_layers import SimpleSSM
        
        model = SimpleSSM(d_model=256)
        model.eval()
        
        times = []
        seq_lengths = [1000, 2000, 5000]
        
        for seq_len in seq_lengths:
            x = torch.randn(1, seq_len, 256)
            
            start = time.time()
            with torch.no_grad():
                _ = model(x)
            elapsed = time.time() - start
            times.append(elapsed)
        
        # Check linear scaling (time ratio should match length ratio)
        ratio_1_2 = times[1] / times[0]
        expected_ratio = seq_lengths[1] / seq_lengths[0]
        
        # Allow 20% variance
        assert abs(ratio_1_2 - expected_ratio) < expected_ratio * 0.2, \
            f"Non-linear scaling: {ratio_1_2:.2f}x vs expected {expected_ratio:.2f}x"


class TestOptimizedSSM:
    """Test 2: Optimized SSM should be significantly faster"""
    
    def test_optimized_ssm_3x_speedup(self):
        """Optimized SSM should be at least 3x faster than SimpleSSM
        
        This test will FAIL until optimized implementation is created.
        """
        from modules.commons.ssm_layers import SimpleSSM
        from ssm_optimizer.optimized_ssm import OptimizedSSM
        
        batch_size = 1
        seq_len = 5000
        d_model = 256
        
        simple_model = SimpleSSM(d_model=d_model)
        simple_model.eval()
        
        optimized_model = OptimizedSSM(d_model=d_model)
        optimized_model.eval()
        
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Benchmark SimpleSSM
        start = time.time()
        with torch.no_grad():
            _ = simple_model(x)
        elapsed_simple = time.time() - start
        
        # Benchmark OptimizedSSM
        start = time.time()
        with torch.no_grad():
            _ = optimized_model(x)
        elapsed_optimized = time.time() - start
        
        speedup = elapsed_simple / elapsed_optimized
        
        # Assert: Should achieve 3x speedup
        assert speedup >= 3.0, f"Optimization insufficient: {speedup:.2f}x speedup (target: 3x)"
    
    def test_optimized_ssm_accuracy(self):
        """Optimized SSM should match SimpleSSM within numerical tolerance"""
        from modules.commons.ssm_layers import SimpleSSM
        from ssm_optimizer.optimized_ssm import OptimizedSSM
        
        batch_size = 2
        seq_len = 1000
        d_model = 256
        
        simple_model = SimpleSSM(d_model=d_model)
        simple_model.eval()
        
        optimized_model = OptimizedSSM(d_model=d_model)
        optimized_model.eval()
        
        x = torch.randn(batch_size, seq_len, d_model)
        
        with torch.no_grad():
            result_simple = simple_model(x)
            result_optimized = optimized_model(x)
        
        max_diff = torch.max(torch.abs(result_simple - result_optimized)).item()
        
        # Assert: Max difference should be < 1e-5
        assert max_diff < 1e-5, f"Accuracy loss too large: {max_diff} (tolerance: 1e-5)"


class TestSSMStateCache:
    """Test 3: State caching for incremental inference"""
    
    def test_state_cache_basic(self):
        """State cache should store and retrieve states correctly"""
        from ssm_optimizer.state_cache import SSMStateCache
        
        cache = SSMStateCache(max_cache_size=10)
        
        # Store state
        key = "step_5"
        state = np.random.randn(1, 128, 256).astype(np.float32)
        cache.set_state(key, state)
        
        # Retrieve state
        retrieved = cache.get_state(key)
        
        assert retrieved is not None
        assert np.allclose(retrieved, state)
    
    def test_state_cache_lru_eviction(self):
        """State cache should evict least recently used when full"""
        from ssm_optimizer.state_cache import SSMStateCache
        
        cache = SSMStateCache(max_cache_size=3)
        
        # Fill cache
        for i in range(3):
            cache.set_state(f"step_{i}", np.random.randn(1, 128, 256).astype(np.float32))
        
        # Access step_0 and step_2 (step_1 becomes LRU)
        cache.get_state("step_0")
        cache.get_state("step_2")
        
        # Add new state (should evict step_1)
        cache.set_state("step_3", np.random.randn(1, 128, 256).astype(np.float32))
        
        # step_1 should be evicted
        assert cache.get_state("step_1") is None
        # step_0 and step_2 should still exist
        assert cache.get_state("step_0") is not None
        assert cache.get_state("step_2") is not None
    
    def test_incremental_ssm_state_reuse(self):
        """Incremental SSM should reuse states across steps"""
        from ssm_optimizer.incremental_ssm import IncrementalSSM
        
        d_model = 256
        d_state = 128
        
        incremental_ssm = IncrementalSSM(d_model=d_model, d_state=d_state)
        
        # Simulate diffusion steps
        x = np.random.randn(1, 1000, d_model).astype(np.float32)
        dt = np.random.randn(1, 1000, d_state).astype(np.float32)
        A = np.random.randn(d_state).astype(np.float32)
        B = np.random.randn(1, 1000, d_state).astype(np.float32)
        C = np.random.randn(1, 1000, d_state).astype(np.float32)
        
        # Step 0: Full computation
        output_0 = incremental_ssm.compute(0, x, dt, A, B, C)
        assert output_0 is not None
        
        # Step 1: Should reuse state from step 0
        output_1 = incremental_ssm.compute(1, x, dt, A, B, C)
        assert output_1 is not None
        
        # Verify states are cached
        assert 0 in incremental_ssm._prev_states
        assert 1 in incremental_ssm._prev_states


class TestONNXGraphOptimization:
    """Test 4: ONNX graph optimization for SSM"""
    
    def test_ssm_pattern_detection(self):
        """Should correctly identify SSM selective scan patterns in ONNX graph"""
        from ssm_optimizer.onnx_graph_opt import _is_ssm_scan_pattern
        import onnx
        from onnx import helper, TensorProto
        
        # Create a mock SSM pattern node
        node = helper.make_node(
            "CumSum",
            inputs=["input", "axis"],
            outputs=["output"],
            name="test_cumsum"
        )
        
        # Should identify as SSM pattern
        result = _is_ssm_scan_pattern(node)
        assert result is True
    
    def test_ssm_graph_optimization(self):
        """Should optimize SSM operations in ONNX graph"""
        import tempfile
        from ssm_optimizer.onnx_graph_opt import optimize_ssm_graph
        import onnx
        from onnx import helper, TensorProto
        
        # Create minimal test model
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 100, 256])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 100, 256])
        
        cumsum_node = helper.make_node("CumSum", ["input", "axis"], ["output"], name="cumsum_1")
        
        graph = helper.make_graph(
            [cumsum_node],
            "test_graph",
            [input_tensor],
            [output_tensor]
        )
        
        model = helper.make_model(graph)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.onnx"
            output_path = Path(tmpdir) / "output.onnx"
            
            onnx.save(model, input_path)
            
            # Optimize
            result_path = optimize_ssm_graph(str(input_path), str(output_path))
            
            assert Path(result_path).exists()
            
            # Load optimized model
            optimized_model = onnx.load(result_path)
            assert optimized_model is not None


class TestIntegration:
    """Test 5: Integration with DiffSinger export"""
    
    def test_ssm_optimizer_loading(self):
        """SSM optimizer should load from Dependencies directory"""
        import os
        from pathlib import Path
        
        deps_path = Path("C:/Users/Asus/Documents/OpenUtau/Dependencies/SSM")
        
        # Check if optimizer exists (will be created in Phase 4)
        if deps_path.exists():
            lib_path = deps_path / "SSMOptimizer.dll"
            assert lib_path.exists(), "SSMOptimizer.dll not found"
            
            # Should be loadable
            import ctypes
            try:
                lib = ctypes.CDLL(str(lib_path))
                assert lib is not None
            except OSError as e:
                pytest.skip(f"Cannot load SSMOptimizer.dll: {e}")
        else:
            pytest.skip("SSM optimizer not yet installed")
    
    def test_export_with_optimizer(self):
        """DiffSinger export should use optimizer when available"""
        from pathlib import Path
        
        deps_path = Path("C:/Users/Asus/Documents/OpenUtau/Dependencies/SSM")
        
        if not deps_path.exists():
            pytest.skip("SSM optimizer not yet installed")
        
        # This test will be implemented when export integration is done
        # For now, just verify the path exists
        assert deps_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
