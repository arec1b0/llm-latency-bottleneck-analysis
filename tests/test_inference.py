"""
Inference Engine Tests

Unit tests for inference engine and metrics.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import torch

from src.inference.metrics import InferenceTimer, TokenTimings
from src.inference.engine import InferenceEngine


class TestTokenTimings:
    """Tests for TokenTimings dataclass."""
    
    def test_token_timings_creation(self):
        """Test creating TokenTimings instance."""
        timings = TokenTimings(
            first_token_time=0.5,
            subsequent_token_times=[0.05, 0.04, 0.06],
            total_time=2.0,
        )
        
        assert timings.first_token_time == 0.5
        assert len(timings.subsequent_token_times) == 3
        assert timings.total_time == 2.0
    
    def test_average_tpot_calculation(self):
        """Test average TPOT calculation."""
        timings = TokenTimings(
            first_token_time=0.5,
            subsequent_token_times=[0.05, 0.04, 0.06],
            total_time=2.0,
        )
        
        expected_avg = (0.05 + 0.04 + 0.06) / 3
        assert timings.average_tpot == pytest.approx(expected_avg)
    
    def test_average_tpot_empty(self):
        """Test average TPOT with no subsequent tokens."""
        timings = TokenTimings(
            first_token_time=0.5,
            subsequent_token_times=[],
            total_time=0.5,
        )
        
        assert timings.average_tpot == 0.0
    
    def test_total_tokens(self):
        """Test total tokens calculation."""
        timings = TokenTimings(
            first_token_time=0.5,
            subsequent_token_times=[0.05, 0.04, 0.06],
            total_time=2.0,
        )
        
        # 1 (first token) + 3 (subsequent) = 4
        assert timings.total_tokens == 4


class TestInferenceTimer:
    """Tests for InferenceTimer."""
    
    def test_timer_initialization(self):
        """Test timer initialization."""
        timer = InferenceTimer()
        
        assert timer.start_time is None
        assert timer.first_token_time is None
        assert timer.token_times == []
        assert timer.end_time is None
    
    def test_timer_start(self):
        """Test starting timer."""
        timer = InferenceTimer()
        timer.start()
        
        assert timer.start_time is not None
        assert timer.token_times == []
    
    def test_timer_stop_without_start_raises_error(self):
        """Test that stopping without starting raises error."""
        timer = InferenceTimer()
        
        with pytest.raises(RuntimeError, match="Timer not started"):
            timer.stop()
    
    def test_record_first_token(self):
        """Test recording first token."""
        timer = InferenceTimer()
        timer.start()
        
        ttft = timer.record_first_token()
        
        assert ttft > 0
        assert timer.first_token_time is not None
    
    def test_record_first_token_without_start_raises_error(self):
        """Test that recording token without starting raises error."""
        timer = InferenceTimer()
        
        with pytest.raises(RuntimeError, match="Timer not started"):
            timer.record_first_token()
    
    def test_complete_timing_flow(self):
        """Test complete timing flow."""
        timer = InferenceTimer()
        
        # Start timer
        timer.start()
        assert timer.start_time is not None
        
        # Record first token
        ttft = timer.record_first_token()
        assert ttft > 0
        
        # Record subsequent tokens
        for _ in range(3):
            token_time = timer.record_token()
            assert token_time >= 0
        
        # Stop timer
        timings = timer.stop()
        
        assert isinstance(timings, TokenTimings)
        assert timings.first_token_time > 0
        assert timings.total_time > 0
        assert timings.total_tokens == 4  # 1 first + 3 subsequent
    
    def test_memory_tracking(self):
        """Test memory usage tracking."""
        timer = InferenceTimer()
        
        timer.start()
        timer.record_first_token()
        timings = timer.stop()
        
        # Memory tracking should have captured values
        assert timer.memory_start_mb is not None
        assert timer.memory_end_mb is not None
        assert timer.memory_start_mb >= 0
        assert timer.memory_end_mb >= 0
    
    def test_context_manager(self):
        """Test using timer as context manager."""
        timer = InferenceTimer()
        
        with timer.measure("test_operation"):
            timer.record_first_token()
            timer.record_token()
        
        # Timer should have recorded timings
        assert timer.start_time is not None
        assert timer.end_time is not None
    
    @patch("src.inference.metrics.trace")
    def test_tracer_integration(self, mock_trace):
        """Test OpenTelemetry tracer integration."""
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        
        timer = InferenceTimer(tracer=mock_tracer)
        
        timer.start("test_span")
        mock_tracer.start_span.assert_called_once_with("test_span")
        
        timer.record_first_token()
        mock_span.set_attribute.assert_called()
        
        timer.stop()
        mock_span.end.assert_called_once()


class TestInferenceEngine:
    """Tests for InferenceEngine."""
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        engine = InferenceEngine(
            model_name="test-model",
            device="cpu",
            load_in_8bit=False,
        )
        
        assert engine.model_name == "test-model"
        assert engine.device == "cpu"
        assert engine.load_in_8bit is False
        assert engine.model is None
        assert engine.tokenizer is None
        assert engine.is_loaded() is False
    
    def test_device_fallback_to_cpu(self):
        """Test device fallback when CUDA unavailable."""
        with patch("torch.cuda.is_available", return_value=False):
            engine = InferenceEngine(
                model_name="test-model",
                device="cuda",
            )
            
            # Should fallback to CPU
            assert engine.device == "cpu"
    
    def test_quantization_disabled_on_cpu(self):
        """Test that quantization is disabled on CPU."""
        engine = InferenceEngine(
            model_name="test-model",
            device="cpu",
            load_in_8bit=True,
        )
        
        # Quantization should be disabled on CPU
        assert engine.load_in_8bit is False
    
    def test_is_loaded_before_loading(self):
        """Test is_loaded returns False before loading."""
        engine = InferenceEngine(
            model_name="test-model",
            device="cpu",
        )
        
        assert engine.is_loaded() is False
    
    def test_generate_without_loading_raises_error(self):
        """Test that generate raises error when model not loaded."""
        engine = InferenceEngine(
            model_name="test-model",
            device="cpu",
        )
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            engine.generate("Test prompt")
    
    def test_get_model_info(self):
        """Test getting model info."""
        engine = InferenceEngine(
            model_name="test-model",
            device="cpu",
            load_in_8bit=False,
            max_length=1024,
        )
        
        info = engine.get_model_info()
        
        assert info["model_name"] == "test-model"
        assert info["device"] == "cpu"
        assert info["loaded"] is False
        assert info["quantization"] is False
        assert info["max_length"] == 1024
    
    @patch("src.inference.engine.AutoTokenizer")
    @patch("src.inference.engine.AutoModelForCausalLM")
    def test_load_model_success(self, mock_model_class, mock_tokenizer_class):
        """Test successful model loading."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.vocab_size = 32000
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock model
        mock_model = Mock()
        mock_model.config.model_type = "test"
        mock_model.config.hidden_size = 768
        mock_model.config.num_hidden_layers = 12
        mock_model.config.num_attention_heads = 12
        mock_model.config.vocab_size = 32000
        mock_model.parameters.return_value = [torch.zeros(100) for _ in range(10)]
        mock_model_class.from_pretrained.return_value = mock_model
        
        engine = InferenceEngine(
            model_name="test-model",
            device="cpu",
        )
        
        engine.load_model()
        
        assert engine.is_loaded() is True
        assert engine.tokenizer is not None
        assert engine.model is not None
        
        # Verify tokenizer was configured
        assert mock_tokenizer.pad_token == mock_tokenizer.eos_token
    
    @patch("src.inference.engine.AutoTokenizer")
    def test_load_model_failure(self, mock_tokenizer_class):
        """Test model loading failure."""
        mock_tokenizer_class.from_pretrained.side_effect = Exception("Download failed")
        
        engine = InferenceEngine(
            model_name="test-model",
            device="cpu",
        )
        
        with pytest.raises(RuntimeError, match="Model loading failed"):
            engine.load_model()
        
        assert engine.is_loaded() is False
    
    def test_unload_model(self):
        """Test model unloading."""
        engine = InferenceEngine(
            model_name="test-model",
            device="cpu",
        )
        
        # Set mock model and tokenizer
        engine.model = Mock()
        engine.tokenizer = Mock()
        engine._model_loaded = True
        
        engine.unload_model()
        
        assert engine.model is None
        assert engine.tokenizer is None
        assert engine.is_loaded() is False


class TestIntegrationInference:
    """Integration tests for inference (optional, slow)."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_actual_model_loading(self):
        """
        Test loading an actual small model.
        
        Note: This test downloads a real model (slow).
        Run with: pytest -m integration
        """
        pytest.skip("Requires actual model download")
        
        engine = InferenceEngine(
            model_name="gpt2",  # Small test model
            device="cpu",
            load_in_8bit=False,
        )
        
        engine.load_model()
        assert engine.is_loaded() is True
        
        # Test generation
        result = engine.generate(
            prompt="Hello",
            max_new_tokens=10,
            do_sample=False,
        )
        
        assert "generated_text" in result
        assert result["prompt_tokens"] > 0
        assert result["completion_tokens"] > 0
        
        engine.unload_model()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
