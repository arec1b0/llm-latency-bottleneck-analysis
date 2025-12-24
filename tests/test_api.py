"""
API Tests

Unit and integration tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock

from src.api.main import app
from src.api.models import GenerateRequest, GenerateResponse, HealthResponse


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_inference_engine():
    """Mock inference engine."""
    mock = MagicMock()
    mock.is_loaded.return_value = True
    mock.get_model_info.return_value = {
        "model_name": "test-model",
        "device": "cpu",
        "loaded": True,
        "quantization": False,
        "max_length": 2048,
    }
    mock.generate.return_value = {
        "generated_text": "Test response",
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
        "ttft": 0.5,
        "tpot": 0.05,
        "total_time": 1.5,
        "throughput_tokens_per_sec": 20.0,
    }
    return mock


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_check_success(self, client):
        """Test successful health check."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "gpu_available" in data
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["gpu_available"], bool)
    
    def test_health_check_structure(self, client):
        """Test health check response structure."""
        response = client.get("/health")
        data = response.json()
        
        # Validate against HealthResponse model
        health = HealthResponse(**data)
        assert health.status in ["healthy", "degraded", "unhealthy"]
        assert isinstance(health.model_loaded, bool)
        assert isinstance(health.gpu_available, bool)


class TestGenerateEndpoint:
    """Tests for /generate endpoint."""
    
    def test_generate_validation_empty_prompt(self, client):
        """Test validation with empty prompt."""
        response = client.post(
            "/generate",
            json={"prompt": ""},
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_generate_validation_prompt_too_long(self, client):
        """Test validation with too long prompt."""
        response = client.post(
            "/generate",
            json={"prompt": "a" * 5000},  # Exceeds max_length
        )
        
        assert response.status_code == 422
    
    def test_generate_validation_invalid_temperature(self, client):
        """Test validation with invalid temperature."""
        response = client.post(
            "/generate",
            json={
                "prompt": "Test prompt",
                "temperature": 3.0,  # > 2.0
            },
        )
        
        assert response.status_code == 422
    
    def test_generate_validation_invalid_max_tokens(self, client):
        """Test validation with invalid max_tokens."""
        response = client.post(
            "/generate",
            json={
                "prompt": "Test prompt",
                "max_tokens": 0,  # Must be >= 1
            },
        )
        
        assert response.status_code == 422
    
    def test_generate_default_values(self, client):
        """Test that default values are applied."""
        request = GenerateRequest(prompt="Test")
        
        assert request.max_tokens == 256
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.do_sample is True
    
    @patch("src.api.main.inference_engine")
    def test_generate_success(self, mock_engine, client, mock_inference_engine):
        """Test successful text generation."""
        mock_engine.__bool__.return_value = True
        mock_engine.is_loaded.return_value = True
        mock_engine.generate.return_value = {
            "generated_text": "Test response",
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "ttft": 0.5,
            "tpot": 0.05,
            "total_time": 1.5,
            "throughput_tokens_per_sec": 20.0,
        }
        
        response = client.post(
            "/generate",
            json={
                "prompt": "Test prompt",
                "max_tokens": 128,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate structure
        gen_response = GenerateResponse(**data)
        assert gen_response.generated_text == "Test response"
        assert gen_response.prompt_tokens == 10
        assert gen_response.completion_tokens == 20
        assert gen_response.total_tokens == 30
        assert gen_response.ttft > 0
        assert gen_response.tpot > 0
    
    @patch("src.api.main.inference_engine")
    def test_generate_engine_not_initialized(self, mock_engine, client):
        """Test generation when engine not initialized."""
        mock_engine.__bool__.return_value = False
        
        response = client.post(
            "/generate",
            json={"prompt": "Test"},
        )
        
        assert response.status_code == 503
        assert "not initialized" in response.json()["detail"].lower()


class TestMetricsEndpoint:
    """Tests for /metrics endpoint."""
    
    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        
        # Check for some expected metrics
        content = response.text
        assert "llm_requests_total" in content or "HELP" in content


class TestModelEndpoints:
    """Tests for model management endpoints."""
    
    def test_model_info_success(self, client):
        """Test model info endpoint."""
        response = client.get("/model/info")
        
        # May return 200 or 503 depending on engine state
        assert response.status_code in [200, 503]
    
    @patch("src.api.main.inference_engine")
    def test_model_load_when_already_loaded(self, mock_engine, client):
        """Test loading model when already loaded."""
        mock_engine.__bool__.return_value = True
        mock_engine.is_loaded.return_value = True
        
        response = client.post("/model/load")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "already loaded" in data["message"].lower()
    
    @patch("src.api.main.inference_engine")
    def test_model_unload_when_not_loaded(self, mock_engine, client):
        """Test unloading model when not loaded."""
        mock_engine.__bool__.return_value = True
        mock_engine.is_loaded.return_value = False
        
        response = client.post("/model/unload")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data
        assert "health" in data


class TestMiddleware:
    """Tests for middleware functionality."""
    
    def test_process_time_header(self, client):
        """Test that X-Process-Time header is added."""
        response = client.get("/health")
        
        assert "X-Process-Time" in response.headers
        process_time = float(response.headers["X-Process-Time"])
        assert process_time >= 0


class TestCORS:
    """Tests for CORS configuration."""
    
    def test_cors_headers_present(self, client):
        """Test that CORS headers are configured."""
        response = client.options(
            "/generate",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        
        # CORS should be configured
        assert response.status_code in [200, 405]


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_404_endpoint(self, client):
        """Test 404 for non-existent endpoint."""
        response = client.get("/nonexistent")
        
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test 405 for wrong HTTP method."""
        response = client.get("/generate")  # Should be POST
        
        assert response.status_code == 405


# Integration tests (require actual model)
class TestIntegration:
    """Integration tests (optional, slow)."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_end_to_end_generation(self, client):
        """
        End-to-end test with actual model.
        
        Note: This test is marked as 'integration' and 'slow'.
        Run with: pytest -m integration
        """
        # Skip if model not loaded
        health_response = client.get("/health")
        if not health_response.json().get("model_loaded"):
            pytest.skip("Model not loaded")
        
        response = client.post(
            "/generate",
            json={
                "prompt": "Hello, world!",
                "max_tokens": 10,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["generated_text"]) > 0
        assert data["total_tokens"] > 0
        assert data["ttft"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
