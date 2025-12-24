"""
API Models

Pydantic models for request/response validation and serialization.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict


class GenerateRequest(BaseModel):
    """Request model for text generation."""
    
    prompt: str = Field(
        ...,
        description="Input prompt for text generation",
        min_length=1,
        max_length=4096,
        examples=["Explain quantum computing in simple terms"],
    )
    
    max_tokens: int = Field(
        default=256,
        description="Maximum number of tokens to generate",
        ge=1,
        le=2048,
    )
    
    temperature: float = Field(
        default=0.7,
        description="Sampling temperature (0.0 = deterministic, 1.0 = more random)",
        ge=0.0,
        le=2.0,
    )
    
    top_p: float = Field(
        default=0.9,
        description="Nucleus sampling probability",
        ge=0.0,
        le=1.0,
    )
    
    do_sample: bool = Field(
        default=True,
        description="Enable sampling (if False, uses greedy decoding)",
    )
    
    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Validate and clean prompt."""
        v = v.strip()
        if not v:
            raise ValueError("Prompt cannot be empty or whitespace only")
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "Write a haiku about machine learning",
                    "max_tokens": 128,
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "do_sample": True,
                }
            ]
        }
    }


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    
    generated_text: str = Field(
        ...,
        description="Generated text output",
    )
    
    prompt_tokens: int = Field(
        ...,
        description="Number of tokens in the input prompt",
        ge=0,
    )
    
    completion_tokens: int = Field(
        ...,
        description="Number of tokens in the generated completion",
        ge=0,
    )
    
    total_tokens: int = Field(
        ...,
        description="Total tokens (prompt + completion)",
        ge=0,
    )
    
    ttft: float = Field(
        ...,
        description="Time to First Token in seconds",
        ge=0.0,
    )
    
    tpot: float = Field(
        ...,
        description="Time Per Output Token in seconds (average)",
        ge=0.0,
    )
    
    total_time: float = Field(
        ...,
        description="Total generation time in seconds",
        ge=0.0,
    )
    
    throughput_tokens_per_sec: float = Field(
        ...,
        description="Throughput in tokens per second",
        ge=0.0,
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "generated_text": "Machine learning models learn patterns from data...",
                    "prompt_tokens": 15,
                    "completion_tokens": 128,
                    "total_tokens": 143,
                    "ttft": 0.456,
                    "tpot": 0.032,
                    "total_time": 4.552,
                    "throughput_tokens_per_sec": 31.4,
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    model_config = ConfigDict(protected_namespaces=())
    
    status: str = Field(
        ...,
        description="Service status",
        examples=["healthy", "degraded", "unhealthy"],
    )
    
    model_loaded: bool = Field(
        ...,
        description="Whether the LLM model is loaded",
    )
    
    model_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Model configuration and metadata",
    )
    
    gpu_available: bool = Field(
        ...,
        description="Whether GPU is available",
    )
    
    active_requests: int = Field(
        default=0,
        description="Number of currently active requests",
        ge=0,
    )


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    error: str = Field(
        ...,
        description="Error type or category",
        examples=["validation_error", "generation_error", "oom_error"],
    )
    
    message: str = Field(
        ...,
        description="Human-readable error message",
    )
    
    detail: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details",
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "error": "generation_error",
                    "message": "Failed to generate text due to model error",
                    "detail": {
                        "prompt_length": 150,
                        "max_tokens": 256,
                    },
                }
            ]
        }
    }


class MetricsSnapshot(BaseModel):
    """Snapshot of current metrics."""
    
    model_config = ConfigDict(protected_namespaces=())
    
    total_requests: int = Field(
        ...,
        description="Total number of requests processed",
        ge=0,
    )
    
    active_requests: int = Field(
        ...,
        description="Number of currently active requests",
        ge=0,
    )
    
    total_errors: int = Field(
        ...,
        description="Total number of errors",
        ge=0,
    )
    
    model_loaded: bool = Field(
        ...,
        description="Whether model is loaded",
    )
    
    memory_usage_mb: float = Field(
        ...,
        description="Current memory usage in MB",
        ge=0.0,
    )
    
    gpu_memory_usage_mb: Optional[float] = Field(
        default=None,
        description="Current GPU memory usage in MB",
        ge=0.0,
    )
