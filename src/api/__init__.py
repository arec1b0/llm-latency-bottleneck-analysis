"""
API Module

FastAPI application with OpenTelemetry instrumentation.
"""

from .main import app
from .models import GenerateRequest, GenerateResponse, HealthResponse

__all__ = ["app", "GenerateRequest", "GenerateResponse", "HealthResponse"]
