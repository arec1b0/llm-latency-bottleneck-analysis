"""
Telemetry Module

OpenTelemetry instrumentation for distributed tracing and metrics collection.
"""

from .tracer import setup_telemetry, get_tracer, shutdown_telemetry
from .metrics_collector import MetricsCollector

__all__ = ["setup_telemetry", "get_tracer", "shutdown_telemetry", "MetricsCollector"]
