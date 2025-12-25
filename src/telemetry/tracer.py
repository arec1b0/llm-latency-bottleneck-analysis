"""
OpenTelemetry Tracer Configuration

Sets up distributed tracing with Jaeger backend.
Provides tracer instances for instrumenting code.
"""

import logging
import os
from typing import Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.logging import LoggingInstrumentor


logger = logging.getLogger(__name__)


def setup_telemetry(
    service_name: Optional[str] = None,
    otlp_endpoint: Optional[str] = None,
) -> TracerProvider:
    """
    Initialize OpenTelemetry tracing with OTLP exporter.
    
    Args:
        service_name: Name of the service for tracing (default: from env)
        otlp_endpoint: OTLP collector endpoint (default: from env)
    
    Returns:
        TracerProvider: Configured tracer provider
    
    Raises:
        ValueError: If required configuration is missing
    """
    # Get configuration from environment or parameters
    service_name = service_name or os.getenv("OTEL_SERVICE_NAME", "llm-inference-api")
    otlp_endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    
    logger.info(
        f"Initializing OpenTelemetry for service '{service_name}' "
        f"with OTLP at {otlp_endpoint}"
    )
    
    # Create resource with service information
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": os.getenv("SERVICE_VERSION", "0.1.0"),
            "deployment.environment": os.getenv("ENVIRONMENT", "development"),
        }
    )
    
    # Create tracer provider
    tracer_provider = TracerProvider(resource=resource)
    
    # Create OTLP exporter
    try:
        otlp_exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            insecure=True,  # For local development
        )
        
        # Create batch span processor for efficient export
        span_processor = BatchSpanProcessor(
            otlp_exporter,
            max_queue_size=2048,
            max_export_batch_size=512,
            schedule_delay_millis=5000,
            export_timeout_millis=30000,
        )
        
        tracer_provider.add_span_processor(span_processor)
        logger.info("OTLP exporter configured successfully")
        
    except Exception as e:
        logger.error(f"Failed to configure OTLP exporter: {e}")
        logger.warning("Continuing without OTLP tracing")
        # Don't raise - allow system to continue without tracing
    
    # Set as global tracer provider
    trace.set_tracer_provider(tracer_provider)
    
    # Instrument logging to include trace context
    try:
        LoggingInstrumentor().instrument(set_logging_format=True)
    except Exception as e:
        logger.warning(f"Failed to instrument logging: {e}")
    
    logger.info("OpenTelemetry instrumentation initialized")
    return tracer_provider


def get_tracer(name: str = __name__) -> trace.Tracer:
    """
    Get a tracer instance for instrumenting code.
    
    Args:
        name: Name of the tracer (typically __name__ of the module)
    
    Returns:
        Tracer: OpenTelemetry tracer instance
    """
    return trace.get_tracer(name)


def shutdown_telemetry() -> None:
    """
    Gracefully shutdown telemetry and flush remaining spans.
    Should be called on application shutdown.
    """
    try:
        tracer_provider = trace.get_tracer_provider()
        if hasattr(tracer_provider, "shutdown"):
            tracer_provider.shutdown()
            logger.info("Telemetry shutdown completed")
    except Exception as e:
        logger.error(f"Error during telemetry shutdown: {e}")
