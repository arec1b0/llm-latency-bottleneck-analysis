"""
Metrics Collector

Collects and exposes Prometheus metrics for LLM inference performance.
Tracks TTFT, TPOT, memory usage, and throughput.
"""

import logging
import time
from typing import Dict, Optional
from dataclasses import dataclass, field
from threading import Lock

import psutil
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)


logger = logging.getLogger(__name__)


@dataclass
class InferenceMetrics:
    """Container for inference performance metrics."""
    
    ttft: float  # Time to First Token (seconds)
    tpot: float  # Time Per Output Token (seconds)
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    total_time: float
    memory_used_mb: float
    gpu_memory_used_mb: Optional[float] = None


class MetricsCollector:
    """
    Singleton metrics collector for Prometheus.
    
    Tracks:
    - Request counts and errors
    - Latency distributions (TTFT, TPOT, total)
    - Token counts
    - Memory utilization
    - Throughput
    """
    
    _instance: Optional["MetricsCollector"] = None
    _lock: Lock = Lock()
    
    def __new__(cls):
        """Ensure singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize Prometheus metrics."""
        if self._initialized:
            return
        
        # Request counters
        self.requests_total = Counter(
            "llm_requests_total",
            "Total number of inference requests",
            ["status"]
        )
        
        self.errors_total = Counter(
            "llm_errors_total",
            "Total number of errors",
            ["error_type"]
        )
        
        # Latency histograms
        self.ttft_histogram = Histogram(
            "llm_ttft_seconds",
            "Time to First Token distribution",
            buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
        )
        
        self.tpot_histogram = Histogram(
            "llm_tpot_seconds",
            "Time Per Output Token distribution",
            buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
        )
        
        self.total_time_histogram = Histogram(
            "llm_inference_duration_seconds",
            "Total inference duration",
            buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0, 60.0]
        )
        
        # Token counters
        self.tokens_processed = Counter(
            "llm_tokens_processed_total",
            "Total tokens processed",
            ["token_type"]
        )
        
        # Current state gauges
        self.active_requests = Gauge(
            "llm_active_requests",
            "Number of currently active requests"
        )
        
        self.memory_usage_mb = Gauge(
            "llm_memory_usage_megabytes",
            "Current memory usage in MB",
            ["memory_type"]
        )
        
        self.throughput_tokens_per_sec = Gauge(
            "llm_throughput_tokens_per_second",
            "Current throughput in tokens/second"
        )
        
        # Model loading gauge
        self.model_loaded = Gauge(
            "llm_model_loaded",
            "Whether model is loaded (1) or not (0)"
        )
        
        # Queue depth gauge
        self.queue_depth = Gauge(
            "llm_queue_depth",
            "Number of requests waiting in queue"
        )
        
        self._initialized = True
        logger.info("MetricsCollector initialized")
    
    def record_request(self, status: str = "success") -> None:
        """
        Record a completed request.
        
        Args:
            status: Request status (success, error, timeout)
        """
        self.requests_total.labels(status=status).inc()
    
    def record_error(self, error_type: str) -> None:
        """
        Record an error occurrence.
        
        Args:
            error_type: Type of error (oom, timeout, model_error, etc.)
        """
        self.errors_total.labels(error_type=error_type).inc()
        self.record_request(status="error")
    
    def record_inference_metrics(self, metrics: InferenceMetrics) -> None:
        """
        Record comprehensive inference metrics.
        
        Args:
            metrics: InferenceMetrics dataclass with performance data
        """
        # Record latencies
        self.ttft_histogram.observe(metrics.ttft)
        self.tpot_histogram.observe(metrics.tpot)
        self.total_time_histogram.observe(metrics.total_time)
        
        # Record token counts
        self.tokens_processed.labels(token_type="prompt").inc(metrics.prompt_tokens)
        self.tokens_processed.labels(token_type="completion").inc(metrics.completion_tokens)
        
        # Update memory gauge
        self.memory_usage_mb.labels(memory_type="ram").set(metrics.memory_used_mb)
        if metrics.gpu_memory_used_mb is not None:
            self.memory_usage_mb.labels(memory_type="vram").set(metrics.gpu_memory_used_mb)
        
        # Calculate throughput
        if metrics.total_time > 0:
            throughput = metrics.total_tokens / metrics.total_time
            self.throughput_tokens_per_sec.set(throughput)
        
        self.record_request(status="success")
        
        logger.debug(
            f"Metrics recorded - TTFT: {metrics.ttft:.3f}s, "
            f"TPOT: {metrics.tpot:.3f}s, "
            f"Tokens: {metrics.total_tokens}"
        )
    
    def increment_active_requests(self) -> None:
        """Increment active request counter."""
        self.active_requests.inc()
    
    def decrement_active_requests(self) -> None:
        """Decrement active request counter."""
        self.active_requests.dec()
    
    def set_model_loaded(self, loaded: bool) -> None:
        """
        Set model loaded status.
        
        Args:
            loaded: True if model is loaded, False otherwise
        """
        self.model_loaded.set(1 if loaded else 0)
    
    def update_queue_depth(self, depth: int) -> None:
        """
        Update queue depth metric.
        
        Args:
            depth: Number of requests waiting in queue
        """
        self.queue_depth.set(depth)
    
    def update_system_metrics(self) -> None:
        """Update system-level metrics (CPU, memory)."""
        try:
            # System RAM
            memory = psutil.virtual_memory()
            self.memory_usage_mb.labels(memory_type="system_ram").set(
                memory.used / (1024 * 1024)
            )
            
            # Process memory
            process = psutil.Process()
            process_memory = process.memory_info().rss / (1024 * 1024)
            self.memory_usage_mb.labels(memory_type="process_ram").set(process_memory)
            
        except Exception as e:
            logger.warning(f"Failed to update system metrics: {e}")
    
    @staticmethod
    def get_metrics_response() -> tuple:
        """
        Generate Prometheus metrics response.
        
        Returns:
            Tuple of (metrics_text, content_type)
        """
        return generate_latest(), CONTENT_TYPE_LATEST


# Global metrics collector instance
metrics_collector = MetricsCollector()
