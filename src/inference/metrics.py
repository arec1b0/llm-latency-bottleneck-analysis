"""
Inference Metrics

Performance timing utilities for tracking TTFT, TPOT, and memory usage.
"""

import logging
import time
from typing import Optional, List
from dataclasses import dataclass
from contextlib import contextmanager

import psutil
from opentelemetry import trace


logger = logging.getLogger(__name__)


@dataclass
class TokenTimings:
    """Container for token-level timing information."""
    
    first_token_time: float  # Time to first token (seconds)
    subsequent_token_times: List[float]  # Time for each subsequent token
    total_time: float  # Total generation time
    
    @property
    def average_tpot(self) -> float:
        """Calculate average time per output token (excluding first)."""
        if not self.subsequent_token_times:
            return 0.0
        return sum(self.subsequent_token_times) / len(self.subsequent_token_times)
    
    @property
    def total_tokens(self) -> int:
        """Total number of tokens generated."""
        return 1 + len(self.subsequent_token_times)


class InferenceTimer:
    """
    High-precision timer for tracking inference performance.
    
    Measures:
    - Time to First Token (TTFT)
    - Time Per Output Token (TPOT)
    - Total inference time
    - Memory usage before/after
    """
    
    def __init__(self, tracer: Optional[trace.Tracer] = None):
        """
        Initialize inference timer.
        
        Args:
            tracer: OpenTelemetry tracer for distributed tracing
        """
        self.tracer = tracer
        self.start_time: Optional[float] = None
        self.first_token_time: Optional[float] = None
        self.token_times: List[float] = []
        self.end_time: Optional[float] = None
        
        self.memory_start_mb: Optional[float] = None
        self.memory_end_mb: Optional[float] = None
        self.gpu_memory_start_mb: Optional[float] = None
        self.gpu_memory_end_mb: Optional[float] = None
        
        self._span: Optional[trace.Span] = None
    
    def start(self, span_name: str = "inference") -> None:
        """
        Start timing inference.
        
        Args:
            span_name: Name for the OpenTelemetry span
        """
        self.start_time = time.perf_counter()
        self.token_times = []
        
        # Capture initial memory
        self.memory_start_mb = self._get_process_memory_mb()
        self.gpu_memory_start_mb = self._get_gpu_memory_mb()
        
        # Start tracing span
        if self.tracer:
            self._span = self.tracer.start_span(span_name)
            self._span.set_attribute("inference.start_time", self.start_time)
        
        logger.debug(f"Inference timer started: {span_name}")
    
    def record_first_token(self) -> float:
        """
        Record time when first token is generated.
        
        Returns:
            Time to first token in seconds
        """
        if self.start_time is None:
            raise RuntimeError("Timer not started. Call start() first.")
        
        self.first_token_time = time.perf_counter()
        ttft = self.first_token_time - self.start_time
        
        if self._span:
            self._span.set_attribute("inference.ttft", ttft)
            self._span.add_event("first_token_generated", {"ttft": ttft})
        
        logger.debug(f"First token generated in {ttft:.3f}s")
        return ttft
    
    def record_token(self) -> float:
        """
        Record time for each subsequent token.
        
        Returns:
            Time since last token in seconds
        """
        if self.start_time is None:
            raise RuntimeError("Timer not started. Call start() first.")
        
        current_time = time.perf_counter()
        
        # Calculate time since last token
        if self.first_token_time is None:
            # This is the first token
            token_time = current_time - self.start_time
            self.first_token_time = current_time
        else:
            # Subsequent token
            last_time = self.token_times[-1] if self.token_times else self.first_token_time
            token_time = current_time - last_time
            self.token_times.append(current_time)
        
        return token_time
    
    def stop(self) -> TokenTimings:
        """
        Stop timing and calculate final metrics.
        
        Returns:
            TokenTimings with complete performance data
        """
        if self.start_time is None:
            raise RuntimeError("Timer not started. Call start() first.")
        
        self.end_time = time.perf_counter()
        total_time = self.end_time - self.start_time
        
        # Capture final memory
        self.memory_end_mb = self._get_process_memory_mb()
        self.gpu_memory_end_mb = self._get_gpu_memory_mb()
        
        # Calculate token-level timings
        if self.first_token_time is None:
            ttft = 0.0
            tpot_list = []
        else:
            ttft = self.first_token_time - self.start_time
            
            # Calculate time per token for subsequent tokens
            tpot_list = []
            times = [self.first_token_time] + self.token_times
            for i in range(1, len(times)):
                tpot_list.append(times[i] - times[i-1])
        
        timings = TokenTimings(
            first_token_time=ttft,
            subsequent_token_times=tpot_list,
            total_time=total_time
        )
        
        # Update span with final metrics
        if self._span:
            self._span.set_attribute("inference.total_time", total_time)
            self._span.set_attribute("inference.total_tokens", timings.total_tokens)
            self._span.set_attribute("inference.average_tpot", timings.average_tpot)
            
            if self.memory_start_mb and self.memory_end_mb:
                memory_delta = self.memory_end_mb - self.memory_start_mb
                self._span.set_attribute("inference.memory_delta_mb", memory_delta)
            
            self._span.end()
        
        logger.info(
            f"Inference completed - TTFT: {ttft:.3f}s, "
            f"Avg TPOT: {timings.average_tpot:.3f}s, "
            f"Total: {total_time:.3f}s, "
            f"Tokens: {timings.total_tokens}"
        )
        
        return timings
    
    def get_memory_delta_mb(self) -> Optional[float]:
        """
        Get memory usage delta.
        
        Returns:
            Memory delta in MB, or None if not available
        """
        if self.memory_start_mb and self.memory_end_mb:
            return self.memory_end_mb - self.memory_start_mb
        return None
    
    def get_gpu_memory_delta_mb(self) -> Optional[float]:
        """
        Get GPU memory usage delta.
        
        Returns:
            GPU memory delta in MB, or None if not available
        """
        if self.gpu_memory_start_mb and self.gpu_memory_end_mb:
            return self.gpu_memory_end_mb - self.gpu_memory_start_mb
        return None
    
    @staticmethod
    def _get_process_memory_mb() -> float:
        """Get current process memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception as e:
            logger.warning(f"Failed to get process memory: {e}")
            return 0.0
    
    @staticmethod
    def _get_gpu_memory_mb() -> Optional[float]:
        """Get current GPU memory usage in MB."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 * 1024)
        except Exception as e:
            logger.debug(f"Failed to get GPU memory: {e}")
        return None
    
    @contextmanager
    def measure(self, span_name: str = "inference"):
        """
        Context manager for automatic timing.
        
        Args:
            span_name: Name for the tracing span
        
        Yields:
            InferenceTimer instance
        
        Example:
            with timer.measure("generate_text"):
                timer.record_first_token()
                # ... generate tokens ...
                timer.record_token()
        """
        self.start(span_name)
        try:
            yield self
        finally:
            self.stop()
