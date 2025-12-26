"""
Request Batching System

Dynamic request batching for improved LLM inference throughput.
Groups incoming requests to leverage CPU SIMD/threading efficiently.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from ..api.models import GenerateRequest, GenerateResponse


logger = logging.getLogger(__name__)


@dataclass
class QueuedRequest:
    """A request waiting in the batch queue."""
    request_id: str
    request: GenerateRequest
    future: asyncio.Future
    queued_at: datetime
    timeout_at: datetime
    
    def is_expired(self) -> bool:
        """Check if request has timed out."""
        return datetime.now() > self.timeout_at


@dataclass
class Batch:
    """A batch of requests to process together."""
    requests: List[QueuedRequest] = field(default_factory=list)
    formed_at: datetime = field(default_factory=datetime.now)
    
    def add_request(self, request: QueuedRequest) -> None:
        """Add a request to the batch."""
        self.requests.append(request)
    
    def is_empty(self) -> bool:
        """Check if batch has no requests."""
        return len(self.requests) == 0
    
    def get_prompts(self) -> List[str]:
        """Extract prompts from all requests in batch."""
        return [req.request.prompt for req in self.requests]
    
    def size(self) -> int:
        """Get number of requests in batch."""
        return len(self.requests)


class RequestQueue:
    """
    Thread-safe request queue for batching.
    
    Manages incoming requests and provides batch formation interface.
    """
    
    def __init__(self, max_size: int = 1000, default_timeout: float = 30.0):
        """
        Initialize request queue.
        
        Args:
            max_size: Maximum number of requests to queue
            default_timeout: Default timeout for requests (seconds)
        """
        self.queue = asyncio.Queue(maxsize=max_size)
        self.pending_requests: Dict[str, QueuedRequest] = {}
        self.default_timeout = default_timeout
        self._lock = asyncio.Lock()
        
        # Metrics
        self.total_enqueued = 0
        self.total_dequeued = 0
        self.total_timeouts = 0
        
        logger.info(f"RequestQueue initialized: max_size={max_size}, timeout={default_timeout}s")
    
    async def enqueue(self, request: GenerateRequest, timeout: Optional[float] = None) -> str:
        """
        Add request to queue and return request ID.
        
        Args:
            request: Generation request to queue
            timeout: Custom timeout for this request
            
        Returns:
            Request ID for tracking
            
        Raises:
            asyncio.QueueFull: If queue is full
        """
        request_id = str(uuid.uuid4())
        timeout = timeout or self.default_timeout
        
        # Create queued request
        queued_request = QueuedRequest(
            request_id=request_id,
            request=request,
            future=asyncio.get_event_loop().create_future(),
            queued_at=datetime.now(),
            timeout_at=datetime.now() + timedelta(seconds=timeout)
        )
        
        # Add to queue and tracking
        await self.queue.put(queued_request)
        
        async with self._lock:
            self.pending_requests[request_id] = queued_request
            self.total_enqueued += 1
        
        logger.debug(f"Request {request_id} enqueued (queue size: {self.queue.qsize()})")
        return request_id
    
    async def get_batch(self, max_batch_size: int, max_wait_time: float) -> List[QueuedRequest]:
        """
        Get batch of requests respecting size and time limits.
        
        Args:
            max_batch_size: Maximum number of requests in batch
            max_wait_time: Maximum time to wait for batch formation (seconds)
            
        Returns:
            List of queued requests forming a batch
        """
        batch_requests = []
        start_time = time.time()
        
        # Collect requests until batch is full or time limit reached
        while (len(batch_requests) < max_batch_size and 
               time.time() - start_time < max_wait_time):
            
            try:
                # Wait for request with timeout
                timeout = max_wait_time - (time.time() - start_time)
                if timeout <= 0:
                    break
                
                request = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                
                # Check if request expired
                if request.is_expired():
                    logger.warning(f"Request {request.request_id} timed out in queue")
                    request.future.set_exception(asyncio.TimeoutError("Request timed out in queue"))
                    self.total_timeouts += 1
                    continue
                
                batch_requests.append(request)
                logger.debug(f"Added request {request.request_id} to batch (size: {len(batch_requests)})")
                
            except asyncio.TimeoutError:
                # No more requests available within time limit
                break
        
        # Update metrics
        self.total_dequeued += len(batch_requests)
        
        if batch_requests:
            logger.info(f"Formed batch of {len(batch_requests)} requests in {time.time() - start_time:.3f}s")
        
        return batch_requests
    
    async def get_pending_request(self, request_id: str) -> Optional[QueuedRequest]:
        """Get pending request by ID."""
        async with self._lock:
            return self.pending_requests.get(request_id)
    
    async def remove_pending(self, request_id: str) -> None:
        """Remove request from pending tracking."""
        async with self._lock:
            self.pending_requests.pop(request_id, None)
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self.queue.qsize()
    
    def get_pending_count(self) -> int:
        """Get number of pending requests being processed."""
        return len(self.pending_requests)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get queue metrics."""
        return {
            "queue_size": self.get_queue_size(),
            "pending_count": self.get_pending_count(),
            "total_enqueued": self.total_enqueued,
            "total_dequeued": self.total_dequeued,
            "total_timeouts": self.total_timeouts,
            "timeout_rate": self.total_timeouts / max(self.total_enqueued, 1)
        }


class BatchScheduler:
    """
    Background task that forms batches from queued requests.
    
    Continuously processes requests from the queue and forms optimal batches
    for the inference engine.
    """
    
    def __init__(
        self, 
        queue: RequestQueue, 
        engine,
        max_batch_size: int = 4,
        max_wait_time: float = 0.05,
        processing_interval: float = 0.01
    ):
        """
        Initialize batch scheduler.
        
        Args:
            queue: Request queue to pull from
            engine: Inference engine for processing batches
            max_batch_size: Maximum requests per batch
            max_wait_time: Maximum wait time for batch formation (seconds)
            processing_interval: Interval between batch checks (seconds)
        """
        self.queue = queue
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.processing_interval = processing_interval
        
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
        # Metrics
        self.total_batches_processed = 0
        self.total_requests_processed = 0
        self.total_processing_time = 0.0
        self.batch_sizes = []
        
        logger.info(
            f"BatchScheduler initialized: max_batch_size={max_batch_size}, "
            f"max_wait_time={max_wait_time}s"
        )
    
    async def start(self) -> None:
        """Start background batch processing loop."""
        if self._running:
            logger.warning("BatchScheduler already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._processing_loop())
        logger.info("BatchScheduler started")
    
    async def stop(self) -> None:
        """Stop background batch processing loop."""
        if not self._running:
            return
        
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info("BatchScheduler stopped")
    
    async def _processing_loop(self) -> None:
        """Main processing loop for batch formation and execution."""
        logger.info("BatchScheduler processing loop started")
        
        while self._running:
            try:
                # Get batch from queue
                batch_requests = await self.queue.get_batch(
                    max_batch_size=self.max_batch_size,
                    max_wait_time=self.max_wait_time
                )
                
                if batch_requests:
                    await self._process_batch(batch_requests)
                else:
                    # No requests, brief sleep before next check
                    await asyncio.sleep(self.processing_interval)
                    
            except asyncio.CancelledError:
                logger.info("BatchScheduler processing loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}", exc_info=True)
                await asyncio.sleep(self.processing_interval)
        
        logger.info("BatchScheduler processing loop ended")
    
    async def _process_batch(self, batch_requests: List[QueuedRequest]) -> None:
        """
        Process batch of requests and distribute results.
        
        Args:
            batch_requests: List of queued requests to process together
        """
        if not batch_requests:
            return
        
        batch = Batch(requests=batch_requests)
        start_time = time.time()
        
        try:
            logger.info(f"Processing batch of {batch.size()} requests")
            
            # Extract prompts for batch processing
            prompts = batch.get_prompts()
            
            # Get generation parameters from first request (all should be similar)
            first_request = batch_requests[0].request
            max_tokens = first_request.max_tokens
            temperature = first_request.temperature
            top_p = first_request.top_p
            do_sample = first_request.do_sample
            
            # Process batch through engine
            if batch.size() == 1:
                # Single request - use regular generate for compatibility
                result = self.engine.generate(
                    prompt=prompts[0],
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                )
                batch_results = [result]
            else:
                # Multiple requests - use batch generation
                batch_results = self.engine.generate_batch(
                    prompts=prompts,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                )
            
            # Distribute results to requesters
            for i, queued_request in enumerate(batch_requests):
                try:
                    # Remove from pending tracking
                    await self.queue.remove_pending(queued_request.request_id)
                    
                    # Set result for future
                    response = GenerateResponse(**batch_results[i])
                    queued_request.future.set_result(response)
                    
                except Exception as e:
                    logger.error(f"Error distributing result for request {queued_request.request_id}: {e}")
                    queued_request.future.set_exception(e)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.total_batches_processed += 1
            self.total_requests_processed += batch.size()
            self.total_processing_time += processing_time
            self.batch_sizes.append(batch.size())
            
            logger.info(
                f"Batch processed: {batch.size()} requests in {processing_time:.3f}s "
                f"(avg: {processing_time/batch.size():.3f}s per request)"
            )
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}", exc_info=True)
            
            # Fail all requests in batch
            for queued_request in batch_requests:
                await self.queue.remove_pending(queued_request.request_id)
                queued_request.future.set_exception(e)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get scheduler metrics."""
        avg_batch_size = sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0
        avg_processing_time = self.total_processing_time / max(self.total_batches_processed, 1)
        
        return {
            "running": self._running,
            "total_batches_processed": self.total_batches_processed,
            "total_requests_processed": self.total_requests_processed,
            "total_processing_time": self.total_processing_time,
            "avg_batch_size": avg_batch_size,
            "avg_processing_time_per_batch": avg_processing_time,
            "avg_processing_time_per_request": avg_processing_time / max(avg_batch_size, 1),
            "max_batch_size": self.max_batch_size,
            "max_wait_time": self.max_wait_time,
            "batch_sizes_distribution": {
                size: self.batch_sizes.count(size) for size in set(self.batch_sizes)
            }
        }
