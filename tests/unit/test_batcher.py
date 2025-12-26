"""
Unit tests for request batching system.

Tests RequestQueue and BatchScheduler components for proper
dynamic request batching behavior.
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from typing import List, Dict, Any

from src.api.models import GenerateRequest, GenerateResponse
from src.inference.batcher import RequestQueue, BatchScheduler, QueuedRequest, Batch


@pytest.fixture
async def mock_engine():
    """Create a mock inference engine for testing."""
    engine = AsyncMock()
    
    # Mock single generation - return dict, not coroutine
    def mock_generate(*args, **kwargs):
        return {
            "generated_text": "Test response",
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8,
            "ttft": 0.5,
            "tpot": 0.1,
            "total_time": 0.8,
            "throughput_tokens_per_sec": 10.0,
        }
    
    # Mock batch generation - return list of dicts, not coroutine
    def mock_generate_batch(*args, **kwargs):
        return [
            {
                "generated_text": f"Test response {i}",
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "total_tokens": 8,
                "ttft": 0.5,
                "tpot": 0.1,
                "total_time": 0.8,
                "throughput_tokens_per_sec": 10.0,
            }
            for i in range(4)
        ]
    
    engine.generate = mock_generate
    engine.generate_batch = mock_generate_batch
    
    return engine


@pytest.fixture
async def request_queue():
    """Create a RequestQueue for testing."""
    return RequestQueue(max_size=100, default_timeout=5.0)


@pytest.fixture
def sample_request():
    """Create a sample GenerateRequest for testing."""
    return GenerateRequest(
        prompt="Test prompt",
        max_tokens=50,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )


class TestRequestQueue:
    """Test RequestQueue functionality."""
    
    @pytest.mark.asyncio
    async def test_enqueue_request(self, request_queue, sample_request):
        """Test enqueuing a single request."""
        request_id = await request_queue.enqueue(sample_request)
        
        assert request_id is not None
        assert len(request_id) > 0
        assert request_queue.get_queue_size() == 1
        assert request_queue.get_pending_count() == 1
        assert request_queue.total_enqueued == 1
    
    @pytest.mark.asyncio
    async def test_get_batch_empty(self, request_queue):
        """Test getting batch from empty queue."""
        batch = await request_queue.get_batch(max_batch_size=4, max_wait_time=0.01)
        
        assert batch == []
        assert request_queue.total_dequeued == 0
    
    @pytest.mark.asyncio
    async def test_get_batch_partial(self, request_queue, sample_request):
        """Test getting partial batch."""
        # Enqueue 2 requests
        request_id1 = await request_queue.enqueue(sample_request)
        request_id2 = await request_queue.enqueue(sample_request)
        
        # Get batch with max size 4
        batch = await request_queue.get_batch(max_batch_size=4, max_wait_time=0.01)
        
        assert len(batch) == 2
        assert batch[0].request_id == request_id1
        assert batch[1].request_id == request_id2
        assert request_queue.total_dequeued == 2
    
    @pytest.mark.asyncio
    async def test_get_batch_time_limit(self, request_queue, sample_request):
        """Test batch formation with time limit."""
        # Enqueue 1 request
        request_id = await request_queue.enqueue(sample_request)
        
        # Get batch with short time limit (should return immediately with 1 request)
        start_time = time.time()
        batch = await request_queue.get_batch(max_batch_size=4, max_wait_time=0.1)
        elapsed = time.time() - start_time
        
        assert len(batch) == 1
        assert batch[0].request_id == request_id
        assert elapsed < 0.2  # Should return quickly
    
    @pytest.mark.asyncio
    async def test_get_expired_request(self, request_queue):
        """Test handling of expired requests."""
        # Create request with very short timeout
        request = GenerateRequest(prompt="Test", max_tokens=10)
        request_id = await request_queue.enqueue(request, timeout=0.001)
        
        # Wait for request to expire
        await asyncio.sleep(0.01)
        
        # Try to get batch (should skip expired request)
        batch = await request_queue.get_batch(max_batch_size=4, max_wait_time=0.01)
        
        assert len(batch) == 0
        assert request_queue.total_timeouts == 1
    
    @pytest.mark.asyncio
    async def test_queue_full(self, request_queue):
        """Test behavior when queue is full."""
        # Fill queue to capacity (max_size=100)
        requests = []
        for i in range(100):  # Fill exactly to capacity
            req = GenerateRequest(prompt=f"Test {i}", max_tokens=10)
            request_id = await request_queue.enqueue(req)
            requests.append(request_id)
        
        # Try to enqueue one more (should fail)
        req = GenerateRequest(prompt="Overflow", max_tokens=10)
        with pytest.raises(asyncio.QueueFull):
            await request_queue.enqueue(req)
    
    @pytest.mark.asyncio
    async def test_pending_request_tracking(self, request_queue, sample_request):
        """Test pending request tracking."""
        request_id = await request_queue.enqueue(sample_request)
        
        # Get pending request
        pending = await request_queue.get_pending_request(request_id)
        assert pending is not None
        assert pending.request_id == request_id
        assert pending.request == sample_request
        
        # Remove pending request
        await request_queue.remove_pending(request_id)
        pending = await request_queue.get_pending_request(request_id)
        assert pending is None
    
    def test_metrics(self, request_queue):
        """Test metrics collection."""
        metrics = request_queue.get_metrics()
        
        assert "queue_size" in metrics
        assert "pending_count" in metrics
        assert "total_enqueued" in metrics
        assert "total_dequeued" in metrics
        assert "total_timeouts" in metrics
        assert "timeout_rate" in metrics
        
        assert metrics["queue_size"] == 0
        assert metrics["pending_count"] == 0
        assert metrics["total_enqueued"] == 0
        assert metrics["total_dequeued"] == 0
        assert metrics["total_timeouts"] == 0


class TestBatch:
    """Test Batch functionality."""
    
    def test_empty_batch(self):
        """Test empty batch creation."""
        batch = Batch()
        assert batch.is_empty()
        assert batch.size() == 0
        assert batch.get_prompts() == []
    
    def test_add_request(self, sample_request):
        """Test adding request to batch."""
        batch = Batch()
        queued_request = QueuedRequest(
            request_id="test-id",
            request=sample_request,
            future=asyncio.Future(),
            queued_at=datetime.now(),
            timeout_at=datetime.now() + timedelta(seconds=30)
        )
        
        batch.add_request(queued_request)
        
        assert not batch.is_empty()
        assert batch.size() == 1
        assert batch.get_prompts() == [sample_request.prompt]


class TestBatchScheduler:
    """Test BatchScheduler functionality."""
    
    @pytest.mark.asyncio
    async def test_scheduler_start_stop(self, request_queue, mock_engine):
        """Test starting and stopping scheduler."""
        scheduler = BatchScheduler(
            queue=request_queue,
            engine=mock_engine,
            max_batch_size=4,
            max_wait_time=0.05
        )
        
        # Start scheduler
        await scheduler.start()
        assert scheduler._running is True
        assert scheduler._task is not None
        
        # Stop scheduler
        await scheduler.stop()
        assert scheduler._running is False
    
    @pytest.mark.asyncio
    async def test_process_single_request(self, request_queue, mock_engine, sample_request):
        """Test processing a single request."""
        scheduler = BatchScheduler(
            queue=request_queue,
            engine=mock_engine,
            max_batch_size=4,
            max_wait_time=0.01
        )
        
        # Enqueue request
        request_id = await request_queue.enqueue(sample_request)
        queued_request = await request_queue.get_pending_request(request_id)
        
        # Process batch directly
        await scheduler._process_batch([queued_request])
        
        # Check result
        assert queued_request.future.done()
        result = queued_request.future.result()
        assert isinstance(result, GenerateResponse)
        assert result.generated_text == "Test response"
        
        # Check metrics
        metrics = scheduler.get_metrics()
        assert metrics["total_batches_processed"] == 1
        assert metrics["total_requests_processed"] == 1
    
    @pytest.mark.asyncio
    async def test_process_multiple_requests(self, request_queue, mock_engine, sample_request):
        """Test processing multiple requests in batch."""
        scheduler = BatchScheduler(
            queue=request_queue,
            engine=mock_engine,
            max_batch_size=4,
            max_wait_time=0.01
        )
        
        # Enqueue multiple requests
        request_ids = []
        queued_requests = []
        for i in range(3):
            request_id = await request_queue.enqueue(sample_request)
            request_ids.append(request_id)
            queued_request = await request_queue.get_pending_request(request_id)
            queued_requests.append(queued_request)
        
        # Process batch
        await scheduler._process_batch(queued_requests)
        
        # Check all results
        for i, queued_request in enumerate(queued_requests):
            assert queued_request.future.done()
            result = queued_request.future.result()
            assert isinstance(result, GenerateResponse)
            assert result.generated_text == f"Test response {i}"
        
        # Check metrics
        metrics = scheduler.get_metrics()
        assert metrics["total_batches_processed"] == 1
        assert metrics["total_requests_processed"] == 3
        assert metrics["avg_batch_size"] == 3.0
    
    @pytest.mark.asyncio
    async def test_scheduler_metrics(self, request_queue, mock_engine):
        """Test scheduler metrics collection."""
        scheduler = BatchScheduler(
            queue=request_queue,
            engine=mock_engine,
            max_batch_size=4,
            max_wait_time=0.01
        )
        
        metrics = scheduler.get_metrics()
        
        assert "running" in metrics
        assert "total_batches_processed" in metrics
        assert "total_requests_processed" in metrics
        assert "total_processing_time" in metrics
        assert "avg_batch_size" in metrics
        assert "avg_processing_time_per_batch" in metrics
        assert "max_batch_size" in metrics
        assert "max_wait_time" in metrics
        assert "batch_sizes_distribution" in metrics
        
        assert metrics["running"] is False
        assert metrics["max_batch_size"] == 4
        assert metrics["max_wait_time"] == 0.01


class TestIntegration:
    """Integration tests for the complete batching system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_batching(self, mock_engine):
        """Test complete batching workflow."""
        # Create queue and scheduler
        queue = RequestQueue(max_size=100, default_timeout=5.0)
        scheduler = BatchScheduler(
            queue=queue,
            engine=mock_engine,
            max_batch_size=2,
            max_wait_time=0.05
        )
        
        # Start scheduler
        await scheduler.start()
        
        try:
            # Enqueue multiple requests
            requests = []
            futures = []
            for i in range(3):
                req = GenerateRequest(prompt=f"Test prompt {i}", max_tokens=50)
                request_id = await queue.enqueue(req)
                queued_request = await queue.get_pending_request(request_id)
                requests.append(queued_request)
                futures.append(queued_request.future)
            
            # Wait for all requests to complete
            try:
                results = await asyncio.wait_for(asyncio.gather(*futures), timeout=2.0)
            except asyncio.TimeoutError:
                pytest.fail("Requests timed out")
            
            # Check results
            assert len(results) == 3
            for result in results:
                assert isinstance(result, GenerateResponse)
                assert result.generated_text is not None
            
            # Check metrics
            queue_metrics = queue.get_metrics()
            scheduler_metrics = scheduler.get_metrics()
            
            assert queue_metrics["total_enqueued"] == 3
            assert queue_metrics["total_dequeued"] == 3
            assert scheduler_metrics["total_requests_processed"] == 3
            assert scheduler_metrics["total_batches_processed"] >= 1  # At least 1 batch
            
        finally:
            await scheduler.stop()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, mock_engine):
        """Test error handling in batching system."""
        # Configure mock to raise exception for both single and batch generation
        def mock_generate_error(*args, **kwargs):
            raise Exception("Test error")
        
        mock_engine.generate = mock_generate_error
        mock_engine.generate_batch = mock_generate_error
        
        queue = RequestQueue(max_size=100, default_timeout=5.0)
        scheduler = BatchScheduler(
            queue=queue,
            engine=mock_engine,
            max_batch_size=2,
            max_wait_time=0.01
        )
        
        # Enqueue request
        req = GenerateRequest(prompt="Test", max_tokens=50)
        request_id = await queue.enqueue(req)
        queued_request = await queue.get_pending_request(request_id)
        
        # Process batch (should handle error)
        await scheduler._process_batch([queued_request])
        
        # Check that future has exception
        assert queued_request.future.done()
        with pytest.raises(Exception, match="Test error"):
            queued_request.future.result()


class TestBatchingLogic:
    """Test specific batching logic scenarios."""
    
    @pytest.mark.asyncio
    async def test_group_5_requests_into_4_and_1(self, mock_engine):
        """Test the specific case: send 5 requests, expect 1 batch of 4 and 1 batch of 1."""
        # Create queue and scheduler with max_batch_size=4
        queue = RequestQueue(max_size=100, default_timeout=5.0)
        scheduler = BatchScheduler(
            queue=queue,
            engine=mock_engine,
            max_batch_size=4,
            max_wait_time=0.05  # Short wait time to ensure quick batching
        )
        
        # Track batch formation
        batch_sizes = []
        
        # Override _process_batch to track batch sizes
        original_process_batch = scheduler._process_batch
        async def track_batch_sizes(batch_requests):
            batch_sizes.append(len(batch_requests))
            await original_process_batch(batch_requests)
        
        scheduler._process_batch = track_batch_sizes
        
        # Start scheduler
        await scheduler.start()
        
        try:
            # Enqueue 5 requests quickly
            requests = []
            futures = []
            for i in range(5):
                req = GenerateRequest(prompt=f"Test prompt {i}", max_tokens=50)
                request_id = await queue.enqueue(req)
                queued_request = await queue.get_pending_request(request_id)
                requests.append(queued_request)
                futures.append(queued_request.future)
            
            # Wait for all requests to complete
            try:
                results = await asyncio.wait_for(asyncio.gather(*futures), timeout=2.0)
            except asyncio.TimeoutError:
                pytest.fail("Requests timed out")
            
            # Verify results
            assert len(results) == 5
            for result in results:
                assert isinstance(result, GenerateResponse)
            
            # Check batch formation: should have formed batches of 4 and 1
            assert len(batch_sizes) >= 2  # At least 2 batches
            assert 4 in batch_sizes, f"Expected a batch of size 4, got batch sizes: {batch_sizes}"
            assert 1 in batch_sizes, f"Expected a batch of size 1, got batch sizes: {batch_sizes}"
            
            # Verify total requests processed
            assert sum(batch_sizes) == 5
            
            # Check metrics
            metrics = scheduler.get_metrics()
            assert metrics["total_requests_processed"] == 5
            
            print(f"Batch sizes formed: {batch_sizes}")
            
        finally:
            await scheduler.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
