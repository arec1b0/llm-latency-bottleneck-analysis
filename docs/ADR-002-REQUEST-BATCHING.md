# ADR-002: Request Batching Strategy

**Status**: Draft  
**Date**: 2025-12-25  
**Decision**: Implement Dynamic Request Batching

---

## Context

The current LLM inference system processes requests serially with a semaphore-based concurrency limit of 1. Under load testing, this approach shows:
- **TTFT**: ~11.8s (Time to First Token)
- **Throughput**: ~22 tokens/sec
- **Error Rate**: 50% (timeouts/overload)

The bottleneck is thread contention and inefficient CPU utilization when processing concurrent requests. Each request waits for the previous one to complete, even though the model could process multiple prompts simultaneously.

---

## Decision

We will implement **Dynamic Request Batching** with the following characteristics:

### 1. Batching Strategy
- **Type**: Dynamic batching (not fixed-size batches)
- **Batch Formation**: Group requests based on time and size criteria
- **Max Batch Size**: 4 requests
- **Max Wait Time**: 50ms (0.05s)
- **Queue Type**: `asyncio.Queue` for thread-safe operations

### 2. Architecture Components

#### RequestQueue (`src/inference/batcher.py`)
```python
class RequestQueue:
    """Thread-safe request queue for batching"""
    def __init__(self, max_size: int = 1000):
        self.queue = asyncio.Queue(maxsize=max_size)
        self.pending_requests = {}  # request_id -> Future
    
    async def enqueue(self, request: GenerateRequest) -> str:
        """Add request to queue and return request ID"""
        
    async def get_batch(self, max_size: int, max_wait: float) -> List[GenerateRequest]:
        """Get batch of requests respecting size and time limits"""
```

#### BatchScheduler (`src/inference/batcher.py`)
```python
class BatchScheduler:
    """Background task that forms batches from queued requests"""
    def __init__(self, queue: RequestQueue, engine: InferenceEngine):
        self.queue = queue
        self.engine = engine
        self.max_batch_size = 4
        self.max_wait_time = 0.05  # 50ms
        
    async def start(self):
        """Start background batch processing loop"""
        
    async def _process_batch(self, batch: List[GenerateRequest]):
        """Process batch and distribute results to requesters"""
```

### 3. API Integration

#### Modified Generate Endpoint
```python
@app.post("/generate")
async def generate_text(request: GenerateRequest):
    # Instead of calling engine.generate directly:
    request_id = await request_queue.enqueue(request)
    result = await wait_for_result(request_id, timeout=30.0)
    return result
```

#### Engine Modification
```python
# In InferenceEngine:
def generate(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
    """Modified to accept list of prompts for batch processing"""
```

---

## Consequences

### Positive
1. **Improved Throughput**: Batch processing leverages CPU SIMD instructions more efficiently
2. **Reduced Error Rate**: Queue prevents request overload and timeouts
3. **Better Resource Utilization**: CPU processes multiple prompts simultaneously
4. **Scalable**: Can adjust batch size and wait time based on load
5. **Backward Compatible**: API contract remains the same

### Negative
1. **Increased Latency for Single Requests**: May wait up to 50ms for batch formation
2. **Memory Overhead**: Queue holds pending requests in memory
3. **Complexity**: Additional components (queue, scheduler) increase system complexity
4. **Debugging Difficulty**: Harder to trace individual request processing

### Trade-offs
- **Latency vs Throughput**: Sacrifice up to 50ms latency for 1.5x+ throughput improvement
- **Memory vs Performance**: Additional memory usage for queue and batch management
- **Complexity vs Scalability**: More moving parts but better load handling

---

## Alternatives Considered

### 1. Fixed-Size Batching
- **Description**: Always wait for exactly N requests before processing
- **Rejected**: Would increase latency significantly for low-traffic periods
- **Impact**: Poor user experience during off-peak hours

### 2. No Batching (Current State)
- **Description**: Maintain current serial processing
- **Rejected**: Cannot meet throughput and error rate targets
- **Impact**: System fails under moderate load

### 3. External Batching Service
- **Description**: Separate microservice for request batching
- **Rejected**: Adds operational complexity and network overhead
- **Impact**: Over-engineering for current scale

---

## Implementation Plan

### Phase 1: Core Components
1. Create `src/inference/batcher.py` with RequestQueue and BatchScheduler
2. Implement unit tests for queue operations and batch formation logic
3. Modify InferenceEngine.generate to accept List[str] prompts

### Phase 2: API Integration
1. Update `/generate` endpoint to use queue instead of direct engine calls
2. Implement request/response correlation with unique IDs
3. Add timeout handling for queued requests

### Phase 3: Performance Validation
1. Run benchmark tests with concurrency > 1
2. Verify error rate drops below 5%
3. Confirm 1.5x+ throughput improvement
4. Tune batch size and wait time parameters

---

## Success Criteria

1. **Unit Tests**: All batcher tests pass (queueing, timeout, batch formation)
2. **Load Test**: `scripts/performance_test.py` handles 4+ concurrent requests
3. **Error Rate**: < 5% (down from 50%)
4. **Throughput**: ≥ 33 tokens/sec (1.5x improvement from 22 tok/s)
5. **Latency**: TTFT remains ≤ 12s (acceptable trade-off)

---

## Rollback Plan

If batching introduces issues:
1. **Immediate**: Set `max_batch_size=1` to disable batching
2. **Configuration**: Add `ENABLE_BATCHING=false` environment variable
3. **Code**: Revert API endpoint to call `engine.generate` directly
4. **Timeline**: Complete rollback in < 5 minutes

---

## Monitoring Requirements

1. **Queue Depth**: Monitor `request_queue_size` metric
2. **Batch Efficiency**: Track `batch_size_distribution` histogram
3. **Wait Times**: Measure `batch_wait_time_seconds`
4. **Error Rate**: Continue monitoring `generation_error_rate`
5. **Throughput**: Track `tokens_per_second_batched`

---

## Future Considerations

1. **Adaptive Batching**: Dynamically adjust batch size based on load
2. **Priority Queues**: Different service levels for different request types
3. **Distributed Batching**: Multiple workers sharing a single queue
4. **GPU Batching**: Leverage GPU batch processing when available
