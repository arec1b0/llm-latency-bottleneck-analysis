# LLM Inference Optimization Plan - PyTorch/Transformers Stack

**Document Version**: 1.0
**Status**: READY FOR EXECUTION
**Author**: High-Performance ML Engineer
**Date**: 2025-12-25

## Executive Summary

**Current Stack Analysis**:
- **Engine**: Custom PyTorch/Transformers (not vLLM)
- **Model**: mistralai/Mistral-7B-Instruct-v0.2
- **Quantization**: 8-bit (bitsandbytes)
- **API**: FastAPI with async support
- **Observability**: OpenTelemetry + Prometheus

**Target Outcomes**:
- TTFT P95: ~7700ms → < 500ms (94% improvement)
- Throughput: 0.12 req/s → 100+ req/s (833x improvement)
- VRAM: ~24GB → < 16GB (33% reduction)
- Recovery time: < 5 minutes

---

## Phase 1: Baseline Measurement

### Step 1.1: Run Comprehensive Benchmarks

**Task**: Establish performance baseline with `scripts/benchmark.py`

**Commands**:
```bash
# Start API service
python -m src.api.main

# Run full benchmark suite
python scripts/benchmark.py --all --output baseline

# Document results in docs/BENCHMARKS_LOG.md
```

**Acceptance Criteria**:
- [ ] All benchmark types complete (single, concurrent, batch, streaming)
- [ ] Baseline metrics recorded in BENCHMARKS_LOG.md
- [ ] VRAM usage documented
- [ ] Error rates documented

---

## Phase 2: Architecture Layer Optimizations

### Step 2.1: Implement Request Batching

**Current State**: Single request processing only

**Target**: Dynamic batching with configurable batch sizes

**Files to Modify**:
- `src/inference/engine.py` - Add `BatchManager` class
- `src/api/main.py` - Add batch endpoint optimization

**Implementation**:
```python
class BatchManager:
    def __init__(self, max_batch_size: int = 8, max_wait_ms: int = 50):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.pending_requests = []
        self.batch_lock = asyncio.Lock()
    
    async def add_request(self, request_data) -> asyncio.Future:
        """Add request to batch queue and return Future for result."""
        # Implementation for dynamic batching
```

**Acceptance Criteria**:
- [ ] Dynamic batching implemented with configurable batch sizes
- [ ] Batch timeout mechanism prevents starvation
- [ ] Benchmark shows >2x throughput improvement
- [ ] Individual request latency doesn't increase significantly

### Step 2.2: Optimize Streaming Implementation

**Current State**: Basic streaming with manual token generation

**Target**: Optimized streaming with proper async handling

**Files to Modify**:
- `src/inference/engine.py` - Fix duplicate `generate_stream` methods
- `src/api/main.py` - Optimize SSE streaming

**Implementation**:
- Remove duplicate `generate_stream` method (lines 312-500 and 697-880)
- Implement proper async streaming with `asyncio.Queue`
- Add backpressure handling

**Acceptance Criteria**:
- [ ] Single, optimized `generate_stream` method
- [ ] Streaming latency < 10ms per token
- [ ] No memory leaks during long streams
- [ ] Proper error handling in streams

### Step 2.3: Add Connection Pooling and Keep-Alive

**Target**: Optimize HTTP connection handling

**Files to Modify**:
- `src/api/main.py` - Add connection pooling configuration

**Implementation**:
```python
# Add to FastAPI app configuration
@app.middleware("http")
async def connection_pooling_middleware(request: Request, call_next):
    # Implement connection pooling logic
```

**Acceptance Criteria**:
- [ ] HTTP keep-alive enabled
- [ ] Connection pooling configured
- [ ] Reduced connection overhead in benchmarks

---

## Phase 3: Engine Layer Optimizations

### Step 3.1: Enable Flash Attention 2

**Current State**: Standard attention (eager mode)

**Target**: Flash Attention 2 for memory and speed optimization

**Files to Modify**:
- `src/inference/engine.py` - Update model loading

**Implementation**:
```python
# In load_model() method:
if self.device == "cuda":
    # Enable Flash Attention 2
    self.model = AutoModelForCausalLM.from_pretrained(
        self.model_name,
        attn_implementation="flash_attention_2",
        ... # other params
    )
```

**Dependencies to Add**:
```bash
pip install flash-attn --no-build-isolation
```

**Acceptance Criteria**:
- [ ] Flash Attention 2 successfully enabled
- [ ] Memory usage reduced by 20-30%
- [ ] TTFT improved by 15-25%
- [ ] Model loading logs confirm FA2 usage

### Step 3.2: Implement KV Cache Optimization

**Current State**: Standard KV cache handling

**Target**: Optimized KV cache with memory management

**Files to Modify**:
- `src/inference/engine.py` - Add KV cache management

**Implementation**:
```python
class KVCacheManager:
    def __init__(self, max_cache_size: int):
        self.max_cache_size = max_cache_size
        self.cache_store = {}
    
    def get_cache_key(self, input_ids: torch.Tensor) -> str:
        """Generate cache key for input sequence."""
    
    def store_cache(self, key: str, past_key_values):
        """Store KV cache with LRU eviction."""
```

**Acceptance Criteria**:
- [ ] KV cache reuse for repeated prompts
- [ ] Cache hit rate > 10% in typical usage
- [ ] Memory usage stays within limits
- [ ] Cache invalidation works correctly

### Step 3.3: Optimize Memory Management

**Target**: Reduce memory fragmentation and improve allocation

**Files to Modify**:
- `src/inference/engine.py` - Add memory optimization

**Implementation**:
```python
def optimize_memory(self):
    """Optimize GPU memory usage."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Set memory fraction if needed
        torch.cuda.set_per_process_memory_fraction(0.95)
```

**Acceptance Criteria**:
- [ ] Memory fragmentation reduced
- [ ] OOM errors eliminated under normal load
- [ ] Memory usage stable over time

---

## Phase 4: Model Layer Optimizations

### Step 4.1: Implement 4-bit Quantization (GPTQ/AWQ)

**Current State**: 8-bit quantization (bitsandbytes)

**Target**: 4-bit quantization for maximum memory savings

**Decision Point**: Choose between GPTQ vs AWQ
- **GPTQ**: Better quality, longer loading time
- **AWQ**: Faster loading, slightly less accurate

**Files to Modify**:
- `src/inference/engine.py` - Add 4-bit quantization support
- `requirements.txt` - Add quantization libraries

**Implementation**:
```python
# Add quantization options
if self.quantization_method == "gptq":
    from transformers import AutoModelForCausalLM
    self.model = AutoModelForCausalLM.from_pretrained(
        self.model_name,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    )
```

**Dependencies**:
```bash
pip install auto-gptq transformers-optimum
# or
pip install auto-awq
```

**Acceptance Criteria**:
- [ ] 4-bit quantization working
- [ ] VRAM usage < 12GB (from ~24GB)
- [ ] Quality loss < 2% (BLEU score comparison)
- [ ] Loading time acceptable (< 2 minutes)

### Step 4.2: Implement Speculative Decoding

**Target**: Use draft model for faster generation

**Files to Modify**:
- `src/inference/engine.py` - Add speculative decoding

**Implementation**:
```python
class SpeculativeDecoder:
    def __init__(self, target_model, draft_model):
        self.target_model = target_model
        self.draft_model = draft_model
    
    async def generate_with_speculation(self, prompt: str, max_tokens: int):
        """Generate using speculative decoding with draft model."""
        # Implementation of speculative decoding algorithm
```

**Acceptance Criteria**:
- [ ] Draft model integration working
- [ ] Acceptance rate > 70%
- [ ] TTFT improved by 30-40%
- [ ] Quality maintained (no regression)

---

## Phase 5: Advanced Optimizations

### Step 5.1: Implement Tensor Parallelism

**Target**: Split model across multiple GPUs

**Files to Modify**:
- `src/inference/engine.py` - Add multi-GPU support

**Implementation**:
```python
def setup_tensor_parallelism(self, num_gpus: int = 2):
    """Setup tensor parallelism across multiple GPUs."""
    import torch.distributed as dist
    # Implementation for tensor parallelism
```

**Acceptance Criteria**:
- [ ] Model successfully splits across GPUs
- [ ] Throughput scales with GPU count
- [ ] Communication overhead minimal

### Step 5.2: Add Semantic Cache

**Target**: Cache responses for semantically similar prompts

**Files to Modify**:
- `src/cache/semantic_cache.py` - New file
- `src/api/main.py` - Integrate cache

**Implementation**:
```python
from sentence_transformers import SentenceTransformer
import redis

class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.95):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.threshold = similarity_threshold
        self.redis_client = redis.Redis()
    
    async def get_cached_response(self, prompt: str) -> Optional[str]:
        """Check cache for semantically similar prompt."""
```

**Acceptance Criteria**:
- [ ] Cache hit rate > 15% for typical queries
- [ ] Cache lookup latency < 5ms
- [ ] Semantic similarity working correctly

---

## Phase 6: Verification and Rollback

### Step 6.1: Comprehensive Testing

**Task**: Run full benchmark suite after each optimization

**Commands**:
```bash
# After each optimization step
python scripts/benchmark.py --all --output optimization_[step_name]

# Compare with baseline
python scripts/compare_benchmarks.py baseline.json optimization_[step_name].json
```

**Acceptance Criteria**:
- [ ] Each optimization shows measurable improvement
- [ ] No regressions in error rates
- [ ] Memory usage within limits
- [ ] Quality metrics maintained

### Step 6.2: Rollback Procedures

**Task**: Document and test rollback procedures

**Files to Create**:
- `scripts/rollback_optimization.py` - Automated rollback script

**Implementation**:
```python
def rollback_optimization(step_name: str):
    """Rollback specific optimization step."""
    rollback_map = {
        "flash_attention": revert_to_eager_attention,
        "4bit_quantization": revert_to_8bit,
        "speculative_decoding": disable_speculation,
        # ... other rollbacks
    }
```

**Acceptance Criteria**:
- [ ] Each optimization can be rolled back independently
- [ ] Rollback time < 2 minutes
- [ ] Service remains available during rollback

---

## Success Metrics

### Primary Metrics
| Metric | Current | Target | Measurement Method |
|--------|---------|--------|-------------------|
| TTFT P95 | ~7700ms | < 500ms | Benchmark script |
| Throughput | 0.12 req/s | > 100 req/s | Load test |
| VRAM Usage | ~24GB | < 16GB | nvidia-smi |
| Error Rate | < 1% | < 1% | Benchmark logs |

### Secondary Metrics
- **Model Loading Time**: < 2 minutes
- **Recovery Time**: < 5 minutes
- **Cache Hit Rate**: > 15%
- **GPU Utilization**: > 80%

---

## Execution Order

1. **Baseline** (Step 1.1) - MUST COMPLETE FIRST
2. **Batching** (Step 2.1) - Highest impact
3. **Flash Attention 2** (Step 3.1) - Medium impact, low risk
4. **4-bit Quantization** (Step 4.1) - High impact, medium risk
5. **Streaming Optimization** (Step 2.2) - Medium impact
6. **Speculative Decoding** (Step 4.2) - Medium impact, high complexity
7. **Semantic Cache** (Step 5.2) - Low impact, easy win
8. **Tensor Parallelism** (Step 5.1) - High impact, high complexity

---

## Risk Assessment

### High Risk
- **4-bit Quantization**: Quality degradation risk
- **Speculative Decoding**: Complexity risk
- **Tensor Parallelism**: Multi-GPU coordination risk

### Medium Risk
- **Flash Attention 2**: Compatibility risk
- **Batching**: Latency increase risk

### Low Risk
- **Streaming Optimization**: Implementation risk only
- **Semantic Cache**: No impact on core functionality
- **Memory Management**: Low risk, high reward

---

## Next Steps

1. **IMMEDIATE**: Run baseline benchmarks
2. **PRIORITY 1**: Implement request batching
3. **PRIORITY 2**: Enable Flash Attention 2
4. **PRIORITY 3**: Evaluate 4-bit quantization options

**Remember**: One optimization at a time, measure everything, document results.
