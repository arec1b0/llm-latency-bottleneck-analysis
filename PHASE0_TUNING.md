# Phase 0 Tuning - Configuration Adjustments

## Changes Made (2025-12-26)

### Problem
Initial benchmark showed:
- ✅ Throughput: 35.0 tok/s (target met)
- ❌ Error Rate: 26% (target: <5%)
- Issue: Requests timing out during batch processing

### Configuration Changes

**File**: `src/api/main.py`

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `REQUEST_TIMEOUT` | 30.0s | **60.0s** | Prevent timeouts during batch processing |
| `MAX_WAIT_TIME` | 0.05s (50ms) | **0.02s (20ms)** | Faster batch dispatch, reduce queue wait time |

### Expected Impact

**Timeout Increase (30s → 60s)**:
- Allows more time for batch processing to complete
- Reduces timeout-related errors
- Trade-off: Slower failure detection

**Wait Time Reduction (50ms → 20ms)**:
- Batches form faster, reducing total request latency
- Requests spend less time waiting in queue
- Trade-off: Potentially smaller batch sizes (but still respects MAX_BATCH_SIZE=4)

### How to Test

1. **Restart the API server** (Terminal 1):
   ```powershell
   # Stop current server (Ctrl+C)
   cd f:\Projects\llm-latency-bottleneck-analysis
   python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Run benchmark** (Terminal 2):
   ```powershell
   cd f:\Projects\llm-latency-bottleneck-analysis
   python scripts/benchmark.py --concurrent --url http://localhost:8000 --output benchmark_phase0_tuned
   ```

3. **Success Criteria**:
   - Error rate < 5% (down from 26%)
   - Throughput ≥ 33 tok/s (maintain current 35.0 tok/s)
   - TTFT should remain similar or improve

### Rollback Plan

If tuning makes things worse:
```python
# In src/api/main.py, revert to:
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "30.0"))
MAX_WAIT_TIME = float(os.getenv("MAX_WAIT_TIME", "0.05"))
```

Or use environment variables:
```bash
export REQUEST_TIMEOUT=30.0
export MAX_WAIT_TIME=0.05
```

### Tuning Attempt Results

#### Attempt 1: Reduce Wait Time + Increase Timeout
**Config**: `MAX_WAIT_TIME=0.02s (20ms)`, `REQUEST_TIMEOUT=60s`
**Results**: 
- Error Rate: 30% ❌ (worse than 26%)
- Throughput: 29.2 tok/s ❌ (worse than 35.0 tok/s)
- **Conclusion**: Reducing wait time caused smaller batches, reducing efficiency

#### Attempt 2: Keep Wait Time + Increase Timeout More
**Config**: `MAX_WAIT_TIME=0.05s (50ms)`, `REQUEST_TIMEOUT=90s`
**Results**:
- Error Rate: 28% ❌
- Throughput: 17.6 tok/s ❌ (Degraded significantly)
- **Observation**: Server connection failures ("WinError 64"). CPU saturated, requests backing up until connection limits/timeouts hit.

### Root Cause Analysis

**The Real Problem**: CPU-based inference is fundamentally too slow for concurrent load
- VRAM Usage: 0.00GB → Model running on CPU, not GPU
- TTFT: 3.6-4.2 seconds → CPU inference is 10-100x slower than GPU
- Error Rate: 26-30% → Requests timing out because CPU can't keep up

**Why Batching Alone Can't Fix This**:
1. Batching improves throughput by ~59% (22 → 35 tok/s)
2. But CPU is still too slow to process requests within timeout window
3. Even with 90s timeout, CPU processing time exceeds capacity under load

**The Solution**: Phase 1 (GPU/vLLM Migration) is required to achieve <5% error rate
- GPU inference: 10-100x faster than CPU
- vLLM: Optimized batching + PagedAttention
- Expected: <5% error rate with GPU acceleration

### Recommendation

**Accept Phase 0 as "Partial Success"** and proceed to Phase 1:
- ✅ Batching system implemented and working
- ✅ Throughput improved by 59% (22 → 35 tok/s)
- ✅ Error rate reduced by 48% (50% → 26%)
- ❌ Cannot achieve <5% error rate on CPU alone

**Phase 1 (GPU/vLLM) will address**:
- GPU acceleration for 10-100x faster inference
- vLLM's optimized batching and memory management
- Expected to achieve both throughput AND error rate targets
