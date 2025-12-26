# LLM Inference Performance Benchmarks

**Purpose**: Track all performance optimizations with measurable metrics

**Metrics Tracked**:
- TTFT (Time to First Token) in milliseconds
- TPOT (Time Per Output Token) in milliseconds  
- Throughput (tokens/second)
- VRAM Usage (GB)
- Model loading time

**Baseline Configuration**:
- Model: mistralai/Mistral-7B-Instruct-v0.2
- Hardware: GPU (CUDA)
- Quantization: 8-bit (bitsandbytes)
- Max Length: 2048 tokens

---

## Baseline Measurement - 2025-12-25

*Configuration*: Stock PyTorch/Transformers inference (GPT2 on CPU)

### Single Request Performance
- **TTFT P50**: 11,798ms
- **TTFT P95**: N/A (insufficient successful requests)
- **TPOT P50**: 0ms (CPU bottleneck)
- **Throughput**: 21.9 tokens/sec
- **VRAM Usage**: 0GB (CPU only)
- **RAM Usage**: 16.9GB
- **Error Rate**: 50% (server stability issues)

### Concurrent Load (10 requests)
- **Average TTFT**: N/A
- **Average TPOT**: N/A
- **Total Throughput**: N/A
- **Error Rate**: N/A

### Batch Performance (batch_size=8)
- **TTFT P95**: N/A
- **TPOT P50**: N/A
- **Throughput**: N/A
- **VRAM Usage**: N/A

### Key Observations
1. **High Latency**: TTFT of ~12 seconds indicates severe performance bottleneck
2. **CPU Only**: No GPU acceleration (VRAM = 0GB)
3. **Stability Issues**: 50% error rate suggests server instability
4. **Low Throughput**: 21.9 tokens/sec is far below target of 100+ tokens/sec

### Baseline Issues Identified
- Model running on CPU instead of GPU
- Server connection issues during load
- No batching optimization
- No Flash Attention or other optimizations

---

## Optimization 1: Request Batching - 2025-12-25

### Changes Applied
- Implemented dynamic request batching system (`src/inference/batcher.py`)
- Added RequestQueue with asyncio.Queue for thread-safe operations
- Added BatchScheduler for background batch formation and processing
- Updated API endpoints to use batching system with fallback to direct generation
- Added configuration options: `ENABLE_BATCHING`, `MAX_BATCH_SIZE`, `MAX_WAIT_TIME`
- Added comprehensive metrics and monitoring endpoints

### Performance Impact
- **Before**: TTFT=11,798ms, Throughput=21.9 tok/s, Error Rate=50%
- **After**: TTFT=3,619ms, Throughput=35.0 tok/s, Error Rate=26%
- **Delta**: -69% TTFT, +59% throughput, -48% error rate

### Configuration
- Max Batch Size: 4 requests
- Max Wait Time: 50ms (0.05s)
- Queue Max Size: 1000 requests
- Request Timeout: 30s

### Tradeoff Analysis
**Gains**:
- Improved throughput through batch processing
- Reduced error rate via queue-based load management
- Better CPU utilization with SIMD/threading
- Backward compatibility maintained

**Costs**:
- Up to 50ms additional latency for batch formation
- Increased memory usage for queue management
- Added system complexity

**Why worth it**: Expected 1.5x+ throughput improvement with <5% error rate

### Rollback Plan
1. Set `ENABLE_BATCHING=false` environment variable
2. Or set `MAX_BATCH_SIZE=1` to disable batching
3. Restart service
4. Recovery time: < 1 minute

### Test Results
**Benchmark Date**: 2025-12-26 06:16:52

**Concurrent Load Test** (10 concurrent, 50 total requests):
- **TTFT P50**: 3,619.4ms
- **TTFT P95**: 4,187.6ms
- **TTFT P99**: 4,309.4ms
- **Throughput**: 35.0 tokens/sec ✅ **MEETS TARGET** (≥33 tok/s)
- **Requests/sec**: 0.26
- **Error Rate**: 26% ❌ **FAILS TARGET** (<5%)
- **Total Tokens**: 4,921
- **Successful Requests**: 37/50 (74%)

**Success Criteria Assessment**:
- ✅ Throughput target met: 35.0 tok/s (59% improvement)
- ❌ Error rate target not met: 26% (needs to be <5%)

**Issues Identified**:
1. High error rate (26%) indicates timeout or queue issues
2. High latency (TTFT P50: 3.6s) suggests processing bottlenecks
3. 13 requests failed, likely due to timeouts during batch processing

**Recommended Next Steps**:
1. Increase `REQUEST_TIMEOUT` from 30s to 60s
2. Adjust `MAX_WAIT_TIME` from 50ms to 20ms for faster batch dispatch
3. Monitor queue metrics to identify bottlenecks
4. Consider implementing request prioritization

---

## Optimization 2: [NAME] - [DATE]

[... repeat format for each optimization ...]

---

## Summary Table

| Optimization | TTFT (ms) | TPOT (ms) | Throughput (tok/s) | VRAM (GB) | Notes |
|-------------|-----------|-----------|-------------------|-----------|-------|
| Baseline | [TBD] | [TBD] | [TBD] | [TBD] | Stock PyTorch |
| [Opt 1] | [TBD] | [TBD] | [TBD] | [TBD] | [Description] |
| [Opt 2] | [TBD] | [TBD] | [TBD] | [TBD] | [Description] |

---

**Target Goals**:
- TTFT P95: < 500ms (from ~7700ms)
- Throughput: > 100 tokens/sec (from ~0.12 req/s)
- VRAM: < 16GB with quantization
- Zero-downtime deployments
