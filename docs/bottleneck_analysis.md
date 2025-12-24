# Bottleneck Analysis Guide

## Introduction

This guide provides a systematic approach to identifying and resolving latency bottlenecks in LLM inference pipelines. It covers analysis methodology, common bottlenecks, diagnostic techniques, and optimization strategies.

## Analysis Methodology

### Step 1: Baseline Establishment

Before optimizing, establish baseline performance metrics.

**Run Baseline Load Test**:

```bash
cd load_testing
locust -f locustfile.py --users 10 --spawn-rate 2 --run-time 5m --host http://localhost:8000 --headless
```

**Collect Baseline Metrics**:

1. **TTFT (Time to First Token)**:
   - P50, P95, P99 values
   - Target: P95 < 1s

2. **TPOT (Time Per Output Token)**:
   - Average across all requests
   - Target: < 0.1s

3. **Throughput**:
   - Requests per second
   - Tokens per second
   - Target: System-dependent

4. **Resource Utilization**:
   - GPU utilization: 70-90%
   - VRAM usage: < 85%
   - CPU usage: < 50%

**Documentation**:

```markdown
# Baseline Performance (Date: YYYY-MM-DD)

## Load Configuration
- Users: 10
- Spawn Rate: 2/sec
- Duration: 5 minutes

## Results
- TTFT P50: 0.456s
- TTFT P95: 0.892s
- TTFT P99: 1.234s
- TPOT Avg: 0.045s
- Throughput: 28.5 tokens/sec
- GPU Utilization: 78%
- VRAM Usage: 12.3GB / 16GB (77%)
```

### Step 2: Trace Analysis

Use Jaeger to identify slow spans and operations.

**Analyze Traces**:

```bash
python scripts/analyze_traces.py llm-inference-api 1h 200
```

**Key Questions**:

1. **What is the slowest operation?**
   - Look at "Slowest Operations" section
   - Typical bottleneck: model_generate

2. **Where is time being spent?**
   - Examine span durations
   - Calculate percentage of total time

3. **Are there unexpected delays?**
   - Check for gaps between spans
   - Look for serialization overhead

4. **What patterns emerge under load?**
   - Compare traces at different loads
   - Identify degradation points

**Example Analysis**:

```
Operation Performance:
  tokenize_input:     45ms (0.5%)
  model_generate:   8,400ms (98.8%)
  decode_output:      45ms (0.5%)
  
Bottleneck: model_generate (98.8% of time)
```

### Step 3: Metrics Correlation

Use Prometheus to correlate metrics with performance.

**Key PromQL Queries**:

```promql
# TTFT over time
histogram_quantile(0.95, rate(llm_ttft_seconds_bucket[5m]))

# TPOT over time
rate(llm_tpot_seconds_sum[5m]) / rate(llm_tpot_seconds_count[5m])

# Memory usage
llm_memory_usage_megabytes{memory_type="vram"}

# Request rate
rate(llm_requests_total[5m])

# Error rate
rate(llm_errors_total[5m]) / rate(llm_requests_total[5m])
```

**Correlation Analysis**:

1. **TTFT vs Load**:
   - Does TTFT increase with concurrent requests?
   - Indicates queue/memory contention

2. **TPOT vs Memory**:
   - Does TPOT increase as VRAM fills?
   - Indicates memory bandwidth bottleneck

3. **Throughput vs GPU Utilization**:
   - Low GPU utilization + low throughput = I/O bottleneck
   - High GPU utilization + low throughput = compute bottleneck

### Step 4: Bottleneck Classification

Classify bottlenecks into categories:

**1. Compute-Bound**:
- **Symptoms**: High GPU utilization (>90%), stable TPOT
- **Causes**: Insufficient compute power
- **Solutions**: Better hardware, quantization, smaller models

**2. Memory-Bound**:
- **Symptoms**: High VRAM usage (>85%), increasing TPOT
- **Causes**: Memory bandwidth limits, cache misses
- **Solutions**: 8-bit quantization, KV cache optimization

**3. I/O-Bound**:
- **Symptoms**: Low GPU utilization (<50%), high TTFT
- **Causes**: Slow data transfer, tokenization overhead
- **Solutions**: Async I/O, pre-tokenization, batching

**4. Queue-Bound**:
- **Symptoms**: TTFT increases with load, requests queued
- **Causes**: Serial processing, no concurrency
- **Solutions**: Request batching, async workers, load balancing

**5. Software-Bound**:
- **Symptoms**: Unexpected delays, inconsistent performance
- **Causes**: Inefficient code, poor algorithms
- **Solutions**: Profiling, code optimization, caching

## Common Bottlenecks

### Bottleneck 1: High TTFT (Time to First Token)

**Symptoms**:
- TTFT P95 > 1.5s
- User-perceived latency
- Poor responsiveness

**Root Causes**:

1. **Cold Start**:
   - Model not preloaded
   - First request loads model
   - **Solution**: Preload model on startup

2. **Tokenization Overhead**:
   - Slow tokenizer initialization
   - Complex preprocessing
   - **Solution**: Cache tokenizer, optimize preprocessing

3. **Memory Transfer**:
   - CPU → GPU data movement
   - Large input sequences
   - **Solution**: Pin memory, optimize batch size

4. **Queue Wait Time**:
   - Requests waiting for GPU
   - Serial processing
   - **Solution**: Request batching, async processing

**Diagnostic Steps**:

```bash
# 1. Check Jaeger spans
python scripts/analyze_traces.py

# Look for:
# - model_load span (should be < 10s)
# - tokenize_input span (should be < 100ms)
# - First request vs subsequent requests

# 2. Test with preloading
curl -X POST http://localhost:8000/model/load
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Test", "max_tokens": 10}'

# 3. Profile tokenization
python -m cProfile -s cumtime -m transformers-cli env
```

**Optimization Strategy**:

```python
# Before: Lazy loading
# TTFT first request: 12s (10s load + 2s generation)

# After: Preloading
# In src/api/main.py lifespan:
async def lifespan(app: FastAPI):
    # ... init code ...
    inference_engine.load_model()  # Preload
    yield
    # ... cleanup ...

# TTFT first request: 0.5s
# Improvement: 24x faster
```

### Bottleneck 2: High TPOT (Time Per Output Token)

**Symptoms**:
- TPOT > 0.1s
- Slow generation speed
- Low tokens/sec throughput

**Root Causes**:

1. **Insufficient GPU Compute**:
   - Older GPU (K80, P100)
   - Shared GPU resources
   - **Solution**: Upgrade to A100, H100

2. **Memory Bandwidth**:
   - Large model, slow VRAM
   - Memory access patterns
   - **Solution**: Quantization, Flash Attention

3. **Inefficient Decoding**:
   - Naive sampling
   - Repeated computations
   - **Solution**: KV cache, optimized kernels

4. **Temperature/Sampling**:
   - High temperature increases variance
   - Top-k/top-p adds overhead
   - **Solution**: Greedy decoding for speed

**Diagnostic Steps**:

```bash
# 1. Check GPU utilization
nvidia-smi dmon -s um -c 60

# Look for:
# - GPU utilization should be 80-95%
# - Memory usage should be stable

# 2. Profile generation
python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    inputs = tokenizer('Test', return_tensors='pt')
    outputs = model.generate(**inputs, max_new_tokens=50)

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
"

# 3. Compare configurations
# Test FP16 vs 8-bit vs 4-bit
```

**Optimization Strategy**:

```python
# Configuration impact on TPOT:

# FP16 (baseline):
# TPOT: 0.045s, Memory: 14GB

# 8-bit quantization:
load_in_8bit=True
# TPOT: 0.052s (+15%), Memory: 7GB (-50%)

# Greedy decoding (vs sampling):
do_sample=False
# TPOT: 0.038s (-15%), deterministic

# Flash Attention (requires compatible model):
attn_implementation="flash_attention_2"
# TPOT: 0.032s (-29%), same memory
```

### Bottleneck 3: Memory Issues (OOM)

**Symptoms**:
- OOM errors
- Request failures
- Degraded performance before crash

**Root Causes**:

1. **Model Too Large**:
   - 7B model in FP16 = ~14GB
   - Limited GPU memory
   - **Solution**: 8-bit quantization, smaller model

2. **Batch Size Too Large**:
   - Memory grows with batch size
   - KV cache accumulation
   - **Solution**: Reduce batch size, stream tokens

3. **Memory Leaks**:
   - Gradients not cleared
   - Cache not freed
   - **Solution**: Use `torch.no_grad()`, clear cache

4. **Concurrent Requests**:
   - Multiple models in memory
   - Memory fragmentation
   - **Solution**: Single model, request queue

**Diagnostic Steps**:

```bash
# 1. Monitor memory
nvidia-smi dmon -s um -c 60

# 2. Track memory over time
watch -n 1 'nvidia-smi | grep MiB'

# 3. Profile memory usage
python -c "
import torch
from transformers import AutoModelForCausalLM

torch.cuda.reset_peak_memory_stats()
model = AutoModelForCausalLM.from_pretrained(
    'mistralai/Mistral-7B-Instruct-v0.2',
    device_map='auto'
)

print(f'Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB')
"
```

**Optimization Strategy**:

```python
# Memory optimization techniques:

# 1. 8-bit quantization
from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)
# Memory: 14GB → 7GB

# 2. Gradient checkpointing (training only)
model.gradient_checkpointing_enable()
# Memory: -30-40%

# 3. KV cache optimization
max_new_tokens=128  # Limit output length
# Memory: Proportional to output length

# 4. Clear cache between requests
torch.cuda.empty_cache()
# Prevents fragmentation
```

### Bottleneck 4: Network/Serialization

**Symptoms**:
- High latency despite fast inference
- Large payload sizes
- Bandwidth saturation

**Root Causes**:

1. **Large Responses**:
   - Long generated text
   - Verbose JSON
   - **Solution**: Compression, streaming

2. **Serialization Overhead**:
   - JSON encoding/decoding
   - Pydantic validation
   - **Solution**: MessagePack, protobuf

3. **Network Latency**:
   - Distant clients
   - Slow network
   - **Solution**: CDN, edge deployment

**Diagnostic Steps**:

```bash
# 1. Measure API overhead
time curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Test", "max_tokens": 10}' \
  -o /dev/null

# 2. Profile serialization
python -m cProfile -s cumtime src/api/main.py

# 3. Check response sizes
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Test", "max_tokens": 256}' \
  -w "\nSize: %{size_download} bytes\nTime: %{time_total}s\n"
```

**Optimization Strategy**:

```python
# 1. Streaming responses (FastAPI)
from fastapi.responses import StreamingResponse

@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    async def token_stream():
        for token in inference_engine.generate_stream(request.prompt):
            yield f"data: {json.dumps({'token': token})}\n\n"
    
    return StreamingResponse(token_stream(), media_type="text/event-stream")

# 2. Response compression
from fastapi.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 3. Minimal responses
# Only include necessary fields
# Remove verbose debugging info
```

## Optimization Checklist

### Quick Wins (< 1 hour)

- [ ] Preload model on startup
- [ ] Enable 8-bit quantization
- [ ] Set appropriate max_tokens limits
- [ ] Add response compression
- [ ] Use greedy decoding (when applicable)
- [ ] Clear CUDA cache between requests

### Medium Effort (1 day)

- [ ] Implement request batching
- [ ] Add async generation
- [ ] Optimize tokenization
- [ ] Implement response streaming
- [ ] Add request queue
- [ ] Profile and optimize hot paths

### Long Term (1 week+)

- [ ] Migrate to vLLM/TGI
- [ ] Implement Flash Attention
- [ ] Add horizontal scaling
- [ ] Optimize model architecture
- [ ] Custom CUDA kernels
- [ ] Distributed inference

## Performance Targets

### Production SLAs

**Latency**:
- TTFT P95: < 1s
- TPOT Avg: < 0.05s
- Total P95: < 10s (for 256 tokens)

**Availability**:
- Uptime: > 99.9%
- Error rate: < 0.1%

**Throughput**:
- Concurrent requests: > 10
- Tokens/sec/GPU: > 50

**Resource Efficiency**:
- GPU utilization: 70-90%
- VRAM usage: < 85%
- Cost per 1M tokens: < $X (define based on cloud costs)

## Continuous Monitoring

### Daily Checks

1. **Review Metrics**:
   - Check Prometheus dashboards
   - Monitor TTFT/TPOT trends
   - Verify error rates

2. **Analyze Traces**:
   - Run trace analysis script
   - Look for new bottlenecks
   - Identify degradation

3. **Resource Monitoring**:
   - Check GPU utilization
   - Monitor memory usage
   - Verify disk space

### Weekly Reviews

1. **Performance Trends**:
   - Compare week-over-week metrics
   - Identify degradation patterns
   - Plan optimizations

2. **Capacity Planning**:
   - Project growth
   - Evaluate scaling needs
   - Budget for upgrades

3. **Incident Analysis**:
   - Review OOM events
   - Analyze timeout errors
   - Document lessons learned

## Tools and Resources

### Profiling Tools

- **PyTorch Profiler**: CUDA kernel analysis
- **Nsight Systems**: GPU profiling
- **cProfile**: Python profiling
- **line_profiler**: Line-by-line profiling
- **memory_profiler**: Memory usage tracking

### Load Testing

- **Locust**: Distributed load testing
- **k6**: Modern load testing
- **Apache JMeter**: Enterprise load testing
- **wrk**: Simple HTTP benchmark

### Monitoring

- **Jaeger**: Distributed tracing
- **Prometheus**: Metrics collection
- **Grafana**: Visualization
- **nvidia-smi**: GPU monitoring
- **htop**: System monitoring

## References

- [Optimizing LLMs for Speed](https://huggingface.co/docs/transformers/main/en/perf_train_gpu_one)
- [vLLM Paper](https://arxiv.org/abs/2309.06180)
- [Flash Attention](https://arxiv.org/abs/2205.14135)
- [Quantization Methods](https://huggingface.co/docs/transformers/main/en/quantization)
- [PyTorch Profiler Guide](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
