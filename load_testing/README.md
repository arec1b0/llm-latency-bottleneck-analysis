# Load Testing Guide

Comprehensive load testing suite for LLM inference API using Locust.

## Overview

This directory contains load testing scripts to identify latency bottlenecks and performance characteristics under various load patterns.

## Files

- **locustfile.py**: Main Locust test file with user behaviors
- **scenarios.py**: Predefined test scenarios with configurations
- **results/**: Directory for storing test results (auto-created)

## Quick Start

### 1. List Available Scenarios

```bash
python scenarios.py --list
```

### 2. View Recommended Workflow

```bash
python scenarios.py --workflow
```

### 3. Run a Scenario

```bash
python scenarios.py baseline
```

This will show the command to run the baseline scenario.

### 4. Execute Load Test

```bash
locust -f locustfile.py --users 10 --spawn-rate 2 --run-time 5m --host http://localhost:8000 --headless
```

Or use the Locust web UI:

```bash
locust -f locustfile.py --host http://localhost:8000
```

Then open http://localhost:8089

## Available Scenarios

### Quick Smoke Test
- **Users**: 5
- **Duration**: 2 minutes
- **Purpose**: Fast sanity check

```bash
locust -f locustfile.py --users 5 --spawn-rate 5 --run-time 2m --host http://localhost:8000 --headless
```

### Baseline
- **Users**: 10
- **Duration**: 5 minutes
- **Purpose**: Establish baseline metrics

```bash
locust -f locustfile.py --users 10 --spawn-rate 2 --run-time 5m --host http://localhost:8000 --headless
```

### Medium Load
- **Users**: 25
- **Duration**: 10 minutes
- **Purpose**: Test typical production load

```bash
locust -f locustfile.py --users 25 --spawn-rate 5 --run-time 10m --host http://localhost:8000 --headless
```

### High Load
- **Users**: 50
- **Duration**: 10 minutes
- **Purpose**: Identify performance degradation

```bash
locust -f locustfile.py --users 50 --spawn-rate 10 --run-time 10m --host http://localhost:8000 --headless
```

### Stress Test
- **Users**: 100
- **Duration**: 15 minutes
- **Purpose**: Find breaking point

```bash
locust -f locustfile.py --users 100 --spawn-rate 20 --run-time 15m --host http://localhost:8000 --headless --user-class HighLoadUser
```

### Burst Traffic
- **Users**: 30
- **Duration**: 10 minutes
- **Purpose**: Test spike handling

```bash
locust -f locustfile.py --users 30 --spawn-rate 30 --run-time 10m --host http://localhost:8000 --headless --user-class BurstLoadUser
```

## User Classes

### LLMInferenceUser
Standard user with realistic wait times (1-3 seconds between requests).

**Task Distribution:**
- 10x: Short prompts (64 tokens)
- 5x: Medium prompts (128 tokens)
- 2x: Long prompts (256 tokens)
- 1x: High temperature generation
- 1x: Health checks
- 1x: Model info requests

### HighLoadUser
Aggressive user with minimal wait time (0.1-0.5 seconds).

### BurstLoadUser
Sends requests in bursts, simulating traffic spikes.

## Metrics Collected

### Standard Locust Metrics
- Request count
- Failure rate
- Response time (min, median, max, percentiles)
- Requests per second

### Custom LLM Metrics
- **TTFT**: Time to First Token
- **TPOT**: Time Per Output Token
- **Token counts**: Prompt, completion, total
- **Throughput**: Tokens per second
- **Memory usage**: Tracked via API

### Results Location

Results are automatically saved to:
```
load_testing/results/metrics_YYYYMMDD_HHMMSS.json
```

## Analyzing Results

### During Test

Monitor in real-time:
1. Locust web UI: http://localhost:8089
2. Jaeger traces: http://localhost:16686
3. Prometheus metrics: http://localhost:9090

### After Test

1. **Check Locust HTML Report**:
   - Generated automatically in headless mode
   - Contains detailed statistics and graphs

2. **Analyze Custom Metrics**:
   ```bash
   cat load_testing/results/metrics_*.json | jq .
   ```

3. **Review Jaeger Traces**:
   - Filter by time range
   - Look for slow spans
   - Identify bottlenecks

4. **Query Prometheus**:
   ```promql
   rate(llm_requests_total[5m])
   histogram_quantile(0.95, llm_ttft_seconds)
   llm_memory_usage_megabytes
   ```

## Interpreting Results

### Good Performance
- **TTFT P95**: < 1 second
- **TPOT Average**: < 0.1 seconds
- **Error Rate**: < 1%
- **Throughput**: Stable across duration

### Performance Degradation Indicators
- **TTFT P95**: > 2 seconds
- **TPOT increasing over time**
- **Memory usage growing**
- **Error rate increasing**

### Common Bottlenecks

#### 1. High TTFT
**Causes:**
- Model loading overhead
- Tokenization latency
- GPU memory transfer

**Solutions:**
- Preload model
- Optimize tokenization
- Use larger batch sizes

#### 2. High TPOT
**Causes:**
- GPU computation limits
- Memory bandwidth
- Inefficient decoding

**Solutions:**
- Enable quantization
- Reduce max tokens
- Use faster hardware

#### 3. Queue Buildup
**Causes:**
- Slow request processing
- Insufficient concurrency
- Memory bottleneck

**Solutions:**
- Add workers
- Implement request queue
- Scale horizontally

#### 4. Memory Issues
**Causes:**
- Model size too large
- Memory leaks
- Batch size too high

**Solutions:**
- Enable 8-bit quantization
- Monitor for leaks
- Reduce batch size

## Best Practices

### Before Testing

1. **Ensure API is running**:
   ```bash
   curl http://localhost:8000/health
   ```

2. **Start observability stack**:
   ```bash
   cd docker
   docker-compose up -d
   ```

3. **Preload model** (optional):
   ```bash
   curl -X POST http://localhost:8000/model/load
   ```

### During Testing

1. **Monitor system resources**:
   - CPU usage
   - GPU utilization
   - Memory (RAM and VRAM)
   - Network I/O

2. **Watch for errors**:
   - OOM errors
   - Timeouts
   - 5xx responses

3. **Check Jaeger traces**:
   - Identify slow operations
   - Look for bottlenecks

### After Testing

1. **Save results**:
   ```bash
   cp load_testing/results/metrics_*.json results/archive/
   ```

2. **Generate report**:
   - Locust automatically generates HTML reports
   - Create custom analysis scripts

3. **Clean up**:
   ```bash
   # Unload model to free memory
   curl -X POST http://localhost:8000/model/unload
   
   # Clear GPU cache
   python -c "import torch; torch.cuda.empty_cache()"
   ```

## Recommended Testing Sequence

1. **Quick Smoke** (2 min) - Verify functionality
2. **Baseline** (5 min) - Establish baseline
3. **Medium Load** (10 min) - Test typical load
4. **Ramp Up** (20 min) - Observe degradation curve
5. **High Load** (10 min) - Test under pressure
6. **Burst Traffic** (10 min) - Test spike handling
7. **Stress Test** (15 min) - Find breaking point
8. **Endurance** (30 min) - Test stability (optional)

Wait 5 minutes between tests to allow the system to stabilize.

## Troubleshooting

### Locust Won't Start

```bash
# Check Python environment
python --version
pip list | grep locust

# Reinstall if needed
pip install locust
```

### Connection Refused

```bash
# Verify API is running
curl http://localhost:8000/health

# Check API logs
docker logs <api-container>
```

### High Error Rate

```bash
# Check API health
curl http://localhost:8000/health

# Review API logs for errors
tail -f logs/api.log

# Monitor GPU memory
nvidia-smi -l 1
```

### Results Not Saved

```bash
# Create results directory
mkdir -p load_testing/results

# Check permissions
ls -la load_testing/
```

## Advanced Usage

### Custom Scenarios

Edit `locustfile.py` to add custom user behaviors:

```python
@task(5)
def custom_task(self):
    """Custom task description."""
    payload = {
        "prompt": "Custom prompt",
        "max_tokens": 128,
    }
    self.client.post("/generate", json=payload)
```

### Distributed Testing

Run Locust in distributed mode for higher load:

```bash
# Master
locust -f locustfile.py --master --host http://localhost:8000

# Workers (run on multiple machines)
locust -f locustfile.py --worker --master-host <master-ip>
```

### Export Results

```bash
# CSV export
locust -f locustfile.py --csv=results/test --headless ...

# JSON export via API
curl http://localhost:8089/stats/requests > results/stats.json
```

## References

- [Locust Documentation](https://docs.locust.io/)
- [Load Testing Best Practices](https://docs.locust.io/en/stable/running-distributed.html)
- [Performance Testing Guide](https://www.blazemeter.com/blog/performance-testing-vs-load-testing-vs-stress-testing)
