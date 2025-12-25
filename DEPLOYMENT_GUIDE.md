# LLM Inference Service - Deployment & Testing Guide

This guide covers deploying the LLM inference service to GPU instances and testing the new features including streaming responses, batch generation, and production monitoring.

## Table of Contents

1. [Overview](#overview)
2. [GPU Deployment](#gpu-deployment)
3. [Streaming Responses](#streaming-responses)
4. [Performance Testing](#performance-testing)
5. [Production Monitoring](#production-monitoring)
6. [Next Steps](#next-steps)

## Overview

### New Features Implemented

✅ **GPU Instance Deployment**
- Terraform configurations for AWS (g5.xlarge, p3, p4 instances)
- Deployment scripts for AWS, GCP, and Azure
- Automated setup with Docker and NVIDIA Container Toolkit
- Production-ready infrastructure with monitoring

✅ **Streaming Responses**
- Token-by-token streaming via Server-Sent Events (SSE)
- Real-time TTFT and TPOT metrics during generation
- `/generate_stream` endpoint with concurrency control
- Improved user experience with immediate feedback

✅ **Performance Testing Suite**
- Comprehensive testing scripts for all generation modes
- Batch vs single vs streaming comparisons
- Concurrency and scaling tests
- Automated performance analysis and recommendations

✅ **Production Monitoring**
- Prometheus alerting rules for health, performance, and resources
- Alertmanager with email, Slack, and PagerDuty integration
- Comprehensive runbooks and troubleshooting guides
- Production-ready dashboards and metrics

## GPU Deployment

### Prerequisites

- Cloud provider account (AWS/GCP/Azure)
- Terraform installed (for AWS)
- Cloud CLI tools (aws-cli, gcloud, or az)
- SSH key pair generated

### AWS Deployment

#### 1. Configure Terraform Variables

```bash
cd deploy/aws

# Create terraform.tfvars
cat > terraform.tfvars <<EOF
aws_region = "us-west-2"
instance_type = "g5.xlarge"  # or g5.2xlarge, p3.2xlarge
ssh_key_name = "your-key-name"
ssh_key_path = "~/.ssh/id_rsa"
environment = "production"
EOF
```

#### 2. Deploy Infrastructure

```bash
# Initialize Terraform
terraform init

# Review deployment plan
terraform plan

# Deploy
terraform apply

# Get instance IP
terraform output instance_public_ip
```

#### 3. Verify Deployment

```bash
# SSH into instance
ssh -i ~/.ssh/id_rsa ubuntu@<INSTANCE_IP>

# Check services
/opt/llm-inference/manage.sh status

# View logs
/opt/llm-inference/manage.sh logs

# Check GPU
nvidia-smi
```

#### 4. Access Services

- **API**: `http://<INSTANCE_IP>:8000`
- **Grafana**: `http://<INSTANCE_IP>:3000` (admin/admin)
- **Jaeger**: `http://<INSTANCE_IP>:16686`
- **Prometheus**: `http://<INSTANCE_IP>:9090`
- **Alertmanager**: `http://<INSTANCE_IP>:9093`

### Quick Deployment Script

```bash
# Deploy to AWS with default settings
./deploy/deploy.sh --provider aws --instance-type g5.xlarge --region us-west-2

# Deploy to GCP
./deploy/deploy.sh --provider gcp --instance-type n1-standard-4 --region us-central1

# Deploy to Azure
./deploy/deploy.sh --provider azure --instance-type Standard_NC4as_T4_v3 --region eastus
```

### Cost Estimates

| Instance Type | GPU | vCPU | RAM | Cost/Hour (approx) |
|--------------|-----|------|-----|-------------------|
| g5.xlarge | 1x A10G (24GB) | 4 | 16GB | $1.00 |
| g5.2xlarge | 1x A10G (24GB) | 8 | 32GB | $1.50 |
| g5.4xlarge | 1x A10G (24GB) | 16 | 64GB | $2.00 |
| p3.2xlarge | 1x V100 (16GB) | 8 | 61GB | $3.06 |
| p4d.24xlarge | 8x A100 (40GB) | 96 | 1.1TB | $32.77 |

## Streaming Responses

### API Endpoint

**POST** `/generate_stream`

Returns Server-Sent Events (SSE) stream with token-by-token generation.

### Request Format

```json
{
  "prompt": "The future of artificial intelligence is",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "do_sample": true
}
```

### Response Format (SSE)

```
data: {"token": "bright", "text": "bright", "finished": false, "metrics": {...}}

data: {"token": " and", "text": "bright and", "finished": false, "metrics": {...}}

data: {"token": " promising", "text": "bright and promising", "finished": true, "metrics": {...}}
```

### Testing Streaming

```bash
# Test streaming endpoint
python scripts/test_streaming.py --url http://localhost:8000

# Compare streaming vs standard
python scripts/test_streaming.py --compare

# Test concurrent streaming
python scripts/test_streaming.py --concurrent 5

# Save results
python scripts/test_streaming.py --output streaming_results.json
```

### Example: Python Client

```python
import aiohttp
import json

async def stream_generate(prompt: str):
    url = "http://localhost:8000/generate_stream"
    request_data = {
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=request_data) as response:
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line.startswith('data: '):
                    data = json.loads(line[6:])
                    token = data.get("token", "")
                    print(token, end="", flush=True)
                    
                    if data.get("finished"):
                        break
```

### Benefits of Streaming

- **Improved UX**: Users see output immediately (lower perceived latency)
- **Better TTFT**: First token arrives faster than waiting for complete response
- **Real-time metrics**: Monitor generation progress in real-time
- **Resource efficiency**: Can cancel long-running generations early

## Performance Testing

### Comprehensive Test Suite

```bash
# Run full performance test suite
python scripts/performance_test.py \
  --url http://localhost:8000 \
  --prompt "The future of AI is" \
  --max-tokens 128 \
  --num-requests 20 \
  --batch-sizes 1 4 8 16 \
  --output performance_results.json
```

### Test Components

1. **Single Generation**: Sequential requests with detailed metrics
2. **Batch Generation**: Multiple batch sizes (1, 4, 8, 16)
3. **Streaming Generation**: Token-by-token performance
4. **Concurrent Requests**: Scaling from 1 to 16 concurrent requests

### Key Metrics Measured

- **Latency**: TTFT, TPOT, total time (p50, p95, p99)
- **Throughput**: Tokens/second, requests/second
- **Efficiency**: Batch efficiency, concurrency scaling
- **Resource Usage**: GPU utilization, memory usage

### Expected Results

Based on analysis, you should see:

- **TTFT**: 0.3-0.5s on GPU (vs 2-3s on CPU)
- **TPOT**: 0.02-0.05s on GPU (vs 0.1-0.2s on CPU)
- **Throughput**: 50-100 tok/s on GPU (vs 10-20 tok/s on CPU)
- **Batch Efficiency**: 1.5-2x improvement with batch size 8-16
- **Concurrency Scaling**: 0.7-0.9x efficiency up to 4-8 concurrent requests

### Profiling Batch vs Single

```bash
# Compare batch performance
python scripts/profile_batching.py \
  --url http://localhost:8000 \
  --batch-sizes 1 2 4 8 16 32 \
  --num-iterations 10 \
  --output batch_profile.json
```

### Load Testing

```bash
# Run Locust load test
cd load_testing
locust -f locustfile.py --host http://localhost:8000

# Access Locust UI
open http://localhost:8089
```

## Production Monitoring

### Setup Alertmanager

1. **Configure Environment Variables**

```bash
# Create .env file for monitoring
cat > monitoring/.env <<EOF
SMTP_PASSWORD=your-smtp-password
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
PAGERDUTY_SERVICE_KEY=your-pagerduty-key
PAGERDUTY_LLM_SERVICE_KEY=your-llm-specific-key
EOF
```

2. **Update Alert Recipients**

Edit `monitoring/alertmanager.yml`:
- Replace `team@example.com` with your team email
- Replace `#alerts-critical` with your Slack channels
- Update PagerDuty service keys

3. **Start Monitoring Stack**

```bash
cd docker
docker-compose up -d

# Verify services
docker-compose ps

# Check Alertmanager
curl http://localhost:9093/api/v1/status
```

### Alert Categories

**Critical Alerts** (Immediate response):
- Service down
- Very high error rate (>10%)
- Critical GPU memory usage (>95%)
- Very high TTFT (>5s)

**Warning Alerts** (Business hours response):
- High error rate (>5%)
- High GPU utilization (>90%)
- High TTFT (>2s)
- Low throughput (<10 tok/s)

### Testing Alerts

```bash
# Validate alert rules
promtool check rules monitoring/production-alerts.yml

# Validate Alertmanager config
amtool check-config monitoring/alertmanager.yml

# Send test alert
curl -X POST http://localhost:9093/api/v1/alerts -d '[{
  "labels": {
    "alertname": "TestAlert",
    "severity": "warning",
    "service": "llm-inference"
  },
  "annotations": {
    "summary": "Test alert",
    "description": "This is a test"
  }
}]'
```

### Monitoring Dashboards

Access Grafana at `http://localhost:3000`:

1. **LLM Metrics Dashboard**
   - Request rate and response time
   - TTFT and TPOT distributions
   - Token generation throughput
   - Error rates and types

2. **Resource Dashboard**
   - GPU utilization and memory
   - CPU and system memory
   - Active requests and queue depth
   - Network I/O

3. **Alerting Dashboard**
   - Active alerts by severity
   - Alert history and trends
   - MTTD and MTTR metrics

## Next Steps

### 1. Baseline Performance Testing

```bash
# Run comprehensive performance test
python scripts/performance_test.py --output baseline_cpu.json

# After GPU deployment
python scripts/performance_test.py \
  --url http://<GPU_INSTANCE_IP>:8000 \
  --output baseline_gpu.json

# Compare results
python scripts/compare_results.py baseline_cpu.json baseline_gpu.json
```

### 2. Optimize Configuration

Based on test results, tune:
- `MAX_CONCURRENT_GENERATIONS`: Start with 4, adjust based on GPU memory
- `MAX_BATCH_SIZE`: Test 8, 16, 32 for optimal throughput
- Model quantization: 8-bit vs 4-bit vs full precision
- Context length: Balance between capability and memory

### 3. Production Deployment Checklist

- [ ] GPU instance deployed and verified
- [ ] Model loaded successfully on GPU
- [ ] Streaming endpoint tested
- [ ] Batch generation tested
- [ ] Performance baseline established
- [ ] Monitoring stack deployed
- [ ] Alerts configured and tested
- [ ] Runbooks reviewed by team
- [ ] Load testing completed
- [ ] Backup and recovery tested
- [ ] Documentation updated
- [ ] Team trained on operations

### 4. Continuous Improvement

**Week 1-2**: Monitor and tune
- Review alert frequency
- Adjust thresholds based on actual traffic
- Optimize batch sizes and concurrency

**Week 3-4**: Scale and optimize
- Implement auto-scaling if needed
- Add caching layer for common prompts
- Optimize model loading and warm-up

**Month 2+**: Advanced features
- Multi-model support
- A/B testing framework
- Cost optimization strategies
- Advanced caching and routing

### 5. Cost Optimization

**Immediate**:
- Use spot instances for non-critical workloads
- Enable 8-bit quantization (50% memory savings)
- Right-size instance based on actual usage

**Short-term**:
- Implement request batching (1.5-2x efficiency)
- Add auto-scaling with scale-to-zero
- Use reserved instances for base capacity

**Long-term**:
- Multi-tenancy with model sharing
- Serverless inference for low-traffic periods
- Custom model optimization and pruning

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose logs llm-inference

# Check GPU availability
nvidia-smi

# Verify model download
ls -lh ~/.cache/huggingface/

# Check disk space
df -h
```

### High Latency

```bash
# Check GPU utilization
nvidia-smi

# Check concurrent requests
curl http://localhost:8000/metrics | grep llm_active_requests

# Review Grafana dashboard
open http://localhost:3000
```

### Out of Memory

```bash
# Enable 8-bit quantization
export MODEL_LOAD_IN_8BIT=true

# Reduce batch size
export MAX_BATCH_SIZE=8

# Reduce concurrent requests
export MAX_CONCURRENT_GENERATIONS=2

# Restart service
docker-compose restart llm-inference
```

## Support and Resources

- **Documentation**: See `docs/` directory
- **Monitoring Guide**: `monitoring/README.md`
- **Deployment Configs**: `deploy/` directory
- **Testing Scripts**: `scripts/` directory
- **Analysis Report**: `ANALYSIS_REPORT.md`

## Summary

All four major improvements have been implemented:

1. ✅ **GPU Deployment**: Terraform configs and deployment scripts for AWS/GCP/Azure
2. ✅ **Streaming Responses**: Real-time token generation with `/generate_stream` endpoint
3. ✅ **Performance Testing**: Comprehensive test suite with batch/streaming/concurrency tests
4. ✅ **Production Monitoring**: Alertmanager with Prometheus rules and multi-channel notifications

The service is now ready for GPU deployment with production-grade monitoring and testing capabilities.
