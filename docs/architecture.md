# System Architecture

## Overview

This document describes the architecture of the LLM Latency Bottleneck Analysis system, a production-grade platform for identifying and analyzing performance bottlenecks in LLM inference pipelines.

---

# PART 1: TARGET STATE ARCHITECTURE

> **Migration Status**: Planning Phase
> **Target Date**: Q1 2025
> **Reference Documents**:
> - `PLAN.md` - Step-by-step implementation guide
> - `docs/ADR-001-INFERENCE-ENGINE.md` - vLLM vs TGI decision record

## The One Diagram (Target State)

```
                                    PRODUCTION LLM INFERENCE PLATFORM
                                    ==================================

                    +------------------------------------------------------------------+
                    |                        CONTROL PLANE                             |
                    |  +------------------+  +-------------------+  +----------------+ |
                    |  | Model Registry   |  | Config Management |  | Secrets Vault  | |
                    |  | (S3 + DynamoDB)  |  | (ConfigMaps)      |  | (AWS SM)       | |
                    |  +--------+---------+  +---------+---------+  +-------+--------+ |
                    +-----------|--------------------|----------------------|-----------+
                                |                    |                      |
    +==========+                v                    v                      v
    | CLIENTS  |     +------------------------------------------------------------------+
    | - Web    |     |                         INGRESS LAYER                            |
    | - Mobile |     |  +------------------+                    +---------------------+ |
    | - API    |====>|  | AWS ALB / NLB    |                    | Rate Limiter        | |
    +==========+     |  | (TLS Termination)|                    | (Token Bucket)      | |
                     |  +--------+---------+                    +----------+----------+ |
                     +-----------|----------------------------------------|-------------+
                                 |                                        |
                                 v                                        v
                     +------------------------------------------------------------------+
                     |                      KUBERNETES LAYER                            |
                     |  +----------------------------------------------------------+   |
                     |  |                    KEDA Scaler                           |   |
                     |  |  Metrics: queue_depth, gpu_util, requests_in_flight      |   |
                     |  +------------------------------+---------------------------+   |
                     |                                 |                               |
                     |            +--------------------|--------------------+          |
                     |            |                    |                    |          |
                     |            v                    v                    v          |
                     |  +-----------------+  +-----------------+  +-----------------+  |
                     |  | Inference Pod 1 |  | Inference Pod 2 |  | Inference Pod N |  |
                     |  | +-------------+ |  | +-------------+ |  | +-------------+ |  |
                     |  | | vLLM Engine | |  | | vLLM Engine | |  | | vLLM Engine | |  |
                     |  | | - Paged Attn| |  | | - Paged Attn| |  | | - Paged Attn| |  |
                     |  | | - Cont Batch| |  | | - Cont Batch| |  | | - Cont Batch| |  |
                     |  | +------+------+ |  | +------+------+ |  | +------+------+ |  |
                     |  |        |        |  |        |        |  |        |        |  |
                     |  |  [NVIDIA GPU]   |  |  [NVIDIA GPU]   |  |  [NVIDIA GPU]   |  |
                     |  +-----------------+  +-----------------+  +-----------------+  |
                     +------------------------------------------------------------------+
                                                    |
                     +------------------------------------------------------------------+
                     |                    OBSERVABILITY LAYER                           |
                     |  +------------------+  +------------------+  +----------------+  |
                     |  | OpenTelemetry    |  | Prometheus       |  | Grafana        |  |
                     |  | Collector        |  | (TSDB)           |  | (Dashboards)   |  |
                     |  +--------+---------+  +--------+---------+  +-------+--------+  |
                     |           |                     |                    |           |
                     |           +----------+----------+--------------------+           |
                     |                      v                                           |
                     |           +---------------------+    +------------------------+  |
                     |           | Jaeger (Tracing)    |    | AlertManager + PagerDuty| |
                     |           +---------------------+    +------------------------+  |
                     +------------------------------------------------------------------+
```

## Target Request Flow

```
+--------+    +--------+    +----------+    +--------+    +----------+    +--------+
| Client | -> | ALB/   | -> | Rate     | -> | K8s    | -> | vLLM     | -> | Client |
|        |    | NLB    |    | Limiter  |    | Service|    | Pod      |    |        |
+--------+    +--------+    +----------+    +--------+    +----------+    +--------+
    |             |              |              |              |              |
    |  1. HTTPS   |              |              |              |              |
    |  Request    |              |              |              |              |
    |------------>|              |              |              |              |
    |             | 2. Check     |              |              |              |
    |             | Rate Limit   |              |              |              |
    |             |------------->|              |              |              |
    |             |              | 3. Route to  |              |              |
    |             |              | healthy pod  |              |              |
    |             |              |------------->|              |              |
    |             |              |              | 4. Queue &   |              |
    |             |              |              | Continuous   |              |
    |             |              |              | Batch        |              |
    |             |              |              |------------->|              |
    |             |              |              |              | 5. Generate  |
    |             |              |              |              | w/ PagedAttn |
    |             |              |              |<-------------|              |
    |             |              |<-------------|              |              |
    |             |<-------------|              |              |              |
    |<------------|              |              |              |              |
    |  6. SSE     |              |              |              |              |
    |  Stream     |              |              |              |              |
```

## Current State vs Target State

| Aspect | Current State | Target State | Improvement |
|--------|---------------|--------------|-------------|
| **Inference Engine** | Custom PyTorch/Transformers | vLLM with PagedAttention | 20x throughput |
| **Deployment** | Bare metal / VM | Kubernetes (EKS) | Zero-downtime |
| **Scaling** | Manual (semaphore=1) | KEDA auto-scaling (1-10 pods) | Auto recovery |
| **Model Versioning** | HuggingFace Hub ID | Immutable S3 Registry + DynamoDB | < 1min rollback |
| **Rollback** | Redeploy (10+ min) | Blue-Green (< 60 seconds) | 10x faster |
| **API** | Custom FastAPI | OpenAI-compatible (vLLM native) | Drop-in replacement |
| **Throughput** | 0.12 req/s | 100+ req/s | 833x |
| **TTFT P95** | 7.7s | < 500ms | 15x faster |
| **Concurrency** | 1 request | 100+ (continuous batching) | 100x |
| **Cost** | $5+/M tokens | < $0.60/M tokens | 8x cheaper |

## Key Architectural Decisions

### 1. Inference Engine: vLLM
- **Decision**: Migrate from custom PyTorch/Transformers to vLLM
- **Why**: PagedAttention provides 24x memory efficiency, continuous batching for 3-5x throughput
- **Trade-off**: Learning curve vs. massive performance gain
- **Reference**: `docs/ADR-001-INFERENCE-ENGINE.md`

### 2. Orchestration: Kubernetes + KEDA
- **Decision**: Event-driven auto-scaling based on queue depth and GPU utilization
- **Why**: Zero-downtime deployments, automatic scaling, industry standard
- **Trade-off**: Operational complexity vs. reliability

### 3. Model Registry: S3 + DynamoDB
- **Decision**: Immutable versioning with SHA256 checksums
- **Why**: Sub-1-minute rollbacks, audit trail, reproducibility
- **Trade-off**: Additional infrastructure vs. safety

### 4. Cost Optimization: Spot Instances + AWQ Quantization
- **Decision**: 60% savings from spot, 40% memory reduction from AWQ
- **Why**: Production costs must be sustainable
- **Trade-off**: Spot interruptions vs. cost savings (mitigated by graceful shutdown)

## Recovery & SLAs (Target)

| Failure Scenario | Detection | Recovery | RTO |
|------------------|-----------|----------|-----|
| Pod crash | 10s (liveness probe) | 30s (restart) | < 1 min |
| Node failure | 30s (node-problem-detector) | 60s (reschedule) | < 2 min |
| Bad deployment | 5s (error rate spike) | 45s (rollback) | < 1 min |
| AZ failure | 60s (health checks) | 120s (failover) | < 3 min |
| Model corruption | Load failure | 60s (rollback) | < 2 min |

## Service Level Objectives (Target)

| Metric | Current | SLO Target |
|--------|---------|------------|
| Availability | 99.5% | 99.9% |
| TTFT P95 | 7.7s | < 1s |
| TPOT P95 | 300ms | < 100ms |
| Error Rate | 0.5% | < 0.1% |
| Throughput | 0.12/s | > 100/s |

---

# PART 2: CURRENT STATE ARCHITECTURE

> The following sections describe the current implementation that will be migrated.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Layer                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │  HTTP    │  │  Locust  │  │  cURL    │  │  Scripts │       │
│  │  Client  │  │  Load    │  │          │  │          │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Layer (FastAPI)                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Middleware Stack                                        │  │
│  │  • CORS                                                  │  │
│  │  • Process Time Tracking                                │  │
│  │  • OpenTelemetry Instrumentation                        │  │
│  │  • Exception Handling                                   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │  /generate  │  │  /health    │  │  /metrics   │           │
│  │  /model/*   │  │  /docs      │  │             │           │
│  └─────────────┘  └─────────────┘  └─────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Inference Engine Layer                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  InferenceEngine                                         │  │
│  │  • Model Management (Load/Unload)                       │  │
│  │  • Generation Pipeline                                  │  │
│  │  • Performance Instrumentation                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │  Tokenizer  │  │  LLM Model  │  │  Timer      │           │
│  │             │  │  (Mistral)  │  │  Metrics    │           │
│  └─────────────┘  └─────────────┘  └─────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Observability Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │  Jaeger     │  │  Prometheus │  │  Metrics    │           │
│  │  (Traces)   │  │  (Metrics)  │  │  Collector  │           │
│  │  :16686     │  │  :9090      │  │             │           │
│  └─────────────┘  └─────────────┘  └─────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. API Layer

**Technology**: FastAPI + Uvicorn

**Responsibilities**:
- HTTP request handling
- Request validation (Pydantic)
- Response serialization
- Error handling
- CORS management
- Middleware execution

**Key Files**:
- `src/api/main.py`: FastAPI application
- `src/api/models.py`: Request/response models

**Design Decisions**:
- **FastAPI**: Chosen for automatic OpenAPI docs, async support, and Pydantic integration
- **Singleton Engine**: Single InferenceEngine instance shared across requests for memory efficiency
- **Lazy Loading**: Model loads on first request to reduce startup time
- **Lifespan Events**: Proper initialization and cleanup using async context managers

### 2. Inference Engine

**Technology**: PyTorch + Transformers + BitsAndBytes

**Responsibilities**:
- Model lifecycle management
- Text generation
- Performance timing
- Memory tracking
- Error handling (OOM, generation failures)

**Key Files**:
- `src/inference/engine.py`: InferenceEngine class
- `src/inference/metrics.py`: InferenceTimer and TokenTimings

**Design Decisions**:
- **8-bit Quantization**: Optional BitsAndBytes integration for memory efficiency
- **Device Management**: Automatic CPU fallback when GPU unavailable
- **Token-by-Token Timing**: High-precision timing for TTFT and TPOT
- **Memory Monitoring**: Track RAM and VRAM usage throughout inference

### 3. Telemetry Layer

**Technology**: OpenTelemetry + Prometheus + Jaeger

**Responsibilities**:
- Distributed tracing
- Metrics collection
- Span management
- Exporter configuration

**Key Files**:
- `src/telemetry/tracer.py`: OpenTelemetry setup
- `src/telemetry/metrics_collector.py`: Prometheus metrics

**Design Decisions**:
- **OpenTelemetry Standard**: Industry-standard observability framework
- **Jaeger Backend**: Mature, battle-tested tracing system
- **Prometheus Metrics**: Standard metrics format with histogram buckets
- **Singleton Pattern**: Single metrics collector instance across application

### 4. Observability Stack

**Technology**: Docker Compose + Jaeger + Prometheus

**Responsibilities**:
- Trace collection and storage
- Metrics storage and querying
- UI for visualization
- Data persistence

**Key Files**:
- `docker/docker-compose.yml`: Service orchestration
- `docker/prometheus/prometheus.yml`: Scrape configuration

**Design Decisions**:
- **Containerized**: Isolated, reproducible deployment
- **Badger Storage**: Embedded storage for Jaeger (simpler than Elasticsearch)
- **Volume Persistence**: Data survives container restarts
- **Health Checks**: Automatic container health monitoring

## Data Flow

### Request Flow

```
1. HTTP Request arrives at FastAPI
   ↓
2. Middleware adds process time tracking
   ↓
3. OpenTelemetry creates root span
   ↓
4. Pydantic validates request body
   ↓
5. InferenceEngine.generate() called
   ↓
6. InferenceTimer starts timing
   ↓
7. Tokenizer processes prompt (child span)
   ↓
8. Model generates tokens (child span)
   ↓
9. First token generated → TTFT recorded
   ↓
10. Subsequent tokens → TPOT recorded
   ↓
11. InferenceTimer stops, calculates metrics
   ↓
12. MetricsCollector updates Prometheus counters
   ↓
13. OpenTelemetry span ends, exported to Jaeger
   ↓
14. Response serialized and returned
   ↓
15. Process time added to response header
```

### Trace Hierarchy

```
/generate (root span)
├── tokenize_input (child span)
│   └── attributes: prompt_tokens=15
├── model_generate (child span)
│   ├── attributes: max_tokens=256, temperature=0.7
│   ├── event: first_token_generated (TTFT)
│   └── attributes: total_tokens=271
├── decode_output (child span)
│   └── attributes: completion_tokens=256
└── attributes: ttft=0.456s, tpot=0.032s
```

### Metrics Flow

```
Generation Request
   ↓
InferenceEngine.generate()
   ↓
Performance Data Collected:
  • TTFT: 0.456s
  • TPOT: 0.032s
  • Tokens: 271
  • Time: 8.5s
  • Memory: 2.3GB
   ↓
MetricsCollector.record_inference_metrics()
   ↓
Prometheus Metrics Updated:
  • llm_ttft_seconds (histogram)
  • llm_tpot_seconds (histogram)
  • llm_tokens_processed_total (counter)
  • llm_memory_usage_megabytes (gauge)
   ↓
Prometheus Scrapes /metrics endpoint (5s interval)
   ↓
Metrics Stored in Prometheus TSDB
   ↓
Available for Querying via PromQL
```

## Performance Characteristics

### Latency Breakdown

Typical inference request (256 tokens):

```
Total Request Time: 8.5s
├── API Overhead: ~50ms (0.6%)
│   ├── Request parsing: 10ms
│   ├── Validation: 5ms
│   ├── Serialization: 15ms
│   └── Middleware: 20ms
├── Tokenization: ~45ms (0.5%)
├── Model Generation: 8.4s (98.8%)
│   ├── TTFT: 0.456s (5.4%)
│   └── Token Generation: 7.944s (93.4%)
└── Response Overhead: ~5ms (0.1%)
```

### Resource Utilization

**Memory (Mistral-7B)**:
- Model Weights (FP16): ~14GB
- Model Weights (8-bit): ~7GB
- Runtime Overhead: ~2GB
- Peak Memory: 16GB (FP16), 9GB (8-bit)

**Compute**:
- GPU Utilization: 70-90% during generation
- CPU Utilization: 10-20% (tokenization, overhead)
- Memory Bandwidth: Critical bottleneck for TPOT

**Throughput**:
- Single Request: ~30 tokens/sec
- Concurrent Requests: Limited by memory
- Queue Handling: Serial processing (no batching)

## Scalability Considerations

### Vertical Scaling

**GPU Memory** (Primary Bottleneck):
- 24GB: Supports FP16 with small batch
- 16GB: Requires 8-bit quantization
- 12GB: Supports 8-bit only, no batching
- 8GB: Not recommended for 7B models

**GPU Compute**:
- A100: ~40 tokens/sec
- RTX 4090: ~35 tokens/sec
- V100: ~25 tokens/sec
- RTX 3090: ~30 tokens/sec

### Horizontal Scaling

**Current Limitations**:
- Single model instance per API server
- No request batching
- No load balancing built-in

**Scaling Options**:
1. **Multiple API Instances**: Load balancer + multiple containers
2. **Model Serving Frameworks**: Ray Serve, TorchServe, Triton
3. **Async Generation**: Queue-based architecture
4. **Batch Processing**: Combine requests (increases TTFT, improves throughput)

### Bottleneck Mitigation

**TTFT Optimization**:
- Model preloading
- KV cache warming
- Speculative decoding
- Faster tokenization

**TPOT Optimization**:
- Better hardware (A100)
- Quantization (8-bit, 4-bit)
- Flash Attention
- Larger batch sizes

**Memory Optimization**:
- 8-bit/4-bit quantization
- Model sharding
- Gradient checkpointing (training only)
- KV cache optimization

## Monitoring and Alerting

### Key Metrics to Monitor

**Latency Metrics**:
- `llm_ttft_seconds` (P50, P95, P99)
- `llm_tpot_seconds` (P50, P95)
- `llm_inference_duration_seconds` (P50, P95, P99)

**Throughput Metrics**:
- `rate(llm_requests_total[5m])`
- `llm_tokens_processed_total`
- `llm_throughput_tokens_per_second`

**Resource Metrics**:
- `llm_memory_usage_megabytes{memory_type="vram"}`
- `llm_active_requests`
- `llm_errors_total`

**System Metrics**:
- GPU utilization (nvidia-smi)
- CPU usage
- Network I/O

### Alert Thresholds

**Critical**:
- TTFT P95 > 3s
- Error rate > 5%
- VRAM usage > 95%
- Active requests > 10

**Warning**:
- TTFT P95 > 1.5s
- Error rate > 1%
- VRAM usage > 85%
- Active requests > 5

## Security Considerations

**Current Implementation** (Development):
- No authentication
- CORS allows all origins
- No rate limiting
- No input sanitization beyond validation

**Production Recommendations**:
- Add API key authentication
- Restrict CORS origins
- Implement rate limiting (per-user/per-IP)
- Add input content filtering
- Enable HTTPS/TLS
- Audit logging
- Secret management for API keys

## Deployment Options

### Local Development

```bash
# Single machine with GPU
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Docker Deployment

```bash
# Build image
docker build -t llm-inference-api .

# Run container
docker run -p 8000:8000 \
  --gpus all \
  -v ./models:/app/models \
  llm-inference-api
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-inference
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: api
        image: llm-inference-api:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 32Gi
```

### Cloud Deployment

**AWS**:
- EC2 with GPU (g4dn, p3, p4)
- ECS/EKS for orchestration
- CloudWatch for monitoring
- ALB for load balancing

**GCP**:
- Compute Engine with GPU (T4, V100, A100)
- GKE for orchestration
- Cloud Monitoring
- Cloud Load Balancing

**Azure**:
- NC-series VMs (NVIDIA GPUs)
- AKS for orchestration
- Azure Monitor
- Application Gateway

## Future Enhancements

### Performance

- [ ] Request batching
- [ ] Async generation with streaming
- [ ] KV cache optimization
- [ ] Flash Attention integration
- [ ] 4-bit quantization (GPTQ, AWQ)
- [ ] Speculative decoding

### Features

- [ ] Multi-model support
- [ ] Model routing
- [ ] A/B testing framework
- [ ] Prompt caching
- [ ] Response caching
- [ ] Cost tracking

### Observability

- [ ] Custom Grafana dashboards
- [ ] Alertmanager integration
- [ ] Log aggregation (ELK, Loki)
- [ ] Distributed tracing across services
- [ ] Real-user monitoring (RUM)

### Operations

- [ ] Helm charts
- [ ] CI/CD pipeline
- [ ] Automated testing
- [ ] Canary deployments
- [ ] Auto-scaling
- [ ] Disaster recovery

## References

- [FastAPI Best Practices](https://fastapi.tiangolo.com/tutorial/)
- [OpenTelemetry Python](https://opentelemetry.io/docs/instrumentation/python/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [Jaeger Architecture](https://www.jaegertracing.io/docs/latest/architecture/)
